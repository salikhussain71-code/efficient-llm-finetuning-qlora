"""SAMSum loading, cleaning, tokenisation (with completion-only loss masking) and batching.

Data comes from the Hugging Face Hub (``--dataset-id``) or from local CSV / JSON / JSONL / Parquet files.
Columns are normalised to ``dialogue`` and ``summary``.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config
from .prompts import build_prompt, normalise_text, prompt_overhead, truncate_dialogue

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------ loading
def _read_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: '{path}'.")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type '{suffix}' (use .csv, .json, .jsonl or .parquet).")


def _clean(raw: pd.DataFrame, dialogue_col: str, summary_col: str, source: str) -> pd.DataFrame:
    missing = {dialogue_col, summary_col} - set(raw.columns)
    if missing:
        raise KeyError(f"Column(s) {sorted(missing)} not found in {source}. Available: {list(raw.columns)}. "
                       "Use --dialogue-col / --summary-col.")
    df = pd.DataFrame({
        "id": raw["id"].astype(str) if "id" in raw.columns else raw.index.astype(str),
        "dialogue": raw[dialogue_col].fillna("").map(normalise_text),
        "summary": raw[summary_col].fillna("").map(normalise_text),
    })
    df = df[(df["dialogue"].str.len() > 0) & (df["summary"].str.len() > 0)]
    return df.reset_index(drop=True)


def load_splits(
    dataset_id: str | None = config.DEFAULT_DATASET_ID,
    train_file: str | Path | None = None,
    validation_file: str | Path | None = None,
    test_file: str | Path | None = None,
    dialogue_col: str = config.DIALOGUE_COL,
    summary_col: str = config.SUMMARY_COL,
) -> dict[str, pd.DataFrame]:
    """Return ``{"train": df, "validation": df, "test": df}`` (only the splits that exist).

    Local files take precedence over ``dataset_id``. Each frame has columns ``id``, ``dialogue``, ``summary``.
    """
    local = {"train": train_file, "validation": validation_file, "test": test_file}
    if any(local.values()):
        return {split: _clean(_read_file(Path(p)), dialogue_col, summary_col, str(p))
                for split, p in local.items() if p}
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id)
    except Exception as err:
        raise RuntimeError(
            f"Could not load dataset '{dataset_id}' from the Hugging Face Hub: {err}\n"
            "Check your internet connection, or pass local files with --train-file / --validation-file / --test-file."
        ) from err
    return {split: _clean(ds[split].to_pandas(), dialogue_col, summary_col, f"{dataset_id}:{split}")
            for split in config.SPLITS if split in ds}


def sample_df(df: pd.DataFrame, n: int | None, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Reproducible random subset of ``n`` rows (all rows when ``n`` is falsy or too large)."""
    if not n or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def dataset_stats(df: pd.DataFrame) -> dict:
    """Word-level statistics for the README."""
    d = df["dialogue"].str.split().str.len()
    s = df["summary"].str.split().str.len()
    turns = df["dialogue"].str.count("\n") + 1
    return {"rows": int(len(df)), "dialogue_words_mean": float(d.mean()), "dialogue_words_p95": float(d.quantile(0.95)),
            "summary_words_mean": float(s.mean()), "summary_words_p95": float(s.quantile(0.95)),
            "turns_mean": float(turns.mean())}


def pick_shots(train: pd.DataFrame, k: int, seed: int = config.RANDOM_STATE, max_words: int = 80) -> list[tuple[str, str]]:
    """``k`` short, fixed (seeded) training examples used for few-shot prompting."""
    if k <= 0:
        return []
    pool = train[train["dialogue"].str.split().str.len() <= max_words]
    pool = pool if len(pool) >= k else train
    rows = pool.sample(n=min(k, len(pool)), random_state=seed)
    return list(zip(rows["dialogue"], rows["summary"]))


# ------------------------------------------------------------------------------------ tokenisation
def encode_example(tokenizer, dialogue: str, summary: str, max_seq_len: int) -> tuple[list[int], list[int], bool]:
    """Token ids and labels for one training example.

    Labels are ``-100`` over the prompt, so the loss is computed only on the summary (completion-only
    training). The dialogue is truncated (never the summary) to fit ``max_seq_len``. Returns
    ``(input_ids, labels, truncated)``.
    """
    summary_ids = tokenizer(summary, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    summary_ids = summary_ids[: max(max_seq_len // 2, 8)]
    budget = max_seq_len - len(summary_ids) - prompt_overhead(tokenizer)
    n_dialogue = len(tokenizer(dialogue, add_special_tokens=False)["input_ids"])
    truncated = n_dialogue > budget
    text = truncate_dialogue(dialogue, tokenizer, max(budget, 1)) if truncated else dialogue
    prompt_ids = tokenizer(build_prompt(text), add_special_tokens=True)["input_ids"]
    ids = prompt_ids + summary_ids
    labels = [-100] * len(prompt_ids) + summary_ids
    if len(ids) > max_seq_len:                       # pathological case: keep the end (the summary)
        ids, labels, truncated = ids[-max_seq_len:], labels[-max_seq_len:], True
    return ids, labels, truncated


class SummarizationDataset(Dataset):
    """Pre-tokenised (dialogue -> summary) examples with completion-only labels."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_seq_len: int = 512) -> None:
        self.items: list[dict[str, list[int]]] = []
        self.n_truncated = 0
        for dialogue, summary in zip(df["dialogue"], df["summary"]):
            ids, labels, truncated = encode_example(tokenizer, dialogue, summary, max_seq_len)
            self.items.append({"input_ids": ids, "labels": labels})
            self.n_truncated += int(truncated)
        self.lengths = [len(x["input_ids"]) for x in self.items]
        self.n_target_tokens = sum(sum(t != -100 for t in x["labels"]) for x in self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        return self.items[i]


def collate(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad to the longest sequence in the batch; padding is ignored by attention and loss."""
    longest = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), longest), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), longest), dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        ids[i, :n] = torch.tensor(x["input_ids"])
        labels[i, :n] = torch.tensor(x["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def make_batches(lengths: list[int], batch_size: int, seed: int, epoch: int, group_by_length: bool = True,
                 shuffle: bool = True) -> list[list[int]]:
    """Batches of indices, reproducible from ``(seed, epoch)`` (needed to resume training exactly).

    With ``group_by_length`` the shuffled indices are cut into mega-batches that are sorted by length, which
    removes most padding without making the batch order predictable.
    """
    idx = list(range(len(lengths)))
    rng = random.Random(seed * 100_003 + epoch)
    if shuffle:
        rng.shuffle(idx)
    if group_by_length and shuffle:
        mega = batch_size * 32
        idx = [i for start in range(0, len(idx), mega)
               for i in sorted(idx[start:start + mega], key=lambda j: -lengths[j])]
    batches = [idx[s:s + batch_size] for s in range(0, len(idx), batch_size)]
    if shuffle and group_by_length:
        rng.shuffle(batches)
    return batches
