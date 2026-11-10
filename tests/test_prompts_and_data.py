import pandas as pd
import pytest
from transformers import AutoTokenizer

from qlora_finetune.data import (
    SummarizationDataset,
    collate,
    dataset_stats,
    encode_example,
    load_splits,
    make_batches,
    pick_shots,
    sample_df,
)
from qlora_finetune.prompts import build_prompt, clean_generation, normalise_text, prompt_overhead, truncate_dialogue


def test_prompt_format_is_stable():
    assert build_prompt("A: hi\r\nB: yo ") == ("Summarize the following dialogue.\n\n### Dialogue:\nA: hi\nB: yo"
                                               "\n\n### Summary:\n")


def test_few_shot_prompt_puts_examples_first():
    prompt = build_prompt("A: q", [("X: a", "X says a."), ("Y: b", "Y says b.")])
    assert prompt.count("### Dialogue:") == 3 and prompt.index("X says a.") < prompt.index("Y says b.") < prompt.index("A: q")
    assert prompt.endswith("### Summary:\n")


def test_clean_generation_cuts_runaway_text():
    assert clean_generation("  Anna greets Ben.\n\n### Dialogue:\nmore") == "Anna greets Ben."
    assert clean_generation("Short summary </s> garbage") == "Short summary"
    assert clean_generation("Line one\nline two") == "Line one line two"
    assert clean_generation("") == "" and clean_generation("###") == ""


def test_normalise_text():
    assert normalise_text(None) == "" and normalise_text(float("nan")) == ""
    assert normalise_text(" a \r\n b  \r\n") == "a\n b"


def test_truncate_dialogue_and_overhead(tiny_model):
    tok = AutoTokenizer.from_pretrained(tiny_model)
    long = "Anna: " + "hello there " * 80
    short = truncate_dialogue(long, tok, 20)
    assert len(tok(short, add_special_tokens=False)["input_ids"]) <= 22 and short.startswith("Anna")
    assert truncate_dialogue("Anna: hi", tok, 50) == "Anna: hi"
    assert prompt_overhead(tok, [("A: x", "A says x.")]) > prompt_overhead(tok)


def test_load_local_splits(files):
    splits = load_splits(None, files["train"], files["validation"], files["test"])
    assert set(splits) == {"train", "validation", "test"} and list(splits["train"].columns) == ["id", "dialogue", "summary"]
    assert len(splits["test"]) == 40 and splits["train"]["dialogue"].str.contains("\r").sum() == 0


def test_load_splits_errors(tmp_path, files):
    with pytest.raises(FileNotFoundError):
        load_splits(None, tmp_path / "nope.csv")
    with pytest.raises(KeyError, match="--dialogue-col"):
        load_splits(None, files["train"], dialogue_col="text")
    bad = tmp_path / "x.txt"
    bad.write_text("a", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_splits(None, bad)


def test_other_file_formats(tmp_path):
    frame = pd.DataFrame({"dialogue": ["A: hi\nB: yo", None, "C: x"], "summary": ["A greets B.", "x", ""]})
    for name, writer in (("d.jsonl", lambda p: frame.to_json(p, orient="records", lines=True)),
                         ("d.json", lambda p: frame.to_json(p, orient="records")),
                         ("d.parquet", lambda p: frame.to_parquet(p))):
        path = tmp_path / name
        writer(path)
        assert len(load_splits(None, path)["train"]) == 1          # empty rows are dropped


def test_hub_failure_gives_actionable_message(monkeypatch):
    import datasets

    def boom(*a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(datasets, "load_dataset", boom)
    with pytest.raises(RuntimeError, match="--train-file"):
        load_splits("some/dataset")


def test_sampling_and_stats(files):
    df = load_splits(None, files["train"])["train"]
    assert sample_df(df, 10, 1).equals(sample_df(df, 10, 1)) and len(sample_df(df, 0)) == len(df)
    stats = dataset_stats(df)
    assert stats["rows"] == len(df) and stats["turns_mean"] >= 2
    shots = pick_shots(df, 3, seed=1)
    assert len(shots) == 3 and shots == pick_shots(df, 3, seed=1) and pick_shots(df, 0) == []


def test_encoding_masks_the_prompt_and_keeps_the_summary(tiny_model):
    tok = AutoTokenizer.from_pretrained(tiny_model)
    ids, labels, truncated = encode_example(tok, "Anna: hi\nBen: yo", "Anna greets Ben.", 128)
    assert len(ids) == len(labels) and not truncated
    n_prompt = sum(x == -100 for x in labels)
    assert n_prompt > 0 and labels[n_prompt:] == ids[n_prompt:] and ids[-1] == tok.eos_token_id
    assert tok.decode(ids[n_prompt:-1]).strip() == "Anna greets Ben."          # exactly the summary is trained on


def test_long_dialogue_is_truncated_but_summary_survives(tiny_model):
    tok = AutoTokenizer.from_pretrained(tiny_model)
    ids, labels, truncated = encode_example(tok, "Anna: " + "word " * 500, "Short summary.", 96)
    assert truncated and len(ids) <= 96 and ids[-1] == tok.eos_token_id
    assert tok.decode([t for t in labels if t != -100]).strip().startswith("Short summary.")


def test_dataset_and_collate(files, tiny_model):
    tok = AutoTokenizer.from_pretrained(tiny_model)
    df = load_splits(None, files["train"])["train"].head(6)
    ds = SummarizationDataset(df, tok, 160)
    assert len(ds) == 6 and ds.n_target_tokens > 0 and ds.lengths == [len(x["input_ids"]) for x in ds.items]
    batch = collate([ds[0], ds[1], ds[2]], pad_token_id=0)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    short = min(range(3), key=lambda i: ds.lengths[i])
    assert (batch["labels"][short][ds.lengths[short]:] == -100).all()
    assert batch["attention_mask"][short].sum() == ds.lengths[short]


def test_batches_are_reproducible_and_complete():
    lengths = list(range(1, 38))
    a = make_batches(lengths, 4, seed=3, epoch=0)
    assert a == make_batches(lengths, 4, seed=3, epoch=0) and a != make_batches(lengths, 4, seed=3, epoch=1)
    assert sorted(i for b in a for i in b) == list(range(37)) and all(len(b) <= 4 for b in a)
    assert make_batches(lengths, 5, 0, 0, shuffle=False)[0] == [0, 1, 2, 3, 4]
