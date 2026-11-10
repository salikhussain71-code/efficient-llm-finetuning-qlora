"""ROUGE, BLEU, bootstrap confidence intervals and paired significance tests."""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer

from . import config

logging.getLogger("absl").setLevel(logging.WARNING)   # rouge-score logs "Using default tokenizer" on every call

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")
LABELS = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "rougeLsum": "ROUGE-Lsum"}
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_TOKEN = re.compile(r"[a-z0-9]+")


def _lines(text: str) -> str:
    """Sentences on separate lines (needed by ROUGE-Lsum)."""
    return "\n".join(s for s in _SENT.split(text.strip()) if s)


def score_pairs(predictions: list[str], references: list[str], use_stemmer: bool = True) -> pd.DataFrame:
    """Per-example precision / recall / F1 (x 100) for every ROUGE type."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=use_stemmer)
    rows = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(_lines(ref), _lines(pred))
        row = {}
        for t in ROUGE_TYPES:
            row[f"{t}_p"], row[f"{t}_r"], row[f"{t}_f"] = (scores[t].precision * 100, scores[t].recall * 100,
                                                          scores[t].fmeasure * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05, seed: int = config.RANDOM_STATE) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        m = float(values.mean()) if len(values) else 0.0
        return m, m
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def aggregate(per_example: pd.DataFrame, n_boot: int = 1000, seed: int = config.RANDOM_STATE) -> dict:
    """Mean F1 per ROUGE type with a 95 % bootstrap CI."""
    out: dict[str, float] = {}
    for t in ROUGE_TYPES:
        f = per_example[f"{t}_f"].to_numpy()
        lo, hi = bootstrap_ci(f, n_boot, seed=seed)
        out[t], out[f"{t}_ci_low"], out[f"{t}_ci_high"] = float(f.mean()), lo, hi
    return out


def bleu(predictions: list[str], references: list[str]) -> float:
    """Corpus-level BLEU (sacreBLEU defaults), 0-100."""
    if not predictions:
        return 0.0
    return float(sacrebleu.corpus_bleu(predictions, [references]).score)


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = config.RANDOM_STATE) -> dict:
    """Paired bootstrap of the mean difference ``a - b`` over the same examples.

    ``p_value`` is the share of resamples in which ``a`` is not better than ``b`` (one-sided: small = ``a``
    is reliably better). The CI is the 95 % percentile interval of the mean difference.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    diff = a - b
    if len(diff) == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return {"delta": float(diff.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "p_value": float((means <= 0).mean())}


def _bigrams(text: str) -> set[tuple[str, str]]:
    toks = _TOKEN.findall(text.lower())
    return set(zip(toks, toks[1:]))


def text_stats(predictions: list[str], dialogues: list[str], references: list[str]) -> dict:
    """Length and abstractiveness statistics of a system's summaries."""
    pw = np.array([len(p.split()) for p in predictions], dtype=float)
    rw = np.array([len(r.split()) for r in references], dtype=float)
    novel = []
    for p, d in zip(predictions, dialogues):
        grams = _bigrams(p)
        novel.append(len(grams - _bigrams(d)) / len(grams) if grams else 0.0)
    return {"summary_words": float(pw.mean()) if len(pw) else 0.0, "reference_words": float(rw.mean()) if len(rw) else 0.0,
            "empty_share": float((pw == 0).mean()) if len(pw) else 0.0,
            "novel_bigrams": float(np.mean(novel)) if novel else 0.0}
