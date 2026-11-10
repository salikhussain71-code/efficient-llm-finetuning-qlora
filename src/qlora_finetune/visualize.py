"""Charts. Every function saves a PNG and returns the output path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: works on servers / CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .resources import estimate_weight_memory_gb  # noqa: E402

PALETTE = ["#9AA5B1", "#1F4E79", "#2E7D32", "#F9A825", "#C62828", "#8E24AA"]
ROUGE_COLORS = {"rouge1": "#1F4E79", "rouge2": "#F9A825", "rougeL": "#2E7D32", "rougeLsum": "#8E24AA"}
ROUGE_LABELS = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "rougeLsum": "ROUGE-Lsum"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.25,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(log: pd.DataFrame, out_dir: Path) -> Path:
    """Training loss per logged step (left) and validation loss / ROUGE-L per epoch (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    train = log.dropna(subset=["loss"]) if "loss" in log else log.iloc[0:0]
    if len(train):
        axes[0].plot(train["step"], train["loss"], color=PALETTE[1], lw=1.6)
        if len(train) >= 8:
            smooth = train["loss"].rolling(max(len(train) // 8, 2), min_periods=1).mean()
            axes[0].plot(train["step"], smooth, color=PALETTE[4], lw=2, label="smoothed")
            axes[0].legend()
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Optimiser step")
    axes[0].set_ylabel("Cross-entropy (summary tokens)")
    val = log.dropna(subset=["val_loss"]) if "val_loss" in log else log.iloc[0:0]
    if len(val):
        axes[1].plot(val["epoch"], val["val_loss"], "o-", color=PALETTE[2], label="validation loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation loss")
        if "val_rougeL" in val and val["val_rougeL"].notna().any():
            ax2 = axes[1].twinx()
            ax2.plot(val["epoch"], val["val_rougeL"], "s--", color=PALETTE[3], label="validation ROUGE-L")
            ax2.set_ylabel("ROUGE-L")
            ax2.grid(False)
            ax2.spines["right"].set_visible(True)
            axes[1].legend(loc="upper left")
            ax2.legend(loc="upper right")
        else:
            axes[1].legend()
    axes[1].set_title("Validation")
    return _save(fig, out_dir / "training_curves.png")


def plot_rouge_comparison(table: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bars: four ROUGE variants per system, in the order given by ``table``."""
    types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    systems = list(table["system"])
    x = np.arange(len(systems))
    w = 0.8 / len(types)
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(systems) + 2), 5))
    for k, t in enumerate(types):
        bars = ax.bar(x + (k - 1.5) * w, table[t], width=w * 0.92, color=ROUGE_COLORS[t], label=ROUGE_LABELS[t])
        ax.bar_label(bars, fmt="{:.1f}", fontsize=7, padding=2)
    ax.set_xticks(x, [s.replace(" (", "\n(") for s in systems], fontsize=9)
    ax.set_ylabel("F1 x 100")
    ax.set_ylim(0, max(table[types].max()) * 1.15)
    ax.set_title("ROUGE on the SAMSum test set")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir / "rouge_comparison.png")


def plot_delta_histogram(delta: np.ndarray, label: str, out_dir: Path) -> Path:
    """Per-dialogue ROUGE-L gain of the fine-tuned model over the baseline."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(delta, bins=30, color=PALETTE[1], alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(float(np.mean(delta)), color=PALETTE[4], lw=2, ls="--", label=f"mean {np.mean(delta):+.1f}")
    ax.set_xlabel("ROUGE-L difference per dialogue")
    ax.set_ylabel("Dialogues")
    ax.set_title(f"Per-example gain: {label}")
    ax.legend()
    return _save(fig, out_dir / "rouge_l_gain_histogram.png")


def plot_length_distributions(lengths: dict[str, list[int]], out_dir: Path) -> Path:
    names = list(lengths)
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(names) + 2))
    import inspect

    key = "tick_labels" if "tick_labels" in inspect.signature(plt.Axes.boxplot).parameters else "labels"
    box = ax.boxplot([lengths[n] for n in names], vert=False, patch_artist=True, showfliers=False, **{key: names})
    for patch, color in zip(box["boxes"], PALETTE * 2):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Words per summary")
    ax.set_title("Summary length by system")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir / "summary_lengths.png")


def plot_memory_estimate(params_b: float, name: str, measured_gb: float | None, out_dir: Path) -> Path:
    """Weights-only memory at different precisions, with the measured training peak for comparison."""
    est = estimate_weight_memory_gb(params_b)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(list(est), list(est.values()), color=[PALETTE[0], PALETTE[1], PALETTE[3], PALETTE[2]])
    ax.bar_label(bars, fmt="{:.1f} GB", padding=3)
    ax.axhline(16, color=PALETTE[4], ls="--", lw=1.5, label="16 GB (Colab T4)")
    if measured_gb:
        ax.axhline(measured_gb, color="k", ls=":", lw=1.5, label=f"measured training peak {measured_gb:.1f} GB")
    ax.set_ylabel("Weights only (GB)")
    ax.set_title(f"{name} ({params_b:g}B parameters): memory of the weights by precision")
    ax.legend()
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir / "memory_estimate.png")
