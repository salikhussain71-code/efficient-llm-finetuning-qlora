"""Central configuration: paths and constants shared across the project."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("QF_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"

MODELS_DIR = PROJECT_ROOT / "models"
# QF_REPORTS_DIR lets smoke tests write somewhere else, so synthetic results never end up in reports/
REPORTS_DIR = Path(os.environ.get("QF_REPORTS_DIR", PROJECT_ROOT / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
README_PATH = PROJECT_ROOT / "README.md"

RANDOM_STATE = 42

# SAMSum on the Hugging Face Hub (columns: id, dialogue, summary; 14,732 / 818 / 819 examples)
DEFAULT_DATASET_ID = "knkarthick/samsum"
DIALOGUE_COL = "dialogue"
SUMMARY_COL = "summary"
SPLITS = ("train", "validation", "test")
