import pytest

from qlora_finetune import config
from qlora_finetune.testing import build_tiny_causal_lm

TRAIN_ARGS = ["--max-train-samples", "0", "--epochs", "10", "--batch-size", "8", "--grad-accum", "1", "--lr", "3e-3",
              "--max-seq-len", "160", "--logging-steps", "10", "--eval-rouge-samples", "6", "--max-new-tokens", "40",
              "--no-gradient-checkpointing", "--quantization", "none"]


@pytest.fixture(scope="session")
def sample_dir():
    assert (config.SAMPLE_DIR / "test.csv").exists(), "Run scripts/generate_sample_data.py"
    return config.SAMPLE_DIR


@pytest.fixture(scope="session")
def files(sample_dir):
    return {"train": str(sample_dir / "train.csv"), "validation": str(sample_dir / "validation.csv"),
            "test": str(sample_dir / "test.csv")}


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory, sample_dir):
    import pandas as pd

    df = pd.read_csv(sample_dir / "train.csv")
    return str(build_tiny_causal_lm(tmp_path_factory.mktemp("tiny_llama"), df["dialogue"].tolist() + df["summary"].tolist()))


@pytest.fixture(scope="session")
def trained_adapter(tmp_path_factory, files, tiny_model):
    """A tiny adapter trained once per test session (about ten seconds on CPU)."""
    from qlora_finetune import train

    out = tmp_path_factory.mktemp("adapter") / "adapter"
    root = tmp_path_factory.mktemp("reports")
    old = (config.METRICS_DIR, config.FIGURES_DIR)
    config.METRICS_DIR, config.FIGURES_DIR = root / "metrics", root / "figures"
    try:
        code = train.main(["--base-model", tiny_model, "--train-file", files["train"], "--validation-file",
                           files["validation"], "--output", str(out), "--no-plots", *TRAIN_ARGS])
    finally:
        config.METRICS_DIR, config.FIGURES_DIR = old
    assert code == 0
    return str(out)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Redirect every output path of the project into a temporary directory."""
    root = config.PROJECT_ROOT
    for name in ("MODELS_DIR", "REPORTS_DIR", "FIGURES_DIR", "METRICS_DIR", "PREDICTIONS_DIR", "README_PATH"):
        monkeypatch.setattr(config, name, tmp_path / getattr(config, name).relative_to(root))
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def merged_model(tmp_path_factory, tiny_model, trained_adapter):
    from qlora_finetune.merge import merge

    out = tmp_path_factory.mktemp("merged") / "merged"
    merge(tiny_model, trained_adapter, out)
    return str(out)
