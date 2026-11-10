import json
import sys

import pandas as pd
import pytest
from safetensors.torch import load_file

from qlora_finetune import train
from qlora_finetune.generate import SummaryGenerator
from qlora_finetune.metrics import aggregate, score_pairs
from qlora_finetune.modeling import QuantizationUnavailable

QUICK = ["--max-train-samples", "16", "--max-steps", "2", "--batch-size", "4", "--grad-accum", "1",
         "--max-seq-len", "128", "--eval-rouge-samples", "0", "--max-new-tokens", "10", "--no-plots",
         "--quantization", "none", "--max-validation-samples", "8"]


def run(tiny_model, files, out, *extra):
    return train.main(["--base-model", tiny_model, "--train-file", files["train"], "--validation-file", files["validation"],
                       "--output", str(out), *extra])


def test_training_produces_a_better_summariser(trained_adapter, tiny_model, files):
    """The adapter trained once per session must beat the untrained base model by a wide margin."""
    summary = json.loads((__import__("pathlib").Path(trained_adapter) / "training_summary.json").read_text())
    assert summary["parameters"]["trainable"] > 0 and 0 < summary["parameters"]["trainable_percent"] < 100
    val = summary["validation"]
    assert len(val) == 10 and val[-1]["val_loss"] < val[0]["val_loss"]
    assert summary["best_val_loss"] == pytest.approx(min(v["val_loss"] for v in val))
    assert summary["resources"]["adapter_size_mb"] > 0 and summary["resources"]["tokens_seen"] > 0
    assert summary["quantization"] == "none" and summary["optimisation"]["effective_batch"] == 8

    dev = pd.read_csv(files["validation"])
    dialogues, refs = dev["dialogue"].tolist(), dev["summary"].tolist()
    base = SummaryGenerator.load(tiny_model, max_new_tokens=40).generate(dialogues)
    tuned = SummaryGenerator.load(tiny_model, trained_adapter, max_new_tokens=40).generate(dialogues)
    b, t = aggregate(score_pairs(base, refs), 50)["rougeL"], aggregate(score_pairs(tuned, refs), 50)["rougeL"]
    assert t > b + 20


def test_adapter_folder_contents(trained_adapter):
    from pathlib import Path

    names = {p.name for p in Path(trained_adapter).iterdir()}
    assert {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "training_summary.json"} <= names
    assert pd.read_csv(Path(trained_adapter) / "train_log.csv")["loss"].notna().any()


def test_resumed_training_matches_uninterrupted_training(isolated, tiny_model, files, tmp_path):
    """Pause at step 10 of 25, resume, finish: the adapter must equal that of an uninterrupted run."""
    common = ["--max-train-samples", "0", "--batch-size", "8", "--grad-accum", "1", "--lr", "3e-3", "--lora-dropout", "0",
              "--max-seq-len", "128", "--eval-rouge-samples", "0", "--no-plots", "--quantization", "none",
              "--no-gradient-checkpointing", "--epochs", "1", "--max-validation-samples", "8"]
    straight, paused, resumed = tmp_path / "straight", tmp_path / "paused", tmp_path / "resumed"
    assert run(tiny_model, files, straight, *common) == 0
    assert run(tiny_model, files, paused, *common, "--stop-after-steps", "10") == 0
    ckpt = paused / "checkpoints" / "step-000010"
    assert ckpt.exists()
    assert run(tiny_model, files, resumed, *common, "--resume-from", str(ckpt)) == 0

    a, b = load_file(str(straight / "adapter_model.safetensors")), load_file(str(resumed / "adapter_model.safetensors"))
    assert a.keys() == b.keys() and max(float((a[k] - b[k]).abs().max()) for k in a) < 1e-4
    state = __import__("torch").load(ckpt / "trainer_state.pt", map_location="cpu", weights_only=False)
    assert state["global_step"] == 10 and {"optimizer", "scheduler", "scaler", "epoch", "batch_in_epoch"} <= set(state)
    assert json.loads((paused / "training_summary.json").read_text())["optimisation"]["steps"] == 10
    assert json.loads((resumed / "training_summary.json").read_text())["optimisation"]["steps"] == 25


def test_periodic_checkpoints(isolated, tiny_model, files, tmp_path):
    assert run(tiny_model, files, tmp_path / "c", *QUICK, "--max-steps", "4", "--save-steps", "2") == 0
    assert sorted(p.name for p in (tmp_path / "c" / "checkpoints").iterdir()) == ["step-000002", "step-000004"]


@pytest.mark.parametrize("scheduler", ["cosine", "linear", "constant"])
def test_lr_schedulers_run(isolated, tiny_model, files, tmp_path, scheduler):
    assert run(tiny_model, files, tmp_path / scheduler, *QUICK, "--lr-scheduler", scheduler) == 0


def test_gradient_checkpointing_and_accumulation(isolated, tiny_model, files, tmp_path):
    code = run(tiny_model, files, tmp_path / "gc", *[a for a in QUICK if a != "--max-steps" and a != "2"], "--max-steps", "3",
               "--grad-accum", "2")
    assert code == 0
    summary = json.loads((tmp_path / "gc" / "training_summary.json").read_text())
    assert summary["optimisation"]["gradient_checkpointing"] is True and summary["optimisation"]["effective_batch"] == 8


def test_target_module_selection(isolated, tiny_model, files, tmp_path):
    run(tiny_model, files, tmp_path / "all", *QUICK)
    run(tiny_model, files, tmp_path / "qv", *QUICK, "--target-modules", "q_proj,v_proj")
    n_all = json.loads((tmp_path / "all" / "training_summary.json").read_text())["parameters"]["trainable"]
    n_qv = json.loads((tmp_path / "qv" / "training_summary.json").read_text())["parameters"]["trainable"]
    assert 0 < n_qv < n_all


def test_validation_is_carved_out_when_missing(isolated, tiny_model, files, tmp_path):
    code = train.main(["--base-model", tiny_model, "--train-file", files["train"], "--output", str(tmp_path / "o"),
                       *QUICK])
    assert code == 0
    summary = json.loads((tmp_path / "o" / "training_summary.json").read_text())
    assert summary["data"]["validation_examples"] > 0


def test_outputs_and_plot(isolated, tiny_model, files, tmp_path):
    code = run(tiny_model, files, tmp_path / "p", *[a for a in QUICK if a != "--no-plots"], "--logging-steps", "1")
    assert code == 0
    assert (isolated / "reports/figures/training_curves.png").exists() and (isolated / "reports/metrics/train_log.csv").exists()
    assert (isolated / "reports/metrics/training_summary.json").exists()


def test_rouge_is_tracked_during_training(isolated, tiny_model, files, tmp_path):
    code = run(tiny_model, files, tmp_path / "r", *[a for a in QUICK if a not in ("--eval-rouge-samples", "0")],
               "--eval-rouge-samples", "4")
    assert code == 0
    log = pd.read_csv(tmp_path / "r" / "train_log.csv")
    assert "val_rougeL" in log and log["val_rougeL"].notna().any()


def test_weights_and_biases_offline(isolated, tiny_model, files, tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    code = run(tiny_model, files, tmp_path / "w", *QUICK, "--report-to", "wandb", "--run-name", "unit-test",
               "--logging-steps", "1")
    assert code == 0 and (tmp_path / "w" / "wandb").exists()


def test_error_handling(isolated, tiny_model, files, tmp_path, monkeypatch):
    assert train.main(["--base-model", tiny_model, "--train-file", str(tmp_path / "missing.csv"), *QUICK]) == 1
    assert train.main(["--base-model", str(tmp_path / "no-model"), "--train-file", files["train"], *QUICK]) == 1
    assert run(tiny_model, files, tmp_path / "q", *[a for a in QUICK if a not in ("--quantization", "none")],
               "--quantization", "4bit") == 1                                      # no GPU here
    monkeypatch.setitem(sys.modules, "wandb", None)
    assert run(tiny_model, files, tmp_path / "w", *QUICK, "--report-to", "wandb") == 1
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    assert run(tiny_model, files, tmp_path / "o", *QUICK, "--optim", "paged_adamw_8bit") == 1


def test_helpers():
    import torch

    assert train.resolve_quantization("auto", torch.device("cpu")) == "none"
    assert train.resolve_quantization("auto", torch.device("cuda")) == "4bit"
    assert train.resolve_quantization("8bit", torch.device("cpu")) == "8bit"
    with pytest.raises(QuantizationUnavailable):
        sys.modules["bitsandbytes"] = None
        try:
            train.make_optimizer([], type("A", (), {"optim": "paged_adamw_8bit", "lr": 1e-3, "weight_decay": 0})())
        finally:
            sys.modules.pop("bitsandbytes", None)
