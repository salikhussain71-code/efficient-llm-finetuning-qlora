import sys
import time

import numpy as np
import pandas as pd
import pytest
import torch

from qlora_finetune import modeling, resources
from qlora_finetune.metrics import (
    aggregate,
    bleu,
    bootstrap_ci,
    paired_bootstrap,
    score_pairs,
    text_stats,
)
from qlora_finetune.presets import PRESETS, display_name, find_preset, resolve_model_id
from qlora_finetune.tracking import Tracker, TrackingUnavailable


# ------------------------------------------------------------------------------------------- presets
def test_presets():
    assert find_preset("phi-2").hub_id == "microsoft/phi-2" and find_preset("mistralai/Mistral-7B-v0.1").key == "mistral-7b"
    assert find_preset("nope") is None and PRESETS["llama3-8b"].gated
    assert resolve_model_id("phi-2") == "microsoft/phi-2" and resolve_model_id("some/model") == "some/model"
    assert display_name("microsoft/phi-2") == "phi-2" and display_name("models/my-merged/") == "my-merged"


# ----------------------------------------------------------------------------------------- quantisation
def test_bnb_config_is_nf4_with_double_quantisation():
    cfg = modeling.build_bnb_config("4bit", torch.float16)
    assert cfg.load_in_4bit and cfg.bnb_4bit_quant_type == "nf4" and cfg.bnb_4bit_use_double_quant
    assert cfg.bnb_4bit_compute_dtype == torch.float16
    assert modeling.build_bnb_config("8bit", torch.float16).load_in_8bit
    assert modeling.build_bnb_config("none", torch.float16) is None
    with pytest.raises(ValueError):
        modeling.build_bnb_config("2bit", torch.float16)


def test_quantisation_without_a_gpu_gives_an_actionable_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(modeling.QuantizationUnavailable, match="NVIDIA GPU"):
        modeling.require_quantization_support("4bit")
    modeling.require_quantization_support("none")                       # no requirement


def test_quantisation_without_bitsandbytes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    with pytest.raises(modeling.QuantizationUnavailable, match="pip install bitsandbytes"):
        modeling.require_quantization_support("4bit")


def test_dtype_helpers():
    cpu = torch.device("cpu")
    assert modeling.compute_dtype(cpu) == torch.float32 and modeling.resolve_dtype("float16", cpu) == torch.float16
    assert modeling.resolve_dtype("auto", cpu) == torch.float32 and modeling.pick_device("cpu").type == "cpu"


# ----------------------------------------------------------------------------------------------- loading
def test_tokenizer_gets_a_pad_token(tiny_model):
    tok = modeling.load_tokenizer(tiny_model, padding_side="left")
    assert tok.pad_token == tok.eos_token and tok.padding_side == "left"


def test_missing_model_message(tmp_path):
    with pytest.raises(modeling.ModelLoadError, match="Hugging Face Hub"):
        modeling.load_base_model(str(tmp_path / "missing"))
    with pytest.raises(modeling.ModelLoadError):
        modeling.load_tokenizer(str(tmp_path / "missing"))


def test_lora_wraps_only_a_small_share_of_parameters(tiny_model):
    model = modeling.load_base_model(tiny_model)
    all_linear = modeling.attach_lora(model, modeling.build_lora_config(r=8, target_modules="all-linear"))
    summary_all = modeling.parameter_summary(all_linear)
    assert 0 < summary_all["trainable"] < summary_all["total"] and 0 < summary_all["trainable_percent"] < 100

    model2 = modeling.load_base_model(tiny_model)
    qv = modeling.attach_lora(model2, modeling.build_lora_config(r=8, target_modules=["q_proj", "v_proj"]))
    assert modeling.parameter_summary(qv)["trainable"] < summary_all["trainable"]
    assert all("lora_" in n for n, p in qv.named_parameters() if p.requires_grad)     # the base stays frozen
    assert modeling.memory_footprint_gb(qv) > 0


def test_prepare_for_training_enables_checkpointing(tiny_model):
    model = modeling.prepare_for_training(modeling.load_base_model(tiny_model), "none", True)
    assert model.is_gradient_checkpointing and model.config.use_cache is False


# ---------------------------------------------------------------------------------------------- resources
def test_memory_estimates_follow_the_bit_width():
    est = resources.estimate_weight_memory_gb(7.0)
    assert est["fp16/bf16"] == pytest.approx(14.0) and est["fp32"] == pytest.approx(28.0)
    assert est["8-bit"] == pytest.approx(7.0) and 3.5 < est["4-bit (NF4)"] < 4.0 and est["4-bit (NF4)"] < est["8-bit"]


def test_resource_helpers(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 1_000_000)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"x" * 500_000)
    assert resources.dir_size_mb(tmp_path) == pytest.approx(1.5)
    assert resources.dir_size_mb(tmp_path / "a.bin") == pytest.approx(1.0)
    assert resources.peak_memory_gb(torch.device("cpu")) > 0
    with resources.Timer() as t:
        time.sleep(0.01)
    assert t.seconds >= 0.01 and resources.environment()["torch"]
    assert resources.gpu_info() is None or "name" in resources.gpu_info()


# ----------------------------------------------------------------------------------------------- tracking
def test_tracker_writes_csv(tmp_path):
    tracker = Tracker(tmp_path / "run")
    tracker.log({"loss": 1.0}, 1)
    tracker.log({"loss": 0.5, "lr": 1e-4}, 2)
    tracker.summary({"x": 1})                           # no-op without W&B
    tracker.finish()
    frame = pd.read_csv(tmp_path / "run" / "train_log.csv")
    assert list(frame["step"]) == [1, 2] and frame["loss"].tolist() == [1.0, 0.5]


def test_tracker_wandb_calls(tmp_path, monkeypatch):
    calls = []

    class FakeRun:
        summary = type("S", (), {"update": staticmethod(lambda v: calls.append(("summary", v)))})()

    class FakeWandb:
        run = FakeRun()

        @staticmethod
        def init(**kw):
            calls.append(("init", kw["project"], kw["name"]))
            return FakeRun()

        @staticmethod
        def log(metrics, step):
            calls.append(("log", step, metrics))

        @staticmethod
        def finish():
            calls.append(("finish",))

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    tracker = Tracker(tmp_path, "wandb", "proj", "run1", {"a": 1})
    tracker.log({"loss": 2.0}, 3)
    tracker.summary({"best": 1})
    tracker.finish()
    assert [c[0] for c in calls] == ["init", "log", "summary", "finish"] and calls[0][1:] == ("proj", "run1")


def test_tracker_errors(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(TrackingUnavailable, match="pip install wandb"):
        Tracker(tmp_path, "wandb")
    with pytest.raises(ValueError):
        Tracker(tmp_path, "tensorboard")


# ------------------------------------------------------------------------------------------------ metrics
def test_rouge_and_bleu_known_values():
    s = score_pairs(["Anna asks Ben to come on Friday."], ["Anna asks Ben to come on Friday."])
    assert s["rouge1_f"][0] == pytest.approx(100) and s["rougeL_f"][0] == pytest.approx(100)
    assert score_pairs(["alpha beta"], ["gamma delta"])["rouge1_f"][0] == 0
    assert score_pairs([""], ["something"])["rougeL_f"][0] == 0
    assert bleu(["the cat sat on the mat"], ["the cat sat on the mat"]) == pytest.approx(100) and bleu([], []) == 0
    with pytest.raises(ValueError):
        score_pairs(["a"], [])


def test_multi_sentence_rouge_lsum_is_order_insensitive():
    s = score_pairs(["A man was arrested. Police said he was calm."], ["Police said he was calm. A man was arrested."])
    assert s["rougeLsum_f"][0] == pytest.approx(100)


def test_confidence_interval_and_aggregate():
    values = np.random.default_rng(0).normal(40, 5, 80)
    lo, hi = bootstrap_ci(values, 500, seed=1)
    assert lo < values.mean() < hi and (lo, hi) == bootstrap_ci(values, 500, seed=1)
    assert bootstrap_ci([3.0]) == (3.0, 3.0) and bootstrap_ci([]) == (0.0, 0.0)
    agg = aggregate(score_pairs(["a b c", "d e f"], ["a b c", "x y z"]), n_boot=50)
    assert agg["rouge1"] == pytest.approx(50) and agg["rougeL_ci_low"] <= agg["rougeL"] <= agg["rougeL_ci_high"]


def test_paired_bootstrap_detects_real_and_absent_differences():
    rng = np.random.default_rng(1)
    base = rng.normal(30, 8, 100)
    better = base + rng.normal(6, 3, 100)
    sig = paired_bootstrap(better, base, 500)
    assert sig["delta"] > 4 and sig["ci_low"] > 0 and sig["p_value"] < 0.01
    noise = np.tile([4.0, -4.0], 50)                       # symmetric: the true mean difference is exactly zero
    same = paired_bootstrap(base + noise, base, 500)
    assert same["delta"] == pytest.approx(0) and same["ci_low"] < 0 < same["ci_high"] and same["p_value"] > 0.05
    assert paired_bootstrap([], [])["p_value"] == 1.0
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 2.0])


def test_text_stats():
    s = text_stats(["Anna and Ben meet.", ""], ["Anna: hi\nBen: yo", "A: x"], ["one two three", "one"])
    assert s["summary_words"] == 2.0 and s["empty_share"] == 0.5 and 0 <= s["novel_bigrams"] <= 1


def test_reports_dir_can_be_redirected_by_environment(tmp_path):
    """`make smoke` relies on this so synthetic results never land in reports/."""
    import os
    import subprocess

    code = "from qlora_finetune import config; print(config.METRICS_DIR, config.FIGURES_DIR, config.PREDICTIONS_DIR)"
    env = {**os.environ, "QF_REPORTS_DIR": str(tmp_path / "elsewhere")}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert out.split() == [str(tmp_path / "elsewhere" / p) for p in ("metrics", "figures", "predictions")]
    default = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert default.split()[0].endswith("reports/metrics")
