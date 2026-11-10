import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from qlora_finetune import merge, modelcard, push
from qlora_finetune.generate import SummaryGenerator


# ------------------------------------------------------------------------------------------------ merge
def test_merged_model_is_standalone_and_equivalent(tiny_model, trained_adapter, merged_model, files):
    out = Path(merged_model)
    assert {"config.json", "model.safetensors", "tokenizer.json", "merge_report.json"} <= {p.name for p in out.iterdir()}
    assert not (out / "adapter_config.json").exists()                             # no adapter needed any more

    report = json.loads((out / "merge_report.json").read_text())
    ver = report["verification"]
    assert ver["identical_outputs"] == ver["dialogues"] == 3 and ver["max_abs_logit_diff"] < 1e-3
    assert report["merged_size_mb"] > report["adapter_size_mb"] > 0 and report["save_dtype"] == "float32"

    dev = pd.read_csv(files["validation"])["dialogue"].tolist()[:10]
    with_adapter = SummaryGenerator.load(tiny_model, trained_adapter, max_new_tokens=30).generate(dev)
    standalone = SummaryGenerator.load(merged_model, max_new_tokens=30).generate(dev)       # loads WITHOUT peft
    assert with_adapter == standalone


def test_merge_without_verification_and_half_precision(tiny_model, trained_adapter, merged_model, tmp_path):
    report = merge.merge(tiny_model, trained_adapter, tmp_path / "m", verify=False, save_dtype="float16")
    assert "verification" not in report and report["save_dtype"] == "float16"
    reference = json.loads((Path(merged_model) / "merge_report.json").read_text())
    assert 0 < report["merged_size_mb"] < reference["merged_size_mb"]            # float16 is smaller than float32


def test_merge_cli(tiny_model, trained_adapter, tmp_path, capsys):
    out = tmp_path / "cli"
    assert merge.main(["--base-model", tiny_model, "--adapter", trained_adapter, "--output", str(out), "--no-verify"]) == 0
    assert (out / "model.safetensors").exists() and "Merged model saved" in capsys.readouterr().out
    assert merge.main(["--base-model", tiny_model, "--adapter", str(tmp_path / "nothing")]) == 1
    assert merge.main(["--base-model", str(tmp_path / "nomodel"), "--adapter", trained_adapter,
                       "--output", str(tmp_path / "x")]) == 1


# ------------------------------------------------------------------------------------------- model card
TRAINING = {"base_model": "microsoft/phi-2", "quantization": "4bit",
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
            "parameters": {"trainable": 1000, "total": 100000, "trainable_percent": 1.0},
            "data": {"train_examples": 4000}, "optimisation": {"epochs": 1, "steps": 250, "effective_batch": 16, "lr": 2e-4,
                                                             "scheduler": "cosine"},
            "resources": {"peak_memory_gb": 9.5, "gpu": {"name": "Tesla T4"}, "device": "cuda"}}
RESULTS = {"n_examples": 200, "split": "test", "results": [
    {"system": "phi-2 (zero-shot)", "rouge1": 20.1, "rouge2": 5.0, "rougeL": 15.0, "rougeLsum": 15.1, "bleu": 2.0},
    {"system": "phi-2 + LoRA (a)", "rouge1": 50.0, "rouge2": 25.0, "rougeL": 42.0, "rougeLsum": 42.1, "bleu": 20.0}]}


def test_model_card_content():
    card = modelcard.build_model_card("adapter", "me/phi2-samsum", "microsoft/phi-2", "knkarthick/samsum", TRAINING, RESULTS)
    assert card.startswith("---\nbase_model: microsoft/phi-2\nlibrary_name: peft")
    for expected in ("- peft", "- samsum", "PeftModel.from_pretrained(base, \"me/phi2-samsum\")", "QLoRA (4-bit NF4",
                     "### Dialogue:", "| Peak memory | 9.5 GB |", "Tesla T4", "| phi-2 + LoRA (a) | 50.00 |", "CC BY-NC-ND 4.0"):
        assert expected in card, expected
    merged = modelcard.build_model_card("merged", "me/m", "microsoft/phi-2")
    assert "library_name: transformers" in merged and 'AutoModelForCausalLM.from_pretrained("me/m"' in merged
    assert "Run `python -m qlora_finetune.evaluate`" in merged and "## Training" not in merged
    plain = modelcard.build_model_card("adapter", "me/x", "b", training={**TRAINING, "quantization": "none"})
    assert "LoRA (trainable low-rank adapters)" in plain and "QLoRA (4-bit" not in plain
    with pytest.raises(ValueError):
        modelcard.build_model_card("full", "me/x", "b")
    assert modelcard.load_json(None) is None and modelcard.load_json("/no/such/file.json") is None


# ----------------------------------------------------------------------------------------------- push
@pytest.fixture
def adapter_copy(trained_adapter, tmp_path):
    import shutil

    dest = tmp_path / "my-adapter"
    shutil.copytree(trained_adapter, dest)
    (dest / "checkpoints" / "step-000010").mkdir(parents=True)
    (dest / "checkpoints" / "step-000010" / "trainer_state.pt").write_bytes(b"x")
    (dest / "optimizer.pt").write_bytes(b"x")
    return dest


def test_kind_detection_and_file_selection(adapter_copy, merged_model, tmp_path):
    assert push.detect_kind(adapter_copy) == "adapter" and push.detect_kind(Path(merged_model)) == "merged"
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        push.detect_kind(tmp_path / "empty")
    names = {p.name for p in push.files_to_upload(adapter_copy)}
    assert "adapter_model.safetensors" in names and not ({"trainer_state.pt", "optimizer.pt", "train_log.csv"} & names)


def test_dry_run_writes_a_real_model_card_and_uploads_nothing(adapter_copy, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)            # any attempt to import the hub client would fail
    summary = str(adapter_copy / "training_summary.json")
    assert push.main(["--path", str(adapter_copy), "--dry-run", "--training-summary", summary]) == 0
    out = capsys.readouterr().out
    assert "salikhussain71-code/my-adapter (private)" in out and "Dry run" in out and "checkpoints" not in out
    card = (adapter_copy / "README.md").read_text(encoding="utf-8")
    assert "[More Information Needed]" not in card and "library_name: peft" in card and "## Training" in card


def test_custom_card_is_kept_unless_overwrite_requested(adapter_copy):
    (adapter_copy / "README.md").write_text("# my own card\n", encoding="utf-8")
    push.main(["--path", str(adapter_copy), "--dry-run"])
    assert (adapter_copy / "README.md").read_text(encoding="utf-8") == "# my own card\n"
    push.main(["--path", str(adapter_copy), "--dry-run", "--overwrite-card"])
    assert "library_name: peft" in (adapter_copy / "README.md").read_text(encoding="utf-8")


def fake_hub(monkeypatch, fail=False):
    calls = []

    class Api:
        def __init__(self, token=None):
            calls.append(("token", token))

        def create_repo(self, **kw):
            if fail:
                raise PermissionError("invalid token")
            calls.append(("create", kw))

        def upload_folder(self, **kw):
            calls.append(("upload", kw))

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Api))
    return calls


def test_upload_is_private_by_default_and_skips_checkpoints(adapter_copy, monkeypatch, capsys):
    calls = fake_hub(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    assert push.main(["--path", str(adapter_copy), "--repo-id", "someone/my-adapter", "--commit-message", "v1"]) == 0
    by = {c[0]: c[1] for c in calls}
    assert by["token"] == "hf_test" and by["create"]["private"] is True and by["create"]["repo_id"] == "someone/my-adapter"
    assert by["upload"]["commit_message"] == "v1" and "checkpoints/*" in by["upload"]["ignore_patterns"]
    assert "https://huggingface.co/someone/my-adapter" in capsys.readouterr().out


def test_public_flag_and_explicit_token(adapter_copy, monkeypatch):
    calls = fake_hub(monkeypatch)
    assert push.main(["--path", str(adapter_copy), "--public", "--token", "hf_x"]) == 0
    assert ("token", "hf_x") in calls and next(c[1] for c in calls if c[0] == "create")["private"] is False


def test_push_errors(adapter_copy, tmp_path, monkeypatch):
    assert push.main(["--path", str(tmp_path / "nope")]) == 1
    (tmp_path / "e").mkdir()
    assert push.main(["--path", str(tmp_path / "e")]) == 1
    assert push.main(["--path", str(adapter_copy), "--repo-id", "not a valid id", "--dry-run"]) == 1
    fake_hub(monkeypatch, fail=True)
    assert push.main(["--path", str(adapter_copy)]) == 1                    # authentication / permission failure


def test_card_helper():
    assert push.is_autogenerated_card("... [More Information Needed] ...") and not push.is_autogenerated_card("# mine")
