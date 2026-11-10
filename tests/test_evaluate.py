import json

import pandas as pd
import pytest

from qlora_finetune import evaluate

COMMON = ["--max-samples", "0", "--max-new-tokens", "40", "--bootstrap", "50", "--quantization", "none", "--examples", "3"]


def args(files, tiny_model, *extra):
    return ["--base-model", tiny_model, "--train-file", files["train"], "--test-file", files["test"], *COMMON, *extra]


def test_end_to_end_comparison(isolated, files, tiny_model, trained_adapter):
    (isolated / "README.md").write_text(f"# t\n{evaluate.RESULTS_START}\nPLACEHOLDER_TEXT\n{evaluate.RESULTS_END}\n",
                                        encoding="utf-8")
    code = evaluate.main(args(files, tiny_model, "--adapter", trained_adapter, "--base-shots", "1",
                              "--reference-model", tiny_model, "--reference-shots", "1"))
    assert code == 0
    table = pd.read_csv(isolated / "reports/metrics/results.csv")
    assert list(table["kind"]) == ["base", "fewshot", "finetuned", "reference"]
    rouge = dict(zip(table["kind"], table["rougeL"]))
    assert rouge["finetuned"] > rouge["base"] + 20 and rouge["finetuned"] > rouge["fewshot"]
    assert table["rougeL_ci_low"].le(table["rougeL"]).all() and table["rougeL"].le(table["rougeL_ci_high"]).all()
    assert table["bleu"].between(0, 100).all() and (table["peak_memory_gb"] > 0).all()

    sig = pd.read_csv(isolated / "reports/metrics/significance.csv")
    assert set(sig["versus_kind"]) == {"base", "fewshot", "reference"} and set(sig["metric"]) == {"rouge1", "rouge2", "rougeL"}
    row = sig[(sig["versus_kind"] == "base") & (sig["metric"] == "rougeL")].iloc[0]
    assert row["ci_low"] > 0 and row["p_value"] < 0.01

    run = json.loads((isolated / "reports/metrics/results.json").read_text())
    assert run["n_examples"] == 40 and "LoRA" in run["primary"]
    assert any("significantly better" in s for s in run["statements"])
    assert len(list((isolated / "reports/predictions").glob("*.csv"))) == 4
    for f in ("rouge_comparison.png", "summary_lengths.png", "rouge_l_gain_histogram.png"):
        assert (isolated / "reports/figures" / f).exists()
    assert "Example 3" in (isolated / "reports/examples.md").read_text(encoding="utf-8")
    readme = (isolated / "README.md").read_text(encoding="utf-8")
    assert "PLACEHOLDER_TEXT" not in readme and "Findings" in readme and "ROUGE-L 95% CI" in readme


def test_merged_model_matches_the_adapter(isolated, files, tiny_model, trained_adapter, merged_model):
    code = evaluate.main(args(files, tiny_model, "--no-base", "--adapter", trained_adapter, "--merged", merged_model,
                              "--no-plots", "--no-update-readme"))
    assert code == 0
    table = pd.read_csv(isolated / "reports/metrics/results.csv").set_index("kind")
    assert abs(table.loc["finetuned", "rougeL"] - table.loc["merged", "rougeL"]) < 1.0


def test_unavailable_system_is_skipped(isolated, files, tiny_model, tmp_path):
    code = evaluate.main(args(files, tiny_model, "--adapter", str(tmp_path / "missing"), "--no-plots", "--no-update-readme"))
    assert code == 0 and list(pd.read_csv(isolated / "reports/metrics/results.csv")["kind"]) == ["base"]


def test_validation_split(isolated, files, tiny_model):
    code = evaluate.main(["--base-model", tiny_model, "--validation-file", files["validation"], "--split", "validation",
                          *COMMON, "--no-plots", "--no-update-readme"])
    assert code == 0 and json.loads((isolated / "reports/metrics/results.json").read_text())["n_examples"] == 30


def test_errors(isolated, files, tiny_model, tmp_path):
    assert evaluate.main(["--base-model", tiny_model, "--no-base", *COMMON]) == 1                       # nothing to evaluate
    assert evaluate.main(["--base-model", tiny_model, "--test-file", files["test"], "--split", "validation", *COMMON]) == 1
    assert evaluate.main(["--base-model", tiny_model, "--test-file", files["test"], "--base-shots", "2", *COMMON]) == 1
    assert evaluate.main(["--base-model", tiny_model, "--test-file", str(tmp_path / "x.csv"), *COMMON]) == 1
    assert evaluate.main(["--base-model", str(tmp_path / "missing"), "--test-file", files["test"], *COMMON]) == 1


def test_specs_and_helpers(tiny_model):
    ns = evaluate.parse_args(["--base-model", "phi-2", "--adapter", "models/a", "models/b", "--merged", "models/m",
                              "--reference-model", "mistral-7b", "--base-shots", "3"])
    specs = evaluate.build_specs(ns)
    assert [s["kind"] for s in specs] == ["base", "fewshot", "finetuned", "finetuned", "merged", "reference"]
    assert specs[0]["name"] == "phi-2 (zero-shot)" and specs[1]["name"] == "phi-2 + 3-shot"
    assert specs[-1]["name"] == "mistral-7b (zero-shot, reference)" and specs[4]["quantization"] == "m"
    assert evaluate.slug("phi-2 + LoRA (a/b)") == "phi-2_LoRA_a_b"
    assert evaluate.md_table(pd.DataFrame({"a": [1], "b": [2]})).count("|") == 9


@pytest.mark.parametrize("ci_low,ci_high,expected", [(1.0, 3.0, "significantly better"), (-3.0, -1.0, "significantly worse"),
                                                     (-1.0, 2.0, "not significantly different")])
def test_headline_wording_follows_the_confidence_interval(ci_low, ci_high, expected):
    sig = pd.DataFrame([{"system": "ft", "versus": "big", "versus_kind": "reference", "metric": "rougeL",
                         "delta": 1.0, "ci_low": ci_low, "ci_high": ci_high, "p_value": 0.1}])
    text = evaluate.headline(sig, "ft")[0]
    assert expected in text and "reference model" in text and evaluate.headline(sig, "other") == []
