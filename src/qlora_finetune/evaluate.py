"""Compare the base model, the fine-tuned model and a larger reference model on SAMSum.

    python -m qlora_finetune.evaluate --base-model phi-2 --adapter models/phi-2-samsum-qlora \\
        --reference-model mistral-7b --max-samples 200
    python -m qlora_finetune.evaluate --base-model phi-2 --adapter models/a models/b --base-shots 3
    python -m qlora_finetune.evaluate --base-model phi-2 --adapter models/phi-2-samsum-qlora \\
        --merged models/phi-2-samsum-merged                      # check that merging preserved the quality

All systems summarise the **same** dialogues with the **same** prompt and the same output cleaning. Reported:
ROUGE-1/2/L/Lsum and BLEU with bootstrap confidence intervals, paired significance tests, length and
abstractiveness, and the resources each system needed (peak memory, time).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import __version__, config, resources, visualize
from .data import dataset_stats, load_splits, pick_shots, sample_df
from .generate import SummaryGenerator
from .metrics import ROUGE_TYPES, aggregate, bleu, paired_bootstrap, score_pairs, text_stats
from .modeling import (
    DTYPES,
    QUANTIZATION_MODES,
    ModelLoadError,
    QuantizationUnavailable,
    parameter_summary,
    pick_device,
)
from .presets import display_name, find_preset

logger = logging.getLogger("qlora_finetune")

RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"
COMPARE = ("rouge1", "rouge2", "rougeL")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate base vs fine-tuned vs reference summarisers.")
    g = p.add_argument_group("systems")
    g.add_argument("--base-model", default="phi-2", help="The small model that was fine-tuned.")
    g.add_argument("--adapter", nargs="*", default=[], help="LoRA adapter folder(s) of the fine-tuned model(s).")
    g.add_argument("--merged", nargs="*", default=[], help="Merged (standalone) model folder(s).")
    g.add_argument("--reference-model", default=None, help="A larger model for the 'small beats large' comparison.")
    g.add_argument("--no-base", action="store_true", help="Skip the zero-shot base model.")
    g.add_argument("--base-shots", type=int, default=0, help="Also evaluate the base model with K in-context examples.")
    g.add_argument("--reference-shots", type=int, default=0, help="In-context examples for the reference model.")
    g.add_argument("--quantization", choices=("auto",) + QUANTIZATION_MODES, default="auto",
                   help="For base / adapter / reference models. auto = 4bit on a CUDA GPU, none otherwise.")
    g.add_argument("--merged-quantization", choices=QUANTIZATION_MODES, default="none")
    g.add_argument("--dtype", choices=DTYPES, default="auto")
    g.add_argument("--device", default="auto")

    g = p.add_argument_group("data")
    g.add_argument("--dataset-id", default=config.DEFAULT_DATASET_ID)
    g.add_argument("--train-file", type=Path, default=None, help="Only needed for few-shot examples.")
    g.add_argument("--validation-file", type=Path, default=None)
    g.add_argument("--test-file", type=Path, default=None)
    g.add_argument("--dialogue-col", default=config.DIALOGUE_COL)
    g.add_argument("--summary-col", default=config.SUMMARY_COL)
    g.add_argument("--split", choices=("test", "validation"), default="test")
    g.add_argument("--max-samples", type=int, default=200, help="Random dialogues to evaluate (0 = the whole split).")

    g = p.add_argument_group("generation / statistics")
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--max-new-tokens", type=int, default=100)
    g.add_argument("--max-prompt-tokens", type=int, default=1024)
    g.add_argument("--num-beams", type=int, default=1)
    g.add_argument("--bootstrap", type=int, default=1000)
    g.add_argument("--examples", type=int, default=6, help="Qualitative examples to write.")
    g.add_argument("--no-plots", action="store_true")
    g.add_argument("--no-update-readme", action="store_true")
    g.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    return p.parse_args(argv)


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def md_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---"] + ["---:"] * (len(df.columns) - 1)) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def build_specs(args: argparse.Namespace) -> list[dict]:
    """Ordered list of systems to evaluate."""
    base = display_name(args.base_model)
    specs: list[dict] = []
    if not args.no_base:
        specs.append({"name": f"{base} (zero-shot)", "kind": "base", "model": args.base_model, "adapter": None,
                      "shots": 0, "quantization": "q"})
    if args.base_shots:
        specs.append({"name": f"{base} + {args.base_shots}-shot", "kind": "fewshot", "model": args.base_model,
                      "adapter": None, "shots": args.base_shots, "quantization": "q"})
    for a in args.adapter:
        specs.append({"name": f"{base} + LoRA ({Path(a).name})", "kind": "finetuned", "model": args.base_model,
                      "adapter": a, "shots": 0, "quantization": "q"})
    for m in args.merged:
        specs.append({"name": f"merged ({Path(m).name})", "kind": "merged", "model": m, "adapter": None,
                      "shots": 0, "quantization": "m"})
    if args.reference_model:
        ref = display_name(args.reference_model)
        suffix = f"{args.reference_shots}-shot" if args.reference_shots else "zero-shot"
        specs.append({"name": f"{ref} ({suffix}, reference)", "kind": "reference", "model": args.reference_model,
                      "adapter": None, "shots": args.reference_shots, "quantization": "q"})
    return specs


def model_size_b(generator: SummaryGenerator, model_name: str) -> float:
    preset = find_preset(model_name)
    if preset:
        return preset.params_b
    return parameter_summary(generator.model)["total"] / 1e9


def headline(sig: pd.DataFrame, primary: str) -> list[str]:
    """Plain-language statements derived from the paired bootstrap (no overclaiming)."""
    out = []
    for r in sig[(sig["system"] == primary) & (sig["metric"] == "rougeL")].itertuples():
        rel = "reference model" if r.versus_kind == "reference" else "base model"
        if r.ci_low > 0:
            verdict = "is significantly better than"
        elif r.ci_high < 0:
            verdict = "is significantly worse than"
        else:
            verdict = "is not significantly different from"
        out.append(f"`{primary}` {verdict} the {rel} `{r.versus}` on ROUGE-L "
                   f"({r.delta:+.2f} points, 95% CI {r.ci_low:+.2f} to {r.ci_high:+.2f}).")
    return out


def update_readme(body: str) -> bool:
    path = config.README_PATH
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if RESULTS_START not in text or RESULTS_END not in text:
        return False
    pattern = re.compile(re.escape(RESULTS_START) + r".*?" + re.escape(RESULTS_END), re.DOTALL)
    path.write_text(pattern.sub(lambda _: f"{RESULTS_START}\n{body}\n{RESULTS_END}", text, count=1), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    device = pick_device(args.device)
    quant_q = ("4bit" if device.type == "cuda" else "none") if args.quantization == "auto" else args.quantization
    specs = build_specs(args)
    if not specs:
        logger.error("Nothing to evaluate: give --base-model (without --no-base), --adapter, --merged or --reference-model.")
        return 1

    # data -------------------------------------------------------------------------------------------
    try:
        splits = load_splits(args.dataset_id, args.train_file, args.validation_file, args.test_file,
                             args.dialogue_col, args.summary_col)
        if args.split not in splits:
            raise ValueError(f"The data has no '{args.split}' split (available: {list(splits)}).")
        df = sample_df(splits[args.split], args.max_samples or None, args.seed)
        needs_shots = max(args.base_shots, args.reference_shots) > 0
        shots_pool = splits.get("train")
        if needs_shots and shots_pool is None:
            raise ValueError("Few-shot prompting needs training examples: pass --train-file or a dataset with a train split.")
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as err:
        logger.error("%s", err.args[0] if err.args else err)
        return 1
    dialogues, references = df["dialogue"].tolist(), df["summary"].tolist()
    logger.info("Evaluating %d %s dialogues with %d system(s).", len(df), args.split, len(specs))

    # run every system -------------------------------------------------------------------------------------
    rows, predictions, scores_by_system, infos = [], {}, {}, {}
    for spec in specs:
        logger.info("Running %s ...", spec["name"])
        quant = quant_q if spec["quantization"] == "q" else args.merged_quantization
        shots = pick_shots(shots_pool, spec["shots"], args.seed) if spec["shots"] else []
        try:
            resources.free_memory()
            resources.reset_peak(device)
            gen = SummaryGenerator.load(spec["model"], spec["adapter"], quant, args.device, args.dtype,
                                        name=spec["name"], max_new_tokens=args.max_new_tokens,
                                        max_prompt_tokens=args.max_prompt_tokens, shots=shots,
                                        num_beams=args.num_beams)
        except (ModelLoadError, QuantizationUnavailable) as err:
            logger.warning("Skipping '%s': %s", spec["name"], err)
            continue
        size_b = model_size_b(gen, spec["model"])
        footprint = gen.model.get_memory_footprint() / 1e9 if hasattr(gen.model, "get_memory_footprint") else 0.0
        with resources.Timer() as timer:
            preds = gen.generate(dialogues, batch_size=args.batch_size)
        peak = resources.peak_memory_gb(device)
        gen.close()

        per_example = score_pairs(preds, references)
        agg = aggregate(per_example, args.bootstrap, args.seed)
        stats = text_stats(preds, dialogues, references)
        rows.append({"system": spec["name"], "kind": spec["kind"], "model": spec["model"], "adapter": spec["adapter"],
                     "quantization": quant, "params_b": round(size_b, 2), "weights_gb": round(footprint, 2),
                     "peak_memory_gb": round(peak, 2), "seconds_per_example": timer.seconds / len(df),
                     **{t: agg[t] for t in ROUGE_TYPES},
                     "rougeL_ci_low": agg["rougeL_ci_low"], "rougeL_ci_high": agg["rougeL_ci_high"],
                     "bleu": bleu(preds, references), **stats})
        predictions[spec["name"]], scores_by_system[spec["name"]] = preds, per_example
        infos[spec["name"]] = {"model": spec["model"], "adapter": spec["adapter"], "shots": spec["shots"]}
        logger.info("  R1 %.2f | R2 %.2f | RL %.2f | RLsum %.2f | BLEU %.2f | %.1f words | %.2fs/example | %.2f GB",
                    rows[-1]["rouge1"], rows[-1]["rouge2"], rows[-1]["rougeL"], rows[-1]["rougeLsum"],
                    rows[-1]["bleu"], rows[-1]["summary_words"], rows[-1]["seconds_per_example"], peak)
    if not rows:
        logger.error("No system could be evaluated.")
        return 1
    table = pd.DataFrame(rows)

    # paired significance ---------------------------------------------------------------------------------------
    sig_rows = []
    for r in rows:
        if r["kind"] not in ("finetuned", "merged"):
            continue
        for other in rows:
            if other["kind"] not in ("base", "fewshot", "reference"):
                continue
            for metric in COMPARE:
                res = paired_bootstrap(scores_by_system[r["system"]][f"{metric}_f"],
                                       scores_by_system[other["system"]][f"{metric}_f"], seed=args.seed)
                sig_rows.append({"system": r["system"], "versus": other["system"], "versus_kind": other["kind"],
                                 "metric": metric, **res})
    sig = pd.DataFrame(sig_rows, columns=["system", "versus", "versus_kind", "metric", "delta", "ci_low", "ci_high",
                                          "p_value"])

    # save --------------------------------------------------------------------------------------------------------
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.METRICS_DIR / "results.csv", index=False)
    sig.to_csv(config.METRICS_DIR / "significance.csv", index=False)
    for name, preds in predictions.items():
        out = pd.DataFrame({"id": df["id"], "dialogue": dialogues, "reference": references, "prediction": preds})
        out = pd.concat([out, scores_by_system[name][[f"{t}_f" for t in ROUGE_TYPES]].round(2)], axis=1)
        out.to_csv(config.PREDICTIONS_DIR / f"{slug(name)}.csv", index=False)

    primary = next((r["system"] for r in rows if r["kind"] == "finetuned"),
                   next((r["system"] for r in rows if r["kind"] == "merged"), None))
    shown = pd.DataFrame({
        "System": [f"**{s}**" if s == primary else s for s in table["system"]],
        "Params (B)": table["params_b"].map("{:.1f}".format),
        "Quant.": table["quantization"],
        "ROUGE-1": table["rouge1"].map("{:.2f}".format), "ROUGE-2": table["rouge2"].map("{:.2f}".format),
        "ROUGE-L": table["rougeL"].map("{:.2f}".format), "ROUGE-Lsum": table["rougeLsum"].map("{:.2f}".format),
        "ROUGE-L 95% CI": [f"{a:.1f}-{b:.1f}" for a, b in zip(table["rougeL_ci_low"], table["rougeL_ci_high"])],
        "BLEU": table["bleu"].map("{:.2f}".format), "Words": table["summary_words"].map("{:.1f}".format),
        "Peak mem (GB)": table["peak_memory_gb"].map("{:.1f}".format),
        "s/example": table["seconds_per_example"].map("{:.2f}".format),
    })
    table_md = md_table(shown)
    sig_md = ""
    if len(sig):
        view = sig[sig["metric"] == "rougeL"]
        sig_md = md_table(pd.DataFrame({
            "Fine-tuned system": view["system"], "versus": view["versus"],
            "ROUGE-L difference": view["delta"].map("{:+.2f}".format),
            "95% CI": [f"{a:+.2f} to {b:+.2f}" for a, b in zip(view["ci_low"], view["ci_high"])],
            "p (one-sided)": view["p_value"].map("{:.3f}".format)}))
    statements = headline(sig, primary) if primary and len(sig) else []
    (config.METRICS_DIR / "results.md").write_text(table_md + "\n\n" + sig_md + "\n", encoding="utf-8")

    summary = {
        "version": __version__, "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "split": args.split, "n_examples": len(df), "seed": args.seed, "dataset": args.dataset_id,
        "dataset_stats": dataset_stats(df), "generation": {"max_new_tokens": args.max_new_tokens,
                                                           "num_beams": args.num_beams, "do_sample": False},
        "systems": infos, "primary": primary, "statements": statements,
        "environment": resources.environment(), "results": json.loads(table.to_json(orient="records")),
        "significance": json.loads(sig.to_json(orient="records")),
    }
    (config.METRICS_DIR / "results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # examples + figures ----------------------------------------------------------------------------------------
    lines = ["# Qualitative examples", "", f"{len(df)} random {args.split} dialogues; first {args.examples} shown.", ""]
    for i in range(min(args.examples, len(df))):
        lines += [f"## Example {i + 1} (id `{df.loc[i, 'id']}`)", "", "**Dialogue**", ""]
        lines += [f"> {ln}" for ln in dialogues[i].split("\n")] + ["", "**Reference**", "", f"> {references[i]}", ""]
        for name, preds in predictions.items():
            lines += [f"**{name}**", "", f"> {preds[i] or '(empty)'}", ""]
    (config.REPORTS_DIR / "examples.md").write_text("\n".join(lines), encoding="utf-8")

    figures: list[str] = []
    if not args.no_plots:
        out_dir = config.FIGURES_DIR
        visualize.plot_rouge_comparison(table, out_dir)
        visualize.plot_length_distributions(
            {"Reference": [len(r.split()) for r in references],
             **{n: [len(p.split()) for p in preds] for n, preds in predictions.items()}}, out_dir)
        figures = ["rouge_comparison.png", "summary_lengths.png"]
        base_row = next((r for r in rows if r["kind"] == "base"), None)
        if primary and base_row:
            delta = (scores_by_system[primary]["rougeL_f"] - scores_by_system[base_row["system"]]["rougeL_f"]).to_numpy()
            visualize.plot_delta_histogram(delta, f"{primary} vs {base_row['system']}", out_dir)
            figures.append("rouge_l_gain_histogram.png")
        preset = find_preset(args.base_model)
        peak_train = None
        summ_path = config.METRICS_DIR / "training_summary.json"
        if summ_path.exists():
            peak_train = json.loads(summ_path.read_text(encoding="utf-8"))["resources"]["peak_memory_gb"]
        if preset:
            visualize.plot_memory_estimate(preset.params_b, preset.key, peak_train, out_dir)
            figures.append("memory_estimate.png")

    # README ---------------------------------------------------------------------------------------------------------
    if not args.no_update_readme:
        parts = [f"_Generated automatically on {summary['created']} - **{len(df)}** random `{args.split}` dialogues "
                 f"({args.dataset_id}), greedy decoding (max {args.max_new_tokens} new tokens), seed {args.seed}. "
                 "ROUGE / BLEU are F1 x 100; the same prompt and output cleaning are used for every system._", ""]
        if statements:
            parts += ["**Findings**", ""] + [f"- {s}" for s in statements] + [""]
        parts += [table_md, ""]
        if sig_md:
            parts += ["Paired bootstrap on per-dialogue ROUGE-L (positive = the fine-tuned system is better):", "",
                      sig_md, ""]
        train_path = config.METRICS_DIR / "training_summary.json"
        if train_path.exists():
            t = json.loads(train_path.read_text(encoding="utf-8"))
            res, par = t["resources"], t["parameters"]
            parts += ["**Training budget:** "
                      f"{par['trainable']:,} trainable parameters ({par['trainable_percent']:.2f}% of the model), "
                      f"{t['quantization']} base weights, peak memory {res['peak_memory_gb']:.1f} GB on "
                      f"`{(res.get('gpu') or {}).get('name', res['device'])}`, {res['train_seconds'] / 60:.0f} min, "
                      f"adapter file {res['adapter_size_mb']} MB.", ""]
        if figures:
            parts += ["| | |", "|:---:|:---:|"]
            cells = [f"![{f}](reports/figures/{f})" for f in figures]
            for i in range(0, len(cells), 2):
                pair = cells[i:i + 2] + [""] * (2 - len(cells[i:i + 2]))
                parts.append("| " + " | ".join(pair) + " |")
            parts.append("")
        parts.append("Side-by-side examples: [`reports/examples.md`](reports/examples.md).")
        if update_readme("\n".join(parts)):
            logger.info("README.md results section updated.")

    print("\n" + table_md)
    for s in statements:
        print("\n" + s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
