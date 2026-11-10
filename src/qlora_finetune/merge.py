"""Merge a LoRA adapter into its base model and save a standalone checkpoint (bonus task).

    python -m qlora_finetune.merge --base-model phi-2 --adapter models/phi-2-samsum-qlora \\
        --output models/phi-2-samsum-merged

The base model is loaded **un-quantised** (merging into 4-bit weights is not possible), the adapter is applied
(W' = W + (alpha / r) * B @ A), and the result is saved in safetensors format together with the tokenizer.
Before and after merging, the model summarises a few built-in dialogues; the report records how many outputs
are identical and the largest logit difference, so you can see that merging did not change the behaviour.

Note: a QLoRA adapter was trained against the *4-bit* base weights. Merged into 16-bit weights it behaves very
slightly differently - confirm with ``python -m qlora_finetune.evaluate --merged <output>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from . import resources
from .generate import SummaryGenerator
from .modeling import (
    DTYPES,
    ModelLoadError,
    load_adapter,
    load_base_model,
    load_tokenizer,
    pick_device,
    resolve_dtype,
)
from .presets import display_name, resolve_model_id
from .prompts import build_prompt

logger = logging.getLogger("qlora_finetune")

VERIFY_DIALOGUES = [
    "Anna: Are you coming to the party on Friday?\nBen: Yes, I will bring cake.",
    "Carla: I'll be 15 minutes late.\nDev: OK, I'll wait at the station.",
    "Emma: I passed the driving test!\nFarid: Congratulations, Emma!",
]


@torch.inference_mode()
def _last_logits(model, tokenizer, dialogue: str) -> torch.Tensor:
    enc = tokenizer(build_prompt(dialogue), return_tensors="pt").to(model.device)
    return model(**enc).logits[0, -1].float().cpu()


def merge(base_model: str, adapter: str, output: Path, dtype: str = "auto", device: str = "auto",
          save_dtype: str = "auto", verify: bool = True, max_shard_size: str = "2GB") -> dict:
    """Merge ``adapter`` into ``base_model`` and write the result to ``output``. Returns the report."""
    dev = pick_device(device)
    tokenizer = load_tokenizer(base_model, padding_side="left")
    model = load_base_model(base_model, quantization="none", device=device, dtype=dtype)
    peft_model = load_adapter(model, adapter, trainable=False).eval()

    report: dict = {"base_model": resolve_model_id(base_model), "adapter": str(adapter), "output": str(output),
                    "adapter_size_mb": round(resources.dir_size_mb(Path(adapter) / "adapter_model.safetensors"), 2)
                    if (Path(adapter) / "adapter_model.safetensors").exists() else None}
    before_text, before_logits = [], []
    if verify:
        gen = SummaryGenerator(peft_model, tokenizer, max_new_tokens=60)
        before_text = gen.generate(VERIFY_DIALOGUES, batch_size=len(VERIFY_DIALOGUES))
        before_logits = [_last_logits(peft_model, tokenizer, d) for d in VERIFY_DIALOGUES]

    merged = peft_model.merge_and_unload()
    merged.eval()
    if verify:
        gen = SummaryGenerator(merged, tokenizer, max_new_tokens=60)
        after_text = gen.generate(VERIFY_DIALOGUES, batch_size=len(VERIFY_DIALOGUES))
        diffs = [float((b - _last_logits(merged, tokenizer, d)).abs().max())
                 for b, d in zip(before_logits, VERIFY_DIALOGUES)]
        report["verification"] = {
            "dialogues": len(VERIFY_DIALOGUES),
            "identical_outputs": sum(a == b for a, b in zip(before_text, after_text)),
            "max_abs_logit_diff": max(diffs), "before": before_text, "after": after_text}
        logger.info("Verification: %d/%d identical summaries, max |logit diff| %.4g",
                    report["verification"]["identical_outputs"], len(VERIFY_DIALOGUES), max(diffs))

    target = resolve_dtype(save_dtype if save_dtype != "auto" else ("float16" if dev.type != "cpu" else "float32"), dev)
    merged.to(target)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size=max_shard_size)
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(output)
    report.update({"save_dtype": str(target).replace("torch.", ""),
                   "merged_size_mb": round(resources.dir_size_mb(output), 1),
                   "files": sorted(p.name for p in output.iterdir())})
    (output / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    p.add_argument("--base-model", default="phi-2")
    p.add_argument("--adapter", required=True, help="Folder with the trained LoRA adapter.")
    p.add_argument("--output", type=Path, default=None, help="Default: models/<model>-samsum-merged")
    p.add_argument("--dtype", choices=DTYPES, default="auto", help="dtype used while merging.")
    p.add_argument("--save-dtype", choices=DTYPES, default="auto", help="dtype of the saved weights.")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-shard-size", default="2GB")
    p.add_argument("--no-verify", action="store_true", help="Skip the before/after comparison.")
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    output = args.output or Path("models") / f"{display_name(args.base_model)}-samsum-merged"
    if not (Path(args.adapter) / "adapter_config.json").exists():
        logger.error("'%s' does not contain a LoRA adapter (adapter_config.json missing).", args.adapter)
        return 1
    try:
        report = merge(args.base_model, args.adapter, output, args.dtype, args.device, args.save_dtype,
                       not args.no_verify, args.max_shard_size)
    except ModelLoadError as err:
        logger.error("%s", err)
        return 1
    print(f"\nMerged model saved to {output} ({report['merged_size_mb']} MB, {report['save_dtype']}); "
          f"the adapter alone was {report['adapter_size_mb']} MB.")
    print(f"Evaluate it: python -m qlora_finetune.evaluate --base-model {args.base_model} --merged {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
