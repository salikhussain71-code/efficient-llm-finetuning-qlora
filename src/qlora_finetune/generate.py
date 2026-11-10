"""Summarise dialogues with a base model, a LoRA adapter or a merged model.

    python -m qlora_finetune.generate --base-model phi-2 --adapter models/phi-2-samsum-qlora --quantization 4bit \\
        "Anna: Are you coming on Friday?\\nBen: Yes, I will bring cake."
    python -m qlora_finetune.generate --base-model models/phi-2-samsum-merged --file chat.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from . import resources
from .modeling import (
    DTYPES,
    QUANTIZATION_MODES,
    ModelLoadError,
    QuantizationUnavailable,
    load_adapter,
    load_base_model,
    load_tokenizer,
)
from .prompts import build_prompt, clean_generation, normalise_text, prompt_overhead, truncate_dialogue

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """Greedy (or beam) generation of dialogue summaries with the exact prompt used in training."""

    def __init__(self, model, tokenizer, name: str = "model", max_new_tokens: int = 100, max_prompt_tokens: int = 1024,
                 shots: list[tuple[str, str]] | None = None, num_beams: int = 1) -> None:
        self.model, self.tokenizer, self.name = model, tokenizer, name
        self.max_new_tokens, self.max_prompt_tokens = max_new_tokens, max_prompt_tokens
        self.shots = shots or []
        self.num_beams = num_beams
        self._overhead = prompt_overhead(tokenizer, self.shots)

    @classmethod
    def load(cls, base_model: str, adapter: str | None = None, quantization: str = "none", device: str = "auto",
             dtype: str = "auto", name: str | None = None, **kwargs) -> SummaryGenerator:
        tokenizer = load_tokenizer(base_model, padding_side="left")
        model = load_base_model(base_model, quantization=quantization, device=device, dtype=dtype)
        if adapter:
            model = load_adapter(model, adapter, trainable=False)
        model.eval()
        return cls(model, tokenizer, name=name or Path(str(adapter or base_model)).name, **kwargs)

    def _prompt(self, dialogue: str) -> str:
        budget = self.max_prompt_tokens - self._overhead
        return build_prompt(truncate_dialogue(dialogue, self.tokenizer, max(budget, 16)), self.shots)

    @torch.inference_mode()
    def generate(self, dialogues: list[str], batch_size: int = 8, progress: bool = False) -> list[str]:
        """One cleaned summary per dialogue, in input order."""
        tok, model = self.tokenizer, self.model
        was_training, old_side = model.training, tok.padding_side
        old_cache = getattr(model.config, "use_cache", True)
        model.eval()
        model.config.use_cache = True
        tok.padding_side = "left"
        try:
            prompts = [self._prompt(d) for d in dialogues]
            order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
            results = [""] * len(prompts)
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=True)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                out = model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                     num_beams=self.num_beams, pad_token_id=tok.pad_token_id,
                                     eos_token_id=tok.eos_token_id, use_cache=True)
                new = out[:, enc["input_ids"].shape[1]:]
                for i, text in zip(idx, tok.batch_decode(new, skip_special_tokens=True)):
                    results[i] = clean_generation(text)
                if progress:
                    logger.info("  %s: %d/%d", self.name, min(start + batch_size, len(order)), len(order))
            return results
        finally:
            tok.padding_side = old_side
            model.config.use_cache = old_cache
            model.train(was_training)

    def close(self) -> None:
        """Release the model (important when several large models are evaluated one after another)."""
        self.model = None
        resources.free_memory()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarise dialogues.")
    p.add_argument("text", nargs="*", help="Dialogue text(s). Use \\n for line breaks or pass --file.")
    p.add_argument("--file", type=Path, nargs="*", default=[], help="Text file(s), one dialogue per file.")
    p.add_argument("--base-model", default="phi-2", help="Preset key, Hugging Face id, or a merged-model folder.")
    p.add_argument("--adapter", default=None, help="Folder with a trained LoRA adapter.")
    p.add_argument("--quantization", choices=QUANTIZATION_MODES, default="none")
    p.add_argument("--dtype", choices=DTYPES, default="auto")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--num-beams", type=int, default=1)
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    dialogues = [t.replace("\\n", "\n") if "\n" not in t else t for t in args.text]
    dialogues += [f.read_text(encoding="utf-8") for f in args.file]
    if not dialogues:
        p.print_help()
        return 1
    try:
        gen = SummaryGenerator.load(args.base_model, args.adapter, args.quantization, args.device, args.dtype,
                                    max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
    except (ModelLoadError, QuantizationUnavailable) as err:
        print(err, file=sys.stderr)
        return 1
    for dialogue, summary in zip(dialogues, gen.generate([normalise_text(d) for d in dialogues])):
        print(f"\n--- Dialogue ---\n{normalise_text(dialogue)}\n--- Summary ({gen.name}) ---\n{summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
