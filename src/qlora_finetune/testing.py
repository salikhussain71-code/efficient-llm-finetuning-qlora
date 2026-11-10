"""Build a tiny, randomly initialised Llama-style causal LM that runs fully offline.

It is far too small to be useful, but it exercises the *real* code path (tokenisation, LoRA, training,
generation, merging, ROUGE) in seconds on a CPU. Used by the unit tests, CI and ``make smoke``.

    python -m qlora_finetune.testing --out models/tiny-llama
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

DEFAULT_CORPUS = [
    "Summarize the following dialogue.", "### Dialogue:", "### Summary:",
    "Anna: Are you coming to the party on Friday?\nBen: Yes, I will bring cake.",
    "Anna asks Ben if Ben is coming to the party on Friday. Ben will bring cake.",
]


def build_tiny_causal_lm(out_dir: str | Path, corpus: list[str] | None = None, seed: int = 0, hidden_size: int = 96,
                         num_layers: int = 2, vocab_size: int = 600, max_length: int = 512) -> Path:
    """Create and save a tiny tokenizer + Llama model; ``corpus`` should contain the text you will train on."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<s>", "</s>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(list(corpus or []) + DEFAULT_CORPUS * 3, trainer)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", tok.token_to_id("<s>"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                                   model_max_length=max_length)

    config = LlamaConfig(vocab_size=len(fast), hidden_size=hidden_size, intermediate_size=hidden_size * 2,
                         num_hidden_layers=num_layers, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=max_length, bos_token_id=1, eos_token_id=2, pad_token_id=None,
                         tie_word_embeddings=False)
    model = LlamaForCausalLM(config)
    fast.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a tiny random causal LM for offline smoke tests.")
    parser.add_argument("--out", type=Path, default=Path("models/tiny-llama"))
    parser.add_argument("--corpus-csv", type=Path, default=None,
                        help="Optional CSV with dialogue/summary columns to train the tokenizer on.")
    args = parser.parse_args(argv)
    corpus = None
    if args.corpus_csv:
        import pandas as pd

        df = pd.read_csv(args.corpus_csv)
        corpus = df["dialogue"].tolist() + df["summary"].tolist()
    path = build_tiny_causal_lm(args.out, corpus)
    print(f"Tiny Llama written to {path} (random weights - it only exercises the pipeline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
