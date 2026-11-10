"""Parameter-efficient fine-tuning (LoRA / QLoRA) for dialogue summarisation.

Pipeline: ``train`` (QLoRA on a single GPU) -> ``evaluate`` (ROUGE / BLEU: base vs fine-tuned vs a larger
reference model) -> ``merge`` (fold the adapter into the base weights) -> ``push`` (Hugging Face Hub).
"""

__version__ = "1.0.0"
