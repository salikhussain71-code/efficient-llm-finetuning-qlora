"""Prompt format shared by training, evaluation and inference.

Training and inference MUST use exactly the same text, otherwise the fine-tuned model sees a different
format at test time. Everything that builds a prompt therefore lives here.
"""

from __future__ import annotations

import re

INSTRUCTION = "Summarize the following dialogue."
HEAD = f"{INSTRUCTION}\n\n### Dialogue:\n"
TAIL = "\n\n### Summary:\n"
_STOP_MARKERS = ("\n\n", "###", "<|endoftext|>", "</s>", "<|im_end|>", "<|eot_id|>")
_WS = re.compile(r"\s+")


def normalise_text(text: object) -> str:
    """Windows newlines -> ``\\n``, trailing spaces removed, surrounding whitespace stripped."""
    if text is None or (isinstance(text, float) and text != text):      # None / NaN -> empty
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def build_prompt(dialogue: str, shots: list[tuple[str, str]] | None = None) -> str:
    """Instruction + dialogue + the ``### Summary:`` marker the model must continue.

    ``shots`` are (dialogue, summary) examples placed before the query (few-shot prompting).
    """
    parts = [f"{HEAD}{normalise_text(d)}{TAIL}{normalise_text(s)}\n\n" for d, s in (shots or [])]
    parts.append(f"{HEAD}{normalise_text(dialogue)}{TAIL}")
    return "".join(parts)


def clean_generation(text: str) -> str:
    """Cut a raw generation at the first stop marker and flatten whitespace.

    Base models tend to keep writing (new ``### Dialogue:`` blocks, etc.). The same cleaning is applied to
    every system, so the comparison stays fair.
    """
    text = text.lstrip()
    cut = len(text)
    for marker in _STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return _WS.sub(" ", text[:cut]).strip()


def truncate_dialogue(dialogue: str, tokenizer, budget: int) -> str:
    """Keep the first ``budget`` tokens of a dialogue (decoded back to text); short dialogues are untouched."""
    dialogue = normalise_text(dialogue)
    ids = tokenizer(dialogue, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return dialogue
    return tokenizer.decode(ids[:max(budget, 1)], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def prompt_overhead(tokenizer, shots: list[tuple[str, str]] | None = None) -> int:
    """Tokens used by the instruction, markers, special tokens and few-shot examples."""
    return len(tokenizer(build_prompt("", shots), add_special_tokens=True)["input_ids"])
