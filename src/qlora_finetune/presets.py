"""Small language models suitable for QLoRA on a single 16 GB GPU (e.g. a Colab T4).

Any other causal-LM id or local directory also works. ``params_b`` (billions of parameters) is used for the
memory estimates in the report.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    key: str
    hub_id: str
    params_b: float
    family: str
    gated: bool = False
    note: str = ""


PRESETS: dict[str, ModelPreset] = {
    p.key: p
    for p in (
        ModelPreset("tinyllama-1.1b", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1.1, "llama",
                    note="Very fast; good for a first end-to-end run"),
        ModelPreset("phi-2", "microsoft/phi-2", 2.7, "phi",
                    note="Recommended small model: strong for its size, comfortable on a T4 in 4-bit"),
        ModelPreset("mistral-7b", "mistralai/Mistral-7B-v0.1", 7.2, "mistral",
                    note="Larger reference model; also fits a T4 in 4-bit"),
        ModelPreset("llama3-8b", "meta-llama/Meta-Llama-3-8B", 8.0, "llama", gated=True,
                    note="Gated: accept the licence on the Hub and log in with a token"),
    )
}
_BY_HUB_ID = {p.hub_id: p for p in PRESETS.values()}


def find_preset(name: str) -> ModelPreset | None:
    """Preset for a short key (``phi-2``) or a hub id (``microsoft/phi-2``), else ``None``."""
    return PRESETS.get(name) or _BY_HUB_ID.get(name)


def resolve_model_id(name: str) -> str:
    """Hub id for a preset key; any other string (hub id or local path) is returned unchanged."""
    preset = find_preset(name)
    return preset.hub_id if preset else name


def display_name(name: str) -> str:
    preset = find_preset(name)
    if preset:
        return preset.key
    return name.rstrip("/\\").split("/")[-1].split("\\")[-1] or name
