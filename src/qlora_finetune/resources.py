"""Resource accounting: the 'efficient' part of efficient fine-tuning, in numbers."""

from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

import torch


def gpu_info() -> dict | None:
    """Name and memory of the first CUDA GPU, or ``None``."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name, "total_gb": round(props.total_memory / 1e9, 2)}


def process_rss_gb() -> float:
    """Peak resident memory of this process (CPU runs)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1e9 if sys.platform == "darwin" else 1e6)     # bytes on macOS, KB on Linux


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device) -> float:
    """Peak GPU memory allocated (CUDA) or peak process memory (CPU / MPS), in GB."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e9
    return process_rss_gb()


def free_memory() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dir_size_mb(path: str | Path) -> float:
    """Total size of the files below ``path`` in MB."""
    path = Path(path)
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return sum(f.stat().st_size for f in files) / 1e6


def estimate_weight_memory_gb(params_b: float) -> dict[str, float]:
    """Weights-only memory of a model with ``params_b`` billion parameters at different precisions.

    4-bit adds ~6 % for the quantisation constants (double quantisation keeps this small). Training also needs
    activations, LoRA gradients / optimiser state and the CUDA context, so treat this as a lower bound.
    """
    n = params_b * 1e9
    return {"fp32": n * 4 / 1e9, "fp16/bf16": n * 2 / 1e9, "8-bit": n * 1 / 1e9, "4-bit (NF4)": n * 0.5 * 1.06 / 1e9}


class Timer:
    """``with Timer() as t: ...; t.seconds``"""

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0


def environment() -> dict:
    import platform

    import peft
    import transformers

    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "cuda": torch.version.cuda, "gpu": gpu_info(),
            "cpu_count": os.cpu_count()}
