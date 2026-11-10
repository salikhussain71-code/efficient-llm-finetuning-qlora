"""Model loading, quantisation (QLoRA) and LoRA adapters.

QLoRA in one paragraph: the frozen base model is stored in **4-bit NormalFloat (NF4)** with **double
quantisation**; computation runs in 16-bit; only small **LoRA** matrices (trained in higher precision) receive
gradients. A 7B model then needs about as much weight memory as a 3-4B model in 16-bit.
"""

from __future__ import annotations

import logging

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .presets import resolve_model_id

logger = logging.getLogger(__name__)

QUANTIZATION_MODES = ("none", "8bit", "4bit")
DTYPES = ("auto", "float32", "float16", "bfloat16")


class QuantizationUnavailable(RuntimeError):
    """Raised when 4-bit / 8-bit loading is requested but bitsandbytes / a CUDA GPU is missing."""


class ModelLoadError(OSError):
    """Raised when a model or tokenizer cannot be loaded."""


# --------------------------------------------------------------------------------------- devices
def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_dtype(device: torch.device) -> torch.dtype:
    """bfloat16 where the GPU supports it, float16 otherwise (e.g. T4), float32 on CPU."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return compute_dtype(device)
    return getattr(torch, name)


# --------------------------------------------------------------------------------- quantisation
def build_bnb_config(quantization: str, dtype: torch.dtype) -> BitsAndBytesConfig | None:
    """``BitsAndBytesConfig`` for ``4bit`` (NF4 + double quantisation) or ``8bit``; ``None`` for ``none``."""
    if quantization not in QUANTIZATION_MODES:
        raise ValueError(f"quantization must be one of {QUANTIZATION_MODES}")
    if quantization == "none":
        return None
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=dtype)


def require_quantization_support(quantization: str) -> None:
    """Fail early with a clear message instead of an obscure bitsandbytes error."""
    if quantization == "none":
        return
    if not torch.cuda.is_available():
        raise QuantizationUnavailable(
            f"--quantization {quantization} needs an NVIDIA GPU (CUDA), which was not found. "
            "Use a GPU runtime (e.g. Google Colab T4) or pass --quantization none (works on CPU, but is slow "
            "and uses more memory).")
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as err:
        raise QuantizationUnavailable(
            "bitsandbytes is not installed. Install it with: pip install bitsandbytes") from err


# ------------------------------------------------------------------------------------- loading
def load_tokenizer(name_or_path: str, padding_side: str = "right"):
    """Tokenizer with a pad token (falls back to EOS, which most causal LMs lack a pad token for)."""
    source = resolve_model_id(name_or_path)
    try:
        tok = AutoTokenizer.from_pretrained(source)
    except Exception as err:
        raise ModelLoadError(_load_hint(source, err)) from err
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def _load_hint(source: str, err: Exception) -> str:
    return (f"Could not load '{source}': {err}\n"
            "Models are downloaded from the Hugging Face Hub on first use (internet required). Gated models "
            "(e.g. Llama 3) also need `huggingface-cli login` after accepting the licence on the model page.")


def load_base_model(name_or_path: str, quantization: str = "none", device: str = "auto", dtype: str = "auto",
                    trust_remote_code: bool = False):
    """Load a causal LM, optionally quantised, and place it on the right device."""
    source = resolve_model_id(name_or_path)
    dev = pick_device(device)
    require_quantization_support(quantization)
    torch_dtype = resolve_dtype(dtype, dev)
    kwargs: dict = {"trust_remote_code": trust_remote_code}
    bnb = build_bnb_config(quantization, torch_dtype)
    if bnb is not None:
        kwargs.update(quantization_config=bnb, device_map={"": dev.index or 0})
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(source, dtype=torch_dtype, **kwargs)
        except TypeError:                                   # transformers < 4.56 calls it torch_dtype
            model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch_dtype, **kwargs)
    except QuantizationUnavailable:
        raise
    except Exception as err:
        raise ModelLoadError(_load_hint(source, err)) from err
    if bnb is None:
        model.to(dev)
    model.eval()
    return model


# ------------------------------------------------------------------------------------- LoRA
def build_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05,
                      target_modules: str | list[str] = "all-linear", bias: str = "none") -> LoraConfig:
    """LoRA hyper-parameters. ``target_modules="all-linear"`` adapts every linear layer except the LM head
    (the QLoRA paper's recommendation); pass a list such as ``["q_proj", "v_proj"]`` for fewer parameters."""
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules, bias=bias,
                      task_type=TaskType.CAUSAL_LM)


def prepare_for_training(model, quantization: str = "none", gradient_checkpointing: bool = True):
    """Make a (possibly quantised) model ready for adapter training."""
    model.config.use_cache = False
    if quantization != "none":
        # casts norms to fp32, enables input grads and (optionally) gradient checkpointing
        return prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model


def attach_lora(model, lora_config: LoraConfig) -> PeftModel:
    return get_peft_model(model, lora_config)


def load_adapter(model, adapter_path: str, trainable: bool = False) -> PeftModel:
    """Attach a saved LoRA adapter to a base model."""
    try:
        return PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    except Exception as err:
        raise ModelLoadError(f"Could not load adapter '{adapter_path}': {err}") from err


def parameter_summary(model) -> dict:
    """Trainable / total parameters (quantisation-aware when PEFT can count them)."""
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
    return {"trainable": int(trainable), "total": int(total), "trainable_percent": 100.0 * trainable / max(total, 1)}


def memory_footprint_gb(model) -> float:
    """Memory used by the model's parameters and buffers, in GB."""
    return float(model.get_memory_footprint()) / 1e9
