"""QLoRA / LoRA fine-tuning of a causal language model for dialogue summarisation (SAMSum).

    python -m qlora_finetune.train --base-model phi-2                       # 4-bit QLoRA on a GPU
    python -m qlora_finetune.train --base-model phi-2 --max-train-samples 0 --epochs 2   # full dataset
    python -m qlora_finetune.train --base-model tinyllama-1.1b --quantization none        # LoRA, no quantisation
    python -m qlora_finetune.train --report-to wandb --run-name phi2-r16                  # Weights & Biases
    python -m qlora_finetune.train --resume-from models/phi-2-samsum-qlora/checkpoints/step-000200

Memory levers (single 16 GB GPU): 4-bit NF4 base weights, LoRA adapters only, gradient checkpointing,
small micro-batches with gradient accumulation, length-grouped batches, optional paged 8-bit AdamW.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from . import __version__, config, resources, visualize
from .data import SummarizationDataset, collate, dataset_stats, load_splits, make_batches, sample_df
from .generate import SummaryGenerator
from .metrics import aggregate, score_pairs
from .modeling import (
    DTYPES,
    QUANTIZATION_MODES,
    ModelLoadError,
    QuantizationUnavailable,
    attach_lora,
    build_lora_config,
    load_adapter,
    load_base_model,
    load_tokenizer,
    memory_footprint_gb,
    parameter_summary,
    pick_device,
    prepare_for_training,
    resolve_dtype,
)
from .presets import display_name, resolve_model_id
from .tracking import Tracker, TrackingUnavailable

logger = logging.getLogger("qlora_finetune")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning on SAMSum.")
    g = p.add_argument_group("model")
    g.add_argument("--base-model", default="phi-2", help="Preset key, Hugging Face id or local folder.")
    g.add_argument("--quantization", choices=("auto",) + QUANTIZATION_MODES, default="auto",
                   help="auto = 4bit on a CUDA GPU, none otherwise.")
    g.add_argument("--dtype", choices=DTYPES, default="auto", help="Compute / non-quantised weight dtype.")
    g.add_argument("--device", default="auto")
    g.add_argument("--trust-remote-code", action="store_true")

    g = p.add_argument_group("data")
    g.add_argument("--dataset-id", default=config.DEFAULT_DATASET_ID)
    g.add_argument("--train-file", type=Path, default=None)
    g.add_argument("--validation-file", type=Path, default=None)
    g.add_argument("--test-file", type=Path, default=None)
    g.add_argument("--dialogue-col", default=config.DIALOGUE_COL)
    g.add_argument("--summary-col", default=config.SUMMARY_COL)
    g.add_argument("--max-train-samples", type=int, default=4000, help="0 = the whole training split.")
    g.add_argument("--max-validation-samples", type=int, default=200)
    g.add_argument("--max-seq-len", type=int, default=512)

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora-r", type=int, default=16)
    g.add_argument("--lora-alpha", type=int, default=32)
    g.add_argument("--lora-dropout", type=float, default=0.05)
    g.add_argument("--target-modules", default="all-linear",
                   help="'all-linear' or a comma-separated list, e.g. q_proj,v_proj")

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=1)
    g.add_argument("--max-steps", type=int, default=0,
                   help="Shorten the whole run (and its LR schedule) to this many optimiser steps (0 = off).")
    g.add_argument("--stop-after-steps", type=int, default=0,
                   help="Pause after this many steps WITHOUT changing the schedule and write a resumable checkpoint "
                        "(for time-limited sessions); continue later with --resume-from.")
    g.add_argument("--batch-size", type=int, default=4, help="Micro-batch size per forward pass.")
    g.add_argument("--grad-accum", type=int, default=4, help="Effective batch = batch-size x grad-accum.")
    g.add_argument("--lr", type=float, default=2e-4)
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--warmup-ratio", type=float, default=0.03)
    g.add_argument("--lr-scheduler", choices=("cosine", "linear", "constant"), default="cosine")
    g.add_argument("--max-grad-norm", type=float, default=0.3)
    g.add_argument("--optim", choices=("adamw", "paged_adamw_8bit"), default="adamw")
    g.add_argument("--no-gradient-checkpointing", action="store_true")
    g.add_argument("--no-group-by-length", action="store_true")
    g.add_argument("--seed", type=int, default=config.RANDOM_STATE)

    g = p.add_argument_group("logging / checkpoints")
    g.add_argument("--output", type=Path, default=None, help="Adapter folder (default: models/<model>-samsum-qlora).")
    g.add_argument("--logging-steps", type=int, default=10)
    g.add_argument("--save-steps", type=int, default=0, help="Write a resumable checkpoint every N steps (0 = off).")
    g.add_argument("--resume-from", type=Path, default=None, help="Checkpoint folder to continue from.")
    g.add_argument("--eval-rouge-samples", type=int, default=32,
                   help="Validation dialogues summarised after each epoch to track ROUGE (0 = off).")
    g.add_argument("--max-new-tokens", type=int, default=100)
    g.add_argument("--report-to", choices=("none", "wandb"), default="none")
    g.add_argument("--wandb-project", default="qlora-samsum")
    g.add_argument("--run-name", default=None)
    g.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


# ------------------------------------------------------------------------------------- helpers
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_quantization(requested: str, device: torch.device) -> str:
    if requested == "auto":
        return "4bit" if device.type == "cuda" else "none"
    return requested


def make_optimizer(params, args):
    if args.optim == "paged_adamw_8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as err:
            raise QuantizationUnavailable("--optim paged_adamw_8bit needs bitsandbytes (pip install bitsandbytes)") from err
        return bnb.optim.PagedAdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def make_scheduler(optimizer, kind: str, warmup_steps: int, total_steps: int):
    if kind == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if kind == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(warmup_steps, 1)))


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (TypeError, AttributeError):                     # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


@torch.inference_mode()
def validation_loss(model, dataset, batch_size: int, pad_id: int, device: torch.device, dtype: torch.dtype) -> float:
    """Token-weighted mean cross-entropy over the summary tokens of ``dataset``."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for batch_idx in make_batches(dataset.lengths, batch_size, 0, 0, group_by_length=False, shuffle=False):
        batch = {k: v.to(device) for k, v in collate([dataset[i] for i in batch_idx], pad_id).items()}
        with autocast_ctx(device, dtype):
            loss = model(**batch).loss
        n = int((batch["labels"][:, 1:] != -100).sum())
        total += float(loss) * n
        count += n
    model.train(was_training)
    return total / max(count, 1)


def save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, state: dict, out_dir: Path) -> Path:
    path = out_dir / "checkpoints" / f"step-{state['global_step']:06d}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), **state}, path / "trainer_state.pt")
    return path


# --------------------------------------------------------------------------------------- training
def train(args: argparse.Namespace) -> dict:
    """Run QLoRA training and return a summary dictionary (also written to ``training_summary.json``)."""
    set_seed(args.seed)
    device = pick_device(args.device)
    quantization = resolve_quantization(args.quantization, device)
    cdtype = resolve_dtype(args.dtype, device)
    out_dir = args.output or (config.MODELS_DIR / f"{display_name(args.base_model)}-samsum-qlora")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Device %s | quantisation %s | compute dtype %s | output %s", device, quantization, cdtype, out_dir)

    # data ------------------------------------------------------------------------------------
    splits = load_splits(args.dataset_id, args.train_file, args.validation_file, args.test_file,
                         args.dialogue_col, args.summary_col)
    if "train" not in splits:
        raise ValueError("No training data: pass --train-file or a dataset with a 'train' split.")
    train_df = sample_df(splits["train"], args.max_train_samples or None, args.seed)
    if "validation" in splits:
        val_df = sample_df(splits["validation"], args.max_validation_samples or None, args.seed)
    else:                                       # carve a validation set out of the training data
        n_val = min(args.max_validation_samples or 100, max(1, len(train_df) // 10))
        val_df, train_df = train_df.iloc[:n_val].reset_index(drop=True), train_df.iloc[n_val:].reset_index(drop=True)
    logger.info("Training on %d dialogues, validating on %d.", len(train_df), len(val_df))

    tokenizer = load_tokenizer(args.base_model, padding_side="right")
    train_ds = SummarizationDataset(train_df, tokenizer, args.max_seq_len)
    val_ds = SummarizationDataset(val_df, tokenizer, args.max_seq_len)
    logger.info("Sequences: mean %.0f tokens, %d of %d truncated; %d target tokens.", np.mean(train_ds.lengths),
                train_ds.n_truncated, len(train_ds), train_ds.n_target_tokens)

    # model -----------------------------------------------------------------------------------
    resources.reset_peak(device)
    base = load_base_model(args.base_model, quantization, args.device, args.dtype, args.trust_remote_code)
    base_footprint = memory_footprint_gb(base)
    model = prepare_for_training(base, quantization, not args.no_gradient_checkpointing)
    targets = args.target_modules if args.target_modules == "all-linear" else args.target_modules.split(",")
    if args.resume_from:
        model = load_adapter(model, str(args.resume_from), trainable=True)
    else:
        model = attach_lora(model, build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, targets))
    params = parameter_summary(model)
    logger.info("Trainable parameters: %s of %s (%.3f%%). Base model memory: %.2f GB.", f"{params['trainable']:,}",
                f"{params['total']:,}", params["trainable_percent"], base_footprint)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(trainable, args)
    steps_per_epoch = max(1, math.ceil(math.ceil(len(train_ds) / args.batch_size) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = make_scheduler(optimizer, args.lr_scheduler, int(args.warmup_ratio * total_steps), total_steps)
    use_fp16 = device.type == "cuda" and cdtype == torch.float16
    scaler = make_scaler(use_fp16)

    tracker = Tracker(out_dir, args.report_to, args.wandb_project, args.run_name,
                      config={**{k: str(v) for k, v in vars(args).items()}, "quantization_resolved": quantization,
                              "trainable_params": params["trainable"], "total_params": params["total"]})

    # resume ----------------------------------------------------------------------------------
    start_epoch, skip_batches, global_step, best_val = 0, 0, 0, float("inf")
    if args.resume_from:
        state = torch.load(args.resume_from / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, skip_batches = state["epoch"], state["batch_in_epoch"]
        global_step, best_val = state["global_step"], state["best_val_loss"]
        logger.info("Resumed from %s (step %d, epoch %d).", args.resume_from, global_step, start_epoch)

    generator = SummaryGenerator(model, tokenizer, name="training", max_new_tokens=args.max_new_tokens)
    val_rows: list[dict] = []
    train_seconds, tokens_seen, last_log_time, last_log_tokens = 0.0, 0, time.perf_counter(), 0
    running_loss, running_n, last_grad_norm, best_epoch = 0.0, 0, 0.0, 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t_start = time.perf_counter()
    stop = False

    for epoch in range(start_epoch, args.epochs):
        batches = make_batches(train_ds.lengths, args.batch_size, args.seed, epoch, not args.no_group_by_length)
        for micro, batch_idx in enumerate(batches):
            if epoch == start_epoch and micro < skip_batches:
                continue
            batch = {k: v.to(device) for k, v in collate([train_ds[i] for i in batch_idx], tokenizer.pad_token_id).items()}
            with autocast_ctx(device, cdtype):
                loss = model(**batch).loss
            scaler.scale(loss / args.grad_accum).backward()
            running_loss += float(loss.detach())
            running_n += 1
            tokens_seen += int(batch["attention_mask"].sum())

            at_boundary = (micro + 1) % args.grad_accum == 0 or micro + 1 == len(batches)
            if not at_boundary:
                continue
            scaler.unscale_(optimizer)
            last_grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.logging_steps == 0 or global_step == total_steps:
                now = time.perf_counter()
                row = {"epoch": epoch + (micro + 1) / len(batches), "loss": running_loss / max(running_n, 1),
                       "lr": scheduler.get_last_lr()[0], "grad_norm": last_grad_norm,
                       "tokens_per_s": (tokens_seen - last_log_tokens) / max(now - last_log_time, 1e-9),
                       "memory_gb": resources.peak_memory_gb(device)}
                tracker.log(row, global_step)
                logger.info("step %d/%d | epoch %.2f | loss %.4f | lr %.2e | grad %.2f | %.0f tok/s | %.2f GB",
                            global_step, total_steps, row["epoch"], row["loss"], row["lr"], row["grad_norm"],
                            row["tokens_per_s"], row["memory_gb"])
                running_loss, running_n, last_log_time, last_log_tokens = 0.0, 0, now, tokens_seen

            if args.save_steps and global_step % args.save_steps == 0:
                ckpt = save_checkpoint(model, tokenizer, optimizer, scheduler, scaler,
                                       {"epoch": epoch, "batch_in_epoch": micro + 1, "global_step": global_step,
                                        "best_val_loss": best_val}, out_dir)
                tracker.flush()
                logger.info("  checkpoint -> %s", ckpt)
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
            if args.stop_after_steps and global_step >= args.stop_after_steps and global_step < total_steps:
                ckpt = save_checkpoint(model, tokenizer, optimizer, scheduler, scaler,
                                       {"epoch": epoch, "batch_in_epoch": micro + 1, "global_step": global_step,
                                        "best_val_loss": best_val}, out_dir)
                tracker.flush()
                logger.info("Paused after %d of %d steps. Resume with: --resume-from %s", global_step, total_steps, ckpt)
                stop = True
                break

        # ---- end of epoch (or of the allowed steps): validate and keep the best adapter --------
        val_loss = validation_loss(model, val_ds, args.batch_size, tokenizer.pad_token_id, device, cdtype)
        row = {"epoch": epoch + 1, "val_loss": val_loss, "val_perplexity": math.exp(min(val_loss, 20))}
        if args.eval_rouge_samples:
            sub = val_df.head(args.eval_rouge_samples)
            preds = generator.generate(sub["dialogue"].tolist(), batch_size=args.batch_size * 2)
            agg = aggregate(score_pairs(preds, sub["summary"].tolist()), n_boot=50)
            row.update({f"val_{t}": agg[t] for t in ("rouge1", "rouge2", "rougeL")})
        tracker.log(row, global_step)
        val_rows.append({"epoch": epoch + 1, "step": global_step, **row})
        logger.info("Epoch %d: val loss %.4f (ppl %.2f)%s", epoch + 1, val_loss, row["val_perplexity"],
                    f" | val ROUGE-L {row['val_rougeL']:.2f}" if "val_rougeL" in row else "")
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch + 1
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            logger.info("  best so far -> adapter saved to %s", out_dir)
        if stop:
            break
    train_seconds = time.perf_counter() - t_start

    # ---- summary ---------------------------------------------------------------------------
    peak = resources.peak_memory_gb(device)
    train_rows = [r for r in tracker.rows if "loss" in r]
    summary = {
        "version": __version__, "base_model": resolve_model_id(args.base_model), "quantization": quantization,
        "compute_dtype": str(cdtype).replace("torch.", ""), "output": str(out_dir),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": args.target_modules},
        "data": {"dataset": args.train_file and str(args.train_file) or args.dataset_id, "train_examples": len(train_ds),
                 "validation_examples": len(val_ds), "mean_tokens": float(np.mean(train_ds.lengths)),
                 "truncated": train_ds.n_truncated, "stats": dataset_stats(train_df)},
        "optimisation": {"epochs": args.epochs, "steps": global_step, "planned_steps": total_steps, "batch_size": args.batch_size,
                         "grad_accum": args.grad_accum, "effective_batch": args.batch_size * args.grad_accum,
                         "lr": args.lr, "scheduler": args.lr_scheduler, "optimizer": args.optim,
                         "max_grad_norm": args.max_grad_norm, "gradient_checkpointing": not args.no_gradient_checkpointing},
        "parameters": params,
        "resources": {"peak_memory_gb": round(peak, 3), "device": str(device), "gpu": resources.gpu_info(),
                      "base_model_memory_gb": round(base_footprint, 3), "train_seconds": round(train_seconds, 1),
                      "tokens_seen": tokens_seen,
                      "tokens_per_second": round(tokens_seen / max(train_seconds, 1e-9), 1),
                      "adapter_size_mb": round(resources.dir_size_mb(out_dir / "adapter_model.safetensors"), 2)
                      if (out_dir / "adapter_model.safetensors").exists() else None},
        "best_epoch": best_epoch, "best_val_loss": best_val if math.isfinite(best_val) else None,
        "final_train_loss": train_rows[-1]["loss"] if train_rows else None,
        "validation": val_rows, "environment": resources.environment(), "seed": args.seed,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (config.METRICS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    tracker.summary({"best_val_loss": summary["best_val_loss"], "peak_memory_gb": peak, "trainable_percent":
                     params["trainable_percent"], "train_seconds": train_seconds})
    log_path = tracker.flush()
    pd.read_csv(log_path).to_csv(config.METRICS_DIR / "train_log.csv", index=False)
    tracker.finish()
    if not args.no_plots and tracker.rows:
        visualize.plot_training_curves(pd.read_csv(log_path), config.FIGURES_DIR)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    try:
        summary = train(args)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, ModelLoadError, TrackingUnavailable) as err:
        logger.error("%s", err.args[0] if err.args else err)
        return 1
    except QuantizationUnavailable as err:
        logger.error("%s", err)
        return 1
    res, par = summary["resources"], summary["parameters"]
    print(f"\nAdapter saved to {summary['output']} (best epoch {summary['best_epoch']}, "
          f"val loss {summary['best_val_loss']:.4f})")
    if summary["optimisation"]["steps"] < summary["optimisation"]["planned_steps"]:
        print(f"Paused at step {summary['optimisation']['steps']} of {summary['optimisation']['planned_steps']} "
              "- see the 'Resume with' line above to continue.")
    print(f"Trainable parameters {par['trainable']:,} ({par['trainable_percent']:.3f}% of {par['total']:,}) | "
          f"peak memory {res['peak_memory_gb']:.2f} GB | {res['train_seconds'] / 60:.1f} min")
    print(f"Next: python -m qlora_finetune.evaluate --base-model {args.base_model} --adapter {summary['output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
