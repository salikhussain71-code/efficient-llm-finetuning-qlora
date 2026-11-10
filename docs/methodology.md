# Methodology

## 1. Problem
Large language models summarise well, but the strongest ones are expensive to call and too big to run privately. The goal
is a **small model that runs on your own hardware** and is *specialised* for one task - summarising chat dialogues
(SAMSum) - and that, on that task, matches or beats a larger general model. Two constraints shape the work:

1. **Single 16 GB GPU** (a Colab T4): full fine-tuning of even a 2.7B model (weights, gradients and Adam state in 16-bit
   and 32-bit) needs far more memory than that.
2. **Evidence**: success is shown by comparing ROUGE (and BLEU) of the base model against the fine-tuned model, with
   uncertainty estimates, not by anecdotes.

## 2. LoRA: train a low-rank update instead of the weights
For a frozen weight matrix `W` (shape `d x k`), LoRA learns `W' = W + (alpha / r) * B @ A` with `A` of shape `r x k` and
`B` of shape `d x r`, `r << min(d, k)`. Only `A` and `B` receive gradients. A `4096 x 4096` layer has 16.8M weights but its
rank-16 adapter has only `16 * (4096 + 4096) = 131k` trainable parameters (0.8 %). Adapting every linear layer
(`target_modules="all-linear"`, the choice recommended in the QLoRA paper) typically trains roughly 0.5-2 % of the parameters (the exact share is printed at the start of every run).
The adapter is a small file (tens of MB) that can be swapped, shared, or merged back into the base weights.

## 3. QLoRA: a 4-bit frozen base
QLoRA keeps the frozen base weights in **4-bit NormalFloat (NF4)**, a data type designed for normally distributed weights,
with **double quantisation** (the quantisation constants are quantised too) and optional **paged optimisers** that spill
optimiser state to CPU memory during memory spikes. Computation is done in 16-bit (`bfloat16`, or `float16` on a T4 which
has no bfloat16 support); the LoRA matrices are kept in higher precision. Gradients flow *through* the 4-bit weights into
the adapters.

Weights-only memory (`reports/figures/memory_estimate.png`):

| Model | fp16 / bf16 | 8-bit | 4-bit NF4 |
|---|---:|---:|---:|
| Phi-2 (2.7B) | 5.4 GB | 2.7 GB | about 1.4 GB |
| Mistral-7B (7.2B) | 14.4 GB | 7.2 GB | about 3.8 GB |
| Llama-3-8B (8.0B) | 16.0 GB | 8.0 GB | about 4.2 GB |

Training additionally needs activations, the adapter gradients and optimiser state, and the CUDA context, so these figures
are a **lower bound**; the *measured* peak memory is written to `reports/metrics/training_summary.json`. Levers used here:
4-bit base, adapters only, **gradient checkpointing** (recompute activations in the backward pass), **micro-batch 4 with
gradient accumulation 4** (effective batch 16), **length-grouped batches**, `max_seq_len=512`, optional `paged_adamw_8bit`.

## 4. Training setup (`train.py`)
| Setting | Default | Rationale |
|---|---|---|
| Quantisation | 4-bit NF4 + double quant (auto on a CUDA GPU) | QLoRA |
| LoRA | r=16, alpha=32, dropout 0.05, all linear layers | common QLoRA configuration |
| Learning rate | 2e-4, cosine decay, 3 % warm-up | QLoRA paper value for 7-13B models |
| Gradient clipping | 0.3 | QLoRA paper |
| Effective batch | 16 | 4 x 4 accumulation |
| Data | 4,000 random training dialogues, 1 epoch (`--max-train-samples 0` for all 14.7k) | fits a Colab session |
| Loss | cross-entropy on the summary tokens only | do not waste capacity on copying the dialogue |
| Mixed precision | bf16 where supported; fp16 + `GradScaler` on a T4 | T4 has no bf16 |

Reproducibility and robustness: fixed seeds; batch order derived from `(seed, epoch)`; **resumable checkpoints**
(`--save-steps`, `--stop-after-steps`, `--resume-from`) storing the adapter, optimiser, scheduler and scaler state - a
test verifies that a run paused at step 10 and resumed reaches the same adapter as an uninterrupted run; every run writes
`training_summary.json` (configuration, parameter counts, peak memory, time, library versions) and `train_log.csv`
(optionally mirrored to Weights & Biases with `--report-to wandb`).

## 5. Evaluation protocol (`evaluate.py`)
Systems, all using the **same dialogues, prompt, greedy decoding and output cleaning**:

| System | Purpose |
|---|---|
| Base model, zero-shot | what you get without any training |
| Base model + K in-context examples (`--base-shots`) | a stronger, fairer baseline than zero-shot |
| Base + LoRA adapter | the fine-tuned model |
| Merged model | checks that merging preserved the quality |
| Larger reference model (`--reference-model`) | the "small model beats large model" comparison |

Metrics: ROUGE-1/2/L/Lsum (`rouge-score`, Porter stemming; Lsum is computed per sentence) and corpus BLEU (sacreBLEU),
reported as F1 x 100, plus summary length, a novel-bigram rate (abstractiveness), peak memory and time per example.

Statistics: **bootstrap 95 % confidence intervals** for every score and a **paired bootstrap** of the per-dialogue
difference between systems. The text of the findings is generated from the intervals: a system is called "significantly
better" only if the 95 % interval of the ROUGE-L difference excludes zero. Small evaluation samples (the default is 200
dialogues) give wide intervals - evaluate the full 819-dialogue test set for final numbers.

## 6. Merging and publishing
`merge.py` loads the base model un-quantised, applies the adapter and saves a standalone safetensors model. It compares
summaries and logits before and after merging on built-in dialogues and writes `merge_report.json`. A QLoRA adapter was
trained against 4-bit weights, so after merging into 16-bit weights the behaviour differs very slightly - run
`evaluate --merged` to confirm. `push.py` uploads an adapter or merged model to the Hub (private unless `--public`) with an
auto-generated model card that includes the training configuration, the evaluation table and the licence note.

## 7. Verification status of this repository
* **Tested offline on CPU (CI):** data handling, prompt format, completion-only masking, LoRA wrapping, training loop,
  checkpoint/resume, generation, ROUGE/BLEU/bootstrap, significance wording, merging (identical outputs, logit difference
  below 1e-3), model card, Hub upload logic (mocked), CLIs. The tests use a tiny random Llama model, and on the synthetic
  sample they confirm that fine-tuning raises ROUGE far above the untrained model.
* **Requires your GPU (not run here):** the 4-bit (bitsandbytes) code path, the paged 8-bit optimiser, fp16 `GradScaler`
  training, real model downloads (Phi-2 / Mistral / Llama), real SAMSum numbers and the actual Hub upload. The code for
  these paths follows the documented PEFT / bitsandbytes APIs and fails early with clear messages when prerequisites
  are missing, but the first GPU run is the real test - the Colab notebook is designed for exactly that.

## 8. Limitations
* ROUGE / BLEU reward word overlap and say nothing about factual consistency; read `reports/examples.md`.
* A zero-shot *base* (non-instruction-tuned) model is a weak baseline; a larger base model is also not "the best a large
  model can do". That is why few-shot baselines and a fixed prompt are provided, and why the claims are stated relative to
  the measured baselines only.
* Single training seed; adapter hyper-parameters were not tuned. Report variability before drawing strong conclusions.
* SAMSum is English, non-commercial (CC BY-NC-ND 4.0) and not representative of all chat styles.
* No safety, bias or hallucination evaluation beyond manual reading of examples.

## 9. Extensions
DoRA / rank and target-module ablations, instruction-tuned base models, longer training on the full dataset, learning-rate
sweeps with W&B, BERTScore or factual-consistency metrics, 8-bit vs 4-bit comparison, inference with a merged model in
GGUF form.

## References
Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021). Dettmers et al., *QLoRA: Efficient Finetuning of
Quantized LLMs* (2023). Gliwa et al., *SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization*
(2019). Hugging Face `transformers`, `peft`, `bitsandbytes`.
