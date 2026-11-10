# Efficient LLM Fine-Tuning with QLoRA: Dialogue Summarization on SAMSum

[![CI](https://github.com/salikhussain71-code/efficient-llm-finetuning-qlora/actions/workflows/ci.yml/badge.svg)](https://github.com/salikhussain71-code/efficient-llm-finetuning-qlora/actions)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

This project fine-tunes a small language model (Phi-2 by default; Mistral-7B and Llama-3-8B are supported) to summarise
chat dialogues, using **QLoRA** so that training fits on a **single 16 GB GPU** such as a free Colab T4. It then measures
the result against the untrained base model and a larger reference model with ROUGE and BLEU, including confidence
intervals and paired significance tests, merges the adapter into a standalone model and publishes it to the Hugging Face Hub.

> **Task 9 - Industry Level (NLP Tasks).** Dataset: SAMSum. Method: LoRA / QLoRA (PEFT, bitsandbytes). Evidence: ROUGE /
> BLEU of base vs fine-tuned model. Bonus items: adapter merging and Hub export.

## Contents
1. [Requirements checklist](#requirements-checklist)
2. [Results](#results)
3. [Repository structure](#repository-structure)
4. [Quick start](#quick-start)
5. [Usage](#usage)
6. [Method in brief](#method-in-brief)
7. [Testing and verification status](#testing-and-verification-status)
8. [Limitations](#limitations)
9. [References and licence](#references-and-licence)

## Requirements checklist

| Requirement | Implementation |
|---|---|
| Dataset: SAMSum | `data.py`, `data/README.md` (Hub id or local files) |
| Fine-tune a small LM (Llama 3 8B, Mistral, Phi-2) with **QLoRA** | `train.py`, `modeling.py` (NF4 + double quantisation, LoRA on all linear layers), presets in `presets.py` |
| **Single GPU** (Colab T4) | 4-bit base, gradient checkpointing, micro-batching with accumulation, fp16 + GradScaler on T4, paged 8-bit AdamW option; peak memory is measured and reported |
| Prove success: ROUGE of base vs fine-tuned model | `evaluate.py`: ROUGE-1/2/L/Lsum + BLEU, bootstrap CIs, paired significance tests, optional larger reference model |
| Tools: Python, Hugging Face (PEFT / BitsAndBytes), PyTorch, WandB | all used; W&B via `--report-to wandb` |
| Topics: LoRA/QLoRA, quantisation, ROUGE/BLEU | see [Method in brief](#method-in-brief) and `docs/methodology.md` |
| **Bonus:** merge LoRA adapters into a standalone model | `merge.py` (with before/after equivalence check) |
| **Bonus:** push the model to the Hub | `push.py` (auto-generated model card, private by default, `--dry-run`) |

## Results

<!-- RESULTS:START -->
_Run the training and evaluation commands below (a Colab T4 is enough) - this section is then filled in automatically
with the ROUGE / BLEU comparison, the significance tests and the training budget._
<!-- RESULTS:END -->

How to read the table: ROUGE and BLEU are F1 x 100. The *Findings* lines are generated from the paired bootstrap
confidence interval of the per-dialogue ROUGE-L difference: a system is called significantly better only when the whole
95 % interval is above zero. With the default 200 evaluation dialogues the intervals are wide; use `--max-samples 0` for
the full 819-dialogue test set.

## Repository structure

```
efficient-llm-finetuning-qlora/
├── README.md
├── LICENSE
├── Makefile                       # make train / evaluate / merge / push / smoke / test
├── pyproject.toml                 # package metadata, pytest and ruff configuration
├── requirements.txt               # core dependencies
├── requirements-gpu.txt           # + bitsandbytes, wandb (NVIDIA GPU)
├── requirements-dev.txt           # + pytest, ruff, jupyter
├── .github/workflows/ci.yml       # lint + tests on every push (CPU, offline)
├── data/
│   ├── README.md                  # SAMSum, licence, local-file format, prompt format
│   ├── raw/                       # optional local data (git-ignored)
│   └── sample/                    # synthetic SAMSum-style CSVs for tests only
├── docs/methodology.md            # LoRA / QLoRA, memory budget, protocol, limitations
├── notebooks/
│   └── colab_qlora_samsum.ipynb   # the whole workflow on a free Colab T4
├── scripts/                       # generate_sample_data.py, build_notebook.py
├── src/qlora_finetune/
│   ├── config.py  presets.py      # paths; model presets (TinyLlama, Phi-2, Mistral-7B, Llama-3-8B)
│   ├── prompts.py  data.py        # prompt format, loading, completion-only masking, batching
│   ├── modeling.py                # quantised loading, LoRA, parameter and memory accounting
│   ├── train.py                   # QLoRA training loop, checkpoint/resume, W&B
│   ├── generate.py                # batched generation with the training prompt
│   ├── metrics.py  evaluate.py    # ROUGE, BLEU, bootstrap, paired significance; comparison CLI
│   ├── merge.py  push.py          # adapter merging; Hub upload; modelcard.py
│   ├── resources.py  tracking.py  # memory/time accounting; CSV + W&B tracking
│   ├── visualize.py  testing.py   # figures; tiny offline model builder
├── tests/                         # 85 pytest tests
├── models/                        # adapters / merged models (git-ignored)
└── reports/                       # figures, metrics, per-example predictions, examples.md
```

## Quick start

### Recommended: Google Colab (free T4 GPU)
Open `notebooks/colab_qlora_samsum.ipynb` in Colab, choose **Runtime > Change runtime type > T4 GPU**, and run the cells.
It installs the requirements, trains, evaluates against Mistral-7B, merges and (optionally) pushes to the Hub.

### Local NVIDIA GPU
```bash
git clone https://github.com/salikhussain71-code/efficient-llm-finetuning-qlora.git
cd efficient-llm-finetuning-qlora
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# install the PyTorch build for your CUDA version first (pytorch.org), then:
pip install -r requirements-gpu.txt
pip install -e .

python -m qlora_finetune.train --base-model phi-2                          # QLoRA, 4,000 dialogues, 1 epoch
python -m qlora_finetune.evaluate --base-model phi-2 --adapter models/phi-2-samsum-qlora \
    --reference-model mistral-7b --base-shots 3 --max-samples 200
```
Models and the dataset are downloaded from the Hugging Face Hub on first use (internet required). Llama 3 is gated: accept
its licence on the Hub and run `huggingface-cli login`.

### No GPU: verify the whole pipeline on a CPU in about a minute
```bash
pip install -r requirements.txt && pip install -e .
make smoke      # tiny random Llama + synthetic sample: trains, evaluates (base vs fine-tuned), merges, dry-runs the push
```
This proves the code path, not the model quality: the tiny model is trained on the synthetic sample only. Its outputs go
to `outputs/smoke-reports/` (git-ignored), so synthetic results never end up in `reports/`.

## Usage

### Train
```bash
python -m qlora_finetune.train --base-model phi-2                                   # defaults: 4-bit, r=16, 4,000 examples
python -m qlora_finetune.train --base-model phi-2 --max-train-samples 0 --epochs 2  # the whole training split
python -m qlora_finetune.train --base-model mistral-7b --optim paged_adamw_8bit     # 7B on a T4
python -m qlora_finetune.train --report-to wandb --wandb-project qlora-samsum --run-name phi2-r16
python -m qlora_finetune.train --stop-after-steps 150          # pause (e.g. before a session limit) ...
python -m qlora_finetune.train --resume-from models/phi-2-samsum-qlora/checkpoints/step-000150   # ... and continue
python -m qlora_finetune.train --help                                                # every option
```
Writes the adapter, `training_summary.json` (parameters, peak memory, time, versions), `train_log.csv` and training curves.

### Evaluate: base vs fine-tuned vs larger model
```bash
python -m qlora_finetune.evaluate --base-model phi-2 --adapter models/phi-2-samsum-qlora \
    --reference-model mistral-7b --base-shots 3 --max-samples 0
```
Several adapters can be compared (`--adapter models/a models/b`). Outputs: `reports/metrics/results.{csv,md,json}`,
`significance.csv`, per-system predictions, figures and `reports/examples.md`, plus the README **Results** section.

### Merge and publish (bonus)
```bash
python -m qlora_finetune.merge --base-model phi-2 --adapter models/phi-2-samsum-qlora --output models/phi-2-samsum-merged
python -m qlora_finetune.evaluate --base-model phi-2 --merged models/phi-2-samsum-merged --no-base   # same quality?
huggingface-cli login
python -m qlora_finetune.push --path models/phi-2-samsum-qlora --repo-id salikhussain71-code/phi-2-samsum-qlora --dry-run
python -m qlora_finetune.push --path models/phi-2-samsum-qlora --repo-id salikhussain71-code/phi-2-samsum-qlora   # private
```
Add `--public` to publish publicly. Training checkpoints are never uploaded. Review the generated model card first.

### Summarise a dialogue
```bash
python -m qlora_finetune.generate --base-model phi-2 --adapter models/phi-2-samsum-qlora --quantization 4bit \
    "Anna: Are you coming on Friday?\nBen: Yes, I will bring cake."
```

## Method in brief
* **LoRA** freezes the pre-trained weights and learns a low-rank update `W + (alpha/r)·B·A` for every linear layer.
* **QLoRA** additionally stores the frozen base in 4-bit NormalFloat with double quantisation, so a 7B model's weights
  need about 4 GB instead of 14 GB; gradients flow through the 4-bit weights into the adapters.
* **Completion-only loss**: only the summary tokens are trained on. Prompt format, truncation and batching are shared
  by training, evaluation and inference.
* **Evaluation** uses one prompt, greedy decoding and one output-cleaning rule for every system, bootstrap confidence
  intervals, and a paired bootstrap for the base-vs-fine-tuned comparison.

Details, memory arithmetic and design choices: [`docs/methodology.md`](docs/methodology.md).

## Testing and verification status

```bash
pip install -r requirements-dev.txt
pytest          # 85 tests, offline, under a minute on a CPU
ruff check .
```
The tests use a tiny random Llama model and a learnable synthetic sample. They cover data handling, masking, LoRA, training
(including that a paused-and-resumed run reproduces an uninterrupted one), generation, metrics, significance wording,
merging (identical outputs after merging), the model card and Hub upload logic (mocked).

**Not exercised in CI because it needs an NVIDIA GPU or network access:** 4-bit loading with bitsandbytes, the paged 8-bit
optimiser, fp16 GradScaler training, downloading real models and SAMSum, and the actual Hub upload. These paths use the
documented PEFT / bitsandbytes APIs and fail early with clear messages if a prerequisite is missing; the Colab notebook is
the intended first end-to-end run, and the numbers in the Results section come from that run.

## Limitations
* ROUGE and BLEU measure word overlap, not factual consistency; always read `reports/examples.md`.
* A zero-shot base model is a weak baseline, and a larger base model is not the best a large model can do - hence the
  few-shot baseline option and the careful wording of the findings.
* One training seed and untuned hyper-parameters; the default evaluation uses 200 dialogues.
* SAMSum is English and licensed CC BY-NC-ND 4.0 (non-commercial); the base models have their own licences.

## References and licence
Released under the [MIT License](LICENSE) (code). Hu et al., *LoRA* (2021); Dettmers et al., *QLoRA* (2023); Gliwa et al.,
*SAMSum Corpus* (2019). Built on Hugging Face `transformers`, `peft`, `datasets`, `bitsandbytes` and PyTorch.
