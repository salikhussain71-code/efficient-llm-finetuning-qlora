.PHONY: install install-gpu install-dev train evaluate merge push generate tiny-model smoke test lint clean

install:
	pip install -r requirements.txt
	pip install -e .

# On an NVIDIA GPU (Colab T4 etc.): adds bitsandbytes and wandb
install-gpu:
	pip install -r requirements-gpu.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

# QLoRA fine-tuning of Phi-2 on 4,000 SAMSum dialogues (needs a GPU; see README for a CPU alternative)
train:
	python -m qlora_finetune.train --base-model phi-2

# Base vs fine-tuned vs a larger reference model, 200 test dialogues
evaluate:
	python -m qlora_finetune.evaluate --base-model phi-2 --adapter models/phi-2-samsum-qlora --reference-model mistral-7b

merge:
	python -m qlora_finetune.merge --base-model phi-2 --adapter models/phi-2-samsum-qlora --output models/phi-2-samsum-merged

push:
	python -m qlora_finetune.push --path models/phi-2-samsum-qlora --dry-run

generate:
	python -m qlora_finetune.generate --base-model phi-2 --adapter models/phi-2-samsum-qlora --quantization 4bit "Anna: Are you coming on Friday?\nBen: Yes, I will bring cake."

# Tiny random Llama so the whole pipeline can be exercised OFFLINE on a CPU (summaries are only meaningful on the
# synthetic sample, which a tiny model can learn)
tiny-model:
	python scripts/generate_sample_data.py
	python -m qlora_finetune.testing --out models/tiny-llama --corpus-csv data/sample/train.csv

# Outputs go to outputs/smoke-reports (git-ignored) so synthetic results never land in reports/
smoke: export QF_REPORTS_DIR = outputs/smoke-reports
smoke: tiny-model
	python -m qlora_finetune.train --base-model models/tiny-llama --train-file data/sample/train.csv \
	  --validation-file data/sample/validation.csv --max-train-samples 0 --epochs 8 --batch-size 8 --grad-accum 1 \
	  --lr 3e-3 --max-seq-len 160 --eval-rouge-samples 8 --max-new-tokens 40 --no-gradient-checkpointing \
	  --output models/tiny-adapter
	python -m qlora_finetune.evaluate --base-model models/tiny-llama --adapter models/tiny-adapter \
	  --train-file data/sample/train.csv --test-file data/sample/test.csv --max-samples 0 --max-new-tokens 40 \
	  --base-shots 2 --no-update-readme
	python -m qlora_finetune.merge --base-model models/tiny-llama --adapter models/tiny-adapter --output models/tiny-merged
	python -m qlora_finetune.push --path models/tiny-adapter --dry-run \
	  --results-file $(QF_REPORTS_DIR)/metrics/results.json --training-summary $(QF_REPORTS_DIR)/metrics/training_summary.json

test:
	pytest

lint:
	ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info wandb
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
