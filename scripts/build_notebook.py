"""Builds notebooks/colab_qlora_samsum.ipynb (kept as a script so the notebook is reproducible)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s.strip())  # noqa: E731
code = lambda s: nbf.v4.new_code_cell(s.strip())  # noqa: E731

nb.cells = [
md("""
# QLoRA fine-tuning of Phi-2 on SAMSum (Google Colab T4)

**Runtime:** *Runtime > Change runtime type > T4 GPU* (free tier). The notebook runs the whole workflow:

1. install the project and check the GPU
2. QLoRA training (4-bit NF4 base, LoRA adapters) with optional Weights & Biases logging
3. evaluation: base model vs fine-tuned model vs a larger reference model (ROUGE / BLEU, confidence intervals)
4. merge the adapter into a standalone model, and check that merging preserved the quality
5. (optional) push to the Hugging Face Hub

Models and the dataset are downloaded from the Hugging Face Hub, so the session needs internet access.
Training 4,000 dialogues for one epoch is sized for a Colab session; raise `--max-train-samples` for better results.
"""),
code("""
!nvidia-smi
!git clone https://github.com/salikhussain71-code/efficient-llm-finetuning-qlora.git
%cd efficient-llm-finetuning-qlora
!pip install -q -r requirements-gpu.txt
!pip install -q -e .
"""),
md("## 1. (Optional) Weights & Biases and Hugging Face logins"),
code("""
# Weights & Biases: skip this cell if you do not want experiment tracking (then drop --report-to wandb below).
import wandb
wandb.login()
"""),
code("""
# Hugging Face: only needed for gated models (Llama 3) and for pushing to the Hub. Needs a token with *write* access.
from huggingface_hub import notebook_login
notebook_login()
"""),
md("## 2. Train (QLoRA)\\nThe run prints the number of trainable parameters, the memory of the 4-bit base model and, every few steps, loss, learning rate, throughput and peak GPU memory."),
code("""
!python -m qlora_finetune.train --base-model phi-2 --max-train-samples 4000 --epochs 1 --report-to wandb --run-name phi2-qlora-r16
"""),
md("The adapter is saved in `models/phi-2-samsum-qlora`. `reports/metrics/training_summary.json` holds the budget (parameters, peak memory, time)."),
code("""
import json
s = json.load(open("reports/metrics/training_summary.json"))
print("trainable parameters:", f"{s['parameters']['trainable']:,}", f"({s['parameters']['trainable_percent']:.2f}%)")
print("peak GPU memory (GB):", s["resources"]["peak_memory_gb"], "| minutes:", round(s["resources"]["train_seconds"] / 60, 1))
from IPython.display import Image
Image("reports/figures/training_curves.png")
"""),
md("## 3. Evaluate: base vs fine-tuned vs a larger model\\nUses the same prompt and decoding for every system. `mistral-7b` (4-bit) is the larger reference; add `--max-samples 0` for the full 819-dialogue test set."),
code("""
!python -m qlora_finetune.evaluate --base-model phi-2 --adapter models/phi-2-samsum-qlora --reference-model mistral-7b --base-shots 3 --max-samples 200
"""),
code("""
import pandas as pd
display(pd.read_csv("reports/metrics/results.csv")[["system", "params_b", "quantization", "rouge1", "rouge2", "rougeL", "bleu", "peak_memory_gb"]].round(2))
print("\\n".join(json.load(open("reports/metrics/results.json"))["statements"]))
Image("reports/figures/rouge_comparison.png")
"""),
md("Read some examples - ROUGE does not detect invented details:"),
code("""
print(open("reports/examples.md").read()[:3000])
"""),
md("## 4. Merge the adapter into a standalone model (bonus)"),
code("""
!python -m qlora_finetune.merge --base-model phi-2 --adapter models/phi-2-samsum-qlora --output models/phi-2-samsum-merged
"""),
code("""
!python -m qlora_finetune.evaluate --base-model phi-2 --adapter models/phi-2-samsum-qlora --merged models/phi-2-samsum-merged --no-base --max-samples 200 --no-update-readme
"""),
md("## 5. Push to the Hugging Face Hub (bonus, optional)\\nStart with `--dry-run` and review the generated model card. Repositories are private unless you add `--public`."),
code("""
!python -m qlora_finetune.push --path models/phi-2-samsum-qlora --repo-id salikhussain71-code/phi-2-samsum-qlora --dry-run
"""),
code("""
# !python -m qlora_finetune.push --path models/phi-2-samsum-qlora --repo-id salikhussain71-code/phi-2-samsum-qlora
"""),
md("## 6. Try it"),
code("""
!python -m qlora_finetune.generate --base-model phi-2 --adapter models/phi-2-samsum-qlora --quantization 4bit "Anna: Are you coming on Friday?\\nBen: Yes, I will bring cake."
"""),
md("""
## Notes
* If you run out of memory: lower `--max-seq-len`, use `--batch-size 2 --grad-accum 8`, or add `--optim paged_adamw_8bit`.
* To resume after a disconnect: `--resume-from models/phi-2-samsum-qlora/checkpoints/step-XXXXXX` (use `--save-steps 100` when training; checkpoints live on the Colab disk, so copy them to Google Drive if you need them to survive a reset).
* Copy `reports/` and the adapter to your repository and commit them.
"""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nb.metadata["accelerator"] = "GPU"
nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
nbf.write(nb, "notebooks/colab_qlora_samsum.ipynb")
print("notebook written")
