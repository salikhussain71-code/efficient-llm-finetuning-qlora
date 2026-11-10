"""Generate a Hugging Face model card (README.md with YAML front matter) for an adapter or merged model."""

from __future__ import annotations

import json
from pathlib import Path

SAMSUM_LICENCE_NOTE = (
    "SAMSum is distributed under the **CC BY-NC-ND 4.0** licence (non-commercial use). Check the dataset card and the "
    "licence of the base model before using this model commercially.")


def _results_table(results: dict | None) -> str:
    if not results or not results.get("results"):
        return "_Run `python -m qlora_finetune.evaluate` and pass `--results-file` to include metrics here._"
    lines = ["| System | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | BLEU |", "|---|---:|---:|---:|---:|---:|"]
    for r in results["results"]:
        lines.append(f"| {r['system']} | {r['rouge1']:.2f} | {r['rouge2']:.2f} | {r['rougeL']:.2f} | "
                     f"{r['rougeLsum']:.2f} | {r['bleu']:.2f} |")
    note = (f"\n{results['n_examples']} random `{results['split']}` dialogues; greedy decoding; scores are F1 x 100. "
            "Small samples have wide confidence intervals.")
    return "\n".join(lines) + note


def build_model_card(kind: str, repo_id: str, base_model: str, dataset_id: str = "knkarthick/samsum",
                     training: dict | None = None, results: dict | None = None) -> str:
    """``kind`` is ``adapter`` (a LoRA adapter) or ``merged`` (a standalone model)."""
    if kind not in ("adapter", "merged"):
        raise ValueError("kind must be 'adapter' or 'merged'")
    tags = ["peft", "lora", "qlora", "summarization", "dialogue-summarization", "samsum"] if kind == "adapter" else \
        ["lora", "qlora", "summarization", "dialogue-summarization", "samsum", "merged"]
    front = ["---", f"base_model: {base_model}", f"library_name: {'peft' if kind == 'adapter' else 'transformers'}",
             "pipeline_tag: text-generation", "language:", "- en", "datasets:", f"- {dataset_id}", "tags:"]
    front += [f"- {t}" for t in tags] + ["---", ""]

    usage_adapter = (f'```python\nimport torch\nfrom peft import PeftModel\n'
                     f'from transformers import AutoModelForCausalLM, AutoTokenizer\n\n'
                     f'base = AutoModelForCausalLM.from_pretrained("{base_model}", torch_dtype=torch.float16, '
                     f'device_map="auto")\ntokenizer = AutoTokenizer.from_pretrained("{base_model}")\n'
                     f'model = PeftModel.from_pretrained(base, "{repo_id}")\n```')
    usage_merged = (f'```python\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\n'
                    f'model = AutoModelForCausalLM.from_pretrained("{repo_id}", device_map="auto")\n'
                    f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")\n```')
    prompt = ('```text\nSummarize the following dialogue.\n\n### Dialogue:\n{dialogue}\n\n### Summary:\n```')

    quant = (training or {}).get("quantization")
    method = ("QLoRA (4-bit NF4 base weights, trainable low-rank adapters)" if quant == "4bit"
              else "LoRA (trainable low-rank adapters)" if quant else "LoRA / QLoRA")
    body = [f"# {repo_id.split('/')[-1]}", "",
            f"{'A LoRA adapter' if kind == 'adapter' else 'A merged model'} that summarises chat dialogues, obtained by "
            f"fine-tuning `{base_model}` on SAMSum with {method}.", "",
            "## Intended use", "",
            "Short summaries of informal two- or multi-person chat dialogues in English. Not a general-purpose "
            "assistant; not evaluated for safety, factual consistency or other languages.", "",
            "## How to use", "", usage_adapter if kind == "adapter" else usage_merged, "",
            "Use exactly this prompt format (it is the format used for training):", "", prompt, ""]
    if training:
        o, lo, par, res = training["optimisation"], training["lora"], training["parameters"], training["resources"]
        body += ["## Training", "", "| | |", "|---|---|",
                 f"| Base model | `{training['base_model']}` |", f"| Quantisation | {training['quantization']} |",
                 f"| LoRA | r={lo['r']}, alpha={lo['alpha']}, dropout={lo['dropout']}, targets={lo['target_modules']} |",
                 f"| Trainable parameters | {par['trainable']:,} ({par['trainable_percent']:.2f}%) |",
                 f"| Training examples | {training['data']['train_examples']:,} |",
                 f"| Epochs / steps | {o['epochs']} / {o['steps']} |",
                 f"| Effective batch size | {o['effective_batch']} |",
                 f"| Learning rate / scheduler | {o['lr']} / {o['scheduler']} |",
                 f"| Peak memory | {res['peak_memory_gb']:.1f} GB |",
                 f"| Hardware | {(res.get('gpu') or {}).get('name', res['device'])} |", ""]
    body += ["## Evaluation", "", _results_table(results), "",
             "## Limitations and licence", "",
             "- Summaries can contain errors or invented details; ROUGE does not measure factual consistency.",
             "- Trained on SAMSum (English, messenger-style conversations written by linguists); other kinds of text "
             "or languages will give lower quality.",
             f"- {SAMSUM_LICENCE_NOTE}", ""]
    return "\n".join(front + body)


def load_json(path: str | Path | None) -> dict | None:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return None
