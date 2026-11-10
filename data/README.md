# Data

Datasets are not committed to Git (see `.gitignore`).

## SAMSum (Hugging Face)

* Messenger-style chat dialogues written by linguists, each with a human-written abstractive summary.
* Size: 14,732 training / 818 validation / 819 test dialogues. Columns: `id`, `dialogue`, `summary`.
* Default Hub id used by the scripts: `knkarthick/samsum` (a Parquet copy of the corpus). The original corpus is also
  available as `Samsung/samsum`; pass it with `--dataset-id`.
* **Licence: CC BY-NC-ND 4.0** (non-commercial). Check the dataset card before any commercial use; the model card that
  `qlora_finetune.push` generates repeats this note.

The dataset is downloaded automatically (and cached by `datasets`) the first time you run `train` or `evaluate`; an
internet connection is required for that step. The Hub id and the split sizes above are from memory - confirm them on
the dataset page.

## Local files

Any CSV / JSON / JSONL / Parquet files with a dialogue column and a summary column work:

```bash
python -m qlora_finetune.train --train-file my_train.csv --validation-file my_val.csv \
    --dialogue-col chat --summary-col abstract
python -m qlora_finetune.evaluate --test-file my_test.csv --train-file my_train.csv --base-model phi-2 --adapter models/x
```
Local files take precedence over `--dataset-id`. Rows with an empty dialogue or summary are dropped; line endings are
normalised.

## What the pipeline does to each example (`prompts.py`, `data.py`)

```text
Summarize the following dialogue.

### Dialogue:
Anna: Are you coming on Friday?
Ben: Yes, I will bring cake.

### Summary:
Anna asks Ben if he is coming on Friday. Ben will bring cake.<eos>
```

* **Completion-only loss**: label ids are masked (`-100`) over the instruction and dialogue, so the model is trained only
  on the summary tokens and the end-of-sequence token.
* **Truncation**: dialogues longer than `--max-seq-len` are cut (never the summary).
* **Length-grouped batches**: batches are built from similar-length examples to waste less compute on padding; the
  order is reproducible from `(seed, epoch)`, which is what makes exact resumption possible.
* The same prompt builder is used for training, evaluation and inference.

## Bundled sample (`data/sample/`)

`train.csv` (200), `validation.csv` (30) and `test.csv` (40) are **synthetic**, template-generated dialogues in the SAMSum
layout, created by `scripts/generate_sample_data.py`. Their summaries are deterministic functions of the dialogue so that
a tiny model can learn them - this lets the tests and CI verify that fine-tuning really improves ROUGE without any
download. **They are not real SAMSum data; never report results obtained on them.**
