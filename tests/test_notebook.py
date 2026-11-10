"""The Colab notebook is not run in CI (it needs a GPU), so check that every command it contains is valid."""
import shlex
from pathlib import Path

import nbformat
import pytest

from qlora_finetune import evaluate, generate, merge, push, train

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_qlora_samsum.ipynb"


def commands():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    found = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for line in cell.source.splitlines():
            line = line.lstrip("# ").strip() if line.startswith("# !") else line.strip()
            if line.startswith("!python -m qlora_finetune."):
                found.append(line[1:])
    return found


def test_notebook_has_the_expected_steps():
    mods = [shlex.split(c)[2].split(".")[-1] for c in commands()]
    assert mods == ["train", "evaluate", "merge", "evaluate", "push", "push", "generate"]


PARSERS = {"qlora_finetune.train": lambda: train.parse_args, "qlora_finetune.evaluate": lambda: evaluate.parse_args,
           "qlora_finetune.merge": lambda: merge.build_parser().parse_args,
           "qlora_finetune.push": lambda: push.build_parser().parse_args,
           "qlora_finetune.generate": lambda: generate.build_parser().parse_args}


@pytest.mark.parametrize("command", commands())
def test_every_notebook_command_parses(command):
    """A mistyped option in the notebook would raise SystemExit here instead of failing on the user's Colab runtime."""
    parts = shlex.split(command)
    PARSERS[parts[2]]()(parts[3:])
