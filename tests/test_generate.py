import pandas as pd
import torch

from qlora_finetune import generate
from qlora_finetune.generate import SummaryGenerator


def echo_first_line(gen):
    """Replace ``model.generate`` by a stub that 'summarises' each prompt as the first line of its dialogue."""
    tok = gen.tokenizer
    gen.seen = []

    def fake(input_ids=None, attention_mask=None, **kw):
        rows = []
        for row in input_ids:
            text = tok.decode(row, skip_special_tokens=True)
            gen.seen.append(text)
            dialogue = text.split("### Dialogue:\n")[-1].split("\n\n### Summary:")[0]
            rows.append(torch.tensor(tok(dialogue.split("\n")[0], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]))
        width = max(len(r) for r in rows)
        new = torch.stack([torch.cat([r, torch.full((width - len(r),), tok.pad_token_id)]) for r in rows])
        return torch.cat([input_ids, new], dim=1)

    gen.model.generate = fake
    return gen


def test_results_come_back_in_input_order(tiny_model):
    gen = echo_first_line(SummaryGenerator.load(tiny_model))
    dialogues = [f"Name{i}: " + "word " * (i * 3 % 17 + 1) for i in range(11)]       # lengths all different
    out = gen.generate(dialogues, batch_size=3)
    assert out == [d.strip() for d in dialogues]


def test_state_is_restored_after_generation(tiny_model):
    gen = SummaryGenerator.load(tiny_model, max_new_tokens=5)
    gen.model.train()
    gen.model.config.use_cache = False
    gen.tokenizer.padding_side = "right"
    gen.generate(["A: hi\nB: yo"])
    assert gen.model.training and gen.model.config.use_cache is False and gen.tokenizer.padding_side == "right"


def test_left_padding_is_used_for_generation(tiny_model):
    gen = echo_first_line(SummaryGenerator.load(tiny_model))
    seen_sides = []
    original = gen.model.generate

    def spy(input_ids=None, attention_mask=None, **kw):
        seen_sides.append((gen.tokenizer.padding_side, bool(attention_mask[:, 0].min() == 0)))
        return original(input_ids=input_ids, attention_mask=attention_mask, **kw)

    gen.model.generate = spy
    gen.generate(["A: short", "A: " + "much longer dialogue " * 5], batch_size=2)
    assert seen_sides == [("left", True)]            # padded on the left: the first position of the short row is padding


def test_long_dialogues_are_truncated_to_the_prompt_budget(tiny_model):
    gen = echo_first_line(SummaryGenerator.load(tiny_model, max_prompt_tokens=60))
    gen.generate(["Anna: " + "blah " * 400])
    assert len(gen.tokenizer(gen.seen[0])["input_ids"]) <= 62


def test_few_shot_examples_are_in_the_prompt(tiny_model):
    gen = echo_first_line(SummaryGenerator.load(tiny_model, shots=[("X: a", "X says a.")]))
    gen.generate(["Y: query"])
    assert "X says a." in gen.seen[0] and gen.seen[0].count("### Dialogue:") == 2


def test_real_generation_with_and_without_adapter(tiny_model, trained_adapter, files):
    dev = pd.read_csv(files["validation"])["dialogue"].tolist()[:5]
    base = SummaryGenerator.load(tiny_model, max_new_tokens=20).generate(dev)
    tuned = SummaryGenerator.load(tiny_model, trained_adapter, max_new_tokens=20, name="tuned")
    out = tuned.generate(dev, batch_size=2)
    assert len(base) == len(out) == 5 and all(isinstance(x, str) for x in out) and base != out
    assert tuned.name == "tuned" and tuned.generate([]) == []
    tuned.close()
    assert tuned.model is None


def test_beam_search_runs(tiny_model):
    assert len(SummaryGenerator.load(tiny_model, max_new_tokens=8, num_beams=2).generate(["A: hi\nB: yo"])) == 1


def test_cli(tiny_model, trained_adapter, tmp_path, capsys):
    f = tmp_path / "chat.txt"
    f.write_text("Anna: Are you coming on Friday?\nBen: Yes, I will bring cake.", encoding="utf-8")
    assert generate.main(["--base-model", tiny_model, "--adapter", trained_adapter, "--file", str(f),
                          "--max-new-tokens", "30"]) == 0
    out = capsys.readouterr().out
    assert "--- Dialogue ---" in out and "Anna: Are you coming on Friday?\nBen:" in out and "--- Summary (" in out
    assert generate.main(["--base-model", tiny_model, "--max-new-tokens", "5", "A: hi\\nB: yo"]) == 0
    assert "A: hi\nB: yo" in capsys.readouterr().out                         # literal \n becomes a line break
    assert generate.main([]) == 1
    assert generate.main(["--base-model", str(tmp_path / "missing"), "text"]) == 1
    assert generate.main(["--base-model", tiny_model, "--quantization", "4bit", "text"]) == 1      # no GPU here
    capsys.readouterr()
