"""Generate SYNTHETIC SAMSum-style CSV files (train / validation / test) for smoke tests and CI.

Columns: ``id``, ``dialogue`` (``Name: text`` lines), ``summary`` - the same layout as the SAMSum data. The
dialogues come from a few templates, so the summary is a deterministic function of the dialogue: a tiny
model can learn it, which lets the tests verify that fine-tuning really improves ROUGE. This is NOT real
SAMSum data - never report results obtained on it.

    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

NAMES = ["Anna", "Ben", "Carla", "Dev", "Emma", "Farid", "Grace", "Hugo", "Iris", "Jack"]
EVENTS = ["the party", "the concert", "the match", "the dinner", "the meeting", "the picnic"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
ITEMS = ["cake", "bread", "juice", "snacks", "cheese", "fruit"]
PLACES = ["station", "cafe", "library", "park", "office", "garage"]
EXAMS = ["driving test", "math exam", "language exam", "final exam"]
THINGS = ["keys", "umbrella", "charger", "wallet", "laptop"]


def invitation(r, a, b):
    ev, day, item = r.choice(EVENTS), r.choice(DAYS), r.choice(ITEMS)
    return ([f"{a}: Are you coming to {ev} on {day}?", f"{b}: Yes, I will bring {item}."],
            f"{a} asks {b} if {b} is coming to {ev} on {day}. {b} will bring {item}.")


def reschedule(r, a, b):
    d1, d2 = r.sample(DAYS, 2)
    return ([f"{a}: I can't make it on {d1}, sorry.", f"{b}: No problem, let's meet on {d2}."],
            f"{a} cannot make it on {d1}. {a} and {b} will meet on {d2}.")


def shopping(r, a, b):
    item, n = r.choice(ITEMS), r.randint(2, 6)
    return ([f"{a}: I'm at the store. Do we need {item}?", f"{b}: Yes, please buy {n} {item}."],
            f"{a} is at the store. {b} asks {a} to buy {n} {item}.")


def delay(r, a, b):
    n, place = r.choice([5, 10, 15, 20, 30]), r.choice(PLACES)
    return ([f"{a}: I'll be {n} minutes late.", f"{b}: OK, I'll wait at the {place}."],
            f"{a} will be {n} minutes late. {b} will wait at the {place}.")


def congrats(r, a, b):
    exam = r.choice(EXAMS)
    return ([f"{a}: I passed the {exam}!", f"{b}: Congratulations, {a}!"],
            f"{a} passed the {exam}. {b} congratulates {a}.")


def question(r, a, b):
    thing, place = r.choice(THINGS), r.choice(PLACES)
    return ([f"{a}: Where is the {thing}?", f"{b}: It is in the {place}."],
            f"{a} asks where the {thing} is. {b} says it is in the {place}.")


TEMPLATES = [invitation, reschedule, shopping, delay, congrats, question]


def make_row(rng: random.Random, idx: int) -> tuple[str, str, str]:
    a, b = rng.sample(NAMES, 2)
    lines, summary = rng.choice(TEMPLATES)(rng, a, b)
    if rng.random() < 0.4:
        lines.append(f"{a}: Thanks!")
        summary += f" {a} thanks {b}."
    return f"{idx:05d}", "\n".join(lines), summary


def write(path: Path, n: int, seed: int) -> None:
    rng = random.Random(seed)
    seen, rows = set(), []
    while len(rows) < n:
        row = make_row(rng, len(rows))
        if row[1] in seen:
            continue
        seen.add(row[1])
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "dialogue", "summary"])
        w.writerows(rows)
    print(f"Wrote {n} synthetic dialogues -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/sample"))
    args = parser.parse_args()
    write(args.out_dir / "train.csv", 200, seed=1)
    write(args.out_dir / "validation.csv", 30, seed=2)
    write(args.out_dir / "test.csv", 40, seed=3)


if __name__ == "__main__":
    main()
