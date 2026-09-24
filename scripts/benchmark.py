"""Benchmark response time and routing over a small set of test questions.

Complements LangSmith (which traces every run in detail): this script gives a
quick local CSV/console summary of latency per agent, useful for the README
and for comparing different local models.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --questions data/eval_questions.txt
"""
from __future__ import annotations

import argparse
import csv
import sys
import uuid
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medassist.graph import build_graph
from medassist.runner import run_turn
from medassist.utils.timing import setup_logging

DEFAULT_QUESTIONS = [
    "I am a 29 year old woman, I have a fever, a dry cough and body aches since yesterday.",
    "I have no allergies and I don't take any medication currently.",
    "What medicine could help with my symptoms and what are the side effects?",
    "What is ibuprofen usually used for?",
    "Hello, thank you for your help!",
]


def load_questions(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_QUESTIONS
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=None, help="Path to a .txt file, one question per line")
    parser.add_argument("--out", default="results/benchmark.csv")
    args = parser.parse_args()

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    questions = load_questions(args.questions)

    rows = []
    for q in questions:
        result = run_turn(graph, q, thread_id, tags=["benchmark"])
        rows.append(
            {
                "question": q,
                "route": result.route,
                "total_seconds": result.total_seconds,
                **{f"t_{k}": v for k, v in result.timings.items()},
            }
        )
        print(f"[{result.total_seconds:>5.2f}s | {str(result.route):9s}] {q}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    avg = mean(r["total_seconds"] for r in rows)
    print(f"\nAverage total latency: {avg:.2f}s over {len(rows)} turns")
    print(f"Saved details to {out_path}")
    print("Full traces (per-node latency, prompts, retrieved chunks) are in LangSmith.")


if __name__ == "__main__":
    main()
