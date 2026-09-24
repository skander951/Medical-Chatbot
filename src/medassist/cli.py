"""Simple terminal chat loop.

Usage:
    python -m medassist.cli
"""
from __future__ import annotations

import uuid

from medassist.graph import build_graph
from medassist.runner import run_turn
from medassist.utils.timing import setup_logging


def main() -> None:
    setup_logging()
    graph = build_graph()
    thread_id = str(uuid.uuid4())

    print("MedAssist (multi-agent medical RAG assistant) — type 'exit' to quit\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        result = run_turn(graph, user_text, thread_id)
        print(f"\nAssistant: {result.answer}\n")
        if result.route:
            print(f"[route={result.route} | total={result.total_seconds}s | {result.timings}]\n")


if __name__ == "__main__":
    main()
