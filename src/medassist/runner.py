"""Run one conversation turn and collect answer, route and per-node latencies."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage


@dataclass
class TurnResult:
    answer: str = ""
    route: str | None = None
    timings: dict = field(default_factory=dict)
    total_seconds: float = 0.0


def run_turn(graph, user_text: str, thread_id: str, extra_state: dict | None = None,
             tags: list[str] | None = None) -> TurnResult:
    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": "medical_assistant_turn",   # shows up as the trace name in LangSmith
        "tags": tags or ["medassist"],
        "metadata": {"thread_id": thread_id},
    }
    inputs = {"messages": [HumanMessage(content=user_text)], **(extra_state or {})}

    result = TurnResult()
    start = time.perf_counter()
    # stream_mode="updates" yields only the nodes that ran during THIS turn
    for chunk in graph.stream(inputs, config, stream_mode="updates"):
        for update in chunk.values():
            if not update:
                continue
            result.timings.update(update.get("timings", {}))
            if "route" in update:
                result.route = update["route"]
            for msg in update.get("messages", []):
                if isinstance(msg, AIMessage):
                    result.answer = str(msg.content)
    result.total_seconds = round(time.perf_counter() - start, 2)
    return result
