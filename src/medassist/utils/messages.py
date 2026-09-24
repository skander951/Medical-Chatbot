"""Small helpers to read the chat history."""
from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage


def latest_user_text(state: dict) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def format_history(messages: list[BaseMessage], n: int = 6) -> str:
    lines = []
    for m in (messages or [])[-n:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines) or "(no previous messages)"
