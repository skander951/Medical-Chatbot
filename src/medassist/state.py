"""Shared state passed between the LangGraph nodes."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    # onboarding
    profile: dict
    onboarding_complete: bool

    # orchestrator
    route: str
    emergency: bool

    # specialist agents
    disease_findings: str
    disease_sources: list[str]
    medicine_findings: str
    medicine_sources: list[str]

    # output / monitoring
    final_answer: str
    timings: dict
