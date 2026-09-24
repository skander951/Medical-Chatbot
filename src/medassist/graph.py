"""LangGraph workflow.

START ─┬─ emergency keywords ───────────────────────────────► response ─► END
       ├─ profile incomplete ─► onboarding ─┬─ still missing ─► END (asks user)
       │                                    └─ complete ──┐
       └─ profile complete ──────────────────────────────►orchestrator
                                                           │
                    ┌────────── disease/both ──────────────┤
                    ▼                                      │
               disease_agent ── both ─► medicine_agent ─┐  │ medicine
                    │ disease-only                      │  ▼
                    └──────────────────────────────────►response ─► END
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from medassist.agents.disease import disease_node
from medassist.agents.medicine import medicine_node
from medassist.agents.onboarding import onboarding_node
from medassist.agents.orchestrator import orchestrator_node
from medassist.agents.response import response_node
from medassist.state import AgentState
from medassist.utils.messages import latest_user_text
from medassist.utils.safety import is_emergency


# ---- routing functions -------------------------------------------------
def route_entry(state: AgentState) -> str:
    if is_emergency(latest_user_text(state)):
        return "response_agent"
    return "orchestrator_agent" if state.get("onboarding_complete") else "onboarding_agent"


def after_onboarding(state: AgentState) -> str:
    return "orchestrator_agent" if state.get("onboarding_complete") else "end"


def after_orchestrator(state: AgentState) -> str:
    route = state.get("route", "direct")
    if route in ("disease", "both"):
        return "disease_agent"
    if route == "medicine":
        return "medicine_agent"
    return "response_agent"  # direct / emergency


def after_disease(state: AgentState) -> str:
    return "medicine_agent" if state.get("route") == "both" else "response_agent"


# ---- graph -------------------------------------------------------------
def build_graph(checkpointer=None):
    g = StateGraph(AgentState)

    g.add_node("onboarding_agent", onboarding_node)
    g.add_node("orchestrator_agent", orchestrator_node)
    g.add_node("disease_agent", disease_node)
    g.add_node("medicine_agent", medicine_node)
    g.add_node("response_agent", response_node)

    g.add_conditional_edges(
        START,
        route_entry,
        ["onboarding_agent", "orchestrator_agent", "response_agent"],
    )
    g.add_conditional_edges(
        "onboarding_agent",
        after_onboarding,
        {"orchestrator_agent": "orchestrator_agent", "end": END},
    )
    g.add_conditional_edges(
        "orchestrator_agent",
        after_orchestrator,
        ["disease_agent", "medicine_agent", "response_agent"],
    )
    g.add_conditional_edges("disease_agent", after_disease, ["medicine_agent", "response_agent"])
    g.add_edge("medicine_agent", "response_agent")
    g.add_edge("response_agent", END)

    # MemorySaver keeps the conversation + patient profile per thread_id.
    return g.compile(checkpointer=checkpointer or MemorySaver())
