"""Structural tests for the graph's routing functions (no LLM calls)."""
from langchain_core.messages import HumanMessage

from medassist.graph import after_disease, after_onboarding, after_orchestrator, route_entry


def test_entry_routes_to_onboarding_when_incomplete():
    state = {"messages": [HumanMessage(content="hi")], "onboarding_complete": False}
    assert route_entry(state) == "onboarding_agent"


def test_entry_routes_to_orchestrator_when_complete():
    state = {"messages": [HumanMessage(content="I have a fever")], "onboarding_complete": True}
    assert route_entry(state) == "orchestrator_agent"


def test_entry_routes_to_response_on_emergency_keywords():
    state = {"messages": [HumanMessage(content="I have severe chest pain")], "onboarding_complete": False}
    assert route_entry(state) == "response_agent"


def test_after_onboarding_waits_when_incomplete():
    assert after_onboarding({"onboarding_complete": False}) == "end"


def test_after_onboarding_proceeds_when_complete():
    assert after_onboarding({"onboarding_complete": True}) == "orchestrator_agent"


def test_after_orchestrator_routes_by_decision():
    assert after_orchestrator({"route": "disease"}) == "disease_agent"
    assert after_orchestrator({"route": "medicine"}) == "medicine_agent"
    assert after_orchestrator({"route": "both"}) == "disease_agent"
    assert after_orchestrator({"route": "direct"}) == "response_agent"
    assert after_orchestrator({"route": "emergency"}) == "response_agent"


def test_after_disease_continues_to_medicine_only_when_both():
    assert after_disease({"route": "both"}) == "medicine_agent"
    assert after_disease({"route": "disease"}) == "response_agent"
