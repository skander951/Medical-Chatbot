"""response_node must short-circuit to the fixed emergency text, with no LLM call."""
from langchain_core.messages import AIMessage, HumanMessage

from medassist.agents.response import response_node
from medassist.utils.safety import EMERGENCY_MESSAGE


def test_emergency_short_circuits_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM should not be called on an emergency message")

    monkeypatch.setattr("medassist.agents.response.get_llm", boom)

    state = {
        "messages": [HumanMessage(content="I have severe chest pain")],
        "emergency": True,
    }
    out = response_node(state)
    ai_messages = [m for m in out["messages"] if isinstance(m, AIMessage)]
    assert ai_messages and ai_messages[0].content == EMERGENCY_MESSAGE
    assert out["final_answer"] == EMERGENCY_MESSAGE
