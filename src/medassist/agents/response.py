"""Response agent: writes the final user-facing answer."""
from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from medassist import prompts
from medassist.llm import get_llm
from medassist.state import AgentState
from medassist.utils.messages import format_history, latest_user_text
from medassist.utils.safety import DISCLAIMER, EMERGENCY_MESSAGE, is_emergency
from medassist.utils.timing import timed

logger = logging.getLogger("medassist.response")


@timed("response_agent")
def response_node(state: AgentState) -> dict:
    text = latest_user_text(state)

    if state.get("emergency") or is_emergency(text):
        answer = EMERGENCY_MESSAGE  # fixed text: no LLM on emergencies
    else:
        system = prompts.RESPONSE.format(
            profile=json.dumps(state.get("profile") or {}),
            history=format_history(state.get("messages", [])),
            disease=state.get("disease_findings") or "(not consulted)",
            medicine=state.get("medicine_findings") or "(not consulted)",
        )
        try:
            body = get_llm("response", 0.2).invoke(
                [SystemMessage(content=system), HumanMessage(content=text)]
            ).content
        except Exception:
            logger.exception("Response generation failed")
            body = "Sorry, I could not generate an answer right now. Please try again."

        sources = sorted(
            set((state.get("disease_sources") or []) + (state.get("medicine_sources") or []))
        )
        parts = [str(body).strip()]
        if sources:
            parts.append("**Sources:** " + "; ".join(sources))
        parts.append(DISCLAIMER)
        answer = "\n\n".join(parts)

    return {"messages": [AIMessage(content=answer)], "final_answer": answer}
