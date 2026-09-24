"""Orchestrator agent: decides which specialists must run."""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from medassist import prompts
from medassist.llm import structured
from medassist.schemas import RouteDecision
from medassist.state import AgentState
from medassist.utils.messages import format_history, latest_user_text
from medassist.utils.safety import is_emergency
from medassist.utils.timing import timed

logger = logging.getLogger("medassist.orchestrator")


@timed("orchestrator_agent")
def orchestrator_node(state: AgentState) -> dict:
    text = latest_user_text(state)
    profile = state.get("profile") or {}

    if is_emergency(text):  # rule-based check always wins over the LLM
        route = "emergency"
    else:
        try:
            decision = structured("orchestrator", RouteDecision).invoke(
                [
                    SystemMessage(
                        content=prompts.ORCHESTRATOR.format(
                            profile=json.dumps(profile),
                            history=format_history(state.get("messages", [])),
                        )
                    ),
                    HumanMessage(content=text),
                ]
            )
            route = decision.route
            logger.info("route=%s (%s)", route, decision.reasoning)
        except Exception:
            logger.exception("Routing failed, falling back to 'both'")
            route = "both"

    # reset per-turn fields so nothing stale leaks into this turn's answer
    return {
        "route": route,
        "emergency": route == "emergency",
        "disease_findings": "",
        "disease_sources": [],
        "medicine_findings": "",
        "medicine_sources": [],
    }
