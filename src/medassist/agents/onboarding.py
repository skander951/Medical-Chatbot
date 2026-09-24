"""Onboarding agent: collects the patient profile over one or more turns."""
from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from medassist import prompts
from medassist.llm import get_llm, structured
from medassist.schemas import ProfileUpdate
from medassist.state import AgentState
from medassist.utils.messages import format_history, latest_user_text
from medassist.utils.profile import FIELD_LABELS, merge_profile, missing_fields
from medassist.utils.timing import timed

logger = logging.getLogger("medassist.onboarding")


@timed("onboarding_agent")
def onboarding_node(state: AgentState) -> dict:
    profile = dict(state.get("profile") or {})
    text = latest_user_text(state)

    # 1) extract whatever the user just told us
    try:
        update = structured("onboarding", ProfileUpdate).invoke(
            [
                SystemMessage(content=prompts.ONBOARDING_EXTRACT.format(profile=json.dumps(profile))),
                HumanMessage(
                    content=f"Recent conversation:\n{format_history(state.get('messages', []))}\n\n"
                    f"Latest user message: {text}"
                ),
            ]
        )
        profile = merge_profile(profile, update.model_dump())
    except Exception:
        logger.exception("Profile extraction failed")

    missing = missing_fields(profile)
    if not missing:
        return {"profile": profile, "onboarding_complete": True}

    # 2) ask for what is still missing
    labels = ", ".join(FIELD_LABELS[m] for m in missing)
    try:
        question = (
            get_llm("onboarding", 0.3)
            .invoke([SystemMessage(content=prompts.ONBOARDING_QUESTION.format(missing=labels))])
            .content
        )
    except Exception:
        question = f"To help you safely, could you tell me your {labels}?"

    return {
        "profile": profile,
        "onboarding_complete": False,
        "messages": [AIMessage(content=question)],
    }
