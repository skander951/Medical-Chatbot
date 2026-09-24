"""Disease agent: RAG over the disease reference book."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from medassist import prompts
from medassist.llm import get_llm
from medassist.rag.retriever import format_context, retrieve, source_labels
from medassist.state import AgentState
from medassist.utils.messages import latest_user_text
from medassist.utils.timing import timed


@timed("disease_agent")
def disease_node(state: AgentState) -> dict:
    profile = state.get("profile") or {}
    query = f"{profile.get('symptoms', '')} {latest_user_text(state)}".strip()

    docs = retrieve("disease", query)
    if not docs:
        return {"disease_findings": prompts.NO_INFO, "disease_sources": []}

    findings = (
        get_llm("disease", 0.1)
        .invoke(
            [
                SystemMessage(
                    content=prompts.DISEASE.format(
                        profile=json.dumps(profile), context=format_context(docs)
                    )
                ),
                HumanMessage(content=f"Symptoms / question: {query}"),
            ]
        )
        .content
    )
    return {"disease_findings": findings, "disease_sources": source_labels(docs)}
