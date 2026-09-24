"""Medicine agent: RAG over the medicine reference book."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from medassist import prompts
from medassist.llm import get_llm
from medassist.rag.retriever import format_context, retrieve, source_labels
from medassist.state import AgentState
from medassist.utils.messages import latest_user_text
from medassist.utils.timing import timed


@timed("medicine_agent")
def medicine_node(state: AgentState) -> dict:
    profile = state.get("profile") or {}
    text = latest_user_text(state)
    disease = (state.get("disease_findings") or "").strip()

    # enrich the search query with the conditions found by the disease agent
    query = f"{text} {disease[:300]}".strip()

    docs = retrieve("medicine", query)
    if not docs:
        return {"medicine_findings": prompts.NO_INFO, "medicine_sources": []}

    findings = (
        get_llm("medicine", 0.1)
        .invoke(
            [
                SystemMessage(
                    content=prompts.MEDICINE.format(
                        profile=json.dumps(profile),
                        disease=disease or "(none identified)",
                        context=format_context(docs),
                    )
                ),
                HumanMessage(content=text),
            ]
        )
        .content
    )
    return {"medicine_findings": findings, "medicine_sources": source_labels(docs)}
