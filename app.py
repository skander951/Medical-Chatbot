"""Streamlit chat UI.

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from medassist.graph import build_graph
from medassist.runner import run_turn

st.set_page_config(page_title="MedAssist", page_icon="🩺")
st.title("🩺 MedAssist — Multi-Agent Medical RAG Assistant")
st.caption(
    "Onboarding → Orchestrator → Disease/Medicine RAG agents → Response agent · "
    "runs on local Ollama models · educational use only, not medical advice."
)


@st.cache_resource
def get_graph():
    return build_graph()


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

if prompt := st.chat_input("Describe your symptoms or ask about a medicine..."):
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_turn(get_graph(), prompt, st.session_state.thread_id)
        st.markdown(result.answer)
        if result.route:
            st.caption(f"route: `{result.route}` · total: {result.total_seconds}s · {result.timings}")

    st.session_state.history.append(("assistant", result.answer))

with st.sidebar:
    st.subheader("Session")
    st.code(st.session_state.thread_id, language=None)
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.markdown(
        "**Agents:** onboarding · orchestrator · disease (RAG) · medicine (RAG) · response\n\n"
        "**Stack:** LangGraph · LangChain · Ollama (local LLMs) · Chroma · LangSmith"
    )
