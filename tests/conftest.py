"""Shared pytest fixtures.

These tests stub out the LLM and retriever so the LangGraph wiring, routing
logic and safety rules can be checked instantly, without Ollama or a real
vector store. Treat them as structural / integration tests for the graph,
not as an evaluation of answer quality (that's what LangSmith + benchmark.py
are for, against a running local model).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from langchain_core.documents import Document

FAKE_PROFILE = {
    "age": 29,
    "sex": "female",
    "symptoms": "fever, dry cough, body aches",
    "allergies": "none",
    "current_medications": "none",
}


@pytest.fixture
def complete_profile():
    return dict(FAKE_PROFILE)


@pytest.fixture
def fake_docs():
    return [
        Document(page_content="Influenza: fever, cough, body aches.", metadata={"source": "disease_book.pdf", "page": 3}),
        Document(page_content="Paracetamol reduces fever and pain.", metadata={"source": "medicine_book.pdf", "page": 7}),
    ]
