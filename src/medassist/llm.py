"""LLM / embedding factories (all local through Ollama)."""
from __future__ import annotations

import os
from functools import lru_cache

from langchain_ollama import ChatOllama, OllamaEmbeddings

from medassist.config import settings


@lru_cache(maxsize=None)
def get_llm(role: str = "default", temperature: float = 0.0) -> ChatOllama:
    """Return a ChatOllama for an agent role.

    A per-agent model can be set with e.g. ORCHESTRATOR_MODEL=llama3.2:3b,
    otherwise OLLAMA_MODEL is used.
    """
    model = os.getenv(f"{role.upper()}_MODEL", settings.default_model)
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_ctx=settings.num_ctx,
    )


def structured(role: str, schema, temperature: float = 0.0):
    """LLM constrained to return a Pydantic `schema` (JSON-schema decoding)."""
    return get_llm(role, temperature).with_structured_output(schema, method="json_schema")


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)
