"""Retrieval helpers over the two vector stores."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from medassist.config import KNOWLEDGE_BASES, settings
from medassist.llm import get_embeddings


@lru_cache(maxsize=None)
def _store(kind: str) -> Chroma:
    if kind not in KNOWLEDGE_BASES:
        raise ValueError(f"Unknown knowledge base: {kind}")
    if not settings.persist_dir.exists() or not any(settings.persist_dir.iterdir()):
        raise RuntimeError(
            "Vector store is empty. Put your PDFs in data/ and run: python -m medassist.rag.ingest"
        )
    return Chroma(
        collection_name=KNOWLEDGE_BASES[kind][1],
        embedding_function=get_embeddings(),
        persist_directory=str(settings.persist_dir),
    )


def retrieve(kind: str, query: str, k: int | None = None) -> list[Document]:
    """MMR search: relevant AND diverse chunks."""
    k = k or settings.top_k
    return _store(kind).max_marginal_relevance_search(query, k=k, fetch_k=max(20, k * 4))


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] ({source_label(d)})\n{d.page_content}" for i, d in enumerate(docs, 1))


def source_label(doc: Document) -> str:
    src = Path(str(doc.metadata.get("source", "unknown"))).name
    page = doc.metadata.get("page")
    return f"{src} p.{int(page) + 1}" if page is not None else src


def source_labels(docs: list[Document]) -> list[str]:
    seen, out = set(), []
    for d in docs:
        label = source_label(d)
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out
