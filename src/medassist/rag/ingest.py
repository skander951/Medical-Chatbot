"""Build the two Chroma vector stores from the PDF books.

Usage:
    python -m medassist.rag.ingest            # both books
    python -m medassist.rag.ingest disease    # only the disease book
"""
from __future__ import annotations

import logging
import sys

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from medassist.config import KNOWLEDGE_BASES, settings
from medassist.llm import get_embeddings
from medassist.utils.timing import setup_logging

logger = logging.getLogger("medassist.ingest")


def _open_store(collection: str) -> Chroma:
    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.persist_dir),
    )


def build_collection(kind: str) -> int:
    folder, collection = KNOWLEDGE_BASES[kind]
    pdf_dir = settings.data_dir / folder
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDF found in %s. Add your %s book there first.", pdf_dir, kind)
        return 0

    docs = []
    for pdf in pdfs:
        logger.info("Loading %s", pdf.name)
        docs.extend(PyPDFLoader(str(pdf)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    logger.info("%s: %d pages -> %d chunks", kind, len(docs), len(chunks))

    # Rebuild from scratch so re-running the script never duplicates chunks.
    try:
        _open_store(collection).delete_collection()
    except Exception:  # collection did not exist yet
        pass
    store = _open_store(collection)
    for i in range(0, len(chunks), 64):
        store.add_documents(chunks[i : i + 64])
        logger.info("%s: embedded %d/%d", kind, min(i + 64, len(chunks)), len(chunks))
    return len(chunks)


def main() -> None:
    setup_logging()
    kinds = sys.argv[1:] or list(KNOWLEDGE_BASES)
    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        if kind not in KNOWLEDGE_BASES:
            raise SystemExit(f"Unknown knowledge base '{kind}'. Choose from {list(KNOWLEDGE_BASES)}")
        n = build_collection(kind)
        logger.info("Indexed %d chunks for '%s'", n, kind)


if __name__ == "__main__":
    main()
