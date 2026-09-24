"""Central configuration, read from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _path(env: str, default: str) -> Path:
    p = Path(os.getenv(env, default))
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    default_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

    top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    data_dir: Path = _path("DATA_DIR", "data")
    persist_dir: Path = _path("PERSIST_DIR", "vectorstore")


settings = Settings()

# knowledge base name -> (PDF folder name, Chroma collection name)
KNOWLEDGE_BASES = {
    "disease": ("disease_book", "disease_book"),
    "medicine": ("medicine_book", "medicine_book"),
}
