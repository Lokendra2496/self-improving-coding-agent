from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agent.state import RunConfig


@dataclass
class RAGConfig:
    repo_path: str
    backend: str
    top_k: int
    index_dir: str
    pg_dsn: Optional[str]
    chunk_size: int
    chunk_overlap: int
    lexical_top_k: int
    vector_top_k: int
    embedding_provider: str
    embedding_model: str
    embedding_batch_size: int
    embedding_dim: int

    @property
    def faiss_index_path(self) -> Path:
        return Path(self.index_dir) / "faiss.index"

    @property
    def faiss_meta_path(self) -> Path:
        return Path(self.index_dir) / "faiss_meta.jsonl"


def resolve_rag_config(config: RunConfig) -> RAGConfig:
    repo_path = config.repo_path
    backend = _pick_str(config.rag_backend, "RAG_BACKEND", "faiss")
    index_dir = _pick_str(config.rag_index_dir, "RAG_INDEX_DIR", ".rag")
    pg_dsn = _pick_optional_str(config.rag_pg_dsn, "RAG_PG_DSN")
    top_k = _pick_int(config.rag_top_k, "RAG_TOP_K", 8)
    chunk_size = _pick_int(config.rag_chunk_size, "RAG_CHUNK_SIZE", 2000)
    chunk_overlap = _pick_int(config.rag_chunk_overlap, "RAG_CHUNK_OVERLAP", 200)
    lexical_top_k = _pick_int(config.rag_lexical_top_k, "RAG_LEXICAL_TOP_K", 20)
    vector_top_k = _pick_int(config.rag_vector_top_k, "RAG_VECTOR_TOP_K", 12)
    embedding_provider = _pick_str(config.embedding_provider, "EMBEDDING_PROVIDER", "litellm")
    embedding_model = _pick_str(config.embedding_model, "EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_batch_size = _pick_int(config.embedding_batch_size, "EMBEDDING_BATCH_SIZE", 64)
    embedding_dim = _pick_int(config.embedding_dim, "EMBEDDING_DIM", 1536)
    if embedding_provider == "mock" and config.embedding_dim is None and os.getenv("EMBEDDING_DIM") is None:
        embedding_dim = 16

    resolved_index_dir = (
        str(Path(repo_path) / index_dir)
        if not Path(index_dir).is_absolute()
        else index_dir
    )

    return RAGConfig(
        repo_path=repo_path,
        backend=backend,
        top_k=top_k,
        index_dir=resolved_index_dir,
        pg_dsn=pg_dsn,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        lexical_top_k=lexical_top_k,
        vector_top_k=vector_top_k,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        embedding_dim=embedding_dim,
    )


def _pick_str(override: Optional[str], env_key: str, default: str) -> str:
    if override:
        return override
    return os.getenv(env_key, default)


def _pick_optional_str(override: Optional[str], env_key: str) -> Optional[str]:
    if override:
        return override
    return os.getenv(env_key)


def _pick_int(override: Optional[int], env_key: str, default: int) -> int:
    if override is not None:
        return override
    raw = os.getenv(env_key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
