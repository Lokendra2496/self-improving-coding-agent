from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from rag.chunking import Chunk, chunk_repo
from rag.config import RAGConfig, resolve_rag_config
from rag.embeddings import get_embedder, resolve_embedding_config_for_rag
from rag.index_meta import write_index_meta
from rag.stores.faiss_store import FaissStore
from rag.stores.pgvector_store import PGVectorStore
from observability.tracing import span


@dataclass
class IndexStats:
    backend: str
    chunk_count: int
    embedding_dim: int


def index_repo(rag_config: RAGConfig) -> IndexStats:
    with span(
        "rag.index",
        {
            "rag.backend": rag_config.backend,
            "rag.chunk_size": rag_config.chunk_size,
            "rag.chunk_overlap": rag_config.chunk_overlap,
        },
    ):
        chunks = chunk_repo(
            rag_config.repo_path,
            chunk_size=rag_config.chunk_size,
            chunk_overlap=rag_config.chunk_overlap,
        )
        if not chunks:
            raise ValueError("No chunks found to index.")
        embedder = get_embedder(resolve_embedding_config_for_rag(rag_config))

        if rag_config.backend == "faiss":
            embeddings = _embed_all(embedder, chunks, rag_config.embedding_batch_size)
            if len(embeddings) != len(chunks):
                raise ValueError("Embedding count mismatch while building FAISS index")
            store = FaissStore.build(embeddings, chunks)
            store.save(rag_config.faiss_index_path, rag_config.faiss_meta_path)
        elif rag_config.backend == "pgvector":
            if not rag_config.pg_dsn:
                raise ValueError("RAG_PG_DSN is required for pgvector backend")
            store = PGVectorStore(dsn=rag_config.pg_dsn, embedding_dim=rag_config.embedding_dim)
            store.initialize()
            _upsert_pgvector(store, embedder, chunks, rag_config.embedding_batch_size)
        else:
            raise ValueError(f"Unsupported RAG backend: {rag_config.backend}")

        stats = IndexStats(
            backend=rag_config.backend,
            chunk_count=len(chunks),
            embedding_dim=rag_config.embedding_dim,
        )
        write_index_meta(rag_config, chunk_count=len(chunks))
        return stats


def _embed_all(embedder, chunks: List[Chunk], batch_size: int) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for batch in _batch(chunks, batch_size):
        texts = [chunk.content for chunk in batch]
        embeddings.extend(embedder.embed_texts(texts))
    return embeddings


def _upsert_pgvector(store: PGVectorStore, embedder, chunks: List[Chunk], batch_size: int) -> None:
    for batch in _batch(chunks, batch_size):
        texts = [chunk.content for chunk in batch]
        embeddings = embedder.embed_texts(texts)
        store.upsert_chunks(embeddings, batch)


def _batch(items: List[Chunk], batch_size: int) -> Iterable[List[Chunk]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def index_repo_from_env() -> IndexStats:
    return index_repo(resolve_rag_config(_dummy_run_config()))


def _dummy_run_config():
    from agent.state import RunConfig

    return RunConfig(goal="index_repo")
