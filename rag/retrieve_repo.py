from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from rag.config import RAGConfig
from rag.embeddings import get_embedder, resolve_embedding_config_for_rag
from rag.index_meta import is_index_stale, read_index_meta
from rag.lexical import lexical_recall
from rag.stores.faiss_store import FaissStore
from rag.stores.pgvector_store import PGVectorStore
from rag.types import RetrievalChunk
from observability.tracing import span

def retrieve(query: str, config: RAGConfig) -> List[RetrievalChunk]:
    with span(
        "rag.retrieve",
        {
            "rag.backend": config.backend,
            "rag.top_k": config.top_k,
            "rag.lexical_top_k": config.lexical_top_k,
            "rag.vector_top_k": config.vector_top_k,
        },
    ):
        lexical_paths = lexical_recall(query, config.repo_path, config.lexical_top_k)
        embedder = get_embedder(resolve_embedding_config_for_rag(config))
        embedding_list = embedder.embed_texts([query])
        if not embedding_list:
            return []
        query_embedding = embedding_list[0]

        vector_hits: List[RetrievalChunk] = []
        if config.backend == "faiss":
            index_path = Path(config.faiss_index_path)
            meta_path = Path(config.faiss_meta_path)
            if not index_path.exists() or not meta_path.exists():
                raise FileNotFoundError("FAISS index files not found. Run indexing first.")
            index_meta = read_index_meta(config.index_dir)
            if index_meta is None:
                raise FileNotFoundError("FAISS index metadata missing. Run indexing first.")
            if is_index_stale(config, index_meta):
                raise ValueError("FAISS index is stale. Re-run indexing for this repo.")
            store = FaissStore.load(index_path, meta_path)
            vector_hits = store.search(query_embedding, config.vector_top_k)
            if lexical_paths:
                filtered = [hit for hit in vector_hits if hit.source in lexical_paths]
                if filtered:
                    vector_hits = filtered
        elif config.backend == "pgvector":
            if not config.pg_dsn:
                raise ValueError("RAG_PG_DSN is required for pgvector backend")
            store = PGVectorStore(dsn=config.pg_dsn, embedding_dim=config.embedding_dim)
            vector_hits = store.search(query_embedding, config.vector_top_k, filter_paths=lexical_paths or None)
        else:
            raise ValueError(f"Unsupported RAG backend: {config.backend}")

        return vector_hits[: config.top_k]
