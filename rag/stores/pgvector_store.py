from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Optional, Sequence

import psycopg
from psycopg.types.json import Json
from pgvector.psycopg import Vector, register_vector

from rag.chunking import Chunk
from rag.types import RetrievalChunk


@dataclass
class PGVectorStore:
    dsn: str
    embedding_dim: int

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS repo_chunks (
                    id UUID PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding VECTOR({self.embedding_dim}) NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS repo_chunks_embedding_idx
                ON repo_chunks USING ivfflat (embedding vector_cosine_ops)
                """
            )
            conn.commit()

    def upsert_chunks(
        self, embeddings: Sequence[Sequence[float]], chunks: Sequence[Chunk]
    ) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks size mismatch")

        paths = list({chunk.source_path for chunk in chunks})
        with self._connect() as conn:
            if paths:
                conn.execute("DELETE FROM repo_chunks WHERE source_path = ANY(%s)", (paths,))

            for embedding, chunk in zip(embeddings, chunks):
                conn.execute(
                    """
                    INSERT INTO repo_chunks (id, source_path, chunk_index, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        uuid.uuid4(),
                        chunk.source_path,
                        chunk.chunk_index,
                        chunk.content,
                        Json(chunk.metadata),
                        embedding,
                    ),
                )
            conn.commit()

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 8,
        filter_paths: Optional[Sequence[str]] = None,
    ) -> List[RetrievalChunk]:
        query_vector = Vector(list(query_embedding))
        with self._connect() as conn:
            if filter_paths:
                rows = conn.execute(
                    """
                    SELECT source_path, content, metadata, 1 - (embedding <=> %s) AS score
                    FROM repo_chunks
                    WHERE source_path = ANY(%s)
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (query_vector, list(filter_paths), query_vector, top_k),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT source_path, content, metadata, 1 - (embedding <=> %s) AS score
                    FROM repo_chunks
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (query_vector, query_vector, top_k),
                ).fetchall()

        results: List[RetrievalChunk] = []
        for row in rows:
            results.append(
                RetrievalChunk(
                    source=row[0],
                    content=row[1],
                    score=float(row[3]),
                )
            )
        return results

    def _connect(self):
        conn = psycopg.connect(self.dsn)
        register_vector(conn)
        return conn
