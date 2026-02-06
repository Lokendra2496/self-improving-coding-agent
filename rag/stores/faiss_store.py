from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np

from rag.chunking import Chunk
from rag.types import RetrievalChunk


@dataclass
class FaissStore:
    index: faiss.Index
    metadata: List[Chunk]

    @classmethod
    def build(cls, embeddings: Sequence[Sequence[float]], chunks: List[Chunk]) -> "FaissStore":
        if not embeddings:
            raise ValueError("No embeddings provided for FAISS index")
        vectors = _normalize(np.array(embeddings, dtype="float32"))
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index=index, metadata=chunks)

    def save(self, index_path: Path, meta_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with meta_path.open("w", encoding="utf-8") as handle:
            for chunk in self.metadata:
                handle.write(
                    json.dumps(
                        {
                            "source_path": chunk.source_path,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                        }
                    )
                    + "\n"
                )

    @classmethod
    def load(cls, index_path: Path, meta_path: Path) -> "FaissStore":
        index = faiss.read_index(str(index_path))
        metadata = []
        with meta_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                data = json.loads(line)
                metadata.append(
                    Chunk(
                        source_path=data["source_path"],
                        chunk_index=data["chunk_index"],
                        content=data["content"],
                        metadata=data.get("metadata", {}),
                    )
                )
        return cls(index=index, metadata=metadata)

    def search(
        self, query_embedding: Sequence[float], top_k: int = 8
    ) -> List[RetrievalChunk]:
        vector = _normalize(np.array([query_embedding], dtype="float32"))
        scores, indices = self.index.search(vector, top_k)
        results: List[RetrievalChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            chunk = self.metadata[idx]
            results.append(
                RetrievalChunk(
                    source=chunk.source_path,
                    content=chunk.content,
                    score=float(score),
                )
            )
        return results


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
