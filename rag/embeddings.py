from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import litellm

from agent.state import RunConfig
from llm.config import resolve_llm_config
from rag.config import RAGConfig


@dataclass
class EmbeddingConfig:
    provider: str
    model: str
    base_url: Optional[str]
    api_key: Optional[str]
    batch_size: int
    embedding_dim: int


class Embedder(Protocol):
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...


def resolve_embedding_config(config: RunConfig) -> EmbeddingConfig:
    llm_config = resolve_llm_config(config)
    provider = config.embedding_provider or "litellm"
    model = config.embedding_model or "text-embedding-3-small"
    batch_size = config.embedding_batch_size or 64
    embedding_dim = config.embedding_dim or 1536
    if provider == "mock" and config.embedding_dim is None:
        embedding_dim = 16

    return EmbeddingConfig(
        provider=provider,
        model=model,
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
    )


def resolve_embedding_config_for_rag(
    config: RAGConfig,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider=config.embedding_provider,
        model=config.embedding_model,
        base_url=base_url or os.getenv("LITELLM_BASE_URL"),
        api_key=api_key or os.getenv("LITELLM_API_KEY"),
        batch_size=config.embedding_batch_size,
        embedding_dim=config.embedding_dim,
    )


def get_embedder(config: EmbeddingConfig) -> Embedder:
    if config.provider == "mock":
        return MockEmbedder(config.embedding_dim)
    return LiteLLMEmbedder(config)


class LiteLLMEmbedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        kwargs: Dict[str, Any] = {"model": self.config.model, "input": texts}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        response = litellm.embedding(**kwargs)
        data = _get(response, "data") or []
        embeddings: List[List[float]] = []
        for item in data:
            embedding = _get(item, "embedding")
            if embedding is None:
                continue
            vector = list(embedding)
            if len(vector) != self.config.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.config.embedding_dim}, got {len(vector)}"
                )
            embeddings.append(vector)
        return embeddings


class MockEmbedder:
    def __init__(self, dims: int = 16) -> None:
        self.dims = dims

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [_hash_to_vector(text, self.dims) for text in texts]


def _hash_to_vector(text: str, dims: int = 16) -> List[float]:
    vec: List[float] = []
    counter = 0
    while len(vec) < dims:
        digest = hashlib.sha256(f"{text}:{counter}".encode("utf-8")).digest()
        for byte in digest:
            vec.append(byte / 255.0)
            if len(vec) >= dims:
                break
        counter += 1
    return vec


def _get(obj: Any, key: str) -> Optional[Any]:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
