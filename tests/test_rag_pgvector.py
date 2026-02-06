import os
from pathlib import Path

import pytest

from agent.state import RunConfig
from rag.config import resolve_rag_config
from rag.index_repo import index_repo
from rag.retrieve_repo import retrieve


def test_pgvector_index_and_retrieve(tmp_path: Path, monkeypatch) -> None:
    dsn = os.getenv("RAG_PG_DSN")
    if not dsn:
        pytest.skip("RAG_PG_DSN not set")

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "sample.py").write_text("def add(a, b):\\n    return a + b\\n")

    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")

    config = RunConfig(
        goal="index",
        repo_path=str(repo_dir),
        rag_backend="pgvector",
        rag_pg_dsn=dsn,
        rag_index_dir=str(tmp_path / ".rag"),
    )
    rag_config = resolve_rag_config(config)
    index_repo(rag_config)

    results = retrieve("add function", rag_config)
    assert results
