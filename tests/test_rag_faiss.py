from pathlib import Path

from agent.state import RunConfig
from rag.config import resolve_rag_config
from rag.index_repo import index_repo
from rag.retrieve_repo import retrieve


def test_faiss_index_and_retrieve(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "sample.py").write_text("def add(a, b):\\n    return a + b\\n")

    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_DIM", "16")

    config = RunConfig(
        goal="index",
        repo_path=str(repo_dir),
        rag_backend="faiss",
        rag_index_dir=str(tmp_path / ".rag"),
    )
    rag_config = resolve_rag_config(config)
    index_repo(rag_config)

    results = retrieve("add function", rag_config)
    assert results
