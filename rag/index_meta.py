from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from rag.chunking import iter_repo_files
from rag.config import RAGConfig


@dataclass
class IndexMeta:
    repo_path: str
    indexed_at: str
    indexed_at_epoch: float
    latest_mtime: float
    file_count: int
    backend: str
    embedding_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    chunk_count: int


def meta_path(index_dir: str | Path) -> Path:
    return Path(index_dir) / "index_meta.json"


def write_index_meta(rag_config: RAGConfig, chunk_count: int) -> IndexMeta:
    latest_mtime, file_count = _repo_stats(rag_config.repo_path)
    indexed_at_epoch = datetime.now(timezone.utc).timestamp()
    indexed_at = datetime.now(timezone.utc).isoformat()
    meta = IndexMeta(
        repo_path=str(Path(rag_config.repo_path).resolve()),
        indexed_at=indexed_at,
        indexed_at_epoch=indexed_at_epoch,
        latest_mtime=latest_mtime,
        file_count=file_count,
        backend=rag_config.backend,
        embedding_model=rag_config.embedding_model,
        embedding_dim=rag_config.embedding_dim,
        chunk_size=rag_config.chunk_size,
        chunk_overlap=rag_config.chunk_overlap,
        chunk_count=chunk_count,
    )
    path = meta_path(rag_config.index_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta.__dict__, indent=2), encoding="utf-8")
    return meta


def read_index_meta(index_dir: str | Path) -> IndexMeta | None:
    path = meta_path(index_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    try:
        return IndexMeta(**data)
    except TypeError:
        return None


def is_index_stale(rag_config: RAGConfig, meta: IndexMeta) -> bool:
    repo_path = str(Path(rag_config.repo_path).resolve())
    if meta.repo_path != repo_path:
        return True
    if meta.backend != rag_config.backend:
        return True
    if meta.embedding_model != rag_config.embedding_model:
        return True
    if meta.embedding_dim != rag_config.embedding_dim:
        return True
    if meta.chunk_size != rag_config.chunk_size or meta.chunk_overlap != rag_config.chunk_overlap:
        return True

    latest_mtime, file_count = _repo_stats(repo_path)
    if file_count != meta.file_count:
        return True
    if latest_mtime > meta.latest_mtime:
        return True
    return False


def _repo_stats(repo_path: str | Path) -> Tuple[float, int]:
    latest_mtime = 0.0
    file_count = 0
    for path in iter_repo_files(repo_path):
        try:
            stat = path.stat()
        except OSError:
            continue
        file_count += 1
        if stat.st_mtime > latest_mtime:
            latest_mtime = stat.st_mtime
    return latest_mtime, file_count
