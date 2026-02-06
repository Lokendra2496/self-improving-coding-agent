from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


@dataclass
class Chunk:
    source_path: str
    chunk_index: int
    content: str
    metadata: Dict[str, str]


def iter_repo_files(
    repo_path: str | Path,
    include_exts: Sequence[str] | None = None,
    exclude_dirs: Sequence[str] | None = None,
) -> Iterator[Path]:
    repo_root = Path(repo_path).resolve()
    include_exts = include_exts or [".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json"]
    exclude_dirs = exclude_dirs or [".git", ".venv", "node_modules", "__pycache__", ".rag"]

    for path in repo_root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in exclude_dirs for part in path.parts):
            continue
        if path.suffix not in include_exts:
            continue
        yield path


def chunk_repo(
    repo_path: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    include_exts: Sequence[str] | None = None,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in iter_repo_files(repo_path, include_exts=include_exts):
        file_chunks = chunk_file(path, chunk_size, chunk_overlap)
        chunks.extend(file_chunks)
    return chunks


def chunk_file(path: Path, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".py":
        return _chunk_python(path, content, chunk_size, chunk_overlap)
    return _chunk_text(path, content, chunk_size, chunk_overlap)


def _chunk_python(path: Path, content: str, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _chunk_text(path, content, chunk_size, chunk_overlap)

    source_lines = content.splitlines()
    chunk_index = 0

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            continue
        snippet = "\n".join(source_lines[start - 1 : end])
        if len(snippet) > chunk_size * 2:
            sub_chunks = _chunk_text(path, snippet, chunk_size, chunk_overlap)
            for sub in sub_chunks:
                sub.metadata["symbol"] = node.name
                chunks.append(_with_index(sub, chunk_index))
                chunk_index += 1
        else:
            chunks.append(
                Chunk(
                    source_path=str(path),
                    chunk_index=chunk_index,
                    content=snippet,
                    metadata={"symbol": node.name},
                )
            )
            chunk_index += 1

    if not chunks:
        return _chunk_text(path, content, chunk_size, chunk_overlap)
    return chunks


def _chunk_text(path: Path, content: str, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    start = 0
    chunk_index = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        snippet = content[start:end]
        chunks.append(
            Chunk(
                source_path=str(path),
                chunk_index=chunk_index,
                content=snippet,
                metadata={},
            )
        )
        chunk_index += 1
        if end == len(content):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def _with_index(chunk: Chunk, index: int) -> Chunk:
    return Chunk(
        source_path=chunk.source_path,
        chunk_index=index,
        content=chunk.content,
        metadata=dict(chunk.metadata),
    )
