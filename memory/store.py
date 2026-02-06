from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List

from agent.state import RunConfig
from memory.config import MemoryConfig, resolve_memory_config
from memory.mem0_client import Mem0Client


@dataclass
class MemoryItem:
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    def __init__(self, backend: "MemoryBackend") -> None:
        self._backend = backend

    @classmethod
    def from_config(cls, config: RunConfig, memory_config: MemoryConfig | None = None) -> "MemoryStore":
        memory_config = memory_config or resolve_memory_config(config)
        if memory_config.backend != "mem0":
            raise ValueError(f"Unsupported memory backend: {memory_config.backend}")
        backend: MemoryBackend = Mem0MemoryStore(memory_config)
        return cls(backend)

    def get_similar(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        return self._backend.get_similar(query, top_k=top_k)

    def store_episode(self, episode: Dict[str, Any]) -> None:
        self._backend.store_episode(episode)

    def store_fix_card(self, fix_card: Dict[str, Any]) -> None:
        self._backend.store_fix_card(fix_card)


class MemoryBackend:
    def get_similar(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        raise NotImplementedError

    def store_episode(self, episode: Dict[str, Any]) -> None:
        raise NotImplementedError

    def store_fix_card(self, fix_card: Dict[str, Any]) -> None:
        raise NotImplementedError


class LocalMemoryStore(MemoryBackend):
    def __init__(self, config: MemoryConfig) -> None:
        self._items: List[MemoryItem] = []
        self.config = config

    def get_similar(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        _ = query
        return self._items[:top_k]

    def store_episode(self, episode: Dict[str, Any]) -> None:
        summary = episode.get("summary", "stored episode")
        self._items.append(MemoryItem(summary=summary, metadata=episode))

    def store_fix_card(self, fix_card: Dict[str, Any]) -> None:
        summary = fix_card.get("summary", "stored fix card")
        self._items.append(MemoryItem(summary=summary, metadata=fix_card))


class Mem0MemoryStore(MemoryBackend):
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.client = Mem0Client(config)

    def get_similar(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        try:
            response = self.client.search(query, top_k=top_k)
        except Exception:
            if self.config.fail_fast:
                raise
            return []

        results = response.get("results", response) or []
        items: List[MemoryItem] = []
        for result in results:
            items.append(
                MemoryItem(
                    summary=_extract_summary(result),
                    metadata=result,
                )
            )
        return items

    def store_episode(self, episode: Dict[str, Any]) -> None:
        summary = episode.get("summary", "stored episode")
        metadata = _fit_metadata(episode)
        try:
            self.client.add(summary, metadata=metadata)
        except Exception:
            if self.config.fail_fast:
                raise

    def store_fix_card(self, fix_card: Dict[str, Any]) -> None:
        summary = fix_card.get("summary", "stored fix card")
        metadata = _fit_metadata(fix_card)
        try:
            self.client.add(summary, metadata=metadata)
        except Exception:
            if self.config.fail_fast:
                raise


def _extract_summary(result: Dict[str, Any]) -> str:
    for key in ("memory", "summary", "text", "content"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    if "metadata" in result and isinstance(result["metadata"], dict):
        meta_summary = result["metadata"].get("summary")
        if isinstance(meta_summary, str) and meta_summary:
            return meta_summary
    return str(result)


_MEM0_METADATA_LIMIT = 2000
_MAX_STRING_CHARS = 400
_MAX_LIST_ITEMS = 25
_MAX_DICT_ITEMS = 50


def _fit_metadata(metadata: Dict[str, Any], max_chars: int = _MEM0_METADATA_LIMIT) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        metadata = {"summary": str(metadata)}

    trimmed = _truncate_value(metadata)
    if _metadata_size(trimmed) <= max_chars:
        return trimmed

    priority_keys = [
        "summary",
        "root_cause",
        "fix",
        "verification",
        "error_signature",
        "files_changed",
        "goal",
        "run_id",
        "iteration",
        "status",
    ]
    reduced: Dict[str, Any] = {}
    for key in priority_keys:
        if key in trimmed:
            reduced[key] = trimmed[key]

    if not reduced:
        reduced = {"summary": _truncate_str(str(trimmed), _MAX_STRING_CHARS)}

    if _metadata_size(reduced) > max_chars:
        summary = str(reduced.get("summary", ""))
        reduced = {"summary": _truncate_str(summary, max(50, max_chars - 100))}

    reduced["_truncated"] = True
    if _metadata_size(reduced) > max_chars:
        reduced.pop("_truncated", None)
    return reduced


def _metadata_size(metadata: Dict[str, Any]) -> int:
    return len(json.dumps(metadata, ensure_ascii=True, separators=(",", ":")))


def _truncate_str(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...(truncated)"


def _truncate_value(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate_str(value, _MAX_STRING_CHARS)
    if isinstance(value, list):
        trimmed = []
        for item in value[:_MAX_LIST_ITEMS]:
            trimmed.append(_truncate_value(item))
        return trimmed
    if isinstance(value, dict):
        trimmed_dict: Dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= _MAX_DICT_ITEMS:
                break
            trimmed_dict[key] = _truncate_value(item)
        return trimmed_dict
    return value
