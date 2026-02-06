from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalChunk:
    source: str
    content: str
    score: float
