from __future__ import annotations

import shlex
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import List, Sequence


def lexical_recall(query: str, repo_path: str | Path, top_k: int = 20) -> List[str]:
    if shutil.which("rg") is None:
        return []

    terms = _tokenize(query)
    if not terms:
        return []

    repo_path = str(Path(repo_path).resolve())
    counts: Counter[str] = Counter()

    for term in terms:
        command = ["rg", "-l", term, repo_path]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return []

        if result.returncode not in (0, 1):
            continue
        for line in result.stdout.splitlines():
            counts[line.strip()] += 1

    return [path for path, _ in counts.most_common(top_k)]


def _tokenize(query: str) -> List[str]:
    try:
        terms = shlex.split(query)
    except ValueError:
        # Fall back when the query contains unmatched quotes from LLM output.
        terms = query.split()
    cleaned: List[str] = []
    for term in terms:
        term = term.strip().strip('"').strip("'")
        if len(term) >= 3:
            cleaned.append(term)
    return cleaned
