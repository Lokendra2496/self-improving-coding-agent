from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Checkpoint:
    run_id: str
    state: Dict[str, Any]


class CheckpointStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save(self, run_id: str, state: Dict[str, Any]) -> None:
        payload = {"run_id": run_id, "state": state}
        self._path_for(run_id).write_text(json.dumps(payload, indent=2))

    def load(self, run_id: str) -> Optional[Checkpoint]:
        path = self._path_for(run_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return Checkpoint(run_id=data["run_id"], state=data["state"])

    def _path_for(self, run_id: str) -> Path:
        return self.root_dir / f"{run_id}.json"
