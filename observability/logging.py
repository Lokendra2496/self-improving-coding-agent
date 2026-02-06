from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("agent")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, fields: Dict[str, Any] | None = None) -> None:
    payload = {"event": event, "ts": _utc_now_iso()}
    if fields:
        payload.update(fields)
    logger.info(json.dumps(payload, sort_keys=True))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
