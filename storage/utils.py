from __future__ import annotations

from typing import Any, Dict


def redact_config(config: Dict[str, Any]) -> Dict[str, Any]:
    redacted = {}
    for key, value in config.items():
        if any(token in key.lower() for token in ("key", "password", "secret", "token")):
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted
