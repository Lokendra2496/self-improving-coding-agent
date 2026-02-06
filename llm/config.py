from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from agent.state import RunConfig


@dataclass
class LLMConfig:
    enabled: bool
    fast_model: str
    strong_model: str
    base_url: Optional[str]
    api_key: Optional[str]
    fast_temperature: float
    strong_temperature: float
    timeout_s: Optional[int]


def resolve_llm_config(config: RunConfig) -> LLMConfig:
    enabled = _pick_bool(config.llm_enabled, "LLM_ENABLED", default=False)
    fast_model = _pick_str(config.fast_model, "FAST_MODEL", "gpt-4o-mini")
    strong_model = _pick_str(config.strong_model, "STRONG_MODEL", "gpt-4o")
    base_url = _pick_optional_str(config.litellm_base_url, "LITELLM_BASE_URL")
    api_key = _pick_optional_str(config.litellm_api_key, "LITELLM_API_KEY")
    fast_temp = _pick_float(config.fast_temperature, "FAST_TEMPERATURE", 0.2)
    strong_temp = _pick_float(config.strong_temperature, "STRONG_TEMPERATURE", 0.2)
    timeout_s = _pick_optional_int(config.litellm_timeout_s, "LITELLM_TIMEOUT_S")

    return LLMConfig(
        enabled=enabled,
        fast_model=fast_model,
        strong_model=strong_model,
        base_url=base_url,
        api_key=api_key,
        fast_temperature=fast_temp,
        strong_temperature=strong_temp,
        timeout_s=timeout_s,
    )


def _pick_str(override: Optional[str], env_key: str, default: str) -> str:
    if override:
        return override
    return os.getenv(env_key, default)


def _pick_optional_str(override: Optional[str], env_key: str) -> Optional[str]:
    if override:
        return override
    return os.getenv(env_key)


def _pick_bool(override: Optional[bool], env_key: str, default: bool) -> bool:
    if override is not None:
        return override
    raw = os.getenv(env_key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _pick_float(override: Optional[float], env_key: str, default: float) -> float:
    if override is not None:
        return override
    raw = os.getenv(env_key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _pick_optional_int(override: Optional[int], env_key: str) -> Optional[int]:
    if override is not None:
        return override
    raw = os.getenv(env_key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None
