from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from agent.state import RunConfig


@dataclass
class MemoryConfig:
    backend: str
    api_key: Optional[str]
    host: Optional[str]
    org_id: Optional[str]
    project_id: Optional[str]
    user_id: Optional[str]
    agent_id: Optional[str]
    app_id: Optional[str]
    top_k: int
    fail_fast: bool


def resolve_memory_config(config: RunConfig) -> MemoryConfig:
    backend = _pick_str(config.memory_backend, "MEMORY_BACKEND", "mem0")
    api_key = _pick_optional_str(config.mem0_api_key, "MEM0_API_KEY")
    host = _pick_optional_str(config.mem0_host, "MEM0_HOST")
    org_id = _pick_optional_str(config.mem0_org_id, "MEM0_ORG_ID")
    project_id = _pick_optional_str(config.mem0_project_id, "MEM0_PROJECT_ID")
    user_id = _pick_optional_str(config.mem0_user_id, "MEM0_USER_ID")
    agent_id = _pick_optional_str(config.mem0_agent_id, "MEM0_AGENT_ID")
    app_id = _pick_optional_str(config.mem0_app_id, "MEM0_APP_ID")
    top_k = _pick_int(config.memory_top_k, "MEMORY_TOP_K", 5)
    fail_fast = _pick_bool(config.memory_fail_fast, "MEMORY_FAIL_FAST", True)

    return MemoryConfig(
        backend=backend,
        api_key=api_key,
        host=host,
        org_id=org_id,
        project_id=project_id,
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        top_k=top_k,
        fail_fast=fail_fast,
    )


def _pick_str(override: Optional[str], env_key: str, default: str) -> str:
    if override:
        return override
    return os.getenv(env_key, default)


def _pick_optional_str(override: Optional[str], env_key: str) -> Optional[str]:
    if override:
        return override
    return os.getenv(env_key)


def _pick_int(override: Optional[int], env_key: str, default: int) -> int:
    if override is not None:
        return override
    raw = os.getenv(env_key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _pick_bool(override: Optional[bool], env_key: str, default: bool) -> bool:
    if override is not None:
        return override
    raw = os.getenv(env_key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
