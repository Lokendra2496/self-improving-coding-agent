from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from agent.state import RunConfig


@dataclass
class GraphConfig:
    backend: str
    uri: Optional[str]
    user: Optional[str]
    password: Optional[str]
    database: Optional[str]
    fail_fast: bool


def resolve_graph_config(config: RunConfig) -> GraphConfig:
    backend = _pick_str(config.graph_backend, "GRAPH_BACKEND", "neo4j")
    uri = _pick_optional_str(config.neo4j_uri, "NEO4J_URI")
    user = _pick_optional_str(config.neo4j_user, "NEO4J_USER")
    password = _pick_optional_str(config.neo4j_password, "NEO4J_PASSWORD")
    database = _pick_optional_str(config.neo4j_database, "NEO4J_DATABASE")
    fail_fast = _pick_bool(config.graph_fail_fast, "GRAPH_FAIL_FAST", True)

    return GraphConfig(
        backend=backend,
        uri=uri,
        user=user,
        password=password,
        database=database,
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


def _pick_bool(override: Optional[bool], env_key: str, default: bool) -> bool:
    if override is not None:
        return override
    raw = os.getenv(env_key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
