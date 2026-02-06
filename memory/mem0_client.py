from __future__ import annotations

from typing import Any, Dict, List, Optional

from mem0 import MemoryClient

from memory.config import MemoryConfig


class Mem0Client:
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.client = MemoryClient(
            api_key=config.api_key,
            host=config.host,
            org_id=config.org_id,
            project_id=config.project_id,
        )

    def add(self, message: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        user_id = self.config.user_id
        agent_id = self.config.agent_id
        app_id = self.config.app_id
        if not (user_id or agent_id or app_id):
            app_id = "self-improving-agent"
        kwargs: Dict[str, Any] = {"metadata": metadata}
        if user_id:
            kwargs["user_id"] = user_id
        if agent_id:
            kwargs["agent_id"] = agent_id
        if app_id:
            kwargs["app_id"] = app_id
        return self.client.add(message, **kwargs)

    def search(self, query: str, top_k: int) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        if self.config.user_id:
            filters["user_id"] = self.config.user_id
        if self.config.agent_id:
            filters["agent_id"] = self.config.agent_id
        if self.config.app_id:
            filters["app_id"] = self.config.app_id
        if not filters:
            filters["app_id"] = "self-improving-agent"
        return self.client.search(
            query,
            user_id=self.config.user_id,
            agent_id=self.config.agent_id,
            app_id=self.config.app_id,
            top_k=top_k,
            filters=filters,
        )
