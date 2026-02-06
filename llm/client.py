from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import litellm

from llm.config import LLMConfig
from observability.tracing import span


@dataclass
class LLMResponse:
    content: str
    model: str


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def fast_complete(self, prompt: str) -> LLMResponse:
        return self._complete(prompt, model=self.config.fast_model, temperature=self.config.fast_temperature)

    def strong_complete(self, prompt: str) -> LLMResponse:
        return self._complete(prompt, model=self.config.strong_model, temperature=self.config.strong_temperature)

    def _complete(self, prompt: str, model: str, temperature: float) -> LLMResponse:
        with span(
            "llm.complete",
            {
                "llm.model": model,
                "llm.temperature": temperature,
                "llm.enabled": self.config.enabled,
            },
        ):
            if not self.config.enabled:
                raise ValueError("LLM is disabled. Set LLM_ENABLED=1.")

            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            if self.config.timeout_s is not None:
                kwargs["timeout"] = self.config.timeout_s

            response = litellm.completion(**kwargs)
            return LLMResponse(content=_extract_content(response), model=model)


def _extract_content(response: Any) -> str:
    choices = _get(response, "choices")
    if not choices:
        return str(response)

    first = choices[0]
    message = _get(first, "message")
    if message is None:
        return str(first)
    content = _get(message, "content")
    return content if isinstance(content, str) else str(message)




def _get(obj: Any, key: str) -> Optional[Any]:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
