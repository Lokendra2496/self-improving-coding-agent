import pytest

from llm.client import LLMClient
from llm.config import LLMConfig


def test_llm_disabled_raises() -> None:
    config = LLMConfig(
        enabled=False,
        fast_model="fast",
        strong_model="strong",
        base_url=None,
        api_key=None,
        fast_temperature=0.2,
        strong_temperature=0.2,
        timeout_s=None,
    )
    client = LLMClient(config)
    with pytest.raises(ValueError):
        client.fast_complete("hello")


def test_llm_enabled_uses_litellm(monkeypatch) -> None:
    def fake_completion(**kwargs):
        return {"choices": [{"message": {"content": f"ok:{kwargs['model']}"}}]}

    import litellm

    monkeypatch.setattr(litellm, "completion", fake_completion)

    config = LLMConfig(
        enabled=True,
        fast_model="fast-model",
        strong_model="strong-model",
        base_url=None,
        api_key="test",
        fast_temperature=0.1,
        strong_temperature=0.2,
        timeout_s=30,
    )
    client = LLMClient(config)
    fast = client.fast_complete("hello")
    strong = client.strong_complete("hello")

    assert fast.content == "ok:fast-model"
    assert strong.content == "ok:strong-model"
