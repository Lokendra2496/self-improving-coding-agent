from __future__ import annotations

import sys
from pathlib import Path
import os
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEMORY_BACKEND", "mem0")
os.environ.setdefault("MEM0_API_KEY", "test-key")
os.environ.setdefault("MEM0_ORG_ID", "test-org")
os.environ.setdefault("MEM0_PROJECT_ID", "test-project")
os.environ.setdefault("GRAPH_BACKEND", "neo4j")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-pass")
os.environ.setdefault("LLM_ENABLED", "1")
os.environ.setdefault("TRACING_ENABLED", "0")
os.environ.setdefault("TRACING_DEFAULT_PHOENIX", "0")


@pytest.fixture(autouse=True)
def _mock_litellm_completion(monkeypatch):
    def fake_completion(**kwargs):
        messages = kwargs.get("messages", [])
        prompt = ""
        if messages and isinstance(messages, list):
            prompt = messages[0].get("content", "")
        if "Decide whether the request requires code review" in prompt:
            content = '{"sql_review": true, "code_review": true, "reason": "tests"}'
        elif "Review the code snippets" in prompt:
            content = "[]"
        elif "Generate a unified diff" in prompt:
            content = (
                "diff --git a/mock_response.txt b/mock_response.txt\n"
                "new file mode 100644\n"
                "index 0000000..e69de29\n"
                "--- /dev/null\n"
                "+++ b/mock_response.txt\n"
                "@@ -0,0 +1 @@\n"
                "+mock\n"
            )
        elif "Create a JSON fix card" in prompt:
            content = (
                '{"summary": "Fixed failing tests", '
                '"root_cause": "Test setup missing config", '
                '"fix": "Added required config to setup", '
                '"verification": "pytest -q", '
                '"error_signature": "AssertionError", '
                '"files_changed": ["mock_response.txt"]}'
            )
        elif "Review the SQL queries" in prompt:
            content = "[]"
        elif "Rewrite the following goal" in prompt:
            content = prompt.splitlines()[-1].strip() if prompt else "query"
        elif "Provide a concise plan" in prompt:
            content = "Plan: analyze, patch, test."
        elif "Provide a brief reflection" in prompt:
            content = "Reflection: review results and adjust next steps."
        else:
            content = f"ok:{kwargs.get('model', '')}"
        return {"choices": [{"message": {"content": content}}]}

    import litellm

    monkeypatch.setattr(litellm, "completion", fake_completion)


@pytest.fixture(autouse=True)
def _mock_mem0_client(monkeypatch):
    from mem0.client.main import MemoryClient
    from memory.mem0_client import Mem0Client

    monkeypatch.setattr(MemoryClient, "_validate_api_key", lambda self: "test@example.com")

    def fake_add(self, message, metadata):
        return {"id": "test", "message": message, "metadata": metadata}

    def fake_search(self, query, top_k):
        return {"results": []}

    monkeypatch.setattr(Mem0Client, "add", fake_add)
    monkeypatch.setattr(Mem0Client, "search", fake_search)


@pytest.fixture(autouse=True)
def _mock_neo4j_driver(monkeypatch):
    from neo4j import GraphDatabase

    class DummySession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute_write(self, func, episode):
            return func(self, episode)

        def run(self, *args, **kwargs):
            return None

    class DummyDriver:
        def session(self, **kwargs):
            return DummySession()

        def close(self):
            return None

    monkeypatch.setattr(GraphDatabase, "driver", lambda *args, **kwargs: DummyDriver())
