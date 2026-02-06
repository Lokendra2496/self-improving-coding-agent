# LiteLLM Routing And Two-Tier Models

## Purpose
Provide a centralized LLM gateway and enforce a two-tier model strategy:
FAST for planning/retrieval and STRONG for patch/reflection.

## Dependencies
- `02_repo_layout_execution_loop.md` for loop integration points.
- `03_tools_safety.md` for execution boundary.

## Connects To
- `04_memory_retrieval.md` for query rewriting.
- `05_observability.md` for logging LLM usage.

## Implementation
- `llm/config.py` resolves models, API key, and optional base URL.
- `llm/client.py` exposes `fast_complete()` and `strong_complete()`.
- `agent/graph.py` calls FAST in `plan` and `retrieve`, STRONG in `patch` and `reflect`.

## Build Tasks
1. Add LiteLLM dependency and config resolution.
2. Implement `LLMClient` with FAST/STRONG calls.
3. Wire the client into graph nodes.
4. Add tests that mock LiteLLM for offline runs.
