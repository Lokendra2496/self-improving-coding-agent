# Mem0 Memory Integration

## Purpose
Use Mem0 as the production memory layer for failure episodes and fix cards.

## Dependencies
- `04_memory_retrieval.md` for the memory interface.
- `07_llm_routing.md` for shared provider configuration.

## Connects To
- `05_observability.md` for memory usage logs.
- `06_api_eval.md` for memory inspection endpoints.

## Implementation
- `memory/config.py` resolves Mem0 config from env or CLI.
- `memory/mem0_client.py` wraps `mem0.MemoryClient`.
- `memory/store.py` selects backend (`mem0` or `local`) and enforces fail-fast.

## Build Tasks
1. Add Mem0 client and config resolution.
2. Wire memory store into the agent loop.
3. Add tests for local backend and Mem0 config validation.
