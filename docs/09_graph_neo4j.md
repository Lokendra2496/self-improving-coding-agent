# Neo4j Graph Memory

## Purpose
Persist relationships between runs, iterations, steps, and retrieved files in Neo4j.

## Dependencies
- `08_memory_mem0.md` for memory context.
- `04_memory_retrieval.md` for retrieved file metadata.

## Connects To
- `06_api_eval.md` for graph inspection APIs (future).
- `05_observability.md` for graph usage logs.

## Implementation
- `memory/graph_config.py` resolves Neo4j config.
- `memory/graph_store.py` writes nodes/edges for runs and iterations.
- `agent/graph.py` calls graph store during `memory_update`.

## Build Tasks
1. Add Neo4j driver and config support.
2. Implement graph upsert for run/iteration/step/file nodes.
3. Wire graph store into the agent loop with fail-fast control.
