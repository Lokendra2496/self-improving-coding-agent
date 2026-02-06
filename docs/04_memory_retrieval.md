# Memory And Retrieval Interfaces

## Purpose
Define the RAG and memory interfaces so implementations can be swapped later.

## Dependencies
- `02_repo_layout_execution_loop.md` for loop integration points.
- `03_tools_safety.md` for context that memory stores.

## Connects To
- `05_observability.md` for logging retrieval and memory hits.
- `06_api_eval.md` for exposure through API endpoints and evals.

## MVP Scope
- Repo retrieval with lexical recall + vector search (FAISS or pgvector).
- FAST model rewrites the goal into a retrieval query.
- Memory store uses Mem0 in production, with a local fallback for tests.

## Build Tasks
1. Add chunking, embeddings, and indexer in `rag/`.
2. Add FAISS and pgvector backends for retrieval.
3. Wire retrieval into the agent loop.
4. Integrate Mem0 via `memory/mem0_client.py`.
