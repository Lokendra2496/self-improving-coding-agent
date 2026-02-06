# Section Index

This index breaks the production plan into buildable sections. Each section
explains purpose, dependencies, and concrete build tasks.

## Sections
- `01_outcomes.md` — success criteria and MVP scope
- `02_repo_layout_execution_loop.md` — repo layout + core agent loop
- `03_tools_safety.md` — sandbox execution and safety gates
- `04_memory_retrieval.md` — retrieval + memory interfaces
- `07_llm_routing.md` — LiteLLM routing and two-tier models
- `08_memory_mem0.md` — Mem0 memory integration
- `09_graph_neo4j.md` — Neo4j graph memory integration
- `05_observability.md` — logging and metrics
- `06_api_eval.md` — API server and evaluation harness

## How To Use
Build sections in order. After each section is implemented, review how it
connects to the rest of the system and adjust the next section accordingly.
