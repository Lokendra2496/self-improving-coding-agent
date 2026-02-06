# Repo Layout And Execution Loop

## Purpose
Establish the directory structure and a minimal agent loop that can run from
the CLI. This is the backbone of the app.

## Dependencies
- `01_outcomes.md` for acceptance criteria.

## Connects To
- `03_tools_safety.md` for execution steps and safety gates.
- `04_memory_retrieval.md` for retrieval inputs and memory updates.
- `05_observability.md` for logging the loop.

## Proposed Layout (MVP)
```
app/            # CLI and optional API entrypoints
agent/          # loop state, orchestration graph
tools/          # sandbox, test runner, static checks
llm/            # LiteLLM routing + model wrappers
rag/            # repo retrieval stubs
memory/         # memory store stubs
observability/  # structured logging and metrics
eval/           # evaluation harness
```

## Execution Loop (MVP)
1. Reproduce (placeholder)
2. Plan (FAST model)
3. Retrieve (FAST query rewrite + stub)
4. Patch (STRONG model)
5. Safety (stub checks)
6. Execute (tool runner)
7. Reflect (STRONG model)
8. Memory update (stub)
9. Decide (stop after 1 iteration by default)

## Build Tasks
1. Implement `agent/state.py` with a minimal run/iteration schema.
2. Implement `agent/graph.py` with a linear loop driver.
3. Add `app/cli.py` to execute the loop and print a summary.
