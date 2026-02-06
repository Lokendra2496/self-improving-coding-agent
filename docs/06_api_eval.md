# API Server And Evaluation Harness

## Purpose
Expose run control and evaluation outputs, enabling a working app beyond CLI.

## Dependencies
- `02_repo_layout_execution_loop.md` for run orchestration.
- `05_observability.md` for run data.
- `04_memory_retrieval.md` for memory endpoints.

## Connects To
- All sections, because API and evals surface their results.

## MVP Scope
- API endpoints to create runs, list runs, and fetch run details.
- Evaluation endpoint for pass@k, time-to-green, and repeat rate.

## Build Tasks
1. Add `app/main.py` with production API endpoints and DB persistence.
2. Add `eval/runner.py` with pass@k and time-to-green metrics.
3. Persist run and eval results in Postgres.
