# Outcomes And MVP Scope

## Purpose
Define what "done" means and how to validate progress. This anchors all other
sections and prevents scope creep.

## Dependencies
- None. This is the root of the build plan.

## Connects To
- `02_repo_layout_execution_loop.md` by defining required loop outputs.
- `05_observability.md` because outcomes require logs and metrics.
- `06_api_eval.md` because outcomes include run reporting.

## MVP Definition
- A CLI command can run a single iteration loop end-to-end.
- The loop produces structured logs and a run record in memory.
- Tools and retrieval are stubbed but return deterministic placeholders.
- Static checks and a test runner wrapper exist and can be invoked.

## Build Tasks
1. Document the MVP acceptance criteria in `README.md`.
2. Ensure the CLI has a clear `run` entrypoint.
3. Define a minimal run state schema with iteration history.
4. Provide a no-op default plan/patch path that still exercises the loop.
