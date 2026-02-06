# Tools And Safety Gates

## Purpose
Provide a safe execution environment and static checks that guard the agent.

## Dependencies
- `02_repo_layout_execution_loop.md` for tool invocation points.

## Connects To
- `04_memory_retrieval.md` because tool outputs feed reflection/memory.
- `05_observability.md` for logging command execution.

## MVP Scope
- A sandbox runner wrapper that can execute a command and return exit code.
- `ruff`, `bandit`, and test runner support.
- Docker sandbox support with `--network none`, CPU/memory limits, and non-root user.

## Build Tasks
1. Implement `tools/sandbox.py` with a `run_command()` wrapper.
2. Add Docker command builder and sandbox config.
3. Implement `tools/run_tests.py` and `tools/static_checks.py` for sandbox usage.
4. Ensure the agent loop calls safety checks before execution.
