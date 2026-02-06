# Observability

## Purpose
Provide structured logs and metrics so runs are debuggable and measurable.

## Dependencies
- `02_repo_layout_execution_loop.md` for logging points.
- `03_tools_safety.md` for command execution results.

## Connects To
- `06_api_eval.md` for exposing run data and evaluations.

## MVP Scope
- Structured logging helper.
- OpenTelemetry tracing to Phoenix.
- Prometheus-compatible `/metrics` endpoint (optional).

## Build Tasks
1. Implement `observability/logging.py` with a JSON logger helper.
2. Implement `observability/tracing.py` for OTLP export.
3. Implement `observability/metrics.py` for Prometheus + CSV export.
4. Add spans around run + step execution and around LLM/RAG calls.
