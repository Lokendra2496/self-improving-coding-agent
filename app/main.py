from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.graph import run_agent_loop
from agent.state import RunConfig
from eval.runner import run_eval
from observability.logging import configure_logging
from observability.metrics import configure_metrics
from observability.tracing import configure_tracing
from storage.db import init_db
from storage.eval_store import get_eval_record
from storage.run_store import create_run_record, get_run_record, list_runs, update_run_record


class RunRequest(BaseModel):
    goal: str = "Fix failing tests"
    run_mode: str = Field(default="sync", pattern="^(sync|async)$")
    config: Dict[str, Any] = Field(default_factory=dict)


class EvalRequest(BaseModel):
    goal: str = "Fix failing tests"
    runs: int = 3
    max_iters: int = 1
    memory_backend: str = "local"
    run_mode: str = Field(default="sync", pattern="^(sync|async)$")


def create_app() -> FastAPI:
    configure_logging()
    configure_tracing()
    configure_metrics()
    init_db()

    app = FastAPI(title="Self-Improving Coding Agent")

    @app.post("/runs")
    def create_run(request: RunRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
        run_id = str(uuid4())
        config = _build_run_config(request.goal, request.config, run_id)
        create_run_record(run_id, request.goal, "queued", config)

        if request.run_mode == "async":
            background_tasks.add_task(_run_and_store, config)
            return {"run_id": run_id, "status": "queued"}

        run_state = _run_and_store(config)
        return {"run_id": run_state.run_id, "status": run_state.status, "run": run_state.to_dict()}

    @app.get("/runs")
    def list_run_records(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        runs = list_runs(limit=limit, offset=offset)
        return {"runs": runs}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> Dict[str, Any]:
        record = get_run_record(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return record

    @app.post("/evals")
    def create_eval(request: EvalRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
        if request.run_mode == "async":
            background_tasks.add_task(_run_eval_and_store, request)
            return {"status": "queued"}
        result = _run_eval_and_store(request)
        return result

    @app.get("/evals/{eval_id}")
    def get_eval(eval_id: str) -> Dict[str, Any]:
        record = get_eval_record(eval_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Eval not found")
        return record

    return app


def _build_run_config(goal: str, overrides: Dict[str, Any], run_id: str) -> RunConfig:
    fields = RunConfig.__dataclass_fields__.keys()
    sanitized = {k: v for k, v in overrides.items() if k in fields}
    sanitized["goal"] = goal
    sanitized["run_id"] = run_id
    return RunConfig(**sanitized)


def _run_and_store(config: RunConfig):
    try:
        run_state = run_agent_loop(config)
        update_run_record(run_state, config)
        return run_state
    except Exception as exc:
        update_run_record(
            _failed_placeholder(config, str(exc)),
            config,
            error=str(exc),
        )
        raise


def _failed_placeholder(config: RunConfig, error: str):
    from agent.state import RunState

    state = RunState.start(goal=config.goal, run_id=config.run_id)
    state.status = "failed"
    state.completed_at = state.created_at
    state.metrics = {"error": error}
    return state


def _run_eval_and_store(request: EvalRequest) -> Dict[str, Any]:
    result = run_eval(
        goal=request.goal,
        runs=request.runs,
        max_iters=request.max_iters,
        memory_backend=request.memory_backend,
        persist=True,
    )
    return result
