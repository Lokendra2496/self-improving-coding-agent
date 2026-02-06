from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agent.graph import run_agent_loop
from agent.state import RunConfig, RunState
from eval.metrics import (
    compute_eval_summary,
    compute_repeated_mistake_rate,
    compute_run_success,
    compute_time_to_green_ms,
)
from storage.db import init_db
from storage.eval_store import add_eval_run, create_eval_record
from storage.run_store import update_run_record


def run_eval(
    goal: str,
    runs: int = 3,
    max_iters: int = 1,
    memory_backend: str = "local",
    persist: bool = True,
) -> Dict[str, Any]:
    run_states: List[RunState] = []

    if persist:
        init_db()

    for _ in range(runs):
        config = RunConfig(
            run_id=str(uuid4()),
            goal=goal,
            max_iters=max_iters,
            memory_backend=memory_backend,
        )
        state = run_agent_loop(config)
        run_states.append(state)
        if persist:
            update_run_record(state, config)

    summary = compute_eval_summary(run_states, memory_backend)
    if persist:
        eval_id = create_eval_record(
            config={"goal": goal, "runs": runs, "max_iters": max_iters, "memory_backend": memory_backend},
            summary=summary,
        )
        for state in run_states:
            add_eval_run(
                eval_id=eval_id,
                run_id=state.run_id,
                success=compute_run_success(state),
                iterations=len(state.iterations),
                time_to_green_ms=compute_time_to_green_ms(state),
                memory_on=memory_backend != "local",
            )
        summary["eval_id"] = eval_id

    return {"summary": summary, "runs": [state.to_dict() for state in run_states]}


def run_eval_from_cli(goal: str, runs: int, max_iters: int, memory_backend: str) -> None:
    result = run_eval(
        goal=goal,
        runs=runs,
        max_iters=max_iters,
        memory_backend=memory_backend,
        persist=True,
    )
    print(result["summary"])
