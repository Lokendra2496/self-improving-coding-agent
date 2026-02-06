from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from psycopg.types.json import Json

from agent.state import RunConfig, RunState
from storage.db import connect
from storage.utils import redact_config


def create_run_record(run_id: str, goal: str, status: str, config: RunConfig) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, goal, status, created_at, config, metrics, iterations)
            VALUES (%s, %s, %s, now(), %s, %s, %s)
            ON CONFLICT (run_id) DO NOTHING
            """,
            (
                run_id,
                goal,
                status,
                Json(redact_config(asdict(config))),
                Json({}),
                Json([]),
            ),
        )
        conn.commit()


def update_run_record(run_state: RunState, config: RunConfig, error: Optional[str] = None) -> None:
    run_dict = run_state.to_dict()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, goal, status, created_at, completed_at, config, metrics, iterations, error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id)
            DO UPDATE SET
                goal = EXCLUDED.goal,
                status = EXCLUDED.status,
                completed_at = EXCLUDED.completed_at,
                config = EXCLUDED.config,
                metrics = EXCLUDED.metrics,
                iterations = EXCLUDED.iterations,
                error = EXCLUDED.error
            """,
            (
                run_dict["run_id"],
                run_dict["goal"],
                run_dict["status"],
                run_dict["created_at"],
                run_dict.get("completed_at"),
                Json(redact_config(asdict(config))),
                Json(run_dict.get("metrics", {})),
                Json(run_dict.get("iterations", [])),
                error,
            ),
        )
        conn.commit()


def get_run_record(run_id: str) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT run_id, goal, status, created_at, completed_at, config, metrics, iterations, error
            FROM runs
            WHERE run_id = %s
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def list_runs(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT run_id, goal, status, created_at, completed_at, config, metrics, iterations, error
            FROM runs
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def _row_to_dict(row) -> Dict[str, Any]:
    return {
        "run_id": row[0],
        "goal": row[1],
        "status": row[2],
        "created_at": row[3],
        "completed_at": row[4],
        "config": row[5],
        "metrics": row[6],
        "iterations": row[7],
        "error": row[8],
    }
