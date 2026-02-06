from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any, Dict, List

from psycopg.types.json import Json

from storage.db import connect
from storage.utils import redact_config


def create_eval_record(config: Dict[str, Any], summary: Dict[str, Any]) -> str:
    eval_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO evals (eval_id, created_at, config, summary)
            VALUES (%s, now(), %s, %s)
            """,
            (eval_id, Json(redact_config(config)), Json(summary)),
        )
        conn.commit()
    return eval_id


def add_eval_run(
    eval_id: str,
    run_id: str,
    success: bool,
    iterations: int,
    time_to_green_ms: int | None,
    memory_on: bool,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO eval_runs (eval_id, run_id, success, iterations, time_to_green_ms, memory_on)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (eval_id, run_id, success, iterations, time_to_green_ms, memory_on),
        )
        conn.commit()


def get_eval_record(eval_id: str) -> Dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT eval_id, created_at, config, summary
            FROM evals
            WHERE eval_id = %s
            """,
            (eval_id,),
        ).fetchone()
        if row is None:
            return None
        runs = conn.execute(
            """
            SELECT run_id, success, iterations, time_to_green_ms, memory_on
            FROM eval_runs
            WHERE eval_id = %s
            """,
            (eval_id,),
        ).fetchall()
    return {
        "eval_id": row[0],
        "created_at": row[1],
        "config": row[2],
        "summary": row[3],
        "runs": [
            {
                "run_id": r[0],
                "success": r[1],
                "iterations": r[2],
                "time_to_green_ms": r[3],
                "memory_on": r[4],
            }
            for r in runs
        ],
    }
