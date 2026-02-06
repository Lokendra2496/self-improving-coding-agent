from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg


def get_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise ValueError("DATABASE_URL is required for database operations.")
    return dsn


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(get_dsn())
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id UUID PRIMARY KEY,
                goal TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                completed_at TIMESTAMPTZ,
                config JSONB NOT NULL,
                metrics JSONB,
                iterations JSONB NOT NULL,
                error TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evals (
                eval_id UUID PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL,
                config JSONB NOT NULL,
                summary JSONB NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_runs (
                eval_id UUID NOT NULL REFERENCES evals(eval_id) ON DELETE CASCADE,
                run_id UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                success BOOLEAN NOT NULL,
                iterations INT NOT NULL,
                time_to_green_ms INT,
                memory_on BOOLEAN NOT NULL,
                PRIMARY KEY (eval_id, run_id)
            )
            """
        )
        conn.commit()
