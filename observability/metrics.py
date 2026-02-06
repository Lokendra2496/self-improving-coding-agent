from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from prometheus_client import Counter, Histogram, start_http_server

_METRICS_STARTED = False
_RUNS_TOTAL = Counter("agent_runs_total", "Total agent runs")
_ITERATIONS_TOTAL = Counter("agent_iterations_total", "Total iterations")
_STEPS_TOTAL = Counter("agent_steps_total", "Total steps")
_RUN_DURATION = Histogram(
    "agent_run_duration_seconds",
    "Run duration in seconds",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)


class Metrics:
    def __init__(self) -> None:
        self._counters: Dict[str, int] = {}
        self._timings: Dict[str, List[int]] = {}

    def incr(self, name: str, value: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value
        _record_counter(name, value)

    def record_timing(self, name: str, duration_ms: int) -> None:
        self._timings.setdefault(name, []).append(duration_ms)
        _record_timing(name, duration_ms)

    def snapshot(self) -> Dict[str, object]:
        timing_summary = {}
        for name, values in self._timings.items():
            if not values:
                continue
            timing_summary[name] = {
                "count": len(values),
                "avg_ms": int(sum(values) / len(values)),
                "min_ms": min(values),
                "max_ms": max(values),
            }
        return {"counters": dict(self._counters), "timings": timing_summary}


def configure_metrics() -> None:
    global _METRICS_STARTED
    if _METRICS_STARTED:
        return
    port = os.getenv("METRICS_PORT")
    if port:
        try:
            start_http_server(int(port))
        except OSError:
            # Port already in use; skip starting another server.
            pass
    _METRICS_STARTED = True


def write_metrics_csv(run_id: str, metrics: Dict[str, object], path: Optional[str] = None) -> None:
    path = path or os.getenv("METRICS_CSV_PATH")
    if not path:
        return
    counters = metrics.get("counters", {})
    timings = metrics.get("timings", {})
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "counters": counters,
        "timings": timings,
    }
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _record_counter(name: str, value: int) -> None:
    if name == "runs_total":
        _RUNS_TOTAL.inc(value)
    elif name == "iterations_total":
        _ITERATIONS_TOTAL.inc(value)
    elif name == "steps_total":
        _STEPS_TOTAL.inc(value)


def _record_timing(name: str, duration_ms: int) -> None:
    if name == "run_duration_ms":
        _RUN_DURATION.observe(duration_ms / 1000.0)


def record_counter(name: str, value: int = 1) -> None:
    _record_counter(name, value)


def record_timing(name: str, duration_ms: int) -> None:
    _record_timing(name, duration_ms)
