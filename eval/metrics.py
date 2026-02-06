from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from agent.state import RunState


def compute_run_success(state: RunState) -> bool:
    for iteration in state.iterations:
        for step in iteration.steps:
            if step.name == "execute" and step.status == "ok":
                result = step.details.get("result", {})
                if result.get("exit_code", 1) == 0:
                    return True
    return False


def compute_time_to_green_ms(state: RunState) -> Optional[int]:
    created_at = _parse_ts(state.created_at)
    if not created_at:
        return None
    best = None
    for iteration in state.iterations:
        for step in iteration.steps:
            if step.name == "execute" and step.status == "ok":
                result = step.details.get("result", {})
                if result.get("exit_code", 1) != 0:
                    continue
                ended_at = _parse_ts(step.ended_at)
                if ended_at is None:
                    continue
                delta_ms = int((ended_at - created_at).total_seconds() * 1000)
                if best is None or delta_ms < best:
                    best = delta_ms
    return best


def compute_repeated_mistake_rate(state: RunState) -> float:
    fingerprints: List[str] = []
    for iteration in state.iterations:
        for step in iteration.steps:
            if step.name == "patch":
                fp = step.details.get("fingerprint")
                if isinstance(fp, str):
                    fingerprints.append(fp)
    if not fingerprints:
        return 0.0
    unique = len(set(fingerprints))
    return (len(fingerprints) - unique) / len(fingerprints)


def compute_eval_summary(states: List[RunState], memory_backend: str) -> Dict[str, object]:
    if not states:
        return {
            "runs": 0,
            "pass_at_1": False,
            "pass_at_k": False,
            "avg_iterations": 0.0,
            "avg_time_to_green_ms": None,
            "repeated_mistake_rate": 0.0,
            "memory_on": memory_backend != "local",
        }

    successes = [compute_run_success(state) for state in states]
    pass_at_1 = successes[0] if successes else False
    pass_at_k = any(successes)
    avg_iterations = sum(len(state.iterations) for state in states) / len(states)

    times = [t for t in (compute_time_to_green_ms(s) for s in states) if t is not None]
    avg_time_to_green = int(sum(times) / len(times)) if times else None
    repeated = sum(compute_repeated_mistake_rate(s) for s in states) / len(states)

    return {
        "runs": len(states),
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
        "avg_iterations": avg_iterations,
        "avg_time_to_green_ms": avg_time_to_green,
        "repeated_mistake_rate": repeated,
        "memory_on": memory_backend != "local",
    }


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
