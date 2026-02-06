from agent.state import IterationRecord, RunState, StepResult, utc_now_iso
from eval.metrics import compute_eval_summary, compute_run_success, compute_time_to_green_ms


def _make_run(success: bool) -> RunState:
    run = RunState.start(goal="test")
    iteration = IterationRecord(index=0)
    status = "ok" if success else "error"
    details = {"result": {"exit_code": 0 if success else 1}}
    step = StepResult(name="execute", status=status, details=details)
    step.ended_at = utc_now_iso()
    iteration.steps.append(step)
    run.iterations.append(iteration)
    run.completed_at = utc_now_iso()
    return run


def test_compute_success() -> None:
    assert compute_run_success(_make_run(True)) is True
    assert compute_run_success(_make_run(False)) is False


def test_compute_time_to_green() -> None:
    run = _make_run(True)
    assert compute_time_to_green_ms(run) is not None


def test_compute_eval_summary() -> None:
    runs = [_make_run(True), _make_run(False)]
    summary = compute_eval_summary(runs, memory_backend="local")
    assert summary["runs"] == 2
    assert summary["pass_at_k"] is True
