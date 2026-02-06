from pathlib import Path

from agent.graph import run_agent_loop
from agent.state import RunConfig


TEST_CMD = ["python", "-c", "print('ok')"]


def test_graph_runs_and_checkpoints(tmp_path: Path) -> None:
    config = RunConfig(
        goal="test run",
        checkpoint_dir=str(tmp_path),
        test_cmd=TEST_CMD,
        ruff_cmd=TEST_CMD,
        bandit_cmd=TEST_CMD,
    )
    state = run_agent_loop(config)

    checkpoint_file = tmp_path / f"{state.run_id}.json"
    assert checkpoint_file.exists()
    assert state.status == "completed"


def test_resume_from_checkpoint_and_retry(tmp_path: Path) -> None:
    pause_config = RunConfig(
        goal="resume test",
        checkpoint_dir=str(tmp_path),
        stop_after_step="plan",
        test_cmd=TEST_CMD,
        ruff_cmd=TEST_CMD,
        bandit_cmd=TEST_CMD,
    )
    paused = run_agent_loop(pause_config)
    assert paused.status == "paused"

    resume_config = RunConfig(
        goal="resume test",
        checkpoint_dir=str(tmp_path),
        resume_run_id=paused.run_id,
        force_retry_once=True,
        max_retries=1,
        test_cmd=TEST_CMD,
        ruff_cmd=TEST_CMD,
        bandit_cmd=TEST_CMD,
    )
    resumed = run_agent_loop(resume_config)
    assert resumed.status == "completed"

    execute_steps = [
        step for step in resumed.iterations[0].steps if step.name == "execute"
    ]
    assert len(execute_steps) >= 2
