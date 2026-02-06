from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from tools.sandbox import CommandResult, SandboxConfig, build_docker_command, run_command


def run_static_checks(
    ruff_cmd: Optional[List[str] | str] = None,
    bandit_cmd: Optional[List[str] | str] = None,
    use_sandbox: bool = False,
    sandbox_config: Optional[SandboxConfig] = None,
    repo_path: str | Path = ".",
) -> List[CommandResult]:
    results: List[CommandResult] = []

    if ruff_cmd is None:
        raise ValueError("Ruff command not configured.")
    else:
        results.append(
            _run_with_sandbox(
                ruff_cmd,
                name="ruff",
                use_sandbox=use_sandbox,
                sandbox_config=sandbox_config,
                repo_path=repo_path,
            )
        )

    if bandit_cmd is None:
        raise ValueError("Bandit command not configured.")
    else:
        results.append(
            _run_with_sandbox(
                bandit_cmd,
                name="bandit",
                use_sandbox=use_sandbox,
                sandbox_config=sandbox_config,
                repo_path=repo_path,
            )
        )

    return results


def _run_with_sandbox(
    command: List[str] | str,
    name: str,
    use_sandbox: bool,
    sandbox_config: Optional[SandboxConfig],
    repo_path: str | Path,
) -> CommandResult:
    if use_sandbox:
        if sandbox_config is None:
            raise ValueError("Sandbox config missing.")
        docker_cmd = build_docker_command(repo_path, command, sandbox_config)
        return run_command(docker_cmd, name=name)
    return run_command(command, name=name, cwd=str(Path(repo_path).resolve()))
