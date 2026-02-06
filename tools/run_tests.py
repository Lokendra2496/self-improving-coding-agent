from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from tools.sandbox import CommandResult, SandboxConfig, build_docker_command, run_command


def run_tests(
    command: Optional[List[str] | str] = None,
    use_sandbox: bool = False,
    sandbox_config: Optional[SandboxConfig] = None,
    repo_path: str | Path = ".",
) -> CommandResult:
    if command is None:
        raise ValueError("Test command not configured.")
    if use_sandbox:
        if sandbox_config is None:
            raise ValueError("Sandbox config missing.")
        docker_cmd = build_docker_command(repo_path, command, sandbox_config)
        return run_command(docker_cmd, name="tests")
    return run_command(command, name="tests", cwd=str(Path(repo_path).resolve()))
