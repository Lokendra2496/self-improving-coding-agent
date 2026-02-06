from __future__ import annotations

import subprocess
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class CommandResult:
    name: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int


@dataclass
class SandboxConfig:
    image: str
    cpus: float = 1.0
    memory: str = "1g"
    network: str = "none"
    user: Optional[str] = None
    workdir: str = "/work"


def run_command(
    command: List[str] | str,
    name: str = "command",
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout_s: Optional[int] = None,
) -> CommandResult:
    start = time.perf_counter()
    use_shell = isinstance(command, str)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            timeout=timeout_s,
            shell=use_shell,
            check=False,
        )
        duration_ms = int((time.perf_counter() - start) * 1000)
        return CommandResult(
            name=name,
            command=_format_command(command),
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=duration_ms,
        )
    except FileNotFoundError as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        missing = exc.filename or _format_command(command)
        return CommandResult(
            name=name,
            command=_format_command(command),
            exit_code=127,
            stdout="",
            stderr=f"Command not found: {missing}",
            duration_ms=duration_ms,
        )
    except subprocess.TimeoutExpired:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return CommandResult(
            name=name,
            command=_format_command(command),
            exit_code=124,
            stdout="",
            stderr="Command timed out.",
            duration_ms=duration_ms,
        )


def noop_result(name: str, message: str) -> CommandResult:
    return CommandResult(
        name=name,
        command="noop",
        exit_code=0,
        stdout=message,
        stderr="",
        duration_ms=0,
    )




def _format_command(command: List[str] | str) -> str:
    if isinstance(command, str):
        return command
    return " ".join(command)


def build_docker_command(
    repo_path: str | Path,
    command: Sequence[str] | str,
    config: SandboxConfig,
) -> List[str]:
    repo_path = str(Path(repo_path).resolve())
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        config.network,
        "--cpus",
        str(config.cpus),
        "--memory",
        config.memory,
        "-v",
        f"{repo_path}:{config.workdir}",
        "-w",
        config.workdir,
    ]

    user = config.user or _default_user()
    if user:
        docker_cmd.extend(["--user", user])

    docker_cmd.append(config.image)
    docker_cmd.extend(_normalize_command(command))
    return docker_cmd


def _normalize_command(command: Sequence[str] | str) -> List[str]:
    if isinstance(command, str):
        return ["/bin/sh", "-lc", command]
    return list(command)


def _default_user() -> Optional[str]:
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        return f"{os.getuid()}:{os.getgid()}"
    return None
