from pathlib import Path

from tools.sandbox import SandboxConfig, build_docker_command


def test_build_docker_command_includes_safety_flags(tmp_path: Path) -> None:
    config = SandboxConfig(
        image="self-improving-agent:latest",
        cpus=1.0,
        memory="1g",
        network="none",
        user="1000:1000",
    )
    cmd = build_docker_command(tmp_path, ["pytest", "-q"], config)

    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--network" in cmd
    assert "none" in cmd
    assert "--cpus" in cmd
    assert "--memory" in cmd
    assert "-v" in cmd
    assert "-w" in cmd
    assert config.image in cmd
