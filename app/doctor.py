"""Config doctor - checks dependencies and configuration."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CheckResult:
    name: str
    status: str  # "ok", "warn", "error"
    message: str
    fix_hint: Optional[str] = None


def run_doctor() -> List[CheckResult]:
    """Run all health checks and return results."""
    results: List[CheckResult] = []

    # Core tools
    results.append(_check_binary("docker", "Docker is required for sandboxed execution"))
    results.append(_check_binary("rg", "ripgrep (rg) is used for lexical search in RAG"))
    results.append(_check_binary("git", "Git is required for patch application"))

    # Docker services
    results.append(_check_docker_running())
    results.append(_check_postgres())
    results.append(_check_neo4j())

    # API keys and config
    results.append(_check_env_var("OPENAI_API_KEY", "OpenAI API key for LLM calls", optional=True))
    results.append(_check_env_var("GROQ_API_KEY", "Groq API key for fast LLM calls", optional=True))
    results.append(_check_env_var("MEM0_API_KEY", "Mem0 API key for memory backend", optional=True))

    # LLM config
    results.append(_check_llm_config())

    return results


def print_doctor_report(results: List[CheckResult]) -> int:
    """Print formatted doctor report and return exit code."""
    print("\nSelf-Improving Agent Health Check\n")
    print("=" * 50)

    ok_count = 0
    warn_count = 0
    error_count = 0

    for result in results:
        icon = _status_icon(result.status)
        print(f"{icon} {result.name}: {result.message}")
        if result.fix_hint:
            print(f"   Hint: {result.fix_hint}")

        if result.status == "ok":
            ok_count += 1
        elif result.status == "warn":
            warn_count += 1
        else:
            error_count += 1

    print("=" * 50)
    print(f"\n{ok_count} passed, {warn_count} warnings, {error_count} errors\n")

    if error_count > 0:
        print("Some required dependencies are missing. Fix errors above to proceed.")
        return 1
    elif warn_count > 0:
        print("Some optional features may not work. See warnings above.")
        return 0
    else:
        print("All checks passed! You're ready to go.")
        return 0


def _status_icon(status: str) -> str:
    return {"ok": "[OK]", "warn": "[WARN]", "error": "[ERROR]"}.get(status, "[?]")


def _check_binary(name: str, description: str) -> CheckResult:
    """Check if a binary is available in PATH."""
    path = shutil.which(name)
    if path:
        return CheckResult(name=name, status="ok", message=f"Found at {path}")
    return CheckResult(
        name=name,
        status="warn" if name == "rg" else "error",
        message=f"Not found - {description}",
        fix_hint=f"Install {name} and ensure it's in your PATH",
    )


def _check_docker_running() -> CheckResult:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return CheckResult(name="Docker daemon", status="ok", message="Running")
        return CheckResult(
            name="Docker daemon",
            status="error",
            message="Not running",
            fix_hint="Start Docker Desktop or run 'sudo systemctl start docker'",
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return CheckResult(
            name="Docker daemon",
            status="error",
            message="Could not connect",
            fix_hint="Install and start Docker",
        )


def _check_postgres() -> CheckResult:
    """Check Postgres connection."""
    dsn = os.getenv("DATABASE_URL") or os.getenv("RAG_PG_DSN")
    if not dsn:
        return CheckResult(
            name="Postgres",
            status="warn",
            message="No DATABASE_URL or RAG_PG_DSN configured",
            fix_hint="Set DATABASE_URL for persistent storage, or run 'docker compose up -d'",
        )

    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return CheckResult(name="Postgres", status="ok", message="Connected")
    except ImportError:
        return CheckResult(
            name="Postgres",
            status="warn",
            message="psycopg not installed",
            fix_hint="Run 'uv pip install psycopg[binary]'",
        )
    except Exception as e:
        return CheckResult(
            name="Postgres",
            status="warn",
            message=f"Connection failed: {type(e).__name__}",
            fix_hint="Check DATABASE_URL and ensure Postgres is running",
        )


def _check_neo4j() -> CheckResult:
    """Check Neo4j connection."""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri:
        return CheckResult(
            name="Neo4j",
            status="warn",
            message="NEO4J_URI not configured",
            fix_hint="Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD or use --graph-backend local",
        )

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        return CheckResult(name="Neo4j", status="ok", message="Connected")
    except ImportError:
        return CheckResult(
            name="Neo4j",
            status="warn",
            message="neo4j driver not installed",
            fix_hint="Run 'uv pip install neo4j'",
        )
    except Exception as e:
        return CheckResult(
            name="Neo4j",
            status="warn",
            message=f"Connection failed: {type(e).__name__}",
            fix_hint="Check NEO4J_* env vars and ensure Neo4j is running",
        )


def _check_env_var(name: str, description: str, optional: bool = False) -> CheckResult:
    """Check if an environment variable is set."""
    value = os.getenv(name)
    if value:
        masked = value[:4] + "..." + value[-4:] if len(value) > 10 else "***"
        return CheckResult(name=name, status="ok", message=f"Set ({masked})")
    status = "warn" if optional else "error"
    return CheckResult(
        name=name,
        status=status,
        message=f"Not set - {description}",
        fix_hint=f"Set {name} in .env or shell",
    )


def _check_llm_config() -> CheckResult:
    """Check LLM configuration."""
    enabled = os.getenv("LLM_ENABLED", "").lower() in ("1", "true", "yes")
    if not enabled:
        return CheckResult(
            name="LLM",
            status="warn",
            message="LLM disabled (mock responses)",
            fix_hint="Set LLM_ENABLED=1 and configure FAST_MODEL/STRONG_MODEL",
        )

    fast = os.getenv("FAST_MODEL")
    strong = os.getenv("STRONG_MODEL")
    if fast and strong:
        return CheckResult(
            name="LLM",
            status="ok",
            message=f"Enabled (fast={fast}, strong={strong})",
        )
    return CheckResult(
        name="LLM",
        status="warn",
        message="LLM enabled but models not configured",
        fix_hint="Set FAST_MODEL and STRONG_MODEL",
    )
