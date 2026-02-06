from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from agent.graph import run_agent_loop
from agent.state import RunConfig
from observability.logging import configure_logging
from observability.metrics import configure_metrics
from observability.tracing import configure_tracing
from rag.config import resolve_rag_config
from rag.index_repo import index_repo


def _load_dotenv() -> None:
    """Auto-load .env file if present."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        # Manual fallback if python-dotenv not installed
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-Improving Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run --goal "Fix failing tests" --repo-path ./myproject \
    --ruff-cmd "ruff check ." --bandit-cmd "bandit -r ." --test-cmd "pytest -q"
  python main.py doctor
  python main.py index --repo-path .
  python main.py runs
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the agent loop")
    index_parser = subparsers.add_parser("index", help="Index repository for RAG")
    eval_parser = subparsers.add_parser("eval", help="Run evaluation suite")
    subparsers.add_parser("doctor", help="Check dependencies and configuration")
    subparsers.add_parser("runs", help="List recent runs")

    run_parser.add_argument(
        "--goal",
        default="Fix failing tests",
        help="Goal description for the agent run",
    )
    run_parser.add_argument(
        "--max-iters",
        type=int,
        default=1,
        help="Maximum iterations to execute",
    )
    run_parser.add_argument(
        "--until-green",
        action="store_true",
        help="Continue iterations until tests pass (respects --max-iters).",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Disabled in full runs (will error if set)",
    )
    run_parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run plan/retrieve/reflect only (no patch or tool execution).",
    )
    run_parser.add_argument(
        "--auto-index",
        action="store_true",
        default=None,
        help="Auto-index repo if the RAG index is missing or stale.",
    )
    run_parser.add_argument(
        "--no-auto-index",
        action="store_false",
        dest="auto_index",
        help="Disable auto-indexing.",
    )
    run_parser.add_argument(
        "--ruff-cmd",
        default="",
        help="Command to run ruff checks (e.g. 'ruff check .')",
    )
    run_parser.add_argument(
        "--bandit-cmd",
        default="",
        help="Command to run bandit checks (e.g. 'bandit -r .')",
    )
    run_parser.add_argument(
        "--test-cmd",
        default="",
        help="Command to run tests (e.g. 'pytest -q')",
    )
    run_parser.add_argument(
        "--resume-run-id",
        default="",
        help="Resume a run from a checkpoint by run_id",
    )
    run_parser.add_argument(
        "--checkpoint-dir",
        default=".checkpoints",
        help="Directory for storing checkpoints",
    )
    run_parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum retries for execution failures",
    )
    run_parser.add_argument(
        "--force-retry-once",
        action="store_true",
        help="Force a single retry in the execute step for testing",
    )
    run_parser.add_argument(
        "--stop-after",
        default="",
        help="Pause the run after a given step name (e.g. 'plan')",
    )
    run_parser.add_argument(
        "--use-sandbox",
        action="store_true",
        help="Run tool commands inside a Docker sandbox",
    )
    run_parser.add_argument(
        "--sandbox-image",
        default="self-improving-agent:latest",
        help="Docker image to use for the sandbox",
    )
    run_parser.add_argument(
        "--sandbox-cpus",
        type=float,
        default=1.0,
        help="CPU limit for the sandbox container",
    )
    run_parser.add_argument(
        "--sandbox-memory",
        default="1g",
        help="Memory limit for the sandbox container",
    )
    run_parser.add_argument(
        "--sandbox-network",
        default="none",
        help="Network mode for the sandbox container",
    )
    run_parser.add_argument(
        "--sandbox-user",
        default="",
        help="User to run inside the container (uid:gid).",
    )
    run_parser.add_argument(
        "--repo-path",
        default=".",
        help="Repository path to mount into the sandbox.",
    )
    run_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply patch to target repo (otherwise workspace only).",
    )
    run_parser.add_argument(
        "--workspace-root",
        default=".workspaces",
        help="Root directory for patch workspaces.",
    )
    run_parser.add_argument(
        "--patch-output-dir",
        default="runs/patches",
        help="Directory to store generated patch files.",
    )
    run_parser.add_argument(
        "--patch-max-bytes",
        type=int,
        default=200_000,
        help="Maximum patch size in bytes.",
    )
    run_parser.add_argument(
        "--patch-blocked-prefix",
        action="append",
        default=[],
        help="Blocked path prefix for patches (can be repeated).",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output for each step.",
    )
    _add_rag_args(run_parser)
    _add_memory_args(run_parser)
    _add_graph_args(run_parser)
    run_parser.add_argument(
        "--llm-enabled",
        action="store_true",
        default=None,
        help="Enable LiteLLM calls (requires provider configuration).",
    )
    run_parser.add_argument(
        "--fast-model",
        default="",
        help="FAST model name for planning/retrieval.",
    )
    run_parser.add_argument(
        "--strong-model",
        default="",
        help="STRONG model name for patch/reflection.",
    )
    run_parser.add_argument(
        "--litellm-base-url",
        default="",
        help="Optional LiteLLM base URL.",
    )
    run_parser.add_argument(
        "--litellm-api-key",
        default="",
        help="Optional API key for LiteLLM.",
    )
    run_parser.add_argument(
        "--fast-temperature",
        type=float,
        default=None,
        help="Temperature for FAST model.",
    )
    run_parser.add_argument(
        "--strong-temperature",
        type=float,
        default=None,
        help="Temperature for STRONG model.",
    )
    run_parser.add_argument(
        "--litellm-timeout-s",
        type=int,
        default=None,
        help="Optional request timeout for LiteLLM.",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="Print run state as JSON",
    )

    index_parser.add_argument(
        "--repo-path",
        default=".",
        help="Repository path to index.",
    )
    _add_rag_args(index_parser)
    eval_parser.add_argument(
        "--goal",
        default="Fix failing tests",
        help="Goal description for the eval runs",
    )
    eval_parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of eval runs",
    )
    eval_parser.add_argument(
        "--max-iters",
        type=int,
        default=1,
        help="Maximum iterations per run",
    )
    eval_parser.add_argument(
        "--memory-backend",
        default="local",
        help="Memory backend for eval runs",
    )

    return parser


def _parse_command(value: str):
    if not value:
        return None
    value = value.strip()
    if _requires_shell(value):
        return value
    return shlex.split(value)


def _requires_shell(value: str) -> bool:
    if any(op in value for op in ["&&", "||", "|", ";", "<", ">", "$(", "`"]):
        return True
    parts = value.split()
    if not parts:
        return False
    first = parts[0]
    if "=" in first and not first.startswith("-"):
        name, _, _ = first.partition("=")
        if name and (name[0].isalpha() or name[0] == "_"):
            if all(char.isalnum() or char == "_" for char in name):
                return True
    return False


def main() -> None:
    _load_dotenv()
    configure_logging()
    configure_tracing()
    configure_metrics()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "doctor":
        from app.doctor import print_doctor_report, run_doctor

        results = run_doctor()
        sys.exit(print_doctor_report(results))

    if args.command == "runs":
        _list_runs()
        return

    if args.command == "run":
        _validate_cli_args(args)
        config = RunConfig(
            goal=args.goal,
            max_iters=args.max_iters,
            until_green=args.until_green,
            dry_run=args.dry_run,
            analysis_only=args.analysis_only,
            ruff_cmd=_parse_command(args.ruff_cmd),
            bandit_cmd=_parse_command(args.bandit_cmd),
            test_cmd=_parse_command(args.test_cmd),
            resume_run_id=args.resume_run_id or None,
            checkpoint_dir=args.checkpoint_dir,
            max_retries=args.max_retries,
            force_retry_once=args.force_retry_once,
            stop_after_step=args.stop_after or None,
            use_sandbox=args.use_sandbox,
            sandbox_image=args.sandbox_image,
            sandbox_cpus=args.sandbox_cpus,
            sandbox_memory=args.sandbox_memory,
            sandbox_network=args.sandbox_network,
            sandbox_user=args.sandbox_user or None,
            repo_path=args.repo_path,
            llm_enabled=args.llm_enabled,
            fast_model=args.fast_model or None,
            strong_model=args.strong_model or None,
            litellm_base_url=args.litellm_base_url or None,
            litellm_api_key=args.litellm_api_key or None,
            fast_temperature=args.fast_temperature,
            strong_temperature=args.strong_temperature,
            litellm_timeout_s=args.litellm_timeout_s,
            rag_backend=args.rag_backend or None,
            rag_top_k=args.rag_top_k,
            rag_index_dir=args.rag_index_dir or None,
            rag_pg_dsn=args.rag_pg_dsn or None,
            auto_index=_pick_bool_env(args.auto_index, "RAG_AUTO_INDEX", True),
            rag_chunk_size=args.rag_chunk_size,
            rag_chunk_overlap=args.rag_chunk_overlap,
            rag_lexical_top_k=args.rag_lexical_top_k,
            rag_vector_top_k=args.rag_vector_top_k,
            embedding_provider=args.embedding_provider or None,
            embedding_model=args.embedding_model or None,
            embedding_batch_size=args.embedding_batch_size,
            embedding_dim=args.embedding_dim,
            memory_backend=args.memory_backend or None,
            mem0_api_key=args.mem0_api_key or None,
            mem0_host=args.mem0_host or None,
            mem0_org_id=args.mem0_org_id or None,
            mem0_project_id=args.mem0_project_id or None,
            mem0_user_id=args.mem0_user_id or None,
            mem0_agent_id=args.mem0_agent_id or None,
            mem0_app_id=args.mem0_app_id or None,
            memory_top_k=args.memory_top_k,
            memory_fail_fast=args.memory_fail_fast,
            graph_backend=args.graph_backend or None,
            neo4j_uri=args.neo4j_uri or None,
            neo4j_user=args.neo4j_user or None,
            neo4j_password=args.neo4j_password or None,
            neo4j_database=args.neo4j_database or None,
            graph_fail_fast=args.graph_fail_fast,
            apply_to_repo=args.apply,
            workspace_root=args.workspace_root,
            patch_output_dir=args.patch_output_dir,
            patch_max_bytes=args.patch_max_bytes,
            patch_blocked_prefixes=args.patch_blocked_prefix or None,
        )

        # Print run header
        _print_header(config)

        try:
            state = run_agent_loop(config)
        except ValueError as exc:
            print(f"Configuration error: {exc}")
            sys.exit(2)

        # Save artifacts
        from app.run_artifacts import save_run_artifacts

        patch_text = _extract_patch_text(state)
        workspace_path = _extract_workspace_path(state)
        artifacts = save_run_artifacts(state, patch_text, workspace_path)

        if args.json:
            print(json.dumps(state.to_dict(), indent=2))
            return

        # Print rich summary
        _print_summary(state, artifacts, verbose=args.verbose)
        return

    if args.command == "index":
        config = RunConfig(
            goal="index_repo",
            repo_path=args.repo_path,
            rag_backend=args.rag_backend or None,
            rag_top_k=args.rag_top_k,
            rag_index_dir=args.rag_index_dir or None,
            rag_pg_dsn=args.rag_pg_dsn or None,
            rag_chunk_size=args.rag_chunk_size,
            rag_chunk_overlap=args.rag_chunk_overlap,
            rag_lexical_top_k=args.rag_lexical_top_k,
            rag_vector_top_k=args.rag_vector_top_k,
            embedding_provider=args.embedding_provider or None,
            embedding_model=args.embedding_model or None,
            embedding_batch_size=args.embedding_batch_size,
            embedding_dim=args.embedding_dim,
        )
        rag_config = resolve_rag_config(config)
        stats = index_repo(rag_config)
        print(f"Indexed {stats.chunk_count} chunks using {stats.backend}.")
        return

    if args.command == "eval":
        from eval.runner import run_eval_from_cli

        run_eval_from_cli(
            goal=args.goal,
            runs=args.runs,
            max_iters=args.max_iters,
            memory_backend=args.memory_backend,
        )
        return


def _print_header(config: RunConfig) -> None:
    """Print run start header."""
    print()
    print("=== Self-Improving Coding Agent ===")
    print(f"Goal: {config.goal}")
    print(f"Repo: {config.repo_path}")
    print(f"Max iterations: {config.max_iters}")
    if config.analysis_only:
        print("Mode: analysis-only")
    elif config.until_green:
        print("Mode: until-green")
    _print_preflight_warnings(config)
    print()


def _validate_cli_args(args: argparse.Namespace) -> None:
    errors = []
    if args.dry_run:
        errors.append("dry-run mode is disabled. Remove --dry-run.")
    if args.until_green and args.analysis_only:
        errors.append("--until-green requires a full run (disable --analysis-only).")
    if not args.analysis_only:
        if not args.test_cmd:
            errors.append("Missing --test-cmd.")
        if not args.ruff_cmd:
            errors.append("Missing --ruff-cmd.")
        if not args.bandit_cmd:
            errors.append("Missing --bandit-cmd.")
    llm_enabled = args.llm_enabled
    if llm_enabled is None:
        llm_enabled = os.getenv("LLM_ENABLED", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    if not llm_enabled:
        errors.append("LLM is disabled. Set LLM_ENABLED=1.")
    memory_backend = _resolve_backend(args.memory_backend, "MEMORY_BACKEND", "mem0")
    graph_backend = _resolve_backend(args.graph_backend, "GRAPH_BACKEND", "neo4j")
    if memory_backend == "local":
        errors.append("Local memory backend is not allowed. Configure MEM0 settings.")
    if graph_backend == "local":
        errors.append("Local graph backend is not allowed. Configure NEO4J settings.")
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print("Run aborted.")
        sys.exit(2)


def _print_preflight_warnings(config: RunConfig) -> None:
    warnings = []
    if not config.analysis_only and not config.test_cmd:
        warnings.append("No test command configured (use --test-cmd).")
    if not config.analysis_only and not config.ruff_cmd and not config.bandit_cmd:
        warnings.append("No static checks configured (use --ruff-cmd/--bandit-cmd).")
    if not config.auto_index:
        warnings.append("Auto-index disabled (missing/stale RAG index will fail).")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")


def _resolve_backend(override: str, env_key: str, default: str) -> str:
    if override:
        return override
    return os.getenv(env_key, default)


def _pick_bool_env(override: Optional[bool], env_key: str, default: bool) -> bool:
    if override is not None:
        return override
    raw = os.getenv(env_key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _print_summary(state, artifacts, verbose: bool = False) -> None:
    """Print rich run summary."""
    from app.run_artifacts import RunArtifacts

    artifacts: RunArtifacts = artifacts
    status_label = {"completed": "OK", "failed": "FAILED", "running": "RUNNING"}.get(
        state.status, "UNKNOWN"
    )

    print()
    print("=== Run Complete ===")
    print(f"Run ID: {state.run_id}")
    print(f"Status: {state.status} ({status_label})")

    duration = state.metrics.get("run_duration_ms")
    if duration:
        print(f"Duration: {duration}ms")

    print()

    # Steps summary
    for iteration in state.iterations:
        print(f"Iteration {iteration.index}:")
        for step in iteration.steps:
            label = {"ok": "OK", "skipped": "SKIP", "error": "ERROR"}.get(step.status, step.status)
            step_line = f"  - {step.name}: {label}"
            if verbose and step.details:
                detail_str = _brief_details(step.details)
                if detail_str:
                    step_line += f" ({detail_str})"
            print(step_line)

    print()
    print("Artifacts:")
    print(f"  Summary: {artifacts.summary_path}")
    if artifacts.patch_path:
        print(f"  Patch:   {artifacts.patch_path}")
    if artifacts.workspace_path:
        print(f"  Workspace: {artifacts.workspace_path}")

    print()
    print(f"Full report: {artifacts.summary_path}")
    print()


def _brief_details(details: dict) -> str:
    """Extract brief detail string."""
    if "model" in details:
        return f"model={details['model']}"
    if "exit_code" in details:
        return f"exit={details['exit_code']}"
    if "applied" in details:
        return f"applied={details['applied']}"
    if "error" in details:
        return f"error={details['error']}"
    return ""


def _extract_patch_text(state) -> str | None:
    """Extract patch text from state iterations."""
    for iteration in state.iterations:
        for step in iteration.steps:
            if step.name == "patch" and step.details:
                # The patch text is stored in state, not step details
                pass
    # Check checkpoint for patch_text
    checkpoint_path = Path(".checkpoints") / f"{state.run_id}.json"
    if checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text())
        return data.get("state", {}).get("patch_text")
    return None


def _extract_workspace_path(state) -> str | None:
    """Extract workspace path from state."""
    for iteration in state.iterations:
        for step in iteration.steps:
            if step.name == "apply_patch" and step.details:
                return step.details.get("workspace_path")
    return None


def _list_runs() -> None:
    """List recent runs with summaries."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("No runs found.")
        return

    # Find run directories (exclude patches subdirectory)
    run_dirs = []
    for item in runs_dir.iterdir():
        if item.is_dir() and item.name != "patches":
            summary = item / "summary.md"
            state_file = item / "state.json"
            if summary.exists() or state_file.exists():
                run_dirs.append(item)

    if not run_dirs:
        print("No runs found. Run 'python main.py run --goal \"...\"' to start.")
        return

    # Sort by modification time
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    print()
    print("Recent Runs")
    print("=" * 60)

    for run_dir in run_dirs[:10]:
        run_id = run_dir.name
        state_file = run_dir / "state.json"

        if state_file.exists():
            data = json.loads(state_file.read_text())
            status = data.get("status", "?")
            goal = data.get("goal", "")[:40]
            created = data.get("created_at", "")[:19]
            status_label = {"completed": "OK", "failed": "FAILED", "running": "RUNNING"}.get(
                status, "UNKNOWN"
            )
            print(f"{run_id[:36]} | {status_label:<8} | {goal}")
            print(f"   Created: {created} | Summary: {run_dir / 'summary.md'}")
        else:
            print(f"{run_id}")

        print()

    print(f"Total: {len(run_dirs)} runs")
    print()


if __name__ == "__main__":
    main()


def _add_rag_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rag-backend",
        default="",
        help="RAG backend: faiss or pgvector.",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=None,
        help="Total chunks to return after merging.",
    )
    parser.add_argument(
        "--rag-index-dir",
        default="",
        help="Directory for FAISS index files.",
    )
    parser.add_argument(
        "--rag-pg-dsn",
        default="",
        help="Postgres DSN for pgvector backend.",
    )
    parser.add_argument(
        "--rag-chunk-size",
        type=int,
        default=None,
        help="Chunk size for indexing.",
    )
    parser.add_argument(
        "--rag-chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap for indexing.",
    )
    parser.add_argument(
        "--rag-lexical-top-k",
        type=int,
        default=None,
        help="Max files returned from lexical recall.",
    )
    parser.add_argument(
        "--rag-vector-top-k",
        type=int,
        default=None,
        help="Max vector results before filtering.",
    )
    parser.add_argument(
        "--embedding-provider",
        default="",
        help="Embedding provider (litellm or mock).",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Batch size for embedding requests.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding vector dimension.",
    )


def _add_memory_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--memory-backend",
        default="",
        help="Memory backend: mem0 or local.",
    )
    parser.add_argument(
        "--mem0-api-key",
        default="",
        help="Mem0 API key (overrides MEM0_API_KEY).",
    )
    parser.add_argument(
        "--mem0-host",
        default="",
        help="Mem0 API host (overrides MEM0_HOST).",
    )
    parser.add_argument(
        "--mem0-org-id",
        default="",
        help="Mem0 organization ID.",
    )
    parser.add_argument(
        "--mem0-project-id",
        default="",
        help="Mem0 project ID.",
    )
    parser.add_argument(
        "--mem0-user-id",
        default="",
        help="Mem0 user ID override.",
    )
    parser.add_argument(
        "--mem0-agent-id",
        default="",
        help="Mem0 agent ID override.",
    )
    parser.add_argument(
        "--mem0-app-id",
        default="",
        help="Mem0 app ID override.",
    )
    parser.add_argument(
        "--memory-top-k",
        type=int,
        default=None,
        help="Top-k memories to retrieve.",
    )
    parser.add_argument(
        "--memory-fail-fast",
        action="store_true",
        default=None,
        help="Fail fast on memory errors.",
    )
    parser.add_argument(
        "--no-memory-fail-fast",
        action="store_false",
        dest="memory_fail_fast",
        help="Allow memory errors without failing the run.",
    )


def _add_graph_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--graph-backend",
        default="",
        help="Graph backend: neo4j or local.",
    )
    parser.add_argument(
        "--neo4j-uri",
        default="",
        help="Neo4j URI (e.g. bolt://localhost:7687).",
    )
    parser.add_argument(
        "--neo4j-user",
        default="",
        help="Neo4j username.",
    )
    parser.add_argument(
        "--neo4j-password",
        default="",
        help="Neo4j password.",
    )
    parser.add_argument(
        "--neo4j-database",
        default="",
        help="Neo4j database name (optional).",
    )
    parser.add_argument(
        "--graph-fail-fast",
        action="store_true",
        default=None,
        help="Fail fast on graph errors.",
    )
    parser.add_argument(
        "--no-graph-fail-fast",
        action="store_false",
        dest="graph_fail_fast",
        help="Allow graph errors without failing the run.",
    )
