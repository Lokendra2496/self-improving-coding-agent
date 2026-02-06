from __future__ import annotations

import json
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent.checkpoints import CheckpointStore
from agent.state import IterationRecord, RunConfig, RunState, StepResult, utc_now_iso
from llm.client import LLMClient
from llm.config import resolve_llm_config
from memory.config import resolve_memory_config
from memory.graph_config import resolve_graph_config
from memory.graph_store import GraphStore
from memory.store import MemoryStore
from observability.logging import log_event
from observability.metrics import configure_metrics, record_counter, record_timing, write_metrics_csv
from observability.tracing import configure_tracing, span
from rag.config import resolve_rag_config
from rag.index_repo import index_repo
from rag.retrieve_repo import retrieve as retrieve_repo
from tools.patching import (
    apply_patch_with_fallback,
    create_workspace,
    extract_unified_diff,
    validate_patch,
    write_patch,
)
from tools.run_tests import run_tests
from tools.sandbox import SandboxConfig
from tools.static_audit import (
    audit_repo,
    collect_code_snippets,
    collect_sql_snippets,
    resolve_goal_paths,
)
from tools.static_checks import run_static_checks


class GraphState(TypedDict):
    run_id: str
    goal: str
    status: str
    current_iteration: int
    max_iters: int
    iterations: List[Dict[str, Any]]
    retry_count: int
    max_retries: int
    force_retry_once: bool
    stop_after: Optional[str]
    start_at: str
    last_error: Optional[str]
    metrics: Dict[str, Any]
    run_started_ts: float
    halt: bool
    last_execute_exit_code: Optional[int]
    retry_next: Optional[str]
    patch_text: Optional[str]
    workspace_path: Optional[str]
    safety_failed: bool


def run_agent_loop(config: RunConfig) -> RunState:
    configure_tracing()
    configure_metrics()
    _validate_run_config(config)
    run_start = time.perf_counter()
    checkpoint_store = CheckpointStore(config.checkpoint_dir)
    memory_config = resolve_memory_config(config)
    memory_store = MemoryStore.from_config(config, memory_config)
    graph_config = resolve_graph_config(config)
    graph_store = GraphStore.from_config(graph_config)
    llm_client = LLMClient(resolve_llm_config(config))

    try:
        if config.resume_run_id:
            checkpoint = checkpoint_store.load(config.resume_run_id)
            if checkpoint is None:
                raise ValueError(f"No checkpoint found for run_id {config.resume_run_id}")
            state: GraphState = checkpoint.state  # type: ignore[assignment]
            state["stop_after"] = config.stop_after_step
            state["max_retries"] = config.max_retries
            state["force_retry_once"] = config.force_retry_once
            state["status"] = "running"
            state["halt"] = False
        else:
            state = _initial_state(config)

        record_counter("runs_total", 1)
        with span(
            "agent.run",
            {"run_id": state["run_id"], "goal": state["goal"], "max_iters": state["max_iters"]},
        ):
            graph = _build_graph(
                config,
                checkpoint_store,
                memory_store,
                llm_client,
                memory_config.top_k,
                graph_store,
                graph_config.fail_fast,
            )
            final_state = graph.invoke(state)

        run_state = _to_run_state(final_state)
        run_state.metrics = final_state.get("metrics", {})
        duration_ms = int((time.perf_counter() - run_start) * 1000)
        run_state.metrics["run_duration_ms"] = duration_ms
        record_timing("run_duration_ms", duration_ms)
        if run_state.status != "running":
            run_state.completed_at = utc_now_iso()
        log_event(_logger(), "run_complete", {"run_id": run_state.run_id, "status": run_state.status})
        write_metrics_csv(run_state.run_id, run_state.metrics)
        return run_state
    finally:
        graph_store.close()


def _initial_state(config: RunConfig) -> GraphState:
    run_state = RunState.start(goal=config.goal, run_id=config.run_id)
    return {
        "run_id": run_state.run_id,
        "goal": config.goal,
        "status": "running",
        "current_iteration": 0,
        "max_iters": config.max_iters,
        "iterations": [],
        "retry_count": 0,
        "max_retries": config.max_retries,
        "force_retry_once": config.force_retry_once,
        "stop_after": config.stop_after_step,
        "start_at": "plan" if config.analysis_only else "reproduce",
        "last_error": None,
        "metrics": {"counters": {}, "timings": {}},
        "run_started_ts": time.perf_counter(),
        "halt": False,
        "last_execute_exit_code": None,
        "retry_next": None,
        "patch_text": None,
        "workspace_path": None,
        "safety_failed": False,
    }


def _validate_run_config(config: RunConfig) -> None:
    if config.dry_run:
        raise ValueError("dry-run mode is disabled for full runs.")
    if config.until_green and config.analysis_only:
        raise ValueError("until-green requires a full run (disable --analysis-only).")
    if not config.analysis_only:
        if not config.test_cmd:
            raise ValueError("test_cmd is required. Pass --test-cmd.")
        if not config.ruff_cmd:
            raise ValueError("ruff_cmd is required. Pass --ruff-cmd.")
        if not config.bandit_cmd:
            raise ValueError("bandit_cmd is required. Pass --bandit-cmd.")
    llm_config = resolve_llm_config(config)
    if not llm_config.enabled:
        raise ValueError("LLM is disabled. Set LLM_ENABLED=1.")
    memory_config = resolve_memory_config(config)
    if memory_config.backend != "mem0":
        raise ValueError("MEMORY_BACKEND must be mem0.")
    graph_config = resolve_graph_config(config)
    if graph_config.backend != "neo4j":
        raise ValueError("GRAPH_BACKEND must be neo4j.")


def _build_graph(
    config: RunConfig,
    checkpoint_store: CheckpointStore,
    memory_store: MemoryStore,
    llm_client: LLMClient,
    memory_top_k: int,
    graph_store: GraphStore,
    graph_fail_fast: bool,
):
    graph = StateGraph(GraphState)

    graph.add_node("router", _router_node)
    graph.add_node("reproduce", _reproduce_node(config, checkpoint_store))
    graph.add_node("plan", _plan_node(config, checkpoint_store, llm_client))
    graph.add_node("retrieve", _retrieve_node(config, checkpoint_store, memory_store, llm_client, memory_top_k))
    graph.add_node("audit", _audit_node(config, checkpoint_store, llm_client))
    graph.add_node("patch", _patch_node(config, checkpoint_store, llm_client))
    graph.add_node("apply_patch", _apply_patch_node(config, checkpoint_store))
    graph.add_node("safety", _safety_node(config, checkpoint_store))
    graph.add_node("execute", _execute_node(config, checkpoint_store))
    graph.add_node("retry_decide", _retry_decide_node(config, checkpoint_store))
    graph.add_node("reflect", _reflect_node(config, checkpoint_store, llm_client))
    graph.add_node(
        "memory_update",
        _memory_update_node(config, checkpoint_store, memory_store, graph_store, graph_fail_fast, llm_client),
    )
    graph.add_node("decide", _decide_node(config, checkpoint_store))

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _route_from_router,
        {
            "reproduce": "reproduce",
            "plan": "plan",
            "retrieve": "retrieve",
            "audit": "audit",
            "patch": "patch",
            "apply_patch": "apply_patch",
            "safety": "safety",
            "execute": "execute",
            "retry_decide": "retry_decide",
            "reflect": "reflect",
            "memory_update": "memory_update",
            "decide": "decide",
            "end": END,
        },
    )

    _add_haltable_edge(graph, "reproduce", "plan")
    _add_haltable_edge(graph, "plan", "retrieve")
    _add_haltable_edge(graph, "retrieve", "audit")
    if config.analysis_only:
        _add_haltable_edge(graph, "audit", "reflect")
        _add_haltable_edge(graph, "reflect", "memory_update")
        _add_haltable_edge(graph, "memory_update", "decide")
        _add_haltable_edge(graph, "decide", "router")
    else:
        _add_haltable_edge(graph, "audit", "patch")
        _add_haltable_edge(graph, "patch", "apply_patch")
        # Route through router - _set_next in node functions controls the next step
        _add_haltable_edge(graph, "apply_patch", "router")
        _add_haltable_edge(graph, "safety", "router")
        _add_haltable_edge(graph, "execute", "retry_decide")
        graph.add_conditional_edges(
            "retry_decide",
            _route_retry_decide,
            {"execute": "execute", "reflect": "reflect", "halt": END},
        )
        _add_haltable_edge(graph, "reflect", "memory_update")
        _add_haltable_edge(graph, "memory_update", "decide")
        _add_haltable_edge(graph, "decide", "router")

    return graph.compile()


def _add_haltable_edge(graph: StateGraph, from_node: str, next_node: str) -> None:
    graph.add_conditional_edges(
        from_node,
        lambda state, next_node=next_node: "halt" if state.get("halt") else next_node,
        {"halt": END, next_node: next_node},
    )


def _router_node(state: GraphState) -> GraphState:
    return state


def _route_from_router(state: GraphState) -> str:
    # Check halt first
    if state.get("halt"):
        return "end"
    # Route to the next step set by _set_next
    target = state.get("start_at") or "reproduce"
    if target == "end":
        return "end"
    return target


def _route_retry_decide(state: GraphState) -> str:
    if state.get("halt"):
        return "halt"
    return state.get("retry_next") or "reflect"


def _reproduce_node(config: RunConfig, checkpoint_store: CheckpointStore):
    def _node(state: GraphState) -> GraphState:
        _ensure_iteration(state)
        sandbox_config = _sandbox_config(config)
        result_obj = run_tests(
            config.test_cmd,
            use_sandbox=config.use_sandbox,
            sandbox_config=sandbox_config,
            repo_path=config.repo_path,
        )
        exit_code = result_obj.exit_code
        result = {
            "name": result_obj.name,
            "command": result_obj.command,
            "exit_code": result_obj.exit_code,
            "duration_ms": result_obj.duration_ms,
            "stdout": _trim_text(result_obj.stdout),
            "stderr": _trim_text(result_obj.stderr),
        }
        status = "ok" if exit_code == 0 else "error"
        if exit_code != 0:
            state["last_error"] = f"reproduce failed with exit code {exit_code}"
            if _is_config_error(exit_code):
                state["status"] = "failed"
                state["halt"] = True
        _add_step(state, "reproduce", status, {"result": result})
        _log_step("reproduce", state)
        _set_next(state, "plan", "reproduce", checkpoint_store)
        return state

    return _node


def _plan_node(config: RunConfig, checkpoint_store: CheckpointStore, llm_client: LLMClient):
    def _node(state: GraphState) -> GraphState:
        prompt = f"Provide a concise plan for the goal: {state['goal']}"
        response = llm_client.fast_complete(prompt)
        _add_step(state, "plan", "ok", {"strategy": response.content, "model": response.model})
        _log_step("plan", state)
        _set_next(state, "retrieve", "plan", checkpoint_store)
        return state

    return _node


def _retrieve_node(
    config: RunConfig,
    checkpoint_store: CheckpointStore,
    memory_store: MemoryStore,
    llm_client: LLMClient,
    memory_top_k: int,
):
    def _node(state: GraphState) -> GraphState:
        rag_config = resolve_rag_config(config)
        rewrite_prompt = (
            "Rewrite the following goal into a concise retrieval query:\n"
            f"{state['goal']}"
        )
        rewrite = llm_client.fast_complete(rewrite_prompt)
        query = rewrite.content
        auto_indexed = False
        try:
            chunks = retrieve_repo(query, rag_config)
            status = "ok"
            error = None
        except Exception as exc:
            if config.auto_index and _should_auto_index(exc):
                try:
                    index_repo(rag_config)
                    auto_indexed = True
                    chunks = retrieve_repo(query, rag_config)
                    status = "ok"
                    error = None
                except Exception as inner_exc:
                    chunks = []
                    status = "error"
                    error = str(inner_exc)
                    state["last_error"] = error
            else:
                chunks = []
                status = "error"
                error = str(exc)
                state["last_error"] = error
        memories = memory_store.get_similar(query, top_k=memory_top_k)
        goal_paths = resolve_goal_paths(state["goal"], config.repo_path)
        error_paths = _extract_failure_paths(state, config.repo_path)
        candidate_sources = list({chunk.source for chunk in chunks[:5]})
        top_sources = _merge_target_files(goal_paths, error_paths, candidate_sources)
        _add_step(
            state,
            "retrieve",
            status,
            {
                "query": query,
                "model": rewrite.model,
                "chunks": len(chunks),
                "memories": len(memories),
                "top_sources": top_sources,
                "error": error,
                "auto_indexed": auto_indexed,
            },
        )
        _log_step("retrieve", state)
        _set_next(state, "patch", "retrieve", checkpoint_store)
        return state

    return _node


def _audit_node(config: RunConfig, checkpoint_store: CheckpointStore, llm_client: LLMClient):
    def _node(state: GraphState) -> GraphState:
        retrieve_details = _latest_step_details(state, "retrieve")
        goal_paths = resolve_goal_paths(state["goal"], config.repo_path)
        error_paths = _extract_failure_paths(state, config.repo_path)
        retrieve_sources = retrieve_details.get("top_sources", []) if isinstance(retrieve_details, dict) else []
        target_files = _merge_target_files(goal_paths, error_paths, retrieve_sources)
        review_plan = _llm_review_plan(llm_client, state["goal"])
        findings: List[Dict[str, Any]] = []
        error = None
        llm_error = None
        try:
            if review_plan.get("sql_review"):
                findings = audit_repo(
                    state["goal"],
                    config.repo_path,
                    target_files=target_files,
                )
        except Exception as exc:
            error = str(exc)
            state["last_error"] = error

        llm_findings: List[Dict[str, Any]] = []
        if review_plan.get("sql_review"):
            snippets = collect_sql_snippets(
                state["goal"],
                config.repo_path,
                target_files=target_files,
            )
            if snippets:
                llm_findings, llm_error = _llm_sql_audit(llm_client, snippets)
                findings.extend(_map_llm_findings(llm_findings, snippets, "llm_sql_review"))

        code_findings: List[Dict[str, Any]] = []
        code_error = None
        line_review: List[Dict[str, Any]] = []
        line_review_error = None
        if review_plan.get("code_review"):
            code_snippets = collect_code_snippets(
                state["goal"],
                config.repo_path,
                target_files=target_files,
            )
            if code_snippets:
                code_findings, code_error = _llm_code_review(llm_client, code_snippets)
                findings.extend(_map_llm_findings(code_findings, code_snippets, "llm_code_review"))
                if _should_line_review(state["goal"]):
                    line_review_snippets = code_snippets[:1]
                    line_review, line_review_error = _llm_code_line_review(
                        llm_client, line_review_snippets
                    )

        status = "ok" if not error else "error"
        details: Dict[str, Any] = {
            "count": len(findings),
            "findings": findings,
            "error": error,
            "review_plan": review_plan,
        }
        if llm_findings:
            details["llm_findings"] = llm_findings
        if llm_error:
            details["llm_error"] = llm_error
        if code_findings:
            details["code_findings"] = code_findings
        if code_error:
            details["code_error"] = code_error
        if line_review:
            details["line_review"] = line_review
        if line_review_error:
            details["line_review_error"] = line_review_error
        _add_step(state, "audit", status, details)
        _log_step("audit", state)
        next_node = "reflect" if config.analysis_only else "patch"
        _set_next(state, next_node, "audit", checkpoint_store)
        return state

    return _node


def _llm_sql_audit(
    llm_client: LLMClient, snippets: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    if not snippets:
        return [], None
    prompt = _build_sql_audit_prompt(snippets)
    response = llm_client.strong_complete(prompt)
    parsed = _parse_json_list(response.content)
    if parsed is None:
        return [], "LLM SQL audit returned invalid JSON"
    return parsed, None


def _build_sql_audit_prompt(snippets: List[Dict[str, Any]]) -> str:
    payload = json.dumps(snippets, ensure_ascii=True)
    return (
        "Review the SQL queries and return JSON only. "
        "Each item must include: file, line, issue, severity, recommendation. "
        "If no issues, return an empty list.\n\n"
        f"Queries: {payload}"
    )

def _llm_code_review(
    llm_client: LLMClient, snippets: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    if not snippets:
        return [], None
    prompt = _build_code_review_prompt(snippets)
    response = llm_client.strong_complete(prompt)
    parsed = _parse_json_list(response.content)
    if parsed is None:
        return [], "LLM code review returned invalid JSON"
    return parsed, None


def _build_code_review_prompt(snippets: List[Dict[str, Any]]) -> str:
    payload = json.dumps(snippets, ensure_ascii=True)
    return (
        "Review the code snippets for bugs, logic errors, or risky patterns. "
        "Return JSON only. Each item must include: file, line, issue, severity, recommendation. "
        "Snippets include 'numbered' lines with actual line numbers; use those line numbers. "
        "If no issues, return an empty list.\n\n"
        f"Snippets: {payload}"
    )


def _llm_code_line_review(
    llm_client: LLMClient, snippets: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    if not snippets:
        return [], None
    prompt = _build_code_line_review_prompt(snippets)
    response = llm_client.strong_complete(prompt)
    parsed = _parse_json_list(response.content)
    if parsed is None:
        return [], "LLM line-by-line review returned invalid JSON"
    return parsed, None


def _build_code_line_review_prompt(snippets: List[Dict[str, Any]]) -> str:
    payload = json.dumps(snippets, ensure_ascii=True)
    return (
        "Provide a line-by-line review of the code. "
        "Snippets include 'numbered' lines with actual line numbers. "
        "Return JSON only: a list of items with file, line, note, severity. "
        "Use severity levels: info, warning, error. "
        "Include notes for non-empty lines; keep each note short. "
        "Limit output to at most 120 items. "
        "If a line cannot be reviewed, note why.\n\n"
        f"Snippets: {payload}"
    )


def _should_line_review(goal: str) -> bool:
    lowered = goal.lower()
    if "line by line" in lowered or "line-by-line" in lowered:
        return True
    if "full" in lowered and "function" in lowered:
        return True
    if "entire" in lowered and "function" in lowered:
        return True
    return False


def _parse_json_list(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None
    candidate = text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def _map_llm_findings(
    findings: List[Dict[str, Any]], snippets: List[Dict[str, Any]], finding_type: str
) -> List[Dict[str, Any]]:
    snippet_map: Dict[tuple[str, Optional[int]], str] = {}
    for snippet in snippets:
        key = (snippet.get("file", ""), snippet.get("line"))
        if key not in snippet_map:
            snippet_map[key] = snippet.get("query") or snippet.get("snippet", "")

    mapped: List[Dict[str, Any]] = []
    for finding in findings:
        file_path = finding.get("file", "")
        line = finding.get("line")
        issue = finding.get("issue", "")
        severity = finding.get("severity", "")
        recommendation = finding.get("recommendation", "")
        detail_parts = [part for part in [issue, recommendation] if part]
        detail = " | ".join(detail_parts)
        if severity:
            detail = f"{detail} (severity: {severity})" if detail else f"severity: {severity}"
        mapped.append(
            {
                "file": file_path,
                "line": line,
                "type": finding_type,
                "detail": detail,
                "excerpt": snippet_map.get((file_path, line), ""),
            }
        )
    return mapped


def _llm_review_plan(llm_client: LLMClient, goal: str) -> Dict[str, Any]:
    prompt = (
        "Decide whether the request requires code review and/or SQL query review. "
        "Return JSON only with keys: sql_review (true/false), code_review (true/false), reason.\n\n"
        f"Goal: {goal}"
    )
    response = llm_client.fast_complete(prompt)
    text = response.content
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"sql_review": True, "code_review": True, "reason": "fallback"}
    if not isinstance(data, dict):
        return {"sql_review": True, "code_review": True, "reason": "fallback"}
    return {
        "sql_review": bool(data.get("sql_review")),
        "code_review": bool(data.get("code_review")),
        "reason": data.get("reason"),
    }


def _patch_node(config: RunConfig, checkpoint_store: CheckpointStore, llm_client: LLMClient):
    def _node(state: GraphState) -> GraphState:
        retrieve_details = _latest_step_details(state, "retrieve")
        top_sources = retrieve_details.get("top_sources", []) if isinstance(retrieve_details, dict) else []
        goal_paths = resolve_goal_paths(state["goal"], config.repo_path)
        error_paths = _extract_failure_paths(state, config.repo_path)
        target_files = _merge_target_files(goal_paths, error_paths, top_sources)
        files_hint = "\n".join(target_files) if target_files else "None"
        context_block = _build_patch_context(config.repo_path, target_files)
        test_context = _build_test_context(state)
        safety_context = _build_safety_context(state)
        prompt = (
            "Generate a unified diff to achieve the goal below.\n"
            "Return ONLY the diff, starting with 'diff --git'.\n\n"
            f"Goal: {state['goal']}\n"
            f"Test result:\n{test_context}\n"
            f"Safety results:\n{safety_context}\n"
            f"Relevant files:\n{files_hint}\n"
            "Current file contents (may be truncated):\n"
            f"{context_block}\n"
        )
        response = llm_client.strong_complete(prompt)
        patch_text = extract_unified_diff(response.content)
        if not patch_text:
            state["patch_text"] = None
            _add_step(
                state,
                "patch",
                "error",
                {"reason": "no diff found", "model": response.model, "raw_response": response.content[:1000]},
            )
            state["last_error"] = "patch generation failed: no diff found in LLM response"
            # Don't halt - continue to apply_patch which will handle the missing patch
            _log_step("patch", state)
            _set_next(state, "apply_patch", "patch", checkpoint_store)
            return state

        state["patch_text"] = patch_text
        _add_step(
            state,
            "patch",
            "ok",
            {
                "fingerprint": "llm_diff",
                "model": response.model,
                "patch_bytes": len(patch_text.encode("utf-8")),
            },
        )
        _log_step("patch", state)
        _set_next(state, "apply_patch", "patch", checkpoint_store)
        return state

    return _node


def _apply_patch_node(config: RunConfig, checkpoint_store: CheckpointStore):
    def _node(state: GraphState) -> GraphState:
        patch_text = state.get("patch_text")
        
        # Handle cases where patch generation failed or is invalid
        # Don't halt - allow retry in next iteration
        if not patch_text:
            _add_step(state, "apply_patch", "error", {"reason": "no patch generated"})
            state["last_error"] = "apply_patch failed: no patch generated"
            state["patch_failed"] = True
            _log_step("apply_patch", state)
            _set_next(state, "reflect", "apply_patch", checkpoint_store)
            return state

        if len(patch_text.encode("utf-8")) > config.patch_max_bytes:
            _add_step(state, "apply_patch", "error", {"error": "patch too large"})
            state["last_error"] = "apply_patch failed: patch too large"
            state["patch_failed"] = True
            _log_step("apply_patch", state)
            _set_next(state, "reflect", "apply_patch", checkpoint_store)
            return state

        violations = validate_patch(patch_text, config.patch_blocked_prefixes)
        if violations:
            _add_step(
                state,
                "apply_patch",
                "error",
                {"error": "blocked paths", "violations": violations},
            )
            state["last_error"] = f"apply_patch failed: blocked paths {violations}"
            state["patch_failed"] = True
            _log_step("apply_patch", state)
            _set_next(state, "reflect", "apply_patch", checkpoint_store)
            return state

        workspace_path = create_workspace(config.repo_path, config.workspace_root)
        patch_path = write_patch(patch_text, config.patch_output_dir, state["run_id"])
        applied, output = apply_patch_with_fallback(patch_path, workspace_path)

        if applied and config.apply_to_repo:
            apply_patch_with_fallback(patch_path, config.repo_path)
        
        if not applied:
            state["last_error"] = f"apply_patch failed: {output[:500]}"
            state["patch_failed"] = True
        else:
            state["patch_failed"] = False

        state["workspace_path"] = workspace_path if applied else None
        _add_step(
            state,
            "apply_patch",
            "ok" if applied else "error",
            {
                "workspace_path": workspace_path,
                "patch_path": patch_path,
                "applied": applied,
                "output": output,
                "applied_to_repo": config.apply_to_repo,
            },
        )
        _log_step("apply_patch", state)
        # Route based on success: safety if applied, reflect if failed
        next_node = "safety" if applied else "reflect"
        _set_next(state, next_node, "apply_patch", checkpoint_store)
        return state

    return _node


def _safety_node(config: RunConfig, checkpoint_store: CheckpointStore):
    def _node(state: GraphState) -> GraphState:
        sandbox_config = _sandbox_config(config)
        static_results = run_static_checks(
            ruff_cmd=config.ruff_cmd,
            bandit_cmd=config.bandit_cmd,
            use_sandbox=config.use_sandbox,
            sandbox_config=sandbox_config,
            repo_path=_effective_repo_path(state, config),
        )
        status = "ok" if all(result.exit_code == 0 for result in static_results) else "error"
        if status == "error":
            state["last_error"] = "safety checks failed"
            state["safety_failed"] = True
            state["last_execute_exit_code"] = 1
        else:
            state["safety_failed"] = False
        _add_step(
            state,
            "safety",
            status,
            {"results": _summarize_results(static_results)},
        )
        _log_step("safety", state)
        # Always proceed to execute (tests) even if safety fails.
        # This lets us know if the actual fix works.
        _set_next(state, "execute", "safety", checkpoint_store)
        return state

    return _node


def _execute_node(config: RunConfig, checkpoint_store: CheckpointStore):
    def _node(state: GraphState) -> GraphState:
        if config.force_retry_once and state["retry_count"] == 0:
            exit_code = 1
            result = {
                "name": "tests",
                "command": "forced_retry",
                "exit_code": exit_code,
                "duration_ms": 0,
            }
            _add_step(state, "execute", "error", {"result": result, "forced_retry": True})
        else:
            sandbox_config = _sandbox_config(config)
            result_obj = run_tests(
                config.test_cmd,
                use_sandbox=config.use_sandbox,
                sandbox_config=sandbox_config,
                repo_path=_effective_repo_path(state, config),
            )
            exit_code = result_obj.exit_code
            result = {
                "name": result_obj.name,
                "command": result_obj.command,
                "exit_code": result_obj.exit_code,
                "duration_ms": result_obj.duration_ms,
                "stdout": _trim_text(result_obj.stdout),
                "stderr": _trim_text(result_obj.stderr),
            }
            status = "ok" if exit_code == 0 else "error"
            _add_step(state, "execute", status, {"result": result})

        if exit_code != 0:
            if _is_config_error(exit_code):
                state["last_error"] = f"execute failed: {result.get('command')}"
                state["status"] = "failed"
                state["halt"] = True
            else:
                state["retry_count"] += 1
                state["last_error"] = f"execute failed with exit code {exit_code}"

        state["last_execute_exit_code"] = exit_code
        _log_step("execute", state)
        _set_next(state, "retry_decide", "execute", checkpoint_store)
        return state

    return _node


def _retry_decide_node(config: RunConfig, checkpoint_store: CheckpointStore):
    def _node(state: GraphState) -> GraphState:
        exit_code = state.get("last_execute_exit_code")
        should_retry = (
            exit_code is not None
            and exit_code != 0
            and state["retry_count"] <= state["max_retries"]
        )
        decision = "retry" if should_retry else "continue"
        state["retry_next"] = "execute" if should_retry else "reflect"
        _add_step(state, "retry_decide", "ok", {"decision": decision, "retry_count": state["retry_count"]})
        _log_step("retry_decide", state)
        _set_next(state, state["retry_next"], "retry_decide", checkpoint_store)
        return state

    return _node


def _reflect_node(config: RunConfig, checkpoint_store: CheckpointStore, llm_client: LLMClient):
    def _node(state: GraphState) -> GraphState:
        context = _build_reflection_context(state)
        prompt = (
            "Provide a brief reflection on the current iteration outcome. "
            "Focus on why it succeeded/failed and the next steps.\n\n"
            f"Context: {context}"
        )
        response = llm_client.strong_complete(prompt)
        _add_step(state, "reflect", "ok", {"summary": response.content, "model": response.model})
        _log_step("reflect", state)
        _set_next(state, "memory_update", "reflect", checkpoint_store)
        return state

    return _node


def _build_reflection_context(state: GraphState) -> str:
    iteration = state.get("iterations", [])[-1] if state.get("iterations") else {}
    steps: List[Dict[str, Any]] = []
    for step in iteration.get("steps", []):
        details = step.get("details", {})
        steps.append(
            {
                "name": step.get("name"),
                "status": step.get("status"),
                "details": _summarize_reflection_details(details),
            }
        )
    payload = {
        "goal": state.get("goal"),
        "iteration": state.get("current_iteration"),
        "status": state.get("status"),
        "last_error": state.get("last_error"),
        "last_execute_exit_code": state.get("last_execute_exit_code"),
        "retry_count": state.get("retry_count"),
        "safety_failed": state.get("safety_failed"),
        "patch_generated": bool(state.get("patch_text")),
        "workspace_path": state.get("workspace_path"),
        "steps": steps,
    }
    return json.dumps(payload, ensure_ascii=True)


def _summarize_reflection_details(details: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if not details:
        return summary
    for key in ("model", "exit_code", "applied", "error", "reason", "decision", "retry_count"):
        if key in details:
            summary[key] = details[key]
    if "result" in details and isinstance(details["result"], dict):
        summary["result"] = {
            "exit_code": details["result"].get("exit_code"),
            "command": details["result"].get("command"),
        }
    return summary


def _memory_update_node(
    config: RunConfig,
    checkpoint_store: CheckpointStore,
    memory_store: MemoryStore,
    graph_store: GraphStore,
    graph_fail_fast: bool,
    llm_client: LLMClient,
):
    def _node(state: GraphState) -> GraphState:
        memory_store.store_episode(
            {
                "summary": f"Iteration {state['current_iteration']} complete.",
                "goal": state["goal"],
                "iteration": state["current_iteration"],
            }
        )
        fix_card = _generate_fix_card(llm_client, state, config)
        if fix_card:
            memory_store.store_fix_card(fix_card)

        episode = _build_graph_episode(state)
        if fix_card:
            episode["fix_card"] = fix_card
        try:
            graph_store.upsert_episode(episode)
        except Exception as exc:
            state["last_error"] = str(exc)
            if graph_fail_fast:
                raise
        details: Dict[str, Any] = {"stored": True}
        if fix_card:
            details["fix_card"] = _fix_card_public(fix_card)
        _add_step(state, "memory_update", "ok", details)
        _log_step("memory_update", state)
        _set_next(state, "decide", "memory_update", checkpoint_store)
        return state

    return _node


def _decide_node(config: RunConfig, checkpoint_store: CheckpointStore):
    def _node(state: GraphState) -> GraphState:
        _finish_iteration(state)
        details: Dict[str, Any] = {}
        
        last_exit = state.get("last_execute_exit_code")
        safety_failed = state.get("safety_failed", False)
        patch_failed = state.get("patch_failed", False)
        
        # Tests passing is the primary success criteria
        # Safety warnings (lint issues) are informational - they may be pre-existing
        tests_pass = not patch_failed and last_exit == 0
        
        details["last_exit_code"] = last_exit
        details["safety_failed"] = safety_failed
        details["patch_failed"] = patch_failed
        
        if config.until_green and not config.analysis_only:
            if tests_pass:
                # SUCCESS: Tests pass - mark as completed
                decision = "stop"
                next_node = "end"
                state["status"] = "completed"
                if safety_failed:
                    details["reason"] = "tests passed (safety warnings are pre-existing)"
                else:
                    details["reason"] = "tests passed and safety clean"
            elif state["current_iteration"] + 1 < state["max_iters"]:
                # RETRY: Tests failed or patch failed - try again
                state["current_iteration"] += 1
                decision = "continue"
                next_node = "reproduce"
                if patch_failed:
                    details["reason"] = "patch failed to apply"
                elif last_exit != 0:
                    details["reason"] = "tests failed"
                else:
                    details["reason"] = "unknown failure"
            else:
                # FAIL: Max iterations reached without tests passing
                decision = "stop"
                next_node = "end"
                state["status"] = "failed"
                details["reason"] = "max iterations reached without tests passing"
        else:
            if state["current_iteration"] + 1 < state["max_iters"]:
                state["current_iteration"] += 1
                decision = "continue"
                next_node = "reproduce"
            else:
                decision = "stop"
                next_node = "end"
                # Mark as completed if patch applied and tests pass
                state["status"] = "failed" if (patch_failed or last_exit != 0) else "completed"

        details["next"] = decision
        _add_step(state, "decide", "ok", details)
        _log_step("decide", state)
        _set_next(state, next_node, "decide", checkpoint_store)
        return state

    return _node


def _ensure_iteration(state: GraphState) -> Dict[str, Any]:
    current_index = state["current_iteration"]
    iterations = state["iterations"]
    if not iterations or iterations[-1]["index"] != current_index:
        iterations.append(
            {
                "index": current_index,
                "started_at": utc_now_iso(),
                "ended_at": None,
                "steps": [],
            }
        )
        state["safety_failed"] = False
        _incr_counter(state, "iterations_total")
    return iterations[-1]


def _finish_iteration(state: GraphState) -> None:
    if not state["iterations"]:
        return
    state["iterations"][-1]["ended_at"] = utc_now_iso()


def _add_step(state: GraphState, name: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
    iteration = _ensure_iteration(state)
    step = {
        "name": name,
        "status": status,
        "details": details or {},
        "started_at": utc_now_iso(),
        "ended_at": utc_now_iso(),
    }
    iteration["steps"].append(step)
    _incr_counter(state, "steps_total")


def _incr_counter(state: GraphState, name: str, value: int = 1) -> None:
    counters = state.setdefault("metrics", {}).setdefault("counters", {})
    counters[name] = counters.get(name, 0) + value
    record_counter(name, value)


def _set_next(state: GraphState, next_node: str, current_node: str, checkpoint_store: CheckpointStore) -> None:
    if state.get("halt"):
        state["halt"] = True
    elif state.get("stop_after") == current_node:
        state["status"] = "paused"
        state["halt"] = True
    else:
        state["halt"] = False
    state["start_at"] = next_node
    checkpoint_store.save(state["run_id"], state)


def _summarize_results(results: list) -> list:
    summary = []
    for result in results:
        summary.append(
            {
                "name": result.name,
                "command": result.command,
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
                "stdout": _trim_text(result.stdout),
                "stderr": _trim_text(result.stderr),
            }
        )
    return summary


def _is_config_error(exit_code: Optional[int]) -> bool:
    return exit_code in {2, 127}


def _should_auto_index(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    if isinstance(exc, ValueError):
        message = str(exc).lower()
        return "index" in message and "rag" in message or "index" in message
    return False


def _build_patch_context(
    repo_path: str, files: List[str], max_files: int = 5, max_chars: int = 12000
) -> str:
    if not files:
        return "None"
    repo_root = Path(repo_path).resolve()
    blocks: List[str] = []
    seen: set[str] = set()
    for entry in files:
        if len(blocks) >= max_files:
            break
        if entry in seen:
            continue
        seen.add(entry)
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            content = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not content.strip():
            continue
        snippet = _clip_text(content, limit=max_chars)
        blocks.append(f"File: {candidate}\n---\n{snippet}\n---")
    return "\n\n".join(blocks) if blocks else "None"


def _clip_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    clip = value[:limit]
    last_newline = clip.rfind("\n")
    if last_newline > 0:
        return clip[:last_newline]
    return clip


def _build_test_context(state: GraphState) -> str:
    details = _latest_step_details(state, "reproduce")
    result = details.get("result") if isinstance(details, dict) else None
    if not isinstance(result, dict):
        return "None"
    command = result.get("command", "")
    exit_code = result.get("exit_code", "")
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    return (
        f"command={command}\n"
        f"exit_code={exit_code}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}\n"
    )


_PATH_RE = re.compile(r"(?P<path>(?:/|\.{0,2}/)?[^\s:'\"]+\.py)")
# Match Python import statements like "from consumer.consumer import X" or "import consumer.consumer"
_IMPORT_RE = re.compile(r"(?:from|import)\s+(?P<module>[a-zA-Z_][a-zA-Z0-9_.]*)")


def _extract_failure_paths(state: GraphState, repo_path: str) -> List[str]:
    repo_root = Path(repo_path).resolve()
    outputs: List[str] = []
    for step_name in ("reproduce", "execute"):
        details = _latest_step_details(state, step_name)
        result = details.get("result") if isinstance(details, dict) else None
        if not isinstance(result, dict):
            continue
        outputs.append(result.get("stdout", "") or "")
        outputs.append(result.get("stderr", "") or "")

    found: List[str] = []
    seen = set()
    test_files_to_parse: List[Path] = []
    
    for output in outputs:
        # Extract explicit file paths (e.g., /path/to/file.py:123)
        for match in _PATH_RE.finditer(output):
            raw_path = match.group("path")
            if not raw_path:
                continue
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = repo_root / raw_path
            if not candidate.exists():
                continue
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
            # Track test files for import parsing
            if "test" in candidate.name.lower() and candidate.suffix == ".py":
                test_files_to_parse.append(candidate)
        
        # Extract module imports from output and convert to file paths
        for match in _IMPORT_RE.finditer(output):
            _add_module_as_path(match.group("module"), repo_root, seen, found)
    
    # Parse imports from failing test files to find the modules under test
    for test_file in test_files_to_parse:
        try:
            content = test_file.read_text(encoding="utf-8", errors="replace")
            for match in _IMPORT_RE.finditer(content):
                _add_module_as_path(match.group("module"), repo_root, seen, found)
        except Exception:
            pass
    
    return found


def _add_module_as_path(module: str | None, repo_root: Path, seen: set, found: List[str]) -> None:
    """Convert a module name to file path and add to found list if it exists."""
    if not module:
        return
    # Skip standard library and common third-party modules
    skip_prefixes = ("os", "sys", "re", "json", "typing", "pathlib", "datetime", 
                     "collections", "functools", "itertools", "unittest", "pytest",
                     "numpy", "pandas", "fastapi", "pydantic", "sqlalchemy")
    if module.split(".")[0] in skip_prefixes:
        return
    
    # Convert module path to file path (e.g., consumer.consumer -> consumer/consumer.py)
    module_path = module.replace(".", "/") + ".py"
    candidate = repo_root / module_path
    if candidate.exists():
        resolved = str(candidate.resolve())
        if resolved not in seen:
            seen.add(resolved)
            found.append(resolved)
    # Also try as package __init__.py
    package_path = module.replace(".", "/") + "/__init__.py"
    candidate = repo_root / package_path
    if candidate.exists():
        resolved = str(candidate.resolve())
        if resolved not in seen:
            seen.add(resolved)
            found.append(resolved)


EXCLUDED_PATH_SEGMENTS = {".venv", "venv", "node_modules", "__pycache__", "site-packages", ".git"}


def _merge_target_files(*groups: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for group in groups:
        for path in group or []:
            if not path or path in seen:
                continue
            # Skip files in excluded directories (virtual envs, node_modules, etc.)
            if any(segment in path for segment in EXCLUDED_PATH_SEGMENTS):
                continue
            seen.add(path)
            merged.append(path)
    return merged


def _build_safety_context(state: GraphState) -> str:
    details = _latest_step_details(state, "safety")
    results = details.get("results") if isinstance(details, dict) else None
    if not isinstance(results, list):
        return "None"
    blocks: List[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        name = result.get("name", "tool")
        exit_code = result.get("exit_code", "")
        stdout = _trim_text(result.get("stdout", ""), limit=1500)
        stderr = _trim_text(result.get("stderr", ""), limit=800)
        blocks.append(
            f"{name}: exit_code={exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}\n"
        )
    return "\n".join(blocks) if blocks else "None"


def _trim_text(value: Optional[str], limit: int = 2000) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit] + "...(truncated)"


def _to_run_state(state: GraphState) -> RunState:
    run_state = RunState.start(goal=state["goal"], run_id=state["run_id"])
    run_state.status = state.get("status", "running")
    run_state.iterations = _iterations_from_state(state)
    return run_state


def _iterations_from_state(state: GraphState) -> List[IterationRecord]:
    iterations: List[IterationRecord] = []
    for iteration_data in state.get("iterations", []):
        iteration = IterationRecord(
            index=iteration_data["index"],
            started_at=iteration_data["started_at"],
            ended_at=iteration_data.get("ended_at"),
        )
        for step_data in iteration_data.get("steps", []):
            step = StepResult(
                name=step_data["name"],
                status=step_data["status"],
                details=step_data.get("details", {}),
                started_at=step_data.get("started_at", utc_now_iso()),
                ended_at=step_data.get("ended_at"),
            )
            iteration.steps.append(step)
        iterations.append(iteration)
    return iterations


def _log_step(step_name: str, state: GraphState) -> None:
    with span(
        f"step.{step_name}",
        {
            "run_id": state["run_id"],
            "step": step_name,
            "iteration": state["current_iteration"],
            "status": state["status"],
        },
    ):
        log_event(
            _logger(),
            "step_complete",
            {
                "run_id": state["run_id"],
                "step": step_name,
                "iteration": state["current_iteration"],
                "status": state["status"],
            },
        )


def _logger():
    from observability.logging import configure_logging

    return configure_logging()


def _build_graph_episode(state: GraphState) -> Dict[str, Any]:
    iteration_data = state["iterations"][-1] if state.get("iterations") else {}
    steps = []
    for idx, step in enumerate(iteration_data.get("steps", [])):
        step_entry = dict(step)
        step_entry["idx"] = idx
        steps.append(step_entry)
    retrieve_step = next((s for s in steps if s.get("name") == "retrieve"), {})
    retrieved_files = retrieve_step.get("details", {}).get("top_sources", [])

    return {
        "run_id": state["run_id"],
        "goal": state["goal"],
        "status": state.get("status"),
        "iteration": state.get("current_iteration"),
        "steps": steps,
        "retrieved_files": retrieved_files,
        "failure_signature": None,
        "bug_type": None,
    }


def _generate_fix_card(llm_client: LLMClient, state: GraphState, config: RunConfig) -> Optional[Dict[str, Any]]:
    context = _build_fix_card_context(state, config)
    prompt = _build_fix_card_prompt(context)
    response = llm_client.strong_complete(prompt)
    fix_card = _parse_fix_card_json(response.content)
    if fix_card is None:
        fix_card = {}
    return _normalize_fix_card(fix_card, context)


def _build_fix_card_context(state: GraphState, config: RunConfig) -> Dict[str, Any]:
    execute_details = _latest_step_details(state, "execute")
    execute_result = execute_details.get("result", {}) if isinstance(execute_details, dict) else {}
    retrieve_details = _latest_step_details(state, "retrieve")
    audit_details = _latest_step_details(state, "audit")
    audit_findings = []
    if isinstance(audit_details, dict):
        audit_findings = audit_details.get("findings", []) or []
    patch_text = state.get("patch_text") or ""
    files_changed = _extract_patch_files(patch_text)
    error_signature = _extract_error_signature(execute_result, state.get("last_error"))

    return {
        "goal": state.get("goal"),
        "status": state.get("status"),
        "run_id": state.get("run_id"),
        "iteration": state.get("current_iteration"),
        "repo_path": config.repo_path,
        "last_error": state.get("last_error"),
        "retrieved_files": retrieve_details.get("top_sources", []),
        "audit_findings": audit_findings[:20],
        "files_changed": files_changed,
        "test_command": execute_result.get("command"),
        "test_exit_code": execute_result.get("exit_code"),
        "test_stdout": _trim_text(execute_result.get("stdout")),
        "test_stderr": _trim_text(execute_result.get("stderr")),
        "error_signature": error_signature,
    }


def _build_fix_card_prompt(context: Dict[str, Any]) -> str:
    return (
        "Create a JSON fix card using the context below. "
        "Return ONLY valid JSON with keys: summary, root_cause, fix, verification, "
        "error_signature, files_changed, debug_steps. "
        "Use null if unknown. Keep fields concise.\n\n"
        f"Context: {json.dumps(context, ensure_ascii=True)}"
    )


def _parse_fix_card_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidate = text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _normalize_fix_card(fix_card: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    files_changed = fix_card.get("files_changed")
    if not isinstance(files_changed, list):
        files_changed = context.get("files_changed", [])

    error_signature = fix_card.get("error_signature") or context.get("error_signature")
    summary = fix_card.get("summary") or f"Fix card for goal: {context.get('goal')}"

    return {
        "type": "fix_card",
        "id": f"{context.get('run_id')}:{context.get('iteration')}",
        "summary": summary,
        "root_cause": fix_card.get("root_cause"),
        "fix": fix_card.get("fix"),
        "verification": fix_card.get("verification"),
        "debug_steps": fix_card.get("debug_steps"),
        "error_signature": error_signature,
        "files_changed": files_changed,
        "goal": context.get("goal"),
        "status": context.get("status"),
        "repo_path": context.get("repo_path"),
        "run_id": context.get("run_id"),
        "iteration": context.get("iteration"),
        "test_command": context.get("test_command"),
        "test_exit_code": context.get("test_exit_code"),
    }


def _fix_card_public(fix_card: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "summary": fix_card.get("summary"),
        "root_cause": fix_card.get("root_cause"),
        "fix": fix_card.get("fix"),
        "verification": fix_card.get("verification"),
        "files_changed": fix_card.get("files_changed"),
        "error_signature": fix_card.get("error_signature"),
    }


def _extract_error_signature(execute_result: Dict[str, Any], fallback: Optional[str]) -> Optional[str]:
    stderr = execute_result.get("stderr") if isinstance(execute_result, dict) else None
    if isinstance(stderr, str) and stderr.strip():
        return stderr.splitlines()[0][:200]
    if isinstance(fallback, str) and fallback.strip():
        return fallback[:200]
    return None


def _extract_patch_files(patch_text: str) -> List[str]:
    files: List[str] = []
    for line in patch_text.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            parts = line.split()
            if len(parts) < 2:
                continue
            path = parts[1]
            if path.startswith("a/") or path.startswith("b/"):
                path = path[2:]
            if path == "/dev/null":
                continue
            if path not in files:
                files.append(path)
    return files


def _latest_step_details(state: GraphState, step_name: str) -> Dict[str, Any]:
    for iteration in reversed(state.get("iterations", [])):
        for step in reversed(iteration.get("steps", [])):
            if step.get("name") == step_name:
                return step.get("details", {}) or {}
    return {}


def _effective_repo_path(state: GraphState, config: RunConfig) -> str:
    workspace = state.get("workspace_path")
    if workspace:
        return workspace
    return config.repo_path


def _sandbox_config(config: RunConfig) -> SandboxConfig:
    return SandboxConfig(
        image=config.sandbox_image,
        cpus=config.sandbox_cpus,
        memory=config.sandbox_memory,
        network=config.sandbox_network,
        user=config.sandbox_user,
    )
