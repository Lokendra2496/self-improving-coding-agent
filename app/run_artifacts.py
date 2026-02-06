"""Run artifact management - bundles all outputs under runs/<run_id>/."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.state import RunState


@dataclass
class RunArtifacts:
    run_dir: Path
    summary_path: Path
    state_path: Path
    patch_path: Optional[Path]
    workspace_path: Optional[Path]


def create_run_dir(run_id: str, runs_root: str = "runs") -> Path:
    """Create the run artifact directory."""
    run_dir = Path(runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_artifacts(
    state: RunState,
    patch_text: Optional[str],
    workspace_path: Optional[str],
    runs_root: str = "runs",
) -> RunArtifacts:
    """Save all run artifacts to runs/<run_id>/."""
    run_dir = create_run_dir(state.run_id, runs_root)

    # Save full state as JSON
    state_path = run_dir / "state.json"
    state_path.write_text(json.dumps(state.to_dict(), indent=2, default=str))

    # Save patch if generated
    patch_path: Optional[Path] = None
    if patch_text:
        patch_path = run_dir / "changes.patch"
        patch_path.write_text(patch_text)

    # Link or note workspace path
    workspace_ref: Optional[Path] = None
    if workspace_path:
        workspace_ref = Path(workspace_path)
        # Write a reference file pointing to workspace
        (run_dir / "workspace.txt").write_text(str(workspace_ref.resolve()))

    # Generate summary
    summary_path = run_dir / "summary.md"
    summary_content = generate_summary(state, patch_text, workspace_path)
    summary_path.write_text(summary_content)

    return RunArtifacts(
        run_dir=run_dir,
        summary_path=summary_path,
        state_path=state_path,
        patch_path=patch_path,
        workspace_path=workspace_ref,
    )


def generate_summary(
    state: RunState,
    patch_text: Optional[str],
    workspace_path: Optional[str],
) -> str:
    """Generate a markdown summary report for the run."""
    lines: List[str] = []

    # Header
    lines.append(f"# Run Summary: {state.run_id}")
    lines.append("")
    lines.append(f"**Goal:** {state.goal}")
    lines.append(f"**Status:** {state.status}")
    lines.append(f"**Created:** {state.created_at}")
    if state.completed_at:
        lines.append(f"**Completed:** {state.completed_at}")
    lines.append("")

    # Metrics
    if state.metrics:
        lines.append("## Metrics")
        lines.append("")
        duration = state.metrics.get("run_duration_ms")
        if duration:
            lines.append(f"- **Duration:** {duration}ms")
        counters = state.metrics.get("counters", {})
        if counters:
            for name, value in counters.items():
                lines.append(f"- **{name}:** {value}")
        lines.append("")

    warnings = _collect_run_warnings(state)

    # Iterations
    lines.append("## Iterations")
    lines.append("")
    for iteration in state.iterations:
        lines.append(f"### Iteration {iteration.index}")
        lines.append("")
        if iteration.started_at:
            lines.append(f"- Started: {iteration.started_at}")
        if iteration.ended_at:
            lines.append(f"- Ended: {iteration.ended_at}")
        lines.append("")

        # Steps table
        lines.append("| Step | Status | Details |")
        lines.append("|------|--------|---------|")
        for step in iteration.steps:
            details_summary = _summarize_details(step.details)
            lines.append(f"| {step.name} | {step.status} | {details_summary} |")
        lines.append("")

    audit_findings = _extract_audit_findings(state)
    if audit_findings:
        lines.append("## Static Audit Findings")
        lines.append("")
        for finding in audit_findings:
            location = finding.get("file", "")
            line = finding.get("line")
            location_text = f"{location}:{line}" if location and line else location
            detail = finding.get("detail", "")
            issue_type = finding.get("type", "issue")
            excerpt = finding.get("excerpt", "")
            lines.append(f"- [{issue_type}] {location_text} {detail}".strip())
            if excerpt:
                lines.append(f"  - `{excerpt}`")
        lines.append("")

    line_review, line_review_error = _extract_audit_line_review(state)
    if line_review or line_review_error:
        lines.append("## Static Audit Review Notes")
        lines.append("")
        if line_review_error:
            lines.append(f"- Line-by-line review error: {line_review_error}")
        max_notes = 60
        for note in line_review[:max_notes]:
            location = note.get("file", "")
            line = note.get("line")
            location_text = f"{location}:{line}" if location and line else location
            severity = note.get("severity", "info")
            detail = note.get("note", "")
            lines.append(f"- [{severity}] {location_text} {detail}".strip())
        if len(line_review) > max_notes:
            lines.append(f"- ... {len(line_review) - max_notes} more notes omitted")
        lines.append("")

    # Patch info
    lines.append("## Patch")
    lines.append("")
    if patch_text:
        patch_lines = patch_text.count("\n") + 1
        patch_bytes = len(patch_text.encode("utf-8"))
        lines.append(f"- **Size:** {patch_bytes} bytes ({patch_lines} lines)")
        lines.append("")
        lines.append("### Files Changed")
        lines.append("")
        for path in _extract_changed_files(patch_text):
            lines.append(f"- `{path}`")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>View Patch</summary>")
        lines.append("")
        lines.append("```diff")
        lines.append(patch_text)
        lines.append("```")
        lines.append("")
        lines.append("</details>")
    else:
        lines.append("No patch generated.")
    lines.append("")

    # Workspace
    lines.append("## Workspace")
    lines.append("")
    if workspace_path:
        lines.append(f"Workspace with applied patch: `{workspace_path}`")
    else:
        lines.append("No workspace created.")
    lines.append("")

    fix_card = _extract_fix_card(state)

    # Lessons learned
    if fix_card:
        lines.append("## Lessons Learned")
        lines.append("")
        if fix_card.get("summary"):
            lines.append(f"- **Summary:** {fix_card.get('summary')}")
        if fix_card.get("root_cause"):
            lines.append(f"- **Root cause:** {fix_card.get('root_cause')}")
        if fix_card.get("fix"):
            lines.append(f"- **Fix:** {fix_card.get('fix')}")
        if fix_card.get("verification"):
            lines.append(f"- **Verification:** {fix_card.get('verification')}")
        files_changed = fix_card.get("files_changed") or []
        if files_changed:
            lines.append("- **Files changed:**")
            for path in files_changed:
                lines.append(f"  - `{path}`")
        lines.append("")

    # Notes
    if warnings:
        lines.append("## Notes")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    # Next steps
    lines.append("## Next Steps")
    lines.append("")
    if state.status == "completed":
        if patch_text:
            lines.append("- Review the generated patch")
            lines.append("- Run tests in the workspace to verify")
            lines.append("- If satisfied, use `--apply` to write changes to the repo")
        else:
            lines.append("- Provide a more specific goal or failing test output")
            lines.append("- Configure tools with --test-cmd/--ruff-cmd/--bandit-cmd")
            lines.append("- Re-run without --dry-run to capture real failures")
    elif state.status == "failed":
        lines.append("- Check the error details above")
        lines.append("- Review logs for more context")
        lines.append("- Adjust goal or configuration and retry")
    else:
        lines.append("- Run is still in progress or was interrupted")
        lines.append("- Use `--resume-run-id` to continue")
    lines.append("")

    return "\n".join(lines)


def _summarize_details(details: Dict[str, Any]) -> str:
    """Create a short summary of step details."""
    if not details:
        return "-"

    parts: List[str] = []

    # Common fields
    if "model" in details:
        parts.append(f"model={details['model']}")
    if "exit_code" in details:
        parts.append(f"exit={details['exit_code']}")
    if "applied" in details:
        parts.append(f"applied={details['applied']}")
    if "error" in details:
        parts.append(f"error={details['error']}")
    if "reason" in details:
        parts.append(f"reason={details['reason']}")
    if "decision" in details:
        parts.append(f"decision={details['decision']}")

    if not parts:
        # Fallback to first few keys
        for key in list(details.keys())[:2]:
            val = details[key]
            if isinstance(val, str) and len(val) > 30:
                val = val[:30] + "..."
            parts.append(f"{key}={val}")

    return ", ".join(parts) if parts else "-"


def _extract_fix_card(state: RunState) -> Dict[str, Any] | None:
    for iteration in reversed(state.iterations):
        for step in reversed(iteration.steps):
            if step.name == "memory_update":
                fix_card = step.details.get("fix_card") if step.details else None
                if isinstance(fix_card, dict):
                    return fix_card
    return None


def _extract_audit_findings(state: RunState) -> List[Dict[str, Any]]:
    for iteration in reversed(state.iterations):
        for step in reversed(iteration.steps):
            if step.name == "audit":
                findings = step.details.get("findings") if step.details else None
                if isinstance(findings, list):
                    return findings
    return []


def _extract_audit_line_review(
    state: RunState,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    for iteration in reversed(state.iterations):
        for step in reversed(iteration.steps):
            if step.name == "audit":
                details = step.details or {}
                review = details.get("line_review")
                error = details.get("line_review_error")
                if isinstance(review, list) or error:
                    return review if isinstance(review, list) else [], error
    return [], None


def _extract_changed_files(patch_text: str) -> List[str]:
    """Extract list of changed files from patch text."""
    files: List[str] = []
    for line in patch_text.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            parts = line.split()
            if len(parts) >= 2:
                path = parts[1]
                if path.startswith("a/") or path.startswith("b/"):
                    path = path[2:]
                if path != "/dev/null" and path not in files:
                    files.append(path)
    return files


def _collect_run_warnings(state: RunState) -> List[str]:
    warnings: List[str] = []
    safety_skipped = False
    execute_skipped = False

    for iteration in state.iterations:
        for step in iteration.steps:
            if step.name == "safety":
                if step.status == "skipped":
                    safety_skipped = True
                else:
                    results = step.details.get("results", [])
                    for result in results:
                        if not isinstance(result, dict):
                            continue
                        exit_code = result.get("exit_code")
                        tool_name = result.get("name", "tool")
                        if exit_code == 2:
                            warnings.append(f"Static check not configured: {tool_name}.")
                        if exit_code == 127:
                            warnings.append(f"Static check tool not found: {tool_name}.")
            if step.name == "execute":
                if step.status == "skipped":
                    execute_skipped = True
                else:
                    result = step.details.get("result", {})
                    if isinstance(result, dict):
                        exit_code = result.get("exit_code")
                        if exit_code == 2:
                            warnings.append("Tests were not configured (missing --test-cmd).")
                        if exit_code == 127:
                            warnings.append("Test command not found. Check your --test-cmd.")
            if step.name == "patch" and step.status == "skipped":
                reason = step.details.get("reason")
                if reason == "no diff found":
                    warnings.append("No patch generated; refine the goal or provide failing output.")

    if safety_skipped and execute_skipped:
        warnings.append("Dry-run mode: safety and execute steps were skipped.")

    return sorted(set(warnings))
