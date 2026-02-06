from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass
class PatchResult:
    workspace_path: str
    patch_path: str
    applied: bool
    output: str


DEFAULT_BLOCKED_PREFIXES = (".git/", ".env", "../", "/")
DEFAULT_IGNORE_DIRS = (".git", ".venv", ".rag", ".workspaces", "__pycache__")


def create_workspace(source_repo: str, root_dir: str | Path) -> str:
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(prefix="workspace-", dir=str(root))
    )
    shutil.copytree(
        source_repo,
        workspace,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*DEFAULT_IGNORE_DIRS),
    )
    _init_git_repo(workspace)
    return str(workspace)


def write_patch(patch_text: str, output_dir: str | Path, run_id: str) -> str:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    patch_path = output / f"{run_id}.patch"
    patch_path.write_text(patch_text, encoding="utf-8")
    return str(patch_path)


def validate_patch(patch_text: str, blocked_prefixes: Sequence[str] | None = None) -> List[str]:
    blocked_prefixes = blocked_prefixes or DEFAULT_BLOCKED_PREFIXES
    violations: List[str] = []
    for path in _extract_paths(patch_text):
        if any(path.startswith(prefix) for prefix in blocked_prefixes):
            violations.append(path)
        if path.startswith(".."):
            violations.append(path)
    return sorted(set(violations))


def apply_patch(patch_path: str, repo_path: str) -> Tuple[bool, str]:
    command = ["git", "apply", "--whitespace=nowarn", patch_path]
    result = subprocess.run(
        command,
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def apply_patch_with_fallback(patch_path: str, repo_path: str) -> Tuple[bool, str]:
    """Apply a patch with multiple fallback strategies for LLM-generated diffs."""
    outputs: List[str] = []

    # Strategy 1: Standard apply
    ok, out = apply_patch(patch_path, repo_path)
    outputs.append(out)
    if ok:
        return True, "\n".join(outputs).strip()

    # Strategy 2: Recount (fixes incorrect hunk line counts)
    command = ["git", "apply", "--recount", "--whitespace=nowarn", patch_path]
    result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True, check=False)
    outputs.append((result.stdout or "") + (result.stderr or ""))
    if result.returncode == 0:
        return True, "\n".join(outputs).strip()

    # Strategy 3: Realign patch line numbers by searching for context in files
    patch_file = Path(patch_path)
    try:
        original_patch = patch_file.read_text(encoding="utf-8")
        realigned = realign_patch_to_file(original_patch, repo_path)
        if realigned != original_patch:
            realigned_path = patch_file.with_suffix(".realigned.patch")
            realigned_path.write_text(realigned, encoding="utf-8")
            command = ["git", "apply", "--whitespace=nowarn", str(realigned_path)]
            result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True, check=False)
            outputs.append(f"(realigned patch) {(result.stdout or '')} {(result.stderr or '')}")
            if result.returncode == 0:
                return True, "\n".join(outputs).strip()
            # Try realigned with recount
            command = ["git", "apply", "--recount", "--whitespace=nowarn", str(realigned_path)]
            result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True, check=False)
            outputs.append((result.stdout or "") + (result.stderr or ""))
            if result.returncode == 0:
                return True, "\n".join(outputs).strip()
    except Exception as e:
        outputs.append(f"(realign failed: {e})")

    # Strategy 4: 3-way merge (uses index to resolve conflicts)
    command = ["git", "apply", "--3way", "--whitespace=nowarn", patch_path]
    result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True, check=False)
    outputs.append((result.stdout or "") + (result.stderr or ""))
    if result.returncode == 0:
        return True, "\n".join(outputs).strip()

    # Strategy 5: Minimal context matching
    command = ["git", "apply", "--recount", "-C0", "--whitespace=nowarn", patch_path]
    result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True, check=False)
    outputs.append((result.stdout or "") + (result.stderr or ""))
    if result.returncode == 0:
        return True, "\n".join(outputs).strip()

    # Strategy 6: Unix patch with fuzz factor to allow context mismatch
    command = ["patch", "--batch", "--forward", "-p1", "--fuzz=3", "-i", patch_path]
    result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True, check=False)
    outputs.append((result.stdout or "") + (result.stderr or ""))
    if result.returncode == 0:
        return True, "\n".join(outputs).strip()

    return False, "\n".join(outputs).strip()


def extract_unified_diff(text: str) -> str | None:
    marker = "diff --git"
    if marker in text:
        diff = text[text.index(marker) :].strip()
        return sanitize_unified_diff(diff)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("--- "):
            diff = "\n".join(lines[i:]).strip()
            return sanitize_unified_diff(diff)
    return None


def sanitize_unified_diff(text: str) -> str:
    """Remove markdown fences, stray non-diff lines, and fix hunk headers."""
    if not text:
        return text
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            continue
        lines.append(line)

    allowed_prefixes = (
        "diff --git ",
        "index ",
        "--- ",
        "+++ ",
        "@@ ",
        "@@",
        "+",
        "-",
        " ",
        "\\ No newline at end of file",
        "new file mode ",
        "deleted file mode ",
        "similarity index ",
        "dissimilarity index ",
        "rename from ",
        "rename to ",
        "copy from ",
        "copy to ",
        "old mode ",
        "new mode ",
        "Binary files ",
        "GIT binary patch",
        "literal ",
        "delta ",
    )

    cleaned: List[str] = []
    in_binary = False
    for line in lines:
        if line.startswith("diff --git "):
            in_binary = False
            cleaned.append(line)
            continue
        if line.startswith("GIT binary patch"):
            in_binary = True
            cleaned.append(line)
            continue
        if in_binary:
            cleaned.append(line)
            continue
        if any(line.startswith(prefix) for prefix in allowed_prefixes):
            cleaned.append(line)
            continue
        # Drop stray non-diff lines.
        continue

    normalized = _normalize_hunk_headers(cleaned)
    # Ensure the patch ends with a newline (required for proper patch format)
    result = "\n".join(normalized).strip()
    if result and not result.endswith("\n"):
        result += "\n"
    return result


HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def realign_patch_to_file(patch_text: str, repo_path: str | Path) -> str:
    """Auto-correct hunk line numbers by searching for context in actual files."""
    if not patch_text:
        return patch_text
    
    repo = Path(repo_path)
    lines = patch_text.splitlines()
    result: List[str] = []
    
    current_file: str | None = None
    file_lines: List[str] = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Track which file we're patching
        if line.startswith("--- a/"):
            current_file = line[6:].split("\t")[0].strip()
            file_path = repo / current_file
            if file_path.is_file():
                try:
                    file_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
                except Exception:
                    file_lines = []
            else:
                file_lines = []
            result.append(line)
            i += 1
            continue
        
        # Process hunk headers - realign line numbers
        if line.startswith("@@") and current_file:
            match = HUNK_HEADER_RE.match(line)
            if match and file_lines:
                suffix = match.group(5) or ""
                
                # Collect context/changed lines for this hunk
                hunk_lines: List[str] = []
                j = i + 1
                while j < len(lines):
                    hunk_line = lines[j]
                    if hunk_line.startswith("diff --git ") or hunk_line.startswith("@@"):
                        break
                    hunk_lines.append(hunk_line)
                    j += 1
                
                # Extract old-side lines (context + removed) to search for
                old_lines_content = []
                for hl in hunk_lines:
                    if hl.startswith(" ") or hl.startswith("-"):
                        old_lines_content.append(hl[1:])  # Strip the prefix
                    elif hl.startswith("\\ No newline"):
                        continue
                    elif not hl.startswith("+"):
                        # Treat unprefixed as context
                        old_lines_content.append(hl)
                
                # Try to find where this context actually appears in the file
                new_start = _find_context_in_file(file_lines, old_lines_content)
                if new_start is not None:
                    old_start = new_start
                else:
                    old_start = int(match.group(1))
                
                # Recompute counts
                old_count = 0
                new_count = 0
                for hl in hunk_lines:
                    if hl.startswith("+"):
                        new_count += 1
                    elif hl.startswith("-"):
                        old_count += 1
                    elif hl.startswith(" "):
                        old_count += 1
                        new_count += 1
                    elif hl.startswith("\\ No newline"):
                        pass
                    else:
                        old_count += 1
                        new_count += 1
                
                result.append(f"@@ -{old_start},{old_count} +{old_start},{new_count} @@{suffix}")
                result.extend(hunk_lines)
                i = j
                continue
        
        result.append(line)
        i += 1
    
    # Ensure trailing newline
    output = "\n".join(result)
    if output and not output.endswith("\n"):
        output += "\n"
    return output


def _find_context_in_file(file_lines: List[str], context_lines: List[str]) -> int | None:
    """Find where context_lines appear in file_lines, return 1-based line number."""
    if not context_lines or not file_lines:
        return None
    
    # Use first few non-empty context lines as search anchor
    search_lines = [ln.strip() for ln in context_lines if ln.strip()][:3]
    if not search_lines:
        return None
    
    first_search = search_lines[0]
    
    for i, file_line in enumerate(file_lines):
        if file_line.strip() == first_search:
            # Check if subsequent lines match
            match = True
            for offset, search_line in enumerate(search_lines[1:], start=1):
                if i + offset >= len(file_lines):
                    match = False
                    break
                if file_lines[i + offset].strip() != search_line:
                    match = False
                    break
            if match:
                return i + 1  # 1-based line number
    
    return None


def _normalize_hunk_headers(lines: List[str]) -> List[str]:
    normalized: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("@@"):
            normalized.append(line)
            i += 1
            continue
        match = HUNK_HEADER_RE.match(line)
        if not match:
            normalized.append(line)
            i += 1
            continue
        old_start = match.group(1)
        new_start = match.group(3)
        suffix = match.group(5) or ""

        old_count = 0
        new_count = 0
        j = i + 1
        while j < len(lines):
            hunk_line = lines[j]
            if hunk_line.startswith("diff --git ") or hunk_line.startswith("@@"):
                break
            if hunk_line.startswith("+"):
                new_count += 1
            elif hunk_line.startswith("-"):
                old_count += 1
            elif hunk_line.startswith(" "):
                old_count += 1
                new_count += 1
            elif hunk_line.startswith("\\ No newline"):
                pass
            else:
                old_count += 1
                new_count += 1
            j += 1

        normalized.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{suffix}")
        normalized.extend(lines[i + 1 : j])
        i = j

    return normalized


def _extract_paths(patch_text: str) -> Iterable[str]:
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
            yield path


def _init_git_repo(path: Path) -> None:
    subprocess.run(
        ["git", "init"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=False,
    )
