from __future__ import annotations

import ast
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rag.chunking import iter_repo_files

SQL_KEYWORDS = [
    "select",
    "insert",
    "update",
    "delete",
    "from",
    "where",
    "join",
    "group",
    "order",
    "values",
    "into",
    "limit",
]

SQL_TYPO_MAP = {
    "selct": "select",
    "selec": "select",
    "fromm": "from",
    "frmo": "from",
    "whre": "where",
    "wher": "where",
    "joon": "join",
    "jion": "join",
    "orde": "order",
    "ordr": "order",
    "gropu": "group",
    "gruop": "group",
    "vales": "values",
    "valeus": "values",
    "inser": "insert",
    "updae": "update",
    "delte": "delete",
}

SQL_START_RE = re.compile(r"\b(select|insert|update|delete|with)\b", re.IGNORECASE)
WORD_RE = re.compile(r"\b[a-zA-Z_]+\b")
CODE_EXTS = [
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rb",
    ".rs",
    ".cpp",
    ".c",
    ".cs",
    ".kt",
    ".php",
    ".scala",
]


def audit_repo(
    goal: str,
    repo_path: str | Path,
    target_files: Optional[Sequence[str]] = None,
    max_findings: int = 20,
    max_files: int = 200,
) -> List[Dict[str, Any]]:
    repo_root = Path(repo_path).resolve()
    targets = _resolve_targets(goal, repo_root, target_files, max_files)
    findings: List[Dict[str, Any]] = []

    for path in targets:
        if len(findings) >= max_findings:
            break
        try:
            if path.suffix == ".py":
                findings.extend(_audit_python_file(path))
            elif path.suffix == ".sql":
                findings.extend(_audit_sql_file(path))
        except Exception as exc:
            findings.append(
                {
                    "file": str(path),
                    "line": None,
                    "type": "audit_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "excerpt": "",
                }
            )

    return findings[:max_findings]


def collect_sql_snippets(
    goal: str,
    repo_path: str | Path,
    target_files: Optional[Sequence[str]] = None,
    max_snippets: int = 10,
    max_files: int = 200,
) -> List[Dict[str, Any]]:
    repo_root = Path(repo_path).resolve()
    targets = _resolve_targets(goal, repo_root, target_files, max_files)
    snippets: List[Dict[str, Any]] = []
    seen: set[tuple[str, Optional[int], str]] = set()

    for path in targets:
        if len(snippets) >= max_snippets:
            break
        if path.suffix == ".py":
            items = _collect_python_sql(path)
        elif path.suffix == ".sql":
            items = _collect_sql_file_sql(path)
        else:
            items = []
        for item in items:
            key = (item["file"], item.get("line"), item.get("query", ""))
            if key in seen:
                continue
            seen.add(key)
            snippets.append(item)
            if len(snippets) >= max_snippets:
                break

    return snippets[:max_snippets]


def collect_code_snippets(
    goal: str,
    repo_path: str | Path,
    target_files: Optional[Sequence[str]] = None,
    max_snippets: int = 8,
    max_chars: int = 2000,
    max_files: int = 200,
) -> List[Dict[str, Any]]:
    repo_root = Path(repo_path).resolve()
    targets = _resolve_targets(goal, repo_root, target_files, max_files)
    snippets: List[Dict[str, Any]] = []
    focus_terms = _extract_focus_terms(goal)

    if not targets:
        targets = list(iter_repo_files(repo_root, include_exts=CODE_EXTS))[:max_files]

    for path in targets:
        if len(snippets) >= max_snippets:
            break
        if path.suffix not in CODE_EXTS:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        snippet, start_line = _focused_snippet(content, focus_terms, max_chars)
        numbered = _numbered_snippet(snippet, start_line)
        snippets.append(
            {
                "file": str(path),
                "line": start_line,
                "start_line": start_line,
                "snippet": snippet,
                "numbered": numbered,
            }
        )

    return snippets[:max_snippets]


def _resolve_targets(
    goal: str,
    repo_root: Path,
    target_files: Optional[Sequence[str]],
    max_files: int,
) -> List[Path]:
    resolved: List[Path] = []
    if target_files:
        for entry in target_files:
            candidate = Path(entry)
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            if candidate.exists():
                resolved.append(candidate.resolve())

    for token in _extract_file_tokens(goal):
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if candidate.exists():
            resolved.append(candidate.resolve())
    if resolved:
        return _dedupe_paths(resolved)

    files = list(
        iter_repo_files(
            repo_root,
            include_exts=CODE_EXTS,
        )
    )
    files.sort()
    return files[:max_files]


def resolve_goal_paths(goal: str, repo_path: str | Path) -> List[str]:
    repo_root = Path(repo_path).resolve()
    resolved: List[Path] = []
    for token in _extract_file_tokens(goal):
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if candidate.exists():
            resolved.append(candidate.resolve())
    return [str(path) for path in _dedupe_paths(resolved)]


def _extract_file_tokens(text: str) -> List[str]:
    tokens = re.split(r"\s+", text)
    results: List[str] = []
    for token in tokens:
        cleaned = token.strip().strip('"').strip("'").strip(",.:;")
        if len(cleaned) < 3:
            continue
        if "/" in cleaned or "\\" in cleaned:
            results.append(cleaned)
            continue
        if "." in Path(cleaned).name:
            results.append(cleaned)
    return results


def _audit_python_file(path: Path) -> List[Dict[str, Any]]:
    source = path.read_text(encoding="utf-8", errors="ignore")
    findings: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [
            {
                "file": str(path),
                "line": getattr(exc, "lineno", None),
                "type": "syntax_error",
                "detail": str(exc),
                "excerpt": "",
            }
        ]

    visitor = _SQLVisitor(str(path), source.splitlines())
    visitor.visit(tree)
    findings.extend(visitor.findings)
    return findings


def _audit_sql_file(path: Path) -> List[Dict[str, Any]]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    findings: List[Dict[str, Any]] = []
    for idx, line in enumerate(content.splitlines(), start=1):
        if not _looks_like_sql(line):
            continue
        findings.extend(_find_sql_typos(line, str(path), idx, line.strip()))
    return findings


def _collect_python_sql(path: Path) -> List[Dict[str, Any]]:
    source = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    collector = _SQLCollector(str(path), source.splitlines())
    collector.visit(tree)
    return collector.snippets


def _collect_sql_file_sql(path: Path) -> List[Dict[str, Any]]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    statements: List[Dict[str, Any]] = []
    buffer: List[str] = []
    start_line = 1
    for idx, line in enumerate(content.splitlines(), start=1):
        if not buffer:
            start_line = idx
        buffer.append(line)
        if ";" in line:
            text = "\n".join(buffer).strip()
            if _looks_like_sql(text):
                statements.append(
                    {
                        "file": str(path),
                        "line": start_line,
                        "query": _trim_sql(text),
                    }
                )
            buffer = []
    if buffer:
        text = "\n".join(buffer).strip()
        if _looks_like_sql(text):
            statements.append(
                {
                    "file": str(path),
                    "line": start_line,
                    "query": _trim_sql(text),
                }
            )
    return statements


class _SQLVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, lines: List[str]) -> None:
        self.file_path = file_path
        self.lines = lines
        self.findings: List[Dict[str, Any]] = []

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Any:
        text = _joined_str_text(node)
        if _looks_like_sql(text):
            self._add_typo_findings(node, text)
            if _has_dynamic_part(node):
                self._add_injection_finding(node, "f-string used in SQL query")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if isinstance(node.op, ast.Add):
            text, dynamic = _binop_text(node)
            if text and _looks_like_sql(text):
                self._add_typo_findings(node, text)
                if dynamic:
                    self._add_injection_finding(node, "string concatenation used in SQL query")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if _is_format_call(node):
            base = _format_base_string(node)
            if base and _looks_like_sql(base):
                self._add_typo_findings(node, base)
                self._add_injection_finding(node, "format() used in SQL query")
        if _is_execute_call(node):
            arg = node.args[0] if node.args else None
            text = _string_from_node(arg)
            if text and _looks_like_sql(text):
                self._add_typo_findings(node, text)
                if _has_dynamic_expression(arg):
                    self._add_injection_finding(node, "dynamic SQL passed to execute()")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, str):
            if _looks_like_sql(node.value):
                self._add_typo_findings(node, node.value)
        self.generic_visit(node)

    def _add_typo_findings(self, node: ast.AST, text: str) -> None:
        line = getattr(node, "lineno", None)
        excerpt = _line_excerpt(self.lines, line)
        self.findings.extend(_find_sql_typos(text, self.file_path, line, excerpt))

    def _add_injection_finding(self, node: ast.AST, detail: str) -> None:
        line = getattr(node, "lineno", None)
        self.findings.append(
            {
                "file": self.file_path,
                "line": line,
                "type": "sql_injection_risk",
                "detail": detail,
                "excerpt": _line_excerpt(self.lines, line),
            }
        )


class _SQLCollector(ast.NodeVisitor):
    def __init__(self, file_path: str, lines: List[str]) -> None:
        self.file_path = file_path
        self.lines = lines
        self.snippets: List[Dict[str, Any]] = []

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Any:
        text = _joined_str_text(node)
        if _looks_like_sql(text):
            self._add_snippet(node, text)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if isinstance(node.op, ast.Add):
            text, _ = _binop_text(node)
            if text and _looks_like_sql(text):
                self._add_snippet(node, text)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if _is_format_call(node):
            base = _format_base_string(node)
            if base and _looks_like_sql(base):
                self._add_snippet(node, base)
        if _is_execute_call(node):
            arg = node.args[0] if node.args else None
            text = _string_from_node(arg)
            if text and _looks_like_sql(text):
                self._add_snippet(node, text)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, str):
            if _looks_like_sql(node.value):
                self._add_snippet(node, node.value)
        self.generic_visit(node)

    def _add_snippet(self, node: ast.AST, text: str) -> None:
        line = getattr(node, "lineno", None)
        self.snippets.append(
            {
                "file": self.file_path,
                "line": line,
                "query": _trim_sql(text),
            }
        )


def _looks_like_sql(text: str) -> bool:
    if SQL_START_RE.search(text):
        return True
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    return any(_token_is_keywordish(token) for token in tokens)


def _find_sql_typos(text: str, file_path: str, line: Optional[int], excerpt: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    tokens = [token.lower() for token in WORD_RE.findall(text)]

    for token in tokens:
        plural_keyword = _plural_keyword(token)
        if plural_keyword:
            findings.append(
                {
                    "file": file_path,
                    "line": line,
                    "type": "sql_typo",
                    "detail": f"Possible typo '{token}' -> '{plural_keyword}' (extra 's')",
                    "excerpt": excerpt,
                }
            )
            continue
        if token in SQL_TYPO_MAP:
            findings.append(
                {
                    "file": file_path,
                    "line": line,
                    "type": "sql_typo",
                    "detail": f"Possible typo '{token}' -> '{SQL_TYPO_MAP[token]}'",
                    "excerpt": excerpt,
                }
            )
            continue
        if token not in SQL_KEYWORDS and len(token) >= 4:
            close = get_close_matches(token, SQL_KEYWORDS, n=1, cutoff=0.84)
            if close:
                findings.append(
                    {
                        "file": file_path,
                        "line": line,
                        "type": "sql_typo",
                        "detail": f"Possible typo '{token}' -> '{close[0]}'",
                        "excerpt": excerpt,
                    }
                )

    return findings


def _token_is_keywordish(token: str) -> bool:
    if token in SQL_KEYWORDS or token in SQL_TYPO_MAP:
        return True
    if _plural_keyword(token):
        return True
    close = get_close_matches(token, SQL_KEYWORDS, n=1, cutoff=0.84)
    return bool(close)


def _plural_keyword(token: str) -> Optional[str]:
    if token.endswith("s") and token[:-1] in SQL_KEYWORDS:
        return token[:-1]
    return None


def _line_excerpt(lines: List[str], line: Optional[int]) -> str:
    if line is None or line <= 0 or line > len(lines):
        return ""
    return lines[line - 1].strip()


def _string_from_node(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return _joined_str_text(node)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        text, _ = _binop_text(node)
        return text
    return None


def _joined_str_text(node: ast.JoinedStr) -> str:
    parts: List[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        else:
            parts.append("?")
    return "".join(parts)


def _binop_text(node: ast.BinOp) -> tuple[Optional[str], bool]:
    left_text, left_dynamic = _binop_part(node.left)
    right_text, right_dynamic = _binop_part(node.right)
    if left_text is None or right_text is None:
        return None, True
    return left_text + right_text, left_dynamic or right_dynamic


def _binop_part(node: ast.AST) -> tuple[Optional[str], bool]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, False
    if isinstance(node, ast.JoinedStr):
        return _joined_str_text(node), _has_dynamic_part(node)
    return None, True


def _has_dynamic_part(node: ast.JoinedStr) -> bool:
    return any(isinstance(value, ast.FormattedValue) for value in node.values)


def _has_dynamic_expression(node: Optional[ast.AST]) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.JoinedStr):
        return _has_dynamic_part(node)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        _, dynamic = _binop_text(node)
        return dynamic
    if isinstance(node, ast.Call) and _is_format_call(node):
        return True
    return False


def _is_execute_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr in {"execute", "executemany"}
    return False


def _is_format_call(node: ast.Call) -> bool:
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr == "format"


def _format_base_string(node: ast.Call) -> Optional[str]:
    if not isinstance(node.func, ast.Attribute):
        return None
    if not isinstance(node.func.value, ast.Constant):
        return None
    if not isinstance(node.func.value.value, str):
        return None
    return node.func.value.value


def _trim_sql(text: str, limit: int = 400) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "...(truncated)"


def _trim_text(text: str, limit: int = 2000) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "...(truncated)"


def _extract_focus_terms(goal: str) -> List[str]:
    patterns = [
        r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\bmethod\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\bhandler\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s+function\b",
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s+handler\b",
    ]
    focus: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, goal, flags=re.IGNORECASE):
            focus.append(match.group(1))

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", goal)
    stopwords = {
        "review",
        "full",
        "function",
        "logic",
        "errors",
        "error",
        "file",
        "files",
        "code",
        "class",
        "method",
        "handler",
    }
    for token in tokens:
        if len(token) < 3:
            continue
        if token.lower() in stopwords:
            continue
        if token[0].islower() and any(char.isupper() for char in token):
            focus.append(token)
        elif "_" in token:
            focus.append(token)
        elif token[0].isupper() and any(char.islower() for char in token):
            focus.append(token)

    return _dedupe_preserve_order(focus)


def _focused_snippet(
    content: str,
    focus_terms: Sequence[str],
    max_chars: int,
) -> tuple[str, int]:
    cleaned = content.replace("\r\n", "\n").replace("\r", "\n")
    if not focus_terms:
        return _trim_text(cleaned, limit=max_chars), 1

    for term in focus_terms:
        index = cleaned.find(term)
        if index == -1:
            continue
        start = max(0, index - (max_chars // 2))
        end = min(len(cleaned), index + (max_chars // 2))
        snippet = cleaned[start:end]
        start_line = cleaned[:start].count("\n") + 1
        return _trim_text(snippet, limit=max_chars), start_line

    return _trim_text(cleaned, limit=max_chars), 1


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _dedupe_paths(items: Sequence[Path]) -> List[Path]:
    seen = set()
    ordered: List[Path] = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _numbered_snippet(snippet: str, start_line: int) -> str:
    lines = snippet.splitlines()
    numbered: List[str] = []
    line_no = start_line
    for line in lines:
        numbered.append(f"{line_no}: {line}")
        line_no += 1
    return "\n".join(numbered)
