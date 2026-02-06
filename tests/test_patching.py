import shutil
from pathlib import Path

import pytest

from tools.patching import (
    apply_patch_with_fallback,
    create_workspace,
    validate_patch,
    write_patch,
)


def test_validate_patch_blocks_env() -> None:
    patch_text = """diff --git a/.env b/.env
--- a/.env
+++ b/.env
@@
+SECRET=1
"""
    violations = validate_patch(patch_text)
    assert ".env" in violations


def test_apply_patch_to_workspace(tmp_path: Path) -> None:
    if not (shutil.which("git") or shutil.which("patch")):
        pytest.skip("git or patch is required for patch application")

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "foo.txt").write_text("", encoding="utf-8")

    patch_text = """diff --git a/foo.txt b/foo.txt
index e69de29..0fef1c1 100644
--- a/foo.txt
+++ b/foo.txt
@@ -0,0 +1 @@
+hello
"""
    workspace = create_workspace(str(repo), tmp_path / "workspaces")
    patch_path = write_patch(patch_text, tmp_path / "patches", "run-1")
    applied, _ = apply_patch_with_fallback(patch_path, workspace)
    assert applied is True
    assert (Path(workspace) / "foo.txt").read_text(encoding="utf-8").strip() == "hello"
