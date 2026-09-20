"""The guard must reject altered/missing baselines without repairing them."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.check_spec_lock import verify_spec_lock


def fixture(root):
    (root / "docs").mkdir()
    (root / "research").mkdir()
    target = root / "docs/MATHEMATICAL_SPEC.md"
    target.write_bytes(b"immutable fixture\r\n")
    lock = {"schema": "ergt-phi-specification-lock-v1", "path": "docs/MATHEMATICAL_SPEC.md",
            "bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (root / "research/specification.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    return target


def test_intact_baseline_is_read_without_changes(tmp_path):
    target = fixture(tmp_path)
    before = target.read_bytes()
    assert verify_spec_lock(tmp_path)["pass"] is True
    assert target.read_bytes() == before


@pytest.mark.parametrize("replacement", [b"immutable fixture\n", b"Immutable fixture\r\n"])
def test_newline_or_same_size_content_change_is_rejected(tmp_path, replacement):
    target = fixture(tmp_path)
    target.write_bytes(replacement)
    with pytest.raises(ValueError, match="differs"):
        verify_spec_lock(tmp_path)
    assert target.read_bytes() == replacement  # No silent reset/reseal.


def test_missing_baseline_is_not_recreated(tmp_path):
    target = fixture(tmp_path)
    target.unlink()
    with pytest.raises(FileNotFoundError):
        verify_spec_lock(tmp_path)
    assert not target.exists()


def test_lock_cannot_redirect_check_to_another_document(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "research/specification.lock.json"
    lock = json.loads(path.read_text())
    lock["path"] = "docs/another.md"
    path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        verify_spec_lock(tmp_path)


def test_workflow_cli_stops_before_creating_any_experiment(tmp_path):
    target = fixture(tmp_path)
    target.write_bytes(b"changed")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    repo = Path(__file__).resolve().parents[1]
    for name in ("research.py", "check_spec_lock.py"):
        (scripts / name).write_bytes((repo / "scripts" / name).read_bytes())
    # A tiny import fixture proves main cannot be reached after guard failure.
    package = tmp_path / "research_tools"
    package.mkdir()
    (package / "workflow.py").write_text(
        "def main(*, root):\n    (root / 'unexpected-main-call').touch()\n    return 0\n", encoding="utf-8")
    result = subprocess.run([sys.executable, "-B", str(scripts / "research.py"), "status"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Specification lock failed" in result.stderr
    assert not (tmp_path / "unexpected-main-call").exists()


def test_workflow_cli_verifies_the_root_override(tmp_path):
    target = fixture(tmp_path)
    target.write_bytes(b"changed override")
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-B", str(repo / "scripts/research.py"),
                             "--root", str(tmp_path), "status"],
                            cwd=repo, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Specification lock failed" in result.stderr
