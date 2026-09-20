"""Resolve relocated historical evidence separately from new scratch outputs.

Historical mappings are authoritative: a missing archived checkpoint must fail
instead of silently selecting a different file left at its former location.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError(f"Expected a repository-relative POSIX path: {relative!r}")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or any(p in {".", ".."} for p in relative.split("/")):
        raise ValueError(f"Unsafe repository-relative path: {relative!r}")
    root = root.resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Path escapes repository: {relative!r}")
    return result


def legacy_input(root: Path, old_relative: str) -> Path:
    """Return archived evidence, or its old location only if it was never mapped."""
    root = Path(root).resolve()
    _inside(root, old_relative)
    mapping_path = root / "research/legacy/path_map.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}
    if mapping and mapping.get("schema") != "ergt-phi-legacy-path-map-v1":
        raise ValueError("Unsupported legacy path-map schema")
    destination = mapping.get("files", {}).get(old_relative)
    if destination is None:
        for old_dir, new_dir in sorted(mapping.get("directories", {}).items(), key=lambda row: len(row[0]), reverse=True):
            _inside(root, old_dir)
            if old_relative == old_dir or old_relative.startswith(old_dir + "/"):
                destination = new_dir + old_relative[len(old_dir):]
                break
    path = _inside(root, destination if destination is not None else old_relative)
    if not path.exists():
        raise FileNotFoundError(f"Required evidence is unavailable: {old_relative} -> {path.relative_to(root).as_posix()}")
    return path


def workspace_path(root: Path, relative: str) -> Path:
    """Return a generated-output path, creating its parents outside the archive."""
    root = Path(root).resolve()
    _inside(root, relative)
    destination = _inside(root, "research/workspace/" + relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def current_input(root: Path, old_relative: str) -> Path:
    """Prefer a fresh working result; otherwise read its preserved predecessor.

Use this for explicit finalization of scratch outputs. Accepted scientific
inputs must use legacy_input so a scratch rerun cannot change their identity.
"""
    root = Path(root).resolve()
    work = _inside(root, "research/workspace/" + old_relative)
    if work.exists():
        return work
    original = _inside(root, old_relative)
    if original.exists():
        return original
    return legacy_input(root, old_relative)
