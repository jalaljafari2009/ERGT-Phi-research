"""SHA-256 integrity checks for the standalone release."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_integrity(package_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(package_root or Path(__file__).resolve().parents[1]).resolve()
    manifest_path = root / "MANIFEST.sha256"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing integrity manifest: {manifest_path}")
    expected: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    missing = sorted(relative for relative in expected if not (root / relative).is_file())
    mismatched = sorted(
        relative
        for relative, digest in expected.items()
        if (root / relative).is_file() and _sha256(root / relative) != digest
    )
    if missing or mismatched:
        raise RuntimeError(
            "release integrity failure; missing=" + repr(missing) + ", mismatched=" + repr(mismatched)
        )
    return {
        "pass": True,
        "package_root": str(root),
        "verified_file_count": len(expected),
        "manifest_sha256": _sha256(manifest_path),
    }
