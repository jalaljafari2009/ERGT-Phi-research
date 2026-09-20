"""Verify the immutable mathematical baseline without modifying any file."""

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = "docs/MATHEMATICAL_SPEC.md"
LOCK_PATH = "research/specification.lock.json"


def verify_spec_lock(root=ROOT, *, require_read_only=False):
    root = Path(root).resolve()
    lock = json.loads((root / LOCK_PATH).read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or lock.get("schema") != "ergt-phi-specification-lock-v1":
        raise ValueError("Unknown specification lock schema")
    if lock.get("path") != SPEC_PATH:
        raise ValueError("The specification lock must identify the canonical baseline")
    target = root / SPEC_PATH
    if target.is_symlink() or target.resolve() != root / SPEC_PATH:
        raise ValueError("The specification must be a regular file at its canonical path")
    data = target.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != lock.get("bytes") or digest != lock.get("sha256"):
        raise ValueError("Mathematical specification differs from the locked baseline; do not reseal it")
    metadata = target.stat()
    if hasattr(metadata, "st_file_attributes"):
        read_only = bool(metadata.st_file_attributes & stat.FILE_ATTRIBUTE_READONLY)
    else:
        read_only = not bool(metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    if require_read_only and not read_only:
        raise ValueError("Content is intact but local read-only protection has not been applied")
    return {"pass": True, "path": SPEC_PATH, "bytes": len(data),
            "sha256": digest, "read_only": read_only,
            "acl_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-read-only", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_spec_lock(require_read_only=args.require_read_only)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"pass": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
