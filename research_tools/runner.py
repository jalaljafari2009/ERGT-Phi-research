"""Execute a locked Colab/local protocol and export evidence, including failures.

No remote service is contacted here. Run with ``python -m research_tools.runner``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_path(root: Path, relative: str) -> Path:
    """Require a portable repository-relative path without traversal or symlinks."""
    posix = PurePosixPath(relative)
    if not relative or "\\" in relative or ":" in relative or posix.is_absolute() or any(
        part in {"", "..", "."} for part in relative.split("/")
    ):
        raise ValueError(f"Unsafe relative path: {relative!r}")
    root = root.resolve()
    path = root.joinpath(*posix.parts)
    current = root
    for part in posix.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink is not allowed: {relative}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes root: {relative}")
    return path


def environment() -> dict:
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "colab_detected": bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION")),
        "packages": {},
    }
    for name in ("numpy", "pandas", "torch", "pytest", "nbformat"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        result["gpu_inventory"] = gpu.stdout.strip() if gpu.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        result["gpu_inventory"] = None
    return result


def _json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _verify_package(root: Path, revision_dir: Path, package_sha256: str) -> dict:
    package = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    protocol = json.loads((revision_dir / "protocol.json").read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{64}", package_sha256):
        raise ValueError("A SHA-256 for the uploaded source archive is required")
    for name in ("experiment_id", "revision"):
        if package.get(name) != protocol.get(name):
            raise ValueError(f"Package/protocol {name} mismatch")
    for name, filename in (("protocol_sha256", "protocol.json"), ("notebook_sha256", "experiment.ipynb")):
        if sha256(revision_dir / filename) != package[name]:
            raise ValueError(f"Package {name} mismatch")
    sources = package.get("files")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Package has no source inventory")
    for relative, expected in sources.items():
        path = safe_path(root, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Source integrity failed: {relative}")
    return package


def run_protocol(
    root: Path, revision_dir: Path, output: Path, *, run_id: str, package_sha256: str,
) -> dict:
    """Return manifest; write RUN_MANIFEST.json and payload into a new result ZIP.

    Preconditions and execution failures are evidence, not scientific acceptance.
    The repository importer separately checks the package hash against its lock.
    """
    root, revision_dir, output = root.resolve(), revision_dir.resolve(), output.resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_id):
        raise ValueError("Invalid run_id")
    if output.exists():
        raise FileExistsError(f"Run bundle already exists: {output}")
    if not revision_dir.is_relative_to(root):
        raise ValueError("Revision must be inside the project root")
    output.parent.mkdir(parents=True, exist_ok=True)
    started, started_clock = now(), time.perf_counter()
    protocol = {}
    package = {}
    declared_outputs = []
    can_collect_outputs = False
    result = {
        "schema": "ergt-phi-run-v1", "run_id": run_id,
        "experiment_id": None, "revision": None,
        "protocol_sha256": None, "notebook_sha256": None,
        "package_sha256": package_sha256, "source_commit": None,
        "command": [], "started_at": started, "finished_at": None,
        "returncode": None, "status": "preflight_failed",
        "environment": environment(), "artifacts": [],
        "missing_required_artifacts": [], "error": None,
    }
    with tempfile.TemporaryDirectory(prefix="ergt-evidence-", dir=output.parent) as temp:
        work = Path(temp)
        payload = work / "payload"
        logs = payload / "runner"
        logs.mkdir(parents=True)
        log_path = logs / "stdout_stderr.log"
        log_path.write_text("", encoding="utf-8")
        try:
            protocol_path = revision_dir / "protocol.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            result.update(
                experiment_id=protocol["experiment_id"], revision=protocol["revision"],
                protocol_sha256=sha256(protocol_path),
                notebook_sha256=sha256(revision_dir / "experiment.ipynb"),
            )
            package = _verify_package(root, revision_dir, package_sha256)
            result["source_commit"] = package["source_commit"]
            raw_command = protocol["command"]
            if not isinstance(raw_command, list) or not raw_command or not all(
                isinstance(arg, str) and arg for arg in raw_command
            ):
                raise ValueError("Protocol command must be a nonempty argv list")
            command = [sys.executable if arg == "{python}" else arg for arg in raw_command]
            # Preserve the exact preregistered command; record runtime resolution separately.
            result["command"] = raw_command
            result["executed_command"] = command
            for item in protocol.get("inputs", []):
                path = safe_path(root, item["path"])
                if not path.is_file() or sha256(path) != item["sha256"]:
                    raise ValueError(f"Missing input or SHA mismatch: {item['path']}")
            for relative in protocol.get("output_paths", []):
                path = safe_path(root, relative)
                if relative == "runner" or relative.startswith("runner/"):
                    raise ValueError("Output prefix runner/ is reserved for execution evidence")
                if path.exists():
                    raise FileExistsError(f"Stale output exists; start a clean runtime: {relative}")
                declared_outputs.append((relative, path))
            for relative in protocol.get("required_artifacts", []):
                safe_path(root, relative)
                if not any(relative == item or relative.startswith(item + "/") for item, _ in declared_outputs):
                    raise ValueError(f"Required artifact lies outside output_paths: {relative}")
            can_collect_outputs = True
            env = os.environ.copy()
            env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
            with log_path.open("a", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    command, cwd=root, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                )
                timed_out = []
                timer = None
                timeout = protocol.get("timeout_seconds")
                if timeout is not None:
                    if not isinstance(timeout, (float, int)) or timeout <= 0:
                        process.terminate()
                        process.wait()
                        raise ValueError("timeout_seconds must be positive")
                    def terminate_on_timeout():
                        timed_out.append(True)
                        process.terminate()
                    timer = threading.Timer(timeout, terminate_on_timeout)
                    timer.daemon = True
                    timer.start()
                try:
                    assert process.stdout is not None
                    for line in process.stdout:
                        stream.write(line)
                        stream.flush()
                        print(line, end="", flush=True)
                    result["returncode"] = process.wait()
                    if timed_out:
                        raise TimeoutError(f"Execution exceeded {timeout} seconds")
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
                finally:
                    if timer is not None:
                        timer.cancel()
            result["status"] = "execution_completed" if result["returncode"] == 0 else "execution_failed"
            if result["returncode"] != 0:
                result["error"] = f"Command exited with code {result['returncode']}"
        except (Exception, KeyboardInterrupt) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            if can_collect_outputs:
                result["status"] = "execution_failed"
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write("\n" + result["error"] + "\n")
        # A rejected preflight must never export stale outputs as new evidence.
        if can_collect_outputs:
            try:
                for relative, path in declared_outputs:
                    if not path.exists():
                        continue
                    members = [path] if path.is_file() else sorted(path.rglob("*"))
                    for source in members:
                        if source.is_symlink():
                            raise ValueError(f"Refusing output symlink: {source}")
                        if not source.is_file():
                            continue
                        relative_file = source.relative_to(root).as_posix()
                        source = safe_path(root, relative_file)
                        target = payload / relative_file
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
            except Exception as exc:
                result["status"] = "execution_failed"
                result["error"] = f"Artifact collection failed: {type(exc).__name__}: {exc}"
        result["missing_required_artifacts"] = [
            relative for relative in protocol.get("required_artifacts", [])
            if not (payload / relative).is_file()
        ]
        result["finished_at"] = now()
        result["seconds"] = time.perf_counter() - started_clock
        _json(logs / "environment.json", result["environment"])
        for path in sorted(payload.rglob("*")):
            if path.is_file():
                result["artifacts"].append({
                    "path": path.relative_to(payload).as_posix(),
                    "sha256": sha256(path), "bytes": path.stat().st_size,
                })
        _json(work / "RUN_MANIFEST.json", result)
        # Exclusive creation prevents accidental replacement of a previous run.
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(work / "RUN_MANIFEST.json", "RUN_MANIFEST.json")
            for item in result["artifacts"]:
                archive.write(payload / item["path"], "payload/" + item["path"])
    print(json.dumps({"status": result["status"], "result_bundle": str(output)}, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_protocol(args.root, args.revision_dir, args.output,
                          run_id=args.run_id, package_sha256=args.package_sha256)
    if result["status"] == "execution_completed":
        return 0
    return result["returncode"] or 2


if __name__ == "__main__":
    raise SystemExit(main())
