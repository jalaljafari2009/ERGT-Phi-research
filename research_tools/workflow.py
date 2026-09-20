"""Versioned local experiment lifecycle, verified result import, and reminders.

No model code, training framework, or cloud account is imported here. Paths in
``registry.json`` are relative to ``research/``; artifact paths are relative to
the repository. The protocol and package lock are the authority for a run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any
import uuid
import zipfile


PROTOCOL_SCHEMA = "ergt-phi-protocol-v1"
RUN_SCHEMA = "ergt-phi-run-v1"
REGISTRY_SCHEMA = "ergt-phi-registry-v1"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
REVISION = re.compile(r"v[0-9]{3,}\Z")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".log", ".csv", ".tsv", ".yaml", ".yml"}
MAX_TRACKED_BYTES = 2 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = 20 * 1024 * 1024 * 1024
MAX_BUNDLE_ENTRIES = 100000


class WorkflowError(ValueError):
    """An actionable contract error; the workflow must not silently continue."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"),
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              WorkflowError(f"Non-finite JSON number: {value}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(_json_bytes(value))
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value: Any, label: str = "identifier") -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise WorkflowError(f"Invalid {label}: {value!r}")
    if value.split(".", 1)[0].upper() in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    } or value.endswith("."):
        raise WorkflowError(f"Reserved filesystem {label}: {value!r}")
    return value


def _relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise WorkflowError(f"Expected canonical relative POSIX path: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} or part.endswith((" ", "."))
           or any(ord(char) < 32 for char in part) for part in parts):
        raise WorkflowError(f"Unsafe relative path: {value!r}")
    for part in parts:
        if part.split(".", 1)[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        } or any(char in part for char in '*?"<>|'):
            raise WorkflowError(f"Unsafe Windows path component: {part!r}")
    return PurePosixPath(value).as_posix()


def _contained(base: Path, relative: str) -> Path:
    path = base.joinpath(*_relative(relative).split("/"))
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError as exc:
        raise WorkflowError(f"Path escapes {base}: {relative}") from exc
    return path


def _required_text(value: dict, key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise WorkflowError(f"Protocol requires nonempty {key}")
    return result


def validate_protocol(protocol: dict) -> dict:
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise WorkflowError(f"Expected protocol schema {PROTOCOL_SCHEMA}")
    _identifier(protocol.get("experiment_id"), "experiment_id")
    if not isinstance(protocol.get("revision"), str) or not REVISION.fullmatch(protocol["revision"]):
        raise WorkflowError("Revision must have format v001")
    if protocol.get("stage") not in {"workflow", *(f"M{i}" for i in range(9))}:
        raise WorkflowError("Stage must be workflow or M0..M8")
    if protocol.get("kind") not in {"infrastructure", "research"}:
        raise WorkflowError("Kind must be infrastructure or research")
    for key in ("title", "hypothesis", "goal_reference", "created_at"):
        _required_text(protocol, key)
    parent = protocol.get("parent")
    if parent is not None:
        if not isinstance(parent, dict):
            raise WorkflowError("parent must be an experiment/revision object or null")
        _identifier(parent.get("experiment_id"), "parent experiment")
        if not REVISION.fullmatch(str(parent.get("revision", ""))):
            raise WorkflowError("Invalid parent revision")
    command = protocol.get("command")
    if not isinstance(command, list) or not command or any(
        not isinstance(part, str) or not part or "\x00" in part for part in command
    ):
        raise WorkflowError("command must be a nonempty argv array, never a shell string")
    for key in ("output_paths", "required_artifacts", "inputs", "gates"):
        if not isinstance(protocol.get(key), list):
            raise WorkflowError(f"Protocol {key} must be an array")
    output_paths = [_relative(path) for path in protocol["output_paths"]]
    if len(set(path.casefold() for path in output_paths)) != len(output_paths):
        raise WorkflowError("Duplicate output_paths")
    required = [_relative(path) for path in protocol["required_artifacts"]]
    if len(set(path.casefold() for path in required)) != len(required):
        raise WorkflowError("Duplicate required_artifacts")
    def covered(path: str) -> bool:
        return any(path == output or path.startswith(output + "/") for output in output_paths)
    if any(not covered(path) for path in required):
        raise WorkflowError("Each required artifact must be covered by output_paths")
    for item in protocol["inputs"]:
        if not isinstance(item, dict):
            raise WorkflowError("inputs entries must be objects")
        _relative(item.get("path"))
        if not SHA256.fullmatch(str(item.get("sha256", ""))):
            raise WorkflowError("Every input requires its SHA-256 before packaging")
        if not isinstance(item.get("uri", ""), str):
            raise WorkflowError("Input URI must be a string")
    gate_names = set()
    for gate in protocol["gates"]:
        if not isinstance(gate, dict):
            raise WorkflowError("Gate entries must be objects")
        name = _required_text(gate, "name")
        if name in gate_names:
            raise WorkflowError(f"Duplicate gate name: {name}")
        gate_names.add(name)
        artifact = _relative(gate.get("artifact"))
        if not covered(artifact):
            raise WorkflowError(f"Gate artifact is outside output_paths: {artifact}")
        pointer = gate.get("pointer")
        if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
            raise WorkflowError(f"Gate {name} needs a JSON pointer")
        if re.search(r"~(?![01])", pointer):
            raise WorkflowError(f"Invalid JSON pointer escape in {name}")
        if gate.get("op") not in {"eq", "ge", "le"} or "value" not in gate:
            raise WorkflowError(f"Gate {name} needs op eq/ge/le and value")
        value = gate["value"]
        if gate["op"] in {"ge", "le"} and not _finite_number(value):
            raise WorkflowError(f"Gate {name} threshold must be a finite number")
    _json_bytes(protocol)  # Reject non-JSON or non-finite custom metadata, too.
    return protocol


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _research(root: Path) -> Path:
    return Path(root).resolve() / "research"


def load_registry(root: Path) -> dict:
    path = _research(root) / "registry.json"
    if not path.exists():
        return {"schema": REGISTRY_SCHEMA, "experiments": {}}
    value = _read_json(path)
    if value.get("schema") != REGISTRY_SCHEMA or not isinstance(value.get("experiments"), dict):
        raise WorkflowError("Invalid research/registry.json")
    return value


def revision_path(root: Path, experiment_id: str, revision: str) -> Path:
    _identifier(experiment_id, "experiment_id")
    if not REVISION.fullmatch(revision):
        raise WorkflowError("Invalid revision")
    registry = load_registry(root)
    try:
        relative = registry["experiments"][experiment_id]["revisions"][revision]
    except KeyError as exc:
        raise WorkflowError(f"Unknown registered revision {experiment_id}/{revision}") from exc
    expected = f"experiments/{experiment_id}/{revision}"
    if relative != expected:
        raise WorkflowError(f"Registry path must be {expected}, got {relative!r}")
    return _contained(_research(root), relative)


def record_event(root: Path, kind: str, message: str, **details: Any) -> dict:
    if not kind.strip() or not message.strip():
        raise WorkflowError("Progress events need a type and message")
    event = {"schema": "ergt-phi-progress-v1", "event_id": uuid.uuid4().hex,
             "created_at": utc_now(), "kind": kind, "message": message, **details}
    path = _research(root) / "progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    return event


def create_experiment(root: Path, protocol: dict) -> Path:
    protocol = json.loads(json.dumps(protocol))
    protocol.setdefault("schema", PROTOCOL_SCHEMA)
    protocol.setdefault("revision", "v001")
    protocol.setdefault("created_at", utc_now())
    protocol.setdefault("parent", None)
    validate_protocol(protocol)
    experiment_id, revision = protocol["experiment_id"], protocol["revision"]
    registry = load_registry(root)
    existing = registry["experiments"].get(experiment_id)
    if existing is not None and revision in existing["revisions"]:
        raise WorkflowError(f"Revision already registered: {experiment_id}/{revision}; use revise")
    parent = protocol["parent"]
    if parent:
        revision_path(root, parent["experiment_id"], parent["revision"])
    if existing and not parent:
        raise WorkflowError("Further revisions must cite a parent; use revise")
    relative = f"experiments/{experiment_id}/{revision}"
    directory = _contained(_research(root), relative)
    if directory.exists():
        raise WorkflowError(f"Refusing to overwrite existing directory {directory}")
    directory.mkdir(parents=True)
    _write_json(directory / "protocol.json", protocol)
    (directory / "README.md").write_text(
        f"# {protocol['title']}\n\n"
        f"Experiment: `{experiment_id}/{revision}` · stage `{protocol['stage']}` · "
        f"kind `{protocol['kind']}`.\n\n"
        f"Hypothesis: {protocol['hypothesis']}\n\n"
        f"Goal reference: {protocol['goal_reference']}\n\n"
        "- [Preregistered protocol](protocol.json)\n"
        "- Notebook: `experiment.ipynb` (generate with the workflow CLI).\n"
        "- Immutable release lock: `package.json` (created when packaged).\n"
        "- Imported evidence: `runs/<run_id>/run.json`, `artifacts.json`, `summary.md`.\n"
        "- Reviewed decisions: `runs/<run_id>/reviews/*.json`, linked to research/decisions.\n\n"
        "An execution success is not a scientific pass. Review gates and limitations before proceeding.\n",
        encoding="utf-8", newline="\n",
    )
    registry["experiments"].setdefault(experiment_id, {"revisions": {}})["revisions"][revision] = relative
    _write_json(_research(root) / "registry.json", registry)
    record_event(root, "experiment_created", f"Registered {experiment_id}/{revision}",
                 experiment_id=experiment_id, revision=revision, path=f"research/{relative}")
    return directory


def revise_experiment(root: Path, experiment_id: str, from_revision: str, protocol: dict) -> Path:
    revision_path(root, experiment_id, from_revision)
    versions = load_registry(root)["experiments"][experiment_id]["revisions"]
    next_revision = f"v{max(int(version[1:]) for version in versions) + 1:03d}"
    revised = dict(protocol)
    revised.update({"experiment_id": experiment_id, "revision": next_revision,
                    "parent": {"experiment_id": experiment_id, "revision": from_revision},
                    "created_at": utc_now()})
    if not isinstance(revised.get("revision_reason"), str) or not revised["revision_reason"].strip():
        raise WorkflowError("A revision requires revision_reason documenting the changed strategy")
    return create_experiment(root, revised)


def build_notebook(root: Path, experiment_id: str, revision: str) -> Path:
    from research_tools.notebooks import generate_notebook
    directory = revision_path(root, experiment_id, revision)
    validate_protocol(_read_json(directory / "protocol.json"))
    result = generate_notebook(Path(root), directory)
    record_event(root, "notebook_generated", f"Generated notebook for {experiment_id}/{revision}",
                 experiment_id=experiment_id, revision=revision)
    return Path(result)


def build_package(root: Path, experiment_id: str, revision: str) -> Path:
    from research_tools.notebooks import build_source_package
    directory = revision_path(root, experiment_id, revision)
    validate_protocol(_read_json(directory / "protocol.json"))
    result = build_source_package(Path(root), directory)
    record_event(root, "package_built", f"Packaged {experiment_id}/{revision}",
                 experiment_id=experiment_id, revision=revision,
                 package_sha256=_read_json(directory / "package.json")["package_sha256"])
    return Path(result)


def _pointer(document: Any, pointer: str) -> Any:
    value = document
    if not pointer:
        return value
    for token in pointer[1:].split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value[key]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", key):
            value = value[int(key)]
        else:
            raise KeyError(pointer)
    return value


def evaluate_run(protocol: dict, manifest: dict, artifact_files: dict[str, Path]) -> dict:
    """Compute gates only from verified files. This never grants stage acceptance."""
    missing = sorted(set(protocol["required_artifacts"]) - set(artifact_files))
    gates = []
    for gate in protocol["gates"]:
        result = dict(gate)
        result["observed"] = None
        file = artifact_files.get(gate["artifact"])
        if file is None:
            result.update(status="missing", explanation="Verified artifact is absent")
        elif file.stat().st_size > MAX_JSON_BYTES:
            result.update(status="invalid", explanation="Gate JSON exceeds 16 MiB limit")
        else:
            try:
                observed = _pointer(_read_json(file), gate["pointer"])
                result["observed"] = observed
                if gate["op"] == "eq":
                    passed = type(observed) is type(gate["value"]) and observed == gate["value"]
                    if _finite_number(observed) and _finite_number(gate["value"]):
                        passed = observed == gate["value"]
                elif not _finite_number(observed):
                    raise WorkflowError("Observed value is not a finite number")
                elif gate["op"] == "ge":
                    passed = observed >= gate["value"]
                else:
                    passed = observed <= gate["value"]
                result.update(status="pass" if passed else "fail", explanation="Threshold comparison")
            except (WorkflowError, KeyError, IndexError, TypeError) as exc:
                result.update(status="invalid", explanation=str(exc))
        gates.append(result)
    success = manifest["returncode"] == 0 and manifest["status"] == "execution_completed"
    all_pass = bool(gates) and all(gate["status"] == "pass" for gate in gates)
    return {"schema": "ergt-phi-evaluation-v1", "execution_success": success,
            "required_artifacts_present": not missing, "missing_required_artifacts": missing,
            "gates": gates, "all_gates_pass": all_pass,
            "eligible_for_pass": success and not missing and all_pass,
            "scientific_phase_pass": False, "kind": protocol["kind"],
            "limitation": "Recorded gates require a linked review; no stage is promoted automatically."}


def _validate_zip(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    inventory = {}
    folded = set()
    entries = archive.infolist()
    if len(entries) > MAX_BUNDLE_ENTRIES or sum(item.file_size for item in entries) > MAX_BUNDLE_BYTES:
        raise WorkflowError("Bundle exceeds configured entry or uncompressed-size limit")
    for item in entries:
        name = item.filename.rstrip("/") if item.is_dir() else item.filename
        _relative(name)
        if name.casefold() in folded:
            raise WorkflowError(f"Duplicate or case-colliding ZIP member: {name}")
        folded.add(name.casefold())
        mode = (item.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (
            stat.S_ISREG(mode) or stat.S_ISDIR(mode)
        )) or item.flag_bits & 1:
            raise WorkflowError(f"Unsafe or encrypted ZIP member: {name}")
        if not item.is_dir():
            inventory[name] = item
    return inventory


def _validate_manifest(manifest: dict, protocol: dict, lock: dict, inventory: dict) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema") != RUN_SCHEMA:
        raise WorkflowError(f"Expected bundle manifest schema {RUN_SCHEMA}")
    _identifier(manifest.get("run_id"), "run_id")
    for key in ("experiment_id", "revision"):
        if manifest.get(key) != protocol[key]:
            raise WorkflowError(f"Manifest {key} does not match the registered protocol")
    for key in ("protocol_sha256", "notebook_sha256", "package_sha256"):
        if not SHA256.fullmatch(str(manifest.get(key, ""))) or manifest[key] != lock.get(key):
            raise WorkflowError(f"Manifest {key} does not match the immutable package lock")
    if manifest.get("source_commit") != lock.get("source_commit"):
        raise WorkflowError("Manifest source_commit does not match the package lock")
    if manifest.get("command") != protocol["command"]:
        raise WorkflowError("Manifest command does not match the preregistered command")
    if manifest.get("status") not in {"execution_completed", "execution_failed", "preflight_failed"}:
        raise WorkflowError("Unknown execution status")
    returncode = manifest.get("returncode")
    if returncode is not None and (not isinstance(returncode, int) or isinstance(returncode, bool)):
        raise WorkflowError("returncode must be an integer or null")
    if manifest["status"] == "execution_completed" and (returncode != 0 or manifest.get("error")):
        raise WorkflowError("Completed status contradicts return code or error")
    if manifest["status"] == "execution_failed" and returncode in {None, 0} and not manifest.get("error"):
        raise WorkflowError("Failed execution needs a nonzero return code or an explicit wrapper error")
    for key in ("started_at", "finished_at"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise WorkflowError(f"Manifest needs {key}")
    if not isinstance(manifest.get("environment"), dict):
        raise WorkflowError("Manifest needs environment metadata")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise WorkflowError("Manifest artifacts must be an array")
    paths = set()
    folded = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise WorkflowError("Artifact entries must be objects")
        path = _relative(artifact.get("path"))
        if path.casefold() in folded:
            raise WorkflowError(f"Duplicate artifact: {path}")
        folded.add(path.casefold())
        paths.add(path)
        if not SHA256.fullmatch(str(artifact.get("sha256", ""))):
            raise WorkflowError(f"Invalid artifact hash: {path}")
        size = artifact.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise WorkflowError(f"Invalid artifact size: {path}")
        entry = inventory.get("payload/" + path)
        if entry is None or entry.file_size != size:
            raise WorkflowError(f"Artifact missing or size mismatch: {path}")
    expected = {"RUN_MANIFEST.json", *("payload/" + path for path in paths)}
    if set(inventory) != expected:
        raise WorkflowError("ZIP contents do not exactly match the declared artifact inventory")
    declared_missing = manifest.get("missing_required_artifacts")
    computed_missing = sorted(set(protocol["required_artifacts"]) - paths)
    if not isinstance(declared_missing, list) or sorted(declared_missing) != computed_missing:
        raise WorkflowError("Declared missing artifacts disagree with the verified inventory")


def _verify_release(directory: Path, protocol: dict, lock: dict) -> None:
    for key, filename in (("protocol_sha256", "protocol.json"), ("notebook_sha256", "experiment.ipynb")):
        path = directory / filename
        if not path.is_file() or sha256_file(path) != lock.get(key):
            raise WorkflowError(f"Released {filename} changed; restore it or create a new revision")
    for key in ("experiment_id", "revision"):
        if key in lock and lock[key] != protocol[key]:
            raise WorkflowError(f"Package lock {key} does not match its directory")


def _tracked_artifact(file: Path) -> bool:
    if file.suffix.lower() not in TEXT_SUFFIXES or file.stat().st_size > MAX_TRACKED_BYTES:
        return False
    try:
        content = file.read_text(encoding="utf-8")
        return "\x00" not in content
    except UnicodeError:
        return False


def _summary(protocol: dict, manifest: dict, evaluation: dict) -> str:
    lines = [f"# Run {manifest['run_id']}", "",
             f"Experiment: `{protocol['experiment_id']}/{protocol['revision']}`; "
             f"stage `{protocol['stage']}`; kind `{protocol['kind']}`.", "",
             f"Hypothesis: {protocol['hypothesis']}", "",
             f"Execution: `{manifest['status']}`, return code `{manifest['returncode']}`.",
             f"Started: {manifest['started_at']}; finished: {manifest['finished_at']}.", "",
             "| Preregistered gate | Observed | Expected | Status |",
             "|---|---|---|---|"]
    for gate in evaluation["gates"]:
        observed = json.dumps(gate["observed"], ensure_ascii=False).replace("|", "\\|")
        expected = json.dumps(gate["value"], ensure_ascii=False).replace("|", "\\|")
        lines.append(f"| {gate['name'].replace('|', '/')} | `{observed}` | "
                     f"{gate['op']} `{expected}` | {gate['status']} |")
    if not evaluation["gates"]:
        lines.append("| No preregistered gates | — | — | cannot pass |")
    lines.extend(["", f"Missing required artifacts: {evaluation['missing_required_artifacts']}.", "",
                  f"Eligible for explicit review as pass: `{evaluation['eligible_for_pass']}`.", "",
                  "This is a deterministic evidence summary, not a scientific conclusion. "
                  "Execution success does not prove the hypothesis. Record interpretation, "
                  "limitations, alternatives, and the next action in a linked decision.", "",
                  "- [Original run manifest](run.json)", "- [Artifact locations and hashes](artifacts.json)",
                  "- [Machine gate evaluation](evaluation.json)",
                  "- Reviews: `reviews/*.json`; linked architecture decisions: `research/decisions/`.", ""])
    return "\n".join(lines)


def _record_location(root: Path, run: Path, source_uri: str, bundle_hash: str) -> None:
    if not source_uri:
        return
    existing = sorted((run / "locations").glob("*.json"))
    if any(_read_json(path).get("source_uri") == source_uri for path in existing):
        return
    _write_json(run / "locations" / f"{len(existing) + 1:03d}.json", {
        "schema": "ergt-phi-artifact-location-v1", "source_uri": source_uri,
        "bundle_sha256": bundle_hash, "recorded_at": utc_now(),
        "remote_availability_verified": False,
    })
    record_event(root, "artifact_location_recorded", f"Recorded archive locator for {run.name}",
                 source_uri=source_uri, bundle_sha256=bundle_hash)


def import_bundle(root: Path, bundle: Path, source_uri: str = "") -> Path:
    """Verify first, then import. ZIP paths never control an extraction location."""
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    if not bundle.is_file():
        raise WorkflowError(f"Bundle does not exist: {bundle}")
    try:
        with zipfile.ZipFile(bundle) as archive:
            inventory = _validate_zip(archive)
            info = inventory.get("RUN_MANIFEST.json")
            if info is None or info.file_size > MAX_JSON_BYTES:
                raise WorkflowError("Bundle needs a reasonably sized RUN_MANIFEST.json")
            manifest = json.loads(archive.read(info).decode("utf-8"))
            directory = revision_path(root, manifest.get("experiment_id", ""), manifest.get("revision", ""))
            protocol = validate_protocol(_read_json(directory / "protocol.json"))
            lock = _read_json(directory / "package.json")
            _verify_release(directory, protocol, lock)
            _validate_manifest(manifest, protocol, lock, inventory)
            run_id = manifest["run_id"]
            target = _contained(directory, f"runs/{run_id}")
            external = _contained(_research(root),
                                  f"artifacts/{protocol['experiment_id']}/{protocol['revision']}/{run_id}")
            bundle_hash = sha256_file(bundle)
            if target.exists():
                stored = _read_json(target / "artifacts.json")
                if _read_json(target / "run.json") != manifest or stored.get("bundle_sha256") != bundle_hash:
                    raise WorkflowError(f"Run id {run_id} already exists with different evidence")
                for item in stored["artifacts"]:
                    file = _contained(root, item["local_path"])
                    if not file.is_file() or sha256_file(file) != item["sha256"]:
                        raise WorkflowError(f"Previously imported artifact is missing or changed: {file}")
                _record_location(root, target, source_uri, bundle_hash)
                return target
            if external.exists():
                raise WorkflowError(f"External artifact destination already exists: {external}")
            inbox = _research(root) / "inbox"
            inbox.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".import-", dir=inbox) as staging_name:
                staging = Path(staging_name)
                run_staging, artifact_staging = staging / "run", staging / "artifacts"
                run_staging.mkdir()
                verified = {}
                locators = []
                for item in manifest["artifacts"]:
                    temporary = _contained(staging / "verified", item["path"])
                    temporary.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with archive.open(inventory["payload/" + item["path"]]) as source, temporary.open("wb") as sink:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(block)
                            sink.write(block)
                    if digest.hexdigest() != item["sha256"]:
                        raise WorkflowError(f"Artifact SHA-256 mismatch: {item['path']}")
                    verified[item["path"]] = temporary
                evaluation = evaluate_run(protocol, manifest, verified)
                for item in manifest["artifacts"]:
                    temporary = verified[item["path"]]
                    tracked = _tracked_artifact(temporary)
                    storage_root = run_staging / "files" if tracked else artifact_staging
                    destination = _contained(storage_root, item["path"])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(temporary), str(destination))
                    final_file = (target / "files" if tracked else external).joinpath(*item["path"].split("/"))
                    locators.append({**item, "local_path": final_file.relative_to(root).as_posix(),
                                     "tracked": tracked})
                _write_json(run_staging / "run.json", manifest)
                _write_json(run_staging / "evaluation.json", evaluation)
                _write_json(run_staging / "artifacts.json", {
                    "schema": "ergt-phi-artifacts-v1", "bundle_sha256": bundle_hash,
                    "bundle_name": bundle.name, "imported_at": utc_now(),
                    "source_uri": source_uri,
                    "run_manifest_sha256": hashlib.sha256(_json_bytes(manifest)).hexdigest(),
                    "artifacts": locators,
                })
                (run_staging / "summary.md").write_text(_summary(protocol, manifest, evaluation),
                                                        encoding="utf-8", newline="\n")
                target.parent.mkdir(parents=True, exist_ok=True)
                if artifact_staging.exists():
                    external.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(artifact_staging, external)
                try:
                    os.replace(run_staging, target)
                except OSError:
                    if external.exists():
                        # This directory was created by this invocation, under the
                        # verified artifact root; preserve it for manual recovery.
                        raise WorkflowError(f"Run publication failed; verified artifacts retained at {external}")
                    raise
    except (zipfile.BadZipFile, json.JSONDecodeError, UnicodeError) as exc:
        raise WorkflowError(f"Invalid result bundle: {exc}") from exc
    record_event(root, "result_imported", f"Verified and imported {protocol['experiment_id']}/{run_id}",
                 experiment_id=protocol["experiment_id"], revision=protocol["revision"], run_id=run_id,
                 bundle_sha256=bundle_hash, eligible_for_pass=evaluation["eligible_for_pass"],
                 returncode=manifest["returncode"], source_commit=manifest["source_commit"],
                 execution_status=manifest["status"])
    return target


def _verified_import(root: Path, directory: Path, run_id: str) -> tuple[dict, dict, dict]:
    _identifier(run_id, "run_id")
    run = _contained(directory, f"runs/{run_id}")
    protocol = validate_protocol(_read_json(directory / "protocol.json"))
    lock = _read_json(directory / "package.json")
    _verify_release(directory, protocol, lock)
    manifest = _read_json(run / "run.json")
    locators = _read_json(run / "artifacts.json")
    if hashlib.sha256(_json_bytes(manifest)).hexdigest() != locators.get("run_manifest_sha256"):
        raise WorkflowError("Imported run manifest changed after verification")
    inventory = {"RUN_MANIFEST.json": zipfile.ZipInfo("RUN_MANIFEST.json")}
    for item in manifest["artifacts"]:
        entry = zipfile.ZipInfo("payload/" + item["path"])
        entry.file_size = item["bytes"]
        inventory[entry.filename] = entry
    _validate_manifest(manifest, protocol, lock, inventory)
    files = {}
    expected = {item["path"]: item for item in manifest["artifacts"]}
    for item in locators["artifacts"]:
        if item["path"] not in expected or any(item[key] != expected[item["path"]][key]
                                              for key in ("sha256", "bytes")):
            raise WorkflowError("Artifact locator disagrees with the imported run manifest")
        file = _contained(Path(root).resolve(), item["local_path"])
        if not file.is_file() or file.stat().st_size != item["bytes"] or sha256_file(file) != item["sha256"]:
            raise WorkflowError(f"Artifact changed or missing since import: {file}")
        files[item["path"]] = file
    if set(files) != set(expected):
        raise WorkflowError("Artifact locator inventory is incomplete")
    return protocol, manifest, evaluate_run(protocol, manifest, files)


def review_run(root: Path, experiment_id: str, revision: str, run_id: str,
               decision: str, reason: str, decision_id: str, next_action: str) -> Path:
    if decision not in {"pass", "revise", "inconclusive"}:
        raise WorkflowError("Decision must be pass, revise, or inconclusive")
    if not reason.strip() or not next_action.strip():
        raise WorkflowError("A review needs interpretation (reason) and a concrete next_action")
    _identifier(decision_id, "decision_id")
    adr = _research(root) / "decisions" / f"{decision_id}.md"
    if not adr.is_file() or not adr.read_text(encoding="utf-8-sig").strip():
        raise WorkflowError(f"Create the linked architecture decision first: {adr}")
    directory = revision_path(root, experiment_id, revision)
    protocol, manifest, evaluation = _verified_import(root, directory, run_id)
    if decision == "pass" and not evaluation["eligible_for_pass"]:
        raise WorkflowError("Cannot pass: execution, required artifacts, and every preregistered gate must pass")
    reviews = directory / "runs" / run_id / "reviews"
    existing = sorted(reviews.glob("*.json")) if reviews.exists() else []
    record = {"schema": "ergt-phi-review-v1", "experiment_id": experiment_id,
              "revision": revision, "run_id": run_id, "decision": decision,
              "reason": reason, "next_action": next_action, "decision_id": decision_id,
              "decision_path": adr.relative_to(Path(root).resolve()).as_posix(),
              "decision_sha256": sha256_file(adr), "kind": protocol["kind"],
              "phase_promoted": False, "evaluation": evaluation}
    if existing:
        last = _read_json(existing[-1])
        if all(last.get(key) == value for key, value in record.items()):
            return existing[-1]
        if last["decision_id"] == decision_id:
            raise WorkflowError("A revised review needs a new linked decision; earlier conclusions stay immutable")
    record["created_at"] = utc_now()
    record["supersedes"] = existing[-1].name if existing else None
    destination = reviews / f"{len(existing) + 1:03d}.json"
    _write_json(destination, record)
    record_event(root, "run_reviewed", f"{experiment_id}/{revision}/{run_id}: {decision}",
                 experiment_id=experiment_id, revision=revision, run_id=run_id,
                 decision=decision, decision_id=decision_id, next_action=next_action)
    return destination


def workflow_status(root: Path) -> dict:
    root = Path(root).resolve()
    registry = load_registry(root)
    experiments, reminders, imported_hashes = [], [], set()
    for experiment_id, entry in sorted(registry["experiments"].items()):
        for revision in sorted(entry["revisions"]):
            directory = revision_path(root, experiment_id, revision)
            label = f"{experiment_id}/{revision}"
            item = {"experiment_id": experiment_id, "revision": revision,
                    "path": directory.relative_to(root).as_posix(), "runs": []}
            try:
                protocol = validate_protocol(_read_json(directory / "protocol.json"))
                item.update(stage=protocol["stage"], kind=protocol["kind"], state="draft")
                if not protocol["gates"]:
                    reminders.append(f"{label}: preregister measurable acceptance gates before execution.")
                if not (directory / "experiment.ipynb").is_file():
                    reminders.append(f"{label}: generate and explain the versioned notebook.")
                elif not (directory / "package.json").is_file():
                    item["state"] = "notebook_ready"
                    reminders.append(f"{label}: build the immutable source package before Colab execution.")
                else:
                    _verify_release(directory, protocol, _read_json(directory / "package.json"))
                    item["state"] = "awaiting_execution_or_import"
                    for run in sorted((directory / "runs").glob("*")):
                        if not run.is_dir() or not (run / "run.json").is_file():
                            continue
                        manifest = _read_json(run / "run.json")
                        artifact_index = _read_json(run / "artifacts.json")
                        imported_hashes.add(artifact_index["bundle_sha256"])
                        if hashlib.sha256(_json_bytes(manifest)).hexdigest() != artifact_index.get("run_manifest_sha256"):
                            raise WorkflowError(f"Run manifest changed: {run.name}")
                        live_files = {}
                        for artifact in artifact_index["artifacts"]:
                            file = _contained(root, artifact["local_path"])
                            if not file.is_file() or file.stat().st_size != artifact["bytes"]:
                                raise WorkflowError(f"Imported artifact missing or resized: {artifact['local_path']}")
                            if file.stat().st_size <= MAX_TRACKED_BYTES and sha256_file(file) != artifact["sha256"]:
                                raise WorkflowError(f"Imported small artifact changed: {artifact['local_path']}")
                            live_files[artifact["path"]] = file
                        if not (run / "interpretation.md").is_file():
                            reminders.append(f"{label}/{run.name}: write interpretation.md beside the evidence: "
                                             "meaning, limitations, alternatives, paper claims and next action.")
                        location_uris = [artifact_index.get("source_uri", "")]
                        location_uris.extend(_read_json(location).get("source_uri", "")
                                             for location in sorted((run / "locations").glob("*.json")))
                        durable_locator = any(uri.startswith(("https://", "http://", "gs://", "s3://", "drive://"))
                                              for uri in location_uris)
                        if any(not artifact["tracked"] and artifact["path"] in protocol["required_artifacts"]
                               for artifact in artifact_index["artifacts"]) and not durable_locator:
                            reminders.append(f"{label}/{run.name}: required large/binary artifacts have only local "
                                             "storage; back up the result ZIP and re-import with --source-uri.")
                        evaluation = evaluate_run(protocol, manifest, live_files)
                        if _read_json(run / "evaluation.json") != evaluation:
                            raise WorkflowError(f"Saved gate evaluation disagrees with current evidence: {run.name}")
                        reviews = sorted((run / "reviews").glob("*.json"))
                        detail = {"run_id": run.name, "execution_status": manifest["status"],
                                  "eligible_for_pass": evaluation["eligible_for_pass"],
                                  "integrity_check": "existence, sizes, and small-file hashes; large hashes rechecked at review",
                                  "review": None}
                        if not reviews:
                            item["state"] = "awaiting_review"
                            reminders.append(f"{label}/{run.name}: review imported gates, interpretation, "
                                             "limitations and next action; link an architecture decision.")
                        else:
                            review = _read_json(reviews[-1])
                            detail["review"] = review["decision"]
                            detail["next_action"] = review["next_action"]
                            decision_file = _contained(root, review["decision_path"])
                            if not decision_file.is_file() or sha256_file(decision_file) != review["decision_sha256"]:
                                reminders.append(f"{label}/{run.name}: linked decision changed or is missing; "
                                                 "restore it and record amendments as a new decision.")
                            if item["state"] != "awaiting_review":
                                item["state"] = ("passed_infrastructure" if protocol["kind"] == "infrastructure"
                                                 else "passed_experiment") if review["decision"] == "pass" else review["decision"]
                            if review["decision"] == "revise":
                                successors = []
                                for other_id, other in registry["experiments"].items():
                                    for other_rev in other["revisions"]:
                                        other_protocol = _read_json(revision_path(root, other_id, other_rev) / "protocol.json")
                                        if other_protocol.get("parent") == {"experiment_id": experiment_id, "revision": revision}:
                                            successors.append(f"{other_id}/{other_rev}")
                                if not successors:
                                    reminders.append(f"{label}: create a successor revision/branch for "
                                                     f"the recorded next action: {review['next_action']}")
                            if review["decision"] == "inconclusive":
                                reminders.append(f"{label}/{run.name}: resolve the uncertainty: {review['next_action']}")
                            if review["decision"] == "pass" and protocol["kind"] == "research":
                                reminders.append(f"{label}: experiment passed its gates; a separate "
                                                 "phase review must check the mathematical goal and independent evidence.")
                        item["runs"].append(detail)
                    if not item["runs"]:
                        reminders.append(f"{label}: execute its notebook in Colab, return the result ZIP, "
                                         "then import it; cloud execution is not observed automatically.")
            except (WorkflowError, KeyError, TypeError) as exc:
                item["state"] = "integrity_error"
                item["error"] = str(exc)
                reminders.append(f"{label}: repair the recorded integrity error: {exc}")
            experiments.append(item)
    inbox = []
    for path in sorted((_research(root) / "inbox").glob("*.zip")):
        if sha256_file(path) not in imported_hashes:
            inbox.append(path.relative_to(root).as_posix())
            reminders.append(f"Import pending Colab bundle: {path.relative_to(root).as_posix()}")
    if not experiments:
        reminders.append("Register a protocol and generate a versioned notebook for the next planned experiment.")
    git_dirty = None
    try:
        from dulwich import porcelain
        status = porcelain.status(str(root))
        git_dirty = any(status.staged.values()) or bool(status.unstaged) or bool(status.untracked)
        if git_dirty:
            reminders.append("Commit each completed, verified unit of progress with its protocol, evidence and decision.")
    except Exception:
        # Git is optional for this stdlib workflow; do not mistake an unavailable
        # backend for a clean repository.
        git_dirty = None
    return {"schema": "ergt-phi-workflow-status-v1", "experiments": experiments,
            "inbox_pending": inbox, "reminders": reminders, "git_dirty": git_dirty,
            "cloud_execution_observed": False, "automatic_phase_promotion": False}


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root or Path.cwd(), help="Repository root")
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create", help="Register a new experiment protocol")
    create.add_argument("--protocol", required=True, type=Path)
    revise = sub.add_parser("revise", help="Create a new immutable revision from a registered parent")
    revise.add_argument("--id", required=True)
    revise.add_argument("--from", dest="from_revision", required=True)
    revise.add_argument("--protocol", required=True, type=Path)
    revise.add_argument("--reason", help="Change of strategy; otherwise protocol.revision_reason is required")
    for name in ("notebook", "package"):
        action = sub.add_parser(name, help=f"Build the versioned {name}")
        action.add_argument("--id", required=True)
        action.add_argument("--revision", default="v001")
    incoming = sub.add_parser("import", help="Verify and import a Colab result ZIP without executing it")
    incoming.add_argument("bundle", type=Path)
    incoming.add_argument("--source-uri", default="", help="Durable Drive/archive locator; recorded, not remotely verified")
    review = sub.add_parser("review", help="Record a conclusion linked to an architecture decision")
    review.add_argument("--id", required=True)
    review.add_argument("--revision", default="v001")
    review.add_argument("--run-id", required=True)
    review.add_argument("--decision", required=True, choices=("pass", "revise", "inconclusive"))
    review.add_argument("--reason", required=True)
    review.add_argument("--decision-id", required=True)
    review.add_argument("--next-action", required=True)
    status_parser = sub.add_parser("status", help="Report missing deliverables and next actions")
    status_parser.add_argument("--json", action="store_true")
    event = sub.add_parser("event", help="Record an implementation/discussion milestone")
    event.add_argument("--type", required=True)
    event.add_argument("--message", required=True)
    event.add_argument("--path", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        if args.action == "create":
            result = create_experiment(args.root, _read_json(args.protocol))
        elif args.action == "revise":
            protocol = _read_json(args.protocol)
            if args.reason:
                protocol["revision_reason"] = args.reason
            result = revise_experiment(args.root, args.id, args.from_revision, protocol)
        elif args.action == "notebook":
            result = build_notebook(args.root, args.id, args.revision)
        elif args.action == "package":
            result = build_package(args.root, args.id, args.revision)
        elif args.action == "import":
            result = import_bundle(args.root, args.bundle, args.source_uri)
        elif args.action == "review":
            result = review_run(args.root, args.id, args.revision, args.run_id,
                                args.decision, args.reason, args.decision_id, args.next_action)
        elif args.action == "event":
            result = record_event(args.root, args.type, args.message, paths=args.path)
        else:
            result = workflow_status(args.root)
            if not args.json:
                for item in result["experiments"]:
                    print(f"{item['experiment_id']}/{item['revision']}: {item['state']}")
                for reminder in result["reminders"]:
                    print(f"- {reminder}")
                return 0
        if isinstance(result, Path):
            print(result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (WorkflowError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"research workflow: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
