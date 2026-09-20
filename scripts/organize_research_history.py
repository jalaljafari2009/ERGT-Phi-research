"""Migrate existing evidence into named historical dossiers, with byte inventories.

This is a one-time, resumable repository organization command. It never runs a
model, promotes a scientific phase, or alters the contents of archived evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "20260920-history-consolidation"
TEXT = {".json", ".jsonl", ".md", ".txt", ".log", ".csv", ".tsv", ".xml", ".yaml", ".yml"}
GROUPS = {
    "LEGACY-M0": {
        "reports": ["M0_ACCEPTANCE.md", "M0_REPORT_FA.md", "M0_TRAINED_REPORT_FA.md"],
        "manifests": ["reference.json", "environment.json", "resolved_paper_protocol.json", "version_control.json", "research_source.json", "engineering_fixture.json", "reference_training_status.json", "m0_status.json", "imported_m0_evidence.json", "trained_m0_audit.json"],
        "notebooks": ["M0_Reference_Training.ipynb"], "runs": ["imported_m0", "m0"],
    },
    "LEGACY-M1": {"reports": ["M1_REPORT_FA.md"], "manifests": ["m1_status.json", "m1_example.json"], "runs": ["m1"]},
    "LEGACY-M2-INITIAL": {"reports": ["M2_REPORT_FA.md", "M2_ANCHOR_REVISION_FA.md"]},
    "LEGACY-M2-CONTEXT": {"reports": ["M2_CONTEXTUAL_REVISION_FA.md"], "manifests": ["m2_status.json", "m2_protocol.json", "anchor_context_audit.json"], "runs": ["m2", "m2_context"]},
    "LEGACY-M2-DIAGNOSTIC": {"reports": ["M2_DIAGNOSTIC_FA.md"], "manifests": ["m2_diagnostic.json"], "runs": ["m2_diagnostic"]},
    "LEGACY-M2-INFORMATION": {"reports": ["M2_INFORMATION_PROBE_FA.md"], "manifests": ["m2_information.json"], "runs": ["m2_information"]},
    "LEGACY-M2-PROPOSAL": {"reports": ["M2_PROPOSAL_PROBE_FA.md"], "manifests": ["m2_proposal.json"], "runs": ["m2_proposal"]},
    "LEGACY-M2-Q-TIMING": {"reports": ["M2_LAGGED_Q_FA.md"]},
    "LEGACY-M2-Q-SHADOW": {"reports": ["M2_Q_SHADOW_FA.md"], "manifests": ["m2_q_shadow.json"], "runs": ["m2_q_shadow"]},
    "LEGACY-M2-Q-CALIBRATION": {"reports": ["M2_Q_CALIBRATION_FA.md"], "manifests": ["m2_q_calibration.json"], "runs": ["m2_q_calibration"]},
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def contained(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative or ":" in relative or any(x in {"", ".", ".."} for x in relative.split("/")):
        raise ValueError(f"Unsafe repository path: {relative}")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError(f"Target outside intended repository: {relative}")
    return target


def save_once(path: Path, value: dict) -> None:
    data = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite a different migration record: {path}")
    else:
        path.write_bytes(data)


def build_plan(root: Path) -> dict:
    from dulwich.repo import Repo
    entries, directories = [], {}
    for group, definition in GROUPS.items():
        for kind, old_folder in (("reports", "docs"), ("manifests", "manifests"), ("notebooks", "notebook")):
            for name in definition.get(kind, []):
                old = f"{old_folder}/{name}"
                source = contained(root, old)
                destination = f"research/legacy/{group}/{kind}/{name}"
                if not source.is_file():
                    raise FileNotFoundError(source)
                entries.append({"experiment_id": group, "old_path": old, "new_path": destination,
                                "sha256": digest(source), "bytes": source.stat().st_size,
                                "action": "move_file", "tracked_evidence": destination})
        for name in definition.get("runs", []):
            old_root = f"runs/{name}"
            new_root = f"research/artifacts/legacy/{group}/{name}"
            source_root = contained(root, old_root)
            directories[old_root] = new_root
            for source in sorted(source_root.rglob("*")):
                if source.is_symlink():
                    raise ValueError(f"Unexpected symlink in historical evidence: {source}")
                if not source.is_file():
                    continue
                suffix = source.relative_to(source_root).as_posix()
                tracked = None
                if source.stat().st_size <= 2 * 1024**2 and source.suffix.lower() in TEXT | {".png", ".svg"}:
                    tracked = f"research/legacy/{group}/evidence/{name}/{suffix}"
                entries.append({"experiment_id": group, "old_path": f"{old_root}/{suffix}",
                                "new_path": f"{new_root}/{suffix}", "sha256": digest(source),
                                "bytes": source.stat().st_size, "action": "move_directory_member",
                                "tracked_evidence": tracked})
    with Repo(str(root)) as repo:
        baseline = repo.head().decode()
    return {"schema": "ergt-phi-history-migration-v1", "migration_id": MIGRATION,
            "created_at": datetime.now(timezone.utc).isoformat(), "baseline_commit": baseline,
            "entries": entries, "directories": directories,
            "scope": "Byte-preserving organization; no new scientific execution or retrospective preregistration."}


def verify(path: Path, item: dict) -> None:
    if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
        raise ValueError(f"Historical evidence changed or is missing: {path}")


def apply(root: Path, plan: dict) -> None:
    # Validate every source/destination BEFORE any move, including resolved roots.
    for item in plan["entries"]:
        source, destination = contained(root, item["old_path"]), contained(root, item["new_path"])
        verify(source if source.exists() else destination, item)
        if source.exists() and destination.exists():
            raise FileExistsError(f"Both old and new evidence paths exist: {source}, {destination}")
    for old, new in plan["directories"].items():
        source, destination = contained(root, old), contained(root, new)
        if source.exists():
            if destination.exists():
                raise FileExistsError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
    for item in plan["entries"]:
        source, destination = contained(root, item["old_path"]), contained(root, item["new_path"])
        if item["action"] == "move_file" and source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
        verify(destination, item)
        tracked = item["tracked_evidence"]
        if tracked and tracked != item["new_path"]:
            copy = contained(root, tracked)
            copy.parent.mkdir(parents=True, exist_ok=True)
            if not copy.exists():
                shutil.copyfile(destination, copy)
            verify(copy, item)
    mapping = {"schema": "ergt-phi-legacy-path-map-v1", "migration_id": MIGRATION,
               "files": {item["old_path"]: item["new_path"] for item in plan["entries"] if item["action"] == "move_file"},
               "directories": plan["directories"]}
    save_once(root / "research/legacy/path_map.json", mapping)
    for group in GROUPS:
        selected = [item for item in plan["entries"] if item["experiment_id"] == group]
        save_once(root / f"research/legacy/{group}/inventory.json", {
            "schema": "ergt-phi-legacy-inventory-v1", "experiment_id": group,
            "migration_id": MIGRATION, "entries": selected, "new_execution_claimed": False,
        })
    print(json.dumps({"migrated_files": len(plan["entries"]),
                      "tracked_evidence_files": sum(bool(item["tracked_evidence"]) for item in plan["entries"]),
                      "raw_bytes": sum(item["bytes"] for item in plan["entries"]),
                      "historical_dossiers": len(GROUPS)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply the checked migration plan")
    args = parser.parse_args()
    record = ROOT / f"research/migrations/{MIGRATION}.json"
    if record.exists():
        plan = json.loads(record.read_text(encoding="utf-8"))
    else:
        plan = build_plan(ROOT)
        save_once(record, plan)
    if args.apply:
        apply(ROOT, plan)
    else:
        print(json.dumps({"plan": str(record), "files": len(plan["entries"]), "directories": plan["directories"]}, indent=2))


if __name__ == "__main__":
    main()
