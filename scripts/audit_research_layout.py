"""Verify consolidation integrity; never execute imported scientific artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_tools.workflow import _pointer, _verified_import, load_registry, revision_path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def audit(require_artifacts=False):
    errors, missing_external = [], []
    migration = read(ROOT / "research/migrations/20260920-history-consolidation.json")
    def check(relative, expected, external=False):
        path = ROOT / relative
        if external and not path.exists():
            missing_external.append(relative)
        elif not path.is_file() or sha(path) != expected:
            errors.append("Missing or changed bytes: " + relative)
    for item in migration["entries"]:
        check(item["new_path"], item["sha256"], item["action"] == "move_directory_member")
        if item["tracked_evidence"]:
            check(item["tracked_evidence"], item["sha256"])
    sources = read(ROOT / "research/sources/register.json")
    for item in sources["sources"]:
        check(item["archived_path"], item["sha256"], item["archived_path"].startswith("research/artifacts/"))

    from dulwich.repo import Repo
    source_records = 0
    with Repo(str(ROOT)) as repository:
        for provenance in (ROOT / "research/legacy").glob("LEGACY-*/source_provenance.json"):
            for row in read(provenance)["records"]:
                source_records += 1
                for location in row["git_locations"]:
                    blob = location["git_blob"].encode()
                    data = repository[blob].data
                    commit = repository[location["locator_commit"].encode()]
                    _, recorded_blob = repository[commit.tree].lookup_path(repository.__getitem__, location["git_path"].encode())
                    if recorded_blob != blob:
                        errors.append("Git content locator mismatch: " + location["git_path"])
                    if row["hash_comparison"] == "sha256-canonical-lf-v1":
                        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                    if hashlib.sha256(data).hexdigest() != row["recorded_sha256"]:
                        errors.append("Historical source SHA mismatch: " + row["historical_path"])
                for origin in row["recorded_by"]:
                    if _pointer(read(ROOT / origin["evidence_path"]), origin["json_pointer"]) != row["recorded_sha256"]:
                        errors.append("Historical source reference mismatch: " + origin["evidence_path"])

    # Compare the actual imported Colab files with the original delivered ZIP.
    reconciled = 0
    source_zip = next(item for item in sources["sources"] if item["id"] == "SRC-005")
    archive = ROOT / source_zip["archived_path"]
    raw = ROOT / "research/artifacts/legacy/LEGACY-M0/imported_m0"
    if archive.exists() and raw.exists():
        with zipfile.ZipFile(archive) as stream:
            for item in stream.infolist():
                if item.is_dir():
                    continue
                path = (raw / item.filename).resolve()
                if not path.is_relative_to(raw.resolve()):
                    raise ValueError("Unexpected result archive member")
                expected = hashlib.sha256(stream.read(item)).hexdigest()
                check(path.relative_to(ROOT).as_posix(), expected, True)
                reconciled += 1

    # Sealed versioned notebooks retain their hashes even after source reorganization.
    sealed_runs = 0
    for experiment_id, entry in load_registry(ROOT)["experiments"].items():
        for revision in entry["revisions"]:
            directory = revision_path(ROOT, experiment_id, revision)
            for path in (directory / "runs").glob("*/run.json"):
                _verified_import(ROOT, directory, path.parent.name)
                sealed_runs += 1

    checked_links = 0
    historical_link_notes = []
    documents = [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "scripts/README.md"]
    documents += list((ROOT / "docs").glob("*.md"))
    documents += [path for path in (ROOT / "research").rglob("*.md") if not any(
        part in {"artifacts", "packages", "inbox", ".staging", "workspace", "spec"}
        for part in path.relative_to(ROOT).parts)]
    for document in documents:
        original = any(part in {"reports", "git_evidence", "evidence"} for part in document.parts)
        for found in re.finditer(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            target = found.group(1).split("#")[0].strip("<>")
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            checked_links += 1
            if not (document.parent / target).exists():
                note = f"{document.relative_to(ROOT).as_posix()}: {target}"
                (historical_link_notes if original else errors).append(note)
    result = {"schema": "ergt-phi-layout-audit-v1",
              "pass": not errors and (not require_artifacts or not missing_external),
              "migrated_originals_checked": len(migration["entries"]),
              "tracked_evidence_checked": sum(bool(item["tracked_evidence"]) for item in migration["entries"]),
              "registered_sources_checked": len(sources["sources"]),
              "historical_source_records_verified": source_records,
              "original_colab_zip_members_reconciled": reconciled,
              "sealed_runs_reverified": sealed_runs, "markdown_links_checked": checked_links,
              "preserved_historical_link_notes": historical_link_notes,
              "missing_local_archives": missing_external, "errors": errors,
              "scientific_phase_promoted": False}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.require_artifacts)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        destination = args.report.resolve()
        if not destination.is_relative_to((ROOT / "research/validation").resolve()):
            raise ValueError("Audit reports belong under research/validation/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    print(payload)
    raise SystemExit(0 if result["pass"] else 1)
