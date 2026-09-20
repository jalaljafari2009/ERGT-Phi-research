"""Index consolidated historical dossiers without inventing new executions."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def record(path):
    file = ROOT / path
    return {"path": file.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "bytes": file.stat().st_size}


def main():
    definitions = [
        ("LEGACY-M0", "M0", "development_reference_only", "m0_status.json", "M0_TRAINED_REPORT_FA.md", "M0_Reference_Training.ipynb", "imported_colab_single_seed"),
        ("LEGACY-M1", "M1", "isolated_kernel_passed", "m1_status.json", "M1_REPORT_FA.md", None, "local_cpu"),
        ("LEGACY-M2-INITIAL", "M2", "calibration_failed_original_raw_unavailable", None, "M2_REPORT_FA.md", None, "local_cpu"),
        ("LEGACY-M2-CONTEXT", "M2", "calibration_not_qualified", "m2_status.json", "M2_CONTEXTUAL_REVISION_FA.md", None, "local_cpu"),
        ("LEGACY-M2-DIAGNOSTIC", "M2", "diagnostic_only", "m2_diagnostic.json", "M2_DIAGNOSTIC_FA.md", None, "local_cpu"),
        ("LEGACY-M2-INFORMATION", "M2", "diagnostic_only", "m2_information.json", "M2_INFORMATION_PROBE_FA.md", None, "local_cpu"),
        ("LEGACY-M2-PROPOSAL", "M2", "diagnostic_only", "m2_proposal.json", "M2_PROPOSAL_PROBE_FA.md", None, "local_cpu"),
        ("LEGACY-M2-Q-TIMING", "M2", "isolated_contract_not_integrated_loop", None, "M2_LAGGED_Q_FA.md", None, "design_and_unit_tests"),
        ("LEGACY-M2-Q-SHADOW", "M2", "disconnected_shadow_audit_only", "m2_q_shadow.json", "M2_Q_SHADOW_FA.md", None, "local_cpu"),
        ("LEGACY-M2-Q-CALIBRATION", "M2", "metrics_passed_handoff_incomplete", "m2_q_calibration.json", "M2_Q_CALIBRATION_FA.md", None, "local_cpu"),
    ]
    entries = []
    migration_path = "research/migrations/20260920-history-consolidation.json"
    migration = json.loads((ROOT / migration_path).read_text(encoding="utf-8"))
    parent = None
    for ident, stage, scope, manifest, report, notebook, location in definitions:
        directory = Path("research/legacy") / ident
        manifest_path = directory / "manifests" / manifest if manifest else None
        if ident == "LEGACY-M2-INITIAL":
            manifest_path = directory / "git_evidence/manifests/m2_status.json"
        item = {"id": ident, "stage": stage, "scope": scope,
                "root": directory.as_posix(), "readme": record(directory / "README.md"),
                "inventory": record(directory / "inventory.json"),
                "manifest": record(manifest_path) if manifest_path else None,
                "report": record(directory / "reports" / report),
                "execution_location_recorded": location, "versioned_bundle_verified": False,
                "parent": parent, "parent_semantics": "reconstructed chronology, not preregistered dependency",
                "retrospective_preregistration": False}
        if notebook:
            item["notebook"] = record(directory / "notebooks" / notebook)
            item["notebook_result_association"] = "historical_reference_not_authenticated_by_new_bundle_contract"
        else:
            item["notebook"] = None
            item["notebook_result_association"] = "no_versioned_notebook_recorded"
        item["git_evidence"] = [record(path.relative_to(ROOT)) for path in sorted((ROOT / directory / "git_evidence").rglob("*")) if path.is_file()]
        source_provenance = ROOT / directory / "source_provenance.json"
        if source_provenance.exists():
            item["source_provenance"] = record(source_provenance.relative_to(ROOT))
        item["raw_archive_roots"] = [new for old, new in migration["directories"].items() if f"/{ident}/" in new]
        item["tracked_evidence_count"] = sum(1 for path in (ROOT / directory / "evidence").rglob("*") if path.is_file())
        entries.append(item)
        parent = ident
    document = {
        "schema": "ergt-phi-legacy-index-v2",
        "migration": record(migration_path),
        "review_basis": "docs/OPERATIONAL_ROADMAP_FA.md",
        "purpose": "Consolidated historical dossiers; source bytes preserved. No new execution or checkpoint is claimed.",
        "source_register": "research/sources/register.json",
        "path_map": "research/legacy/path_map.json",
        "entries": entries,
    }
    destination = ROOT / "research/legacy/index.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Indexed {len(entries)} historical evidence groups: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
