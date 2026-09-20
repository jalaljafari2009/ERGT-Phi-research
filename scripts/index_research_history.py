"""Index historical evidence without reclassifying it as versioned Colab runs."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def record(path):
    file = ROOT / path
    return {"path": path, "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "bytes": file.stat().st_size}


def main():
    definitions = [
        ("LEGACY-M0", "M0", "development_reference_only", "manifests/m0_status.json", "docs/M0_TRAINED_REPORT_FA.md", "notebook/M0_Reference_Training.ipynb", "imported_colab_single_seed"),
        ("LEGACY-M1", "M1", "isolated_kernel_passed", "manifests/m1_status.json", "docs/M1_REPORT_FA.md", None, "local_cpu"),
        ("LEGACY-M2-CONTEXT", "M2", "calibration_not_qualified", "manifests/m2_status.json", "docs/M2_CONTEXTUAL_REVISION_FA.md", None, "local_cpu"),
        ("LEGACY-M2-DIAGNOSTIC", "M2", "diagnostic_only", "manifests/m2_diagnostic.json", "docs/M2_DIAGNOSTIC_FA.md", None, "local_cpu"),
        ("LEGACY-M2-INFORMATION", "M2", "diagnostic_only", "manifests/m2_information.json", "docs/M2_INFORMATION_PROBE_FA.md", None, "local_cpu"),
        ("LEGACY-M2-PROPOSAL", "M2", "diagnostic_only", "manifests/m2_proposal.json", "docs/M2_PROPOSAL_PROBE_FA.md", None, "local_cpu"),
        ("LEGACY-M2-Q-SHADOW", "M2", "disconnected_shadow_audit_only", "manifests/m2_q_shadow.json", "docs/M2_Q_SHADOW_FA.md", None, "local_cpu"),
        ("LEGACY-M2-Q-CALIBRATION", "M2", "metrics_passed_handoff_incomplete", "manifests/m2_q_calibration.json", "docs/M2_Q_CALIBRATION_FA.md", None, "local_cpu"),
    ]
    entries = []
    for ident, stage, scope, manifest, report, notebook, location in definitions:
        item = {"id": ident, "stage": stage, "scope": scope, "manifest": record(manifest), "report": record(report), "execution_location_recorded": location, "versioned_bundle_verified": False}
        if notebook:
            item["notebook"] = record(notebook)
            item["notebook_result_association"] = "historical_reference_not_authenticated_by_new_bundle_contract"
        else:
            item["notebook"] = None
            item["notebook_result_association"] = "no_versioned_notebook_recorded"
        entries.append(item)
    document = {
        "schema": "ergt-phi-legacy-index-v1",
        "source_commit": "9072e12465f6227656dbb9e640eb8244899f4a8f",
        "review_basis": "docs/OPERATIONAL_ROADMAP_FA.md",
        "purpose": "Historical lookup only; original artifacts remain unchanged. No reconstructed execution or checkpoint is claimed.",
        "entries": entries,
    }
    destination = ROOT / "research/legacy/index.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Indexed {len(entries)} historical evidence groups: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
