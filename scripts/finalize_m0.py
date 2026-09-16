"""Audit the deliverable and derive status from saved acceptance evidence."""
from pathlib import Path
import ast
import hashlib
import json
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


reference = json.loads((ROOT / "manifests/reference.json").read_text())
for relative, expected in reference["files"].items():
    for base in (ROOT / "reference", Path(reference["original_root"])):
        if base.exists() and digest(base / relative) != expected:
            raise RuntimeError(f"Reference mutated: {base / relative}")
records = {}
for directory in ("ergt_phi", "scripts", "tests"):
    for path in sorted((ROOT / directory).rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        records[path.relative_to(ROOT).as_posix()] = digest(path)
for path in sorted((ROOT / "configs").glob("*.json")):
    json.loads(path.read_text(encoding="utf-8"))
    records[path.relative_to(ROOT).as_posix()] = digest(path)
notebook = json.loads((ROOT / "notebook/M0_Reference_Training.ipynb").read_text())
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), "colab_cell", "exec")
test_summaries = {}
for name in ("reference", "research"):
    tree = ET.parse(ROOT / f"runs/m0/{name}-tests.xml")
    totals = {key: sum(int(s.attrib.get(key, 0)) for s in tree.iter("testsuite")) for key in ("tests", "failures", "errors", "skipped")}
    test_summaries[name] = totals
    if totals["failures"] or totals["errors"] or totals["skipped"]:
        raise RuntimeError(f"Unresolved acceptance tests: {name}: {totals}")
(ROOT / "manifests/research_source.json").write_text(json.dumps({"files": records}, indent=2), encoding="utf-8")
status = {
    "M0_A": "passed_on_registered_cpu_fixtures", "M0_B": "pending_cuda_training_and_trained_checkpoint_audit",
    "M0_complete": False, "tests": test_summaries,
    "original_reference_unchanged": True, "reference_manifest_entries": len(reference["files"]),
    "registered_data_cohorts_verified": 80, "phase_extension_active": False,
    "untrained_fixture_is_scientific_evidence": False,
    "gpu_training_executed": False,
}
trained_path = ROOT / "manifests/trained_m0_audit.json"
if trained_path.exists():
    trained = json.loads(trained_path.read_text())
    imported = json.loads((ROOT / "manifests/imported_m0_evidence.json").read_text())
    run = ROOT / "runs/imported_m0/m0_single_seed_reference"
    for relative, expected in imported["files"].items():
        assert digest(run / relative) == expected, relative
    assert trained["status"] == "passed"
    assert trained["checkpoint_sha256"] == imported["files"]["opt_12011_data_16301/native_ergt_training.pt"]
    status.update(M0_B="passed_single_seed_development_reference_cpu_parity",
                  M0_complete=True, completion_scope="development_reference_only",
                  gpu_training_executed=True, gpu_training_location="user_Colab_T4",
                  scientific_claim_status="open", four_seed_reproduction=False,
                  trained_parity_evidence="manifests/trained_m0_audit.json")
(ROOT / "manifests/m0_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
print(json.dumps(status, indent=2))
