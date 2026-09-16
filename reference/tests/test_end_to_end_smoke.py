import csv
import json
from pathlib import Path

from ergt_four_seed.runtime import activate_locked_runtime

activate_locked_runtime()
from ergt_reviewer.suite_v9 import run_reviewer_suite


def test_standalone_end_to_end_smoke(tmp_path: Path):
    result = run_reviewer_suite(
        profile="standalone_smoke",
        device="cpu",
        output_root=tmp_path,
        run_id="standalone_smoke",
        resume=False,
        copy_outputs_to_downloads=False,
    )
    root = Path(result["run_root"])
    verdict = json.loads((root / "final_verdict.json").read_text(encoding="utf-8"))
    observer = json.loads(
        (root / "spectral_observer_numerical_audit.json").read_text(encoding="utf-8")
    )
    assert verdict["status"] == "smoke_passed"
    assert verdict["execution_integrity_pass"] is True
    assert verdict["gates"]["locked_native_core_unchanged"] is True
    assert verdict["gates"]["architecture_and_shared_input"] is True
    assert verdict["gates"]["data_protocol_integrity"] is True
    assert int(observer["graphs_unavailable_after_numerical_failure"]) == 0
    assert int(observer["graphs_solved_cpu_float64"]) > 0
    assert observer["governing_answer_path_changed"] is False
    assert observer["reporting_only"] is True
    assert Path(result["bundle_path"]).is_file()
    table_manifest = json.loads(
        (root / "compact_tables/compact_table_manifest.json").read_text(encoding="utf-8")
    )
    assert len(table_manifest["display_order"]) == 14
    with (root / "causal_interventions.csv").open(encoding="utf-8", newline="") as handle:
        interventions = {row["intervention"] for row in csv.DictReader(handle)}
    assert {"shuffled_geometry", "random_geometry", "only_world_0"} <= interventions
