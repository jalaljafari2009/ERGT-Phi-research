import importlib.util
from pathlib import Path


def _validator_module():
    path = Path(__file__).resolve().parent / "validate_completed_run.py"
    spec = importlib.util.spec_from_file_location("standalone_result_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_result_validator_accepts_both_official_record_shapes():
    normalize = _validator_module().normalize_verdict
    direct = normalize({
        "execution_integrity_pass": True,
        "native_ergt_claim_status": "supported",
        "native_32_hop_accuracy_mean": 0.955357,
        "direct_transformer_32_hop_accuracy_mean": 0.5,
        "native_minus_transformer_32_hop": 0.455357,
    })
    registered = normalize({
        "evidence_status": "bounded_four_seed_claim_supported",
        "architecture_and_fairness": {"integrity_checks_all_passed": True},
        "horizon_accuracy_mean": {
            "32": {"ergt": 0.955357, "direct_transformer": 0.5}
        },
    })
    assert direct["format"] == "direct_final_verdict"
    assert registered["format"] == "registered_evidence_summary"
    assert direct["native_minus_transformer_32_hop"] == 0.455357
    assert abs(registered["native_minus_transformer_32_hop"] - 0.455357) < 1.0e-12
