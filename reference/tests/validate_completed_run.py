"""Validate a completed four-seed output against registered bounds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def normalize_verdict(payload):
    if "execution_integrity_pass" in payload:
        return {
            "format": "direct_final_verdict",
            "execution_integrity_pass": bool(payload["execution_integrity_pass"]),
            "native_claim_supported": payload.get("native_ergt_claim_status") == "supported",
            "native_32_hop_accuracy_mean": float(payload["native_32_hop_accuracy_mean"]),
            "direct_transformer_32_hop_accuracy_mean": float(
                payload["direct_transformer_32_hop_accuracy_mean"]
            ),
            "native_minus_transformer_32_hop": float(
                payload["native_minus_transformer_32_hop"]
            ),
        }
    if "architecture_and_fairness" in payload and "horizon_accuracy_mean" in payload:
        hop_32 = payload["horizon_accuracy_mean"]["32"]
        native = float(hop_32["ergt"])
        transformer = float(hop_32["direct_transformer"])
        return {
            "format": "registered_evidence_summary",
            "execution_integrity_pass": bool(
                payload["architecture_and_fairness"]["integrity_checks_all_passed"]
            ),
            "native_claim_supported": payload.get("evidence_status")
            == "bounded_four_seed_claim_supported",
            "native_32_hop_accuracy_mean": native,
            "direct_transformer_32_hop_accuracy_mean": transformer,
            "native_minus_transformer_32_hop": native - transformer,
        }
    raise ValueError("unrecognized four-seed verdict or evidence-summary schema")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    package_root = Path(__file__).resolve().parents[1]
    contract = json.loads((package_root / "contracts/result_acceptance.json").read_text())
    verdict_payload = json.loads((run_root / "final_verdict.json").read_text())
    verdict = normalize_verdict(verdict_payload)
    invariants = list(csv.DictReader((run_root / "native_stability_invariants.csv").open(newline="")))
    checks = {
        "execution_integrity": bool(verdict["execution_integrity_pass"]),
        "native_claim_supported": verdict["native_claim_supported"],
        "native_32_hop_floor": float(verdict["native_32_hop_accuracy_mean"])
        >= float(contract["native_32_hop_accuracy_mean_minimum"]),
        "paired_margin_floor": float(verdict["native_minus_transformer_32_hop"])
        >= float(contract["native_minus_direct_transformer_32_hop_minimum"]),
        "invariant_count": len(invariants) == int(contract["native_stability_invariants_required"]),
        "all_invariants_pass": all(row["passed"].lower() == "true" for row in invariants),
        "baseline_is_nonblocking": contract["baseline_failure_blocks_native_result"] is False,
    }
    print(json.dumps({"pass": all(checks.values()), "checks": checks, "normalized_verdict": verdict}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
