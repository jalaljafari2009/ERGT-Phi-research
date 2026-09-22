"""Run M2-E004/v001 target/shortcut audit with all accepted weights frozen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from ergt_phi.checkpoint import restore
from ergt_phi.m2_fresh import (
    EndpointClassifier,
    build_fresh_development,
    cache_q_features,
    confirmation_windows,
    parameter_count,
    train_control,
)
from ergt_phi.m2_q_training import sha256
from ergt_phi.m2_shortcut_audit import (
    evaluate_classifier_arm,
    evaluate_phase_arm,
    explicit_target_audit,
    prediction_change_rate,
    relation_accuracy_by_answer_outcome,
    transform_q_records,
)
from ergt_phi.native_steps import SteppedNative
from ergt_phi.shadow_data import pair_partition, training_only
from ergt_phi.shadow_phase import CalibrationConfig, PhaseAnchorNetwork
from ergt_reviewer.evaluation_v9 import evaluate_native
from ergt_reviewer.matched_data import RawTokenInputContract
from ergt_reviewer.native_solver import ERGT43Config


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--phase-checkpoint", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--reference-metadata", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--fresh-lock", required=True)
    parser.add_argument("--experiment-protocol", required=True)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _state_sha(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _identity(examples: tuple[Any, ...], indices: list[int]) -> dict[str, list[str]]:
    return {
        "example_ids": [examples[index].example_id for index in indices],
        "pair_ids": sorted({examples[index].pair_id for index in indices}),
    }


def _relative_drop(full: dict[str, Any], ablated: dict[str, Any]) -> float:
    return float(full["balanced_accuracy"] - ablated["balanced_accuracy"])


def main() -> None:
    args = _args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output must stay inside the repository")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = {
        name: (ROOT / value).resolve()
        for name, value in {
            "reference": args.reference_checkpoint,
            "phase": args.phase_checkpoint,
            "controls": args.control_checkpoint,
            "metadata": args.reference_metadata,
            "contract": args.contract,
            "fresh_lock": args.fresh_lock,
            "protocol": args.experiment_protocol,
        }.items()
    }
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    lock = json.loads(paths["fresh_lock"].read_text(encoding="utf-8"))
    if contract.get("schema") != "ergt-phi-m2-target-shortcut-audit-contract-v1":
        raise ValueError("unknown target/shortcut audit contract")
    if protocol.get("experiment_id") != "M2-E004" or protocol.get("revision") != "v001":
        raise ValueError("runner requires M2-E004/v001")
    registered = {item["path"]: item["sha256"] for item in protocol["inputs"]}
    observed_hashes = {
        Path(args.reference_checkpoint).as_posix(): sha256(paths["reference"]),
        Path(args.phase_checkpoint).as_posix(): sha256(paths["phase"]),
        Path(args.control_checkpoint).as_posix(): sha256(paths["controls"]),
    }
    expected_hashes = {
        Path(args.reference_checkpoint).as_posix(): contract["reference_checkpoint_sha256"],
        Path(args.phase_checkpoint).as_posix(): contract["selected_checkpoint_sha256"],
        Path(args.control_checkpoint).as_posix(): contract["control_checkpoint_sha256"],
    }
    if observed_hashes != expected_hashes or any(
        registered.get(path) != digest for path, digest in observed_hashes.items()
    ):
        raise ValueError("a frozen checkpoint differs from the registered inputs")
    input_file_hashes_before = dict(observed_hashes)

    fresh_cfg = contract["fresh_development"]
    fresh, tokenizer, fresh_audit = build_fresh_development(
        paths["metadata"], data_seed=int(fresh_cfg["data_seed"]),
        pair_counts_by_hop=fresh_cfg["pair_counts_by_hop"],
        unsupported_pairs=int(fresh_cfg["unsupported_pairs"]),
    )
    if len(fresh) != int(fresh_cfg["expected_examples"]) or fresh_audit["pairs"] != int(fresh_cfg["expected_pairs"]):
        raise ValueError("fresh cohort size differs from contract")
    first, second = confirmation_windows(fresh, seed=int(fresh_cfg["window_seed"]))
    observed_windows = {"window_1": _identity(fresh, first), "window_2": _identity(fresh, second)}
    fresh_lock_match = bool(
        fresh_audit["fresh_cohort_sha256"] == lock["fresh_cohort_sha256"]
        and fresh_audit["rows_sha256"] == lock["rows_sha256"]
        and observed_windows == lock["windows"]
    )
    if not fresh_lock_match:
        raise ValueError("fresh cohort/windows differ from the accepted v004 lock")

    reference_state = torch.load(paths["reference"], map_location="cpu", weights_only=True)
    native_cfg = dict(reference_state["config"])
    native_cfg["raw_input_contract"] = RawTokenInputContract(**native_cfg["raw_input_contract"])
    baseline = SteppedNative(ERGT43Config(**native_cfg)).eval()
    baseline.load_state_dict(reference_state["best_state"], strict=True)
    baseline.requires_grad_(False)
    baseline_before = _state_sha(baseline)

    selected = torch.load(paths["phase"], map_location="cpu", weights_only=False)
    selected_contract = selected["contract"]
    phase_cfg = CalibrationConfig(**selected_contract["calibration_config"])
    input_dim = int(selected["model"]["network.0.weight"].numel())
    worlds, relations = selected["model"]["initial_offsets"].shape
    phase = PhaseAnchorNetwork(input_dim, worlds, relations, phase_cfg)
    optimizer = torch.optim.AdamW(
        phase.parameters(), lr=phase_cfg.learning_rate, weight_decay=phase_cfg.weight_decay,
    )
    progress = restore(selected, phase, optimizer, contract=selected_contract)
    if not progress.get("frozen") or any(parameter.requires_grad for parameter in phase.parameters()):
        raise ValueError("selected phase checkpoint is not frozen")
    phase.eval()
    phase_before = _state_sha(phase)

    control_state = torch.load(paths["controls"], map_location="cpu", weights_only=False)
    if control_state.get("schema") != "ergt-phi-m2-control-checkpoints-v1":
        raise ValueError("unknown v004 control checkpoint schema")
    q_meta = control_state["q_only_training"]
    no_meta = control_state["no_phase_training"]
    q_only = EndpointClassifier(
        int(q_meta["node_dim"]), int(q_meta["hidden_dim"]), int(relations),
        seed=int(contract.get("q_only_seed", 29092026)),
    )
    no_phase = EndpointClassifier(
        int(no_meta["node_dim"]), int(no_meta["hidden_dim"]), int(relations),
        seed=int(contract.get("no_phase_seed", 30092026)),
    )
    q_only.load_state_dict(control_state["q_only"], strict=True)
    no_phase.load_state_dict(control_state["no_phase"], strict=True)
    q_only.eval().requires_grad_(False)
    no_phase.eval().requires_grad_(False)
    q_before, no_before = _state_sha(q_only), _state_sha(no_phase)

    old, old_tokenizer, old_sha = training_only(paths["metadata"])
    fit, historical_monitor = pair_partition(old, phase_cfg.split_seed)
    print("Caching historical frozen features", flush=True)
    old_records, old_cache = cache_q_features(baseline, old, old_tokenizer)
    print("Caching locked fresh-development features", flush=True)
    fresh_records, fresh_cache = cache_q_features(baseline, fresh, tokenizer)
    q_cfg = contract["q_ablation"]
    q_width = int(q_cfg["q_feature_width"])
    permutation = tuple(int(value) for value in q_cfg["relation_permutation"])
    old_raw = transform_q_records(old_records, q_width=q_width, mode="raw_only", permutation=permutation)
    fresh_views = {
        mode: transform_q_records(fresh_records, q_width=q_width, mode=mode, permutation=permutation)
        for mode in ("full", "zero", "permute", "raw_only")
    }

    raw_cfg = contract["raw_only_control"]
    raw_only, raw_training = train_control(
        old_raw, fit, q_only=False, q_width=q_width,
        hidden_dim=int(raw_cfg["hidden_dim"]), relations=int(relations),
        seed=int(raw_cfg["seed"]), epochs=int(raw_cfg["fixed_epochs"]),
        batch_size=int(raw_cfg["batch_size"]), learning_rate=float(raw_cfg["learning_rate"]),
        weight_decay=float(raw_cfg["weight_decay"]),
    )
    raw_capacity_difference = abs(parameter_count(raw_only) - parameter_count(phase)) / parameter_count(phase)

    target_source = explicit_target_audit(fresh)
    _write_json(output / "target_source_audit.json", target_source)
    predictions: list[dict[str, Any]] = []
    native_rows_all: list[dict[str, Any]] = []
    windows: dict[str, Any] = {}
    batch_size = int(raw_cfg["batch_size"])
    for window, indices in (("window_1", first), ("window_2", second)):
        arm_metrics: dict[str, Any] = {}
        arm_rows: dict[str, list[dict[str, Any]]] = {}
        for mode in ("full", "zero", "permute"):
            arm = f"q_only_{mode}"
            arm_metrics[arm], arm_rows[arm] = evaluate_classifier_arm(
                q_only, fresh_views[mode], indices, q_only=True, q_width=q_width,
                batch_size=batch_size, arm=arm, window=window,
            )
            arm = f"no_phase_{mode}"
            arm_metrics[arm], arm_rows[arm] = evaluate_classifier_arm(
                no_phase, fresh_views[mode], indices, q_only=False, q_width=q_width,
                batch_size=batch_size, arm=arm, window=window,
            )
            arm = f"phase_{mode}"
            arm_metrics[arm], arm_rows[arm] = evaluate_phase_arm(
                phase, fresh_views[mode], indices, arm=arm, window=window,
            )
        arm_metrics["raw_only"], arm_rows["raw_only"] = evaluate_classifier_arm(
            raw_only, fresh_views["raw_only"], indices, q_only=False, q_width=q_width,
            batch_size=batch_size, arm="raw_only", window=window,
        )
        predictions.extend(row for rows in arm_rows.values() for row in rows)

        print(f"Evaluating frozen native answer/path panel: {window}", flush=True)
        native_metrics, native_rows, _ = evaluate_native(
            baseline, tuple(fresh[index] for index in indices), tokenizer,
            device=torch.device("cpu"), batch_size=4, cohort=window,
            max_pairwise_cells=2_500_000,
        )
        native_rows_all.extend(native_rows)
        effects = {}
        for family in ("q_only", "no_phase", "phase"):
            full_rows = arm_rows[f"{family}_full"]
            effects[family] = {
                "zero_balanced_accuracy_drop": _relative_drop(
                    arm_metrics[f"{family}_full"], arm_metrics[f"{family}_zero"],
                ),
                "permutation_balanced_accuracy_drop": _relative_drop(
                    arm_metrics[f"{family}_full"], arm_metrics[f"{family}_permute"],
                ),
                "zero_prediction_change_rate": prediction_change_rate(
                    full_rows, arm_rows[f"{family}_zero"],
                ),
                "permutation_prediction_change_rate": prediction_change_rate(
                    full_rows, arm_rows[f"{family}_permute"],
                ),
            }
        conditional = {
            family: relation_accuracy_by_answer_outcome(arm_rows[f"{family}_full"], native_rows)
            for family in ("q_only", "no_phase", "phase")
        }
        expected = contract["parent_full_balanced_accuracy"][window]
        parent_reproduced = all(
            math.isclose(
                float(arm_metrics[f"{family}_full"]["balanced_accuracy"]),
                float(expected[family]), abs_tol=1e-12, rel_tol=0.0,
            )
            for family in ("phase", "q_only", "no_phase")
        )
        windows[window] = {
            "examples": len(indices),
            "pairs": len(observed_windows[window]["pair_ids"]),
            "relation_arms": arm_metrics,
            "ablation_effects": effects,
            "native_answer_path": native_metrics,
            "relation_accuracy_by_native_answer_outcome": conditional,
            "parent_v004_full_baselines_reproduced": parent_reproduced,
        }

    with (output / "relation_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output / "native_answer_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in native_rows_all:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    torch.save({
        "schema": "ergt-phi-m2-raw-only-control-v1",
        "model": raw_only.state_dict(),
        "training": raw_training,
    }, output / "raw_only_checkpoint.pt")

    policy = contract["shortcut_policy"]
    shortcut_windows = []
    for value in windows.values():
        q_metrics = value["relation_arms"]["q_only_full"]
        q_effect = value["ablation_effects"]["q_only"]
        shortcut_windows.append(
            q_metrics["balanced_accuracy"] >= float(policy["full_q_only_floor"])
            and max(
                q_effect["zero_balanced_accuracy_drop"],
                q_effect["permutation_balanced_accuracy_drop"],
            ) >= float(policy["minimum_zero_or_permutation_drop"])
        )
    shortcut_supported = bool(
        target_source["direct_token_label_exposure"] and all(shortcut_windows)
    )
    relation_target_choice_mismatch = bool(
        target_source["pair_relation_sequence_identity_rate"] == 1.0
        and target_source["supported_answer_flip_rate"] == 1.0
    )
    input_file_hashes_after = {
        Path(args.reference_checkpoint).as_posix(): sha256(paths["reference"]),
        Path(args.phase_checkpoint).as_posix(): sha256(paths["phase"]),
        Path(args.control_checkpoint).as_posix(): sha256(paths["controls"]),
    }
    integrity = {
        "fresh_lock_match": fresh_lock_match,
        "raw_text_disjoint_from_m0_training": fresh_audit["raw_text_disjoint_from_m0_training"],
        "counterfactual_pair_disjoint_from_m0_training": fresh_audit["counterfactual_pair_disjoint_from_m0_training"],
        "confirmation_windows_pair_disjoint": not bool(
            set(observed_windows["window_1"]["pair_ids"]) & set(observed_windows["window_2"]["pair_ids"])
        ),
        "reference_checkpoint_unchanged": baseline_before == _state_sha(baseline),
        "selected_checkpoint_unchanged": phase_before == _state_sha(phase),
        "q_only_checkpoint_unchanged": q_before == _state_sha(q_only),
        "no_phase_checkpoint_unchanged": no_before == _state_sha(no_phase),
        "input_files_unchanged": input_file_hashes_before == input_file_hashes_after,
        "fresh_labels_used_for_training_or_selection": bool(raw_training["fresh_labels_used"]),
        "historical_fit_monitor_disjoint": set(fit).isdisjoint(historical_monitor),
    }
    _write_json(output / "input_integrity.json", {
        **integrity,
        "input_hashes_before": input_file_hashes_before,
        "input_hashes_after": input_file_hashes_after,
        "fresh_cohort_sha256": fresh_audit["fresh_cohort_sha256"],
        "historical_training_cohort_sha256": old_sha,
        "windows": observed_windows,
    })
    result = {
        "schema": "ergt-phi-m2-target-shortcut-audit-result-v1",
        "status": "audit_completed",
        "experiment_id": "M2-E004",
        "revision": "v001",
        "target_shortcut_audit_complete": True,
        "zero_relation_channels_executed": True,
        "q_relation_permutation_executed": True,
        "raw_only_control_executed": True,
        "frozen_native_answer_path_target_used": True,
        "fresh_labels_used_for_training_or_selection": integrity["fresh_labels_used_for_training_or_selection"],
        "current_relation_target_direct_shortcut_risk_supported": shortcut_supported,
        "relation_target_cannot_discriminate_supported_counterfactual_choice": relation_target_choice_mismatch,
        "better_target_panel": {
            "name": contract["target_panel"]["name"],
            "scientifically_preferred_for_next_design": relation_target_choice_mismatch,
            "reason": "It measures frozen answer correctness and event/path execution, while relation labels are literal input tokens and remain identical when the supported counterfactual answer flips.",
            "weights_updated": False,
        },
        "M3_authorized_by_results": False,
        "m8_final_horizons_exposed": False,
        "phase_superiority_claimed": False,
        "target_source_audit": target_source,
        "windows": windows,
        "raw_only_training": raw_training,
        "raw_only_parameters": parameter_count(raw_only),
        "phase_parameters": parameter_count(phase),
        "raw_only_capacity_relative_difference": raw_capacity_difference,
        "integrity": integrity,
        "cost": {
            "historical_cache_seconds": old_cache["seconds"],
            "fresh_cache_seconds": fresh_cache["seconds"],
            "elapsed_seconds": time.perf_counter() - started,
            "device": "cpu",
            "threads": 1,
        },
        "contract_sha256": sha256(paths["contract"]),
        "fresh_lock_sha256": sha256(paths["fresh_lock"]),
        "experiment_protocol_sha256": sha256(paths["protocol"]),
    }
    _write_json(output / "result.json", result)
    _write_json(output / "m2_target_audit_handoff.json", {
        "schema": "ergt-phi-m2-target-audit-handoff-v1",
        "ready": True,
        "experiment_id": "M2-E004",
        "revision": "v001",
        "shortcut_risk_supported": shortcut_supported,
        "relation_target_choice_mismatch": relation_target_choice_mismatch,
        "preferred_next_target": contract["target_panel"]["name"],
        "M3_authorized": False,
        "m8_final_horizons_exposed": False,
        "next_stage": "Review M2 target choice and preregister a new M2 answer/path-aligned diagnostic; do not open M3.",
    })
    print(json.dumps({
        "status": result["status"],
        "shortcut_risk_supported": shortcut_supported,
        "relation_target_choice_mismatch": relation_target_choice_mismatch,
        "window_1_native_accuracy": windows["window_1"]["native_answer_path"]["accuracy"],
        "window_2_native_accuracy": windows["window_2"]["native_answer_path"]["accuracy"],
        "M3_authorized_by_results": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
