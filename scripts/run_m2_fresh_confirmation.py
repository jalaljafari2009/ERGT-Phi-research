"""Evaluate the frozen M2-Q checkpoint on preregistered fresh development data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ergt_phi.checkpoint import restore
from ergt_phi.m2_fresh import (
    build_fresh_development,
    cache_q_features,
    confirmation_windows,
    event_predictions,
    evaluate_control,
    matched_hidden_dim,
    parameter_count,
    phase_window_checks,
    train_control,
)
from ergt_phi.m2_q_training import sha256
from ergt_phi.native_steps import SteppedNative
from ergt_phi.shadow_data import pair_partition, training_only
from ergt_phi.shadow_metrics import evaluate_calibration
from ergt_phi.shadow_phase import CalibrationConfig, PhaseAnchorNetwork
from ergt_reviewer.matched_data import RawTokenInputContract
from ergt_reviewer.native_solver import ERGT43Config


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


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--phase-checkpoint", required=True)
    parser.add_argument("--reference-metadata", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--fresh-lock", required=True)
    parser.add_argument("--experiment-protocol", required=True)
    return parser.parse_args()


def _window_identity(examples: list[Any] | tuple[Any, ...], indices: list[int]) -> dict[str, list[str]]:
    return {
        "example_ids": [examples[index].example_id for index in indices],
        "pair_ids": sorted({examples[index].pair_id for index in indices}),
    }


def main() -> None:
    args = _args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output must remain inside the repository")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    paths = {
        name: (ROOT / value).resolve()
        for name, value in {
            "reference_checkpoint": args.reference_checkpoint,
            "phase_checkpoint": args.phase_checkpoint,
            "reference_metadata": args.reference_metadata,
            "contract": args.contract,
            "fresh_lock": args.fresh_lock,
            "protocol": args.experiment_protocol,
        }.items()
    }
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    fresh_lock = json.loads(paths["fresh_lock"].read_text(encoding="utf-8"))
    if contract.get("schema") != "ergt-phi-m2-fresh-confirmation-contract-v2":
        raise ValueError("unknown M2 fresh confirmation contract")
    if protocol.get("experiment_id") != "M2-E003" or protocol.get("revision") != "v004":
        raise ValueError("runner requires M2-E003/v004 protocol")
    maximum_minutes = float(protocol["resources"]["maximum_runtime_minutes"])
    deadline = time.monotonic() + maximum_minutes * 60

    input_hashes = {item["path"]: item["sha256"] for item in protocol["inputs"]}
    reference_sha = sha256(paths["reference_checkpoint"])
    phase_sha = sha256(paths["phase_checkpoint"])
    if reference_sha != contract["reference_checkpoint_sha256"]:
        raise ValueError("reference checkpoint differs from fresh confirmation contract")
    if phase_sha != contract["selected_checkpoint_sha256"]:
        raise ValueError("phase checkpoint differs from accepted M2-E002 selection")
    if input_hashes.get(Path(args.reference_checkpoint).as_posix()) != reference_sha:
        raise ValueError("protocol does not lock the reference checkpoint")
    if input_hashes.get(Path(args.phase_checkpoint).as_posix()) != phase_sha:
        raise ValueError("protocol does not lock the phase checkpoint")

    fresh_cfg = contract["fresh_development"]
    fresh_examples, tokenizer, fresh_audit = build_fresh_development(
        paths["reference_metadata"],
        data_seed=int(fresh_cfg["data_seed"]),
        pair_counts_by_hop=fresh_cfg["pair_counts_by_hop"],
        unsupported_pairs=int(fresh_cfg["unsupported_pairs"]),
    )
    if fresh_audit["examples"] != int(fresh_cfg["expected_examples"]):
        raise ValueError("fresh example count differs from contract")
    if fresh_audit["pairs"] != int(fresh_cfg["expected_pairs"]):
        raise ValueError("fresh pair count differs from contract")
    first, second = confirmation_windows(
        fresh_examples, seed=int(fresh_cfg["window_seed"]),
    )
    observed_windows = {
        "window_1": _window_identity(fresh_examples, first),
        "window_2": _window_identity(fresh_examples, second),
    }
    lock_match = bool(
        fresh_lock["fresh_cohort_sha256"] == fresh_audit["fresh_cohort_sha256"]
        and fresh_lock["rows_sha256"] == fresh_audit["rows_sha256"]
        and fresh_lock["windows"] == observed_windows
    )
    if not lock_match:
        raise ValueError("regenerated fresh cohort/windows differ from preregistered lock")
    fresh_audit.update({
        "fresh_lock_sha256": sha256(paths["fresh_lock"]),
        "fresh_lock_match": True,
        "windows": observed_windows,
        "window_pair_disjoint": not bool(
            set(observed_windows["window_1"]["pair_ids"])
            & set(observed_windows["window_2"]["pair_ids"])
        ),
    })
    _write_json(output / "fresh_development_audit.json", fresh_audit)

    reference_state = torch.load(paths["reference_checkpoint"], map_location="cpu", weights_only=True)
    model_config = dict(reference_state["config"])
    model_config["raw_input_contract"] = RawTokenInputContract(
        **model_config["raw_input_contract"]
    )
    baseline = SteppedNative(ERGT43Config(**model_config)).eval()
    baseline.load_state_dict(reference_state["best_state"], strict=True)
    baseline.requires_grad_(False)
    baseline_before = _state_sha(baseline)

    selected = torch.load(paths["phase_checkpoint"], map_location="cpu", weights_only=False)
    selected_contract = selected["contract"]
    phase_config = CalibrationConfig(**selected_contract["calibration_config"])
    input_dim = int(selected["model"]["network.0.weight"].numel())
    worlds, relations = selected["model"]["initial_offsets"].shape
    phase = PhaseAnchorNetwork(input_dim, worlds, relations, phase_config)
    phase_optimizer = torch.optim.AdamW(
        phase.parameters(), lr=phase_config.learning_rate,
        weight_decay=phase_config.weight_decay,
    )
    progress = restore(
        selected, phase, phase_optimizer, contract=selected_contract,
    )
    if not progress.get("frozen") or any(parameter.requires_grad for parameter in phase.parameters()):
        raise ValueError("accepted M2 phase checkpoint is not frozen")
    phase.eval()
    phase_before = _state_sha(phase)

    old_examples, old_tokenizer, old_sha = training_only(paths["reference_metadata"])
    fit, historical_monitor = pair_partition(old_examples, phase_config.split_seed)
    print("Caching historical fit features for controls", flush=True)
    old_records, old_cache = cache_q_features(
        baseline, old_examples, old_tokenizer, deadline=deadline,
    )
    print("Caching preregistered fresh-development features", flush=True)
    fresh_records, fresh_cache = cache_q_features(
        baseline, fresh_examples, tokenizer, deadline=deadline,
    )
    _write_json(output / "fresh_feature_cache_manifest.json", fresh_cache)

    control_cfg = contract["controls"]
    phase_parameters = parameter_count(phase)
    no_phase_hidden = int(control_cfg["no_phase_hidden_dim"])
    expected_hidden = matched_hidden_dim(input_dim, relations, phase_parameters)
    if no_phase_hidden != expected_hidden:
        raise ValueError("registered no-phase width is not nearest capacity match")
    q_only, q_training = train_control(
        old_records, fit, q_only=True,
        q_width=int(control_cfg["q_feature_width"]),
        hidden_dim=int(control_cfg["q_only_hidden_dim"]),
        relations=relations, seed=int(control_cfg["q_only_seed"]),
        epochs=int(control_cfg["fixed_epochs"]),
        batch_size=int(control_cfg["batch_size"]),
        learning_rate=float(control_cfg["learning_rate"]),
        weight_decay=float(control_cfg["weight_decay"]),
    )
    no_phase, no_phase_training = train_control(
        old_records, fit, q_only=False,
        q_width=int(control_cfg["q_feature_width"]),
        hidden_dim=no_phase_hidden,
        relations=relations, seed=int(control_cfg["no_phase_seed"]),
        epochs=int(control_cfg["fixed_epochs"]),
        batch_size=int(control_cfg["batch_size"]),
        learning_rate=float(control_cfg["learning_rate"]),
        weight_decay=float(control_cfg["weight_decay"]),
    )
    capacity_relative_difference = abs(
        no_phase_training["parameters"] - phase_parameters
    ) / phase_parameters
    if capacity_relative_difference > float(control_cfg["capacity_relative_difference_max"]):
        raise ValueError("no-phase control capacity is outside the registered tolerance")

    windows: dict[str, Any] = {}
    all_predictions: list[dict[str, Any]] = []
    for name, indices in (("window_1", first), ("window_2", second)):
        phase_metrics = evaluate_calibration(phase, fresh_records, indices)
        checks = phase_window_checks(phase_metrics, relations=relations)
        q_metrics = evaluate_control(
            q_only, fresh_records, indices, q_only=True,
            q_width=int(control_cfg["q_feature_width"]),
            batch_size=int(control_cfg["batch_size"]),
        )
        no_phase_metrics = evaluate_control(
            no_phase, fresh_records, indices, q_only=False,
            q_width=int(control_cfg["q_feature_width"]),
            batch_size=int(control_cfg["batch_size"]),
        )
        windows[name] = {
            "examples": len(indices),
            "pairs": len(observed_windows[name]["pair_ids"]),
            "phase": phase_metrics,
            "phase_checks": checks,
            "phase_all_checks": all(checks.values()),
            "q_only": q_metrics,
            "no_phase": no_phase_metrics,
            "phase_minus_q_only_balanced_accuracy": (
                phase_metrics["balanced_accuracy"] - q_metrics["balanced_accuracy"]
            ),
            "phase_minus_no_phase_balanced_accuracy": (
                phase_metrics["balanced_accuracy"] - no_phase_metrics["balanced_accuracy"]
            ),
        }
        all_predictions.extend(event_predictions(
            phase, q_only, no_phase, fresh_records, indices,
            q_width=int(control_cfg["q_feature_width"]), window=name,
        ))

    with (output / "event_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in all_predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    torch.save({
        "schema": "ergt-phi-m2-control-checkpoints-v1",
        "q_only": q_only.state_dict(),
        "no_phase": no_phase.state_dict(),
        "q_only_training": q_training,
        "no_phase_training": no_phase_training,
    }, output / "control_checkpoints.pt")

    reference_unchanged = baseline_before == _state_sha(baseline)
    phase_unchanged = phase_before == _state_sha(phase)
    fresh_labels_not_used = bool(
        not q_training["fresh_labels_used"]
        and not no_phase_training["fresh_labels_used"]
        and len(fit) + len(historical_monitor) == len(old_records)
        and set(fit).isdisjoint(historical_monitor)
    )
    confirmation_pass = all(window["phase_all_checks"] for window in windows.values())
    phase_leads_both = all(
        window["phase_minus_q_only_balanced_accuracy"] > 0
        and window["phase_minus_no_phase_balanced_accuracy"] > 0
        for window in windows.values()
    )
    result = {
        "schema": "ergt-phi-m2-fresh-confirmation-result-v1",
        "status": "fresh_development_confirmed" if confirmation_pass else "fresh_development_not_confirmed",
        "experiment_id": "M2-E003",
        "revision": "v004",
        "fresh_development_confirmation_pass": confirmation_pass,
        "M2_fresh_development_confirmed": confirmation_pass,
        "M3_authorized_by_results": False,
        "native_answer_improvement_claimed": False,
        "phase_specific_signal_descriptive": phase_leads_both,
        "phase_specific_causal_claim_confirmed": False,
        "fresh_lock_match": lock_match,
        "raw_text_disjoint_from_m0_training": fresh_audit["raw_text_disjoint_from_m0_training"],
        "counterfactual_pair_disjoint_from_m0_training": fresh_audit["counterfactual_pair_disjoint_from_m0_training"],
        "confirmation_windows_pair_disjoint": fresh_audit["window_pair_disjoint"],
        "m8_final_horizons_exposed": False,
        "fresh_labels_used_for_training_or_selection": not fresh_labels_not_used,
        "controls_executed": True,
        "selected_checkpoint_frozen": bool(progress["frozen"]),
        "selected_checkpoint_unchanged": phase_unchanged,
        "reference_checkpoint_unchanged": reference_unchanged,
        "selected_checkpoint_sha256": phase_sha,
        "reference_checkpoint_sha256": reference_sha,
        "fresh_cohort_sha256": fresh_audit["fresh_cohort_sha256"],
        "historical_training_cohort_sha256": old_sha,
        "windows": windows,
        "controls": {
            "q_only_training": q_training,
            "no_phase_training": no_phase_training,
            "phase_parameters": phase_parameters,
            "no_phase_capacity_relative_difference": capacity_relative_difference,
            "fixed_training_epochs": int(control_cfg["fixed_epochs"]),
            "selection_on_fresh_data": False,
        },
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
    handoff = {
        "schema": "ergt-phi-m2-fresh-handoff-v1",
        "ready": confirmation_pass,
        "experiment_id": "M2-E003",
        "revision": "v004",
        "selected_checkpoint_sha256": phase_sha,
        "fresh_cohort_sha256": fresh_audit["fresh_cohort_sha256"],
        "two_locked_windows_passed": confirmation_pass,
        "controls_completed": True,
        "phase_specific_signal_descriptive": phase_leads_both,
        "M3_authorized": False,
        "next_stage": (
            "M2 operational step 4 native/phase adapter design"
            if confirmation_pass else
            "M2 revision decision; do not activate native/phase adapter"
        ),
        "limitations": [
            "This is confirmatory development, not the untouched M8 final panel.",
            "The target is auxiliary relation calibration, not native answer quality.",
            "Control comparisons are descriptive in one development cohort.",
        ],
    }
    _write_json(output / "m2_fresh_handoff.json", handoff)
    print(json.dumps({
        "status": result["status"],
        "fresh_cohort_sha256": result["fresh_cohort_sha256"],
        "window_1_phase_balanced_accuracy": windows["window_1"]["phase"]["balanced_accuracy"],
        "window_2_phase_balanced_accuracy": windows["window_2"]["phase"]["balanced_accuracy"],
        "phase_specific_signal_descriptive": phase_leads_both,
        "M3_authorized_by_results": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
