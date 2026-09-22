"""Run M2-E005/v001 frozen answer/path challenge and matched readout audit."""
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
from ergt_phi.m2_challenge import (
    build_challenge_panel,
    challenge_support,
    evaluate_scalar_readout,
    readout_parameter_count,
    representation_vectors,
    train_scalar_readout,
)
from ergt_phi.m2_fresh import build_fresh_development, cache_q_features, confirmation_windows
from ergt_phi.m2_q_training import sha256
from ergt_phi.m2_shortcut_audit import transform_q_records
from ergt_phi.native_steps import SteppedNative
from ergt_phi.shadow_data import pair_partition, training_only
from ergt_phi.shadow_phase import CalibrationConfig, PhaseAnchorNetwork
from ergt_reviewer.matched_data import RawTokenInputContract
from ergt_reviewer.native_solver import ERGT43Config


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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


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


def _load_frozen_reference(path: Path) -> SteppedNative:
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(state["config"])
    config["raw_input_contract"] = RawTokenInputContract(**config["raw_input_contract"])
    model = SteppedNative(ERGT43Config(**config)).eval()
    model.load_state_dict(state["best_state"], strict=True)
    return model.requires_grad_(False)


def _load_frozen_phase(path: Path) -> tuple[PhaseAnchorNetwork, dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    contract = state["contract"]
    config = CalibrationConfig(**contract["calibration_config"])
    input_dim = int(state["model"]["network.0.weight"].numel())
    worlds, relations = state["model"]["initial_offsets"].shape
    model = PhaseAnchorNetwork(input_dim, int(worlds), int(relations), config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    progress = restore(state, model, optimizer, contract=contract)
    if not progress.get("frozen") or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("selected phase checkpoint is not frozen")
    return model.eval(), {"config": config, "worlds": int(worlds), "relations": int(relations)}


def _support_pass(support: dict[str, Any], policy: dict[str, Any]) -> bool:
    return bool(
        (not policy["require_all_full_answers_correct"] or support["full_answer_correct"] == support["examples"])
        and support["answer_failures"] >= int(policy["minimum_answer_failures_per_window"])
        and support["answer_successes"] >= int(policy["minimum_answer_successes_per_window"])
        and support["event_chain_degradations"] >= int(policy["minimum_event_chain_degradations_per_window"])
        and support["response_margin_drop_std"] >= float(policy["minimum_response_margin_drop_std"])
        and (not policy["require_finite_margin_drop"] or support["response_margin_drop_finite"])
    )


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
            "metadata": args.reference_metadata,
            "contract": args.contract,
            "fresh_lock": args.fresh_lock,
            "protocol": args.experiment_protocol,
        }.items()
    }
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    lock = json.loads(paths["fresh_lock"].read_text(encoding="utf-8"))
    if contract.get("schema") != "ergt-phi-m2-answer-path-challenge-contract-v1":
        raise ValueError("unknown answer/path challenge contract")
    if protocol.get("experiment_id") != "M2-E005" or protocol.get("revision") != "v001":
        raise ValueError("runner requires M2-E005/v001")

    registered = {item["path"]: item["sha256"] for item in protocol["inputs"]}
    observed_hashes = {
        Path(args.reference_checkpoint).as_posix(): sha256(paths["reference"]),
        Path(args.phase_checkpoint).as_posix(): sha256(paths["phase"]),
    }
    expected_hashes = {
        Path(args.reference_checkpoint).as_posix(): contract["reference_checkpoint_sha256"],
        Path(args.phase_checkpoint).as_posix(): contract["selected_checkpoint_sha256"],
    }
    if observed_hashes != expected_hashes or any(
        registered.get(path) != digest for path, digest in observed_hashes.items()
    ):
        raise ValueError("a frozen checkpoint differs from the registered inputs")
    input_hashes_before = dict(observed_hashes)

    fresh_config = contract["fresh_development"]
    fresh, fresh_tokenizer, fresh_audit = build_fresh_development(
        paths["metadata"], data_seed=int(fresh_config["data_seed"]),
        pair_counts_by_hop=fresh_config["pair_counts_by_hop"],
        unsupported_pairs=int(fresh_config["unsupported_pairs"]),
    )
    if len(fresh) != int(fresh_config["expected_examples"]) or fresh_audit["pairs"] != int(fresh_config["expected_pairs"]):
        raise ValueError("fresh cohort size differs from contract")
    first, second = confirmation_windows(fresh, seed=int(fresh_config["window_seed"]))
    observed_windows = {"window_1": _identity(fresh, first), "window_2": _identity(fresh, second)}
    fresh_lock_match = bool(
        fresh_audit["fresh_cohort_sha256"] == lock["fresh_cohort_sha256"]
        and fresh_audit["rows_sha256"] == lock["rows_sha256"]
        and observed_windows == lock["windows"]
    )
    if not fresh_lock_match:
        raise ValueError("fresh cohort/windows differ from the accepted v004 lock")

    baseline = _load_frozen_reference(paths["reference"])
    phase, phase_metadata = _load_frozen_phase(paths["phase"])
    baseline_before, phase_before = _state_sha(baseline), _state_sha(phase)
    old, old_tokenizer, old_sha = training_only(paths["metadata"])
    fit, historical_monitor = pair_partition(old, phase_metadata["config"].split_seed)
    fit_examples = tuple(old[index] for index in fit)

    panel_config = contract["challenge_panel"]
    interventions = panel_config["intervention_by_scenario"]
    intervention_seed = int(panel_config["intervention_seed"])
    native_batch_size = int(panel_config["batch_size"])
    print("Building historical-fit frozen challenge targets", flush=True)
    historical_panel = build_challenge_panel(
        baseline, fit_examples, old_tokenizer,
        intervention_by_scenario=interventions, intervention_seed=intervention_seed,
        batch_size=native_batch_size,
    )
    print("Building locked fresh-development challenge targets", flush=True)
    fresh_panel = build_challenge_panel(
        baseline, fresh, fresh_tokenizer,
        intervention_by_scenario=interventions, intervention_seed=intervention_seed,
        batch_size=native_batch_size,
    )
    _write_jsonl(output / "historical_fit_challenge_rows.jsonl", historical_panel)
    _write_jsonl(output / "fresh_challenge_rows.jsonl", fresh_panel)

    print("Caching historical frozen Q features", flush=True)
    old_records, old_cache = cache_q_features(baseline, old, old_tokenizer)
    print("Caching locked fresh-development Q features", flush=True)
    fresh_records, fresh_cache = cache_q_features(baseline, fresh, fresh_tokenizer)
    q_config = contract["q_ablation"]
    q_width = int(q_config["q_feature_width"])
    permutation = tuple(int(value) for value in q_config["relation_permutation"])
    old_full = transform_q_records(old_records, q_width=q_width, mode="full", permutation=permutation)
    fresh_views = {
        mode: transform_q_records(fresh_records, q_width=q_width, mode=mode, permutation=permutation)
        for mode in ("full", "zero", "permute")
    }
    old_fit_records = [old_full[index] for index in fit]
    target = torch.tensor([row["response_margin_drop"] for row in historical_panel], dtype=torch.float32)
    fresh_target = torch.tensor([row["response_margin_drop"] for row in fresh_panel], dtype=torch.float32)

    readout_config = contract["readouts"]
    representations = ("raw_only", "q_only", "no_phase", "phase")
    readouts: dict[str, torch.nn.Module] = {}
    training: dict[str, dict[str, Any]] = {}
    for name in representations:
        print(f"Training historical-fit {name} scalar readout", flush=True)
        vectors = representation_vectors(
            old_fit_records, fit_examples, mode=name, q_width=q_width,
            phase=phase if name == "phase" else None,
        )
        readouts[name], training[name] = train_scalar_readout(
            vectors, target, target_parameters=int(readout_config["target_parameters"]),
            seed=int(readout_config["seeds"][name]), epochs=int(readout_config["fixed_epochs"]),
            batch_size=int(readout_config["batch_size"]),
            learning_rate=float(readout_config["learning_rate"]),
            weight_decay=float(readout_config["weight_decay"]),
        )

    parameter_counts = {name: readout_parameter_count(readouts[name]) for name in representations}
    target_parameters = int(readout_config["target_parameters"])
    capacity_differences = {
        name: abs(count - target_parameters) / target_parameters
        for name, count in parameter_counts.items()
    }
    capacity_matched = all(
        value <= float(readout_config["capacity_relative_difference_max"])
        for value in capacity_differences.values()
    )

    vector_cache: dict[str, torch.Tensor] = {
        "raw_only": representation_vectors(
            fresh_views["full"], fresh, mode="raw_only", q_width=q_width,
        )
    }
    for name in ("q_only", "no_phase", "phase"):
        for view in ("full", "zero", "permute"):
            vector_cache[f"{name}_{view}"] = representation_vectors(
                fresh_views[view], fresh, mode=name, q_width=q_width,
                phase=phase if name == "phase" else None,
            )

    predictions: list[dict[str, Any]] = []
    windows: dict[str, Any] = {}
    support_policy = contract["support_gates"]
    for window, indices in (("window_1", first), ("window_2", second)):
        subset_rows = [fresh_panel[index] for index in indices]
        support = challenge_support(subset_rows)
        support["passes_preregistered_gates"] = _support_pass(support, support_policy)
        metrics: dict[str, Any] = {}
        arms = ["raw_only"] + [
            f"{name}_{view}"
            for name in ("q_only", "no_phase", "phase")
            for view in ("full", "zero", "permute")
        ]
        index_tensor = torch.tensor(indices, dtype=torch.long)
        truth = fresh_target[index_tensor]
        for arm in arms:
            family = arm if arm == "raw_only" else arm.rsplit("_", 1)[0]
            info = training[family]
            arm_metrics, arm_prediction = evaluate_scalar_readout(
                readouts[family], vector_cache[arm][index_tensor], truth,
                target_mean=info["target_mean"], target_std=info["target_std"],
            )
            metrics[arm] = arm_metrics
            for local_index, example_index in enumerate(indices):
                row = fresh_panel[example_index]
                predictions.append({
                    "window": window,
                    "arm": arm,
                    "example_id": row["example_id"],
                    "pair_id": row["pair_id"],
                    "scenario": row["scenario"],
                    "target": float(truth[local_index]),
                    "prediction": float(arm_prediction[local_index]),
                    "residual": float(arm_prediction[local_index] - truth[local_index]),
                })
        windows[window] = {"support": support, "readout_metrics": metrics}

    _write_jsonl(output / "readout_predictions.jsonl", predictions)
    torch.save({
        "schema": "ergt-phi-m2-answer-path-readouts-v1",
        "models": {name: readouts[name].state_dict() for name in representations},
        "training": training,
        "parameter_counts": parameter_counts,
        "fresh_targets_used": False,
    }, output / "readout_checkpoints.pt")

    input_hashes_after = {
        Path(args.reference_checkpoint).as_posix(): sha256(paths["reference"]),
        Path(args.phase_checkpoint).as_posix(): sha256(paths["phase"]),
    }
    integrity = {
        "fresh_lock_match": fresh_lock_match,
        "raw_text_disjoint_from_m0_training": fresh_audit["raw_text_disjoint_from_m0_training"],
        "counterfactual_pair_disjoint_from_m0_training": fresh_audit["counterfactual_pair_disjoint_from_m0_training"],
        "confirmation_windows_pair_disjoint": not bool(
            set(observed_windows["window_1"]["pair_ids"]) & set(observed_windows["window_2"]["pair_ids"])
        ),
        "historical_fit_monitor_disjoint": set(fit).isdisjoint(historical_monitor),
        "historical_targets_only_for_readout_training": True,
        "fresh_targets_used_for_training_selection_or_tuning": False,
        "reference_checkpoint_unchanged": baseline_before == _state_sha(baseline),
        "selected_checkpoint_unchanged": phase_before == _state_sha(phase),
        "input_files_unchanged": input_hashes_before == input_hashes_after,
        "capacity_matched": capacity_matched,
    }
    _write_json(output / "input_integrity.json", {
        **integrity,
        "input_hashes_before": input_hashes_before,
        "input_hashes_after": input_hashes_after,
        "fresh_cohort_sha256": fresh_audit["fresh_cohort_sha256"],
        "historical_training_cohort_sha256": old_sha,
        "windows": observed_windows,
    })
    panel_suitable = all(value["support"]["passes_preregistered_gates"] for value in windows.values())
    result = {
        "schema": "ergt-phi-m2-answer-path-challenge-result-v1",
        "status": "challenge_panel_validated" if panel_suitable else "challenge_panel_not_suitable",
        "experiment_id": "M2-E005",
        "revision": "v001",
        "challenge_panel_complete": True,
        "challenge_panel_suitable_for_answer_path_readout": panel_suitable,
        "primary_target": panel_config["primary_target"],
        "target_definition": panel_config["target_definition"],
        "target_positive_sign": panel_config["positive_sign"],
        "secondary_targets": panel_config["secondary_targets"],
        "intervention_by_scenario": interventions,
        "readout_training": training,
        "readout_parameters": parameter_counts,
        "capacity_relative_difference_from_target": capacity_differences,
        "capacity_matched": capacity_matched,
        "windows": windows,
        "integrity": integrity,
        "fresh_targets_used_for_training_selection_or_tuning": False,
        "M3_authorized_by_results": False,
        "m8_final_horizons_exposed": False,
        "phase_superiority_claimed": False,
        "scientific_scope": "Development challenge validation and descriptive frozen-readout audit only.",
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
    _write_json(output / "m2_answer_path_handoff.json", {
        "schema": "ergt-phi-m2-answer-path-handoff-v1",
        "ready": True,
        "experiment_id": "M2-E005",
        "revision": "v001",
        "challenge_panel_suitable": panel_suitable,
        "M3_authorized": False,
        "m8_final_horizons_exposed": False,
        "next_stage": (
            "Review the descriptive answer/path readouts and preregister the next M2 diagnostic; do not open M3."
            if panel_suitable else
            "Revise the M2 development target or challenge data without tuning thresholds; do not open M3."
        ),
    })
    print(json.dumps({
        "status": result["status"],
        "window_1_support": windows["window_1"]["support"],
        "window_2_support": windows["window_2"]["support"],
        "capacity_matched": capacity_matched,
        "M3_authorized_by_results": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
