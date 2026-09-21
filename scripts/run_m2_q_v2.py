"""Execute the registered M2-Q v2 checkpoint-handoff experiment.

This runner is the offline calibration entrypoint defined by ADR-0006. It does
not replace the public native forward, activate geometry coupling, or authorize
M3. Its job is to produce a first-qualified frozen checkpoint with exact
provenance, fresh-process reload and resume evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ergt_phi.lagged_q import native_proposal_q
from ergt_phi.m2_q_contract import load_m2_q_contract
from ergt_phi.m2_q_training import (
    calibration_config,
    sha256,
    train_with_checkpoints,
    verify_resume_in_fresh_process,
    verify_selected_reload_in_fresh_process,
)
from ergt_phi.native_steps import SteppedNative
from ergt_phi.shadow_data import pair_partition, training_only
from ergt_reviewer.matched_data import (
    RawTokenInputContract,
    collate_matched_topology_examples,
)
from ergt_reviewer.native_solver import ERGT43Config


def _exact(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError("native outputs have different keys")
        for key in left:
            _exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError("native outputs have different lengths")
        for first, second in zip(left, right):
            _exact(first, second)
    elif left != right:
        raise AssertionError(f"native output mismatch: {left!r} != {right!r}")


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _native_output_audit(outputs: dict[str, Any]) -> dict[str, Any]:
    """Persist a compact digest of every native field plus the actual answer logits."""
    fields: dict[str, Any] = {}
    for name, value in sorted(outputs.items()):
        if isinstance(value, torch.Tensor):
            fields[name] = {
                "kind": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
        elif isinstance(value, (str, int, float, bool)) or value is None:
            fields[name] = {"kind": "scalar", "value": value}
        else:
            fields[name] = {"kind": type(value).__name__, "repr": repr(value)}
    logits = outputs["native_answer_logits"].detach().cpu()
    return {
        "schema": "ergt-phi-native-output-audit-v1",
        "fields": fields,
        "native_answer_logits": logits.tolist(),
        "native_hard_decision": logits.argmax(-1).tolist(),
    }


def _deadline_guard(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TimeoutError("registered M2-Q runtime budget exceeded")


@torch.no_grad()
def cache_q_features(
    baseline: SteppedNative,
    examples: list[Any] | tuple[Any, ...],
    tokenizer: Any,
    *,
    deadline: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cache raw-only Psi0 and detached Q marginals before reading gold edges."""
    raw_records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, example in enumerate(examples):
        _deadline_guard(deadline)
        batch = collate_matched_topology_examples((example,), tokenizer)
        semantic, identity = baseline.raw_input_adapter(**batch.model_inputs())
        seed = baseline.substrate.seed_native_state(
            semantic, batch.base.attention_mask, identity_fibre=identity,
        )
        snapshot = baseline.substrate(
            semantic, batch.base.attention_mask, identity_fibre=identity,
        )
        proposal = baseline.probe_native_proposals(snapshot, batch.base.attention_mask)
        mask = batch.base.attention_mask[0]
        width = mask.numel()
        valid = mask[:, None] & mask[None, :]
        valid &= ~torch.eye(width, dtype=torch.bool, device=mask.device)
        edge_index = torch.nonzero(valid, as_tuple=False).t().contiguous()
        q_pre, _, _ = native_proposal_q(
            proposal, edge_index, attention_mask=batch.base.attention_mask,
        )
        q_pre = q_pre.detach()
        relations = q_pre.shape[1]
        source = q_pre.new_zeros((width, relations)).index_add(0, edge_index[0], q_pre)
        target = q_pre.new_zeros((width, relations)).index_add(0, edge_index[1], q_pre)
        q_context = torch.log1p(torch.cat((source, target), dim=-1))
        features = torch.cat((seed["psi"][0].detach(), q_context), dim=-1).cpu()
        raw_records.append({
            "psi": features,
            "example_id": example.example_id,
            "pair_id": example.pair_id,
        })
        manifest_rows.append({
            "example_id": example.example_id,
            "pair_id": example.pair_id,
            "feature_sha256": _tensor_sha256(features),
            "valid_tokens": int(mask.sum()),
            "candidate_edges": int(edge_index.shape[1]),
            "q_mass": float(q_pre.sum()),
        })
        if (index + 1) % 50 == 0:
            print(f"Cached Q-conditioned raw features: {index + 1}/{len(examples)}", flush=True)

    # This second pass is the first point at which gold endpoints/relations are read.
    records: list[dict[str, Any]] = []
    for raw, example in zip(raw_records, examples):
        records.append({
            **raw,
            "events": torch.tensor(
                [[edge.source_position, edge.target_position] for edge in example.base.edges],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [edge.relation_id - 1 for edge in example.base.edges], dtype=torch.long,
            ),
        })
    return records, {
        "schema": "ergt-phi-m2-q-feature-cache-v1",
        "raw_features_completed_before_supervision_attached": True,
        "reference_prepass_calls": len(examples),
        "proposal_probe_calls": len(examples),
        "seed_only_initializations": len(examples),
        "examples": len(examples),
        "seconds": time.perf_counter() - started,
        "rows": manifest_rows,
    }


def _peak_memory() -> dict[str, Any]:
    try:
        import resource
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB, macOS bytes. Colab is Linux.
        scale = 1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0
        return {"peak_rss_mb": value / scale, "method": "resource.ru_maxrss"}
    except (ImportError, AttributeError, OSError):
        return {"peak_rss_mb": None, "method": "unavailable_on_runtime"}


def _checkpoint_hashes(output: Path) -> dict[str, str]:
    return {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted(output.rglob("*.pt"))
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--reference-metadata", required=True)
    parser.add_argument("--contract", default="configs/m2_q_v2.json")
    parser.add_argument("--experiment-protocol", required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output must remain inside the repository")
    output.mkdir(parents=True, exist_ok=False)

    contract_path = (ROOT / args.contract).resolve()
    experiment_protocol_path = (ROOT / args.experiment_protocol).resolve()
    reference_checkpoint = (ROOT / args.reference_checkpoint).resolve()
    metadata_dir = (ROOT / args.reference_metadata).resolve()
    architecture_contract = load_m2_q_contract(contract_path)
    experiment_protocol = json.loads(experiment_protocol_path.read_text(encoding="utf-8"))
    if experiment_protocol.get("experiment_id") != "M2-E002" or experiment_protocol.get("revision") != "v001":
        raise ValueError("runner requires the registered M2-E002/v001 protocol")
    maximum_minutes = experiment_protocol.get("resources", {}).get("maximum_runtime_minutes")
    if not isinstance(maximum_minutes, (int, float)) or isinstance(maximum_minutes, bool) or maximum_minutes <= 0:
        raise ValueError("experiment protocol must pin a positive maximum_runtime_minutes")
    deadline = time.monotonic() + float(maximum_minutes) * 60.0

    audit_path = ROOT / "research/legacy/LEGACY-M0/manifests/trained_m0_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    reference_sha = sha256(reference_checkpoint)
    if audit.get("status") != "passed" or reference_sha != audit.get("checkpoint_sha256"):
        raise ValueError("accepted M0 reference checkpoint hash/status mismatch")
    input_lock = {item["path"]: item["sha256"] for item in experiment_protocol["inputs"]}
    expected_input = Path(args.reference_checkpoint).as_posix()
    if input_lock.get(expected_input) != reference_sha:
        raise ValueError("experiment input lock does not match the reference checkpoint")

    config = calibration_config(architecture_contract)
    random.seed(config.init_seed)
    np.random.seed(config.init_seed)
    torch.manual_seed(config.init_seed)

    state = torch.load(reference_checkpoint, map_location="cpu", weights_only=True)
    model_config = dict(state["config"])
    model_config["raw_input_contract"] = RawTokenInputContract(
        **model_config["raw_input_contract"]
    )
    baseline = SteppedNative(ERGT43Config(**model_config)).eval()
    baseline.load_state_dict(state["best_state"], strict=True)
    baseline.requires_grad_(False)
    baseline_state = {
        key: value.detach().clone() for key, value in baseline.state_dict().items()
    }
    examples, tokenizer, train_sha = training_only(metadata_dir)
    fit, monitor = pair_partition(examples, config.split_seed)
    split_manifest = {
        "fit_example_ids": [examples[index].example_id for index in fit],
        "monitor_example_ids": [examples[index].example_id for index in monitor],
        "pair_disjoint": not ({examples[index].pair_id for index in fit} &
                              {examples[index].pair_id for index in monitor}),
    }
    split_sha = hashlib.sha256(
        json.dumps(split_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    sources = [
        "configs/m2_q_v2.json",
        "ergt_phi/checkpoint.py",
        "ergt_phi/lagged_q.py",
        "ergt_phi/m2_q_contract.py",
        "ergt_phi/m2_q_training.py",
        "ergt_phi/shadow_data.py",
        "ergt_phi/shadow_metrics.py",
        "ergt_phi/shadow_phase.py",
        "scripts/m2_q_checkpoint_worker.py",
        "scripts/run_m2_q_v2.py",
    ]
    checkpoint_contract = {
        "schema": "ergt-phi-m2-q-checkpoint-contract-v1",
        "experiment_id": "M2-E002",
        "revision": "v001",
        "architecture_contract_sha256": sha256(contract_path),
        "experiment_protocol_sha256": sha256(experiment_protocol_path),
        "source_data_config_hashes": {
            "reference_checkpoint_sha256": reference_sha,
            "training_cohort_sha256": train_sha,
            "partition_sha256": split_sha,
            "reference_protocol_sha256": sha256(metadata_dir / "protocol.json"),
            "reference_data_manifest_sha256": sha256(metadata_dir / "data_manifest.json"),
            "source_sha256": {path: sha256(ROOT / path) for path in sources},
        },
        "calibration_config": asdict(config),
        "checkpoint_boundary": "after_complete_optimizer_step_no_gradient_accumulation",
    }
    _json_write(output / "checkpoint_contract.json", checkpoint_contract)
    _json_write(output / "split_manifest.json", {**split_manifest, "sha256": split_sha})

    first_batch = collate_matched_topology_examples((examples[0],), tokenizer)
    with torch.no_grad():
        native_before = baseline(**first_batch.model_inputs())
    torch.save(_native_output_audit(native_before), output / "native_output_before.pt")
    records, cache_manifest = cache_q_features(
        baseline, examples, tokenizer, deadline=deadline,
    )
    _json_write(output / "feature_cache_manifest.json", cache_manifest)
    _exact(baseline.state_dict(), baseline_state)

    worker = ROOT / "scripts/m2_q_checkpoint_worker.py"
    resume = verify_resume_in_fresh_process(
        records, fit, config,
        worlds=baseline.config.n_worlds,
        relations=3,
        checkpoint_contract=checkpoint_contract,
        output_dir=output,
        worker_script=worker,
    )

    def report_epoch(row: dict[str, Any]) -> None:
        print(json.dumps({
            "epoch": row["epoch"],
            "step": row["step"],
            "fit_balanced_accuracy": row["fit"]["balanced_accuracy"],
            "monitor_balanced_accuracy": row["monitor"]["balanced_accuracy"],
            "ready": row["all_readiness_checks"],
            "streak": row["streak"],
        }), flush=True)

    real = train_with_checkpoints(
        records, fit, monitor, config,
        worlds=baseline.config.n_worlds,
        relations=3,
        order_seed=config.shuffle_seed,
        checkpoint_contract=checkpoint_contract,
        output_dir=output,
        deadline=deadline,
        progress_callback=report_epoch,
    )
    _deadline_guard(deadline)
    shuffled = train_with_checkpoints(
        records, fit, monitor, config,
        worlds=baseline.config.n_worlds,
        relations=3,
        order_seed=config.shuffle_seed + 1,
        checkpoint_contract={**checkpoint_contract, "control": "within_batch_label_permutation"},
        output_dir=output / "shuffled_control",
        shuffle_labels=True,
        deadline=deadline,
    )

    with torch.no_grad():
        native_after_training = baseline(**first_batch.model_inputs())
    torch.save(
        _native_output_audit(native_after_training),
        output / "native_output_after_training.pt",
    )
    _exact(native_before, native_after_training)
    _exact(baseline.state_dict(), baseline_state)

    reload_report = verify_selected_reload_in_fresh_process(
        real, records, monitor, config,
        worlds=baseline.config.n_worlds,
        relations=3,
        checkpoint_contract=checkpoint_contract,
        output_dir=output,
        worker_script=worker,
    )
    if not real.qualified:
        _json_write(output / "reload_verification.json", reload_report)
    with torch.no_grad():
        native_after_reload = baseline(**first_batch.model_inputs())
    torch.save(
        _native_output_audit(native_after_reload),
        output / "native_output_after_reload.pt",
    )
    _exact(native_before, native_after_reload)
    _exact(baseline.state_dict(), baseline_state)
    reference_frozen = all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in baseline.parameters()
    )
    if not reference_frozen:
        raise AssertionError("reference model changed trainability or accumulated gradients")

    final = real.curves[-1]
    shuffled_final = shuffled.curves[-1]
    native_hard_decision_parity = torch.equal(
        native_before["native_answer_logits"].argmax(-1),
        native_after_reload["native_answer_logits"].argmax(-1),
    )
    engineering = {
        "selected_checkpoint_present": (output / "selected_full_state.pt").is_file(),
        "selected_checkpoint_frozen": bool(real.qualified and real.freeze_update_blocked),
        "fresh_process_reload_verified": bool(reload_report.get("pass")),
        "reload_metric_reproduced_exactly": bool(
            reload_report.get("metric_reproduced_exactly")
        ),
        "fresh_process_resume_verified": bool(resume["pass"]),
        "native_output_exact_before_after_and_reload": True,
        "native_hard_decision_parity": bool(native_hard_decision_parity),
        "reference_weights_and_gradients_unchanged": bool(reference_frozen),
        "shuffled_label_control_executed": True,
        "raw_features_completed_before_supervision_attached": True,
    }
    handoff_complete = bool(real.qualified and all(engineering.values()))
    result = {
        "schema": "ergt-phi-m2-q-v2-result-v1",
        "status": "qualified_handoff_ready" if handoff_complete else (
            "calibration_qualified_handoff_failed" if real.qualified
            else "calibration_not_qualified"
        ),
        "experiment_id": "M2-E002",
        "revision": "v001",
        "offline_shadow_entrypoint": True,
        "public_or_integrated_forward_used": False,
        "M2_checkpoint_handoff_complete": handoff_complete,
        "M3_authorized_by_results": False,
        "model_answer_improvement_claimed": False,
        "selected_epoch": real.selected_epoch,
        "selected_step": real.selected_step,
        "first_qualified_checkpoint_policy": True,
        "initial_fit": real.initial_fit,
        "initial_monitor": real.initial_monitor,
        "selected_fit": final["fit"],
        "selected_monitor": final["monitor"],
        "selected_checks": final["checks"],
        "selected_all_readiness_checks": final["all_readiness_checks"],
        "selected_streak": final["streak"],
        "curves": real.curves,
        "shuffled_initial_monitor": shuffled.initial_monitor,
        "shuffled_final_monitor": shuffled_final["monitor"],
        "shuffled_curves": shuffled.curves,
        "shuffle_method": "deterministic_within_batch_label_permutation_before_loss",
        "engineering_checks": engineering,
        "reload_verification": reload_report,
        "resume_verification": resume,
        "checkpoint_hashes": _checkpoint_hashes(output),
        "reference_checkpoint_sha256": reference_sha,
        "training_cohort_sha256": train_sha,
        "partition_sha256": split_sha,
        "architecture_contract_sha256": sha256(contract_path),
        "experiment_protocol_sha256": sha256(experiment_protocol_path),
        "fit_examples": len(fit),
        "monitor_examples": len(monitor),
        "cost": {
            "reference_prepass_seconds": cache_manifest["seconds"],
            "reference_prepass_calls": cache_manifest["reference_prepass_calls"],
            "proposal_probe_calls": cache_manifest["proposal_probe_calls"],
            "total_seconds": time.perf_counter() - started,
            "registered_maximum_runtime_minutes": maximum_minutes,
            **_peak_memory(),
        },
    }
    handoff = {
        "schema": "ergt-phi-m2-handoff-v1",
        "experiment_id": "M2-E002",
        "revision": "v001",
        "ready": handoff_complete,
        "selected_checkpoint": (
            "runs/m2_q_v2_handoff/selected_full_state.pt" if real.qualified else None
        ),
        "selected_checkpoint_sha256": (
            sha256(output / "selected_full_state.pt") if real.qualified else None
        ),
        "selected_epoch": real.selected_epoch,
        "selected_step": real.selected_step,
        "engineering_checks": engineering,
        "calibration_checks": final["checks"],
        "fresh_development_confirmation_complete": False,
        "next_stage": "M2 fresh-development confirmation (operational step 3)",
        "M3_authorized": False,
    }
    _json_write(output / "result.json", result)
    _json_write(output / "m2_handoff.json", handoff)
    print(json.dumps({
        "status": result["status"],
        "selected_epoch": result["selected_epoch"],
        "selected_monitor_balanced_accuracy": result["selected_monitor"]["balanced_accuracy"],
        "shuffled_monitor_balanced_accuracy": result["shuffled_final_monitor"]["balanced_accuracy"],
        "M2_checkpoint_handoff_complete": handoff_complete,
        "M3_authorized_by_results": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
