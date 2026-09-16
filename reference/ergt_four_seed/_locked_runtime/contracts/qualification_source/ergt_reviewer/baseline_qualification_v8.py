"""Prospective V8.1 qualification of a strong direct-Transformer baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    bundle,
    copy_to_colab_downloads,
    environment_record,
    write_csv,
    write_json,
)
from .baseline import (
    DirectTransformer,
    architecture_contract,
    count_parameters,
    inference_active_parameter_count,
    matched_transformer_config,
)
from .evaluation import evaluate_transformer
from .fair_data import (
    assert_shared_raw_input_contract,
    audit_label_contract,
    build_dataset_bundle,
)
from .matched_data import RawTokenInputContract, manifest_hash
from .native_solver import ERGT43Config, NativeGeometricBoundaryModel
from .suite import _locked_native_core_audit
from .training import (
    protocol_hash,
    set_seed,
    train_transformer_staged_curriculum_candidate,
)

SCHEMA_VERSION = "ergt-reviewer-baseline-qualification-v8.1"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_SOURCE_FILES = (
    "ergt_reviewer/baseline.py",
    "ergt_reviewer/baseline_qualification_v8.py",
    "ergt_reviewer/fair_data.py",
    "ergt_reviewer/training.py",
    "configs/baseline_qualification_v8.json",
)


def _device(value: str | torch.device | None) -> torch.device:
    if value is None or str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return selected


def _load_config() -> dict[str, Any]:
    path = PACKAGE_ROOT / "configs" / "baseline_qualification_v8.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _native_parameter_target(config: dict[str, Any], datasets: Any) -> int:
    native_config = ERGT43Config(
        vocab_size=datasets.tokenizer.vocab_size,
        max_tokens=int(config["max_tokens"]),
        raw_input_contract=RawTokenInputContract.from_tokenizer(datasets.tokenizer),
        hidden_dim=int(config["native_hidden_dim"]),
        psi_rank=int(config["native_psi_rank"]),
        field_steps=int(config["native_field_steps"]),
        sparse_top_k=int(config["native_sparse_top_k"]),
        max_hops=int(config["native_max_hops"]),
        geodesic_closure_steps=int(config["native_geodesic_closure_steps"]),
        geodesic_backbone_levels=int(config["native_geodesic_backbone_levels"]),
    )
    return count_parameters(NativeGeometricBoundaryModel(native_config))


def _candidate_model(
    config: dict[str, Any],
    datasets: Any,
    recipe: dict[str, Any],
    target_parameters: int,
) -> tuple[DirectTransformer, dict[str, Any]]:
    transformer_config = matched_transformer_config(
        vocab_size=datasets.tokenizer.vocab_size,
        max_tokens=int(config["max_tokens"]),
        query_token_id=int(datasets.tokenizer.token_to_id["query"]),
        candidate_a_token_id=int(datasets.tokenizer.token_to_id["candidate_a"]),
        candidate_b_token_id=int(datasets.tokenizer.token_to_id["candidate_b"]),
        target_parameters=target_parameters,
        candidate_layers=(int(recipe["n_layers"]),),
    )
    model = DirectTransformer(transformer_config)
    inference_parameters = inference_active_parameter_count(model)
    relative_difference = abs(inference_parameters - target_parameters) / max(1, target_parameters)
    return model, {
        "native_inference_active_parameters": target_parameters,
        "transformer_inference_active_parameters": inference_parameters,
        "inference_active_relative_parameter_difference": relative_difference,
        "within_ten_percent": relative_difference <= 0.10,
        "transformer_config": asdict(transformer_config),
    }


def _manifest_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _qualification_source_sha256() -> dict[str, str]:
    return {
        relative: hashlib.sha256((PACKAGE_ROOT / relative).read_bytes()).hexdigest()
        for relative in QUALIFICATION_SOURCE_FILES
    }


def _readout(
    verdict: dict[str, Any],
    selected: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
) -> str:
    lines = [
        "# V8.1 Direct-Transformer Baseline Qualification",
        "",
        f"- Schema: `{SCHEMA_VERSION}`",
        f"- Status: `{verdict['status']}`",
        "- Final 12--32-hop panels evaluated during qualification: `0`",
        "- Compiler/executor/geometry supplied to Transformer: `false`",
        "",
        "## Candidate Search",
        "",
        (
            "Development 6/8-hop thresholds are reported as observers and do not "
            "determine convergence eligibility."
        ),
        "",
        "| candidate | depth | loss | ID eligible | dev observer | dev accuracy | dev pair |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    for row in candidates:
        lines.append(
            f"| {row['candidate_id']} | {row['n_layers']} | {row['loss_mode']} | "
            f"{str(bool(row['selection_eligible'])).lower()} | "
            f"{str(bool(row['development_observer_pass'])).lower()} | "
            f"{float(row['development_supported_accuracy']):.3f} | "
            f"{float(row['development_pair_exact']):.3f} |"
        )
    lines.extend(("", "## Selected Baseline", ""))
    if selected is None:
        lines.append("No candidate satisfied the preregistered qualification contract.")
    else:
        recipe = dict(selected["recipe"])
        lines.extend(
            (
                f"- Candidate: `{recipe['candidate_id']}`",
                f"- Loss mode: `{recipe['loss_mode']}`",
                (
                    "- Validation supported accuracy: "
                    f"`{selected['validation_supported_accuracy']:.6f}`"
                ),
                f"- Validation pair exact: `{selected['validation_pair_exact']:.6f}`",
                "",
                (
                    "The selected recipe may be used only by the V8.1 final runner on fresh "
                    "optimization/data seeds. Qualification metrics are developmental and "
                    "are not confirmatory paper endpoints."
                ),
            )
        )
    return "\n".join(lines) + "\n"


def run_baseline_qualification_v8(
    *,
    device: str | torch.device | None = "auto",
    output_root: str | Path | None = None,
    run_id: str = "ergt_v8_1_transformer_baseline_qualification",
    resume: bool = True,
    copy_outputs_to_downloads: bool = True,
) -> dict[str, Any]:
    """Run a bounded development-only search and emit a locked recipe manifest."""

    config = _load_config()
    selected_device = _device(device)
    root = Path(output_root or (PACKAGE_ROOT / "runs")) / run_id
    root.mkdir(parents=True, exist_ok=True)
    locked_core = _locked_native_core_audit()
    if not bool(locked_core["locked_native_core_unchanged"]):
        raise RuntimeError("V8.1 qualification refuses a modified native core")

    optimization_seed = int(config["optimization_seed"])
    data_seed = int(config["data_seed"])
    datasets = build_dataset_bundle(config, data_seed)
    selection_cohorts = (
        datasets.train,
        datasets.tuning,
        datasets.mechanism_tuning,
        datasets.validation,
    )
    data_audit = assert_shared_raw_input_contract(
        selection_cohorts,
        datasets.tokenizer,
        training_examples=datasets.train,
    )
    label_audit = audit_label_contract(
        {
            "train": datasets.train,
            "tuning": datasets.tuning,
            "baseline_development": datasets.mechanism_tuning,
            "validation": datasets.validation,
        }
    )
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "protocol_revision": config["qualification_protocol_revision"],
        "prospective": True,
        "development_only": True,
        "v8_failed_qualification_preserved": True,
        "v7_successful_warmup_contract_restored": True,
        "warmup_budget_is_a_ceiling_with_early_convergence_stop": True,
        "optimization_seed": optimization_seed,
        "data_seed": data_seed,
        "same_unique_training_examples_for_all_candidates": True,
        "one_hop_warmup_then_staged_1_2_1_4_1_8": True,
        "exact_complete_pair_coverage_for_every_active_hop": True,
        "one_hop_replay_in_every_curriculum_batch": True,
        "id_convergence_determines_selection_eligibility": True,
        "six_eight_hop_development_only_ranks_eligible_checkpoints": True,
        "development_floors_are_nonblocking_observers": True,
        "validation_used_for_selection": False,
        "final_12_32_hop_panels_evaluated": 0,
        "transformer_compiler_or_executor_used": False,
        "transformer_geometry_used": False,
        "candidate_recipes": config["baseline_candidates"],
        "config": config,
        "locked_native_core_audit": locked_core,
        "dataset_fingerprints": {
            "train": manifest_hash(datasets.train),
            "tuning": manifest_hash(datasets.tuning),
            "development_6_8": manifest_hash(datasets.mechanism_tuning),
            "validation": manifest_hash(datasets.validation),
        },
    }
    protocol["protocol_sha256"] = _manifest_sha256(protocol)
    write_json(root / "qualification_protocol.json", protocol)
    write_json(root / "environment.json", environment_record(selected_device))
    write_json(root / "data_contract_audit.json", data_audit)
    write_json(root / "label_contract_audit.json", label_audit)
    write_json(root / "locked_native_core_audit.json", locked_core)

    target_parameters = _native_parameter_target(config, datasets)
    all_curves: list[dict[str, Any]] = []
    all_resources: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    candidate_records: dict[str, dict[str, Any]] = {}
    print(
        f"[reviewer-v8-qualification] schema={SCHEMA_VERSION} "
        f"candidates={len(config['baseline_candidates'])} device={selected_device}"
    )
    for recipe_value in config["baseline_candidates"]:
        recipe = dict(recipe_value)
        candidate_id = str(recipe["candidate_id"])
        set_seed(optimization_seed)
        model, parameter_audit = _candidate_model(config, datasets, recipe, target_parameters)
        model.to(selected_device)
        candidate_root = root / candidate_id
        candidate_root.mkdir(parents=True, exist_ok=True)
        run_hash = protocol_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_sha256": protocol["protocol_sha256"],
                "recipe": recipe,
                "parameter_audit": parameter_audit,
            }
        )
        curves, resource = train_transformer_staged_curriculum_candidate(
            model,
            datasets.train,
            datasets.tuning,
            datasets.mechanism_tuning,
            datasets.tokenizer,
            config=config,
            recipe=recipe,
            device=selected_device,
            seed=optimization_seed,
            checkpoint=candidate_root / "training.pt",
            run_hash=run_hash,
            resume=resume,
        )
        tuning_metrics, _ = evaluate_transformer(
            model,
            datasets.tuning,
            datasets.tokenizer,
            device=selected_device,
            batch_size=int(config["eval_batch_size"]),
            cohort="baseline_id_tuning",
        )
        development_metrics, _ = evaluate_transformer(
            model,
            datasets.mechanism_tuning,
            datasets.tokenizer,
            device=selected_device,
            batch_size=int(config["eval_batch_size"]),
            cohort="baseline_6_8_development",
        )
        development_observer_pass = float(development_metrics["supported_accuracy"]) >= float(
            config["baseline_development_accuracy_floor"]
        ) and float(development_metrics["counterfactual_pair_exact"]) >= float(
            config["baseline_development_pair_floor"]
        )
        selection_eligible = (
            bool(resource["selected_tuning_checkpoint"])
            and int(resource["stable_tuning_windows"]) >= int(config["checkpoint_stable_windows"])
            and bool(parameter_audit["within_ten_percent"])
            and all(architecture_contract(model).values())
        )
        row = {
            "candidate_id": candidate_id,
            "n_layers": int(recipe["n_layers"]),
            "loss_mode": str(recipe["loss_mode"]),
            "selection_eligible": selection_eligible,
            "development_observer_pass": development_observer_pass,
            "tuning_supported_accuracy": float(tuning_metrics["supported_accuracy"]),
            "tuning_pair_exact": float(tuning_metrics["counterfactual_pair_exact"]),
            "tuning_unsupported_error_rate": float(tuning_metrics["unsupported_error_rate"]),
            "development_supported_accuracy": float(development_metrics["supported_accuracy"]),
            "development_pair_exact": float(development_metrics["counterfactual_pair_exact"]),
            "selection_score": float(resource.get("selected_development_score") or -1.0),
            **parameter_audit,
        }
        candidate_rows.append(row)
        candidate_records[candidate_id] = {
            "recipe": recipe,
            "parameter_audit": parameter_audit,
            "resource": resource,
            "checkpoint_path": str(candidate_root / "training.pt"),
            "metrics": row,
        }
        all_curves.extend(curves)
        all_resources.append({**resource, **parameter_audit})
        model.to("cpu")
        del model
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()

    eligible = [row for row in candidate_rows if bool(row["selection_eligible"])]
    selected_row = max(
        eligible,
        key=lambda row: (float(row["selection_score"]), -int(row["n_layers"])),
        default=None,
    )
    selected_record: dict[str, Any] | None = None
    validation_evaluated = False
    validation_pass: bool | None = None
    if selected_row is not None:
        validation_evaluated = True
        selected_record = candidate_records[str(selected_row["candidate_id"])]
        recipe = dict(selected_record["recipe"])
        model, _ = _candidate_model(config, datasets, recipe, target_parameters)
        state = torch.load(
            selected_record["checkpoint_path"], map_location=selected_device, weights_only=False
        )
        if state.get("best_state") is None:
            raise RuntimeError("selected V8.1 baseline checkpoint has no eligible best_state")
        model.load_state_dict(state["best_state"])
        model.to(selected_device)
        validation_metrics, _ = evaluate_transformer(
            model,
            datasets.validation,
            datasets.tokenizer,
            device=selected_device,
            batch_size=int(config["eval_batch_size"]),
            cohort="baseline_validation_post_selection",
        )
        train_id = tuple(
            example
            for example in datasets.train
            if int(example.base.metadata["path_hops"]) == int(config["id_hops"])
        )
        train_id_metrics, _ = evaluate_transformer(
            model,
            train_id,
            datasets.tokenizer,
            device=selected_device,
            batch_size=int(config["eval_batch_size"]),
            cohort="baseline_train_id_post_selection",
        )
        validation_pass = (
            float(validation_metrics["supported_accuracy"])
            >= float(config["convergence_accuracy_floor"])
            and float(validation_metrics["counterfactual_pair_exact"])
            >= float(config["transformer_tuning_pair_floor"])
            and float(validation_metrics["unsupported_error_rate"])
            <= float(config["transformer_tuning_unsupported_error_ceiling"])
            and float(train_id_metrics["supported_accuracy"])
            >= float(config["convergence_accuracy_floor"])
        )
        selected_record = {
            **selected_record,
            "validation_supported_accuracy": float(validation_metrics["supported_accuracy"]),
            "validation_pair_exact": float(validation_metrics["counterfactual_pair_exact"]),
            "validation_unsupported_error_rate": float(
                validation_metrics["unsupported_error_rate"]
            ),
            "train_id_supported_accuracy": float(train_id_metrics["supported_accuracy"]),
        }

    qualification_pass = bool(
        selected_record
        and validation_pass is True
        and bool(data_audit["data_contract_pass"])
        and bool(label_audit["label_contract_pass"])
        and bool(locked_core["locked_native_core_unchanged"])
    )
    selected_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_revision": config["qualification_protocol_revision"],
        "qualification_pass": qualification_pass,
        "selection_is_development_only": True,
        "final_12_32_hop_panels_evaluated": 0,
        "optimization_seed": optimization_seed,
        "data_seed": data_seed,
        "protocol_sha256": protocol["protocol_sha256"],
        "selected_candidate": (
            {
                "candidate_id": selected_record["recipe"]["candidate_id"],
                "recipe": selected_record["recipe"],
                "parameter_audit": selected_record["parameter_audit"],
                "qualification_metrics": {
                    "tuning_supported_accuracy": selected_record["metrics"][
                        "tuning_supported_accuracy"
                    ],
                    "tuning_pair_exact": selected_record["metrics"]["tuning_pair_exact"],
                    "development_supported_accuracy": selected_record["metrics"][
                        "development_supported_accuracy"
                    ],
                    "development_pair_exact": selected_record["metrics"]["development_pair_exact"],
                    "development_observer_pass": selected_record["metrics"][
                        "development_observer_pass"
                    ],
                    "validation_supported_accuracy": selected_record[
                        "validation_supported_accuracy"
                    ],
                    "validation_pair_exact": selected_record["validation_pair_exact"],
                    "validation_unsupported_error_rate": selected_record[
                        "validation_unsupported_error_rate"
                    ],
                    "train_id_supported_accuracy": selected_record["train_id_supported_accuracy"],
                },
            }
            if selected_record is not None
            else None
        ),
        "forbidden_components": [
            "compiler",
            "generic_executor",
            "ERGT_graph",
            "geometry_side_input",
            "final_horizon_selection",
        ],
        "fresh_final_profile": "paper_final_v8",
        "qualification_source_sha256": _qualification_source_sha256(),
    }
    selected_manifest["manifest_sha256"] = _manifest_sha256(selected_manifest)
    verdict = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if qualification_pass else "failed",
        "qualification_pass": qualification_pass,
        "candidate_count": len(candidate_rows),
        "eligible_candidate_count": len(eligible),
        "development_observer_pass_count": sum(
            int(bool(row["development_observer_pass"])) for row in candidate_rows
        ),
        "selected_candidate_id": (
            selected_record["recipe"]["candidate_id"] if selected_record is not None else None
        ),
        "validation_evaluated_after_selection": validation_evaluated,
        "validation_pass_after_selection": validation_pass,
        "final_12_32_hop_panels_evaluated": 0,
        "next_step": (
            "run_paper_final_v8_on_fresh_seeds"
            if qualification_pass
            else "inspect_baseline_qualification_without_opening_final_panels"
        ),
    }
    write_json(root / "candidate_records.json", candidate_records)
    write_json(root / "selected_baseline_manifest.json", selected_manifest)
    write_json(root / "qualification_verdict.json", verdict)
    write_csv(root / "candidate_metrics.csv", candidate_rows)
    write_csv(root / "training_curves.csv", all_curves)
    write_csv(root / "resource_disclosure.csv", all_resources)
    readout_path = root / "qualification_readout.md"
    readout_path.write_text(_readout(verdict, selected_record, candidate_rows), encoding="utf-8")
    artifacts = [
        root / "qualification_protocol.json",
        root / "environment.json",
        root / "data_contract_audit.json",
        root / "label_contract_audit.json",
        root / "locked_native_core_audit.json",
        root / "candidate_records.json",
        root / "selected_baseline_manifest.json",
        root / "qualification_verdict.json",
        root / "candidate_metrics.csv",
        root / "training_curves.csv",
        root / "resource_disclosure.csv",
        readout_path,
    ]
    archive = bundle(root, artifacts)
    target_archive = root / "ergt_v8_baseline_qualification_bundle.zip"
    archive.replace(target_archive)
    copied = (
        copy_to_colab_downloads([*artifacts, target_archive]) if copy_outputs_to_downloads else []
    )
    print(
        f"[reviewer-v8-qualification] completed status={verdict['status']} "
        f"selected={verdict['selected_candidate_id']}"
    )
    print("manifest_path=", root / "selected_baseline_manifest.json")
    print("bundle_path=", target_archive)
    return {
        "run_root": str(root),
        "manifest_path": str(root / "selected_baseline_manifest.json"),
        "verdict_path": str(root / "qualification_verdict.json"),
        "readout_path": str(readout_path),
        "bundle_path": str(target_archive),
        "decision": verdict,
        "copied_outputs": copied,
    }


__all__ = ["SCHEMA_VERSION", "run_baseline_qualification_v8"]
