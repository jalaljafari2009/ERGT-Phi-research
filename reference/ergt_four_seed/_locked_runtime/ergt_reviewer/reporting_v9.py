"""Compact publication-facing tables for the geometric reasoning study."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import write_csv, write_json


MODEL_DISPLAY_NAMES = {
    "native_ergt": "ERGT Geometric Model",
    "direct_transformer": "Direct Transformer Baseline",
}


def _compact(row: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {
        field: (
            json.dumps(row[field], sort_keys=True)
            if isinstance(row.get(field), (dict, list, tuple))
            else row.get(field)
        )
        for field in fields
        if field in row
    }


def _compact_with_model(
    row: Mapping[str, Any], fields: Sequence[str]
) -> dict[str, Any]:
    result = _compact(row, fields)
    if "model" in result:
        result["model"] = MODEL_DISPLAY_NAMES.get(str(result["model"]), result["model"])
    return result


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _control_family(intervention: str) -> str:
    if intervention in {"shuffled_geometry", "random_geometry"}:
        return "geometry_randomization"
    if intervention.startswith("only_world_"):
        return "single_world_restriction"
    return "geometric_component_ablation"


def write_compact_tables(
    root: Path,
    *,
    claim_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
    verdict: Mapping[str, Any],
    resources: Sequence[Mapping[str, Any]],
    cohort_metrics: Sequence[Mapping[str, Any]],
    interventions: Sequence[Mapping[str, Any]],
    checkpoint_interventions: Sequence[Mapping[str, Any]],
    observers: Sequence[Mapping[str, Any]],
    statistics: Sequence[Mapping[str, Any]],
    data_audits: Sequence[Mapping[str, Any]],
    internalization_audits: Sequence[Mapping[str, Any]],
    runtime_invariance_audits: Sequence[Mapping[str, Any]],
    stability_invariant_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """Write T01--T13 before the notebook exposes complete raw exports."""

    table_root = root / "compact_tables"
    table_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    t01_rows = [
        {
            "row_type": "claim",
            "name": row.get("claim"),
            "status": row.get("status"),
            "gate": row.get("gate"),
            "scope_or_detail": row.get("scope"),
            "blocking_for_execution_integrity": False,
        }
        for row in claim_rows
    ] + [
        {
            "row_type": "gate",
            "name": row.get("gate"),
            "status": (
                "not_tested"
                if row.get("passed") is None
                else ("passed" if bool(row.get("passed")) else "open")
            ),
            "gate": row.get("gate"),
            "scope_or_detail": row.get("detail"),
            "blocking_for_execution_integrity": row.get("blocking_for_execution_integrity", False),
        }
        for row in gate_rows
    ]
    path = table_root / "T01_claim_and_gate_matrix.csv"
    write_csv(path, t01_rows)
    paths.append(path)

    t02_rows: list[dict[str, Any]] = []
    for row in resources:
        if row.get("model") != "native_ergt":
            continue
        exposure = dict(row.get("training_exposure_audit") or {})
        t02_rows.append(
            {
                "optimization_seed": row.get("optimization_seed"),
                "data_seed": row.get("data_seed"),
                "field_ready_step": row.get("independent_field_readiness_step"),
                "selected_checkpoint_step": row.get("selected_checkpoint_step"),
                "readiness_hops": json.dumps(row.get("native_readiness_hops", [])),
                "strict_v21_contract": row.get("strict_v21_chain_closure_overflow_contract"),
                "pair_group_count": exposure.get("pair_group_count"),
                "mean_group_exposure": exposure.get("mean_group_exposure"),
                "minimum_group_exposure": exposure.get("minimum_group_exposure"),
                "required_minimum_group_exposure": exposure.get("required_minimum_group_exposure"),
                "unseen_group_count": exposure.get("unseen_group_count"),
                "zero_teacher_optimizer_steps": row.get("zero_teacher_optimizer_steps"),
                "mechanism_shards_passed": row.get("mechanism_tuning_shards_passed"),
                "mechanism_shards_expected": row.get("mechanism_tuning_shards_expected"),
                "evaluation_state_status": row.get("evaluation_state_status"),
            }
        )
    path = table_root / "T02_checkpoint_and_exposure_audit.csv"
    write_csv(path, t02_rows)
    paths.append(path)

    path = table_root / "T02b_native_stability_invariants.csv"
    write_csv(path, stability_invariant_rows)
    paths.append(path)

    t03_rows: list[dict[str, Any]] = []
    matched_seeds = set(verdict.get("matched_converged_optimization_seeds", []))
    for row in verdict.get("convergence_by_seed", []):
        for model in ("native_ergt", "direct_transformer"):
            t03_rows.append(
                {
                    "optimization_seed": row["optimization_seed"],
                    "data_seed": row["data_seed"],
                    "model": MODEL_DISPLAY_NAMES[model],
                    "train_id_accuracy": row[f"{model}_train_id_accuracy"],
                    "validation_id_accuracy": row[f"{model}_validation_id_accuracy"],
                    "model_specific_convergence": row[f"{model}_id_converged"],
                    "blocks_ergt_claim": model == "native_ergt",
                    "included_in_matched_convergence_analysis": (
                        row["optimization_seed"] in matched_seeds
                    ),
                }
            )
    path = table_root / "T03_independent_convergence.csv"
    write_csv(path, t03_rows)
    paths.append(path)

    horizon_rows = [row for row in cohort_metrics if str(row.get("cohort", "")).startswith("hop_")]
    horizon_pivot: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for row in horizon_rows:
        key = (row.get("optimization_seed"), row.get("data_seed"), row.get("model"))
        target = horizon_pivot.setdefault(
            key,
            {
                "optimization_seed": key[0],
                "data_seed": key[1],
                "model": MODEL_DISPLAY_NAMES.get(str(key[2]), key[2]),
            },
        )
        target[f"accuracy_{row['cohort']}"] = row.get("accuracy")
        target[f"pair_exact_{row['cohort']}"] = row.get("counterfactual_pair_exact")
    for model in ("native_ergt", "direct_transformer"):
        aggregate = {
            "optimization_seed": "mean",
            "data_seed": "mean",
            "model": MODEL_DISPLAY_NAMES[model],
        }
        for cohort in sorted({str(row["cohort"]) for row in horizon_rows}):
            selected = [
                float(row["accuracy"])
                for row in horizon_rows
                if row.get("model") == model and row.get("cohort") == cohort
            ]
            aggregate[f"accuracy_{cohort}"] = _mean(selected)
        horizon_pivot[("mean", "mean", model)] = aggregate
    path = table_root / "T04_horizon_matrix.csv"
    write_csv(path, list(horizon_pivot.values()))
    paths.append(path)

    t05_rows = [
        _compact_with_model(
            row,
            (
                "optimization_seed",
                "data_seed",
                "model",
                "cohort",
                "accuracy",
                "supported_accuracy",
                "counterfactual_pair_exact",
                "inference_seconds",
            ),
        )
        for row in cohort_metrics
        if str(row.get("cohort", "")).startswith("context_")
    ]
    path = table_root / "T05_context_matrix.csv"
    write_csv(path, t05_rows)
    paths.append(path)

    t06_rows = [
        _compact_with_model(
            row,
            (
                "optimization_seed",
                "data_seed",
                "model",
                "cohort",
                "accuracy",
                "unsupported_examples",
                "unsupported_error_rate",
            ),
        )
        for row in cohort_metrics
        if str(row.get("cohort", "")).startswith("unsupported_")
    ]
    path = table_root / "T06_unsupported_rejection.csv"
    write_csv(path, t06_rows)
    paths.append(path)

    intervention_fields = (
        "optimization_seed",
        "data_seed",
        "mechanism_shard",
        "scenario",
        "intervention",
        "panel_full_accuracy",
        "full_accuracy",
        "intervention_accuracy",
        "targeted_drop",
        "eligible_pairs",
        "attribution_evaluable",
    )
    t07_rows = [
        {"panel": "checkpoint_selection", **_compact(row, intervention_fields)}
        for row in checkpoint_interventions
    ] + [
        {"panel": "heldout_final", **_compact(row, intervention_fields)}
        for row in interventions
        if row.get("scenario") != "all"
        and not str(row.get("intervention", "")).startswith("only_world_")
    ]
    path = table_root / "T07_registered_mechanism_matrix.csv"
    write_csv(path, t07_rows)
    paths.append(path)

    geometry_control_rows = [
        row
        for row in interventions
        if row.get("scenario") == "all"
        or str(row.get("intervention", "")).startswith("only_world_")
    ]
    t08_per_seed_rows = [
        {
            "aggregation": "per_seed",
            "control_family": _control_family(str(row.get("intervention", ""))),
            **_compact(row, intervention_fields),
        }
        for row in geometry_control_rows
    ]
    control_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in geometry_control_rows:
        control_groups[str(row.get("intervention", ""))].append(row)
    t08_aggregate_rows: list[dict[str, Any]] = []
    for intervention, rows in sorted(control_groups.items()):
        evaluable_rows = [
            row for row in rows if bool(row.get("attribution_evaluable", False))
        ]
        value_rows = evaluable_rows or rows
        t08_aggregate_rows.append(
            {
                "aggregation": "mean_across_seeds",
                "control_family": _control_family(intervention),
                "optimization_seed": "mean",
                "data_seed": "mean",
                "scenario": "all",
                "intervention": intervention,
                "panel_full_accuracy": _mean(
                    [
                        float(row["panel_full_accuracy"])
                        for row in value_rows
                        if row.get("panel_full_accuracy") is not None
                    ]
                ),
                "full_accuracy": _mean(
                    [
                        float(row["full_accuracy"])
                        for row in value_rows
                        if row.get("full_accuracy") is not None
                    ]
                ),
                "intervention_accuracy": _mean(
                    [
                        float(row["intervention_accuracy"])
                        for row in value_rows
                        if row.get("intervention_accuracy") is not None
                    ]
                ),
                "targeted_drop": _mean(
                    [
                        float(row["targeted_drop"])
                        for row in value_rows
                        if row.get("targeted_drop") is not None
                    ]
                ),
                "eligible_pairs": sum(int(row.get("eligible_pairs", 0)) for row in rows),
                "attribution_evaluable": len(evaluable_rows) == len(rows),
                "evaluable_seed_count": len(evaluable_rows),
                "total_seed_count": len(rows),
            }
        )
    t08_rows = t08_aggregate_rows + t08_per_seed_rows
    path = table_root / "T08_geometry_and_world_controls.csv"
    write_csv(path, t08_rows)
    paths.append(path)

    t09_rows = [
        _compact_with_model(
            row,
            (
                "optimization_seed",
                "data_seed",
                "cohort",
                "program_event_recall",
                "program_event_precision",
                "event_chain_exact",
                "event_world_path_coverage",
                "geodesic_closure_non_degradation_rate",
                "event_selection_overflow_rate",
                "effective_action_mae",
                "typed_conservation_residual",
                "payload_conservation_residual",
            ),
        )
        for row in cohort_metrics
        if row.get("model") == "native_ergt"
        and (row.get("cohort") == "validation" or str(row.get("cohort", "")).startswith("hop_"))
    ]
    path = table_root / "T09_chain_and_closure_diagnostics.csv"
    write_csv(path, t09_rows)
    paths.append(path)

    internal_by_seed = {
        (row.get("optimization_seed"), row.get("data_seed")): row for row in internalization_audits
    }
    invariance_by_seed = {
        (row.get("optimization_seed"), row.get("data_seed")): row
        for row in runtime_invariance_audits
    }
    t10_rows: list[dict[str, Any]] = []
    for audit in data_audits:
        key = (audit.get("optimization_seed"), audit.get("data_seed"))
        internal = internal_by_seed.get(key, {})
        invariance = invariance_by_seed.get(key, {})
        t10_rows.append(
            {
                "optimization_seed": key[0],
                "data_seed": key[1],
                "raw_fingerprint_collision_free": audit.get("raw_fingerprint_collision_free"),
                "unknown_token_free": audit.get("unknown_token_free"),
                "train_evaluation_raw_overlap_count": audit.get(
                    "train_evaluation_raw_overlap_count"
                ),
                "native_readiness_domain_matches_config": audit.get(
                    "native_readiness_domain_matches_config"
                ),
                "public_raw_token_only": internal.get("public_raw_token_only"),
                "identity_fibre_exact": internal.get(
                    "identity_fibre_bitwise_equal_to_internal_reference"
                ),
                "full_forward_exact": internal.get(
                    "full_forward_bitwise_equal_to_internal_reference"
                ),
                "batch_invariance_pass": invariance.get("batch_hard_and_mechanism_invariance_pass"),
                "padding_invariance_pass": invariance.get(
                    "padding_hard_and_mechanism_invariance_pass"
                ),
                "batch_mismatches": json.dumps(invariance.get("batch_mismatches", [])),
                "padding_mismatches": json.dumps(invariance.get("padding_mismatches", [])),
                "contract_pass": bool(internal.get("contract_pass"))
                and bool(invariance.get("contract_pass")),
            }
        )
    path = table_root / "T10_identity_and_invariance_audit.csv"
    write_csv(path, t10_rows)
    paths.append(path)

    observer_groups: dict[tuple[Any, Any, Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observers:
        observer_groups[
            (
                row.get("optimization_seed"),
                row.get("data_seed"),
                row.get("cohort"),
                row.get("hops"),
            )
        ].append(row)
    native_metrics_by_cohort = {
        (row.get("optimization_seed"), row.get("data_seed"), row.get("cohort")): row
        for row in cohort_metrics
        if row.get("model") == "native_ergt"
    }
    spectral_audit = dict(verdict.get("spectral_observer_numerical_audit") or {})
    observer_audit_fields = {
        "observer_role": "detached_reporting_only",
        "numerical_policy": spectral_audit.get("policy"),
        "fail_open": spectral_audit.get("fail_open"),
        "governing_answer_path_changed": spectral_audit.get(
            "governing_answer_path_changed"
        ),
        "graphs_solved_cpu_float64": spectral_audit.get("graphs_solved_cpu_float64"),
        "graphs_unavailable_after_numerical_failure": spectral_audit.get(
            "graphs_unavailable_after_numerical_failure"
        ),
        "nonfinite_values_sanitized": spectral_audit.get("nonfinite_values_sanitized"),
        "cuda_eigensolver_calls": spectral_audit.get("cuda_eigensolver_calls"),
    }
    t11_per_seed_rows: list[dict[str, Any]] = []
    for key, rows in observer_groups.items():
        native_metrics = native_metrics_by_cohort.get((key[0], key[1], key[2]), {})
        result = {
            "aggregation": "per_seed",
            "optimization_seed": key[0],
            "data_seed": key[1],
            "cohort": key[2],
            "hops": key[3],
            "examples": len(rows),
            "observer_only": True,
            "mean_curvature_proxy_negative_fraction": native_metrics.get(
                "curvature_proxy_negative_fraction"
            ),
            "mean_spectral_world_diversity": native_metrics.get(
                "event_spectral_world_diversity"
            ),
            **observer_audit_fields,
        }
        for field in (
            "curvature_proxy_mean",
            "curvature_gradient_mean_abs",
            "spectral_entropy",
            "spectral_effective_rank",
            "spectral_gap",
        ):
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            result[f"mean_{field}"] = _mean(values)
        t11_per_seed_rows.append(result)
    observer_cohort_groups: dict[
        tuple[Any, Any], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in t11_per_seed_rows:
        observer_cohort_groups[(row.get("cohort"), row.get("hops"))].append(row)
    t11_aggregate_rows: list[dict[str, Any]] = []
    observer_value_fields = (
        "mean_curvature_proxy_mean",
        "mean_curvature_proxy_negative_fraction",
        "mean_curvature_gradient_mean_abs",
        "mean_spectral_entropy",
        "mean_spectral_effective_rank",
        "mean_spectral_gap",
        "mean_spectral_world_diversity",
    )
    for (cohort, hops), rows in sorted(
        observer_cohort_groups.items(),
        key=lambda item: (str(item[0][0]), int(item[0][1] or 0)),
    ):
        aggregate: dict[str, Any] = {
            "aggregation": "mean_across_seeds",
            "optimization_seed": "mean",
            "data_seed": "mean",
            "cohort": cohort,
            "hops": hops,
            "examples": sum(int(row.get("examples", 0)) for row in rows),
            "observer_only": True,
            **observer_audit_fields,
        }
        for field in observer_value_fields:
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            aggregate[field] = _mean(values)
        t11_aggregate_rows.append(aggregate)
    t11_rows = t11_aggregate_rows + t11_per_seed_rows
    path = table_root / "T11_curvature_and_spectrum_observers.csv"
    write_csv(path, t11_rows)
    paths.append(path)

    t12_rows = [
        _compact_with_model(
            row,
            (
                "optimization_seed",
                "data_seed",
                "model",
                "training_seconds",
                "steps_completed",
                "optimizer_steps",
                "peak_cuda_allocated_gib",
                "selected_checkpoint_step",
                "zero_teacher_optimizer_steps",
                "evaluation_state_status",
            ),
        )
        for row in resources
    ]
    path = table_root / "T12_resource_disclosure.csv"
    write_csv(path, t12_rows)
    paths.append(path)

    t13_rows = [
        _compact(
            row,
            (
                "optimization_seed",
                "data_seed",
                "cohort",
                "paired_examples",
                "paired_counterfactual_clusters",
                "statistical_unit_is_counterfactual_pair",
                "independent_data_replicates",
                "hierarchical_data_seed_bootstrap",
                "native_minus_transformer_accuracy",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "mcnemar_exact_pvalue",
                "mcnemar_native_only_correct",
                "mcnemar_transformer_only_correct",
            ),
        )
        for row in statistics
        if row.get("optimization_seed") == "aggregate"
        or str(row.get("cohort", "")).startswith("hop_")
    ]
    path = table_root / "T13_paired_statistics.csv"
    write_csv(path, t13_rows)
    paths.append(path)

    manifest_path = table_root / "compact_table_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "ergt-geometric-study-compact-tables-v1",
            "display_order": [str(path.relative_to(root)) for path in paths],
            "raw_exports_follow_compact_tables": True,
            "curvature_and_spectrum_are_observer_only": True,
        },
    )
    paths.append(manifest_path)
    return paths


__all__ = ["write_compact_tables"]
