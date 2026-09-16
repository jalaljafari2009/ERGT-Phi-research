"""One-command ERGT geometric reasoning study with independent scientific statuses."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    bundle,
    copy_to_colab_downloads,
    environment_record,
    write_csv,
    write_figures,
    write_json,
    write_latex_tables,
)
from .baseline import architecture_contract as transformer_architecture_contract
from .evaluation_v9 import (
    TARGETED_MECHANISM_INTERVENTIONS,
    evaluate_native,
    evaluate_targeted_interventions,
    evaluate_transformer,
)
from .fair_data_v9 import (
    assert_shared_raw_input_contract,
    audit_label_contract,
    build_dataset_bundle,
    partition_mechanism_tuning_shards,
)
from .matched_data import (
    collate_matched_topology_examples,
    manifest_hash,
    reference_identity_fibre_tensor,
)
from .native_solver import (
    architecture_contract as native_architecture_contract,
)
from .native_solver import (
    hard_solutions_from_outputs,
)
from .reporting_v9 import write_compact_tables
from .spectral_runtime import (
    POLICY as SPECTRAL_NUMERICAL_POLICY,
)
from .spectral_runtime import (
    install_stable_event_spectral_observer,
    reset_spectral_observer_diagnostics,
    spectral_observer_diagnostics,
)
from .statistics import linear_slope, paired_comparison
from .training_v9 import protocol_hash, set_seed, train_models

SCHEMA_VERSION = "ergt-geometric-long-horizon-study-v1"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _load_profile(profile: str) -> dict[str, Any]:
    path = PACKAGE_ROOT / "configs" / f"{profile}.json"
    if not path.exists():
        raise ValueError(f"unknown study protocol: {profile}")
    return json.loads(path.read_text(encoding="utf-8"))


def _device(value: str | torch.device | None) -> torch.device:
    if value is None or str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return selected


def _publication_figure_curves(
    curves: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize heterogeneous V9 training rows for the legacy figure writer."""

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    metric_priority = (
        "tuning_supported_accuracy",
        "id_tuning_supported_accuracy",
        "tuning_accuracy",
        "readiness_supported_accuracy",
    )
    for row in curves:
        model = str(row.get("model", ""))
        raw_step = row.get("total_step", row.get("step"))
        if not model or raw_step is None:
            continue
        metric = next(
            (row.get(name) for name in metric_priority if row.get(name) is not None),
            None,
        )
        if metric is None:
            continue
        grouped[(model, int(raw_step))].append(float(metric))
    return [
        {
            "model": model,
            "step": step,
            "tuning_accuracy": sum(values) / len(values),
        }
        for (model, step), values in sorted(grouped.items())
    ]


def _source_manifest() -> dict[str, str]:
    records: dict[str, str] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        excluded = {"__pycache__", ".pytest_cache", "_smoke_runs", "runs"}
        if (
            path.is_file()
            and not any(part in excluded for part in path.parts)
            and path.suffix != ".pyc"
            and path.name != "ERGT_reviewer_package.zip"
        ):
            records[str(path.relative_to(PACKAGE_ROOT)).replace("\\", "/")] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return records


def _ergt43_v21_reference() -> dict[str, Any]:
    path = PACKAGE_ROOT / "manifests" / "ergt43_v21_reference.json"
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("immutable") is not True:
        raise ValueError("ERGT-43 V21 reference manifest must remain immutable")
    return reference


def _v6_repair_design() -> dict[str, Any]:
    path = PACKAGE_ROOT / "manifests" / "v6_repair_design.json"
    design = json.loads(path.read_text(encoding="utf-8"))
    if design.get("prospective") is not True:
        raise ValueError("V6 repair design must remain prospective")
    return design


def _canonical_source_sha256(path: Path) -> str:
    """Hash text source independently of Git checkout line endings."""

    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _locked_native_core_audit() -> dict[str, Any]:
    path = PACKAGE_ROOT / "manifests" / "v5_locked_native_core_hashes.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    observed = {
        relative: _canonical_source_sha256(PACKAGE_ROOT / relative) for relative in lock["files"]
    }
    mismatches = [
        relative
        for relative, expected in lock["files"].items()
        if observed.get(relative) != expected
    ]
    return {
        "schema_version": lock["schema_version"],
        "hash_policy": lock["hash_policy"],
        "immutable": bool(lock.get("immutable", False)),
        "expected_sha256": dict(lock["files"]),
        "observed_sha256": observed,
        "mismatched_files": mismatches,
        "locked_native_core_unchanged": bool(lock.get("immutable", False)) and not mismatches,
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / max(1, len(values))


def _raw_example_hashes(examples: Sequence[Any]) -> set[str]:
    return {
        hashlib.sha256(example.base.raw_text.encode("utf-8")).hexdigest() for example in examples
    }


def _claim_row(claim: str, passed: bool | None, gate: str, scope: str) -> dict[str, Any]:
    status = "not_tested" if passed is None else ("supported" if passed else "open")
    return {"claim": claim, "status": status, "gate": gate, "scope": scope}


def _v9_run_status(
    *,
    verdict_enabled: bool,
    integrity_pass: bool,
    native_claim_bundle_pass: bool,
    observed_paired_advantage_pass: bool,
) -> str:
    """Classify completion without using baseline convergence as a veto."""

    if not verdict_enabled:
        return "smoke_passed"
    if not integrity_pass:
        return "invalid_protocol"
    if native_claim_bundle_pass and observed_paired_advantage_pass:
        return "passed"
    return "completed_with_open_claims"


def _native_stability_invariant_rows(
    *,
    config: Mapping[str, Any],
    resources: Sequence[Mapping[str, Any]],
    architecture_audits: Sequence[Mapping[str, Any]],
    internalization_audits: Sequence[Mapping[str, Any]],
    runtime_invariance_audits: Sequence[Mapping[str, Any]],
    interventions: Sequence[Mapping[str, Any]],
    spectral_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Materialize the cumulative S01--S14 non-regression contract per seed."""

    architecture_by_seed = {
        (int(row["optimization_seed"]), int(row["data_seed"])): row["native"]
        for row in architecture_audits
    }
    internal_by_seed = {
        (int(row["optimization_seed"]), int(row["data_seed"])): row
        for row in internalization_audits
    }
    invariance_by_seed = {
        (int(row["optimization_seed"]), int(row["data_seed"])): row
        for row in runtime_invariance_audits
    }
    rows: list[dict[str, Any]] = []

    def append(
        seed: int,
        data_seed: int,
        invariant_id: str,
        name: str,
        value: bool | None,
        evidence: str,
    ) -> None:
        rows.append(
            {
                "optimization_seed": seed,
                "data_seed": data_seed,
                "invariant_id": invariant_id,
                "invariant": name,
                "status": (
                    "not_evaluated" if value is None else ("passed" if bool(value) else "failed")
                ),
                "passed": value,
                "evidence": evidence,
            }
        )

    for resource in resources:
        if resource.get("model") != "native_ergt":
            continue
        seed = int(resource["optimization_seed"])
        data_seed = int(resource["data_seed"])
        key = (seed, data_seed)
        architecture = architecture_by_seed[key]
        internal = internal_by_seed[key]
        invariance = invariance_by_seed[key]
        trained = int(resource.get("steps", 0)) > 0
        field_ready = bool(resource.get("independent_field_readiness_pass")) if trained else None
        stable = (
            bool(field_ready)
            and int(resource.get("stable_zero_teacher_windows", 0))
            >= int(config["checkpoint_stable_windows"])
            if trained
            else None
        )
        world_rows = [
            row
            for row in interventions
            if int(row.get("optimization_seed", -1)) == seed
            and int(row.get("data_seed", -1)) == data_seed
            and str(row.get("intervention", "")).startswith("only_world_")
            and bool(row.get("attribution_evaluable", False))
        ]
        world_pass = None
        if bool(config.get("verdict_enabled", False)):
            world_pass = len(world_rows) == 8 and all(
                float(row["full_accuracy"]) - float(row["intervention_accuracy"])
                >= float(config["intervention_drop_floor"])
                for row in world_rows
            )
        append(
            seed,
            data_seed,
            "S01",
            "registered_measurement_and_action",
            field_ready,
            "strict readiness includes measurement, action MAE, and conservation",
        )
        append(
            seed,
            data_seed,
            "S02",
            "independent_multihop_field_readiness",
            None
            if not trained
            else bool(field_ready and resource.get("native_readiness_is_multihop")),
            str(resource.get("native_readiness_hops")),
        )
        append(
            seed,
            data_seed,
            "S03",
            "consecutive_readiness_and_zero_teacher_windows",
            stable,
            f"stable_zero_teacher_windows={resource.get('stable_zero_teacher_windows')}",
        )
        append(
            seed,
            data_seed,
            "S04",
            "zero_optimizer_updates_after_freeze",
            None if not trained else int(resource.get("zero_teacher_optimizer_steps", -1)) == 0,
            f"zero_teacher_optimizer_steps={resource.get('zero_teacher_optimizer_steps')}",
        )
        append(
            seed,
            data_seed,
            "S05",
            "complete_event_chain_closure_overflow_contract",
            None
            if not trained
            else bool(field_ready and resource.get("strict_v21_chain_closure_overflow_contract")),
            "readiness event recall/precision, chain exactness, closure, and overflow",
        )
        append(
            seed,
            data_seed,
            "S06",
            "product_manifold_fixed_point_closure",
            bool(architecture.get("product_manifold_geodesic_closure"))
            and bool(architecture.get("hard_min_plus_target_solver")),
            "native architecture audit",
        )
        append(
            seed,
            data_seed,
            "S07",
            "valid_domain_multiscale_backbone",
            bool(architecture.get("multiscale_causal_backbone"))
            and bool(architecture.get("padding_independent_multiscale_support")),
            "native architecture audit",
        )
        append(
            seed,
            data_seed,
            "S08",
            "batch_and_padding_invariance",
            bool(invariance.get("contract_pass")),
            f"batch={invariance.get('batch_hard_and_mechanism_invariance_pass')} "
            f"padding={invariance.get('padding_hard_and_mechanism_invariance_pass')}",
        )
        append(
            seed,
            data_seed,
            "S09",
            "semantic_identity_factorization",
            bool(architecture.get("conserved_identity_initial_condition"))
            and bool(architecture.get("identity_derived_inside_ergt"))
            and bool(architecture.get("identity_fibre_metric_binding"))
            and bool(internal.get("identity_fibre_bitwise_equal_to_internal_reference")),
            "architecture plus raw-input identity parity",
        )
        append(
            seed,
            data_seed,
            "S10",
            "registered_local_action_cells",
            bool(architecture.get("registered_local_action_cell_before_world_residual")),
            "native architecture audit",
        )
        append(
            seed,
            data_seed,
            "S11",
            "functional_multiworld_dependence",
            world_pass,
            f"evaluable_single_world_rows={len(world_rows)}",
        )
        append(
            seed,
            data_seed,
            "S12",
            "internal_raw_input_identity_parity",
            bool(internal.get("contract_pass")),
            "bitwise public/internal forward audit",
        )
        append(
            seed,
            data_seed,
            "S13",
            "same_frozen_checkpoint_and_no_final_selection",
            None
            if not trained
            else bool(resource.get("selected_zero_teacher_checkpoint"))
            and not bool(resource.get("selection_uses_heldout_interventions", True))
            and int(resource.get("final_12_32_hop_evaluations_during_training", -1)) == 0,
            str(resource.get("selection_split")),
        )
        append(
            seed,
            data_seed,
            "S14",
            "detached_fail_open_geometry_observers",
            bool(architecture.get("curvature_is_observer_only"))
            and bool(architecture.get("spectrum_is_observer_only"))
            and bool(spectral_audit.get("fail_open"))
            and not bool(spectral_audit.get("governing_answer_path_changed")),
            str(spectral_audit.get("policy")),
        )
    return rows


def _raw_input_internalization_audit(
    model: Any,
    batch: Any,
) -> dict[str, Any]:
    """Prove parity between the public boundary and audit-only internal path."""

    reference_identity = reference_identity_fibre_tensor(
        batch.examples,
        pad_to_tokens=batch.raw_token_ids.size(1),
    ).to(batch.raw_token_ids.device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        semantic_ids, internal_identity = model.raw_input_adapter(
            batch.raw_token_ids,
            batch.base.attention_mask,
        )
        public_outputs = model(**batch.model_inputs())
        reference_outputs = model._forward_from_internal_state(
            batch.base.token_ids,
            batch.base.attention_mask,
            identity_fibre=reference_identity,
            intervention="full",
            intervention_seed=0,
            observe_spectrum=False,
        )
    model.train(was_training)
    common_tensor_keys = sorted(
        key
        for key in reference_outputs
        if key in public_outputs
        and isinstance(reference_outputs[key], torch.Tensor)
        and isinstance(public_outputs[key], torch.Tensor)
    )
    mismatched = [
        key
        for key in common_tensor_keys
        if not torch.equal(public_outputs[key], reference_outputs[key])
    ]
    model_input_keys = sorted(batch.model_inputs())
    public_contract = model_input_keys == ["attention_mask", "raw_token_ids"]
    semantic_exact = bool(torch.equal(semantic_ids, batch.base.token_ids))
    identity_exact = bool(torch.equal(internal_identity, reference_identity))
    return {
        "public_model_input_keys": model_input_keys,
        "public_raw_token_only": public_contract,
        "external_identity_sidecar_used": False,
        "semantic_ids_bitwise_equal_to_internal_reference": semantic_exact,
        "identity_fibre_bitwise_equal_to_internal_reference": identity_exact,
        "full_forward_bitwise_equal_to_internal_reference": not mismatched,
        "mismatched_forward_tensor_keys": mismatched,
        "compared_forward_tensor_count": len(common_tensor_keys),
        "identity_reference_is_audit_only": True,
        "contract_pass": public_contract and semantic_exact and identity_exact and not mismatched,
    }


def _native_runtime_invariance_audit(
    model: Any,
    examples: Sequence[Any],
    tokenizer: Any,
    *,
    device: torch.device,
    maximum_tokens: int,
) -> dict[str, Any]:
    """Compare frozen hard/mechanism outputs across batch and padding layouts."""

    if not examples:
        raise ValueError("runtime invariance audit requires at least one example")
    observed_padding = max(len(example.tokens) for example in examples)
    extended_padding = min(int(maximum_tokens), observed_padding + 16)
    if extended_padding <= observed_padding:
        extended_padding = observed_padding

    def snapshot(values: Sequence[Any], pad_to_tokens: int) -> dict[str, dict[str, torch.Tensor]]:
        batch = collate_matched_topology_examples(
            values,
            tokenizer,
            pad_to_tokens=pad_to_tokens,
        ).to(device)
        with torch.inference_mode():
            outputs = model(**batch.model_inputs())
            solution, _ = hard_solutions_from_outputs(outputs, model.config)
        records: dict[str, dict[str, torch.Tensor]] = {}
        fields = {
            "answer_ids": solution.answer_ids,
            "candidate_supported": solution.candidate_supported,
            "candidate_action": solution.candidate_action,
            "candidate_payload_mass": solution.candidate_payload_mass,
            "candidate_boundary_deficit": solution.candidate_boundary_deficit,
            "selected_action": solution.selected_action,
            "failure_code": solution.failure_code,
            "selected_event_indices": outputs["selected_event_indices"],
            "event_selection_overflow": outputs["event_selection_overflow"],
        }
        for index, example in enumerate(values):
            records[str(example.base.example_id)] = {
                name: tensor[index].detach().cpu() for name, tensor in fields.items()
            }
        return records

    def compare(
        reference: Mapping[str, Mapping[str, torch.Tensor]],
        observed: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> tuple[bool, list[str]]:
        mismatches: list[str] = []
        exact_fields = {
            "answer_ids",
            "candidate_supported",
            "failure_code",
            "selected_event_indices",
            "event_selection_overflow",
        }
        for example_id, reference_fields in reference.items():
            observed_fields = observed.get(example_id)
            if observed_fields is None:
                mismatches.append(f"{example_id}:missing")
                continue
            for name, expected in reference_fields.items():
                actual = observed_fields[name]
                equal = (
                    torch.equal(expected, actual)
                    if name in exact_fields
                    else torch.allclose(expected, actual, atol=1.0e-6, rtol=1.0e-6)
                )
                if not bool(equal):
                    mismatches.append(f"{example_id}:{name}")
        return not mismatches, mismatches

    was_training = model.training
    model.eval()
    reference = snapshot(examples, observed_padding)
    padded = snapshot(examples, extended_padding)
    single = snapshot(examples[:1], extended_padding)
    model.train(was_training)
    padding_pass, padding_mismatches = compare(reference, padded)
    batch_pass, batch_mismatches = compare(
        {str(examples[0].base.example_id): reference[str(examples[0].base.example_id)]},
        single,
    )
    return {
        "observed_pad_tokens": observed_padding,
        "extended_pad_tokens": extended_padding,
        "padding_hard_and_mechanism_invariance_pass": padding_pass,
        "padding_mismatches": padding_mismatches,
        "batch_hard_and_mechanism_invariance_pass": batch_pass,
        "batch_mismatches": batch_mismatches,
        "contract_pass": padding_pass and batch_pass,
    }


def _markdown_report(
    protocol: Mapping[str, Any],
    claim_rows: Sequence[Mapping[str, Any]],
    verdict: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
    failed_rows: Sequence[Mapping[str, Any]],
    open_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# ERGT Attention-Free Geometric Reasoning Study",
        "",
        f"- Schema: `{protocol['schema_version']}`",
        f"- Protocol: `{protocol.get('study_protocol_name', protocol['profile'])}`",
        "- Primary models: Direct Transformer Baseline and ERGT Geometric Model.",
        "- Input: the same collision-free raw serialization for both models.",
        "- Shared training: balanced 1--8 hop raw examples for both models.",
        "- ID convergence: one-hop train subset plus disjoint one-hop validation.",
        (
            "- Direct Transformer recipe: locked by a bounded development-only "
            "qualification stage; final 12--32-hop panels were not used for selection."
            if protocol.get("baseline_qualification_manifest")
            else "- Direct Transformer recipe: preregistered by the active profile."
        ),
        (
            "- Native checkpoint selection: strict independent 2--8-hop readiness, "
            "minimum deterministic exposure, immediate freeze, then two disjoint "
            "mechanism shards; final interventions remain held out."
        ),
        "- Frozen horizon comparison: 4--32 hops; the registered endpoint is 32 hops.",
        "- ERGT inference: compiler-free and free of a generic graph executor.",
        "- Curvature and spectrum: observer-only; neither changes training nor answers.",
        "",
        "## Claim Matrix",
        "",
        "| Claim | Status | Gate | Scope |",
        "|---|---|---|---|",
    ]
    for row in claim_rows:
        lines.append(f"| {row['claim']} | {row['status']} | `{row['gate']}` | {row['scope']} |")
    lines.extend(
        (
            "",
            "## Registered Endpoint",
            "",
            f"- ERGT Geometric Model: `{float(verdict['native_32_hop_accuracy_mean']):.3f}`",
            (
                "- Direct Transformer Baseline: "
                f"`{float(verdict['direct_transformer_32_hop_accuracy_mean']):.3f}`"
            ),
            (
                "- ERGT minus Direct Transformer: "
                f"`{float(verdict['native_minus_transformer_32_hop']):.3f}`"
            ),
            (f"- ERGT ID convergence: `{bool(verdict.get('native_ergt_converged_id', False))}`"),
            (
                "- Direct Transformer convergence by seed: "
                f"`{verdict.get('direct_transformer_baseline_status', 'not_evaluated')}`"
            ),
            (
                "- Observed paired 32-hop comparison: "
                f"`{verdict.get('observed_comparison_status', 'open')}`"
            ),
            (
                "- Conditional matched-convergence readout: "
                f"`{verdict.get('matched_convergence_readout', 'not_evaluated')}` "
                f"({verdict.get('matched_convergence_seed_count', 0)}/"
                f"{verdict.get('matched_convergence_required_seed_count', 3)} registered seeds)"
            ),
            "",
            "## Gate Results",
            "",
            "| Gate | Passed | Detail |",
            "|---|---:|---|",
            *(f"| {row['gate']} | {row['passed']} | {row['detail']} |" for row in gate_rows),
            "",
            "## Failed Rows",
            "",
            (
                "No failed rows."
                if not failed_rows
                else "\n".join(
                    (
                        "| Category | Gate/scenario | Reason | Value | Required |",
                        "|---|---|---|---:|---:|",
                        *(
                            f"| {row['category']} | {row['name']} | {row['reason']} | "
                            f"{row.get('value', '')} | {row.get('required', '')} |"
                            for row in failed_rows
                        ),
                    )
                )
            ),
            "",
            "## Open Scientific Rows",
            "",
            (
                "No open scientific rows."
                if not open_rows
                else "\n".join(
                    (
                        "| Category | Gate/scenario | Reason | Value | Required |",
                        "|---|---|---|---:|---:|",
                        *(
                            f"| {row['category']} | {row['name']} | {row['reason']} | "
                            f"{row.get('value', '')} | {row.get('required', '')} |"
                            for row in open_rows
                        ),
                    )
                )
            ),
            "",
            "## Final Decision",
            "",
            f"- Status: `{verdict['status']}`",
            f"- Interpretation: {verdict['interpretation']}",
            "",
            (
                "No universal Transformer superiority, language-modeling dominance, "
                "or compute-efficiency claim is made."
            ),
        )
    )
    return "\n".join(lines) + "\n"


def run_reviewer_suite(
    *,
    profile: str = "smoke",
    device: str | torch.device | None = "auto",
    output_root: str | Path | None = None,
    run_id: str | None = None,
    resume: bool = True,
    copy_outputs_to_downloads: bool = True,
    schema_version: str | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    baseline_qualification_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every registered paper panel, then issue one non-early verdict."""

    reset_spectral_observer_diagnostics()
    install_stable_event_spectral_observer()
    active_schema = str(schema_version or SCHEMA_VERSION)
    config = _load_profile(profile)
    if config_overrides:
        config.update(dict(config_overrides))
    qualified_baseline = baseline_qualification_manifest is not None
    qualified_candidate = (
        dict(baseline_qualification_manifest.get("selected_candidate") or {})
        if qualified_baseline
        else {}
    )
    qualified_recipe = dict(qualified_candidate.get("recipe") or {})
    qualified_loss_mode = str(qualified_recipe.get("loss_mode", "matched_auxiliary"))
    if len(config["seeds"]) != len(config["data_seeds"]):
        raise ValueError("optimization seeds and data seeds must have equal length")
    selected_device = _device(device)
    root_base = Path(output_root or (PACKAGE_ROOT / "runs"))
    root = root_base / (run_id or f"{profile}_v1")
    root.mkdir(parents=True, exist_ok=True)
    locked_core_audit = _locked_native_core_audit()
    if not locked_core_audit["locked_native_core_unchanged"]:
        raise RuntimeError(
            "locked native core mismatch: " + ", ".join(locked_core_audit["mismatched_files"])
        )
    protocol = {
        "schema_version": active_schema,
        "profile": profile,
        "study_name": "ERGT Attention-Free Geometric Reasoning at Long Relational Horizons",
        "study_protocol_name": config.get("study_protocol_name", profile),
        "models": ["direct_transformer", "native_ergt"],
        "model_display_names": {
            "native_ergt": "ERGT Geometric Model",
            "direct_transformer": "Direct Transformer Baseline",
        },
        "forbidden_model_arms": ["any_transformer_with_external_program_or_solver"],
        "shared_raw_input": True,
        "native_public_inputs": ["raw_token_ids", "attention_mask"],
        "external_identity_sidecar": False,
        "native_internal_path_preserved_by_bitwise_audit": True,
        "ergt43_v21_is_immutable_reference": True,
        "ergt43_v21_reference_manifest": _ergt43_v21_reference(),
        "v6_repair_design": _v6_repair_design(),
        "baseline_qualification_manifest": (
            dict(baseline_qualification_manifest)
            if baseline_qualification_manifest is not None
            else None
        ),
        "ergt43_v21_reference_contract": {
            "semantic_identity_factorization_preserved": True,
            "independent_2_8_hop_readiness_before_teacher_cut": True,
            "minimum_pair_group_exposure_before_teacher_cut": int(
                config.get("native_min_group_exposure_before_freeze", 0)
            ),
            "complete_chain_closure_overflow_readiness": True,
            "zero_teacher_optimizer_steps": 0,
            "stable_zero_teacher_windows": int(config["checkpoint_stable_windows"]),
        },
        "checkpoint_selection_split": (
            {
                "native_ergt": (
                    "independent_2_8_hop_readiness+minimum_exposure+two_frozen_6_8_mechanism_shards"
                ),
                "direct_transformer": (
                    "one_hop_eligibility+bounded_6_8_ranking"
                    if str(config.get("transformer_training_protocol", ""))
                    == "v8_staged_qualified_curriculum"
                    else "one_hop_tuning+bounded_6_8_development"
                ),
            }
            if qualified_baseline
            else "one_hop_tuning+two_frozen_mechanism_shards"
        ),
        "mechanism_tuning_is_disjoint_from_final_interventions": True,
        "mechanism_tuning_shards_are_raw_disjoint": True,
        "mechanism_tuning_controls_teacher": False,
        "field_readiness_controls_immediate_freeze": True,
        "final_interventions_are_never_used_for_checkpoint_selection": True,
        "decoupled_native_selection": [
            "one_hop_id_reporting_only",
            "independent_2_8_hop_field_readiness",
            "deterministic_epoch_coverage",
            "minimum_pair_group_exposure",
            "complete_chain_closure_overflow_contract",
            "immediate_field_freeze",
            "zero_teacher_stability",
            "six_mechanism_interventions_on_two_frozen_shards",
        ],
        "validation_is_never_used_for_selection": True,
        "training_supervision_is_label_matched": (
            qualified_loss_mode == "matched_auxiliary" if qualified_baseline else True
        ),
        "training_supervision_policy": (
            "bounded_development_qualification_then_locked_recipe"
            if qualified_baseline
            else "matched_auxiliary_supervision"
        ),
        "qualified_transformer_loss_mode": (qualified_loss_mode if qualified_baseline else None),
        "transformer_auxiliary_heads_are_training_only": True,
        "transformer_answer_path_remains_direct": True,
        "transformer_readout_uses_raw_candidate_identity_attention": True,
        "parameter_match_basis": "inference_active_parameters",
        "shared_training_hops": list(
            range(int(config["train_min_hops"]), int(config["train_max_hops"]) + 1)
        ),
        "id_calibration_hops": [int(config.get("id_hops", 1))],
        "horizon_extrapolation_hops": list(config["test_hops"]),
        "accepted_raw_duplicates_across_cohorts_policy": "forbidden",
        "pair_level_duplicate_rejection": True,
        "raw_horizon_slope_is_descriptive_not_blocking": True,
        "scenario_hop_assignment": "orthogonal_grid",
        "statistics_unit": "counterfactual_pair_nested_in_data_seed",
        "matched_convergence_rule": {
            "total_seed_pairs": len(config["seeds"]),
            "minimum_id_converged_seed_pairs": int(
                config.get("matched_convergence_minimum_seed_count", len(config["seeds"]))
            ),
            "selection_uses_final_horizon_results": False,
            "all_seed_results_retained": True,
        },
        "no_early_failure": True,
        "curvature_observer_only": True,
        "spectrum_observer_only": True,
        "spectral_observer_numerical_policy": SPECTRAL_NUMERICAL_POLICY,
        "spectral_observer_failure_is_nonblocking": True,
        "locked_native_core_audit": locked_core_audit,
        "final_execution_lock_audit": (
            dict(config["final_execution_lock_audit"])
            if config.get("final_execution_lock_audit") is not None
            else None
        ),
        "config": config,
        "source_sha256": _source_manifest(),
    }
    protocol["protocol_sha256"] = protocol_hash(protocol)
    write_json(root / "protocol.json", protocol)
    write_json(root / "environment.json", environment_record(selected_device))
    write_json(root / "locked_native_core_audit.json", locked_core_audit)
    write_json(root / "v6_repair_design.json", protocol["v6_repair_design"])
    if baseline_qualification_manifest is not None:
        write_json(
            root / "baseline_qualification_manifest.json",
            dict(baseline_qualification_manifest),
        )
    if config.get("qualification_manifest_audit") is not None:
        write_json(
            root / "qualification_manifest_audit.json",
            dict(config["qualification_manifest_audit"]),
        )
    if config.get("fresh_seed_audit") is not None:
        write_json(root / "fresh_seed_audit.json", dict(config["fresh_seed_audit"]))
    if config.get("final_execution_lock_audit") is not None:
        write_json(
            root / "final_execution_lock_audit.json",
            dict(config["final_execution_lock_audit"]),
        )

    all_curves: list[dict[str, Any]] = []
    all_resources: list[dict[str, Any]] = []
    all_cohort_metrics: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    all_interventions: list[dict[str, Any]] = []
    all_checkpoint_selection_interventions: list[dict[str, Any]] = []
    all_observers: list[dict[str, Any]] = []
    all_statistics: list[dict[str, Any]] = []
    data_manifests: dict[str, Any] = {}
    parameter_audits: list[dict[str, Any]] = []
    architecture_audits: list[dict[str, Any]] = []
    data_audits: list[dict[str, Any]] = []
    label_audits: list[dict[str, Any]] = []
    internalization_audits: list[dict[str, Any]] = []
    runtime_invariance_audits: list[dict[str, Any]] = []
    checkpoint_manifest: dict[str, Any] = {}
    raw_hashes_by_data_seed: dict[int, set[str]] = {}
    cross_seed_forbidden_raw_texts: set[str] = set()

    print(
        f"[study-progress] schema={active_schema} "
        f"protocol={config.get('study_protocol_name', profile)} "
        f"opt_seeds={config['seeds']} data_seeds={config['data_seeds']} "
        f"device={selected_device}"
    )
    for seed_value, data_seed_value in zip(config["seeds"], config["data_seeds"], strict=True):
        seed = int(seed_value)
        data_seed = int(data_seed_value)
        set_seed(seed)
        seed_root = root / f"opt_{seed}_data_{data_seed}"
        datasets = build_dataset_bundle(
            config,
            data_seed,
            forbidden_raw_texts=cross_seed_forbidden_raw_texts,
        )
        all_cohorts = (
            datasets.train,
            datasets.tuning,
            datasets.native_readiness,
            datasets.mechanism_tuning,
            datasets.validation,
            *datasets.hop_cohorts.values(),
            *datasets.context_cohorts.values(),
            *datasets.unsupported_cohorts.values(),
        )
        named_cohorts = {
            "train": datasets.train,
            "tuning": datasets.tuning,
            "native_readiness": datasets.native_readiness,
            "mechanism_tuning": datasets.mechanism_tuning,
            "validation": datasets.validation,
            **datasets.hop_cohorts,
            **datasets.context_cohorts,
            **datasets.unsupported_cohorts,
        }
        raw_hashes_by_data_seed[data_seed] = set().union(
            *(_raw_example_hashes(cohort) for cohort in all_cohorts)
        )
        training_hop_counts: dict[str, int] = defaultdict(int)
        for example in datasets.train:
            if not bool(example.base.metadata.get("unsupported_answer", False)):
                training_hop_counts[str(int(example.base.metadata["path_hops"]))] += 1
        observed_training_pair_counts = {
            hops: count // 2 for hops, count in training_hop_counts.items()
        }
        expected_training_pair_counts = {
            str(hops): int(count)
            for hops, count in dict(config.get("training_hop_pair_counts", {})).items()
        }
        training_distribution_pass = (
            not expected_training_pair_counts
            or observed_training_pair_counts == expected_training_pair_counts
        )
        id_hops = int(config.get("id_hops", 1))
        id_split_pass = all(
            int(example.base.metadata["path_hops"]) == id_hops
            for example in (*datasets.tuning, *datasets.validation)
        )
        mechanism_shards = partition_mechanism_tuning_shards(datasets.mechanism_tuning)
        readiness_hops = sorted(
            {
                int(example.base.metadata["path_hops"])
                for example in datasets.native_readiness
                if not bool(example.base.metadata.get("unsupported_answer", False))
            }
        )
        expected_readiness_hops = list(
            range(
                int(config.get("native_readiness_min_hops", 2)),
                int(config.get("native_readiness_max_hops", 8)) + 1,
            )
        )
        readiness_domain_pass = readiness_hops == expected_readiness_hops
        shard_raw_sets = [
            {example.base.raw_text for example in values} for values in mechanism_shards.values()
        ]
        mechanism_shard_disjoint_pass = all(
            left.isdisjoint(right)
            for index, left in enumerate(shard_raw_sets)
            for right in shard_raw_sets[index + 1 :]
        )
        mechanism_shard_count_pass = len(mechanism_shards) == int(
            config.get("mechanism_tuning_shard_count", 1)
        )
        data_audit = assert_shared_raw_input_contract(
            all_cohorts,
            datasets.tokenizer,
            training_examples=datasets.train,
        )
        data_audit.update(
            {
                "training_pair_counts_by_hop": observed_training_pair_counts,
                "expected_training_pair_counts_by_hop": expected_training_pair_counts,
                "training_hop_distribution_matches_config": training_distribution_pass,
                "supported_one_hop_pair_fraction": (
                    observed_training_pair_counts.get("1", 0)
                    / max(1, sum(observed_training_pair_counts.values()))
                ),
                "id_tuning_and_validation_are_one_hop": id_split_pass,
                "mechanism_tuning_shard_count": len(mechanism_shards),
                "mechanism_tuning_shards_raw_disjoint": mechanism_shard_disjoint_pass,
                "mechanism_tuning_shard_count_matches_config": (mechanism_shard_count_pass),
                "native_readiness_hops": readiness_hops,
                "native_readiness_domain_matches_config": readiness_domain_pass,
            }
        )
        data_audit["data_contract_pass"] = bool(data_audit["data_contract_pass"]) and all(
            (
                training_distribution_pass,
                id_split_pass,
                mechanism_shard_disjoint_pass,
                mechanism_shard_count_pass,
                readiness_domain_pass,
            )
        )
        data_audits.append({"optimization_seed": seed, "data_seed": data_seed, **data_audit})
        label_audit = audit_label_contract(named_cohorts)
        label_audits.append({"optimization_seed": seed, "data_seed": data_seed, **label_audit})
        manifest_key = f"opt_{seed}_data_{data_seed}"
        data_manifests[manifest_key] = {
            "train_sha256": manifest_hash(datasets.train),
            "training_supported_examples_by_hop": dict(training_hop_counts),
            "training_pair_counts_by_hop": {
                hops: count // 2 for hops, count in training_hop_counts.items()
            },
            "id_hops": id_hops,
            "tuning_sha256": manifest_hash(datasets.tuning),
            "native_readiness_sha256": manifest_hash(datasets.native_readiness),
            "native_readiness_hops": readiness_hops,
            "mechanism_tuning_sha256": manifest_hash(datasets.mechanism_tuning),
            "mechanism_tuning_shards": {
                shard: manifest_hash(values) for shard, values in mechanism_shards.items()
            },
            "validation_sha256": manifest_hash(datasets.validation),
            "hop_cohorts": {
                name: manifest_hash(values) for name, values in datasets.hop_cohorts.items()
            },
            "context_cohorts": {
                name: manifest_hash(values) for name, values in datasets.context_cohorts.items()
            },
            "unsupported_cohorts": {
                name: manifest_hash(values) for name, values in datasets.unsupported_cohorts.items()
            },
            "tokenizer": datasets.tokenizer.to_json_record(),
            "data_audit": data_audit,
            "label_audit": label_audit,
        }
        models = train_models(
            datasets.train,
            datasets.tuning,
            datasets.tokenizer,
            native_readiness=datasets.native_readiness,
            mechanism_tuning=datasets.mechanism_tuning,
            config=config,
            device=selected_device,
            seed=seed,
            data_seed=data_seed,
            run_root=seed_root,
            resume=resume,
        )
        checkpoint_manifest[manifest_key] = models.checkpoint_paths
        parameter_audits.append(
            {"optimization_seed": seed, "data_seed": data_seed, **models.parameter_audit}
        )
        architecture_audits.append(
            {
                "optimization_seed": seed,
                "data_seed": data_seed,
                "native": native_architecture_contract(models.native),
                "transformer": transformer_architecture_contract(models.transformer),
            }
        )
        audit_examples = datasets.validation[: min(2, len(datasets.validation))]
        audit_batch = collate_matched_topology_examples(
            audit_examples,
            datasets.tokenizer,
            pad_to_tokens=max(len(example.tokens) for example in audit_examples),
        ).to(selected_device)
        internalization_audits.append(
            {
                "optimization_seed": seed,
                "data_seed": data_seed,
                **_raw_input_internalization_audit(models.native, audit_batch),
            }
        )
        runtime_invariance_audits.append(
            {
                "optimization_seed": seed,
                "data_seed": data_seed,
                **_native_runtime_invariance_audit(
                    models.native,
                    audit_examples,
                    datasets.tokenizer,
                    device=selected_device,
                    maximum_tokens=int(config["max_tokens"]),
                ),
            }
        )
        all_curves.extend(
            {"optimization_seed": seed, "data_seed": data_seed, **row} for row in models.curves
        )
        all_resources.extend(
            {"optimization_seed": seed, "data_seed": data_seed, **row} for row in models.resources
        )
        selection_intervention_rows: list[dict[str, Any]] = []
        for mechanism_shard, shard_examples in mechanism_shards.items():
            shard_rows, _ = evaluate_targeted_interventions(
                models.native,
                shard_examples,
                datasets.tokenizer,
                device=selected_device,
                batch_size=int(config["eval_batch_size"]),
                max_pairwise_cells=int(config["max_pairwise_cells"]),
                max_pairs_per_scenario=int(config["mechanism_tuning_pairs_per_scenario"]),
                max_global_pairs=max(1, int(config["mechanism_tuning_pairs_per_scenario"])),
                full_accuracy_floor=float(config["mechanism_tuning_full_accuracy_floor"]),
                minimum_eligible_pairs=int(config["mechanism_tuning_minimum_eligible_pairs"]),
                include_global_interventions=False,
                include_single_world_interventions=False,
            )
            selection_intervention_rows.extend(
                {"mechanism_shard": mechanism_shard, **row} for row in shard_rows
            )
        all_checkpoint_selection_interventions.extend(
            {
                "optimization_seed": seed,
                "data_seed": data_seed,
                "selection_split": "frozen_mechanism_shard",
                **row,
            }
            for row in selection_intervention_rows
        )
        minimum_selection_drop = min(
            (float(row["targeted_drop"]) for row in selection_intervention_rows),
            default=0.0,
        )
        print(
            f"[ergt-checkpoint-audit] seed={seed} shards={len(mechanism_shards)} "
            f"mechanismRows={len(selection_intervention_rows)} "
            f"minDrop={minimum_selection_drop:.3f}"
        )

        id_hops = int(config.get("id_hops", 1))
        train_id_examples = tuple(
            example
            for example in datasets.train
            if int(example.base.metadata["path_hops"]) == id_hops
        )
        evaluation_cohorts = {
            "train_id": train_id_examples,
            "validation": datasets.validation,
            **datasets.hop_cohorts,
            **datasets.context_cohorts,
            **datasets.unsupported_cohorts,
        }
        seed_predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for cohort, examples in evaluation_cohorts.items():
            native_started = time.perf_counter()
            native_metrics, native_rows, observers = evaluate_native(
                models.native,
                examples,
                datasets.tokenizer,
                device=selected_device,
                batch_size=int(config["eval_batch_size"]),
                cohort=cohort,
                observe_spectrum=True,
                max_pairwise_cells=int(config["max_pairwise_cells"]),
            )
            native_seconds = time.perf_counter() - native_started
            transformer_started = time.perf_counter()
            transformer_metrics, transformer_rows = evaluate_transformer(
                models.transformer,
                examples,
                datasets.tokenizer,
                device=selected_device,
                batch_size=int(config["eval_batch_size"]),
                cohort=cohort,
            )
            transformer_seconds = time.perf_counter() - transformer_started
            all_cohort_metrics.extend(
                (
                    {
                        "optimization_seed": seed,
                        "data_seed": data_seed,
                        "model": "native_ergt",
                        "cohort": cohort,
                        "inference_seconds": native_seconds,
                        **native_metrics,
                    },
                    {
                        "optimization_seed": seed,
                        "data_seed": data_seed,
                        "model": "direct_transformer",
                        "cohort": cohort,
                        "inference_seconds": transformer_seconds,
                        **transformer_metrics,
                    },
                )
            )
            all_predictions.extend(
                {"optimization_seed": seed, "data_seed": data_seed, **row} for row in native_rows
            )
            all_predictions.extend(
                {"optimization_seed": seed, "data_seed": data_seed, **row}
                for row in transformer_rows
            )
            all_observers.extend(
                {"optimization_seed": seed, "data_seed": data_seed, **row} for row in observers
            )
            seed_predictions[cohort] = {
                "native": native_rows,
                "transformer": transformer_rows,
            }
            comparison = paired_comparison(
                native_rows,
                transformer_rows,
                bootstrap_samples=int(config["bootstrap_samples"]),
                seed=seed + len(cohort),
            )
            all_statistics.append(
                {
                    "optimization_seed": seed,
                    "data_seed": data_seed,
                    "cohort": cohort,
                    **comparison,
                }
            )
            print(
                f"[cohort-evaluation] seed={seed} cohort={cohort} "
                f"ergt={native_metrics['accuracy']:.3f} "
                f"transformer={transformer_metrics['accuracy']:.3f}"
            )

        mechanism_hops = tuple(
            int(hops)
            for hops in config.get("final_intervention_hops", (12, 16))
            if f"hop_{int(hops)}" in datasets.hop_cohorts
        )
        intervention_examples = (
            *datasets.validation,
            *(
                example
                for hops in mechanism_hops
                for example in datasets.hop_cohorts[f"hop_{hops}"]
            ),
        )
        intervention_rows, intervention_drops = evaluate_targeted_interventions(
            models.native,
            intervention_examples,
            datasets.tokenizer,
            device=selected_device,
            batch_size=int(config["eval_batch_size"]),
            max_pairwise_cells=int(config["max_pairwise_cells"]),
            max_pairs_per_scenario=int(config["intervention_pairs_per_scenario"]),
            max_global_pairs=int(config["intervention_global_pairs"]),
            full_accuracy_floor=float(config["intervention_full_accuracy_floor"]),
            minimum_eligible_pairs=int(config["minimum_intervention_eligible_pairs"]),
        )
        all_interventions.extend(
            {"optimization_seed": seed, "data_seed": data_seed, **row} for row in intervention_rows
        )
        all_statistics.extend(
            {
                "optimization_seed": seed,
                "data_seed": data_seed,
                "cohort": "causal_intervention",
                "metric": key,
                "value": value,
            }
            for key, value in intervention_drops.items()
        )
        del models.native, models.transformer
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()

    prediction_cohorts = sorted({str(row["cohort"]) for row in all_predictions})
    for cohort in prediction_cohorts:
        native_rows = [
            row
            for row in all_predictions
            if row["cohort"] == cohort and row["model"] == "native_ergt"
        ]
        transformer_rows = [
            row
            for row in all_predictions
            if row["cohort"] == cohort and row["model"] == "direct_transformer"
        ]
        aggregate_comparison = paired_comparison(
            native_rows,
            transformer_rows,
            bootstrap_samples=int(config["bootstrap_samples"]),
            seed=991 + len(cohort),
            replicate_key="data_seed",
        )
        all_statistics.append(
            {
                "optimization_seed": "aggregate",
                "data_seed": "hierarchical",
                "cohort": cohort,
                **aggregate_comparison,
            }
        )

    cross_seed_rows: list[dict[str, Any]] = []
    distinct_data_seeds = sorted(raw_hashes_by_data_seed)
    for left_index, left_seed in enumerate(distinct_data_seeds):
        for right_seed in distinct_data_seeds[left_index + 1 :]:
            cross_seed_rows.append(
                {
                    "left_data_seed": left_seed,
                    "right_data_seed": right_seed,
                    "exact_raw_overlap_count": len(
                        raw_hashes_by_data_seed[left_seed] & raw_hashes_by_data_seed[right_seed]
                    ),
                }
            )
    cross_seed_data_audit = {
        "data_seed_count": len(distinct_data_seeds),
        "pairwise_comparisons": cross_seed_rows,
        "all_data_seed_raw_sets_disjoint": all(
            int(row["exact_raw_overlap_count"]) == 0 for row in cross_seed_rows
        ),
        "optimization_and_data_seeds_separated": tuple(config["seeds"])
        != tuple(config["data_seeds"]),
    }
    spectral_numerical_audit = spectral_observer_diagnostics()
    write_json(root / "data_manifest.json", data_manifests)
    write_json(root / "cross_seed_data_audit.json", cross_seed_data_audit)
    write_json(root / "checkpoint_manifest.json", checkpoint_manifest)
    write_json(root / "parameter_audit.json", parameter_audits)
    write_json(root / "architecture_audit.json", architecture_audits)
    write_json(root / "shared_input_audit.json", data_audits)
    write_json(root / "label_contract_audit.json", label_audits)
    write_json(root / "raw_input_internalization_audit.json", internalization_audits)
    write_json(root / "runtime_invariance_audit.json", runtime_invariance_audits)
    write_json(root / "spectral_observer_numerical_audit.json", spectral_numerical_audit)
    write_csv(
        root / "training_curves.csv",
        all_curves,
        empty_fields=("optimization_seed", "data_seed", "model", "step"),
    )
    write_csv(root / "cohort_metrics.csv", all_cohort_metrics)
    write_csv(root / "paired_predictions.csv", all_predictions)
    write_csv(root / "causal_interventions.csv", all_interventions)
    write_csv(
        root / "checkpoint_selection_interventions.csv",
        all_checkpoint_selection_interventions,
    )
    write_csv(root / "geometry_observers.csv", all_observers)
    write_csv(root / "paired_statistics.csv", all_statistics)
    write_csv(root / "resource_disclosure.csv", all_resources)
    write_json(
        root / "native_training_exposure_audit.json",
        [
            {
                "optimization_seed": row.get("optimization_seed"),
                "data_seed": row.get("data_seed"),
                **dict(row.get("training_exposure_audit") or {}),
            }
            for row in all_resources
            if row.get("model") == "native_ergt"
        ],
    )

    verdict_enabled = bool(config["verdict_enabled"])
    source_architecture_pass = all(
        all(audit["native"].values()) and all(audit["transformer"].values())
        for audit in architecture_audits
    )
    data_protocol_pass = (
        all(bool(audit["data_contract_pass"]) for audit in data_audits)
        and bool(cross_seed_data_audit["all_data_seed_raw_sets_disjoint"])
        and bool(cross_seed_data_audit["optimization_and_data_seeds_separated"])
    )
    label_contract_pass = all(bool(audit["label_contract_pass"]) for audit in label_audits)
    internalization_pass = all(audit["contract_pass"] for audit in internalization_audits)
    runtime_invariance_pass = all(audit["contract_pass"] for audit in runtime_invariance_audits)
    architecture_pass = (
        source_architecture_pass
        and data_protocol_pass
        and internalization_pass
        and runtime_invariance_pass
    )
    native_resources = [row for row in all_resources if row["model"] == "native_ergt"]
    transformer_resources = [row for row in all_resources if row["model"] == "direct_transformer"]
    native_checkpoint_stability_pass = not verdict_enabled or all(
        bool(row["independent_field_readiness_pass"])
        and bool(row["selected_zero_teacher_checkpoint"])
        and bool(row["selected_mechanism_tuning_pass"])
        and float(row["selected_mechanism_min_targeted_drop"])
        >= float(config["mechanism_tuning_drop_floor"])
        and int(row.get("mechanism_tuning_shards_passed", 0))
        == int(config.get("mechanism_tuning_shard_count", 1))
        and int(row["stable_zero_teacher_windows"]) >= int(config["checkpoint_stable_windows"])
        and int(row["zero_teacher_optimizer_steps"]) == 0
        and not bool(row.get("mechanism_controls_teacher", True))
        and bool(row.get("teacher_stops_immediately_on_field_readiness", False))
        and bool(row.get("native_readiness_is_multihop", False))
        and bool(row.get("strict_v21_chain_closure_overflow_contract", False))
        and bool(row.get("minimum_exposure_pass", False))
        and int(row.get("unseen_group_count_before_freeze", 1)) == 0
        for row in native_resources
    )
    baseline_checkpoint_stability_pass = not verdict_enabled or all(
        bool(row["selected_tuning_checkpoint"])
        and int(row["stable_tuning_windows"]) >= int(config["checkpoint_stable_windows"])
        for row in transformer_resources
    )
    heldout_selection_pass = all(
        row.get("selection_split")
        in {
            "tuning",
            "tuning+two_frozen_mechanism_shards",
            "one_hop_tuning+bounded_6_8_development",
            "one_hop_eligibility+bounded_6_8_ranking",
            "independent_2_8_hop_readiness+two_frozen_mechanism_shards",
        }
        and not bool(row.get("selection_uses_heldout_interventions", False))
        and int(row.get("validation_evaluations_during_training", -1)) == 0
        and int(row.get("final_12_32_hop_evaluations_during_training", 0)) == 0
        for row in all_resources
    )
    parameter_match_pass = all(audit["within_ten_percent"] for audit in parameter_audits)
    baseline_qualification_pass = bool(
        not qualified_baseline
        or (
            bool(baseline_qualification_manifest.get("qualification_pass", False))
            and bool(baseline_qualification_manifest.get("selection_is_development_only", False))
            and int(baseline_qualification_manifest.get("final_12_32_hop_panels_evaluated", -1))
            == 0
            and bool(qualified_recipe)
            and qualified_loss_mode in {"answer_only", "matched_auxiliary"}
            and bool(dict(config.get("qualification_manifest_audit") or {}).get("pass", False))
        )
    )
    fresh_seed_separation_pass = bool(
        not qualified_baseline
        or bool(dict(config.get("fresh_seed_audit") or {}).get("pass", False))
    )
    final_execution_lock_pass = bool(
        not qualified_baseline
        or bool(dict(config.get("final_execution_lock_audit") or {}).get("pass", False))
    )
    matched_supervision_pass = (
        label_contract_pass
        and all(bool(audit["matched_training_label_families"]) for audit in parameter_audits)
        and all(
            not bool(audit["transformer_auxiliary_outputs_used_at_inference"])
            and not bool(audit["transformer_compiler_or_executor_used"])
            for audit in parameter_audits
        )
        and all(
            str(row.get("training_label_families", "")).startswith("answer,entity")
            and not bool(row.get("auxiliary_outputs_used_at_inference", True))
            and not bool(row.get("compiler_or_executor_used", True))
            for row in transformer_resources
        )
    )
    qualified_supervision_disclosure_pass = (
        label_contract_pass
        and baseline_qualification_pass
        and all(
            str(audit.get("transformer_loss_mode")) == qualified_loss_mode
            and not bool(audit["transformer_auxiliary_outputs_used_at_inference"])
            and not bool(audit["transformer_compiler_or_executor_used"])
            for audit in parameter_audits
        )
        and all(
            str(row.get("loss_mode")) == qualified_loss_mode
            and str(row.get("training_label_families", "")).startswith("answer")
            and not bool(row.get("auxiliary_outputs_used_at_inference", True))
            and not bool(row.get("compiler_or_executor_used", True))
            for row in transformer_resources
        )
    )
    supervision_pass = (
        qualified_supervision_disclosure_pass if qualified_baseline else matched_supervision_pass
    )
    supervision_gate_name = (
        "qualified_baseline_and_disclosed_training_supervision"
        if qualified_baseline
        else "label_contract_and_matched_training_supervision"
    )
    validation_rows = [row for row in all_cohort_metrics if row["cohort"] == "validation"]
    train_id_rows = [row for row in all_cohort_metrics if row["cohort"] == "train_id"]
    native_validation = [row for row in validation_rows if row["model"] == "native_ergt"]
    transformer_validation = [
        row for row in validation_rows if row["model"] == "direct_transformer"
    ]
    native_train_id = [row for row in train_id_rows if row["model"] == "native_ergt"]
    transformer_train_id = [row for row in train_id_rows if row["model"] == "direct_transformer"]
    native_convergence_pass = min(
        float(row["supported_accuracy"]) for row in native_train_id
    ) >= float(config["convergence_accuracy_floor"]) and min(
        float(row["supported_accuracy"]) for row in native_validation
    ) >= float(config["convergence_accuracy_floor"])
    baseline_convergence_pass = min(
        float(row["supported_accuracy"]) for row in transformer_train_id
    ) >= float(config["convergence_accuracy_floor"]) and min(
        float(row["supported_accuracy"]) for row in transformer_validation
    ) >= float(config["convergence_accuracy_floor"])
    both_models_converged_id = native_convergence_pass and baseline_convergence_pass
    convergence_by_seed: list[dict[str, Any]] = []
    for seed_value, data_seed_value in zip(config["seeds"], config["data_seeds"], strict=True):
        seed = int(seed_value)
        data_seed = int(data_seed_value)
        row: dict[str, Any] = {"optimization_seed": seed, "data_seed": data_seed}
        for model_name, train_rows, validation_model_rows in (
            ("native_ergt", native_train_id, native_validation),
            ("direct_transformer", transformer_train_id, transformer_validation),
        ):
            train_row = next(item for item in train_rows if int(item["optimization_seed"]) == seed)
            validation_row = next(
                item for item in validation_model_rows if int(item["optimization_seed"]) == seed
            )
            train_accuracy = float(train_row["supported_accuracy"])
            validation_accuracy = float(validation_row["supported_accuracy"])
            row[f"{model_name}_train_id_accuracy"] = train_accuracy
            row[f"{model_name}_validation_id_accuracy"] = validation_accuracy
            row[f"{model_name}_id_converged"] = train_accuracy >= float(
                config["convergence_accuracy_floor"]
            ) and validation_accuracy >= float(config["convergence_accuracy_floor"])
        row["both_models_id_converged"] = bool(
            row["native_ergt_id_converged"] and row["direct_transformer_id_converged"]
        )
        convergence_by_seed.append(row)
    baseline_nonconverged_seeds = [
        int(row["optimization_seed"])
        for row in convergence_by_seed
        if not bool(row["direct_transformer_id_converged"])
    ]
    baseline_converged_seeds = [
        int(row["optimization_seed"])
        for row in convergence_by_seed
        if bool(row["direct_transformer_id_converged"])
    ]
    matched_converged_seeds = [
        int(row["optimization_seed"])
        for row in convergence_by_seed
        if bool(row["both_models_id_converged"])
    ]
    minimum_matched_seed_count = int(
        config.get("matched_convergence_minimum_seed_count", len(config["seeds"]))
    )
    matched_convergence_seed_floor_pass = (
        len(matched_converged_seeds) >= minimum_matched_seed_count
    )
    max_hop = max(int(value) for value in config["test_hops"])
    long_rows = [row for row in all_cohort_metrics if row["cohort"] == f"hop_{max_hop}"]
    native_long = [row for row in long_rows if row["model"] == "native_ergt"]
    transformer_long = [row for row in long_rows if row["model"] == "direct_transformer"]
    native_long_mean = _mean(native_long, "accuracy")
    transformer_long_mean = _mean(transformer_long, "accuracy")
    long_horizon_statistics = next(
        row
        for row in all_statistics
        if row.get("optimization_seed") == "aggregate" and row.get("cohort") == f"hop_{max_hop}"
    )
    observed_paired_advantage_pass = (
        native_long_mean > transformer_long_mean
        and float(long_horizon_statistics["bootstrap_ci_low"]) > 0.0
        and float(long_horizon_statistics["mcnemar_exact_pvalue"]) < 0.05
    )
    native_long_horizon_floor_pass = native_long_mean >= float(
        config["long_horizon_accuracy_floor"]
    )
    registered_margin_pass = native_long_mean - transformer_long_mean >= float(
        config["long_horizon_margin_floor"]
    )
    registered_long_horizon_pass = (
        observed_paired_advantage_pass and native_long_horizon_floor_pass and registered_margin_pass
    )
    matched_seed_set = set(matched_converged_seeds)
    matched_native_long = [
        row for row in native_long if int(row["optimization_seed"]) in matched_seed_set
    ]
    matched_transformer_long = [
        row for row in transformer_long if int(row["optimization_seed"]) in matched_seed_set
    ]
    matched_native_predictions = [
        row
        for row in all_predictions
        if row["cohort"] == f"hop_{max_hop}"
        and row["model"] == "native_ergt"
        and int(row["optimization_seed"]) in matched_seed_set
    ]
    matched_transformer_predictions = [
        row
        for row in all_predictions
        if row["cohort"] == f"hop_{max_hop}"
        and row["model"] == "direct_transformer"
        and int(row["optimization_seed"]) in matched_seed_set
    ]
    matched_long_horizon_statistics = (
        paired_comparison(
            matched_native_predictions,
            matched_transformer_predictions,
            bootstrap_samples=int(config["bootstrap_samples"]),
            seed=1991 + max_hop,
            replicate_key="data_seed",
        )
        if matched_converged_seeds
        else {
            "bootstrap_ci_low": 0.0,
            "bootstrap_ci_high": 0.0,
            "mcnemar_exact_pvalue": 1.0,
        }
    )
    all_statistics.append(
        {
            "optimization_seed": "matched_convergence_subset",
            "data_seed": "hierarchical",
            "cohort": f"hop_{max_hop}",
            "matched_convergence_preregistered": True,
            "matched_converged_seed_count": len(matched_converged_seeds),
            "matched_converged_optimization_seeds": json.dumps(matched_converged_seeds),
            **matched_long_horizon_statistics,
        }
    )
    write_csv(root / "paired_statistics.csv", all_statistics)
    matched_native_long_mean = _mean(matched_native_long, "accuracy")
    matched_transformer_long_mean = _mean(matched_transformer_long, "accuracy")
    matched_convergence_long_horizon_pass = (
        matched_convergence_seed_floor_pass
        and matched_native_long_mean >= float(config["long_horizon_accuracy_floor"])
        and matched_native_long_mean - matched_transformer_long_mean
        >= float(config["long_horizon_margin_floor"])
        and matched_native_long_mean > matched_transformer_long_mean
        and float(matched_long_horizon_statistics["bootstrap_ci_low"]) > 0.0
        and float(matched_long_horizon_statistics["mcnemar_exact_pvalue"]) < 0.05
    )
    hop_native_points: list[tuple[float, float]] = []
    hop_transformer_points: list[tuple[float, float]] = []
    for hops in tuple(int(value) for value in config["test_hops"]):
        rows = [row for row in all_cohort_metrics if row["cohort"] == f"hop_{hops}"]
        hop_native_points.append(
            (hops, _mean([row for row in rows if row["model"] == "native_ergt"], "accuracy"))
        )
        hop_transformer_points.append(
            (
                hops,
                _mean(
                    [row for row in rows if row["model"] == "direct_transformer"],
                    "accuracy",
                ),
            )
        )
    native_slope = linear_slope(hop_native_points)
    transformer_slope = linear_slope(hop_transformer_points)
    slower_degradation_pass = native_slope >= transformer_slope

    targeted = [
        row
        for row in all_interventions
        if row["scenario"] != "all" and not str(row["intervention"]).startswith("only_world_")
    ]
    targeted_by_key: dict[tuple[int, int, str], float] = {
        (
            int(row["optimization_seed"]),
            int(row["data_seed"]),
            str(row["scenario"]),
        ): float(row["targeted_drop"])
        for row in targeted
        if bool(row.get("attribution_evaluable", False))
    }
    causal_mechanism_pass = (
        bool(targeted)
        and len(targeted_by_key) == len(targeted)
        and all(
            value >= float(config["intervention_drop_floor"]) for value in targeted_by_key.values()
        )
    )
    checkpoint_targeted = list(all_checkpoint_selection_interventions)
    checkpoint_targeted_by_key = {
        (
            int(row["optimization_seed"]),
            int(row["data_seed"]),
            str(row["mechanism_shard"]),
            str(row["scenario"]),
        ): float(row["targeted_drop"])
        for row in checkpoint_targeted
        if bool(row.get("attribution_evaluable", False))
    }
    expected_checkpoint_targeted = (
        len(config["seeds"])
        * int(config.get("mechanism_tuning_shard_count", 1))
        * len(TARGETED_MECHANISM_INTERVENTIONS)
    )
    conjunctive_selection_pass = (
        len(checkpoint_targeted) == expected_checkpoint_targeted
        and len(checkpoint_targeted_by_key) == expected_checkpoint_targeted
        and all(
            value >= float(config["mechanism_tuning_drop_floor"])
            for value in checkpoint_targeted_by_key.values()
        )
    )
    world_rows = [
        row for row in all_interventions if str(row["intervention"]).startswith("only_world_")
    ]
    full_by_seed = {
        (int(row["optimization_seed"]), int(row["data_seed"])): float(row["full_accuracy"])
        for row in world_rows
        if bool(row.get("attribution_evaluable", False))
    }
    best_world_by_seed: dict[tuple[int, int], float] = defaultdict(float)
    for row in world_rows:
        if not bool(row.get("attribution_evaluable", False)):
            continue
        seed = (int(row["optimization_seed"]), int(row["data_seed"]))
        best_world_by_seed[seed] = max(
            best_world_by_seed[seed], float(row["intervention_accuracy"])
        )
    expected_world_rows = len(config["seeds"]) * 8
    multiworld_pass = (
        len(world_rows) == expected_world_rows
        and len(full_by_seed) == len(config["seeds"])
        and all(
            full_by_seed[seed] - value >= float(config["intervention_drop_floor"])
            for seed, value in best_world_by_seed.items()
        )
    )
    unsupported_rows = [
        row
        for row in all_cohort_metrics
        if str(row["cohort"]).startswith("unsupported_") and row["model"] == "native_ergt"
    ]
    unsupported_pass = bool(unsupported_rows) and max(
        float(row["unsupported_error_rate"]) for row in unsupported_rows
    ) <= float(config["unsupported_error_ceiling"])
    stability_invariant_rows = _native_stability_invariant_rows(
        config=config,
        resources=all_resources,
        architecture_audits=architecture_audits,
        internalization_audits=internalization_audits,
        runtime_invariance_audits=runtime_invariance_audits,
        interventions=all_interventions,
        spectral_audit=spectral_numerical_audit,
    )
    evaluated_stability_rows = [
        row for row in stability_invariant_rows if row["status"] != "not_evaluated"
    ]
    native_stability_catalog_pass = not verdict_enabled or (
        len(evaluated_stability_rows) == len(stability_invariant_rows)
        and all(bool(row["passed"]) for row in evaluated_stability_rows)
    )
    native_claim_bundle_pass = (
        native_checkpoint_stability_pass
        and native_stability_catalog_pass
        and native_convergence_pass
        and native_long_horizon_floor_pass
        and causal_mechanism_pass
        and multiworld_pass
        and unsupported_pass
    )
    gate_values: dict[str, bool | None] = {
        "locked_native_core_unchanged": bool(locked_core_audit["locked_native_core_unchanged"]),
        "architecture_and_shared_input": architecture_pass,
        "data_protocol_integrity": data_protocol_pass,
        **(
            {
                "locked_baseline_qualification_manifest": baseline_qualification_pass,
                "immutable_final_execution_lock": final_execution_lock_pass,
                "fresh_confirmatory_seed_separation": fresh_seed_separation_pass,
            }
            if qualified_baseline
            else {}
        ),
        supervision_gate_name: supervision_pass,
        "native_multihop_exposure_locked_checkpoint": native_checkpoint_stability_pass,
        "native_stability_capability_catalog": native_stability_catalog_pass,
        "direct_transformer_checkpoint_selection": baseline_checkpoint_stability_pass,
        "decoupled_frozen_mechanism_checkpoint_selection": (
            conjunctive_selection_pass if verdict_enabled else None
        ),
        "heldout_validation_not_used_for_selection": heldout_selection_pass,
        "parameter_count_match": parameter_match_pass,
        "native_ergt_converged_id": native_convergence_pass if verdict_enabled else None,
        "native_ergt_absolute_32_hop_floor": (
            native_long_horizon_floor_pass if verdict_enabled else None
        ),
        "observed_paired_32_hop_advantage": (
            observed_paired_advantage_pass if verdict_enabled else None
        ),
        "registered_large_32_hop_margin": (registered_margin_pass if verdict_enabled else None),
        "registered_causal_interventions": causal_mechanism_pass if verdict_enabled else None,
        "functional_multiworld_dependence": multiworld_pass if verdict_enabled else None,
        "unsupported_answer_rejection": unsupported_pass if verdict_enabled else None,
        "native_ergt_claim_bundle": native_claim_bundle_pass if verdict_enabled else None,
    }
    claim_rows = [
        _claim_row(
            "attention_free_native_geometric_answer_path",
            gate_values["architecture_and_shared_input"],
            "architecture_and_shared_input",
            "Static source and forward-contract audit.",
        ),
        _claim_row(
            "native_ergt_converges_in_distribution",
            gate_values["native_ergt_converged_id"],
            "native_ergt_converged_id",
            "ERGT one-hop train subset and disjoint one-hop validation; baseline-independent.",
        ),
        _claim_row(
            (
                "qualified_direct_transformer_baseline_with_disclosed_supervision"
                if qualified_baseline
                else "balanced_labels_and_matched_training_supervision"
            ),
            gate_values[supervision_gate_name],
            supervision_gate_name,
            (
                "A bounded development-only search selected the direct Transformer "
                "recipe; no labels, compiler, executor, or geometry enter inference."
                if qualified_baseline
                else ("Same training-label families; no labels or auxiliary heads enter inference.")
            ),
        ),
        {
            "claim": "strengthened_direct_transformer_convergence_by_seed",
            "status": "reported",
            "gate": "per_seed_convergence_readout",
            "scope": (
                "Every seed and both one-hop ID accuracies are reported numerically; "
                "this readout does not determine run status or ERGT convergence."
            ),
        },
        _claim_row(
            "multihop_exposure_locked_field_freeze_and_mechanism_selection",
            gate_values["native_multihop_exposure_locked_checkpoint"],
            "native_multihop_exposure_locked_checkpoint",
            (
                "Two strict 2--8-hop readiness windows plus minimum exposure precede "
                "freeze; no optimizer update follows it."
            ),
        ),
        _claim_row(
            "cumulative_native_stability_non_regression",
            gate_values["native_stability_capability_catalog"],
            "native_stability_capability_catalog",
            "Every registered S01--S14 invariant is explicitly evaluated per seed.",
        ),
        _claim_row(
            "native_ergt_absolute_long_horizon_accuracy_to_32_hops",
            gate_values["native_ergt_absolute_32_hop_floor"],
            "native_ergt_absolute_32_hop_floor",
            "Absolute ERGT endpoint; independent of baseline convergence.",
        ),
        _claim_row(
            "observed_paired_advantage_over_strengthened_direct_transformer",
            gate_values["observed_paired_32_hop_advantage"],
            "observed_paired_32_hop_advantage",
            "Paired effect, hierarchical interval, and exact paired test as observed.",
        ),
        {
            "claim": "conditional_matched_convergence_comparison",
            "status": "reported",
            "gate": "conditional_matched_convergence_readout",
            "scope": (
                "The ID-qualified seed count and its frozen 32-hop endpoint are reported "
                "numerically. The registered three-seed criterion controls manuscript "
                "wording only and never marks the notebook run as failed."
            ),
        },
        _claim_row(
            "causal_dependence_on_registered_geometric_mechanisms",
            gate_values["registered_causal_interventions"],
            "registered_causal_interventions",
            "Same-checkpoint action/cone/transport/boundary/terminal/memory interventions.",
        ),
        _claim_row(
            "functional_multiworld_dependence",
            gate_values["functional_multiworld_dependence"],
            "functional_multiworld_dependence",
            "No claim of uniform or independent semantic specialization.",
        ),
        _claim_row(
            "bounded_unsupported_answer_rejection",
            gate_values["unsupported_answer_rejection"],
            "unsupported_answer_rejection",
            "Controlled unsupported candidates, not open-domain hallucination.",
        ),
        _claim_row(
            "native_ergt_bounded_claim_bundle",
            gate_values["native_ergt_claim_bundle"],
            "native_ergt_claim_bundle",
            "Native convergence, absolute 32-hop floor, mechanisms, worlds, and rejection.",
        ),
        _claim_row(
            "curvature_and_spectrum_observers",
            True,
            "observer_only_source_contract",
            "Descriptive only; no causal claim.",
        ),
        {
            "claim": "universal_transformer_superiority",
            "status": "not_claimed",
            "gate": "out_of_scope",
            "scope": "No universal comparison is asserted.",
        },
        {
            "claim": "language_modeling_or_perplexity_dominance",
            "status": "not_claimed",
            "gate": "out_of_scope",
            "scope": "This package is not a language-model benchmark.",
        },
        {
            "claim": "compute_efficiency_superiority",
            "status": "not_claimed",
            "gate": "resource_disclosure_only",
            "scope": "Runtime and memory are disclosed without an efficiency claim.",
        },
    ]
    integrity_gate_names = {
        "locked_native_core_unchanged",
        "architecture_and_shared_input",
        "data_protocol_integrity",
        supervision_gate_name,
        "heldout_validation_not_used_for_selection",
        "parameter_count_match",
        *(
            {
                "locked_baseline_qualification_manifest",
                "immutable_final_execution_lock",
                "fresh_confirmatory_seed_separation",
            }
            if qualified_baseline
            else set()
        ),
    }
    integrity_pass = all(bool(gate_values[name]) for name in integrity_gate_names)
    status = _v9_run_status(
        verdict_enabled=verdict_enabled,
        integrity_pass=integrity_pass,
        native_claim_bundle_pass=native_claim_bundle_pass,
        observed_paired_advantage_pass=observed_paired_advantage_pass,
    )
    verdict = {
        "schema_version": active_schema,
        "status": status,
        "profile": profile,
        "gates": gate_values,
        "native_32_hop_accuracy_mean": native_long_mean,
        "direct_transformer_32_hop_accuracy_mean": transformer_long_mean,
        "native_minus_transformer_32_hop": native_long_mean - transformer_long_mean,
        "native_minus_transformer_32_hop_bootstrap_ci": [
            float(long_horizon_statistics["bootstrap_ci_low"]),
            float(long_horizon_statistics["bootstrap_ci_high"]),
        ],
        "native_vs_transformer_32_hop_mcnemar_pvalue": float(
            long_horizon_statistics["mcnemar_exact_pvalue"]
        ),
        "execution_integrity_pass": integrity_pass,
        "native_ergt_claim_status": ("supported" if native_claim_bundle_pass else "open"),
        "native_stability_capability_catalog_pass": native_stability_catalog_pass,
        "observed_comparison_status": ("supported" if observed_paired_advantage_pass else "open"),
        "direct_transformer_baseline_status": (
            "converged_all_seeds"
            if baseline_convergence_pass
            else f"converged_{len(baseline_converged_seeds)}_of_{len(config['seeds'])}_seeds"
        ),
        "direct_transformer_convergence_readout": (
            f"{len(baseline_converged_seeds)}_of_{len(config['seeds'])}_seeds_met_id_criterion"
        ),
        "matched_convergence_readout": (
            "registered_conditional_endpoint_supported"
            if matched_convergence_long_horizon_pass
            else (
                "conditional_endpoint_reported_below_registered_seed_count"
                if not matched_convergence_seed_floor_pass
                else "conditional_endpoint_reported_without_registered_advantage"
            )
        ),
        "matched_convergence_claim_status": (
            "supported" if matched_convergence_long_horizon_pass else "reported"
        ),
        "raw_32_hop_advantage_observation_pass": observed_paired_advantage_pass,
        "native_32_hop_accuracy_floor_pass": native_long_horizon_floor_pass,
        "registered_32_hop_margin_pass": registered_margin_pass,
        "post_convergence_32_hop_advantage_pass": matched_convergence_long_horizon_pass,
        "native_ergt_converged_id": native_convergence_pass,
        "direct_transformer_baseline_converged_id": baseline_convergence_pass,
        "both_models_converged_id": both_models_converged_id,
        "matched_convergence_seed_count": len(matched_converged_seeds),
        "matched_convergence_required_seed_count": minimum_matched_seed_count,
        "matched_converged_optimization_seeds": matched_converged_seeds,
        "baseline_converged_optimization_seeds": baseline_converged_seeds,
        "baseline_nonconverged_optimization_seeds": baseline_nonconverged_seeds,
        "convergence_by_seed": convergence_by_seed,
        "matched_convergence_32_hop": {
            "native_accuracy_mean": matched_native_long_mean,
            "direct_transformer_accuracy_mean": matched_transformer_long_mean,
            "native_minus_transformer": (
                matched_native_long_mean - matched_transformer_long_mean
            ),
            "bootstrap_ci": [
                float(matched_long_horizon_statistics["bootstrap_ci_low"]),
                float(matched_long_horizon_statistics["bootstrap_ci_high"]),
            ],
            "mcnemar_pvalue": float(
                matched_long_horizon_statistics["mcnemar_exact_pvalue"]
            ),
            "selection_rule": (
                "Both models clear the preregistered one-hop ID floor; final-horizon "
                "performance is not used to include or exclude a seed."
            ),
        },
        "baseline_qualification_manifest_sha256": (
            baseline_qualification_manifest.get("manifest_sha256") if qualified_baseline else None
        ),
        "qualified_transformer_recipe": qualified_recipe if qualified_baseline else None,
        "minimum_final_targeted_intervention_drop": min(targeted_by_key.values(), default=0.0),
        "minimum_checkpoint_selection_intervention_drop": min(
            checkpoint_targeted_by_key.values(), default=0.0
        ),
        "registered_thresholds": {
            "id_convergence_accuracy_floor": float(config["convergence_accuracy_floor"]),
            "matched_convergence_minimum_seed_count": minimum_matched_seed_count,
            "native_32_hop_accuracy_floor": float(config["long_horizon_accuracy_floor"]),
            "native_minus_transformer_32_hop_floor": float(config["long_horizon_margin_floor"]),
            "final_intervention_drop_floor": float(config["intervention_drop_floor"]),
            "mechanism_tuning_drop_floor": float(config["mechanism_tuning_drop_floor"]),
        },
        "native_horizon_accuracy_slope": native_slope,
        "direct_transformer_horizon_accuracy_slope": transformer_slope,
        "horizon_slope_diagnostic": {
            "native_slope": native_slope,
            "direct_transformer_slope": transformer_slope,
            "native_slope_not_lower": slower_degradation_pass,
            "blocking_gate": False,
            "reason": (
                "Raw linear slopes are descriptive because ceiling and floor effects "
                "can reverse their ordering without reversing endpoint accuracy."
            ),
        },
        "spectral_observer_numerical_audit": spectral_numerical_audit,
        "locked_native_core_audit": locked_core_audit,
        "final_execution_lock_audit": (
            dict(config.get("final_execution_lock_audit") or {}) if qualified_baseline else None
        ),
        "interpretation": (
            (
                "The four-seed study reports ERGT convergence, direct-Transformer "
                "convergence by seed, the all-seed paired comparison, and the "
                "preregistered matched-convergence subset separately. A baseline miss "
                "is retained as a result and never converts an ERGT result or completed "
                "all-seed comparison into a failed execution."
            )
            if qualified_baseline
            else (
                "The run completed every registered panel. The verdict is bounded to "
                "the matched controlled relational task family and does not assert "
                "universal, language-modeling, or efficiency dominance."
            )
        ),
    }
    gate_details = {
        "locked_native_core_unchanged": "Immutable V21-derived native-core SHA-256 audit.",
        "architecture_and_shared_input": (
            "Direct Transformer and native ERGT receive the same raw serialization."
        ),
        "data_protocol_integrity": "No train/evaluation or cross-seed raw overlap.",
        "label_contract_and_matched_training_supervision": (
            "Balanced labels and matched training-only label families."
        ),
        "locked_baseline_qualification_manifest": (
            "The direct Transformer recipe was selected by the bounded development-only "
            "qualification stage without evaluating final 12--32-hop panels."
        ),
        "immutable_final_execution_lock": (
            "The V21/V6 native path, final config, evaluator, data path, selected "
            "baseline recipe, and checkpoint policy match their registered SHA-256 lock."
        ),
        "fresh_confirmatory_seed_separation": (
            "Optimization and data seeds are disjoint from qualification and all V3--V6 "
            "development seeds."
        ),
        "qualified_baseline_and_disclosed_training_supervision": (
            "The selected direct Transformer uses the locked qualified loss mode, raw "
            "tokens only, and no compiler, executor, or ERGT geometry."
        ),
        "native_multihop_exposure_locked_checkpoint": (
            "Strict independent 2--8-hop readiness, complete V21 chain/closure/overflow "
            "checks, deterministic coverage, and minimum exposure precede native freeze."
        ),
        "native_stability_capability_catalog": (
            "All cumulative S01--S14 architecture, training, invariance, mechanism, "
            "multiworld, identity, and observer invariants are evaluated per seed."
        ),
        "direct_transformer_checkpoint_selection": (
            "The strengthened direct Transformer checkpoint is selected without final panels; "
            "a miss is reported for that baseline arm and is not an ERGT veto."
        ),
        "decoupled_frozen_mechanism_checkpoint_selection": (
            "All six mechanisms pass independently on both frozen selection shards."
        ),
        "heldout_validation_not_used_for_selection": (
            "Final validation and causal panels are evaluation-only."
        ),
        "parameter_count_match": "Inference-active parameters differ by at most ten percent.",
        "native_ergt_converged_id": (
            "ERGT clears its registered train/validation ID floor on every seed."
        ),
        "native_ergt_absolute_32_hop_floor": (
            "ERGT independently clears the registered absolute 32-hop accuracy floor."
        ),
        "observed_paired_32_hop_advantage": (
            "Observed paired difference is positive, its hierarchical interval excludes zero, "
            "and the exact paired test is significant."
        ),
        "registered_large_32_hop_margin": (
            "The observed mean difference clears the preregistered large-effect margin."
        ),
        "native_ergt_32_hop_advantage": (
            "Accuracy, margin, confidence interval, and paired test all clear their gates."
        ),
        "post_convergence_native_ergt_32_hop_advantage": (
            "Both models first converge on ID; then accuracy, margin, confidence interval, "
            "and paired test clear their 32-hop gates."
        ),
        "registered_causal_interventions": (
            "Every held-out scenario is evaluable and clears the targeted-drop floor."
        ),
        "functional_multiworld_dependence": "No single world retains the complete program.",
        "unsupported_answer_rejection": (
            "Native unsupported error remains below the registered ceiling."
        ),
        "v3_v4_non_regression_contract": (
            "Convergence, 32-hop advantage, mechanisms, multiworld dependence, and "
            "rejection pass together."
        ),
        "native_ergt_claim_bundle": (
            "Native checkpoint stability, ERGT convergence, absolute 32-hop accuracy, "
            "mechanisms, multiworld dependence, and rejection pass together."
        ),
    }
    gate_rows = [
        {
            "gate": gate,
            "passed": value,
            "blocking_for_execution_integrity": gate in integrity_gate_names,
            "detail": gate_details.get(gate, "Registered confirmatory gate."),
        }
        for gate, value in gate_values.items()
    ]
    open_rows: list[dict[str, Any]] = [
        {
            "category": (
                "integrity_failure" if gate in integrity_gate_names else "open_scientific_gate"
            ),
            "name": gate,
            "reason": gate_details.get(gate, "registered gate failed"),
            "value": False,
            "required": gate in integrity_gate_names,
        }
        for gate, value in gate_values.items()
        if value is False
    ]
    failed_rows: list[dict[str, Any]] = [
        row for row in open_rows if row["category"] == "integrity_failure"
    ]
    if verdict_enabled:
        for category, rows, floor in (
            (
                "checkpoint_selection_intervention",
                checkpoint_targeted,
                float(config["mechanism_tuning_drop_floor"]),
            ),
            (
                "heldout_causal_intervention",
                targeted,
                float(config["intervention_drop_floor"]),
            ),
        ):
            for row in rows:
                evaluable = bool(row.get("attribution_evaluable", False))
                drop = float(row.get("targeted_drop", 0.0))
                if evaluable and drop >= floor:
                    continue
                open_rows.append(
                    {
                        "category": category,
                        "name": (
                            f"opt={row.get('optimization_seed')}/data={row.get('data_seed')}/"
                            f"{row.get('mechanism_shard', 'final')}/"
                            f"{row.get('scenario')}/{row.get('intervention')}"
                        ),
                        "reason": (
                            "attribution_not_evaluable"
                            if not evaluable
                            else "targeted_drop_below_floor"
                        ),
                        "value": drop,
                        "required": floor,
                        "panel_full_accuracy": row.get("panel_full_accuracy"),
                        "eligible_pairs": row.get("eligible_pairs"),
                    }
                )
    write_csv(root / "gate_results.csv", gate_rows)
    write_csv(
        root / "failed_rows.csv",
        failed_rows,
        empty_fields=("category", "name", "reason", "value", "required"),
    )
    write_csv(
        root / "open_scientific_rows.csv",
        open_rows,
        empty_fields=("category", "name", "reason", "value", "required"),
    )
    write_json(root / "gate_details.json", gate_rows)
    write_csv(root / "claim_matrix.csv", claim_rows)
    write_json(root / "claim_matrix.json", claim_rows)
    write_csv(root / "native_stability_invariants.csv", stability_invariant_rows)
    write_json(root / "native_stability_invariants.json", stability_invariant_rows)
    write_json(root / "final_verdict.json", verdict)
    compact_table_paths = write_compact_tables(
        root,
        claim_rows=claim_rows,
        gate_rows=gate_rows,
        verdict=verdict,
        resources=all_resources,
        cohort_metrics=all_cohort_metrics,
        interventions=all_interventions,
        checkpoint_interventions=all_checkpoint_selection_interventions,
        observers=all_observers,
        statistics=all_statistics,
        data_audits=data_audits,
        internalization_audits=internalization_audits,
        runtime_invariance_audits=runtime_invariance_audits,
        stability_invariant_rows=stability_invariant_rows,
    )
    report_path = root / "paper_readout.md"
    report_path.write_text(
        _markdown_report(protocol, claim_rows, verdict, gate_rows, failed_rows, open_rows),
        encoding="utf-8",
    )
    table_paths = write_latex_tables(root, all_cohort_metrics, claim_rows)
    figure_paths = write_figures(
        root,
        all_cohort_metrics,
        _publication_figure_curves(all_curves),
        all_interventions,
    )
    artifact_paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name
        not in {
            "ergt_reviewer_evidence_bundle.zip",
            "ergt_geometric_study_evidence_bundle.zip",
        }
    ]
    bundle_path = bundle(root, artifact_paths)
    public_bundle_path = root / "ergt_geometric_study_evidence_bundle.zip"
    bundle_path.replace(public_bundle_path)
    bundle_path = public_bundle_path
    download_artifacts = [
        root / "final_verdict.json",
        root / "claim_matrix.csv",
        root / "gate_results.csv",
        root / "failed_rows.csv",
        root / "open_scientific_rows.csv",
        root / "cohort_metrics.csv",
        root / "causal_interventions.csv",
        root / "checkpoint_selection_interventions.csv",
        root / "paired_statistics.csv",
        root / "training_curves.csv",
        root / "shared_input_audit.json",
        root / "label_contract_audit.json",
        root / "cross_seed_data_audit.json",
        root / "raw_input_internalization_audit.json",
        root / "runtime_invariance_audit.json",
        root / "spectral_observer_numerical_audit.json",
        root / "locked_native_core_audit.json",
        *((root / "final_execution_lock_audit.json",) if qualified_baseline else ()),
        root / "v6_repair_design.json",
        root / "resource_disclosure.csv",
        root / "native_training_exposure_audit.json",
        root / "native_stability_invariants.csv",
        root / "native_stability_invariants.json",
        report_path,
        bundle_path,
        *table_paths,
        *figure_paths,
        *compact_table_paths,
    ]
    if qualified_baseline:
        download_artifacts.extend(
            (
                root / "baseline_qualification_manifest.json",
                root / "qualification_manifest_audit.json",
                root / "fresh_seed_audit.json",
            )
        )
    copied = copy_to_colab_downloads(download_artifacts) if copy_outputs_to_downloads else []
    result = {
        "run_root": str(root),
        "protocol_path": str(root / "protocol.json"),
        "summary_path": str(root / "final_verdict.json"),
        "claim_matrix_path": str(root / "claim_matrix.csv"),
        "gate_results_path": str(root / "gate_results.csv"),
        "failed_rows_path": str(root / "failed_rows.csv"),
        "open_scientific_rows_path": str(root / "open_scientific_rows.csv"),
        "repair_design_path": str(root / "v6_repair_design.json"),
        "label_audit_path": str(root / "label_contract_audit.json"),
        "runtime_invariance_audit_path": str(root / "runtime_invariance_audit.json"),
        "stability_invariants_path": str(root / "native_stability_invariants.csv"),
        "spectral_observer_audit_path": str(root / "spectral_observer_numerical_audit.json"),
        "baseline_qualification_manifest_path": (
            str(root / "baseline_qualification_manifest.json") if qualified_baseline else None
        ),
        "qualification_manifest_audit_path": (
            str(root / "qualification_manifest_audit.json") if qualified_baseline else None
        ),
        "fresh_seed_audit_path": (
            str(root / "fresh_seed_audit.json") if qualified_baseline else None
        ),
        "final_execution_lock_audit_path": (
            str(root / "final_execution_lock_audit.json") if qualified_baseline else None
        ),
        "paper_report_path": str(report_path),
        "compact_table_manifest_path": str(root / "compact_tables" / "compact_table_manifest.json"),
        "bundle_path": str(bundle_path),
        "decision": verdict,
        "copied_outputs": copied,
    }
    print(
        f"[study-progress] completed status={status} "
        f"ergt32={native_long_mean:.3f} transformer32={transformer_long_mean:.3f}"
    )
    print(
        "[geometry-observer] "
        f"policy={spectral_numerical_audit['policy']} "
        f"solved={spectral_numerical_audit['graphs_solved_cpu_float64']} "
        "unavailable="
        f"{spectral_numerical_audit['graphs_unavailable_after_numerical_failure']}"
    )
    print(
        "[study-readout] "
        f"transformer_id={len(baseline_converged_seeds)}/{len(config['seeds'])} "
        f"matched_id={len(matched_converged_seeds)}/{minimum_matched_seed_count} "
        f"matched_endpoint={verdict['matched_convergence_readout']}"
    )
    for row in gate_rows:
        if bool(row.get("blocking_for_execution_integrity", False)):
            print(f"[study-integrity-check] check={row['gate']} passed={row['passed']}")
    if not failed_rows:
        print("[study-integrity] no_integrity_failures")
    else:
        for row in failed_rows:
            print(
                f"[study-integrity] category={row['category']} name={row['name']} "
                f"reason={row['reason']} value={row.get('value')} "
                f"required={row.get('required')}"
            )
    for key, value in result.items():
        if key != "decision":
            print(f"{key}= {value}")
    return result


__all__ = ["SCHEMA_VERSION", "run_reviewer_suite"]
