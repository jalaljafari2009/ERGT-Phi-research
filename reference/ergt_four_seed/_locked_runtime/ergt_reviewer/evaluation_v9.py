"""V9 evaluation with complete chain, closure, and hard-overflow diagnostics."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .baseline import DirectTransformer
from .data_schema import TrainOnlyTokenizer, collate_examples
from .matched_data import MatchedTopologyExample, collate_matched_topology_examples
from .native_solver import (
    NativeGeometricBoundaryModel,
    hard_solutions_from_outputs,
)

TARGETED_MECHANISM_INTERVENTIONS = {
    "geodesic_action": "no_action",
    "finite_speed_cone": "no_cone",
    "boundary_deficit": "no_boundary",
    "payload_transport": "no_transport",
    "terminal_mass": "no_terminal_mass",
    "memory_source_condition": "no_memory_geometry",
}


def chunks(values: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def memory_bounded_batch_size(
    requested: int,
    token_count: int,
    max_pairwise_cells: int,
) -> int:
    per_example = max(1, token_count * token_count)
    return max(1, min(int(requested), int(max_pairwise_cells) // per_example))


def _pair_exact(rows: Sequence[Mapping[str, Any]]) -> float:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if not bool(row.get("unsupported", False)):
            grouped[str(row["pair_id"])].append(bool(row["correct"]))
    valid = [all(values) and len(values) == 2 for values in grouped.values() if len(values) == 2]
    return sum(valid) / max(1, len(valid))


def _raw_fingerprint(example: MatchedTopologyExample) -> str:
    return hashlib.sha256(example.base.raw_text.encode("utf-8")).hexdigest()


def _gather_tokens(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    suffix = values.shape[2:]
    gather = indices.view(*indices.shape, *([1] * len(suffix))).expand(*indices.shape, *suffix)
    return torch.gather(values, 1, gather)


def summarize_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    correct = [float(bool(row["correct"])) for row in rows]
    unsupported = [row for row in rows if bool(row.get("unsupported", False))]
    supported = [row for row in rows if not bool(row.get("unsupported", False))]
    return {
        "examples": float(len(rows)),
        "accuracy": sum(correct) / max(1, len(correct)),
        "supported_accuracy": sum(float(bool(row["correct"])) for row in supported)
        / max(1, len(supported)),
        "counterfactual_pair_exact": _pair_exact(rows),
        "unsupported_examples": float(len(unsupported)),
        "unsupported_error_rate": sum(int(row["prediction"] != 2) for row in unsupported)
        / max(1, len(unsupported)),
        "abstention_rate": sum(int(row["prediction"] == 2) for row in rows) / max(1, len(rows)),
    }


def evaluate_transformer(
    model: DirectTransformer,
    examples: Sequence[MatchedTopologyExample],
    tokenizer: TrainOnlyTokenizer,
    *,
    device: torch.device,
    batch_size: int,
    cohort: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not examples:
        raise ValueError("cannot evaluate an empty cohort")
    model.eval()
    token_count = max(len(example.tokens) for example in examples)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for group in chunks(tuple(examples), batch_size):
            batch = collate_examples(
                tuple(example.base for example in group), tokenizer, pad_to_tokens=token_count
            ).to(device)
            logits = model(**batch.model_inputs())
            predictions = logits.argmax(dim=-1).cpu().tolist()
            for example, prediction in zip(group, predictions, strict=True):
                label = int(example.answer_id)
                rows.append(
                    {
                        "model": "direct_transformer",
                        "cohort": cohort,
                        "example_id": example.example_id,
                        "pair_id": example.pair_id,
                        "scenario": example.scenario,
                        "hops": int(example.base.metadata["path_hops"]),
                        "tokens": len(example.tokens),
                        "counterfactual_variant": int(example.counterfactual_variant),
                        "label": label,
                        "prediction": int(prediction),
                        "correct": int(prediction) == label,
                        "unsupported": label == 2,
                        "raw_input_sha256": _raw_fingerprint(example),
                    }
                )
    return summarize_predictions(rows), rows


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def evaluate_native(
    model: NativeGeometricBoundaryModel,
    examples: Sequence[MatchedTopologyExample],
    tokenizer: TrainOnlyTokenizer,
    *,
    device: torch.device,
    batch_size: int,
    cohort: str,
    intervention: str = "full",
    intervention_seed: int = 0,
    observe_spectrum: bool = False,
    max_pairwise_cells: int = 2_500_000,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    if not examples:
        raise ValueError("cannot evaluate an empty cohort")
    model.eval()
    token_count = max(len(example.tokens) for example in examples)
    effective_batch = memory_bounded_batch_size(batch_size, token_count, max_pairwise_cells)
    rows: list[dict[str, Any]] = []
    observer_rows: list[dict[str, Any]] = []
    scalar_observers: dict[str, list[float]] = defaultdict(list)
    effective_action_errors: list[float] = []
    measured_hits: dict[str, list[float]] = defaultdict(list)
    program_event_recalls: list[float] = []
    program_event_precisions: list[float] = []
    event_chain_exact_hits: list[float] = []
    selected_event_recalls: list[float] = []
    event_world_path_coverages: list[float] = []
    direct_event_world_path_coverages: list[float] = []
    geodesic_closure_non_degradation_hits: list[float] = []
    event_selection_overflow_hits: list[float] = []
    source_hits: list[float] = []
    candidate_hits: list[float] = []
    boundary_hits: list[float] = []
    observer_keys = (
        "world_usage_entropy",
        "world_geometry_diversity",
        "curvature_proxy_mean",
        "curvature_proxy_negative_fraction",
        "curvature_gradient_mean_abs",
        "event_spectral_entropy",
        "event_spectral_effective_rank",
        "event_spectral_gap",
        "event_spectral_world_diversity",
        "typed_conservation_residual",
        "payload_conservation_residual",
    )
    with torch.inference_mode():
        for group in chunks(tuple(examples), effective_batch):
            batch = collate_matched_topology_examples(
                group, tokenizer, pad_to_tokens=token_count
            ).to(device)
            outputs = model.forward_with_intervention(
                **batch.model_inputs(),
                intervention=intervention,
                intervention_seed=intervention_seed,
                observe_spectrum=observe_spectrum,
            )
            solution, _ = hard_solutions_from_outputs(outputs, model.config)
            predictions = solution.answer_ids.cpu().tolist()
            source_hits.extend(
                (outputs["source_logits"].argmax(dim=-1) == batch.base.supervision.source_positions)
                .float()
                .detach()
                .cpu()
                .tolist()
            )
            candidate_hits.extend(
                (
                    outputs["candidate_logits"].argmax(dim=-1)
                    == batch.base.supervision.candidate_positions
                )
                .all(dim=-1)
                .float()
                .detach()
                .cpu()
                .tolist()
            )
            boundary_hits.extend(
                (
                    outputs["boundary_logits"].argmax(dim=-1)
                    == batch.base.supervision.boundary_labels
                )
                .all(dim=-1)
                .float()
                .detach()
                .cpu()
                .tolist()
            )
            selected = outputs["selected_event_indices"]
            selected_mask = _gather_tokens(batch.physical.event_mask, selected).bool()
            selected_action_target = _gather_tokens(batch.physical.action_cost, selected)
            selected_cone_target = _gather_tokens(batch.physical.cone_admissible, selected)
            selected_transport_target = _gather_tokens(batch.physical.transmission, selected)
            selected_boundary_target = _gather_tokens(batch.physical.boundary_deficit, selected)
            selected_terminal_target = _gather_tokens(batch.physical.terminal_capacity, selected)
            relation_full = torch.cat(
                (
                    (1.0 - outputs["event_presence"]).unsqueeze(-1),
                    outputs["event_relation_probability"],
                ),
                dim=-1,
            )
            predicted_relation = relation_full.argmax(dim=-1)
            predicted_active = (
                (outputs["event_presence"] >= model.config.event_presence_floor)
                & (predicted_relation > 0)
                & outputs["event_valid_mask"].bool()
            )
            for row_index in range(len(group)):
                gold_positions = (
                    batch.physical.event_mask[row_index].nonzero(as_tuple=False).flatten()
                )
                gold_program = {
                    (
                        int(position.item()),
                        int(batch.base.supervision.relation_anchor_labels[row_index, position]),
                        int(batch.base.supervision.event_source_positions[row_index, position]),
                        int(batch.base.supervision.event_target_positions[row_index, position]),
                    )
                    for position in gold_positions
                }
                predicted_program = {
                    (
                        int(selected[row_index, slot].item()),
                        int(predicted_relation[row_index, slot].item()),
                        int(outputs["event_source_index"][row_index, slot].item()),
                        int(outputs["event_target_index"][row_index, slot].item()),
                    )
                    for slot in predicted_active[row_index].nonzero(as_tuple=False).flatten()
                }
                matched = len(gold_program & predicted_program)
                program_event_recalls.append(matched / max(1, len(gold_program)))
                program_event_precisions.append(matched / max(1, len(predicted_program)))
                event_chain_exact_hits.append(float(gold_program == predicted_program))
                selected_positions = {
                    int(selected[row_index, slot].item())
                    for slot in outputs["event_valid_mask"][row_index]
                    .bool()
                    .nonzero(as_tuple=False)
                    .flatten()
                }
                selected_event_recalls.append(
                    len(selected_positions & {int(value.item()) for value in gold_positions})
                    / max(1, len(gold_positions))
                )
                selected_gold_mask = selected_mask[row_index]
                closure_covered = int(
                    outputs["event_world_closure_hard_gate"][row_index][selected_gold_mask]
                    .bool()
                    .sum()
                    .item()
                )
                direct_covered = int(
                    outputs["event_world_direct_hard_gate"][row_index][selected_gold_mask]
                    .bool()
                    .sum()
                    .item()
                )
                true_count = max(1, len(gold_positions))
                closure_coverage = closure_covered / true_count
                direct_coverage = direct_covered / true_count
                direct_event_world_path_coverages.append(direct_coverage)
                geodesic_closure_non_degradation_hits.append(
                    float(closure_coverage + 1.0e-8 >= direct_coverage)
                )
                event_selection_overflow_hits.append(
                    float(outputs["event_selection_overflow"][row_index].item() > 0)
                )
            if bool(selected_mask.any()):
                effective_action_errors.extend(
                    (
                        outputs["event_continuous_hard_action"][selected_mask]
                        - selected_action_target[selected_mask]
                    )
                    .abs()
                    .detach()
                    .cpu()
                    .tolist()
                )
                action_hit = (outputs["event_hard_action"] - selected_action_target).abs() <= 1.0e-6
                cone_hit = (outputs["event_cone"] >= 0.5) == (selected_cone_target >= 0.5)
                transport_hit = (
                    outputs["event_measured_transmission_pair"] - selected_transport_target
                ).abs() <= 1.0e-6
                boundary_hit = (
                    outputs["event_measured_boundary_deficit_pair"] - selected_boundary_target
                ).abs() <= 1.0e-6
                terminal_hit = (
                    outputs["event_measured_terminal_mass_pair"] - selected_terminal_target
                ).abs() <= 1.0e-6
                for key, values in (
                    ("action", action_hit),
                    ("cone", cone_hit),
                    ("transport", transport_hit),
                    ("boundary", boundary_hit),
                    ("terminal", terminal_hit),
                ):
                    measured_hits[key].extend(values[selected_mask].float().detach().cpu().tolist())
                measured_hits["event_exact"].extend(
                    (action_hit & cone_hit & transport_hit & boundary_hit & terminal_hit)[
                        selected_mask
                    ]
                    .float()
                    .detach()
                    .cpu()
                    .tolist()
                )
                event_world_path_coverages.extend(
                    outputs["event_world_closure_hard_gate"][selected_mask]
                    .float()
                    .detach()
                    .cpu()
                    .tolist()
                )
            for key in observer_keys:
                value = outputs.get(key)
                if isinstance(value, torch.Tensor):
                    scalar_observers[key].append(float(value.float().mean().cpu().item()))
            for row_index, (example, prediction) in enumerate(zip(group, predictions, strict=True)):
                label = int(example.answer_id)
                rows.append(
                    {
                        "model": "native_ergt",
                        "cohort": cohort,
                        "intervention": intervention,
                        "example_id": example.example_id,
                        "pair_id": example.pair_id,
                        "scenario": example.scenario,
                        "hops": int(example.base.metadata["path_hops"]),
                        "tokens": len(example.tokens),
                        "counterfactual_variant": int(example.counterfactual_variant),
                        "label": label,
                        "prediction": int(prediction),
                        "correct": int(prediction) == label,
                        "unsupported": label == 2,
                        "failure_code": int(solution.failure_code[row_index].cpu().item()),
                        "selected_action": float(solution.selected_action[row_index].cpu().item()),
                        "raw_input_sha256": _raw_fingerprint(example),
                    }
                )
                observer_rows.append(
                    {
                        "cohort": cohort,
                        "intervention": intervention,
                        "example_id": example.example_id,
                        "hops": int(example.base.metadata["path_hops"]),
                        "tokens": len(example.tokens),
                        "curvature_proxy_mean": float(
                            outputs["event_curvature_proxy"][row_index].float().mean().cpu().item()
                        ),
                        "curvature_gradient_mean_abs": float(
                            outputs["event_curvature_graph_gradient"][row_index]
                            .float()
                            .abs()
                            .mean()
                            .cpu()
                            .item()
                        ),
                        "spectral_entropy": float(
                            outputs["event_spectral_entropy"][row_index].cpu().item()
                        ),
                        "spectral_effective_rank": float(
                            outputs["event_spectral_effective_rank"][row_index].cpu().item()
                        ),
                        "spectral_gap": float(
                            outputs["event_spectral_gap"][row_index].cpu().item()
                        ),
                    }
                )
    metrics = summarize_predictions(rows)
    metrics.update({key: _mean(values) for key, values in scalar_observers.items()})
    for key in ("typed_conservation_residual", "payload_conservation_residual"):
        metrics[key] = max(scalar_observers[key], default=0.0)
    metrics.update(
        {
            "effective_action_mae": _mean(effective_action_errors),
            "program_event_recall": _mean(program_event_recalls),
            "program_event_precision": _mean(program_event_precisions),
            "event_chain_exact": _mean(event_chain_exact_hits),
            "selected_event_recall": _mean(selected_event_recalls),
            "event_world_path_coverage": _mean(event_world_path_coverages),
            "event_world_direct_path_coverage": _mean(direct_event_world_path_coverages),
            "geodesic_closure_non_degradation_rate": _mean(geodesic_closure_non_degradation_hits),
            "event_selection_overflow_rate": _mean(event_selection_overflow_hits),
            "source_accuracy": _mean(source_hits),
            "candidate_exact": _mean(candidate_hits),
            "boundary_exact": _mean(boundary_hits),
            "measured_action_accuracy": _mean(measured_hits["action"]),
            "measured_cone_accuracy": _mean(measured_hits["cone"]),
            "measured_transport_accuracy": _mean(measured_hits["transport"]),
            "measured_boundary_accuracy": _mean(measured_hits["boundary"]),
            "measured_terminal_accuracy": _mean(measured_hits["terminal"]),
            "measured_event_exact": _mean(measured_hits["event_exact"]),
        }
    )
    metrics["effective_batch_size"] = float(effective_batch)
    return metrics, rows, observer_rows


def evaluate_targeted_interventions(
    model: NativeGeometricBoundaryModel,
    examples: Sequence[MatchedTopologyExample],
    tokenizer: TrainOnlyTokenizer,
    *,
    device: torch.device,
    batch_size: int,
    max_pairwise_cells: int,
    max_pairs_per_scenario: int = 4,
    max_global_pairs: int = 14,
    full_accuracy_floor: float = 0.80,
    minimum_eligible_pairs: int = 2,
    include_global_interventions: bool = True,
    include_single_world_interventions: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    targeted = TARGETED_MECHANISM_INTERVENTIONS

    def pair_limited(
        values: Sequence[MatchedTopologyExample], maximum_pairs: int
    ) -> tuple[MatchedTopologyExample, ...]:
        grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
        for value in values:
            key = (value.scenario, int(value.base.metadata["path_hops"]))
            if value.pair_id not in grouped[key]:
                grouped[key].append(value.pair_id)
        selected_ids: list[str] = []
        depth = 0
        while len(selected_ids) < maximum_pairs:
            added = False
            for key in sorted(grouped):
                candidates = sorted(grouped[key])
                if depth < len(candidates):
                    selected_ids.append(candidates[depth])
                    added = True
                    if len(selected_ids) >= maximum_pairs:
                        break
            if not added:
                break
            depth += 1
        allowed = set(selected_ids)
        return tuple(value for value in values if value.pair_id in allowed)

    def correctly_solved_pairs(
        values: Sequence[MatchedTopologyExample],
        prediction_rows: Sequence[Mapping[str, Any]],
    ) -> tuple[MatchedTopologyExample, ...]:
        correctness: dict[str, list[bool]] = defaultdict(list)
        for row in prediction_rows:
            correctness[str(row["pair_id"])].append(bool(row["correct"]))
        eligible = {
            pair_id
            for pair_id, outcomes in correctness.items()
            if len(outcomes) == 2 and all(outcomes)
        }
        return tuple(value for value in values if value.pair_id in eligible)

    global_panel = pair_limited(examples, max_global_pairs)
    global_panel_metrics, global_panel_rows, _ = evaluate_native(
        model,
        global_panel,
        tokenizer,
        device=device,
        batch_size=batch_size,
        cohort="intervention_full",
        max_pairwise_cells=max_pairwise_cells,
    )
    global_examples = correctly_solved_pairs(global_panel, global_panel_rows)
    global_pair_count = len({value.pair_id for value in global_examples})
    global_evaluable = float(global_panel_metrics["accuracy"]) >= float(
        full_accuracy_floor
    ) and global_pair_count >= int(minimum_eligible_pairs)
    if global_examples:
        full, _, _ = evaluate_native(
            model,
            global_examples,
            tokenizer,
            device=device,
            batch_size=batch_size,
            cohort="intervention_eligible_full",
            max_pairwise_cells=max_pairwise_cells,
        )
    else:
        full = global_panel_metrics
    global_evaluation_examples = global_examples or global_panel
    rows: list[dict[str, Any]] = []
    drops: dict[str, float] = {}
    for scenario, intervention in targeted.items():
        subset = pair_limited(
            tuple(example for example in examples if example.scenario == scenario),
            max_pairs_per_scenario,
        )
        if not subset:
            continue
        full_panel, full_rows, _ = evaluate_native(
            model,
            subset,
            tokenizer,
            device=device,
            batch_size=batch_size,
            cohort=f"intervention_{scenario}_full",
            max_pairwise_cells=max_pairwise_cells,
        )
        eligible = correctly_solved_pairs(subset, full_rows)
        eligible_pairs = len({value.pair_id for value in eligible})
        evaluable = float(full_panel["accuracy"]) >= float(
            full_accuracy_floor
        ) and eligible_pairs >= int(minimum_eligible_pairs)
        if not eligible:
            rows.append(
                {
                    "scenario": scenario,
                    "intervention": intervention,
                    "panel_full_accuracy": full_panel["accuracy"],
                    "full_accuracy": full_panel["accuracy"],
                    "intervention_accuracy": full_panel["accuracy"],
                    "targeted_drop": 0.0,
                    "examples": len(subset),
                    "eligible_examples": 0,
                    "eligible_pairs": 0,
                    "attribution_evaluable": False,
                }
            )
            drops[f"{scenario}_targeted_drop"] = 0.0
            continue
        full_subset, _, _ = evaluate_native(
            model,
            eligible,
            tokenizer,
            device=device,
            batch_size=batch_size,
            cohort=f"intervention_{scenario}_eligible_full",
            max_pairwise_cells=max_pairwise_cells,
        )
        changed, _, _ = evaluate_native(
            model,
            eligible,
            tokenizer,
            device=device,
            batch_size=batch_size,
            cohort=f"intervention_{scenario}_{intervention}",
            intervention=intervention,
            intervention_seed=911,
            max_pairwise_cells=max_pairwise_cells,
        )
        drop = float(full_subset["accuracy"] - changed["accuracy"])
        drops[f"{scenario}_targeted_drop"] = drop
        rows.append(
            {
                "scenario": scenario,
                "intervention": intervention,
                "panel_full_accuracy": full_panel["accuracy"],
                "full_accuracy": full_subset["accuracy"],
                "intervention_accuracy": changed["accuracy"],
                "targeted_drop": drop,
                "examples": len(subset),
                "eligible_examples": len(eligible),
                "eligible_pairs": eligible_pairs,
                "attribution_evaluable": evaluable,
            }
        )
    global_interventions = (
        "shuffled_geometry",
        "random_geometry",
        "no_phi",
        "no_event_backreaction",
        "direct_world_gate",
        "no_multiscale_backbone",
        "no_world_transitions",
    )
    if include_global_interventions:
        for intervention in global_interventions:
            changed, _, _ = evaluate_native(
                model,
                global_evaluation_examples,
                tokenizer,
                device=device,
                batch_size=batch_size,
                cohort=f"intervention_{intervention}",
                intervention=intervention,
                intervention_seed=917,
                max_pairwise_cells=max_pairwise_cells,
            )
            drop = float(full["accuracy"] - changed["accuracy"])
            drops[f"{intervention}_drop"] = drop
            rows.append(
                {
                    "scenario": "all",
                    "intervention": intervention,
                    "panel_full_accuracy": global_panel_metrics["accuracy"],
                    "full_accuracy": full["accuracy"],
                    "intervention_accuracy": changed["accuracy"],
                    "targeted_drop": drop,
                    "examples": len(global_panel),
                    "eligible_examples": len(global_examples),
                    "eligible_pairs": global_pair_count,
                    "attribution_evaluable": global_evaluable,
                }
            )
    single_world_accuracies: list[float] = []
    if include_single_world_interventions:
        for world in range(8):
            intervention = f"only_world_{world}"
            changed, _, _ = evaluate_native(
                model,
                global_evaluation_examples,
                tokenizer,
                device=device,
                batch_size=batch_size,
                cohort=f"intervention_{intervention}",
                intervention=intervention,
                intervention_seed=919 + world,
                max_pairwise_cells=max_pairwise_cells,
            )
            single_world_accuracies.append(float(changed["accuracy"]))
            rows.append(
                {
                    "scenario": "all",
                    "intervention": intervention,
                    "panel_full_accuracy": global_panel_metrics["accuracy"],
                    "full_accuracy": full["accuracy"],
                    "intervention_accuracy": changed["accuracy"],
                    "targeted_drop": float(full["accuracy"] - changed["accuracy"]),
                    "examples": len(global_panel),
                    "eligible_examples": len(global_examples),
                    "eligible_pairs": global_pair_count,
                    "attribution_evaluable": global_evaluable,
                }
            )
        best_single = max(single_world_accuracies, default=0.0)
        drops["best_single_world_accuracy"] = best_single
        drops["full_minus_best_single_world"] = float(full["accuracy"] - best_single)
    return rows, drops


__all__ = [
    "TARGETED_MECHANISM_INTERVENTIONS",
    "evaluate_native",
    "evaluate_targeted_interventions",
    "evaluate_transformer",
    "memory_bounded_batch_size",
    "summarize_predictions",
]
