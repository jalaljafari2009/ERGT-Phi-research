"""V9 training loops with V21 multi-hop readiness and exposure locking."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .baseline import (
    DirectTransformer,
    TransformerConfig,
    count_parameters,
    inference_active_parameter_count,
    matched_transformer_config,
    training_only_parameter_count,
)
from .data_schema import TrainOnlyTokenizer
from .evaluation_v9 import (
    TARGETED_MECHANISM_INTERVENTIONS,
    evaluate_native,
    evaluate_targeted_interventions,
    evaluate_transformer,
)
from .fair_data_v9 import partition_mechanism_tuning_shards
from .matched_data import (
    MatchedTopologyExample,
    RawTokenInputContract,
    collate_matched_topology_examples,
)
from .native_solver import (
    ERGT43Config,
    NativeGeometricBoundaryModel,
    native_geometric_training_loss,
)


@dataclass
class TrainedModels:
    native: NativeGeometricBoundaryModel
    transformer: DirectTransformer
    native_config: ERGT43Config
    transformer_config: TransformerConfig
    curves: list[dict[str, Any]]
    resources: list[dict[str, Any]]
    checkpoint_paths: dict[str, str]
    parameter_audit: dict[str, Any]


def protocol_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _training_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "baseline.py",
        "data_schema.py",
        "evaluation_v9.py",
        "fair_data_v9.py",
        "matched_data.py",
        "native_solver.py",
        "physics_core.py",
        "product_geodesic.py",
        "training_v9.py",
        "world_contract.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _dataset_fingerprint(examples: Sequence[MatchedTopologyExample]) -> str:
    return protocol_hash(
        {
            "examples": [
                {
                    "example_id": example.example_id,
                    "pair_id": example.pair_id,
                    "raw_text": example.base.raw_text,
                    "answer_id": int(example.answer_id),
                }
                for example in examples
            ]
        }
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _grouped_sample(
    examples: Sequence[MatchedTopologyExample],
    batch_size: int,
    rng: random.Random,
) -> tuple[MatchedTopologyExample, ...]:
    grouped: dict[str, list[MatchedTopologyExample]] = defaultdict(list)
    for example in examples:
        grouped[example.pair_id].append(example)
    groups = tuple(grouped.values())
    count = max(1, (batch_size + 1) // 2)
    selected = rng.sample(groups, min(count, len(groups)))
    batch = [example for group in selected for example in group]
    rng.shuffle(batch)
    return tuple(batch[:batch_size])


def _ordered_pair_groups(
    examples: Sequence[MatchedTopologyExample],
) -> tuple[tuple[MatchedTopologyExample, ...], ...]:
    grouped: dict[str, list[MatchedTopologyExample]] = defaultdict(list)
    for example in examples:
        grouped[example.pair_id].append(example)
    malformed = sorted(pair_id for pair_id, values in grouped.items() if len(values) != 2)
    if malformed:
        raise ValueError(f"V9 requires complete counterfactual pairs: {malformed[:3]}")
    return tuple(tuple(grouped[pair_id]) for pair_id in sorted(grouped))


def _epoch_grouped_sample(
    groups: Sequence[Sequence[MatchedTopologyExample]],
    batch_size: int,
    sample_index: int,
    seed: int,
) -> tuple[tuple[MatchedTopologyExample, ...], tuple[str, ...]]:
    """Visit every pair group once per shuffled epoch before any group repeats."""

    groups_per_batch = max(1, (batch_size + 1) // 2)
    selected: list[MatchedTopologyExample] = []
    selected_ids: list[str] = []
    for offset in range(groups_per_batch):
        position = sample_index * groups_per_batch + offset
        epoch = position // len(groups)
        within_epoch = position % len(groups)
        order = list(range(len(groups)))
        random.Random(seed + 1_000_003 * epoch).shuffle(order)
        group = groups[order[within_epoch]]
        selected.extend(group)
        selected_ids.append(str(group[0].pair_id))
    random.Random(seed + 17_171 * (sample_index + 1)).shuffle(selected)
    return tuple(selected[:batch_size]), tuple(selected_ids)


def _group_metadata(
    groups: Sequence[Sequence[MatchedTopologyExample]],
) -> dict[str, dict[str, Any]]:
    return {
        str(group[0].pair_id): {
            "hops": int(group[0].base.metadata["path_hops"]),
            "scenario": str(group[0].scenario),
            "unsupported": bool(group[0].base.metadata.get("unsupported_answer", False)),
        }
        for group in groups
    }


def _exposure_summary(
    exposures: Mapping[str, int],
    metadata: Mapping[str, Mapping[str, Any]],
    required_minimum: int,
) -> dict[str, Any]:
    values = [int(exposures.get(pair_id, 0)) for pair_id in metadata]
    by_hop: dict[str, list[int]] = defaultdict(list)
    by_cell: dict[str, list[int]] = defaultdict(list)
    for pair_id, record in metadata.items():
        value = int(exposures.get(pair_id, 0))
        hop = str(int(record["hops"]))
        scenario = str(record["scenario"])
        unsupported = bool(record["unsupported"])
        by_hop[hop].append(value)
        by_cell[f"{scenario}|hop_{hop}|unsupported_{int(unsupported)}"].append(value)
    return {
        "pair_group_count": len(values),
        "mean_group_exposure": sum(values) / max(1, len(values)),
        "minimum_group_exposure": min(values, default=0),
        "maximum_group_exposure": max(values, default=0),
        "unseen_group_count": sum(value == 0 for value in values),
        "required_minimum_group_exposure": int(required_minimum),
        "minimum_exposure_pass": bool(values) and min(values) >= int(required_minimum),
        "by_hop": {
            hop: {
                "groups": len(cell),
                "minimum": min(cell),
                "mean": sum(cell) / len(cell),
                "maximum": max(cell),
            }
            for hop, cell in sorted(by_hop.items(), key=lambda item: int(item[0]))
        },
        "by_scenario_hop": {
            key: {
                "groups": len(cell),
                "minimum": min(cell),
                "mean": sum(cell) / len(cell),
                "maximum": max(cell),
            }
            for key, cell in sorted(by_cell.items())
        },
    }


def _native_field_ready_v9(
    metrics: Mapping[str, float],
    config: Mapping[str, Any],
) -> bool:
    return (
        _native_field_ready(metrics, config)
        and float(metrics["geodesic_closure_non_degradation_rate"])
        >= float(config.get("native_geodesic_closure_non_degradation_floor", 1.0))
        and float(metrics["event_selection_overflow_rate"])
        <= float(config.get("native_event_selection_overflow_ceiling", 0.0))
    )


def _save_training_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_state: Mapping[str, torch.Tensor] | None,
    best_score: float,
    curves: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    run_hash: str,
    training_state: Mapping[str, Any] | None = None,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "run_hash": run_hash,
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_state": best_state,
            "best_score": best_score,
            "curves": list(curves),
            "config": dict(config),
            "training_state": dict(training_state or {}),
        },
        temporary,
    )
    temporary.replace(path)


def _load_training_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    run_hash: str,
    device: torch.device,
) -> tuple[
    int,
    Mapping[str, torch.Tensor] | None,
    float,
    list[dict[str, Any]],
    dict[str, Any],
]:
    if not path.exists():
        return 0, None, float("-inf"), [], {}
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("run_hash") != run_hash:
        print(f"[checkpoint-resume] ignored incompatible checkpoint: {path}")
        return 0, None, float("-inf"), [], {}
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    return (
        int(state["step"]),
        state.get("best_state"),
        float(state.get("best_score", float("-inf"))),
        list(state.get("curves", [])),
        dict(state.get("training_state", {})),
    )


def _measurement_min(metrics: Mapping[str, float]) -> float:
    return min(
        float(metrics[key])
        for key in (
            "measured_action_accuracy",
            "measured_cone_accuracy",
            "measured_transport_accuracy",
            "measured_boundary_accuracy",
            "measured_terminal_accuracy",
        )
    )


def _native_field_ready(metrics: Mapping[str, float], config: Mapping[str, Any]) -> bool:
    return (
        float(metrics["supported_accuracy"]) >= float(config["native_readiness_accuracy_floor"])
        and float(metrics["counterfactual_pair_exact"])
        >= float(config["native_readiness_pair_floor"])
        and float(metrics["unsupported_error_rate"]) <= float(config["unsupported_error_ceiling"])
        and _measurement_min(metrics) >= float(config["native_measurement_accuracy_floor"])
        and float(metrics["effective_action_mae"])
        <= float(config["native_effective_action_mae_ceiling"])
        and float(metrics["program_event_recall"])
        >= float(config["native_program_event_recall_floor"])
        and float(metrics["program_event_precision"])
        >= float(config["native_program_event_precision_floor"])
        and float(metrics["event_world_path_coverage"])
        >= float(config["native_path_coverage_floor"])
        and float(metrics["event_chain_exact"]) >= float(config["native_event_chain_exact_floor"])
        and float(metrics["source_accuracy"]) >= float(config["native_query_role_accuracy_floor"])
        and float(metrics["candidate_exact"]) >= float(config["native_query_role_accuracy_floor"])
        and float(metrics["boundary_exact"]) >= float(config["native_query_role_accuracy_floor"])
        and float(metrics["typed_conservation_residual"])
        <= float(config.get("conservation_residual_ceiling", 1.0e-4))
        and float(metrics["payload_conservation_residual"])
        <= float(config.get("conservation_residual_ceiling", 1.0e-4))
    )


def _selection_score(metrics: Mapping[str, float]) -> float:
    unsupported_score = (
        1.0 - float(metrics["unsupported_error_rate"])
        if float(metrics["unsupported_examples"]) > 0
        else 1.0
    )
    return (
        float(metrics["supported_accuracy"])
        + float(metrics["counterfactual_pair_exact"])
        + unsupported_score
    )


def _native_mechanism_tuning_ready(
    model: NativeGeometricBoundaryModel,
    examples: Sequence[MatchedTopologyExample],
    tokenizer: TrainOnlyTokenizer,
    *,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    """Evaluate the six registered mechanisms on a tuning-only cohort."""

    if not bool(config.get("mechanism_tuning_required", True)):
        return (
            True,
            {
                "mechanism_tuning_complete": True,
                "mechanism_tuning_evaluable": True,
                "mechanism_tuning_min_targeted_drop": float(config["mechanism_tuning_drop_floor"]),
                "mechanism_tuning_min_panel_full_accuracy": 0.0,
                "mechanism_tuning_scenarios": 0,
            },
            [],
        )

    rows, _ = evaluate_targeted_interventions(
        model,
        examples,
        tokenizer,
        device=device,
        batch_size=int(config["eval_batch_size"]),
        max_pairwise_cells=int(config["max_pairwise_cells"]),
        max_pairs_per_scenario=int(config["mechanism_tuning_pairs_per_scenario"]),
        max_global_pairs=max(1, int(config["mechanism_tuning_pairs_per_scenario"])),
        full_accuracy_floor=float(config["mechanism_tuning_full_accuracy_floor"]),
        minimum_eligible_pairs=int(config["mechanism_tuning_minimum_eligible_pairs"]),
        include_global_interventions=False,
        include_single_world_interventions=False,
    )
    by_scenario = {str(row["scenario"]): row for row in rows}
    expected = set(TARGETED_MECHANISM_INTERVENTIONS)
    complete = set(by_scenario) == expected
    evaluable = complete and all(
        bool(by_scenario[scenario]["attribution_evaluable"]) for scenario in expected
    )
    minimum_drop = min(
        (float(by_scenario[scenario]["targeted_drop"]) for scenario in expected),
        default=0.0,
    )
    minimum_full_accuracy = min(
        (float(by_scenario[scenario]["panel_full_accuracy"]) for scenario in expected),
        default=0.0,
    )
    passed = complete and evaluable and minimum_drop >= float(config["mechanism_tuning_drop_floor"])
    return (
        passed,
        {
            "mechanism_tuning_complete": complete,
            "mechanism_tuning_evaluable": evaluable,
            "mechanism_tuning_min_targeted_drop": minimum_drop,
            "mechanism_tuning_min_panel_full_accuracy": minimum_full_accuracy,
            "mechanism_tuning_scenarios": len(by_scenario),
        },
        rows,
    )


def _native_mechanism_shards_ready(
    model: NativeGeometricBoundaryModel,
    examples: Sequence[MatchedTopologyExample],
    tokenizer: TrainOnlyTokenizer,
    *,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    """Require every independently generated mechanism shard to pass."""

    shards = partition_mechanism_tuning_shards(examples)
    expected_shards = int(config.get("mechanism_tuning_shard_count", 1))
    summaries: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for shard, values in shards.items():
        passed, summary, shard_rows = _native_mechanism_tuning_ready(
            model,
            values,
            tokenizer,
            config=config,
            device=device,
        )
        summaries[shard] = {"passed": bool(passed), **summary}
        rows.extend({"mechanism_shard": shard, **row} for row in shard_rows)
    complete = len(shards) == expected_shards
    passed = complete and all(bool(summary["passed"]) for summary in summaries.values())
    return (
        passed,
        {
            "mechanism_tuning_complete": complete,
            "mechanism_tuning_evaluable": complete
            and all(bool(summary["mechanism_tuning_evaluable"]) for summary in summaries.values()),
            "mechanism_tuning_min_targeted_drop": min(
                (
                    float(summary["mechanism_tuning_min_targeted_drop"])
                    for summary in summaries.values()
                ),
                default=0.0,
            ),
            "mechanism_tuning_min_panel_full_accuracy": min(
                (
                    float(summary["mechanism_tuning_min_panel_full_accuracy"])
                    for summary in summaries.values()
                ),
                default=0.0,
            ),
            "mechanism_tuning_scenarios": sum(
                int(summary["mechanism_tuning_scenarios"]) for summary in summaries.values()
            ),
            "mechanism_tuning_shards_expected": expected_shards,
            "mechanism_tuning_shards_observed": len(shards),
            "mechanism_tuning_shards_passed": sum(
                int(bool(summary["passed"])) for summary in summaries.values()
            ),
            "mechanism_tuning_shard_summaries": summaries,
        },
        rows,
    )


def _transformer_training_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Any,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Give the direct Transformer the same training-label families as ERGT.

    All auxiliary predictions are discarded at inference and cannot feed the
    direct answer logits. This repairs supervision asymmetry without adding a
    compiler, graph executor, or geometry to the Transformer answer path.
    """

    supervision = batch.base.supervision
    valid = batch.base.attention_mask.bool()
    event_mask = supervision.relation_event_mask.bool()
    answer_class_weight = outputs["answer_logits"].new_tensor((1.0, 1.0, 1.25))
    answer_loss = F.cross_entropy(
        outputs["answer_logits"], supervision.answer_labels, weight=answer_class_weight
    )

    entity_targets = supervision.entity_labels[valid]
    positive = entity_targets.sum().clamp_min(1.0)
    entity_positive_weight = (entity_targets.numel() - positive) / positive
    entity_loss = F.binary_cross_entropy_with_logits(
        outputs["entity_logits"][valid],
        entity_targets,
        pos_weight=entity_positive_weight,
    )
    query_role_loss = F.cross_entropy(
        outputs["query_role_logits"][valid],
        supervision.query_role_labels[valid],
        weight=outputs["answer_logits"].new_tensor((0.10, 1.0, 1.0, 1.0, 1.0, 1.0)),
    )
    relation_anchor_loss = F.cross_entropy(
        outputs["relation_anchor_logits"][valid],
        supervision.relation_anchor_labels[valid],
        weight=outputs["answer_logits"].new_tensor((0.10, 1.0, 1.0, 1.0)),
    )
    if bool(event_mask.any()):
        source_pointer_loss = F.cross_entropy(
            outputs["event_source_logits"][event_mask],
            supervision.event_source_positions[event_mask],
        )
        target_pointer_loss = F.cross_entropy(
            outputs["event_target_logits"][event_mask],
            supervision.event_target_positions[event_mask],
        )
        action_targets = (batch.physical.action_cost[event_mask].round().long() - 1).clamp(0, 2)
        cone_targets = (batch.physical.cone_admissible[event_mask] >= 0.5).long()
        transport_targets = (batch.physical.transmission[event_mask] >= 0.5).long()
        boundary_targets = (batch.physical.boundary_deficit[event_mask] >= 1.0).long()
        terminal_targets = (batch.physical.terminal_capacity[event_mask] >= 0.5).long()
        physical_loss = (
            sum(
                (
                    F.cross_entropy(outputs["physical_action_logits"][event_mask], action_targets),
                    F.cross_entropy(outputs["physical_cone_logits"][event_mask], cone_targets),
                    F.cross_entropy(
                        outputs["physical_transport_logits"][event_mask], transport_targets
                    ),
                    F.cross_entropy(
                        outputs["physical_boundary_logits"][event_mask], boundary_targets
                    ),
                    F.cross_entropy(
                        outputs["physical_terminal_logits"][event_mask], terminal_targets
                    ),
                )
            )
            / 5.0
        )
    else:
        zero = answer_loss.new_tensor(0.0)
        source_pointer_loss = zero
        target_pointer_loss = zero
        physical_loss = zero

    boundary_mask = supervision.query_boundary_value_labels >= 0
    boundary_value_loss = (
        F.cross_entropy(
            outputs["boundary_value_logits"][boundary_mask],
            supervision.query_boundary_value_labels[boundary_mask],
        )
        if bool(boundary_mask.any())
        else answer_loss.new_tensor(0.0)
    )

    grouped: dict[str, list[int]] = defaultdict(list)
    for row, example in enumerate(batch.examples):
        grouped[example.pair_id].append(row)
    equivariance_terms: list[torch.Tensor] = []
    log_probabilities = F.log_softmax(outputs["answer_logits"][:, :2], dim=-1)
    for indices in grouped.values():
        if len(indices) != 2:
            continue
        left, right = indices
        labels = {
            int(supervision.answer_labels[left].item()),
            int(supervision.answer_labels[right].item()),
        }
        if labels != {0, 1}:
            continue
        equivariance_terms.append(
            F.mse_loss(log_probabilities[left], log_probabilities[right].flip(0))
        )
    pair_equivariance_loss = (
        torch.stack(equivariance_terms).mean()
        if equivariance_terms
        else answer_loss.new_tensor(0.0)
    )
    auxiliary_loss = (
        sum(
            (
                entity_loss,
                query_role_loss,
                relation_anchor_loss,
                source_pointer_loss,
                target_pointer_loss,
                boundary_value_loss,
                physical_loss,
            )
        )
        / 7.0
    )
    total = (
        answer_loss
        + float(config.get("transformer_auxiliary_weight", 0.50)) * auxiliary_loss
        + float(config.get("transformer_pair_equivariance_weight", 0.25)) * pair_equivariance_loss
    )
    return total, {
        "answer": answer_loss,
        "auxiliary": auxiliary_loss,
        "pair_equivariance": pair_equivariance_loss,
        "entity": entity_loss,
        "query_role": query_role_loss,
        "relation_anchor": relation_anchor_loss,
        "source_pointer": source_pointer_loss,
        "target_pointer": target_pointer_loss,
        "boundary_value": boundary_value_loss,
        "physical": physical_loss,
    }


def _train_native(
    model: NativeGeometricBoundaryModel,
    train: Sequence[MatchedTopologyExample],
    tuning: Sequence[MatchedTopologyExample],
    mechanism_tuning: Sequence[MatchedTopologyExample],
    tokenizer: TrainOnlyTokenizer,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    checkpoint: Path,
    run_hash: str,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    steps = int(config["native_steps"])
    batch_size = int(config["batch_size"])
    eval_interval = int(config["eval_interval"])
    learning_rate = float(config["native_learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_step, best_state, best_score, curves, restored = (
        0,
        None,
        float("-inf"),
        [],
        {},
    )
    if resume:
        start_step, best_state, best_score, curves, restored = _load_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            run_hash=run_hash,
            device=device,
        )
    rng = random.Random(seed + 55)
    if restored.get("sampling_rng_state") is not None:
        rng.setstate(restored["sampling_rng_state"])
    train_pad = max(len(example.tokens) for example in train)
    teacher_schedule_steps = int(config.get("native_teacher_schedule_steps", steps))
    teacher_cut = max(
        1,
        int(
            math.ceil(
                float(config.get("native_teacher_cut_fraction", 0.78)) * teacher_schedule_steps
            )
        ),
    )
    readiness_start = max(2 * eval_interval, teacher_schedule_steps // 4)
    required_windows = int(config.get("checkpoint_stable_windows", 2))
    readiness_streak = int(restored.get("readiness_streak", 0))
    stable_streak = int(restored.get("stable_streak", 0))
    field_ready = bool(restored.get("field_ready", False))
    field_ready_step = restored.get("field_ready_step")
    mechanism_ready = bool(restored.get("mechanism_ready", False))
    mechanism_evaluations = int(restored.get("mechanism_evaluations", 0))
    selected_mechanism_min_drop = float(restored.get("selected_mechanism_min_drop", 0.0))
    selected_step = restored.get("selected_step")
    gradient_reaches_psi = bool(restored.get("gradient_reaches_psi", False))
    optimizer_steps = int(restored.get("optimizer_steps", 0))
    zero_teacher_optimizer_steps = int(restored.get("zero_teacher_optimizer_steps", 0))
    completed_step = start_step
    training_state: dict[str, Any] = dict(restored)
    training_complete = bool(restored.get("training_complete", False)) or (
        best_state is not None
        and field_ready
        and mechanism_ready
        and stable_streak >= required_windows
    )
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step_iterator = () if training_complete else range(start_step + 1, steps + 1)
    for step in step_iterator:
        completed_step = step
        scheduled_teacher = 1.0 if step < teacher_cut else 0.0
        teacher = 1.0 if scheduled_teacher > 0.0 or not field_ready else 0.0
        answer_weight = 0.15 if teacher > 0.0 else 1.0
        if teacher > 0.0:
            group = _grouped_sample(train, batch_size, rng)
            batch = collate_matched_topology_examples(group, tokenizer, pad_to_tokens=train_pad).to(
                device
            )
            model.train()
            outputs = model(**batch.model_inputs())
            loss, parts = native_geometric_training_loss(
                outputs,
                batch,
                config=model.config,
                teacher_weight=teacher,
                answer_weight=answer_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_reaches_psi = gradient_reaches_psi or any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and float(parameter.grad.abs().sum().item()) > 0.0
                for name, parameter in model.named_parameters()
                if "psi" in name or "world" in name
            )
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            optimizer_steps += 1
            if teacher == 0.0:
                zero_teacher_optimizer_steps += 1
            train_loss = float(loss.detach().cpu().item())
        else:
            parts = {}
            gradient_norm = 0.0
            train_loss = 0.0
        if (
            step == 1
            or step == steps
            or step % eval_interval == 0
            or step in {teacher_cut - 1, teacher_cut, teacher_cut + 1}
        ):
            metrics, _, _ = evaluate_native(
                model,
                tuning,
                tokenizer,
                device=device,
                batch_size=int(config["eval_batch_size"]),
                cohort="tuning",
                max_pairwise_cells=int(config["max_pairwise_cells"]),
            )
            field_metrics_pass = _native_field_ready(metrics, config)
            mechanism_summary: dict[str, Any] = {
                "mechanism_tuning_complete": False,
                "mechanism_tuning_evaluable": False,
                "mechanism_tuning_min_targeted_drop": 0.0,
                "mechanism_tuning_min_panel_full_accuracy": 0.0,
                "mechanism_tuning_scenarios": 0,
            }
            mechanism_pass = False
            if field_metrics_pass and step >= readiness_start:
                mechanism_pass, mechanism_summary, _ = _native_mechanism_tuning_ready(
                    model,
                    mechanism_tuning,
                    tokenizer,
                    config=config,
                    device=device,
                )
                mechanism_evaluations += 1
            readiness_pass = field_metrics_pass and mechanism_pass
            if teacher > 0.0 and step >= readiness_start:
                readiness_streak = readiness_streak + 1 if readiness_pass else 0
                if readiness_streak >= required_windows and not field_ready:
                    field_ready = True
                    field_ready_step = step
                    mechanism_ready = True
            window_pass = teacher == 0.0 and readiness_pass
            stable_streak = stable_streak + 1 if window_pass else 0
            score = _selection_score(metrics) + 0.25 * _measurement_min(metrics)
            if window_pass and score >= best_score - 1.0e-12:
                best_score = score
                selected_step = step
                selected_mechanism_min_drop = float(
                    mechanism_summary["mechanism_tuning_min_targeted_drop"]
                )
                best_state = copy.deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
            curves.append(
                {
                    "model": "native_ergt",
                    "step": step,
                    "teacher_weight": teacher,
                    "scheduled_teacher_weight": scheduled_teacher,
                    "train_loss": train_loss,
                    "gradient_norm": gradient_norm,
                    "tuning_accuracy": metrics["accuracy"],
                    "tuning_supported_accuracy": metrics["supported_accuracy"],
                    "tuning_pair_exact": metrics["counterfactual_pair_exact"],
                    "tuning_unsupported_error_rate": metrics["unsupported_error_rate"],
                    "tuning_measurement_min_accuracy": _measurement_min(metrics),
                    "tuning_effective_action_mae": metrics["effective_action_mae"],
                    "tuning_program_event_recall": metrics["program_event_recall"],
                    "tuning_program_event_precision": metrics["program_event_precision"],
                    "tuning_event_chain_exact": metrics["event_chain_exact"],
                    "tuning_path_coverage": metrics["event_world_path_coverage"],
                    "tuning_source_accuracy": metrics["source_accuracy"],
                    "tuning_candidate_exact": metrics["candidate_exact"],
                    "tuning_boundary_exact": metrics["boundary_exact"],
                    "field_metrics_pass": field_metrics_pass,
                    "mechanism_tuning_pass": mechanism_pass,
                    **mechanism_summary,
                    "readiness_pass": readiness_pass,
                    "readiness_streak": readiness_streak,
                    "field_ready": field_ready,
                    "field_ready_step": field_ready_step,
                    "stable_window_pass": window_pass,
                    "stable_streak": stable_streak,
                    "zero_teacher": teacher == 0.0,
                    **{
                        f"loss_{name}": float(value.detach().cpu().item())
                        for name, value in parts.items()
                        if isinstance(value, torch.Tensor) and value.numel() == 1
                    },
                }
            )
            print(
                f"[ergt-training] step={step} teacher={teacher:.1f} "
                f"loss={train_loss:.4f} tune={metrics['accuracy']:.3f} "
                f"pair={metrics['counterfactual_pair_exact']:.3f} "
                f"measure={_measurement_min(metrics):.3f} "
                f"actionMAE={metrics['effective_action_mae']:.3f} "
                f"events={metrics['program_event_recall']:.3f}/"
                f"{metrics['program_event_precision']:.3f} "
                f"mechanism={mechanism_pass} "
                f"minDrop={mechanism_summary['mechanism_tuning_min_targeted_drop']:.3f} "
                f"ready={field_ready}({readiness_streak}/{required_windows}) "
                f"stable={stable_streak}/{required_windows}"
            )
        training_complete = teacher == 0.0 and stable_streak >= required_windows
        training_state = {
            "sampling_rng_state": rng.getstate(),
            "readiness_streak": readiness_streak,
            "stable_streak": stable_streak,
            "field_ready": field_ready,
            "field_ready_step": field_ready_step,
            "mechanism_ready": mechanism_ready,
            "mechanism_evaluations": mechanism_evaluations,
            "selected_mechanism_min_drop": selected_mechanism_min_drop,
            "selected_step": selected_step,
            "gradient_reaches_psi": gradient_reaches_psi,
            "optimizer_steps": optimizer_steps,
            "zero_teacher_optimizer_steps": zero_teacher_optimizer_steps,
            "training_complete": training_complete,
        }
        if step and (step % max(eval_interval * 2, 100) == 0 or step == steps):
            _save_training_state(
                checkpoint,
                model=model,
                optimizer=optimizer,
                step=step,
                best_state=best_state,
                best_score=best_score,
                curves=curves,
                config=asdict(model.config),
                run_hash=run_hash,
                training_state=training_state,
            )
        if training_complete:
            break
    if steps > 0:
        _save_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=completed_step,
            best_state=best_state,
            best_score=best_score,
            curves=curves,
            config=asdict(model.config),
            run_hash=run_hash,
            training_state=training_state,
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    resource = {
        "model": "native_ergt",
        "training_seconds": elapsed,
        "steps": steps,
        "steps_completed": completed_step,
        "peak_cuda_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device)) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "gradient_reaches_geometric_field": gradient_reaches_psi if steps else None,
        "selected_zero_teacher_checkpoint": best_state is not None if steps else None,
        "selected_checkpoint_step": selected_step,
        "independent_field_readiness_pass": field_ready if steps else None,
        "independent_field_readiness_step": field_ready_step,
        "selected_mechanism_tuning_pass": mechanism_ready if steps else None,
        "selected_mechanism_min_targeted_drop": selected_mechanism_min_drop,
        "mechanism_tuning_evaluations": mechanism_evaluations,
        "stable_zero_teacher_windows": stable_streak,
        "optimizer_steps": optimizer_steps,
        "zero_teacher_optimizer_steps": zero_teacher_optimizer_steps,
        "training_complete": training_complete if steps else None,
        "selection_split": "tuning+mechanism_tuning",
        "selection_uses_heldout_interventions": False,
        "validation_evaluations_during_training": 0,
    }
    if steps == 0:
        _save_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=0,
            best_state=None,
            best_score=best_score,
            curves=curves,
            config=asdict(model.config),
            run_hash=run_hash,
            training_state={},
        )
    return curves, resource


def _save_field_candidate(
    checkpoint: Path,
    *,
    model: NativeGeometricBoundaryModel,
    run_hash: str,
    step: int,
    metrics: Mapping[str, float],
) -> str:
    path = checkpoint.with_name(f"{checkpoint.stem}_field_step_{step}.pt")
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": "ergt-geometric-frozen-field-candidate-v1",
            "run_hash": run_hash,
            "step": int(step),
            "field_metrics": dict(metrics),
            "model_state": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
        },
        temporary,
    )
    temporary.replace(path)
    return str(path)


def _train_native_decoupled_v6(
    model: NativeGeometricBoundaryModel,
    train: Sequence[MatchedTopologyExample],
    tuning: Sequence[MatchedTopologyExample],
    mechanism_tuning: Sequence[MatchedTopologyExample],
    tokenizer: TrainOnlyTokenizer,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    checkpoint: Path,
    run_hash: str,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """V21-style field freeze followed by selection on two frozen shards."""

    steps = int(config["native_steps"])
    batch_size = int(config["batch_size"])
    eval_interval = int(config["eval_interval"])
    learning_rate = float(config["native_learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_step, best_state, best_score, curves, restored = (
        0,
        None,
        float("-inf"),
        [],
        {},
    )
    if resume:
        start_step, best_state, best_score, curves, restored = _load_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            run_hash=run_hash,
            device=device,
        )
    rng = random.Random(seed + 55)
    if restored.get("sampling_rng_state") is not None:
        rng.setstate(restored["sampling_rng_state"])
    train_pad = max(len(example.tokens) for example in train)
    required_windows = int(config.get("checkpoint_stable_windows", 2))
    readiness_start = int(
        config.get(
            "native_readiness_start_step",
            max(2 * eval_interval, int(config.get("native_teacher_schedule_steps", steps)) // 4),
        )
    )
    readiness_streak = int(restored.get("readiness_streak", 0))
    stable_streak = int(restored.get("stable_streak", 0))
    field_ready = bool(restored.get("field_ready", False))
    field_ready_step = restored.get("field_ready_step")
    field_frozen_state = restored.get("field_frozen_state")
    field_candidate_paths = list(restored.get("field_candidate_paths", []))
    mechanism_ready = bool(restored.get("mechanism_ready", False))
    mechanism_evaluated = bool(restored.get("mechanism_evaluated", False))
    mechanism_summary = dict(restored.get("mechanism_summary", {}))
    selected_mechanism_min_drop = float(restored.get("selected_mechanism_min_drop", 0.0))
    selected_step = restored.get("selected_step")
    gradient_reaches_psi = bool(restored.get("gradient_reaches_psi", False))
    optimizer_steps = int(restored.get("optimizer_steps", 0))
    zero_teacher_optimizer_steps = int(restored.get("zero_teacher_optimizer_steps", 0))
    completed_step = start_step
    training_state: dict[str, Any] = dict(restored)
    training_complete = bool(restored.get("training_complete", False))
    if field_ready and field_frozen_state is not None:
        model.load_state_dict(field_frozen_state)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step_iterator = () if training_complete else range(start_step + 1, steps + 1)
    for step in step_iterator:
        completed_step = step
        teacher = 0.0 if field_ready else 1.0
        scheduled_teacher = (
            1.0 if step < int(config.get("native_teacher_schedule_steps", steps)) else 0.0
        )
        answer_weight = 0.15 if teacher > 0.0 else 1.0
        if teacher > 0.0:
            group = _grouped_sample(train, batch_size, rng)
            batch = collate_matched_topology_examples(
                group,
                tokenizer,
                pad_to_tokens=train_pad,
            ).to(device)
            model.train()
            outputs = model(**batch.model_inputs())
            loss, parts = native_geometric_training_loss(
                outputs,
                batch,
                config=model.config,
                teacher_weight=teacher,
                answer_weight=answer_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_reaches_psi = gradient_reaches_psi or any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and float(parameter.grad.abs().sum().item()) > 0.0
                for name, parameter in model.named_parameters()
                if "psi" in name or "world" in name
            )
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            optimizer_steps += 1
            train_loss = float(loss.detach().cpu().item())
        else:
            parts = {}
            gradient_norm = 0.0
            train_loss = 0.0
        should_eval = (
            step == 1
            or step == steps
            or step % eval_interval == 0
            or (field_ready and stable_streak < required_windows)
        )
        just_frozen = False
        mechanism_evaluated_now = False
        if should_eval:
            metrics, _, _ = evaluate_native(
                model,
                tuning,
                tokenizer,
                device=device,
                batch_size=int(config["eval_batch_size"]),
                cohort="tuning",
                max_pairwise_cells=int(config["max_pairwise_cells"]),
            )
            field_metrics_pass = _native_field_ready(metrics, config)
            if teacher > 0.0 and step >= readiness_start:
                readiness_streak = readiness_streak + 1 if field_metrics_pass else 0
                if field_metrics_pass:
                    candidate_path = _save_field_candidate(
                        checkpoint,
                        model=model,
                        run_hash=run_hash,
                        step=step,
                        metrics=metrics,
                    )
                    if candidate_path not in field_candidate_paths:
                        field_candidate_paths.append(candidate_path)
                if readiness_streak >= required_windows and not field_ready:
                    field_ready = True
                    field_ready_step = step
                    field_frozen_state = copy.deepcopy(
                        {name: value.detach().cpu() for name, value in model.state_dict().items()}
                    )
                    just_frozen = True
            if field_ready and teacher == 0.0 and not mechanism_evaluated:
                mechanism_ready, mechanism_summary, _ = _native_mechanism_shards_ready(
                    model,
                    mechanism_tuning,
                    tokenizer,
                    config=config,
                    device=device,
                )
                mechanism_evaluated = True
                mechanism_evaluated_now = True
                selected_mechanism_min_drop = float(
                    mechanism_summary["mechanism_tuning_min_targeted_drop"]
                )
            window_pass = field_ready and teacher == 0.0 and field_metrics_pass
            stable_streak = stable_streak + 1 if window_pass else 0
            score = _selection_score(metrics) + 0.25 * _measurement_min(metrics)
            if (
                window_pass
                and stable_streak >= required_windows
                and mechanism_ready
                and score >= best_score - 1.0e-12
            ):
                best_score = score
                selected_step = field_ready_step
                best_state = copy.deepcopy(field_frozen_state)
            default_mechanism_summary: dict[str, Any] = {
                "mechanism_tuning_complete": False,
                "mechanism_tuning_evaluable": False,
                "mechanism_tuning_min_targeted_drop": 0.0,
                "mechanism_tuning_min_panel_full_accuracy": 0.0,
                "mechanism_tuning_scenarios": 0,
                "mechanism_tuning_shards_expected": int(
                    config.get("mechanism_tuning_shard_count", 1)
                ),
                "mechanism_tuning_shards_observed": 0,
                "mechanism_tuning_shards_passed": 0,
            }
            logged_mechanism = {**default_mechanism_summary, **mechanism_summary}
            curves.append(
                {
                    "model": "native_ergt",
                    "step": step,
                    "teacher_weight": teacher,
                    "scheduled_teacher_weight": scheduled_teacher,
                    "train_loss": train_loss,
                    "gradient_norm": gradient_norm,
                    "tuning_accuracy": metrics["accuracy"],
                    "tuning_supported_accuracy": metrics["supported_accuracy"],
                    "tuning_pair_exact": metrics["counterfactual_pair_exact"],
                    "tuning_unsupported_error_rate": metrics["unsupported_error_rate"],
                    "tuning_measurement_min_accuracy": _measurement_min(metrics),
                    "tuning_effective_action_mae": metrics["effective_action_mae"],
                    "tuning_program_event_recall": metrics["program_event_recall"],
                    "tuning_program_event_precision": metrics["program_event_precision"],
                    "tuning_event_chain_exact": metrics["event_chain_exact"],
                    "tuning_path_coverage": metrics["event_world_path_coverage"],
                    "tuning_source_accuracy": metrics["source_accuracy"],
                    "tuning_candidate_exact": metrics["candidate_exact"],
                    "tuning_boundary_exact": metrics["boundary_exact"],
                    "field_metrics_pass": field_metrics_pass,
                    "mechanism_tuning_pass": mechanism_ready,
                    **{
                        key: value
                        for key, value in logged_mechanism.items()
                        if key != "mechanism_tuning_shard_summaries"
                    },
                    "readiness_pass": field_metrics_pass,
                    "readiness_streak": readiness_streak,
                    "field_ready": field_ready,
                    "field_ready_step": field_ready_step,
                    "stable_window_pass": window_pass,
                    "stable_streak": stable_streak,
                    "zero_teacher": teacher == 0.0,
                    "field_frozen_before_mechanism_selection": field_ready,
                    **{
                        f"loss_{name}": float(value.detach().cpu().item())
                        for name, value in parts.items()
                        if isinstance(value, torch.Tensor) and value.numel() == 1
                    },
                }
            )
            print(
                f"[ergt-training] step={step} teacher={teacher:.1f} "
                f"loss={train_loss:.4f} tune={metrics['accuracy']:.3f} "
                f"pair={metrics['counterfactual_pair_exact']:.3f} "
                f"measure={_measurement_min(metrics):.3f} "
                f"actionMAE={metrics['effective_action_mae']:.3f} "
                f"events={metrics['program_event_recall']:.3f}/"
                f"{metrics['program_event_precision']:.3f} "
                f"field={field_metrics_pass} ready={field_ready}"
                f"({readiness_streak}/{required_windows}) "
                f"mechanism={mechanism_ready} shards="
                f"{logged_mechanism['mechanism_tuning_shards_passed']}/"
                f"{logged_mechanism['mechanism_tuning_shards_expected']} "
                f"stable={stable_streak}/{required_windows}"
            )
        training_complete = field_ready and stable_streak >= required_windows
        training_state = {
            "sampling_rng_state": rng.getstate(),
            "readiness_streak": readiness_streak,
            "stable_streak": stable_streak,
            "field_ready": field_ready,
            "field_ready_step": field_ready_step,
            "field_frozen_state": field_frozen_state,
            "field_candidate_paths": field_candidate_paths,
            "mechanism_ready": mechanism_ready,
            "mechanism_evaluated": mechanism_evaluated,
            "mechanism_summary": mechanism_summary,
            "selected_mechanism_min_drop": selected_mechanism_min_drop,
            "selected_step": selected_step,
            "gradient_reaches_psi": gradient_reaches_psi,
            "optimizer_steps": optimizer_steps,
            "zero_teacher_optimizer_steps": zero_teacher_optimizer_steps,
            "training_complete": training_complete,
        }
        if step and (
            step % max(eval_interval * 2, 100) == 0
            or step == steps
            or just_frozen
            or mechanism_evaluated_now
            or training_complete
        ):
            _save_training_state(
                checkpoint,
                model=model,
                optimizer=optimizer,
                step=step,
                best_state=best_state,
                best_score=best_score,
                curves=curves,
                config=asdict(model.config),
                run_hash=run_hash,
                training_state=training_state,
            )
        if training_complete:
            break
    if steps > 0:
        _save_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=completed_step,
            best_state=best_state,
            best_score=best_score,
            curves=curves,
            config=asdict(model.config),
            run_hash=run_hash,
            training_state=training_state,
        )
    evaluation_state_status = "untrained_smoke"
    if best_state is not None:
        model.load_state_dict(best_state)
        evaluation_state_status = "selected_frozen_field_and_mechanism_checkpoint"
    elif field_frozen_state is not None:
        model.load_state_dict(field_frozen_state)
        evaluation_state_status = "unselected_frozen_field_diagnostic_only"
    elif steps > 0:
        evaluation_state_status = "unselected_final_training_state_diagnostic_only"
    elapsed = time.perf_counter() - started
    resource = {
        "model": "native_ergt",
        "training_seconds": elapsed,
        "steps": steps,
        "steps_completed": completed_step,
        "peak_cuda_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device)) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "gradient_reaches_geometric_field": gradient_reaches_psi if steps else None,
        "selected_zero_teacher_checkpoint": best_state is not None if steps else None,
        "selected_checkpoint_step": selected_step,
        "independent_field_readiness_pass": field_ready if steps else None,
        "independent_field_readiness_step": field_ready_step,
        "selected_mechanism_tuning_pass": mechanism_ready if steps else None,
        "selected_mechanism_min_targeted_drop": selected_mechanism_min_drop,
        "mechanism_tuning_evaluations": int(mechanism_evaluated),
        "mechanism_tuning_shards_expected": int(config.get("mechanism_tuning_shard_count", 1)),
        "mechanism_tuning_shards_passed": int(
            mechanism_summary.get("mechanism_tuning_shards_passed", 0)
        ),
        "stable_zero_teacher_windows": stable_streak,
        "optimizer_steps": optimizer_steps,
        "zero_teacher_optimizer_steps": zero_teacher_optimizer_steps,
        "training_complete": training_complete if steps else None,
        "field_candidate_checkpoint_count": len(field_candidate_paths),
        "field_candidate_checkpoint_paths": json.dumps(field_candidate_paths),
        "evaluation_state_status": evaluation_state_status,
        "selection_split": "tuning+two_frozen_mechanism_shards",
        "selection_uses_heldout_interventions": False,
        "validation_evaluations_during_training": 0,
        "mechanism_controls_teacher": False,
        "teacher_stops_immediately_on_field_readiness": True,
    }
    if steps == 0:
        _save_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=0,
            best_state=None,
            best_score=best_score,
            curves=curves,
            config=asdict(model.config),
            run_hash=run_hash,
            training_state={},
        )
    return curves, resource


def _train_transformer(
    model: DirectTransformer,
    train: Sequence[MatchedTopologyExample],
    tuning: Sequence[MatchedTopologyExample],
    tokenizer: TrainOnlyTokenizer,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    checkpoint: Path,
    run_hash: str,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    steps = int(config["transformer_steps"])
    batch_size = int(config["batch_size"])
    eval_interval = int(config["eval_interval"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["transformer_learning_rate"]))
    start_step, best_state, best_score, curves, restored = (
        0,
        None,
        float("-inf"),
        [],
        {},
    )
    if resume:
        start_step, best_state, best_score, curves, restored = _load_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            run_hash=run_hash,
            device=device,
        )
    rng = random.Random(seed + 55)
    if restored.get("sampling_rng_state") is not None:
        rng.setstate(restored["sampling_rng_state"])
    train_pad = max(len(example.tokens) for example in train)
    stable_streak = int(restored.get("stable_streak", 0))
    selected_step = restored.get("selected_step")
    optimizer_steps = int(restored.get("optimizer_steps", 0))
    required_windows = int(config.get("checkpoint_stable_windows", 2))
    completed_step = start_step
    training_state: dict[str, Any] = dict(restored)
    training_complete = bool(restored.get("training_complete", False)) or (
        best_state is not None and stable_streak >= required_windows
    )
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step_iterator = () if training_complete else range(start_step + 1, steps + 1)
    for step in step_iterator:
        completed_step = step
        group = _grouped_sample(train, batch_size, rng)
        batch = collate_matched_topology_examples(group, tokenizer, pad_to_tokens=train_pad).to(
            device
        )
        model.train()
        outputs = model.forward_with_auxiliary(**batch.base.model_inputs())
        loss, parts = _transformer_training_loss(outputs, batch, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        optimizer_steps += 1
        if step == 1 or step == steps or step % eval_interval == 0:
            metrics, _ = evaluate_transformer(
                model,
                tuning,
                tokenizer,
                device=device,
                batch_size=int(config["eval_batch_size"]),
                cohort="tuning",
            )
            score = _selection_score(metrics)
            window_pass = (
                float(metrics["supported_accuracy"])
                >= float(config["transformer_tuning_accuracy_floor"])
                and float(metrics["counterfactual_pair_exact"])
                >= float(config["transformer_tuning_pair_floor"])
                and float(metrics["unsupported_error_rate"])
                <= float(config["transformer_tuning_unsupported_error_ceiling"])
            )
            stable_streak = stable_streak + 1 if window_pass else 0
            if score >= best_score - 1.0e-12:
                best_score = score
                selected_step = step
                best_state = copy.deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
            curves.append(
                {
                    "model": "direct_transformer",
                    "step": step,
                    "teacher_weight": 0.0,
                    "train_loss": float(loss.detach().cpu().item()),
                    "gradient_norm": gradient_norm,
                    "tuning_accuracy": metrics["accuracy"],
                    "tuning_supported_accuracy": metrics["supported_accuracy"],
                    "tuning_pair_exact": metrics["counterfactual_pair_exact"],
                    "tuning_unsupported_error_rate": metrics["unsupported_error_rate"],
                    "stable_window_pass": window_pass,
                    "stable_streak": stable_streak,
                    "zero_teacher": True,
                    **{
                        f"loss_{name}": float(value.detach().cpu().item())
                        for name, value in parts.items()
                    },
                }
            )
            print(
                f"[transformer-training] step={step} loss={float(loss.detach().cpu()):.4f} "
                f"tune={metrics['accuracy']:.3f} pair={metrics['counterfactual_pair_exact']:.3f} "
                f"unsupportedErr={metrics['unsupported_error_rate']:.3f} "
                f"answerLoss={float(parts['answer'].detach().cpu()):.3f} "
                f"auxLoss={float(parts['auxiliary'].detach().cpu()):.3f} "
                f"stable={stable_streak}/{required_windows}"
            )
        training_complete = stable_streak >= required_windows
        training_state = {
            "sampling_rng_state": rng.getstate(),
            "stable_streak": stable_streak,
            "selected_step": selected_step,
            "optimizer_steps": optimizer_steps,
            "training_complete": training_complete,
        }
        if step and (step % max(eval_interval * 2, 100) == 0 or step == steps):
            _save_training_state(
                checkpoint,
                model=model,
                optimizer=optimizer,
                step=step,
                best_state=best_state,
                best_score=best_score,
                curves=curves,
                config=asdict(model.config),
                run_hash=run_hash,
                training_state=training_state,
            )
        if training_complete:
            break
    if steps > 0:
        _save_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=completed_step,
            best_state=best_state,
            best_score=best_score,
            curves=curves,
            config=asdict(model.config),
            run_hash=run_hash,
            training_state=training_state,
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    resource = {
        "model": "direct_transformer",
        "training_seconds": elapsed,
        "steps": steps,
        "steps_completed": completed_step,
        "peak_cuda_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device)) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "selected_tuning_checkpoint": best_state is not None if steps else None,
        "selected_checkpoint_step": selected_step,
        "stable_tuning_windows": stable_streak,
        "optimizer_steps": optimizer_steps,
        "training_complete": training_complete if steps else None,
        "selection_split": "tuning",
        "validation_evaluations_during_training": 0,
        "training_label_families": (
            "answer,entity,query_role,relation_anchor,event_source,event_target,"
            "boundary_value,action,cone,transport,boundary_deficit,terminal_mass"
        ),
        "auxiliary_outputs_used_at_inference": False,
        "compiler_or_executor_used": False,
    }
    if steps == 0:
        _save_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=0,
            best_state=None,
            best_score=best_score,
            curves=curves,
            config=asdict(model.config),
            run_hash=run_hash,
            training_state={},
        )
    return curves, resource


def _pair_groups_by_hop(
    examples: Sequence[MatchedTopologyExample],
) -> dict[int, tuple[tuple[MatchedTopologyExample, ...], ...]]:
    grouped: dict[str, list[MatchedTopologyExample]] = defaultdict(list)
    for example in examples:
        grouped[example.pair_id].append(example)
    by_hop: dict[int, list[tuple[MatchedTopologyExample, ...]]] = defaultdict(list)
    for values in grouped.values():
        hops = {int(example.base.metadata["path_hops"]) for example in values}
        if len(hops) != 1:
            raise ValueError("counterfactual pair spans multiple hop strata")
        by_hop[next(iter(hops))].append(tuple(values))
    return {hop: tuple(values) for hop, values in by_hop.items()}


def _stratified_curriculum_sample(
    examples_by_hop: Mapping[int, Sequence[Sequence[MatchedTopologyExample]]],
    batch_size: int,
    rng: random.Random,
) -> tuple[MatchedTopologyExample, ...]:
    """Sample complete pairs while anchoring every mixed batch at one hop."""

    if batch_size < 2 or batch_size % 2:
        raise ValueError("V7 curriculum batch_size must be a positive even number")
    one_hop = tuple(examples_by_hop.get(1, ()))
    short = tuple(group for hop in (2, 3, 4) for group in examples_by_hop.get(hop, ()))
    long = tuple(group for hop in (5, 6, 7, 8) for group in examples_by_hop.get(hop, ()))
    all_groups = tuple(group for groups in examples_by_hop.values() for group in groups)
    if not one_hop or not short or not long:
        raise ValueError("V7 curriculum requires nonempty 1, 2-4, and 5-8 hop strata")

    pair_slots = batch_size // 2
    strata = (one_hop, short, long)
    selected: list[Sequence[MatchedTopologyExample]] = []
    selected_ids: set[str] = set()
    for slot in range(pair_slots):
        pool = strata[slot] if slot < len(strata) else all_groups
        available = [group for group in pool if group[0].pair_id not in selected_ids]
        if not available:
            available = [group for group in all_groups if group[0].pair_id not in selected_ids]
        if not available:
            raise ValueError("not enough distinct counterfactual pairs for one batch")
        group = rng.choice(available)
        selected.append(group)
        selected_ids.add(group[0].pair_id)
    batch = [example for group in selected for example in group]
    rng.shuffle(batch)
    return tuple(batch)


def _balanced_hop_curriculum_sample(
    examples_by_hop: Mapping[int, Sequence[Sequence[MatchedTopologyExample]]],
    *,
    max_hop: int,
    batch_size: int,
    rng: random.Random,
) -> tuple[MatchedTopologyExample, ...]:
    """Sample complete pairs with deterministic coverage of every active hop.

    At the registered V8 batch size of 16, the final 1--8-hop stage contains
    exactly one counterfactual pair from each hop. Earlier stages distribute
    the same eight pair slots as evenly as possible across their active hops.
    """

    if batch_size < 2 or batch_size % 2:
        raise ValueError("V8 curriculum batch_size must be a positive even number")
    if max_hop < 1:
        raise ValueError("V8 curriculum max_hop must be positive")
    active_hops = tuple(range(1, max_hop + 1))
    missing = [hop for hop in active_hops if not examples_by_hop.get(hop)]
    if missing:
        raise ValueError(f"V8 curriculum has no examples for hops {missing}")
    pair_slots = batch_size // 2
    if pair_slots < len(active_hops):
        raise ValueError(
            "V8 curriculum needs at least one complete pair per active hop: "
            f"pair_slots={pair_slots}, active_hops={len(active_hops)}"
        )

    hop_schedule = [active_hops[slot % len(active_hops)] for slot in range(pair_slots)]
    selected: list[Sequence[MatchedTopologyExample]] = []
    selected_ids: set[str] = set()
    for hop in hop_schedule:
        available = [
            group for group in examples_by_hop[hop] if group[0].pair_id not in selected_ids
        ]
        if not available:
            raise ValueError(f"not enough distinct hop-{hop} counterfactual pairs for one V8 batch")
        group = rng.choice(available)
        selected.append(group)
        selected_ids.add(group[0].pair_id)
    batch = [example for group in selected for example in group]
    rng.shuffle(batch)
    return tuple(batch)


def _transformer_candidate_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Any,
    config: Mapping[str, Any],
    loss_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_mode == "matched_auxiliary":
        return _transformer_training_loss(outputs, batch, config)
    if loss_mode != "answer_only":
        raise ValueError(f"unknown Transformer loss mode: {loss_mode}")
    answer_class_weight = outputs["answer_logits"].new_tensor((1.0, 1.0, 1.25))
    answer_loss = F.cross_entropy(
        outputs["answer_logits"],
        batch.base.supervision.answer_labels,
        weight=answer_class_weight,
    )
    zero = answer_loss.new_tensor(0.0)
    return answer_loss, {
        "answer": answer_loss,
        "auxiliary": zero,
        "pair_equivariance": zero,
    }


def train_transformer_curriculum_candidate(
    model: DirectTransformer,
    train: Sequence[MatchedTopologyExample],
    tuning: Sequence[MatchedTopologyExample],
    development_tuning: Sequence[MatchedTopologyExample],
    tokenizer: TrainOnlyTokenizer,
    *,
    config: Mapping[str, Any],
    recipe: Mapping[str, Any],
    device: torch.device,
    seed: int,
    checkpoint: Path,
    run_hash: str,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Qualify a direct Transformer without consulting final OOD panels.

    Stage one establishes one-hop ID convergence. Stage two trains on the
    registered 1--8-hop mixture with one-hop replay in every stratified batch.
    Only checkpoints retaining two consecutive ID windows are eligible, and
    the best eligible checkpoint is chosen on the disjoint 6/8-hop development
    cohort. No compiler, executor, geometry, or final 12--32-hop panel enters
    this loop.
    """

    candidate_id = str(recipe["candidate_id"])
    loss_mode = str(recipe["loss_mode"])
    warmup_steps = int(recipe["warmup_steps"])
    mixed_steps = int(recipe["mixed_steps"])
    eval_interval = int(recipe.get("eval_interval", config["eval_interval"]))
    batch_size = int(recipe.get("batch_size", config["batch_size"]))
    base_lr = float(recipe["learning_rate"])
    mixed_lr = base_lr * float(recipe.get("mixed_learning_rate_scale", 0.35))
    required_windows = int(config.get("checkpoint_stable_windows", 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    start_step, best_state, best_score, curves, restored = (
        0,
        None,
        float("-inf"),
        [],
        {},
    )
    if resume:
        start_step, best_state, best_score, curves, restored = _load_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            run_hash=run_hash,
            device=device,
        )

    rng = random.Random(seed + int(recipe.get("seed_offset", 0)) + 701)
    if restored.get("sampling_rng_state") is not None:
        rng.setstate(restored["sampling_rng_state"])
    by_hop = _pair_groups_by_hop(train)
    warmup_examples = tuple(example for group in by_hop.get(1, ()) for example in group)
    if not warmup_examples:
        raise ValueError("V7 baseline qualification requires one-hop training examples")
    train_pad = max(len(example.tokens) for example in train)
    stage = str(restored.get("stage", "warmup"))
    stage_step = int(restored.get("stage_step", start_step if stage == "warmup" else 0))
    warmup_streak = int(restored.get("warmup_streak", 0))
    mixed_streak = int(restored.get("mixed_streak", 0))
    selected_step = restored.get("selected_step")
    selected_development_score = float(restored.get("selected_development_score", best_score))
    selected_stable_windows = int(restored.get("selected_stable_windows", 0))
    optimizer_steps = int(restored.get("optimizer_steps", 0))
    warmup_complete = bool(restored.get("warmup_complete", False))
    training_complete = bool(restored.get("training_complete", False))
    completed_total_step = int(restored.get("completed_total_step", start_step))
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def id_window_pass(metrics: Mapping[str, float]) -> bool:
        return (
            float(metrics["supported_accuracy"])
            >= float(config["transformer_tuning_accuracy_floor"])
            and float(metrics["counterfactual_pair_exact"])
            >= float(config["transformer_tuning_pair_floor"])
            and float(metrics["unsupported_error_rate"])
            <= float(config["transformer_tuning_unsupported_error_ceiling"])
        )

    def save_state(current_stage: str, current_stage_step: int) -> None:
        training_state = {
            "sampling_rng_state": rng.getstate(),
            "stage": current_stage,
            "stage_step": current_stage_step,
            "warmup_streak": warmup_streak,
            "mixed_streak": mixed_streak,
            "selected_step": selected_step,
            "selected_development_score": selected_development_score,
            "selected_stable_windows": selected_stable_windows,
            "optimizer_steps": optimizer_steps,
            "warmup_complete": warmup_complete,
            "training_complete": training_complete,
            "completed_total_step": completed_total_step,
            "candidate_id": candidate_id,
            "loss_mode": loss_mode,
        }
        _save_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=completed_total_step,
            best_state=best_state,
            best_score=selected_development_score,
            curves=curves,
            config=asdict(model.config),
            run_hash=run_hash,
            training_state=training_state,
        )

    if not training_complete and stage == "warmup":
        for current in range(stage_step + 1, warmup_steps + 1):
            group = _grouped_sample(warmup_examples, batch_size, rng)
            batch = collate_matched_topology_examples(group, tokenizer, pad_to_tokens=train_pad).to(
                device
            )
            model.train()
            outputs = model.forward_with_auxiliary(**batch.base.model_inputs())
            loss, parts = _transformer_candidate_loss(outputs, batch, config, loss_mode)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            optimizer_steps += 1
            completed_total_step += 1
            stage_step = current
            if current == 1 or current == warmup_steps or current % eval_interval == 0:
                id_metrics, _ = evaluate_transformer(
                    model,
                    tuning,
                    tokenizer,
                    device=device,
                    batch_size=int(config["eval_batch_size"]),
                    cohort="baseline_id_tuning",
                )
                window = id_window_pass(id_metrics)
                warmup_streak = warmup_streak + 1 if window else 0
                curves.append(
                    {
                        "model": "direct_transformer",
                        "candidate_id": candidate_id,
                        "stage": "warmup_1hop",
                        "step": current,
                        "total_step": completed_total_step,
                        "train_loss": float(loss.detach().cpu()),
                        "gradient_norm": gradient_norm,
                        "tuning_accuracy": id_metrics["accuracy"],
                        "tuning_supported_accuracy": id_metrics["supported_accuracy"],
                        "tuning_pair_exact": id_metrics["counterfactual_pair_exact"],
                        "tuning_unsupported_error_rate": id_metrics["unsupported_error_rate"],
                        "stable_window_pass": window,
                        "stable_streak": warmup_streak,
                        **{
                            f"loss_{name}": float(value.detach().cpu())
                            for name, value in parts.items()
                        },
                    }
                )
                print(
                    f"[transformer-qualification] candidate={candidate_id} "
                    f"stage=warmup step={current} loss={float(loss.detach().cpu()):.4f} "
                    f"id={id_metrics['supported_accuracy']:.3f} "
                    f"pair={id_metrics['counterfactual_pair_exact']:.3f} "
                    f"stable={warmup_streak}/{required_windows}"
                )
                save_state("warmup", current)
                if warmup_streak >= required_windows:
                    warmup_complete = True
                    break
        if not warmup_complete:
            training_complete = True
            save_state("failed_warmup", stage_step)
        else:
            stage = "mixed"
            stage_step = 0
            mixed_streak = 0
            for group in optimizer.param_groups:
                group["lr"] = mixed_lr
            save_state(stage, stage_step)

    if not training_complete and stage == "mixed":
        for group in optimizer.param_groups:
            group["lr"] = mixed_lr
        for current in range(stage_step + 1, mixed_steps + 1):
            examples = _stratified_curriculum_sample(by_hop, batch_size, rng)
            batch = collate_matched_topology_examples(
                examples, tokenizer, pad_to_tokens=train_pad
            ).to(device)
            model.train()
            outputs = model.forward_with_auxiliary(**batch.base.model_inputs())
            loss, parts = _transformer_candidate_loss(outputs, batch, config, loss_mode)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            optimizer_steps += 1
            completed_total_step += 1
            stage_step = current
            if current == 1 or current == mixed_steps or current % eval_interval == 0:
                id_metrics, _ = evaluate_transformer(
                    model,
                    tuning,
                    tokenizer,
                    device=device,
                    batch_size=int(config["eval_batch_size"]),
                    cohort="baseline_id_tuning",
                )
                development_metrics, _ = evaluate_transformer(
                    model,
                    development_tuning,
                    tokenizer,
                    device=device,
                    batch_size=int(config["eval_batch_size"]),
                    cohort="baseline_6_8_development",
                )
                window = id_window_pass(id_metrics)
                mixed_streak = mixed_streak + 1 if window else 0
                development_score = float(development_metrics["supported_accuracy"]) + float(
                    development_metrics["counterfactual_pair_exact"]
                )
                eligible = mixed_streak >= required_windows
                if eligible and development_score > selected_development_score + 1.0e-12:
                    selected_development_score = development_score
                    best_score = development_score
                    selected_step = current
                    selected_stable_windows = mixed_streak
                    best_state = copy.deepcopy(
                        {name: value.detach().cpu() for name, value in model.state_dict().items()}
                    )
                curves.append(
                    {
                        "model": "direct_transformer",
                        "candidate_id": candidate_id,
                        "stage": "mixed_1_8",
                        "step": current,
                        "total_step": completed_total_step,
                        "train_loss": float(loss.detach().cpu()),
                        "gradient_norm": gradient_norm,
                        "tuning_accuracy": id_metrics["accuracy"],
                        "tuning_supported_accuracy": id_metrics["supported_accuracy"],
                        "tuning_pair_exact": id_metrics["counterfactual_pair_exact"],
                        "tuning_unsupported_error_rate": id_metrics["unsupported_error_rate"],
                        "development_accuracy": development_metrics["accuracy"],
                        "development_supported_accuracy": development_metrics["supported_accuracy"],
                        "development_pair_exact": development_metrics["counterfactual_pair_exact"],
                        "stable_window_pass": window,
                        "stable_streak": mixed_streak,
                        "eligible_checkpoint": eligible,
                        "development_selection_score": development_score,
                        **{
                            f"loss_{name}": float(value.detach().cpu())
                            for name, value in parts.items()
                        },
                    }
                )
                print(
                    f"[transformer-qualification] candidate={candidate_id} "
                    f"stage=mixed step={current} loss={float(loss.detach().cpu()):.4f} "
                    f"id={id_metrics['supported_accuracy']:.3f} "
                    f"pair={id_metrics['counterfactual_pair_exact']:.3f} "
                    f"dev={development_metrics['supported_accuracy']:.3f}/"
                    f"{development_metrics['counterfactual_pair_exact']:.3f} "
                    f"eligible={eligible}"
                )
                save_state("mixed", current)
        training_complete = True
        save_state("complete", stage_step)

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    selected = best_state is not None
    resource = {
        "model": "direct_transformer",
        "candidate_id": candidate_id,
        "loss_mode": loss_mode,
        "training_seconds": elapsed,
        "steps": warmup_steps + mixed_steps,
        "steps_completed": completed_total_step,
        "warmup_steps_completed": min(stage_step, warmup_steps)
        if stage.startswith("warmup")
        else warmup_steps,
        "mixed_steps_completed": stage_step if stage in {"mixed", "complete"} else 0,
        "peak_cuda_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device)) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "selected_tuning_checkpoint": selected,
        "selected_checkpoint_step": selected_step,
        "selected_development_score": selected_development_score if selected else None,
        "stable_tuning_windows": selected_stable_windows if selected else 0,
        "optimizer_steps": optimizer_steps,
        "training_complete": training_complete,
        "warmup_convergence_pass": warmup_complete,
        "selection_split": "one_hop_tuning+bounded_6_8_development",
        "validation_evaluations_during_training": 0,
        "final_12_32_hop_evaluations_during_training": 0,
        "training_label_families": (
            "answer"
            if loss_mode == "answer_only"
            else (
                "answer,entity,query_role,relation_anchor,event_source,event_target,"
                "boundary_value,action,cone,transport,boundary_deficit,terminal_mass"
            )
        ),
        "matched_training_label_families": loss_mode == "matched_auxiliary",
        "auxiliary_outputs_used_at_inference": False,
        "compiler_or_executor_used": False,
        "curriculum": "one_hop_warmup_then_stratified_1_to_8_with_replay",
    }
    return curves, resource


def train_transformer_staged_curriculum_candidate(
    model: DirectTransformer,
    train: Sequence[MatchedTopologyExample],
    tuning: Sequence[MatchedTopologyExample],
    development_tuning: Sequence[MatchedTopologyExample],
    tokenizer: TrainOnlyTokenizer,
    *,
    config: Mapping[str, Any],
    recipe: Mapping[str, Any],
    device: torch.device,
    seed: int,
    checkpoint: Path,
    run_hash: str,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train the V8 direct baseline with exact staged hop coverage.

    One-hop tuning determines checkpoint eligibility. The disjoint 6/8-hop
    development cohort only ranks eligible checkpoints; it is deliberately
    not a minimum-performance gate. Final 12--32-hop panels are never read.
    """

    candidate_id = str(recipe["candidate_id"])
    loss_mode = str(recipe["loss_mode"])
    if loss_mode != "matched_auxiliary":
        raise ValueError("V8 staged qualification requires matched auxiliary supervision")
    warmup_steps = int(recipe["warmup_steps"])
    stages = tuple(dict(stage) for stage in recipe["curriculum_stages"])
    if [int(stage["max_hop"]) for stage in stages] != [2, 4, 8]:
        raise ValueError("V8 curriculum stages must be registered as max hops 2, 4, and 8")
    eval_interval = int(recipe.get("eval_interval", config["eval_interval"]))
    batch_size = int(recipe.get("batch_size", 16))
    if batch_size != 16:
        raise ValueError("V8 exact 1--8 hop coverage is registered at batch_size=16")
    warmup_batch_size = int(recipe.get("warmup_batch_size", 8))
    if warmup_batch_size < 2 or warmup_batch_size % 2:
        raise ValueError("V8.1 warmup_batch_size must be a positive even number")
    warmup_sampling_seed_offset = int(recipe.get("warmup_sampling_seed_offset", 701))
    base_lr = float(recipe["learning_rate"])
    required_windows = int(config.get("checkpoint_stable_windows", 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    start_step, best_state, best_score, curves, restored = (
        0,
        None,
        float("-inf"),
        [],
        {},
    )
    if resume:
        start_step, best_state, best_score, curves, restored = _load_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            run_hash=run_hash,
            device=device,
        )

    rng = random.Random(seed + int(recipe.get("seed_offset", 0)) + warmup_sampling_seed_offset)
    if restored.get("sampling_rng_state") is not None:
        rng.setstate(restored["sampling_rng_state"])
    by_hop = _pair_groups_by_hop(train)
    warmup_examples = tuple(example for group in by_hop.get(1, ()) for example in group)
    if not warmup_examples:
        raise ValueError("V8.1 baseline qualification requires one-hop training examples")
    train_pad = max(len(example.tokens) for example in train)
    phase = str(restored.get("phase", "warmup"))
    phase_step = int(restored.get("phase_step", start_step if phase == "warmup" else 0))
    curriculum_stage_index = int(restored.get("curriculum_stage_index", 0))
    warmup_streak = int(restored.get("warmup_streak", 0))
    id_streak = int(restored.get("id_streak", 0))
    selected_step = restored.get("selected_step")
    selected_stage = restored.get("selected_stage")
    selected_development_score = float(restored.get("selected_development_score", best_score))
    selected_stable_windows = int(restored.get("selected_stable_windows", 0))
    optimizer_steps = int(restored.get("optimizer_steps", 0))
    warmup_steps_completed = int(restored.get("warmup_steps_completed", 0))
    warmup_complete = bool(restored.get("warmup_complete", False))
    run_terminated = bool(restored.get("run_terminated", restored.get("training_complete", False)))
    completed_total_step = int(restored.get("completed_total_step", start_step))
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def id_window_pass(metrics: Mapping[str, float]) -> bool:
        return (
            float(metrics["supported_accuracy"])
            >= float(config["transformer_tuning_accuracy_floor"])
            and float(metrics["counterfactual_pair_exact"])
            >= float(config["transformer_tuning_pair_floor"])
            and float(metrics["unsupported_error_rate"])
            <= float(config["transformer_tuning_unsupported_error_ceiling"])
        )

    def save_state(current_phase: str, current_phase_step: int) -> None:
        training_state = {
            "sampling_rng_state": rng.getstate(),
            "phase": current_phase,
            "phase_step": current_phase_step,
            "curriculum_stage_index": curriculum_stage_index,
            "warmup_streak": warmup_streak,
            "id_streak": id_streak,
            "selected_step": selected_step,
            "selected_stage": selected_stage,
            "selected_development_score": selected_development_score,
            "selected_stable_windows": selected_stable_windows,
            "optimizer_steps": optimizer_steps,
            "warmup_steps_completed": warmup_steps_completed,
            "warmup_complete": warmup_complete,
            "run_terminated": run_terminated,
            "completed_total_step": completed_total_step,
            "candidate_id": candidate_id,
            "loss_mode": loss_mode,
        }
        _save_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=completed_total_step,
            best_state=best_state,
            best_score=selected_development_score,
            curves=curves,
            config=asdict(model.config),
            run_hash=run_hash,
            training_state=training_state,
        )

    if not run_terminated and phase == "warmup":
        for current in range(phase_step + 1, warmup_steps + 1):
            # Preserve the successful V7 one-hop optimization contract exactly.
            # V8's larger exact-coverage batch is introduced only after ID
            # convergence, when the staged multi-hop curriculum begins.
            examples = _grouped_sample(warmup_examples, warmup_batch_size, rng)
            batch = collate_matched_topology_examples(
                examples, tokenizer, pad_to_tokens=train_pad
            ).to(device)
            model.train()
            outputs = model.forward_with_auxiliary(**batch.base.model_inputs())
            loss, parts = _transformer_candidate_loss(outputs, batch, config, loss_mode)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            optimizer_steps += 1
            completed_total_step += 1
            phase_step = current
            warmup_steps_completed = current
            if current == 1 or current == warmup_steps or current % eval_interval == 0:
                id_metrics, _ = evaluate_transformer(
                    model,
                    tuning,
                    tokenizer,
                    device=device,
                    batch_size=int(config["eval_batch_size"]),
                    cohort="baseline_id_tuning",
                )
                window = id_window_pass(id_metrics)
                warmup_streak = warmup_streak + 1 if window else 0
                curves.append(
                    {
                        "model": "direct_transformer",
                        "candidate_id": candidate_id,
                        "stage": "warmup_1hop",
                        "stage_max_hop": 1,
                        "step": current,
                        "total_step": completed_total_step,
                        "train_loss": float(loss.detach().cpu()),
                        "gradient_norm": gradient_norm,
                        "tuning_accuracy": id_metrics["accuracy"],
                        "tuning_supported_accuracy": id_metrics["supported_accuracy"],
                        "tuning_pair_exact": id_metrics["counterfactual_pair_exact"],
                        "tuning_unsupported_error_rate": id_metrics["unsupported_error_rate"],
                        "stable_window_pass": window,
                        "stable_streak": warmup_streak,
                        "eligible_checkpoint": False,
                        **{
                            f"loss_{name}": float(value.detach().cpu())
                            for name, value in parts.items()
                        },
                    }
                )
                print(
                    f"[transformer-training] candidate={candidate_id} "
                    f"stage=warmup step={current} loss={float(loss.detach().cpu()):.4f} "
                    f"id={id_metrics['supported_accuracy']:.3f} "
                    f"pair={id_metrics['counterfactual_pair_exact']:.3f} "
                    f"stable={warmup_streak}/{required_windows}"
                )
                save_state("warmup", current)
                if warmup_streak >= required_windows:
                    warmup_complete = True
                    break
        if not warmup_complete:
            run_terminated = True
            phase = "failed_warmup"
            save_state(phase, phase_step)
        else:
            phase = "curriculum"
            phase_step = 0
            curriculum_stage_index = 0
            id_streak = 0
            save_state(phase, phase_step)

    if not run_terminated and phase == "curriculum":
        for stage_index in range(curriculum_stage_index, len(stages)):
            curriculum_stage_index = stage_index
            stage = stages[stage_index]
            stage_name = str(stage["name"])
            stage_max_hop = int(stage["max_hop"])
            stage_steps = int(stage["steps"])
            stage_lr = base_lr * float(stage["learning_rate_scale"])
            for group in optimizer.param_groups:
                group["lr"] = stage_lr
            first_step = (
                phase_step + 1
                if stage_index == int(restored.get("curriculum_stage_index", 0))
                else 1
            )
            for current in range(first_step, stage_steps + 1):
                examples = _balanced_hop_curriculum_sample(
                    by_hop,
                    max_hop=stage_max_hop,
                    batch_size=batch_size,
                    rng=rng,
                )
                batch = collate_matched_topology_examples(
                    examples, tokenizer, pad_to_tokens=train_pad
                ).to(device)
                model.train()
                outputs = model.forward_with_auxiliary(**batch.base.model_inputs())
                loss, parts = _transformer_candidate_loss(outputs, batch, config, loss_mode)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                optimizer.step()
                optimizer_steps += 1
                completed_total_step += 1
                phase_step = current
                if current == 1 or current == stage_steps or current % eval_interval == 0:
                    id_metrics, _ = evaluate_transformer(
                        model,
                        tuning,
                        tokenizer,
                        device=device,
                        batch_size=int(config["eval_batch_size"]),
                        cohort="baseline_id_tuning",
                    )
                    development_metrics, _ = evaluate_transformer(
                        model,
                        development_tuning,
                        tokenizer,
                        device=device,
                        batch_size=int(config["eval_batch_size"]),
                        cohort="baseline_6_8_development",
                    )
                    window = id_window_pass(id_metrics)
                    id_streak = id_streak + 1 if window else 0
                    development_score = float(development_metrics["supported_accuracy"]) + float(
                        development_metrics["counterfactual_pair_exact"]
                    )
                    eligible = id_streak >= required_windows
                    if eligible and development_score > selected_development_score + 1.0e-12:
                        selected_development_score = development_score
                        best_score = development_score
                        selected_step = completed_total_step
                        selected_stage = stage_name
                        selected_stable_windows = id_streak
                        best_state = copy.deepcopy(
                            {
                                name: value.detach().cpu()
                                for name, value in model.state_dict().items()
                            }
                        )
                    curves.append(
                        {
                            "model": "direct_transformer",
                            "candidate_id": candidate_id,
                            "stage": stage_name,
                            "stage_max_hop": stage_max_hop,
                            "step": current,
                            "total_step": completed_total_step,
                            "train_loss": float(loss.detach().cpu()),
                            "gradient_norm": gradient_norm,
                            "tuning_accuracy": id_metrics["accuracy"],
                            "tuning_supported_accuracy": id_metrics["supported_accuracy"],
                            "tuning_pair_exact": id_metrics["counterfactual_pair_exact"],
                            "tuning_unsupported_error_rate": id_metrics["unsupported_error_rate"],
                            "development_accuracy": development_metrics["accuracy"],
                            "development_supported_accuracy": development_metrics[
                                "supported_accuracy"
                            ],
                            "development_pair_exact": development_metrics[
                                "counterfactual_pair_exact"
                            ],
                            "stable_window_pass": window,
                            "stable_streak": id_streak,
                            "eligible_checkpoint": eligible,
                            "development_selection_score": development_score,
                            **{
                                f"loss_{name}": float(value.detach().cpu())
                                for name, value in parts.items()
                            },
                        }
                    )
                    print(
                        f"[transformer-training] candidate={candidate_id} "
                        f"stage={stage_name} step={current} "
                        f"loss={float(loss.detach().cpu()):.4f} "
                        f"id={id_metrics['supported_accuracy']:.3f} "
                        f"pair={id_metrics['counterfactual_pair_exact']:.3f} "
                        f"dev={development_metrics['supported_accuracy']:.3f}/"
                        f"{development_metrics['counterfactual_pair_exact']:.3f} "
                        f"eligible={eligible}"
                    )
                    save_state("curriculum", current)
            phase_step = 0
            curriculum_stage_index = stage_index + 1
            save_state("curriculum", phase_step)
        run_terminated = True
        phase = "complete"
        save_state(phase, phase_step)

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    selected = best_state is not None
    total_registered_steps = warmup_steps + sum(int(stage["steps"]) for stage in stages)
    resource = {
        "model": "direct_transformer",
        "candidate_id": candidate_id,
        "loss_mode": loss_mode,
        "training_seconds": elapsed,
        "steps": total_registered_steps,
        "steps_completed": completed_total_step,
        "warmup_steps_completed": warmup_steps_completed,
        "curriculum_stages_completed": curriculum_stage_index,
        "peak_cuda_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device)) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "selected_tuning_checkpoint": selected,
        "selected_checkpoint_step": selected_step,
        "selected_checkpoint_stage": selected_stage,
        "selected_development_score": (selected_development_score if selected else None),
        "stable_tuning_windows": selected_stable_windows if selected else 0,
        "optimizer_steps": optimizer_steps,
        "training_complete": phase == "complete",
        "run_terminated": run_terminated,
        "termination_reason": (
            "curriculum_complete"
            if phase == "complete"
            else "warmup_budget_exhausted"
            if phase == "failed_warmup"
            else "incomplete"
        ),
        "warmup_convergence_pass": warmup_complete,
        "warmup_batch_size": warmup_batch_size,
        "curriculum_batch_size": batch_size,
        "warmup_sampling_seed_offset": warmup_sampling_seed_offset,
        "warmup_sampling_contract": "v7_complete_pair_replay",
        "selection_split": "one_hop_eligibility+bounded_6_8_ranking",
        "validation_evaluations_during_training": 0,
        "final_12_32_hop_evaluations_during_training": 0,
        "training_label_families": (
            "answer,entity,query_role,relation_anchor,event_source,event_target,"
            "boundary_value,action,cone,transport,boundary_deficit,terminal_mass"
        ),
        "matched_training_label_families": True,
        "auxiliary_outputs_used_at_inference": False,
        "compiler_or_executor_used": False,
        "curriculum": "one_hop_warmup_then_exact_1_2_1_4_1_8_hop_coverage",
        "exact_hop_coverage_per_batch": True,
        "development_thresholds_are_observers_only": True,
    }
    return curves, resource


def _train_native_v9(
    model: NativeGeometricBoundaryModel,
    train: Sequence[MatchedTopologyExample],
    id_tuning: Sequence[MatchedTopologyExample],
    native_readiness: Sequence[MatchedTopologyExample],
    mechanism_tuning: Sequence[MatchedTopologyExample],
    tokenizer: TrainOnlyTokenizer,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    checkpoint: Path,
    run_hash: str,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze only after strict multi-hop readiness and deterministic exposure."""

    steps = int(config["native_steps"])
    batch_size = int(config["batch_size"])
    eval_interval = int(config["eval_interval"])
    required_windows = int(config.get("checkpoint_stable_windows", 2))
    readiness_start = int(config.get("native_readiness_start_step", 200))
    minimum_exposure = int(config.get("native_min_group_exposure_before_freeze", 8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["native_learning_rate"]))
    start_step, best_state, best_score, curves, restored = (0, None, float("-inf"), [], {})
    if resume:
        start_step, best_state, best_score, curves, restored = _load_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            run_hash=run_hash,
            device=device,
        )

    groups = _ordered_pair_groups(train)
    metadata = _group_metadata(groups)
    exposures = {
        pair_id: int(value)
        for pair_id, value in dict(restored.get("group_exposures", {})).items()
        if pair_id in metadata
    }
    for pair_id in metadata:
        exposures.setdefault(pair_id, 0)
    train_pad = max(len(example.tokens) for example in train)
    readiness_hops = sorted(
        {
            int(example.base.metadata["path_hops"])
            for example in native_readiness
            if not bool(example.base.metadata.get("unsupported_answer", False))
        }
    )
    expected_readiness_hops = list(
        range(
            int(config.get("native_readiness_min_hops", 2)),
            int(config.get("native_readiness_max_hops", 8)) + 1,
        )
    )
    if readiness_hops != expected_readiness_hops:
        raise ValueError(
            f"native readiness hops must be {expected_readiness_hops}, got {readiness_hops}"
        )

    readiness_streak = int(restored.get("readiness_streak", 0))
    stable_streak = int(restored.get("stable_streak", 0))
    field_ready = bool(restored.get("field_ready", False))
    field_ready_step = restored.get("field_ready_step")
    field_frozen_state = restored.get("field_frozen_state")
    field_candidate_paths = list(restored.get("field_candidate_paths", []))
    mechanism_ready = bool(restored.get("mechanism_ready", False))
    mechanism_evaluated = bool(restored.get("mechanism_evaluated", False))
    mechanism_summary = dict(restored.get("mechanism_summary", {}))
    selected_mechanism_min_drop = float(restored.get("selected_mechanism_min_drop", 0.0))
    selected_step = restored.get("selected_step")
    gradient_reaches_psi = bool(restored.get("gradient_reaches_psi", False))
    optimizer_steps = int(restored.get("optimizer_steps", 0))
    zero_teacher_optimizer_steps = int(restored.get("zero_teacher_optimizer_steps", 0))
    training_complete = bool(restored.get("training_complete", False))
    completed_step = start_step
    training_state: dict[str, Any] = dict(restored)
    if field_ready and field_frozen_state is not None:
        model.load_state_dict(field_frozen_state)

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step_iterator = () if training_complete else range(start_step + 1, steps + 1)
    for step in step_iterator:
        completed_step = step
        teacher = 0.0 if field_ready else 1.0
        scheduled_teacher = (
            1.0 if step < int(config.get("native_teacher_schedule_steps", steps)) else 0.0
        )
        if teacher > 0.0:
            group, selected_pair_ids = _epoch_grouped_sample(
                groups,
                batch_size,
                optimizer_steps,
                seed + 55,
            )
            for pair_id in selected_pair_ids:
                exposures[pair_id] += 1
            batch = collate_matched_topology_examples(
                group,
                tokenizer,
                pad_to_tokens=train_pad,
            ).to(device)
            model.train()
            outputs = model(**batch.model_inputs())
            loss, parts = native_geometric_training_loss(
                outputs,
                batch,
                config=model.config,
                teacher_weight=1.0,
                answer_weight=0.15,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_reaches_psi = gradient_reaches_psi or any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and float(parameter.grad.abs().sum().item()) > 0.0
                for name, parameter in model.named_parameters()
                if "psi" in name or "world" in name
            )
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            optimizer_steps += 1
            train_loss = float(loss.detach().cpu().item())
        else:
            parts = {}
            gradient_norm = 0.0
            train_loss = 0.0

        should_eval = (
            step == 1
            or step == steps
            or step % eval_interval == 0
            or (field_ready and stable_streak < required_windows)
        )
        just_frozen = False
        mechanism_evaluated_now = False
        if should_eval:
            readiness_metrics, _, _ = evaluate_native(
                model,
                native_readiness,
                tokenizer,
                device=device,
                batch_size=int(config["eval_batch_size"]),
                cohort="native_multihop_readiness",
                max_pairwise_cells=int(config["max_pairwise_cells"]),
            )
            id_metrics, _, _ = evaluate_native(
                model,
                id_tuning,
                tokenizer,
                device=device,
                batch_size=int(config["eval_batch_size"]),
                cohort="id_tuning_reporting_only",
                max_pairwise_cells=int(config["max_pairwise_cells"]),
            )
            exposure = _exposure_summary(exposures, metadata, minimum_exposure)
            strict_field_pass = _native_field_ready_v9(readiness_metrics, config)
            readiness_pass = strict_field_pass and bool(exposure["minimum_exposure_pass"])
            if teacher > 0.0 and step >= readiness_start:
                readiness_streak = readiness_streak + 1 if readiness_pass else 0
                if readiness_pass:
                    candidate_path = _save_field_candidate(
                        checkpoint,
                        model=model,
                        run_hash=run_hash,
                        step=step,
                        metrics={
                            **readiness_metrics,
                            "minimum_group_exposure": exposure["minimum_group_exposure"],
                            "unseen_group_count": exposure["unseen_group_count"],
                        },
                    )
                    if candidate_path not in field_candidate_paths:
                        field_candidate_paths.append(candidate_path)
                if readiness_streak >= required_windows and not field_ready:
                    field_ready = True
                    field_ready_step = step
                    field_frozen_state = copy.deepcopy(
                        {name: value.detach().cpu() for name, value in model.state_dict().items()}
                    )
                    just_frozen = True
            if field_ready and teacher == 0.0 and not mechanism_evaluated:
                mechanism_ready, mechanism_summary, _ = _native_mechanism_shards_ready(
                    model,
                    mechanism_tuning,
                    tokenizer,
                    config=config,
                    device=device,
                )
                mechanism_evaluated = True
                mechanism_evaluated_now = True
                selected_mechanism_min_drop = float(
                    mechanism_summary["mechanism_tuning_min_targeted_drop"]
                )
            stable_window = field_ready and teacher == 0.0 and readiness_pass
            stable_streak = stable_streak + 1 if stable_window else 0
            score = _selection_score(readiness_metrics) + 0.25 * _measurement_min(readiness_metrics)
            if (
                stable_window
                and stable_streak >= required_windows
                and mechanism_ready
                and score >= best_score - 1.0e-12
            ):
                best_score = score
                selected_step = field_ready_step
                best_state = copy.deepcopy(field_frozen_state)
            default_mechanism_summary: dict[str, Any] = {
                "mechanism_tuning_complete": False,
                "mechanism_tuning_evaluable": False,
                "mechanism_tuning_min_targeted_drop": 0.0,
                "mechanism_tuning_min_panel_full_accuracy": 0.0,
                "mechanism_tuning_scenarios": 0,
                "mechanism_tuning_shards_expected": int(
                    config.get("mechanism_tuning_shard_count", 1)
                ),
                "mechanism_tuning_shards_observed": 0,
                "mechanism_tuning_shards_passed": 0,
            }
            logged_mechanism = {**default_mechanism_summary, **mechanism_summary}
            curves.append(
                {
                    "model": "native_ergt",
                    "step": step,
                    "teacher_weight": teacher,
                    "scheduled_teacher_weight": scheduled_teacher,
                    "train_loss": train_loss,
                    "gradient_norm": gradient_norm,
                    "id_tuning_supported_accuracy": id_metrics["supported_accuracy"],
                    "id_tuning_pair_exact": id_metrics["counterfactual_pair_exact"],
                    "readiness_supported_accuracy": readiness_metrics["supported_accuracy"],
                    "readiness_pair_exact": readiness_metrics["counterfactual_pair_exact"],
                    "readiness_unsupported_error_rate": readiness_metrics["unsupported_error_rate"],
                    "readiness_measurement_min_accuracy": _measurement_min(readiness_metrics),
                    "readiness_effective_action_mae": readiness_metrics["effective_action_mae"],
                    "readiness_program_event_recall": readiness_metrics["program_event_recall"],
                    "readiness_program_event_precision": readiness_metrics[
                        "program_event_precision"
                    ],
                    "readiness_event_chain_exact": readiness_metrics["event_chain_exact"],
                    "readiness_path_coverage": readiness_metrics["event_world_path_coverage"],
                    "readiness_closure_non_degradation": readiness_metrics[
                        "geodesic_closure_non_degradation_rate"
                    ],
                    "readiness_hard_overflow_rate": readiness_metrics[
                        "event_selection_overflow_rate"
                    ],
                    "strict_field_metrics_pass": strict_field_pass,
                    "minimum_exposure_pass": exposure["minimum_exposure_pass"],
                    "mean_group_exposure": exposure["mean_group_exposure"],
                    "minimum_group_exposure": exposure["minimum_group_exposure"],
                    "unseen_group_count": exposure["unseen_group_count"],
                    "readiness_pass": readiness_pass,
                    "readiness_streak": readiness_streak,
                    "field_ready": field_ready,
                    "field_ready_step": field_ready_step,
                    "mechanism_tuning_pass": mechanism_ready,
                    **{
                        key: value
                        for key, value in logged_mechanism.items()
                        if key != "mechanism_tuning_shard_summaries"
                    },
                    "stable_window_pass": stable_window,
                    "stable_streak": stable_streak,
                    "zero_teacher": teacher == 0.0,
                    "field_frozen_before_mechanism_selection": field_ready,
                    **{
                        f"loss_{name}": float(value.detach().cpu().item())
                        for name, value in parts.items()
                        if isinstance(value, torch.Tensor) and value.numel() == 1
                    },
                }
            )
            print(
                f"[ergt-training] step={step} teacher={teacher:.1f} "
                f"loss={train_loss:.4f} id={id_metrics['supported_accuracy']:.3f} "
                f"readyAcc={readiness_metrics['supported_accuracy']:.3f} "
                f"pair={readiness_metrics['counterfactual_pair_exact']:.3f} "
                f"measure={_measurement_min(readiness_metrics):.3f} "
                f"actionMAE={readiness_metrics['effective_action_mae']:.3f} "
                f"chain={readiness_metrics['event_chain_exact']:.3f} "
                f"closure={readiness_metrics['geodesic_closure_non_degradation_rate']:.3f} "
                f"overflow={readiness_metrics['event_selection_overflow_rate']:.3f} "
                f"exposure={exposure['minimum_group_exposure']}/"
                f"{minimum_exposure} unseen={exposure['unseen_group_count']} "
                f"ready={field_ready}({readiness_streak}/{required_windows}) "
                f"mechanism={mechanism_ready} stable={stable_streak}/{required_windows}"
            )

        training_complete = (
            field_ready and mechanism_evaluated and stable_streak >= required_windows
        )
        training_state = {
            "readiness_streak": readiness_streak,
            "stable_streak": stable_streak,
            "field_ready": field_ready,
            "field_ready_step": field_ready_step,
            "field_frozen_state": field_frozen_state,
            "field_candidate_paths": field_candidate_paths,
            "mechanism_ready": mechanism_ready,
            "mechanism_evaluated": mechanism_evaluated,
            "mechanism_summary": mechanism_summary,
            "selected_mechanism_min_drop": selected_mechanism_min_drop,
            "selected_step": selected_step,
            "gradient_reaches_psi": gradient_reaches_psi,
            "optimizer_steps": optimizer_steps,
            "zero_teacher_optimizer_steps": zero_teacher_optimizer_steps,
            "group_exposures": exposures,
            "training_complete": training_complete,
        }
        if step and (
            step % max(eval_interval * 2, 100) == 0
            or step == steps
            or just_frozen
            or mechanism_evaluated_now
            or training_complete
        ):
            _save_training_state(
                checkpoint,
                model=model,
                optimizer=optimizer,
                step=step,
                best_state=best_state,
                best_score=best_score,
                curves=curves,
                config=asdict(model.config),
                run_hash=run_hash,
                training_state=training_state,
            )
        if training_complete:
            break

    if steps > 0:
        _save_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=completed_step,
            best_state=best_state,
            best_score=best_score,
            curves=curves,
            config=asdict(model.config),
            run_hash=run_hash,
            training_state=training_state,
        )
    evaluation_state_status = "untrained_smoke"
    if best_state is not None:
        model.load_state_dict(best_state)
        evaluation_state_status = "selected_v9_multihop_ready_frozen_checkpoint"
    elif field_frozen_state is not None:
        model.load_state_dict(field_frozen_state)
        evaluation_state_status = "unselected_frozen_field_diagnostic_only"
    elif steps > 0:
        evaluation_state_status = "unselected_final_training_state_diagnostic_only"
    exposure = _exposure_summary(exposures, metadata, minimum_exposure)
    elapsed = time.perf_counter() - started
    resource = {
        "model": "native_ergt",
        "training_seconds": elapsed,
        "steps": steps,
        "steps_completed": completed_step,
        "peak_cuda_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device)) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "gradient_reaches_geometric_field": gradient_reaches_psi if steps else None,
        "selected_zero_teacher_checkpoint": best_state is not None if steps else None,
        "selected_checkpoint_step": selected_step,
        "independent_field_readiness_pass": field_ready if steps else None,
        "independent_field_readiness_step": field_ready_step,
        "native_readiness_hops": readiness_hops,
        "native_readiness_is_multihop": readiness_hops == expected_readiness_hops,
        "native_id_is_reporting_only": True,
        "strict_v21_chain_closure_overflow_contract": True,
        "selected_mechanism_tuning_pass": mechanism_ready if steps else None,
        "selected_mechanism_min_targeted_drop": selected_mechanism_min_drop,
        "mechanism_tuning_evaluations": int(mechanism_evaluated),
        "mechanism_tuning_shards_expected": int(config.get("mechanism_tuning_shard_count", 1)),
        "mechanism_tuning_shards_passed": int(
            mechanism_summary.get("mechanism_tuning_shards_passed", 0)
        ),
        "stable_zero_teacher_windows": stable_streak,
        "zero_teacher_optimizer_steps": zero_teacher_optimizer_steps,
        "optimizer_steps": optimizer_steps,
        "teacher_stops_immediately_on_field_readiness": True,
        "mechanism_controls_teacher": False,
        "field_candidate_paths": field_candidate_paths,
        "selection_uses_heldout_interventions": False,
        "selection_split": "independent_2_8_hop_readiness+two_frozen_mechanism_shards",
        "validation_evaluations_during_training": 0,
        "final_12_32_hop_evaluations_during_training": 0,
        "evaluation_state_status": evaluation_state_status,
        "training_complete": training_complete,
        "training_exposure_audit": exposure,
        "pair_group_count": exposure["pair_group_count"],
        "mean_group_exposure_before_freeze": exposure["mean_group_exposure"],
        "minimum_group_exposure_before_freeze": exposure["minimum_group_exposure"],
        "unseen_group_count_before_freeze": exposure["unseen_group_count"],
        "minimum_exposure_pass": exposure["minimum_exposure_pass"],
        "matched_training_label_families": True,
        "training_label_families": (
            "answer,entity,query_role,relation_anchor,event_source,event_target,"
            "boundary_value,action,cone,transport,boundary_deficit,terminal_mass"
        ),
        "auxiliary_outputs_used_at_inference": False,
        "compiler_or_executor_used": False,
        "checkpoint_protocol": "v9_multihop_exposure_locked",
    }
    return curves, resource


def train_models(
    train: Sequence[MatchedTopologyExample],
    tuning: Sequence[MatchedTopologyExample],
    tokenizer: TrainOnlyTokenizer,
    *,
    native_readiness: Sequence[MatchedTopologyExample] | None = None,
    mechanism_tuning: Sequence[MatchedTopologyExample] | None = None,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    data_seed: int,
    run_root: Path,
    resume: bool,
) -> TrainedModels:
    run_root.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    mechanism_tuning = tuple(
        mechanism_tuning
        if mechanism_tuning is not None
        else (example for example in tuning if int(example.answer_id) != 2)
    )
    native_readiness = tuple(native_readiness or ())
    if not native_readiness:
        raise ValueError("V9 native training requires an independent multi-hop readiness cohort")
    max_tokens = int(config["max_tokens"])
    native_config = ERGT43Config(
        vocab_size=tokenizer.vocab_size,
        max_tokens=max_tokens,
        raw_input_contract=RawTokenInputContract.from_tokenizer(tokenizer),
        hidden_dim=int(config["native_hidden_dim"]),
        psi_rank=int(config["native_psi_rank"]),
        field_steps=int(config["native_field_steps"]),
        sparse_top_k=int(config["native_sparse_top_k"]),
        max_hops=int(config["native_max_hops"]),
        geodesic_closure_steps=int(config["native_geodesic_closure_steps"]),
        geodesic_backbone_levels=int(config["native_geodesic_backbone_levels"]),
    )
    native = NativeGeometricBoundaryModel(native_config)
    native_parameters = count_parameters(native)
    qualified_recipe = config.get("qualified_transformer_recipe")
    candidate_layers = (
        (int(qualified_recipe["n_layers"]),) if isinstance(qualified_recipe, Mapping) else (4,)
    )
    transformer_config = matched_transformer_config(
        vocab_size=tokenizer.vocab_size,
        max_tokens=max_tokens,
        query_token_id=int(tokenizer.token_to_id["query"]),
        candidate_a_token_id=int(tokenizer.token_to_id["candidate_a"]),
        candidate_b_token_id=int(tokenizer.token_to_id["candidate_b"]),
        target_parameters=native_parameters,
        candidate_layers=candidate_layers,
    )
    transformer = DirectTransformer(transformer_config)
    transformer_parameters = count_parameters(transformer)
    transformer_auxiliary_parameters = training_only_parameter_count(transformer)
    transformer_inference_parameters = inference_active_parameter_count(transformer)
    inference_relative_difference = abs(transformer_inference_parameters - native_parameters) / max(
        1, native_parameters
    )
    total_training_relative_difference = abs(transformer_parameters - native_parameters) / max(
        1, native_parameters
    )
    native.to(device)
    transformer.to(device)
    checkpoint_protocol = str(
        config.get("native_checkpoint_protocol", "v5_conjunctive_teacher_lock")
    )
    run_hash = protocol_hash(
        {
            "training_protocol": checkpoint_protocol,
            "config": dict(config),
            "optimization_seed": seed,
            "data_seed": data_seed,
            "train_fingerprint": _dataset_fingerprint(train),
            "tuning_fingerprint": _dataset_fingerprint(tuning),
            "native_readiness_fingerprint": _dataset_fingerprint(native_readiness),
            "mechanism_tuning_fingerprint": _dataset_fingerprint(mechanism_tuning),
            "training_source_sha256": _training_source_hashes(),
        }
    )
    native_checkpoint = run_root / "native_ergt_training.pt"
    transformer_checkpoint = run_root / "direct_transformer_training.pt"
    if checkpoint_protocol != "v9_multihop_exposure_locked":
        raise ValueError(
            "training_v9 requires native_checkpoint_protocol=v9_multihop_exposure_locked"
        )
    native_curves, native_resource = _train_native_v9(
        native,
        train,
        tuning,
        native_readiness,
        mechanism_tuning,
        tokenizer,
        config=config,
        device=device,
        seed=seed,
        checkpoint=native_checkpoint,
        run_hash=run_hash,
        resume=resume,
    )
    transformer_training_protocol = str(
        config.get("transformer_training_protocol", "v6_fixed_mixture")
    )
    if transformer_training_protocol in {
        "v7_qualified_curriculum",
        "v8_staged_qualified_curriculum",
    }:
        if not isinstance(qualified_recipe, Mapping):
            raise ValueError("qualified Transformer training requires a locked recipe")
        trainer = (
            train_transformer_staged_curriculum_candidate
            if transformer_training_protocol == "v8_staged_qualified_curriculum"
            else train_transformer_curriculum_candidate
        )
        transformer_curves, transformer_resource = trainer(
            transformer,
            train,
            tuning,
            mechanism_tuning,
            tokenizer,
            config=config,
            recipe=qualified_recipe,
            device=device,
            seed=seed,
            checkpoint=transformer_checkpoint,
            run_hash=run_hash,
            resume=resume,
        )
    else:
        transformer_curves, transformer_resource = _train_transformer(
            transformer,
            train,
            tuning,
            tokenizer,
            config=config,
            device=device,
            seed=seed,
            checkpoint=transformer_checkpoint,
            run_hash=run_hash,
            resume=resume,
        )
    return TrainedModels(
        native=native,
        transformer=transformer,
        native_config=native_config,
        transformer_config=transformer_config,
        curves=[*native_curves, *transformer_curves],
        resources=[native_resource, transformer_resource],
        checkpoint_paths={
            "native_ergt": str(native_checkpoint),
            "direct_transformer": str(transformer_checkpoint),
        },
        parameter_audit={
            "native_ergt_trainable_parameters": native_parameters,
            "direct_transformer_trainable_parameters": transformer_parameters,
            "direct_transformer_training_only_auxiliary_parameters": (
                transformer_auxiliary_parameters
            ),
            "direct_transformer_inference_active_parameters": (transformer_inference_parameters),
            "inference_active_relative_parameter_difference": (inference_relative_difference),
            "total_training_relative_parameter_difference": (total_training_relative_difference),
            "relative_parameter_difference": inference_relative_difference,
            "within_ten_percent": inference_relative_difference <= 0.10,
            "matched_training_label_families": bool(
                transformer_resource.get("matched_training_label_families", True)
            ),
            "transformer_loss_mode": str(
                transformer_resource.get("loss_mode", "matched_auxiliary")
            ),
            "transformer_auxiliary_outputs_used_at_inference": False,
            "transformer_compiler_or_executor_used": False,
            "direct_transformer_config": asdict(transformer_config),
            "native_ergt_config": asdict(native_config),
        },
    )


__all__ = [
    "TrainedModels",
    "_epoch_grouped_sample",
    "_exposure_summary",
    "_native_field_ready_v9",
    "_ordered_pair_groups",
    "_train_native_v9",
    "protocol_hash",
    "set_seed",
    "train_transformer_curriculum_candidate",
    "train_transformer_staged_curriculum_candidate",
    "train_models",
]
