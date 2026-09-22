"""Frozen native challenge targets and matched diagnostic readouts for M2-E005."""
from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .runtime import activate_reference
from .shadow_phase import PhaseAnchorNetwork

activate_reference()
from ergt_reviewer.evaluation_v9 import memory_bounded_batch_size  # noqa: E402
from ergt_reviewer.matched_data import collate_matched_topology_examples  # noqa: E402
from ergt_reviewer.native_solver import hard_solutions_from_outputs  # noqa: E402


def _chunks(values: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [values[start:start + size] for start in range(0, len(values), size)]


def _gather_tokens(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 1, positions)


@torch.no_grad()
def native_condition_rows(
    model: nn.Module,
    examples: Sequence[Any],
    tokenizer: Any,
    *,
    intervention: str,
    intervention_seed: int,
    batch_size: int = 4,
    pad_to_tokens: int | None = None,
    max_pairwise_cells: int = 2_500_000,
) -> list[dict[str, Any]]:
    """Return per-example answer, margin and path metrics for one frozen condition."""
    if not examples:
        return []
    token_count = pad_to_tokens or max(len(example.tokens) for example in examples)
    effective_batch = memory_bounded_batch_size(batch_size, token_count, max_pairwise_cells)
    rows: list[dict[str, Any]] = []
    model.eval()
    for group in _chunks(tuple(examples), effective_batch):
        batch = collate_matched_topology_examples(
            group, tokenizer, pad_to_tokens=token_count,
        ).to(torch.device("cpu"))
        outputs = model.forward_with_intervention(
            **batch.model_inputs(), intervention=intervention,
            intervention_seed=intervention_seed, observe_spectrum=False,
        )
        solution, _ = hard_solutions_from_outputs(outputs, model.config)
        selected = outputs["selected_event_indices"]
        selected_gold = _gather_tokens(batch.physical.event_mask, selected).bool()
        relation_full = torch.cat(
            ((1.0 - outputs["event_presence"]).unsqueeze(-1),
             outputs["event_relation_probability"]), dim=-1,
        )
        predicted_relation = relation_full.argmax(dim=-1)
        predicted_active = (
            (outputs["event_presence"] >= model.config.event_presence_floor)
            & (predicted_relation > 0)
            & outputs["event_valid_mask"].bool()
        )
        for row_index, example in enumerate(group):
            gold_positions = batch.physical.event_mask[row_index].nonzero(as_tuple=False).flatten()
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
            selected_positions = {
                int(selected[row_index, slot].item())
                for slot in outputs["event_valid_mask"][row_index].bool().nonzero(as_tuple=False).flatten()
            }
            gold_position_set = {int(value.item()) for value in gold_positions}
            closure_covered = int(
                outputs["event_world_closure_hard_gate"][row_index][selected_gold[row_index]]
                .bool().sum().item()
            )
            gold = int(example.answer_id)
            logits = outputs["native_answer_logits"][row_index].detach().cpu()
            alternatives = torch.cat((logits[:gold], logits[gold + 1:]))
            response_margin = float(logits[gold] - alternatives.max())
            soft = outputs["candidate_soft_action"][row_index].detach().cpu()
            hard = solution.candidate_action[row_index].detach().cpu()
            soft_choice_margin = None
            hard_choice_margin = None
            if gold in (0, 1):
                other = 1 - gold
                soft_choice_margin = float(soft[other] - soft[gold])
                if bool(torch.isfinite(hard).all()) and bool((hard < 5.0e5).all()):
                    hard_choice_margin = float(hard[other] - hard[gold])
            rows.append({
                "example_id": example.example_id,
                "pair_id": example.pair_id,
                "scenario": example.scenario,
                "hops": int(example.base.metadata["path_hops"]),
                "counterfactual_variant": int(example.counterfactual_variant),
                "label": gold,
                "prediction": int(solution.answer_ids[row_index]),
                "answer_correct": int(solution.answer_ids[row_index]) == gold,
                "failure_code": int(solution.failure_code[row_index]),
                "response_logit_margin": response_margin,
                "soft_choice_margin": soft_choice_margin,
                "hard_choice_margin": hard_choice_margin,
                "program_event_recall": matched / max(1, len(gold_program)),
                "program_event_precision": matched / max(1, len(predicted_program)),
                "event_chain_exact": float(gold_program == predicted_program),
                "selected_event_recall": len(selected_positions & gold_position_set) / max(1, len(gold_position_set)),
                "event_world_path_coverage": closure_covered / max(1, len(gold_positions)),
                "intervention": intervention,
            })
    return rows


@torch.no_grad()
def build_challenge_panel(
    model: nn.Module,
    examples: Sequence[Any],
    tokenizer: Any,
    *,
    intervention_by_scenario: Mapping[str, str],
    intervention_seed: int,
    batch_size: int = 4,
) -> list[dict[str, Any]]:
    """Evaluate full and one preregistered scenario intervention per example."""
    if not examples:
        raise ValueError("challenge panel cannot be empty")
    token_count = max(len(example.tokens) for example in examples)
    full_rows = native_condition_rows(
        model, examples, tokenizer, intervention="full",
        intervention_seed=intervention_seed, batch_size=batch_size,
        pad_to_tokens=token_count,
    )
    full = {row["example_id"]: row for row in full_rows}
    grouped: dict[str, list[Any]] = defaultdict(list)
    for example in examples:
        try:
            intervention = intervention_by_scenario[example.scenario]
        except KeyError as exc:
            raise ValueError(f"no registered intervention for {example.scenario}") from exc
        grouped[intervention].append(example)
    changed: dict[str, dict[str, Any]] = {}
    for intervention in sorted(grouped):
        for row in native_condition_rows(
            model, grouped[intervention], tokenizer, intervention=intervention,
            intervention_seed=intervention_seed, batch_size=batch_size,
            pad_to_tokens=token_count,
        ):
            changed[row["example_id"]] = row
    rows: list[dict[str, Any]] = []
    for example in examples:
        left, right = full[example.example_id], changed[example.example_id]
        soft_drop = None
        if left["soft_choice_margin"] is not None and right["soft_choice_margin"] is not None:
            soft_drop = left["soft_choice_margin"] - right["soft_choice_margin"]
        rows.append({
            "example_id": example.example_id,
            "pair_id": example.pair_id,
            "scenario": example.scenario,
            "hops": int(example.base.metadata["path_hops"]),
            "label": int(example.answer_id),
            "intervention": right["intervention"],
            "full": left,
            "challenge": right,
            "response_margin_drop": left["response_logit_margin"] - right["response_logit_margin"],
            "soft_choice_margin_drop": soft_drop,
            "answer_became_wrong": bool(left["answer_correct"] and not right["answer_correct"]),
            "event_chain_degraded": bool(left["event_chain_exact"] > right["event_chain_exact"]),
            "selected_event_recall_drop": left["selected_event_recall"] - right["selected_event_recall"],
            "path_coverage_drop": left["event_world_path_coverage"] - right["event_world_path_coverage"],
        })
    return rows


def challenge_support(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty challenge rows")
    margins = torch.tensor([float(row["response_margin_drop"]) for row in rows], dtype=torch.float64)
    wrong = sum(bool(row["answer_became_wrong"]) for row in rows)
    chain = sum(bool(row["event_chain_degraded"]) for row in rows)
    path = sum(float(row["path_coverage_drop"]) > 1.0e-12 for row in rows)
    pairs: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        pairs[str(row["pair_id"])].append(bool(row["challenge"]["answer_correct"]))
    pair_exact = sum(len(values) == 2 and all(values) for values in pairs.values()) / max(1, len(pairs))
    return {
        "examples": len(rows),
        "pairs": len(pairs),
        "full_answer_correct": sum(bool(row["full"]["answer_correct"]) for row in rows),
        "challenge_answer_correct": sum(bool(row["challenge"]["answer_correct"]) for row in rows),
        "answer_failures": wrong,
        "answer_successes": len(rows) - wrong,
        "challenge_counterfactual_pair_exact": pair_exact,
        "event_chain_degradations": chain,
        "path_coverage_degradations": path,
        "response_margin_drop_mean": float(margins.mean()),
        "response_margin_drop_std": float(margins.std(unbiased=False)),
        "response_margin_drop_min": float(margins.min()),
        "response_margin_drop_max": float(margins.max()),
        "response_margin_drop_finite": bool(torch.isfinite(margins).all()),
    }


@torch.no_grad()
def representation_vectors(
    records: Sequence[dict[str, Any]],
    examples: Sequence[Any],
    *,
    mode: str,
    q_width: int,
    phase: PhaseAnchorNetwork | None = None,
) -> torch.Tensor:
    """Pool source, candidates, mean and maximum into an example vector."""
    if len(records) != len(examples):
        raise ValueError("records/examples length mismatch")
    values: list[torch.Tensor] = []
    for record, example in zip(records, examples, strict=True):
        psi = record["psi"]
        if mode == "raw_only":
            nodes = psi[:, :-q_width]
        elif mode == "q_only":
            nodes = psi[:, -q_width:]
        elif mode == "no_phase":
            nodes = psi
        elif mode == "phase":
            if phase is None:
                raise ValueError("phase representation requires a frozen phase module")
            mask = torch.ones((1, psi.shape[0]), dtype=torch.bool)
            nodes = phase(psi.unsqueeze(0), mask)[0].transpose(0, 1)
        else:
            raise ValueError(f"unknown representation: {mode}")
        positions = (
            int(example.base.source_position),
            int(example.base.candidate_positions[0]),
            int(example.base.candidate_positions[1]),
        )
        pooled = torch.cat((
            nodes[positions[0]], nodes[positions[1]], nodes[positions[2]],
            nodes.mean(dim=0), nodes.max(dim=0).values,
        ))
        values.append(pooled.detach().cpu())
    return torch.stack(values)


class ScalarReadout(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, *, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
                nn.SiLU(), nn.Linear(hidden_dim, 1),
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def readout_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def matched_scalar_hidden(input_dim: int, target_parameters: int) -> int:
    return min(
        range(1, target_parameters + 1),
        key=lambda hidden: abs(2 * input_dim + hidden * (input_dim + 2) + 1 - target_parameters),
    )


def train_scalar_readout(
    vectors: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_parameters: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[ScalarReadout, dict[str, Any]]:
    if vectors.ndim != 2 or targets.shape != (vectors.shape[0],):
        raise ValueError("invalid scalar readout training shapes")
    target_mean = targets.double().mean()
    target_std = targets.double().std(unbiased=False).clamp_min(1.0e-8)
    normalized = ((targets.double() - target_mean) / target_std).float()
    hidden = matched_scalar_hidden(vectors.shape[1], target_parameters)
    model = ScalarReadout(vectors.shape[1], hidden, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps = 0
    for epoch in range(epochs):
        order = list(range(vectors.shape[0]))
        random.Random(seed + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(vectors[indices]), normalized[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite scalar readout loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
    model.eval().requires_grad_(False)
    return model, {
        "examples": vectors.shape[0],
        "epochs": epochs,
        "steps": steps,
        "input_dim": vectors.shape[1],
        "hidden_dim": hidden,
        "parameters": readout_parameter_count(model),
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "fresh_targets_used": False,
    }


@torch.no_grad()
def evaluate_scalar_readout(
    model: ScalarReadout,
    vectors: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_mean: float,
    target_std: float,
) -> tuple[dict[str, Any], torch.Tensor]:
    prediction = model(vectors).double() * float(target_std) + float(target_mean)
    truth = targets.double()
    error = prediction - truth
    mae = error.abs().mean()
    rmse = error.square().mean().sqrt()
    denominator = (truth - truth.mean()).square().sum()
    r2 = 1.0 - error.square().sum() / denominator if float(denominator) > 0 else None
    centered_prediction = prediction - prediction.mean()
    centered_truth = truth - truth.mean()
    pearson_denominator = centered_prediction.square().sum().sqrt() * centered_truth.square().sum().sqrt()
    pearson = (
        (centered_prediction * centered_truth).sum() / pearson_denominator
        if float(pearson_denominator) > 0 else torch.tensor(0.0)
    )
    baseline_error = truth - float(target_mean)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": None if r2 is None else float(r2),
        "pearson": float(pearson),
        "constant_baseline_mae": float(baseline_error.abs().mean()),
        "constant_baseline_rmse": float(baseline_error.square().mean().sqrt()),
        "examples": truth.numel(),
        "target_mean": float(truth.mean()),
        "target_std": float(truth.std(unbiased=False)),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std(unbiased=False)),
    }, prediction.float()
