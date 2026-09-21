"""Fresh-development confirmation primitives for the frozen M2-Q checkpoint.

The module keeps three questions separate: whether the frozen phase checkpoint
generalizes, whether Q alone carries the relation signal, and whether a
capacity-matched ordinary classifier explains the same signal without phase.
Fresh-development labels are evaluation-only.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import random
import time
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .lagged_q import native_proposal_q
from .runtime import activate_reference
from .shadow_data import pack_records, training_only
from .shadow_phase import PhaseAnchorNetwork, relation_logits

activate_reference()
from ergt_reviewer.fair_data_v9 import (  # noqa: E402
    _balanced_training_cohort,
    _unique_cohort,
    append_training_distractors,
    make_unsupported,
    protocol_tokenizer,
)
from ergt_reviewer.matched_data import (  # noqa: E402
    collate_matched_topology_examples,
    manifest_hash,
)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_fresh_development(
    reference_metadata: Any,
    *,
    data_seed: int,
    pair_counts_by_hop: dict[str, int],
    unsupported_pairs: int,
) -> tuple[tuple[Any, ...], Any, dict[str, Any]]:
    """Regenerate old training data, then build a raw-disjoint same-family cohort."""
    old, _, old_sha = training_only(reference_metadata)
    protocol = json.loads((reference_metadata / "protocol.json").read_text(encoding="utf-8"))
    config = protocol["config"]
    old_raw = {example.base.raw_text for example in old}
    old_pairs = {example.pair_id for example in old}
    forbidden = set(old_raw)
    supported = _balanced_training_cohort(
        pair_counts_by_hop=pair_counts_by_hop,
        seed=data_seed,
        split="m2_fresh_development",
        node_label_pool_size=int(config["node_label_pool_size"]),
        forbidden_raw_texts=forbidden,
    )
    unsupported = make_unsupported(_unique_cohort(
        pair_count=unsupported_pairs,
        seed=data_seed + 3,
        split="m2_fresh_development_unsupported",
        min_hops=int(config["train_min_hops"]),
        max_hops=int(config["train_max_hops"]),
        node_label_pool_size=int(config["node_label_pool_size"]),
        forbidden_raw_texts=forbidden,
    ))
    fresh = append_training_distractors(
        (*supported, *unsupported), int(config["training_distractor_tokens"]),
    )
    fresh_raw = {example.base.raw_text for example in fresh}
    fresh_pairs = {example.pair_id for example in fresh}
    rows = [{
        "example_id": example.example_id,
        "pair_id": example.pair_id,
        "split": example.base.split,
        "scenario": example.scenario,
        "counterfactual_variant": example.counterfactual_variant,
        "raw_text_sha256": _sha_text(example.base.raw_text),
        "token_count": len(example.tokens),
        "event_count": len(example.base.edges),
    } for example in fresh]
    audit = {
        "schema": "ergt-phi-m2-fresh-development-v1",
        "data_seed": data_seed,
        "pair_counts_by_hop": pair_counts_by_hop,
        "unsupported_pairs": unsupported_pairs,
        "examples": len(fresh),
        "pairs": len(fresh_pairs),
        "old_training_cohort_sha256": old_sha,
        "fresh_cohort_sha256": manifest_hash(fresh),
        "raw_text_disjoint_from_m0_training": not bool(old_raw & fresh_raw),
        "counterfactual_pair_disjoint_from_m0_training": not bool(old_pairs & fresh_pairs),
        "raw_text_collision_count": len(old_raw & fresh_raw),
        "pair_collision_count": len(old_pairs & fresh_pairs),
        "rows": rows,
    }
    audit["rows_sha256"] = _canonical_sha(rows)
    if not audit["raw_text_disjoint_from_m0_training"]:
        raise ValueError("fresh development raw text collides with M0 training")
    if not audit["counterfactual_pair_disjoint_from_m0_training"]:
        raise ValueError("fresh development pair identity collides with M0 training")
    tokenizer = protocol_tokenizer(int(config["node_label_pool_size"]))
    return fresh, tokenizer, audit


def confirmation_windows(examples: Sequence[Any], *, seed: int) -> tuple[list[int], list[int]]:
    """Create two pair-disjoint, split-stratified windows fixed by a seed."""
    pairs: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        pairs[example.pair_id].append(index)
    strata: dict[str, list[str]] = defaultdict(list)
    for pair_id, indices in pairs.items():
        if len(indices) != 2:
            raise ValueError("confirmation requires complete counterfactual pairs")
        strata[examples[indices[0]].base.split].append(pair_id)
    first: list[int] = []
    second: list[int] = []
    for offset, split in enumerate(sorted(strata)):
        pair_ids = sorted(strata[split])
        random.Random(seed + offset).shuffle(pair_ids)
        if len(pair_ids) < 2:
            raise ValueError("each confirmation stratum needs at least two pairs")
        for position, pair_id in enumerate(pair_ids):
            (first if position % 2 == 0 else second).extend(pairs[pair_id])
    first, second = sorted(first), sorted(second)
    if not first or not second or set(first) & set(second):
        raise ValueError("invalid confirmation windows")
    first_pairs = {examples[index].pair_id for index in first}
    second_pairs = {examples[index].pair_id for index in second}
    if first_pairs & second_pairs:
        raise ValueError("confirmation windows are not pair-disjoint")
    return first, second


@torch.no_grad()
def cache_q_features(
    baseline: nn.Module,
    examples: Sequence[Any],
    tokenizer: Any,
    *,
    deadline: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cache label-free Psi0 and Q node marginals, then attach loss-only labels."""
    raw_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, example in enumerate(examples):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("M2 fresh-development runtime budget exceeded")
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
        rows.append({
            "example_id": example.example_id,
            "pair_id": example.pair_id,
            "feature_sha256": hashlib.sha256(features.numpy().tobytes()).hexdigest(),
            "valid_tokens": int(mask.sum()),
            "candidate_edges": int(edge_index.shape[1]),
            "q_mass": float(q_pre.sum()),
        })
        if (index + 1) % 50 == 0:
            print(f"Cached confirmation features: {index + 1}/{len(examples)}", flush=True)
    records = [{
        **raw,
        "events": torch.tensor(
            [[edge.source_position, edge.target_position] for edge in example.base.edges],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [edge.relation_id - 1 for edge in example.base.edges], dtype=torch.long,
        ),
    } for raw, example in zip(raw_records, examples)]
    return records, {
        "schema": "ergt-phi-m2-fresh-feature-cache-v1",
        "raw_features_completed_before_supervision_attached": True,
        "examples": len(examples),
        "reference_prepass_calls": len(examples),
        "proposal_probe_calls": len(examples),
        "seconds": time.perf_counter() - started,
        "rows": rows,
    }


class EndpointClassifier(nn.Module):
    """Ordinary relation classifier over source/target node features."""
    def __init__(self, node_dim: int, hidden_dim: int, relations: int, *, seed: int):
        super().__init__()
        self.node_dim = node_dim
        self.relations = relations
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.LayerNorm(2 * node_dim),
                nn.Linear(2 * node_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, relations),
            )

    def forward(self, features: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        batch, source, target = events.unbind(-1)
        endpoints = torch.cat((features[batch, source], features[batch, target]), dim=-1)
        return self.network(endpoints)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def matched_hidden_dim(node_dim: int, relations: int, target_parameters: int) -> int:
    """Choose the positive hidden width nearest the frozen phase parameter count."""
    best = min(
        range(1, max(2, target_parameters + 1)),
        key=lambda hidden: abs(
            (4 * node_dim + hidden * (2 * node_dim + 1 + relations) + relations)
            - target_parameters
        ),
    )
    return int(best)


def _control_records(records: Sequence[dict[str, Any]], *, q_only: bool, q_width: int) -> list[dict[str, Any]]:
    return [{**record, "psi": record["psi"][:, -q_width:] if q_only else record["psi"]}
            for record in records]


def train_control(
    records: Sequence[dict[str, Any]],
    fit_indices: Sequence[int],
    *,
    q_only: bool,
    q_width: int,
    hidden_dim: int,
    relations: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[EndpointClassifier, dict[str, Any]]:
    """Train only on the historical M0 fit partition for a fixed step budget."""
    view = _control_records(records, q_only=q_only, q_width=q_width)
    node_dim = int(view[0]["psi"].shape[-1])
    model = EndpointClassifier(node_dim, hidden_dim, relations, seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    steps = 0
    for epoch in range(epochs):
        order = list(fit_indices)
        random.Random(seed + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), batch_size):
            features, _, events, labels = pack_records(view, order[start:start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features, events), labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite control loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
    model.eval().requires_grad_(False)
    return model, {
        "epochs": epochs,
        "steps": steps,
        "examples": len(fit_indices),
        "fresh_labels_used": False,
        "parameters": parameter_count(model),
        "node_dim": node_dim,
        "hidden_dim": hidden_dim,
    }


@torch.no_grad()
def evaluate_control(
    model: EndpointClassifier,
    records: Sequence[dict[str, Any]],
    indices: Sequence[int],
    *,
    q_only: bool,
    q_width: int,
    batch_size: int,
) -> dict[str, Any]:
    view = _control_records(records, q_only=q_only, q_width=q_width)
    confusion = torch.zeros(model.relations, model.relations, dtype=torch.long)
    ce = 0.0
    total = 0
    for start in range(0, len(indices), batch_size):
        features, _, events, labels = pack_records(view, indices[start:start + batch_size])
        logits = model(features, events)
        ce += float(F.cross_entropy(logits, labels, reduction="sum"))
        total += labels.numel()
        prediction = logits.argmax(-1)
        confusion += torch.bincount(
            labels * model.relations + prediction,
            minlength=model.relations ** 2,
        ).reshape(model.relations, model.relations)
    counts = confusion.sum(1)
    recall = confusion.diag().double() / counts.clamp_min(1)
    return {
        "ce": ce / total,
        "accuracy": float(confusion.diag().sum()) / total,
        "balanced_accuracy": float(recall.mean()),
        "per_class_recall": recall.tolist(),
        "class_counts": counts.tolist(),
        "confusion_matrix": confusion.tolist(),
        "events": total,
    }


@torch.no_grad()
def event_predictions(
    phase: PhaseAnchorNetwork,
    q_only: EndpointClassifier,
    no_phase: EndpointClassifier,
    records: Sequence[dict[str, Any]],
    indices: Iterable[int],
    *,
    q_width: int,
    window: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in indices:
        record = records[index]
        features, mask, events, labels = pack_records(records, [index])
        anchors = phase(features, mask)
        phase_logits = relation_logits(
            anchors, phase.offsets, events, phase.config.temperature,
        )
        q_logits = q_only(features[:, :, -q_width:], events)
        no_phase_logits = no_phase(features, events)
        for position, ((_, source, target), label) in enumerate(zip(events.tolist(), labels.tolist())):
            rows.append({
                "window": window,
                "example_id": record["example_id"],
                "pair_id": record["pair_id"],
                "event_position": position,
                "source": source,
                "target": target,
                "label": label,
                "phase_prediction": int(phase_logits[position].argmax()),
                "q_only_prediction": int(q_logits[position].argmax()),
                "no_phase_prediction": int(no_phase_logits[position].argmax()),
            })
    return rows


def phase_window_checks(metrics: dict[str, Any], *, relations: int) -> dict[str, bool]:
    return {
        "all_relation_classes_present": all(value > 0 for value in metrics["class_counts"]),
        "balanced_accuracy": metrics["balanced_accuracy"] >= 0.70,
        "minimum_class_recall": min(metrics["per_class_recall"]) >= 0.50,
        "ce_below_uniform_by_five_percent": metrics["ce"] <= math.log(relations) * 0.95,
        "offset_separation": metrics["minimum_offset_separation"] >= 1.0,
        "phase_not_constant": metrics["phase_resultant"] <= 0.98,
    }
