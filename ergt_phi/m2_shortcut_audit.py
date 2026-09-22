"""Ablations and target diagnostics for the frozen M2 relation readouts.

The helpers in this module never alter a supplied record or checkpoint.  Q is
ordered as three outgoing/source relation channels followed by three
incoming/target relation channels.  Permutations are applied independently to
those two blocks.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import torch

from .m2_fresh import EndpointClassifier, evaluate_control
from .shadow_data import pack_records
from .shadow_metrics import evaluate_calibration
from .shadow_phase import PhaseAnchorNetwork, relation_logits
from .runtime import activate_reference

activate_reference()
from ergt_reviewer.data_schema import RELATION_SURFACES  # noqa: E402


def transform_q_records(
    records: Sequence[dict[str, Any]],
    *,
    q_width: int,
    mode: str,
    permutation: Sequence[int] = (1, 2, 0),
) -> list[dict[str, Any]]:
    """Return a record view with Q intact, zeroed, cyclically permuted or removed."""
    if q_width <= 0 or q_width % 2:
        raise ValueError("q_width must be a positive even number")
    relations = q_width // 2
    if sorted(int(value) for value in permutation) != list(range(relations)):
        raise ValueError("permutation must contain every relation channel once")
    if mode not in {"full", "zero", "permute", "raw_only"}:
        raise ValueError(f"unknown Q transform: {mode}")
    order = [int(value) for value in permutation]
    order += [relations + int(value) for value in permutation]
    transformed: list[dict[str, Any]] = []
    for record in records:
        psi = record["psi"]
        if psi.shape[-1] < q_width:
            raise ValueError("record feature width is smaller than Q")
        if mode == "raw_only":
            changed = psi[:, :-q_width].clone()
        else:
            changed = psi.clone()
            if mode == "zero":
                changed[:, -q_width:] = 0
            elif mode == "permute":
                changed[:, -q_width:] = psi[:, -q_width:][:, order]
        transformed.append({**record, "psi": changed})
    return transformed


@torch.no_grad()
def classifier_predictions(
    model: EndpointClassifier,
    records: Sequence[dict[str, Any]],
    indices: Iterable[int],
    *,
    q_only: bool,
    q_width: int,
    arm: str,
    window: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in indices:
        record = records[index]
        features, _, events, labels = pack_records(records, [index])
        if q_only:
            features = features[:, :, -q_width:]
        logits = model(features, events)
        for position, (event, label, prediction) in enumerate(
            zip(events.tolist(), labels.tolist(), logits.argmax(-1).tolist(), strict=True)
        ):
            _, source, target = event
            rows.append({
                "arm": arm,
                "window": window,
                "example_id": record["example_id"],
                "pair_id": record["pair_id"],
                "event_position": position,
                "source": source,
                "target": target,
                "label": label,
                "prediction": prediction,
                "correct": prediction == label,
            })
    return rows


@torch.no_grad()
def phase_predictions(
    phase: PhaseAnchorNetwork,
    records: Sequence[dict[str, Any]],
    indices: Iterable[int],
    *,
    arm: str,
    window: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in indices:
        record = records[index]
        features, mask, events, labels = pack_records(records, [index])
        logits = relation_logits(
            phase(features, mask), phase.offsets, events, phase.config.temperature,
        )
        for position, (event, label, prediction) in enumerate(
            zip(events.tolist(), labels.tolist(), logits.argmax(-1).tolist(), strict=True)
        ):
            _, source, target = event
            rows.append({
                "arm": arm,
                "window": window,
                "example_id": record["example_id"],
                "pair_id": record["pair_id"],
                "event_position": position,
                "source": source,
                "target": target,
                "label": label,
                "prediction": prediction,
                "correct": prediction == label,
            })
    return rows


def evaluate_classifier_arm(
    model: EndpointClassifier,
    records: Sequence[dict[str, Any]],
    indices: Sequence[int],
    *,
    q_only: bool,
    q_width: int,
    batch_size: int,
    arm: str,
    window: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = evaluate_control(
        model, records, indices, q_only=q_only, q_width=q_width,
        batch_size=batch_size,
    )
    return metrics, classifier_predictions(
        model, records, indices, q_only=q_only, q_width=q_width,
        arm=arm, window=window,
    )


def evaluate_phase_arm(
    phase: PhaseAnchorNetwork,
    records: Sequence[dict[str, Any]],
    indices: Sequence[int],
    *,
    arm: str,
    window: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return (
        evaluate_calibration(phase, records, indices),
        phase_predictions(phase, records, indices, arm=arm, window=window),
    )


def prediction_change_rate(
    baseline: Sequence[Mapping[str, Any]],
    ablated: Sequence[Mapping[str, Any]],
) -> float:
    def keyed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], int]:
        return {
            (str(row["example_id"]), int(row["event_position"])): int(row["prediction"])
            for row in rows
        }
    left, right = keyed(baseline), keyed(ablated)
    if not left or left.keys() != right.keys():
        raise ValueError("prediction panels must have identical nonempty identities")
    return sum(left[key] != right[key] for key in left) / len(left)


def explicit_target_audit(examples: Sequence[Any]) -> dict[str, Any]:
    """Measure whether relation supervision is literally present in raw tokens."""
    exposed = 0
    total = 0
    pair_relations: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    pair_answers: dict[str, list[int]] = defaultdict(list)
    for example in examples:
        labels: list[int] = []
        for edge in example.physical_edges:
            total += 1
            labels.append(int(edge.relation_id))
            expected = RELATION_SURFACES[int(edge.relation_id)]
            exposed += int(example.tokens[int(edge.event_anchor_position)] == expected)
        pair_relations[str(example.pair_id)].append(tuple(labels))
        pair_answers[str(example.pair_id)].append(int(example.answer_id))
    complete_pairs = {
        pair_id: values for pair_id, values in pair_relations.items() if len(values) == 2
    }
    identical = sum(values[0] == values[1] for values in complete_pairs.values())
    supported = [
        pair_id for pair_id, answers in pair_answers.items()
        if len(answers) == 2 and 2 not in answers
    ]
    flips = sum(len(set(pair_answers[pair_id])) == 2 for pair_id in supported)
    return {
        "events": total,
        "relation_surface_at_event_anchor_count": exposed,
        "relation_surface_at_event_anchor_rate": exposed / max(1, total),
        "direct_token_label_exposure": exposed == total and total > 0,
        "complete_counterfactual_pairs": len(complete_pairs),
        "pairs_with_identical_relation_sequence": identical,
        "pair_relation_sequence_identity_rate": identical / max(1, len(complete_pairs)),
        "supported_counterfactual_pairs": len(supported),
        "supported_pairs_with_answer_flip": flips,
        "supported_answer_flip_rate": flips / max(1, len(supported)),
    }


def relation_accuracy_by_answer_outcome(
    relation_rows: Sequence[Mapping[str, Any]],
    answer_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    outcomes = {str(row["example_id"]): bool(row["correct"]) for row in answer_rows}
    grouped: dict[str, list[bool]] = {"answer_correct": [], "answer_wrong": []}
    for row in relation_rows:
        key = "answer_correct" if outcomes[str(row["example_id"])] else "answer_wrong"
        grouped[key].append(bool(row["correct"]))
    return {
        key: {
            "events": len(values),
            "relation_accuracy": sum(values) / len(values) if values else None,
        }
        for key, values in grouped.items()
    }
