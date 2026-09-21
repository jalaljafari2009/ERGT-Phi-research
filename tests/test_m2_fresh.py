from types import SimpleNamespace

import torch

from ergt_phi.m2_fresh import (
    EndpointClassifier,
    confirmation_windows,
    evaluate_control,
    matched_hidden_dim,
    parameter_count,
    phase_window_checks,
    train_control,
)
from ergt_phi.shadow_phase import CalibrationConfig, PhaseAnchorNetwork


def _examples():
    result = []
    for split in ("hop_1_a", "hop_2_b"):
        for pair in range(4):
            for variant in range(2):
                result.append(SimpleNamespace(
                    pair_id=f"{split}-pair-{pair}",
                    base=SimpleNamespace(split=split),
                    counterfactual_variant=variant,
                ))
    return result


def _records():
    generator = torch.Generator().manual_seed(41)
    records = []
    for index in range(8):
        records.append({
            "psi": torch.randn(5, 10, generator=generator),
            "events": torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long),
            "labels": torch.tensor([0, 1, 2], dtype=torch.long),
            "example_id": f"example-{index}",
            "pair_id": f"pair-{index // 2}",
        })
    return records


def test_confirmation_windows_keep_pairs_together_and_are_deterministic():
    examples = _examples()
    first, second = confirmation_windows(examples, seed=17)
    assert (first, second) == confirmation_windows(examples, seed=17)
    assert set(first).isdisjoint(second)
    assert sorted(first + second) == list(range(len(examples)))
    first_pairs = {examples[index].pair_id for index in first}
    second_pairs = {examples[index].pair_id for index in second}
    assert first_pairs.isdisjoint(second_pairs)


def test_matched_hidden_dim_minimizes_parameter_gap():
    target = 700
    hidden = matched_hidden_dim(10, 3, target)
    selected = parameter_count(EndpointClassifier(10, hidden, 3, seed=1))
    neighbours = [
        parameter_count(EndpointClassifier(10, candidate, 3, seed=1))
        for candidate in {max(1, hidden - 1), hidden, hidden + 1}
    ]
    assert abs(selected - target) == min(abs(value - target) for value in neighbours)


def test_registered_shape_needs_no_phase_width_35():
    phase = PhaseAnchorNetwork(62, 8, 3, CalibrationConfig())
    assert parameter_count(phase) == 4700
    assert matched_hidden_dim(62, 3, parameter_count(phase)) == 35
    control = EndpointClassifier(62, 35, 3, seed=1)
    assert parameter_count(control) == 4731
    assert abs(parameter_count(control) - parameter_count(phase)) / parameter_count(phase) < 0.01


def test_controls_train_on_supplied_historical_indices_and_evaluate_elsewhere():
    records = _records()
    model, audit = train_control(
        records, [0, 1, 2, 3], q_only=True, q_width=4, hidden_dim=8,
        relations=3, seed=31, epochs=2, batch_size=2,
        learning_rate=0.002, weight_decay=0.0001,
    )
    metrics = evaluate_control(
        model, records, [4, 5, 6, 7], q_only=True, q_width=4, batch_size=2,
    )
    assert audit["fresh_labels_used"] is False
    assert audit["examples"] == 4
    assert audit["steps"] == 4
    assert metrics["events"] == 12
    assert len(metrics["per_class_recall"]) == 3


def test_phase_window_checks_are_explicit():
    metrics = {
        "class_counts": [4, 4, 4],
        "balanced_accuracy": 0.75,
        "per_class_recall": [0.75, 0.75, 0.75],
        "ce": 0.9,
        "minimum_offset_separation": 1.2,
        "phase_resultant": 0.5,
    }
    checks = phase_window_checks(metrics, relations=3)
    assert all(checks.values())
    metrics["per_class_recall"][1] = 0.49
    assert phase_window_checks(metrics, relations=3)["minimum_class_recall"] is False
