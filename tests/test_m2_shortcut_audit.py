from types import SimpleNamespace

import pytest
import torch

from ergt_phi.m2_shortcut_audit import (
    explicit_target_audit,
    prediction_change_rate,
    transform_q_records,
)


def records():
    return [{
        "psi": torch.arange(20, dtype=torch.float32).reshape(2, 10),
        "events": torch.tensor([[0, 1]]),
        "labels": torch.tensor([0]),
        "example_id": "a",
        "pair_id": "p",
    }]


def test_q_zero_permutation_and_raw_views_do_not_mutate_source():
    source = records()
    original = source[0]["psi"].clone()
    zero = transform_q_records(source, q_width=6, mode="zero")
    permuted = transform_q_records(source, q_width=6, mode="permute")
    raw = transform_q_records(source, q_width=6, mode="raw_only")
    torch.testing.assert_close(source[0]["psi"], original)
    torch.testing.assert_close(zero[0]["psi"][:, -6:], torch.zeros(2, 6))
    torch.testing.assert_close(
        permuted[0]["psi"][:, -6:], original[:, -6:][:, [1, 2, 0, 4, 5, 3]],
    )
    torch.testing.assert_close(raw[0]["psi"], original[:, :-6])


def test_transform_rejects_non_permutation():
    with pytest.raises(ValueError, match="permutation"):
        transform_q_records(records(), q_width=6, mode="permute", permutation=(0, 0, 2))


def test_prediction_change_requires_same_panel():
    a = [{"example_id": "e", "event_position": 0, "prediction": 0}]
    b = [{"example_id": "e", "event_position": 0, "prediction": 1}]
    assert prediction_change_rate(a, b) == 1.0
    with pytest.raises(ValueError, match="identical"):
        prediction_change_rate(a, [])


def test_explicit_target_audit_detects_literal_relation_and_pair_mismatch(monkeypatch):
    import ergt_phi.m2_shortcut_audit as module
    monkeypatch.setattr(module, "RELATION_SURFACES", ("none", "rel_a", "rel_b"))
    def example(name, answer):
        return SimpleNamespace(
            pair_id="pair",
            answer_id=answer,
            tokens=("node", "rel_a", "node"),
            physical_edges=(SimpleNamespace(relation_id=1, event_anchor_position=1),),
        )
    audit = explicit_target_audit((example("a", 0), example("b", 1)))
    assert audit["direct_token_label_exposure"] is True
    assert audit["pair_relation_sequence_identity_rate"] == 1.0
    assert audit["supported_answer_flip_rate"] == 1.0
