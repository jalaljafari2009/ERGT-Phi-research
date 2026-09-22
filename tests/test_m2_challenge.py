from types import SimpleNamespace

import torch

from ergt_phi.m2_challenge import (
    ScalarReadout,
    challenge_support,
    evaluate_scalar_readout,
    matched_scalar_hidden,
    representation_vectors,
    train_scalar_readout,
)


def test_challenge_support_counts_variation_and_pairs():
    rows = []
    for index, (drop, wrong) in enumerate(((0.0, False), (1.0, True), (2.0, False), (3.0, True))):
        rows.append({
            "pair_id": f"p{index // 2}", "response_margin_drop": drop,
            "answer_became_wrong": wrong, "event_chain_degraded": index % 2 == 0,
            "path_coverage_drop": float(index == 3),
            "full": {"answer_correct": True}, "challenge": {"answer_correct": not wrong},
        })
    result = challenge_support(rows)
    assert result["pairs"] == 2
    assert result["answer_failures"] == 2
    assert result["event_chain_degradations"] == 2
    assert result["response_margin_drop_std"] > 0


def test_representation_vectors_pool_expected_nodes():
    psi = torch.arange(40, dtype=torch.float32).reshape(4, 10)
    records = [{"psi": psi}]
    example = SimpleNamespace(base=SimpleNamespace(source_position=0, candidate_positions=(2, 3)))
    q = representation_vectors(records, [example], mode="q_only", q_width=6)
    raw = representation_vectors(records, [example], mode="raw_only", q_width=6)
    assert q.shape == (1, 30)
    assert raw.shape == (1, 20)
    torch.testing.assert_close(q[0, :6], psi[0, -6:])


def test_matched_width_and_scalar_training_are_deterministic():
    width = matched_scalar_hidden(6, 100)
    model = ScalarReadout(6, width, seed=4)
    assert abs(sum(p.numel() for p in model.parameters()) - 100) <= 8
    vectors = torch.arange(48, dtype=torch.float32).reshape(8, 6) / 10
    targets = torch.linspace(-1, 1, 8)
    first, info_a = train_scalar_readout(
        vectors, targets, target_parameters=100, seed=7, epochs=2, batch_size=4,
        learning_rate=0.01, weight_decay=0.0,
    )
    second, info_b = train_scalar_readout(
        vectors, targets, target_parameters=100, seed=7, epochs=2, batch_size=4,
        learning_rate=0.01, weight_decay=0.0,
    )
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert info_a == info_b and info_a["fresh_targets_used"] is False
    metrics, prediction = evaluate_scalar_readout(
        first, vectors, targets, target_mean=info_a["target_mean"], target_std=info_a["target_std"],
    )
    assert prediction.shape == targets.shape
    assert metrics["examples"] == 8
