from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from ergt_phi.checkpoint import save
from ergt_phi.m2_q_contract import load_m2_q_contract
from ergt_phi.m2_q_training import (
    CalibrationOutcome,
    calibration_config,
    train_with_checkpoints,
    verify_resume_in_fresh_process,
    verify_selected_reload_in_fresh_process,
)
from ergt_phi.shadow_metrics import evaluate_calibration
from ergt_phi.shadow_phase import CalibrationConfig, PhaseAnchorNetwork


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "m2_q_checkpoint_worker.py"


def records() -> list[dict]:
    generator = torch.Generator().manual_seed(73)
    result = []
    for index in range(4):
        result.append({
            "psi": torch.randn(5, 4, generator=generator),
            "events": torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long),
            "labels": torch.tensor([0, 1, 2], dtype=torch.long),
            "example_id": f"example-{index}",
            "pair_id": f"pair-{index}",
        })
    return result


def metrics(*, ready: bool, ce: float = 1.0) -> dict:
    recall = [0.8, 0.8, 0.8] if ready else [0.4, 0.4, 0.4]
    return {
        "ce": ce,
        "accuracy": sum(recall) / 3,
        "balanced_accuracy": sum(recall) / 3,
        "per_class_recall": recall,
        "class_counts": [2, 2, 2],
        "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "events": 6,
        "minimum_offset_separation": 1.5,
        "maximum_offset_displacement": 0.1,
        "phase_resultant": 0.5,
        "phase_resultant_per_world": [0.5, 0.5],
    }


def small_config(**changes) -> CalibrationConfig:
    values = dict(hidden_dim=8, epochs=5, batch_size=1, required_windows=2)
    values.update(changes)
    return CalibrationConfig(**values)


def test_contract_maps_to_executable_calibration_config() -> None:
    contract = load_m2_q_contract(ROOT / "configs" / "m2_q_v2.json")
    config = calibration_config(contract)
    assert config.init_seed == 22092026
    assert config.required_windows == 2
    assert config.readiness_balanced_accuracy == 0.7
    assert config.minimum_offset_separation == 1.0


def test_first_qualified_checkpoint_is_immediately_frozen(tmp_path, monkeypatch) -> None:
    data = records()
    fit, monitor = [0, 1], [2, 3]
    monitor_calls = 0

    def fake_evaluate(_phase, _records, indices):
        nonlocal monitor_calls
        if list(indices) == monitor:
            monitor_calls += 1
            # initial, epoch 1, then two consecutive qualified windows.
            qualified = monitor_calls >= 3
            return metrics(ready=qualified, ce=0.8 if qualified else 1.0)
        return metrics(ready=True, ce=0.7)

    monkeypatch.setattr("ergt_phi.m2_q_training.evaluate_calibration", fake_evaluate)
    outcome = train_with_checkpoints(
        data, fit, monitor, small_config(),
        worlds=2, relations=3, order_seed=19,
        checkpoint_contract={"stage": "M2-Q-test"}, output_dir=tmp_path,
    )
    assert outcome.qualified
    assert outcome.selected_epoch == 3
    assert len(outcome.curves) == 3
    assert outcome.freeze_update_blocked
    selected = torch.load(tmp_path / "selected_full_state.pt", weights_only=False)
    latest = torch.load(tmp_path / "latest_full_state.pt", weights_only=False)
    assert selected["progress"]["frozen"] is True
    assert selected["progress"]["selected_epoch"] == 3
    assert latest["progress"]["frozen"] is False
    assert all(value is False for value in selected["requires_grad"].values())
    for name in selected["model"]:
        torch.testing.assert_close(selected["model"][name], latest["model"][name], rtol=0, atol=0)


def test_resume_equivalence_uses_a_fresh_process(tmp_path) -> None:
    report = verify_resume_in_fresh_process(
        records(), [0, 1, 2], small_config(batch_size=1),
        worlds=2, relations=3,
        checkpoint_contract={"stage": "M2-Q-resume-test"},
        output_dir=tmp_path, worker_script=WORKER,
    )
    assert report["pass"] is True
    assert report["fresh_process"] is True
    assert report["completed_steps"] == 2


def test_selected_checkpoint_reloads_frozen_and_reproduces_metrics(tmp_path) -> None:
    data = records()
    config = small_config()
    phase = PhaseAnchorNetwork(4, 2, 3, config)
    optimizer = torch.optim.AdamW(phase.parameters(), lr=config.learning_rate)
    phase.freeze()
    progress = {
        "step": 4, "epoch": 2, "cursor": 2, "order": [0, 1],
        "streak": 2, "frozen": True, "selected_epoch": 2, "selected_step": 4,
    }
    checkpoint_contract = {"stage": "M2-Q-reload-test"}
    save(
        tmp_path / "selected_full_state.pt", phase, optimizer,
        contract=checkpoint_contract, progress=progress,
    )
    monitor = [2, 3]
    selected_metrics = evaluate_calibration(phase, data, monitor)
    outcome = CalibrationOutcome(
        phase=phase, optimizer=optimizer,
        initial_fit=deepcopy(selected_metrics), initial_monitor=deepcopy(selected_metrics),
        curves=[{"monitor": selected_metrics}], qualified=True,
        selected_epoch=2, selected_step=4,
        checkpoint_paths={"selected_full_state": "selected_full_state.pt"},
        freeze_update_blocked=True,
    )
    report = verify_selected_reload_in_fresh_process(
        outcome, data, monitor, config,
        worlds=2, relations=3, checkpoint_contract=checkpoint_contract,
        output_dir=tmp_path, worker_script=WORKER,
    )
    assert report["pass"] is True
    assert report["metric_reproduced_exactly"] is True
    assert report["parameters_frozen"] is True
    assert report["modules_eval"] is True
