"""Checkpoint-safe training primitives for the M2-Q v2 offline calibration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .checkpoint import capture, load, restore, save
from .shadow_data import pack_records
from .shadow_metrics import evaluate_calibration
from .shadow_phase import (
    CalibrationConfig,
    PhaseAnchorNetwork,
    calibration_step,
    readiness,
)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_config(contract: dict[str, Any]) -> CalibrationConfig:
    """Build the executable calibration config from the validated v2 contract."""
    values = contract["calibration"]
    gates = values["readiness"]
    return CalibrationConfig(
        init_seed=int(values["init_seed"]),
        shuffle_seed=int(values["shuffle_seed"]),
        split_seed=int(values["split_seed"]),
        epochs=int(values["epochs"]),
        batch_size=int(values["batch_size"]),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        offset_penalty=float(values["offset_penalty"]),
        readiness_balanced_accuracy=float(gates["balanced_accuracy_min"]),
        readiness_min_recall=float(gates["per_class_recall_min"]),
        readiness_ce_reduction=float(gates["ce_reduction_min"]),
        minimum_offset_separation=float(gates["offset_separation_radians_min"]),
        maximum_phase_resultant=float(gates["phase_resultant_max"]),
        required_windows=int(gates["consecutive_windows"]),
    )


def batch_tensors(
    records: Sequence[dict[str, Any]],
    indices: Sequence[int],
    *,
    shuffle_labels: bool = False,
    shuffle_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = pack_records(records, indices)
    if not shuffle_labels:
        return tensors
    psi, mask, events, labels = tensors
    generator = torch.Generator().manual_seed(shuffle_seed)
    labels = labels[torch.randperm(labels.numel(), generator=generator)]
    return psi, mask, events, labels


def _new_model(
    input_dim: int,
    worlds: int,
    relations: int,
    config: CalibrationConfig,
) -> tuple[PhaseAnchorNetwork, torch.optim.Optimizer]:
    phase = PhaseAnchorNetwork(input_dim, worlds, relations, config)
    optimizer = torch.optim.AdamW(
        phase.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    return phase, optimizer


def _deadline_guard(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("registered M2-Q runtime budget exceeded")


@dataclass
class CalibrationOutcome:
    phase: PhaseAnchorNetwork
    optimizer: torch.optim.Optimizer
    initial_fit: dict[str, Any]
    initial_monitor: dict[str, Any]
    curves: list[dict[str, Any]]
    qualified: bool
    selected_epoch: int | None
    selected_step: int | None
    checkpoint_paths: dict[str, str]
    freeze_update_blocked: bool


def assert_frozen_update_blocked(
    phase: PhaseAnchorNetwork,
    optimizer: torch.optim.Optimizer,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    checkpoint_contract: dict[str, Any],
    progress: dict[str, Any],
) -> bool:
    before = {name: value.detach().clone() for name, value in phase.state_dict().items()}
    try:
        calibration_step(
            phase, optimizer, tensors,
            contract=checkpoint_contract, progress=progress,
        )
    except RuntimeError as exc:
        if "frozen" not in str(exc):
            raise
    else:
        raise AssertionError("frozen phase accepted an optimizer update")
    for name, value in phase.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    return True


def train_with_checkpoints(
    records: Sequence[dict[str, Any]],
    fit_indices: Sequence[int],
    monitor_indices: Sequence[int],
    config: CalibrationConfig,
    *,
    worlds: int,
    relations: int,
    order_seed: int,
    checkpoint_contract: dict[str, Any],
    output_dir: str | Path,
    shuffle_labels: bool = False,
    deadline: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> CalibrationOutcome:
    """Train until the first qualifying streak and freeze that exact state."""
    if not records or not fit_indices or not monitor_indices:
        raise ValueError("M2-Q calibration requires records, fit and monitor partitions")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_dim = int(records[0]["psi"].shape[-1])
    phase, optimizer = _new_model(input_dim, worlds, relations, config)
    progress: dict[str, Any] = {
        "step": 0,
        "epoch": 0,
        "cursor": 0,
        "order": [],
        "streak": 0,
        "frozen": False,
        "selected_epoch": None,
        "selected_step": None,
        "shuffle_labels": bool(shuffle_labels),
    }
    save(
        output / "initial_full_state.pt", phase, optimizer,
        contract=checkpoint_contract, progress=progress,
    )
    initial_monitor = evaluate_calibration(phase, records, monitor_indices)
    initial_fit = evaluate_calibration(phase, records, fit_indices)
    curves: list[dict[str, Any]] = []
    selected_epoch: int | None = None
    selected_step: int | None = None

    for epoch in range(config.epochs):
        _deadline_guard(deadline)
        order = list(fit_indices)
        random.Random(order_seed + epoch).shuffle(order)
        progress.update(epoch=epoch, order=order, cursor=0)
        phase.train()
        for start in range(0, len(order), config.batch_size):
            _deadline_guard(deadline)
            stop = min(start + config.batch_size, len(order))
            tensors = batch_tensors(
                records, order[start:stop], shuffle_labels=shuffle_labels,
                shuffle_seed=order_seed + 100000 + epoch * 1000 + start,
            )
            calibration_step(
                phase, optimizer, tensors,
                contract=checkpoint_contract, progress=progress,
            )
            progress["cursor"] = stop

        monitor = evaluate_calibration(phase, records, monitor_indices)
        fit_metrics = evaluate_calibration(phase, records, fit_indices)
        ready, checks = readiness(monitor, initial_monitor["ce"], config)
        progress["streak"] = progress["streak"] + 1 if ready else 0
        progress["epoch"] = epoch + 1
        row = {
            "epoch": epoch + 1,
            "step": progress["step"],
            "fit": fit_metrics,
            "monitor": monitor,
            "checks": checks,
            "all_readiness_checks": bool(ready),
            "streak": progress["streak"],
        }
        curves.append(row)
        # latest is a resumable optimizer-step boundary and remains trainable.
        save(
            output / "latest_full_state.pt", phase, optimizer,
            contract=checkpoint_contract, progress=progress,
        )
        if progress_callback is not None:
            progress_callback(row)
        if progress["streak"] >= config.required_windows:
            selected_epoch = epoch + 1
            selected_step = int(progress["step"])
            progress.update(
                frozen=True,
                selected_epoch=selected_epoch,
                selected_step=selected_step,
            )
            phase.freeze()
            save(
                output / "selected_full_state.pt", phase, optimizer,
                contract=checkpoint_contract, progress=progress,
            )
            break

    qualified = selected_epoch is not None
    freeze_update_blocked = False
    if qualified:
        probe_indices = list(fit_indices)[: min(config.batch_size, len(fit_indices))]
        freeze_update_blocked = assert_frozen_update_blocked(
            phase, optimizer, batch_tensors(records, probe_indices),
            checkpoint_contract=checkpoint_contract, progress=progress,
        )
    else:
        save(
            output / "rejected_full_state.pt", phase, optimizer,
            contract=checkpoint_contract, progress=progress,
        )
        restored = load(
            output / "initial_full_state.pt", phase, optimizer,
            contract=checkpoint_contract,
        )
        restored.update(frozen=True, rollback_reason="readiness_not_reached")
        phase.freeze()
        save(
            output / "restored_initial_full_state.pt", phase, optimizer,
            contract=checkpoint_contract, progress=restored,
        )

    checkpoints = {
        path.stem: path.name
        for path in sorted(output.glob("*_full_state.pt"))
    }
    return CalibrationOutcome(
        phase=phase,
        optimizer=optimizer,
        initial_fit=initial_fit,
        initial_monitor=initial_monitor,
        curves=curves,
        qualified=qualified,
        selected_epoch=selected_epoch,
        selected_step=selected_step,
        checkpoint_paths=checkpoints,
        freeze_update_blocked=freeze_update_blocked,
    )


def _exact(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError("state dictionaries have different keys")
        for key in left:
            _exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError("state sequences have different lengths")
        for first, second in zip(left, right):
            _exact(first, second)
    elif left != right:
        raise AssertionError(f"state mismatch: {left!r} != {right!r}")


def verify_resume_in_fresh_process(
    records: Sequence[dict[str, Any]],
    fit_indices: Sequence[int],
    config: CalibrationConfig,
    *,
    worlds: int,
    relations: int,
    checkpoint_contract: dict[str, Any],
    output_dir: str | Path,
    worker_script: str | Path,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Compare two optimizer steps with a one-step checkpoint/process break."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    order = list(fit_indices)
    random.Random(config.shuffle_seed).shuffle(order)
    batches = [
        order[start:start + config.batch_size]
        for start in range(0, len(order), config.batch_size)
    ]
    if len(batches) < 2:
        raise ValueError("resume verification requires at least two fit batches")
    first = batch_tensors(records, batches[0])
    second = batch_tensors(records, batches[1])
    input_dim = int(records[0]["psi"].shape[-1])
    phase, optimizer = _new_model(input_dim, worlds, relations, config)
    progress: dict[str, Any] = {
        "step": 0, "epoch": 0, "cursor": 0, "order": order,
        "streak": 0, "frozen": False,
    }
    calibration_step(
        phase, optimizer, first,
        contract=checkpoint_contract, progress=progress,
    )
    progress["cursor"] = len(batches[0])
    cut = output / "resume_cut_full_state.pt"
    save(cut, phase, optimizer, contract=checkpoint_contract, progress=progress)
    calibration_step(
        phase, optimizer, second,
        contract=checkpoint_contract, progress=progress,
    )
    progress["cursor"] += len(batches[1])
    continuous = output / "resume_continuous_full_state.pt"
    save(
        continuous, phase, optimizer,
        contract=checkpoint_contract, progress=progress,
    )

    payload = output / "resume_worker_payload.pt"
    torch.save({
        "input_dim": input_dim,
        "worlds": worlds,
        "relations": relations,
        "config": asdict(config),
        "checkpoint_contract": checkpoint_contract,
        "batch": second,
        "cursor_increment": len(batches[1]),
    }, payload)
    resumed = output / "resume_resumed_full_state.pt"
    process = subprocess.run(
        [str(python_executable), "-B", str(worker_script), "resume",
         str(cut), str(payload), str(resumed)],
        check=False, capture_output=True, text=True,
    )
    (output / "resume_worker.log").write_text(
        process.stdout + process.stderr, encoding="utf-8",
    )
    payload.unlink(missing_ok=True)
    if process.returncode != 0:
        raise RuntimeError(f"fresh-process resume worker failed: {process.returncode}")
    continuous_state = torch.load(continuous, map_location="cpu", weights_only=False)
    resumed_state = torch.load(resumed, map_location="cpu", weights_only=False)
    for key in (
        "model", "optimizer", "scheduler", "scaler", "requires_grad",
        "module_training", "rng", "progress", "contract_sha256",
    ):
        _exact(continuous_state[key], resumed_state[key])
    report = {
        "pass": True,
        "fresh_process": True,
        "completed_steps": int(resumed_state["progress"]["step"]),
        "continuous_sha256": sha256(continuous),
        "resumed_sha256": sha256(resumed),
        "exact_fields": [
            "model", "optimizer", "scheduler", "scaler", "requires_grad",
            "module_training", "rng", "progress", "contract_sha256",
        ],
    }
    (output / "resume_verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    return report


def verify_selected_reload_in_fresh_process(
    outcome: CalibrationOutcome,
    records: Sequence[dict[str, Any]],
    monitor_indices: Sequence[int],
    config: CalibrationConfig,
    *,
    worlds: int,
    relations: int,
    checkpoint_contract: dict[str, Any],
    output_dir: str | Path,
    worker_script: str | Path,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    if not outcome.qualified:
        return {"pass": False, "reason": "no_selected_checkpoint"}
    output = Path(output_dir)
    payload = output / "reload_worker_payload.pt"
    torch.save({
        "input_dim": int(records[0]["psi"].shape[-1]),
        "worlds": worlds,
        "relations": relations,
        "config": asdict(config),
        "checkpoint_contract": checkpoint_contract,
        "records": list(records),
        "monitor_indices": list(monitor_indices),
    }, payload)
    report_path = output / "reload_verification.json"
    process = subprocess.run(
        [str(python_executable), "-B", str(worker_script), "reload",
         str(output / "selected_full_state.pt"), str(payload), str(report_path)],
        check=False, capture_output=True, text=True,
    )
    (output / "reload_worker.log").write_text(
        process.stdout + process.stderr, encoding="utf-8",
    )
    payload.unlink(missing_ok=True)
    if process.returncode != 0:
        raise RuntimeError(f"fresh-process reload worker failed: {process.returncode}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected_metrics = outcome.curves[-1]["monitor"]
    _exact(report["monitor"], selected_metrics)
    report["metric_reproduced_exactly"] = True
    report["selected_checkpoint_sha256"] = sha256(output / "selected_full_state.pt")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
