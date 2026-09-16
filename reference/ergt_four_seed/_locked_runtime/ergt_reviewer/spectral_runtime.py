"""Fail-open numerics for the detached reviewer spectral observer.

This module is intentionally outside the checkpoint-bound training source
manifest.  It only replaces a reporting observer at suite runtime; no value
returned here is read by training, routing, transport, hard measurement, or
checkpoint selection.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import torch
import torch.nn.functional as F

POLICY: Final = "detached_cpu_float64_symmetric_eigvalsh_fail_open_v1"

_COUNTERS: dict[str, int] = {}


def reset_spectral_observer_diagnostics() -> None:
    """Reset process-local reporting counters before a reviewer run."""

    _COUNTERS.clear()
    _COUNTERS.update(
        {
            "calls": 0,
            "cuda_input_calls": 0,
            "graphs_considered": 0,
            "graphs_solved_cpu_float64": 0,
            "graphs_without_spectral_mass": 0,
            "graphs_unavailable_after_numerical_failure": 0,
            "nonfinite_values_sanitized": 0,
            "cuda_eigensolver_calls": 0,
        }
    )


def spectral_observer_diagnostics() -> dict[str, Any]:
    """Return the non-gating observer audit for artifact export."""

    return {
        "policy": POLICY,
        "reporting_only": True,
        "fail_open": True,
        "governing_answer_path_changed": False,
        "training_source_hashes_changed": False,
        "checkpoint_compatibility_preserved": True,
        **_COUNTERS,
    }


def _zero_result(reference: torch.Tensor, batch: int) -> dict[str, torch.Tensor]:
    zero = reference.new_zeros(batch)
    return {
        "event_spectral_entropy": zero,
        "event_spectral_effective_rank": zero,
        "event_spectral_gap": zero,
        "event_spectral_world_diversity": zero,
        "event_spectral_valid_world_fraction": zero,
        "event_spectral_numerical_failure_count": zero,
    }


def stable_event_graph_spectral_observer(
    outputs: Mapping[str, torch.Tensor],
    selected_event_indices: torch.Tensor,
    event_presence: torch.Tensor,
    *,
    event_presence_floor: float,
    eps: float,
) -> dict[str, torch.Tensor]:
    """Measure event-graph spectra without invoking a CUDA eigensolver.

    Graph extraction follows the registered V4 observer.  Only the detached
    eigendecomposition is made numerically conservative.  A graph whose
    spectrum remains unavailable is omitted from observer aggregation and
    counted in the returned/runtime audit; it never aborts model evaluation.
    """

    with torch.no_grad():
        _COUNTERS["calls"] += 1
        world_weight = outputs["program_world_edge_weight"].detach()
        if world_weight.device.type == "cuda":
            _COUNTERS["cuda_input_calls"] += 1
        world_mask = (
            outputs["program_world_sparse_mask"].detach().bool()
            & outputs["program_world_cone"].detach().bool()
        )
        world_weight = world_weight * world_mask.to(world_weight.dtype)
        selected = selected_event_indices.detach()
        active_event = event_presence.detach() >= float(event_presence_floor)
        batch, worlds, _, token_count = world_weight.shape
        if selected.dim() != 2 or selected.size(0) != batch:
            raise ValueError("selected event indices must have shape [B, E]")
        event_count = selected.size(1)
        if event_count == 0:
            return _zero_result(world_weight, batch)

        selected = selected.clamp(0, token_count - 1)
        row_index = selected[:, None, :, None].expand(-1, worlds, -1, token_count)
        event_rows = torch.gather(world_weight, 2, row_index)
        column_index = selected[:, None, None, :].expand(-1, worlds, event_count, -1)
        adjacency = torch.gather(event_rows, 3, column_index)
        valid_pair = active_event[:, None, :, None] & active_event[:, None, None, :]
        off_diagonal = ~torch.eye(
            event_count,
            dtype=torch.bool,
            device=world_weight.device,
        )[None, None]
        adjacency = 0.5 * (adjacency + adjacency.transpose(-1, -2))
        adjacency = adjacency * (valid_pair & off_diagonal).to(adjacency.dtype)

        # One detached transfer per model batch avoids thousands of tiny
        # device synchronizations while keeping every eigensolver call off CUDA.
        adjacency64 = adjacency.to(device="cpu", dtype=torch.float64)
        active_event_cpu = active_event.to(device="cpu")
        finite = torch.isfinite(adjacency64)
        nonfinite_count = int((~finite).sum().item())
        if nonfinite_count:
            _COUNTERS["nonfinite_values_sanitized"] += nonfinite_count
            adjacency64 = torch.nan_to_num(adjacency64, nan=0.0, posinf=0.0, neginf=0.0)
        adjacency64 = adjacency64.clamp_min(0.0)
        adjacency64 = 0.5 * (adjacency64 + adjacency64.transpose(-1, -2))

        entropy_by_world = torch.zeros((batch, worlds), dtype=torch.float64)
        rank_by_world = torch.zeros((batch, worlds), dtype=torch.float64)
        gap_by_world = torch.zeros((batch, worlds), dtype=torch.float64)
        valid_world = torch.zeros((batch, worlds), dtype=torch.bool)
        failure_by_row = torch.zeros(batch, dtype=torch.float64)
        signature = torch.zeros((batch, worlds, 16), dtype=torch.float64)

        for row in range(batch):
            event_mask = active_event_cpu[row]
            valid_count = int(event_mask.sum().item())
            if valid_count < 2:
                continue
            for world in range(worlds):
                _COUNTERS["graphs_considered"] += 1
                graph64 = adjacency64[row, world][event_mask][:, event_mask]
                degree = graph64.sum(dim=-1)
                nonisolated = degree > float(eps)
                inverse_sqrt = degree.clamp_min(float(eps)).rsqrt()
                normalized = inverse_sqrt[:, None] * graph64 * inverse_sqrt[None, :]
                laplacian = torch.diag(nonisolated.to(torch.float64)) - normalized
                laplacian = 0.5 * (laplacian + laplacian.transpose(0, 1))
                try:
                    eigenvalues = torch.linalg.eigvalsh(laplacian).clamp_min(0.0)
                except RuntimeError:
                    _COUNTERS["graphs_unavailable_after_numerical_failure"] += 1
                    failure_by_row[row] += 1.0
                    continue
                if not bool(torch.isfinite(eigenvalues).all()):
                    _COUNTERS["graphs_unavailable_after_numerical_failure"] += 1
                    failure_by_row[row] += 1.0
                    continue
                _COUNTERS["graphs_solved_cpu_float64"] += 1
                mass = eigenvalues.sum()
                if float(mass.item()) <= float(eps):
                    _COUNTERS["graphs_without_spectral_mass"] += 1
                    continue
                probability = eigenvalues / mass
                raw_entropy = -(probability * probability.clamp_min(float(eps)).log()).sum()
                entropy = raw_entropy / torch.tensor(
                    float(valid_count), dtype=torch.float64
                ).log().clamp_min(float(eps))
                entropy_by_world[row, world] = entropy
                rank_by_world[row, world] = raw_entropy.exp()
                gap_by_world[row, world] = eigenvalues[1]
                width = min(16, valid_count)
                signature[row, world, :width] = eigenvalues[:width]
                valid_world[row, world] = True

        valid_weight = valid_world.to(torch.float64)
        denominator = valid_weight.sum(dim=-1).clamp_min(1.0)
        entropy = (entropy_by_world * valid_weight).sum(dim=-1) / denominator
        effective_rank = (rank_by_world * valid_weight).sum(dim=-1) / denominator
        spectral_gap = (gap_by_world * valid_weight).sum(dim=-1) / denominator

        normalized_signature = F.normalize(signature, dim=-1, eps=float(eps))
        cosine = torch.einsum("bwi,bvi->bwv", normalized_signature, normalized_signature)
        world_pair = valid_world[:, :, None] & valid_world[:, None, :]
        world_pair = (
            world_pair
            & torch.triu(
                torch.ones((worlds, worlds), dtype=torch.bool),
                diagonal=1,
            )[None]
        )
        pair_count = world_pair.sum(dim=(-1, -2)).clamp_min(1).to(torch.float64)
        diversity = ((1.0 - cosine) * world_pair.to(cosine.dtype)).sum(dim=(-1, -2))
        diversity = diversity / pair_count
        diversity = torch.where(
            world_pair.any(dim=(-1, -2)),
            diversity,
            torch.zeros_like(diversity),
        )

        def restore(value: torch.Tensor) -> torch.Tensor:
            return value.to(device=world_weight.device, dtype=world_weight.dtype).detach()

        return {
            "event_spectral_entropy": restore(entropy),
            "event_spectral_effective_rank": restore(effective_rank),
            "event_spectral_gap": restore(spectral_gap),
            "event_spectral_world_diversity": restore(diversity),
            "event_spectral_valid_world_fraction": restore(
                valid_weight.sum(dim=-1) / float(worlds)
            ),
            "event_spectral_numerical_failure_count": restore(failure_by_row),
        }


def install_stable_event_spectral_observer() -> None:
    """Install the reporting-only implementation into the loaded solver module."""

    from . import native_solver

    native_solver.event_graph_spectral_observer = stable_event_graph_spectral_observer


reset_spectral_observer_diagnostics()


__all__ = [
    "POLICY",
    "install_stable_event_spectral_observer",
    "reset_spectral_observer_diagnostics",
    "spectral_observer_diagnostics",
    "stable_event_graph_spectral_observer",
]
