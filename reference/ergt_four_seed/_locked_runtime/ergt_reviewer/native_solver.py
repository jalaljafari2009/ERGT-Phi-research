"""Compiler-free native geometric boundary solver for ERGT-43.

The neural path induces fields, event endpoints, typed operator charge, and
physical initial/boundary conditions from raw tokens.  The answer path then
uses a tensor min-plus evolution over those event fields.  It never creates a
``GeometricProgram`` and never calls the ERGT-42 graph compiler or hard graph
executor.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

from .data_schema import REGISTER_CARDINALITY, RELATION_NAMES, apply_relation
from .world_contract import WORLD_NAMES
from .physics_core import (
    ERGT42Config,
    PhysicsNativeSubstrate,
    RolePreservingGeometricFieldInducer as RolePreservingGeometricCompiler,
    physics_native_training_loss,
)
from .matched_data import (
    ERGT43Batch,
    IDENTITY_FIBRE_BITS,
    RawTokenInputContract,
)
from .product_geodesic import (
    ProductManifoldClosure,
    product_manifold_geodesic_closure,
)

SCHEMA_VERSION: Final = "ergt-reviewer-native-geometric-boundary-solver-v3"
INTERVENTIONS: Final[tuple[str, ...]] = (
    "full",
    "no_action",
    "no_cone",
    "no_boundary",
    "no_transport",
    "no_terminal_mass",
    "no_memory_geometry",
    "no_phi",
    "no_event_backreaction",
    "shuffled_geometry",
    "random_geometry",
    "direct_world_gate",
    "no_multiscale_backbone",
    "no_world_transitions",
    *(f"drop_world_{index}" for index in range(len(WORLD_NAMES))),
    *(f"only_world_{index}" for index in range(len(WORLD_NAMES))),
)


class ERGT43Error(ValueError):
    """Raised when the direct geometric solver contract is violated."""


class RawTokenInputAdapter(nn.Module):
    """Derive semantic state and conserved identity inside the ERGT boundary."""

    def __init__(self, contract: RawTokenInputContract) -> None:
        super().__init__()
        self.contract = contract
        self.register_buffer(
            "known_raw_token_ids",
            torch.tensor(contract.known_raw_token_ids, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "known_semantic_token_ids",
            torch.tensor(contract.known_semantic_token_ids, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "identity_bit_positions",
            torch.arange(IDENTITY_FIBRE_BITS, dtype=torch.int64),
            persistent=False,
        )

    def _semantic_ids(self, raw_token_ids: torch.Tensor) -> torch.Tensor:
        contract = self.contract
        if contract.semantic_hash_bucket_count:
            bucket_count = int(contract.semantic_hash_bucket_count)
            signed_remainder = torch.remainder(raw_token_ids, bucket_count)
            unsigned_correction = (1 << 64) % bucket_count
            buckets = torch.remainder(
                signed_remainder
                + (raw_token_ids < 0).to(torch.int64) * unsigned_correction,
                bucket_count,
            )
            semantic = buckets + int(contract.semantic_known_token_count)
        else:
            semantic = torch.full_like(raw_token_ids, int(contract.semantic_unk_id))

        indices = torch.searchsorted(self.known_raw_token_ids, raw_token_ids)
        safe_indices = indices.clamp(max=self.known_raw_token_ids.numel() - 1)
        known = (indices < self.known_raw_token_ids.numel()) & (
            self.known_raw_token_ids[safe_indices] == raw_token_ids
        )
        known_values = self.known_semantic_token_ids[safe_indices]
        return torch.where(known, known_values, semantic)

    def forward(
        self,
        raw_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if raw_token_ids.dtype != torch.int64 or raw_token_ids.dim() != 2:
            raise ERGT43Error("raw_token_ids must be a signed int64 tensor [B, T]")
        if attention_mask.shape != raw_token_ids.shape:
            raise ERGT43Error("raw_token_ids and attention_mask must align")
        valid = attention_mask.bool()
        semantic = self._semantic_ids(raw_token_ids)
        semantic = torch.where(
            valid,
            semantic,
            torch.full_like(semantic, int(self.contract.semantic_pad_id)),
        )
        bits = torch.bitwise_and(
            torch.bitwise_right_shift(
                raw_token_ids.unsqueeze(-1), self.identity_bit_positions
            ),
            1,
        )
        scale = 1.0 / float(IDENTITY_FIBRE_BITS) ** 0.5
        identity = (bits.to(torch.float32) * 2.0 - 1.0) * scale
        identity = identity * valid.unsqueeze(-1).to(identity.dtype)
        return semantic, identity


@dataclass(frozen=True)
class ERGT43Config:
    vocab_size: int
    max_tokens: int
    raw_input_contract: RawTokenInputContract | None = None
    hidden_dim: int = 48
    psi_rank: int = 16
    n_worlds: int = 8
    field_steps: int = 3
    sparse_top_k: int = 16
    max_hops: int = 40
    softmin_temperature: float = 0.35
    action_margin: float = 0.20
    transmission_floor: float = 0.50
    boundary_deficit_ceiling: float = 1.00
    terminal_mass_floor: float = 0.50
    event_presence_floor: float = 0.45
    event_slot_reserve_factor: int = 2
    close_event_source_over_role_worlds: bool = True
    geodesic_closure_steps: int = 12
    geodesic_backbone_levels: int = 10
    geodesic_backbone_support_floor: float = 0.35
    geodesic_backbone_cone_margin: float = 0.05
    world_lens_transition_cost: float = 0.15
    geodesic_path_gate_floor: float = 0.25
    world_action_scale: float = 0.15
    action_measurement_quantum: float = 0.50
    attenuated_mass_level: float = 0.25
    eps: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.vocab_size <= 2 or self.max_tokens <= 1:
            raise ERGT43Error("vocab_size and max_tokens must define a real input space")
        if self.n_worlds != len(WORLD_NAMES):
            raise ERGT43Error("ERGT-43 requires the complete eight-world field")
        if self.max_hops <= 0 or self.sparse_top_k <= 0:
            raise ERGT43Error("max_hops and sparse_top_k must be positive")
        if self.softmin_temperature <= 0.0 or self.action_margin <= 0.0:
            raise ERGT43Error("softmin temperature and action margin must be positive")
        if self.action_measurement_quantum <= 0.0:
            raise ERGT43Error("action measurement quantum must be positive")
        if self.event_slot_reserve_factor < 2:
            raise ERGT43Error("ERGT-43 requires at least two slots per predicted event")
        if self.geodesic_closure_steps <= 0 or self.geodesic_backbone_levels <= 0:
            raise ERGT43Error("product-manifold closure requires positive steps and levels")
        if not 0.0 < self.geodesic_backbone_support_floor < 1.0:
            raise ERGT43Error("backbone support floor must lie in (0, 1)")
        if self.geodesic_backbone_cone_margin < 0.0:
            raise ERGT43Error("backbone cone margin must be non-negative")
        if self.world_lens_transition_cost < 0.0:
            raise ERGT43Error("world lens transition cost must be non-negative")
        if not 0.0 <= self.world_action_scale < 0.5 * self.action_measurement_quantum:
            raise ERGT43Error(
                "world action must remain a sub-cell correction to registered local action"
            )
        if not 0.0 < self.geodesic_path_gate_floor < 1.0:
            raise ERGT43Error("geodesic path gate floor must lie in (0, 1)")
        if not 0.0 < self.attenuated_mass_level < self.transmission_floor:
            raise ERGT43Error("attenuated mass must lie below the registered support floor")
        if (
            self.raw_input_contract is not None
            and self.raw_input_contract.semantic_vocabulary_size != self.vocab_size
        ):
            raise ERGT43Error("raw-input contract and semantic vocabulary must agree")

    def ergt42_config(self) -> ERGT42Config:
        return ERGT42Config(
            vocab_size=self.vocab_size,
            max_tokens=self.max_tokens,
            hidden_dim=self.hidden_dim,
            psi_rank=self.psi_rank,
            n_worlds=self.n_worlds,
            field_steps=self.field_steps,
            sparse_top_k=self.sparse_top_k,
            max_program_hops=self.max_hops,
            adaptive_event_capacity=True,
            event_slot_reserve_factor=self.event_slot_reserve_factor,
            event_selection_floor=self.event_presence_floor,
            close_event_source_over_role_worlds=self.close_event_source_over_role_worlds,
        )

    @property
    def registered_geodesic_closure_steps(self) -> int:
        """Never truncate Bellman relaxation below the registered answer horizon."""

        return max(int(self.geodesic_closure_steps), int(self.max_hops))


def valid_domain_multiscale_backbone_mask(
    attention_mask: torch.Tensor,
    *,
    levels: int,
) -> torch.Tensor:
    """Return the dyadic numerical mesh over physical, non-padding cells only."""

    if attention_mask.dim() != 2:
        raise ERGT43Error("attention_mask must have shape [B, T]")
    if levels <= 0:
        raise ERGT43Error("multiscale backbone levels must be positive")
    batch, token_count = attention_mask.shape
    mesh = torch.zeros(
        (token_count, token_count),
        dtype=torch.bool,
        device=attention_mask.device,
    )
    source = torch.arange(token_count, device=attention_mask.device)
    for level in range(int(levels)):
        offset = 1 << level
        if offset >= token_count:
            break
        forward = source + offset
        valid_forward = forward < token_count
        mesh[source[valid_forward], forward[valid_forward]] = True
        backward = source - offset
        valid_backward = backward >= 0
        mesh[source[valid_backward], backward[valid_backward]] = True
    valid_pair = attention_mask[:, :, None] & attention_mask[:, None, :]
    return mesh.unsqueeze(0).expand(batch, -1, -1) & valid_pair


def padding_independent_multiscale_support(
    state: Mapping[str, torch.Tensor],
    induced: Mapping[str, torch.Tensor],
    attention_mask: torch.Tensor,
    config: ERGT43Config,
) -> dict[str, torch.Tensor]:
    """Place the geodesic mesh on valid cells without using padding charge.

    V18 accidentally obtained diffuse metric support from selected padding
    cells. V19 correctly removed that nonphysical charge, exposing occasional
    disconnected long event fibres. This replacement is label-free: it gives
    only the registered dyadic numerical mesh a minimum metric capacity. Edge
    action, finite-speed admissibility, world authorization, and every typed
    terminal condition remain part of the governing path.
    """

    raw_weight = induced["program_world_edge_weight"]
    raw_length = induced["program_world_edge_length"]
    raw_action = induced["program_world_action"]
    raw_cone = induced["program_world_cone"].bool()
    raw_mask = induced["program_world_mask"].bool()
    batch, worlds, token_count, target_count = raw_weight.shape
    if token_count != target_count or attention_mask.shape != (batch, token_count):
        raise ERGT43Error("program world fields and attention mask must align")

    mesh = valid_domain_multiscale_backbone_mask(
        attention_mask.bool(),
        levels=config.geodesic_backbone_levels,
    )
    active = state["active_worlds"].bool()
    if active.dim() == 1:
        active = active.unsqueeze(0).expand(batch, -1)
    authorized = induced.get("relation_authorized_worlds")
    if authorized is None:
        authorized = torch.ones(worlds, dtype=torch.bool, device=raw_weight.device)
    authorized = authorized.bool()
    if authorized.dim() == 1:
        authorized = authorized.unsqueeze(0).expand(batch, -1)
    support_mask = mesh[:, None] & active[:, :, None, None] & authorized[:, :, None, None]

    floor = raw_weight.new_tensor(float(config.geodesic_backbone_support_floor))
    hard_supported_weight = torch.where(
        support_mask,
        torch.maximum(raw_weight, floor),
        raw_weight,
    )
    # Preserve a gradient to the learned metric while fixing the forward mesh.
    supported_weight = raw_weight + (hard_supported_weight - raw_weight).detach()
    supported_length = -torch.log(supported_weight.clamp_min(float(config.eps)))
    valid_pair = attention_mask[:, None, :, None] & attention_mask[:, None, None, :]
    diagonal = torch.eye(
        token_count,
        dtype=torch.bool,
        device=raw_weight.device,
    ).view(1, 1, token_count, token_count)
    off_diagonal = valid_pair & ~diagonal
    supported_length = supported_length.masked_fill(~off_diagonal, 30.0)
    supported_cone = supported_length <= state["world_cone_budget"]
    supported_action = (
        supported_length
        + 0.75 * (1.0 - state["reconstructibility"][:, None])
        + 0.50 * (1.0 - state["stability"][:, None])
        + (~supported_cone).to(supported_length.dtype) * 8.0
    )
    supported_mask = (
        induced["program_world_sparse_mask"].bool()
        & supported_cone
        & active[:, :, None, None]
        & state["program_transport_enabled"].bool()
    )

    output = dict(induced)
    physical_world_pair = (
        valid_pair & active[:, :, None, None] & authorized[:, :, None, None] & ~diagonal
    )
    support_fraction = support_mask.to(raw_weight.dtype).sum() / physical_world_pair.to(
        raw_weight.dtype
    ).sum().clamp_min(1.0)
    output.update(
        {
            "direct_program_world_edge_weight": raw_weight,
            "direct_program_world_edge_length": raw_length,
            "direct_program_world_action": raw_action,
            "direct_program_world_cone": raw_cone,
            "direct_program_world_mask": raw_mask,
            "program_world_edge_weight": supported_weight,
            "program_world_edge_length": supported_length,
            "program_world_action": supported_action,
            "program_world_cone": supported_cone,
            "program_world_mask": supported_mask,
            "valid_domain_backbone_mask": support_mask,
            "valid_domain_backbone_support_floor": floor,
            "valid_domain_backbone_support_fraction": support_fraction,
        }
    )
    return output


def registered_event_selection_audit(
    event_relation_logits: torch.Tensor,
    selected_event_indices: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Count hard predicted events that were actually omitted by slot selection."""

    if event_relation_logits.dim() != 3:
        raise ERGT43Error("event_relation_logits must have shape [B, T, R]")
    batch, token_count, _ = event_relation_logits.shape
    if attention_mask.shape != (batch, token_count):
        raise ERGT43Error("attention_mask must align with event relation logits")
    if selected_event_indices.dim() != 2 or selected_event_indices.size(0) != batch:
        raise ERGT43Error("selected_event_indices must have shape [B, K]")
    if selected_event_indices.numel() and (
        int(selected_event_indices.min().item()) < 0
        or int(selected_event_indices.max().item()) >= token_count
    ):
        raise ERGT43Error("selected event index lies outside the token axis")

    hard_event_mask = (event_relation_logits.argmax(dim=-1) > 0) & attention_mask.bool()
    selected_token_mask = torch.zeros_like(hard_event_mask)
    selected_token_mask.scatter_(1, selected_event_indices, True)
    hard_predicted_count = hard_event_mask.sum(dim=1)
    selected_hard_count = (hard_event_mask & selected_token_mask).sum(dim=1)
    hard_overflow = (hard_predicted_count - selected_hard_count).clamp_min(0)
    return hard_predicted_count, selected_hard_count, hard_overflow


class ProductManifoldFieldInducer(RolePreservingGeometricCompiler):
    """Bound event slots by the registered maximum physical chain.

    The matched-topology generator emits two typed physical events per
    relational hop. Soft candidate pressure above this bound remains
    observable, while only hard predicted events actually omitted by slot
    selection block readiness.
    """

    def _event_slot_count(self, event_strength: torch.Tensor) -> tuple[int, torch.Tensor]:
        event_count = event_strength.size(1)
        active_count = (event_strength >= self.config.event_selection_floor).sum(dim=1)
        predicted_peak = int(active_count.detach().amax().cpu().item())
        reserved = self.config.event_slot_reserve_factor * predicted_peak
        registered_bound = max(8, 2 * int(self.config.max_program_hops))
        event_slots = min(
            event_count,
            registered_bound,
            max(8, reserved),
        )
        return event_slots, active_count

    def _select_event_indices(
        self,
        event_probability: torch.Tensor,
        event_strength: torch.Tensor,
        event_slots: int,
    ) -> torch.Tensor:
        """Keep every hard event first, then return slots in causal token order."""

        hard_event = event_probability.argmax(dim=-1) > 0
        priority = event_strength + 2.0 * hard_event.to(event_strength.dtype)
        selected = priority.topk(k=event_slots, dim=1).indices
        return selected.sort(dim=1).values

    def forward(
        self,
        state: Mapping[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = super().forward(state, attention_mask)
        selected = outputs["selected_event_indices"]
        actual_capacity = selected.new_tensor(selected.size(1))
        soft_candidate_count = outputs["predicted_event_count"]
        hard_count, selected_hard_count, hard_overflow = registered_event_selection_audit(
            outputs["event_relation_logits"],
            selected,
            attention_mask,
        )
        outputs["soft_event_candidate_count"] = soft_candidate_count
        outputs["soft_event_candidate_overflow"] = (
            soft_candidate_count - actual_capacity
        ).clamp_min(0)
        outputs["predicted_event_count"] = hard_count
        outputs["selected_predicted_event_count"] = selected_hard_count
        outputs["selected_event_capacity"] = actual_capacity
        outputs["event_selection_overflow"] = hard_overflow
        return outputs


@dataclass(frozen=True)
class HardBoundarySolution:
    answer_ids: torch.Tensor
    candidate_supported: torch.Tensor
    candidate_action: torch.Tensor
    candidate_payload_mass: torch.Tensor
    candidate_boundary_deficit: torch.Tensor
    selected_action: torch.Tensor
    failure_code: torch.Tensor


def canonical_identity_fibre_mask(
    identity: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    identity_distance: torch.Tensor | None = None,
    tolerance: float = 1.0e-5,
) -> torch.Tensor:
    """Choose one positional representative for each conserved identity fibre."""

    if identity_distance is None:
        normalized = F.normalize(identity, dim=-1, eps=1.0e-6)
        identity_distance = (
            (normalized[:, :, None, :] - normalized[:, None, :, :]).square().sum(dim=-1)
        )
    same_fibre = identity_distance <= float(tolerance)
    token_count = identity.size(1)
    earlier = torch.tril(
        torch.ones((token_count, token_count), dtype=torch.bool, device=identity.device),
        diagonal=-1,
    )
    has_earlier_representative = (same_fibre & earlier[None] & attention_mask[:, None, :]).any(
        dim=-1
    )
    return attention_mask & ~has_earlier_representative


def exact_identity_fibre_gate(
    anchor: torch.Tensor,
    identity: torch.Tensor,
    *,
    tolerance: float = 1.0e-5,
) -> torch.Tensor:
    """Return the exact conserved-charge equivalence class of each anchor."""

    normalized_anchor = F.normalize(anchor, dim=-1, eps=1.0e-6)
    normalized_identity = F.normalize(identity, dim=-1, eps=1.0e-6)
    similarity = torch.einsum("bkh,bth->bkt", normalized_anchor, normalized_identity)
    return similarity >= 1.0 - float(tolerance)


class IdentityFibreMetricBinder(nn.Module):
    """Bind query roles and event endpoints by an identity-fibre distance.

    This is a symmetric metric energy over the conserved identity charge.  It
    has no learned query/key projections, no softmax attention, and no access
    to target pointers or answer labels.
    """

    def __init__(self) -> None:
        super().__init__()
        self.metric_gain_raw = nn.Parameter(torch.tensor(4.0))
        self.entity_gate_gain_raw = nn.Parameter(torch.tensor(2.0))
        self.residual_gain_raw = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _metric_cost(anchor: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        anchor = F.normalize(anchor, dim=-1, eps=1.0e-6)
        identity = F.normalize(identity, dim=-1, eps=1.0e-6)
        return (anchor[:, :, None, :] - identity[:, None, :, :]).square().sum(dim=-1)

    @staticmethod
    def _oriented_identity(identity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left = torch.roll(identity, shifts=1, dims=1)
        right = torch.roll(identity, shifts=-1, dims=1)
        left[:, 0] = 0.0
        right[:, -1] = 0.0
        return left, right

    def forward(
        self,
        state: Mapping[str, torch.Tensor],
        induced: Mapping[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        identity = state["identity"]
        left_identity, right_identity = self._oriented_identity(identity)
        query_probability = torch.softmax(induced["query_role_logits"], dim=-1)
        valid = attention_mask.to(identity.dtype)
        metric_gain = F.softplus(self.metric_gain_raw)
        entity_gain = F.softplus(self.entity_gate_gain_raw)
        residual_gain = F.softplus(self.residual_gain_raw)
        entity_gate = F.logsigmoid(induced["entity_logits"])
        canonical_fibre = canonical_identity_fibre_mask(
            identity,
            attention_mask,
            identity_distance=state.get("identity_distance"),
        )

        pointer_contexts: list[torch.Tensor] = []
        for role_id in (1, 2, 4):
            weight = query_probability[..., role_id] * valid
            soft_context = (weight.unsqueeze(-1) * right_identity).sum(dim=1)
            soft_context = soft_context / weight.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            role_score = induced["query_role_logits"][..., role_id].masked_fill(
                ~attention_mask, -1.0e4
            )
            hard_anchor = role_score.argmax(dim=-1)
            rows = torch.arange(identity.size(0), device=identity.device)
            hard_context = right_identity[rows, hard_anchor]
            context = soft_context + (hard_context - soft_context).detach()
            pointer_contexts.append(context)
        pointer_context = torch.stack(pointer_contexts, dim=1)
        pointer_cost = self._metric_cost(pointer_context, identity)
        pointer_exact_gate = pointer_cost.detach() <= 1.0e-5
        pointer_quotient_gate = pointer_exact_gate & canonical_fibre[:, None, :]
        pointer_training_logits = (
            -metric_gain * pointer_cost
            + entity_gain * entity_gate[:, None, :]
            + residual_gain * torch.tanh(induced["pointer_logits"] / 4.0)
        )
        pointer_training_logits = pointer_training_logits.masked_fill(
            ~attention_mask[:, None, :], -1.0e4
        )
        pointer_logits = pointer_training_logits.masked_fill(~pointer_quotient_gate, -1.0e4)

        source_cost = self._metric_cost(left_identity, identity)
        target_cost = self._metric_cost(right_identity, identity)
        source_exact_gate = source_cost.detach() <= 1.0e-5
        target_exact_gate = target_cost.detach() <= 1.0e-5
        source_quotient_gate = source_exact_gate & canonical_fibre[:, None, :]
        target_quotient_gate = target_exact_gate & canonical_fibre[:, None, :]
        endpoint_gate = entity_gain * entity_gate[:, None, :]
        event_source_logits = (
            -metric_gain * source_cost
            + endpoint_gate
            + residual_gain * torch.tanh(induced["event_source_logits"] / 4.0)
        )
        event_target_logits = (
            -metric_gain * target_cost
            + endpoint_gate
            + residual_gain * torch.tanh(induced["event_target_logits"] / 4.0)
        )
        event_source_logits = event_source_logits.masked_fill(~source_quotient_gate, -1.0e4)
        event_target_logits = event_target_logits.masked_fill(~target_quotient_gate, -1.0e4)
        return {
            "pointer_logits": pointer_logits,
            "pointer_training_logits": pointer_training_logits,
            "source_logits": pointer_logits[:, 0],
            "candidate_logits": pointer_logits[:, 1:],
            "event_source_logits": event_source_logits,
            "event_target_logits": event_target_logits,
            "identity_fibre_pointer_cost": pointer_cost,
            "identity_fibre_source_cost": source_cost,
            "identity_fibre_target_cost": target_cost,
            "identity_fibre_binding_strength": metric_gain,
            "identity_fibre_representative_mask": canonical_fibre,
            "identity_fibre_pointer_exact_gate": pointer_quotient_gate,
            "identity_fibre_source_exact_gate": source_quotient_gate,
            "identity_fibre_target_exact_gate": target_quotient_gate,
        }


def _gather_tokens(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    suffix = values.shape[2:]
    gather = indices.view(*indices.shape, *([1] * len(suffix))).expand(*indices.shape, *suffix)
    return torch.gather(values, 1, gather)


def _hard_one_hot(probability: torch.Tensor) -> torch.Tensor:
    index = probability.argmax(dim=-1, keepdim=True)
    return torch.zeros_like(probability).scatter_(-1, index, 1.0)


def _soft_min(values: torch.Tensor, *, dim: int, temperature: float) -> torch.Tensor:
    return -float(temperature) * torch.logsumexp(-values / float(temperature), dim=dim)


def registered_hard_measurements(
    *,
    action: torch.Tensor,
    transmission: torch.Tensor,
    boundary_deficit: torch.Tensor,
    terminal_mass: torch.Tensor,
    config: ERGT43Config,
) -> dict[str, torch.Tensor]:
    """Measure continuous fields at the preregistered hard resolution.

    Forward values are discrete observables consumed by the hard target. The
    straight-through form keeps the continuous field available to training and
    prevents sub-resolution errors from becoming an unregistered decision
    channel over long paths.
    """

    hard_action = registered_hard_action(action, config)
    hard_transmission = torch.where(
        transmission >= float(config.transmission_floor),
        torch.ones_like(transmission),
        torch.full_like(transmission, float(config.attenuated_mass_level)),
    )
    hard_boundary = torch.where(
        boundary_deficit <= float(config.boundary_deficit_ceiling),
        torch.zeros_like(boundary_deficit),
        torch.full_like(boundary_deficit, 2.0 * float(config.boundary_deficit_ceiling)),
    )
    hard_terminal = torch.where(
        terminal_mass >= float(config.terminal_mass_floor),
        torch.ones_like(terminal_mass),
        torch.full_like(terminal_mass, float(config.attenuated_mass_level)),
    )
    return {
        "action": hard_action,
        "transmission": transmission + (hard_transmission - transmission).detach(),
        "boundary_deficit": boundary_deficit + (hard_boundary - boundary_deficit).detach(),
        "terminal_mass": terminal_mass + (hard_terminal - terminal_mass).detach(),
    }


def registered_hard_action(
    action: torch.Tensor,
    config: ERGT43Config,
) -> torch.Tensor:
    """Project action onto its registered lattice with straight-through gradients."""

    quantum = float(config.action_measurement_quantum)
    hard_action = (action / quantum).round().mul(quantum).clamp_min(quantum)
    return action + (hard_action - action).detach()


class LocalPhysicalConditionField(nn.Module):
    """Read physical event conditions from an oriented local Psi stencil."""

    def __init__(self, config: ERGT43Config) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.config = config
        self.shared = nn.Sequential(
            nn.LayerNorm(7 * hidden),
            nn.Linear(7 * hidden, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
        )
        self.action_head = nn.Linear(hidden, 1)
        self.cone_head = nn.Linear(hidden, 1)
        self.transmission_head = nn.Linear(hidden, 1)
        self.deficit_head = nn.Linear(hidden, 1)
        self.terminal_head = nn.Linear(hidden, 1)

    @staticmethod
    def _right_offset(hidden: torch.Tensor, offset: int) -> torch.Tensor:
        shifted = torch.zeros_like(hidden)
        if offset < hidden.size(1):
            shifted[:, :-offset] = hidden[:, offset:]
        return shifted

    def forward(self, psi: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        stencil = torch.cat(
            (psi, *(self._right_offset(psi, offset) for offset in range(1, 7))), dim=-1
        )
        hidden = self.shared(stencil)
        valid = attention_mask.to(hidden.dtype)
        return {
            "event_action_cost": (0.10 + F.softplus(self.action_head(hidden).squeeze(-1))) * valid,
            "event_cone_logit": self.cone_head(hidden).squeeze(-1),
            "event_cone_probability": torch.sigmoid(self.cone_head(hidden).squeeze(-1)),
            "event_transmission": torch.sigmoid(self.transmission_head(hidden).squeeze(-1)),
            "event_boundary_deficit": F.softplus(self.deficit_head(hidden).squeeze(-1)),
            "event_terminal_mass": torch.sigmoid(self.terminal_head(hidden).squeeze(-1)),
        }


def _event_world_values(
    outputs: Mapping[str, torch.Tensor],
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    temperature: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, event_count = source_indices.shape
    rows = torch.arange(batch, device=source_indices.device)[:, None].expand(-1, event_count)
    world_action = outputs.get(
        "direct_program_world_action", outputs["program_world_action"]
    ).permute(0, 2, 3, 1)[rows, source_indices, target_indices]
    world_mask = outputs.get("direct_program_world_mask", outputs["program_world_mask"]).permute(
        0, 2, 3, 1
    )[rows, source_indices, target_indices]
    authorized = outputs["relation_authorized_worlds"].bool()
    if authorized.dim() == 1:
        authorized = authorized.unsqueeze(0).expand(batch, -1)
    world_mask = world_mask & authorized[:, None, :]
    world_weight = outputs.get(
        "direct_program_world_edge_weight", outputs["program_world_edge_weight"]
    ).permute(0, 2, 3, 1)[rows, source_indices, target_indices]
    route = outputs["role_routes"][:, 1]
    route = route[:, None, :].expand(-1, event_count, -1)
    soft_gate = (route * world_weight).sum(dim=-1).clamp(eps, 1.0)
    masked_action = world_action - float(temperature) * route.clamp_min(eps).log()
    masked_action = masked_action.masked_fill(~world_mask, 1.0e4)
    soft_action = _soft_min(masked_action, dim=-1, temperature=temperature)
    hard_action = masked_action.min(dim=-1).values
    hard_gate = world_mask.any(dim=-1)
    return soft_action, hard_action, soft_gate, hard_gate


def _product_manifold_event_values(
    outputs: Mapping[str, torch.Tensor],
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    config: ERGT43Config,
    *,
    intervention: str,
) -> ProductManifoldClosure:
    use_raw_geometry = intervention == "no_multiscale_backbone"
    action_key = "direct_program_world_action" if use_raw_geometry else "program_world_action"
    weight_key = (
        "direct_program_world_edge_weight" if use_raw_geometry else "program_world_edge_weight"
    )
    cone_key = "direct_program_world_cone" if use_raw_geometry else "program_world_cone"
    world_action = outputs.get(action_key, outputs["program_world_action"])
    world_weight = outputs.get(weight_key, outputs["program_world_edge_weight"])
    world_cone = outputs.get(cone_key, outputs["program_world_cone"]).bool()
    if intervention == "no_cone":
        cone_penalty = (~world_cone).to(world_action.dtype) * 8.0
        world_action = (world_action - cone_penalty).clamp_min(config.eps)
    return product_manifold_geodesic_closure(
        world_action=world_action,
        world_weight=world_weight,
        world_sparse_mask=outputs["program_world_sparse_mask"],
        world_cone=world_cone,
        attention_mask=outputs["attention_mask"].bool(),
        active_worlds=outputs["active_worlds"].bool(),
        authorized_worlds=outputs["relation_authorized_worlds"].bool(),
        role_route=outputs["role_routes"][:, 1],
        event_source=source_indices,
        event_target=target_indices,
        max_steps=config.registered_geodesic_closure_steps,
        sparse_top_k=config.sparse_top_k,
        backbone_levels=config.geodesic_backbone_levels,
        world_transition_cost=config.world_lens_transition_cost,
        temperature=config.softmin_temperature,
        include_backbone=intervention != "no_multiscale_backbone",
        allow_world_transitions=intervention != "no_world_transitions",
        ignore_cone=intervention == "no_cone",
        eps=config.eps,
    )


def _event_world_observer(
    outputs: Mapping[str, torch.Tensor],
    key: str,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Project a world observable onto events without changing their dynamics."""

    with torch.no_grad():
        batch, event_count = source_indices.shape
        rows = torch.arange(batch, device=source_indices.device)[:, None].expand(-1, event_count)
        values = outputs[key].detach().permute(0, 2, 3, 1)[rows, source_indices, target_indices]
        world_mask = (
            outputs["program_world_mask"]
            .detach()
            .permute(0, 2, 3, 1)[rows, source_indices, target_indices]
        )
        route = outputs["role_routes"].detach()[:, 1, None, :].expand(-1, event_count, -1)
        weight = route * world_mask.to(route.dtype)
        return ((values * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(float(eps))).detach()


def event_graph_spectral_observer(
    outputs: Mapping[str, torch.Tensor],
    selected_event_indices: torch.Tensor,
    event_presence: torch.Tensor,
    *,
    event_presence_floor: float,
    eps: float,
) -> dict[str, torch.Tensor]:
    """Measure the selected-event world spectra outside the governing path.

    The observer uses the symmetric normalized Laplacian of each induced world
    graph after restricting it to active selected events.  It consumes no gold
    labels and every input is detached.  Its outputs are reporting-only and are
    never read by routing, transport, action, or either hard solver.
    """

    with torch.no_grad():
        # Observe the event-backreacted geometry actually available to the
        # product-manifold solver, not the substrate graph that existed before
        # events modified the worlds.  The sparse/cone masks are detached with
        # the weights, so this remains a read-only view of the final geometry.
        world_weight = outputs["program_world_edge_weight"].detach()
        world_mask = (
            outputs["program_world_sparse_mask"].detach().bool()
            & outputs["program_world_cone"].detach().bool()
        )
        world_weight = world_weight * world_mask.to(world_weight.dtype)
        selected = selected_event_indices.detach()
        active_event = event_presence.detach() >= float(event_presence_floor)
        batch, worlds, _, token_count = world_weight.shape
        if selected.dim() != 2 or selected.size(0) != batch:
            raise ERGT43Error("selected event indices must have shape [B, E]")
        event_count = selected.size(1)
        if event_count == 0:
            zero = world_weight.new_zeros(batch)
            return {
                "event_spectral_entropy": zero,
                "event_spectral_effective_rank": zero,
                "event_spectral_gap": zero,
                "event_spectral_world_diversity": zero,
            }

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

        entropy_by_world = world_weight.new_zeros((batch, worlds))
        rank_by_world = world_weight.new_zeros((batch, worlds))
        gap_by_world = world_weight.new_zeros((batch, worlds))
        valid_world = torch.zeros((batch, worlds), dtype=torch.bool, device=world_weight.device)
        signature = world_weight.new_zeros((batch, worlds, 16))
        for row in range(batch):
            event_mask = active_event[row]
            valid_count = int(event_mask.sum().item())
            if valid_count < 2:
                continue
            for world in range(worlds):
                graph = adjacency[row, world][event_mask][:, event_mask]
                degree = graph.sum(dim=-1)
                nonisolated = degree > float(eps)
                inverse_sqrt = degree.clamp_min(float(eps)).rsqrt()
                normalized_adjacency = inverse_sqrt[:, None] * graph * inverse_sqrt[None, :]
                laplacian = torch.diag(nonisolated.to(graph.dtype)) - normalized_adjacency
                eigenvalues = torch.linalg.eigvalsh(laplacian).clamp_min(0.0)
                mass = eigenvalues.sum()
                if float(mass.item()) <= float(eps):
                    continue
                probability = eigenvalues / mass
                raw_entropy = -(probability * probability.clamp_min(float(eps)).log()).sum()
                entropy_by_world[row, world] = raw_entropy / graph.new_tensor(
                    float(valid_count)
                ).log().clamp_min(float(eps))
                rank_by_world[row, world] = raw_entropy.exp()
                gap_by_world[row, world] = eigenvalues[1]
                signature[row, world, : min(16, valid_count)] = eigenvalues[:16]
                valid_world[row, world] = True

        valid_weight = valid_world.to(world_weight.dtype)
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
                torch.ones((worlds, worlds), dtype=torch.bool, device=world_weight.device),
                diagonal=1,
            )[None]
        )
        pair_count = world_pair.sum(dim=(-1, -2)).clamp_min(1).to(world_weight.dtype)
        diversity = ((1.0 - cosine) * world_pair.to(cosine.dtype)).sum(dim=(-1, -2))
        diversity = diversity / pair_count
        diversity = torch.where(
            world_pair.any(dim=(-1, -2)),
            diversity,
            torch.zeros_like(diversity),
        )
        return {
            "event_spectral_entropy": entropy.detach(),
            "event_spectral_effective_rank": effective_rank.detach(),
            "event_spectral_gap": spectral_gap.detach(),
            "event_spectral_world_diversity": diversity.detach(),
        }


def gauge_fixed_world_action(
    soft_action: torch.Tensor,
    hard_action: torch.Tensor,
    hard_gate: torch.Tensor,
    event_strength: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove the unobservable per-example action offset.

    The hard solver depends on action differences, not on an arbitrary additive
    metric zero.  Centering over active event cells prevents entity-specific
    world offsets from leaking into an otherwise transferable local action.
    The remaining correction is bounded but retains relative path ordering.
    """

    active = hard_gate.to(soft_action.dtype)
    weight = event_strength.detach().clamp(0.0, 1.0) * active
    denominator = weight.sum(dim=1, keepdim=True)
    normalized = weight / denominator.clamp_min(float(eps))
    soft_gauge = (normalized * soft_action.clamp(max=30.0)).sum(dim=1, keepdim=True)
    hard_gauge = (normalized * hard_action.clamp(max=30.0)).sum(dim=1, keepdim=True)
    has_active_event = (denominator > float(eps)).to(soft_action.dtype)
    soft_residual = torch.tanh(soft_action.clamp(max=30.0) - soft_gauge) * active * has_active_event
    hard_residual = torch.tanh(hard_action.clamp(max=30.0) - hard_gauge) * active * has_active_event
    return (
        soft_residual,
        hard_residual,
        soft_gauge.squeeze(1),
        hard_gauge.squeeze(1),
    )


def _selected_event_fields(
    outputs: Mapping[str, torch.Tensor],
    config: ERGT43Config,
    *,
    intervention: str,
) -> dict[str, torch.Tensor]:
    selected = outputs["selected_event_indices"]
    relation_full = torch.softmax(outputs["event_relation_logits"], dim=-1)
    source_probability = torch.softmax(outputs["event_source_logits"], dim=-1)
    target_probability = torch.softmax(outputs["event_target_logits"], dim=-1)
    relation_probability = _gather_tokens(relation_full[..., 1:], selected)
    selected_valid = _gather_tokens(outputs["attention_mask"], selected).bool()
    # Keep the gauge and the native solver on the same physical support even
    # when a short sequence reserves more event slots than it has valid cells.
    event_strength = relation_probability.sum(dim=-1) * selected_valid.to(
        relation_probability.dtype
    )
    source_probability = _gather_tokens(source_probability, selected)
    target_probability = _gather_tokens(target_probability, selected)
    source_indices = source_probability.argmax(dim=-1)
    target_indices = target_probability.argmax(dim=-1)

    local_action = _gather_tokens(outputs["event_action_cost"], selected)
    local_registered_action = registered_hard_action(local_action, config)
    cone = _gather_tokens(outputs["event_cone_probability"], selected)
    transmission = _gather_tokens(outputs["event_transmission"], selected)
    deficit = _gather_tokens(outputs["event_boundary_deficit"], selected)
    terminal = _gather_tokens(outputs["event_terminal_mass"], selected)
    (
        direct_world_soft_action,
        direct_world_hard_action,
        direct_world_soft_gate,
        direct_world_hard_gate,
    ) = _event_world_values(
        outputs,
        source_indices,
        target_indices,
        temperature=config.softmin_temperature,
        eps=config.eps,
    )
    closure = _product_manifold_event_values(
        outputs,
        source_indices,
        target_indices,
        config,
        intervention=intervention,
    )
    if intervention == "direct_world_gate":
        world_soft_action = direct_world_soft_action
        world_hard_action = direct_world_hard_action
        world_soft_gate = direct_world_soft_gate
        world_hard_gate = direct_world_hard_gate
    else:
        world_soft_action = closure.soft_action
        world_hard_action = closure.hard_action
        world_soft_gate = closure.soft_gate
        world_hard_gate = closure.hard_gate
    (
        world_soft_action_residual,
        world_hard_action_residual,
        world_soft_action_gauge,
        world_hard_action_gauge,
    ) = gauge_fixed_world_action(
        world_soft_action,
        world_hard_action,
        world_hard_gate,
        event_strength,
        eps=config.eps,
    )
    action = local_action + config.world_action_scale * world_soft_action_residual
    # The registered local action defines the observable lattice cell. World
    # geometry remains a bounded sub-cell correction for the smooth path, but
    # cannot move a correct local observable into an adjacent cell merely
    # because the number of distant events changed the per-example gauge.
    hard_action_value = (
        local_registered_action + config.world_action_scale * world_hard_action_residual
    )
    # The forward value is the actual hard world action consumed by the target
    # solver, while gradients follow the smooth min-plus surrogate. Supervising
    # this tensor prevents another physical channel from hiding inside an
    # otherwise unconstrained world-action residual.
    continuous_hard_action = action + (hard_action_value - action).detach()
    world_gate = world_soft_gate * world_hard_gate.to(world_soft_gate.dtype)
    event_curvature = _event_world_observer(
        outputs,
        "world_forman_curvature_proxy",
        source_indices,
        target_indices,
        eps=config.eps,
    )
    event_curvature_gradient = _event_world_observer(
        outputs,
        "world_curvature_graph_gradient",
        source_indices,
        target_indices,
        eps=config.eps,
    )

    if intervention == "no_action":
        action = torch.ones_like(action)
        continuous_hard_action = torch.ones_like(continuous_hard_action)
    elif intervention == "no_cone":
        cone = torch.ones_like(cone)
    elif intervention == "no_boundary":
        deficit = torch.zeros_like(deficit)
    elif intervention == "no_transport":
        transmission = torch.ones_like(transmission)
    elif intervention == "no_terminal_mass":
        terminal = torch.ones_like(terminal)
    elif intervention == "no_memory_geometry":
        source_pointer = outputs["source_logits"].argmax(dim=-1, keepdim=True)
        leaves_source = source_indices == source_pointer
        transmission = torch.where(leaves_source, torch.ones_like(transmission), transmission)

    measured = registered_hard_measurements(
        action=continuous_hard_action,
        transmission=transmission,
        boundary_deficit=deficit,
        terminal_mass=terminal,
        config=config,
    )

    return {
        "selected_event_indices": selected,
        "event_relation_probability": relation_probability,
        "event_presence": event_strength,
        "event_valid_mask": selected_valid,
        "event_source_probability": source_probability,
        "event_target_probability": target_probability,
        "event_source_index": source_indices,
        "event_target_index": target_indices,
        "event_local_action": local_action,
        "event_local_registered_action": local_registered_action,
        "event_world_soft_action": world_soft_action,
        "event_world_hard_action": world_hard_action,
        "event_world_direct_soft_action": direct_world_soft_action,
        "event_world_direct_hard_action": direct_world_hard_action,
        "event_world_direct_soft_gate": direct_world_soft_gate,
        "event_world_direct_hard_gate": direct_world_hard_gate,
        "event_world_closure_soft_action": closure.soft_action,
        "event_world_closure_hard_action": closure.hard_action,
        "event_world_closure_soft_gate": closure.soft_gate,
        "event_world_closure_hard_gate": closure.hard_gate,
        "event_world_closure_gain": (
            closure.hard_gate.to(world_soft_gate.dtype)
            - direct_world_hard_gate.to(world_soft_gate.dtype)
        ),
        "event_geodesic_path_hops": closure.path_hops,
        "geodesic_closure_step_budget": action.new_tensor(config.registered_geodesic_closure_steps),
        "geodesic_closure_steps_executed": action.new_tensor(closure.steps_executed),
        "geodesic_closure_fixed_point_reached": action.new_tensor(
            float(closure.fixed_point_reached)
        ),
        "product_manifold_graph_edge_count": closure.graph_edge_count,
        "event_world_soft_action_residual": world_soft_action_residual,
        "event_world_hard_action_residual": world_hard_action_residual,
        "world_soft_action_gauge": world_soft_action_gauge,
        "world_hard_action_gauge": world_hard_action_gauge,
        "event_action": action,
        "event_continuous_hard_action": continuous_hard_action,
        "event_hard_action": measured["action"],
        "event_cone": cone,
        "event_transmission_pair": transmission,
        "event_boundary_deficit_pair": deficit,
        "event_terminal_mass_pair": terminal,
        "event_measured_transmission_pair": measured["transmission"],
        "event_measured_boundary_deficit_pair": measured["boundary_deficit"],
        "event_measured_terminal_mass_pair": measured["terminal_mass"],
        "event_world_soft_gate": world_soft_gate,
        "event_world_hard_gate": world_hard_gate,
        "event_world_gate": world_gate,
        "event_curvature_proxy": event_curvature,
        "event_curvature_graph_gradient": event_curvature_gradient,
    }


def soft_min_plus_typed_boundary_solve(
    fields: Mapping[str, torch.Tensor],
    *,
    max_hops: int,
    temperature: float,
    transmission_floor: float,
    boundary_deficit_ceiling: float,
    terminal_mass_floor: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable min-plus surrogate of the hard ordered solver."""

    source_probability = torch.softmax(fields["source_logits"], dim=-1)
    candidate_probability = torch.softmax(fields["candidate_logits"], dim=-1)
    boundary_probability = torch.softmax(fields["boundary_logits"], dim=-1)
    event_source = fields["event_source_probability"].clamp_min(eps)
    event_target = fields["event_target_probability"].clamp_min(eps)
    relation = fields["event_relation_probability"].clamp_min(eps)
    feasibility = fields["event_presence"].clamp(0.0, 1.0)
    feasibility = feasibility * fields["event_world_gate"].clamp(0.0, 1.0)
    feasibility = feasibility * fields["event_cone"].clamp(0.0, 1.0)
    feasibility = feasibility * torch.sigmoid(
        10.0 * (fields["event_transmission_pair"] - transmission_floor)
    )
    feasibility = feasibility * torch.sigmoid(
        10.0 * (boundary_deficit_ceiling - fields["event_boundary_deficit_pair"])
    )
    terminal_binding = torch.einsum("bkt,bct->bk", event_target, candidate_probability).clamp(
        0.0, 1.0
    )
    effective_terminal_mass = (
        1.0 - terminal_binding + terminal_binding * fields["event_terminal_mass_pair"]
    )
    feasibility = feasibility * torch.sigmoid(
        10.0 * (effective_terminal_mass - terminal_mass_floor)
    )
    event_base_cost = fields["event_action"] - float(temperature) * feasibility.clamp_min(eps).log()

    batch, token_count = source_probability.shape
    state = source_probability.new_full((batch, token_count, REGISTER_CARDINALITY), 1.0e4)
    state[..., 0] = -float(temperature) * source_probability.clamp_min(eps).log()
    source_binding = -float(temperature) * event_source.log()
    target_binding = -float(temperature) * event_target.log()
    effective_hops = min(int(max_hops), token_count)
    for _ in range(effective_hops):
        updated = state.clone()
        for relation_id in range(1, len(RELATION_NAMES)):
            relation_cost = -float(temperature) * relation[..., relation_id - 1].log()
            for register in range(REGISTER_CARDINALITY):
                event_origin = _soft_min(
                    state[..., register].unsqueeze(1) + source_binding,
                    dim=-1,
                    temperature=temperature,
                )
                event_cost = event_origin + event_base_cost + relation_cost
                incoming = _soft_min(
                    event_cost.unsqueeze(-1) + target_binding,
                    dim=1,
                    temperature=temperature,
                )
                target_register = apply_relation(register, relation_id)
                updated[..., target_register] = _soft_min(
                    torch.stack((updated[..., target_register], incoming), dim=-1),
                    dim=-1,
                    temperature=temperature,
                )
        state = updated

    candidate_costs: list[torch.Tensor] = []
    for candidate in range(2):
        binding = (
            candidate_probability[:, candidate, :, None]
            * boundary_probability[:, candidate, None, :]
        )
        energy = state - float(temperature) * binding.clamp_min(eps).log()
        candidate_costs.append(
            _soft_min(energy.flatten(start_dim=1), dim=-1, temperature=temperature)
        )
    candidate_cost = torch.stack(candidate_costs, dim=-1)
    candidate_score = -candidate_cost
    abstain_score = candidate_score.min(dim=-1).values - 2.0
    return torch.cat((candidate_score, abstain_score.unsqueeze(-1)), dim=-1), candidate_cost


def hard_min_plus_typed_boundary_solve(
    *,
    event_source: torch.Tensor,
    event_target: torch.Tensor,
    event_relation: torch.Tensor,
    event_action: torch.Tensor,
    event_transmission: torch.Tensor,
    event_boundary_deficit: torch.Tensor,
    event_terminal_mass: torch.Tensor,
    event_admissible: torch.Tensor,
    source: torch.Tensor,
    candidates: torch.Tensor,
    boundaries: torch.Tensor,
    token_count: int,
    max_hops: int,
    action_margin: float,
    terminal_mass_floor: float = 0.50,
    boundary_deficit_ceiling: float = 1.00,
) -> HardBoundarySolution:
    """Hard tensor Bellman evolution with no graph object or generic traversal."""

    batch = source.size(0)
    infinity = event_action.new_tensor(1.0e6)
    state = event_action.new_full((batch, token_count, REGISTER_CARDINALITY), 1.0e6)
    payload = event_action.new_zeros((batch, token_count, REGISTER_CARDINALITY))
    deficit = event_action.new_full((batch, token_count, REGISTER_CARDINALITY), 1.0e6)
    rows = torch.arange(batch, device=source.device)
    terminal_edge = (event_target == candidates[:, 0, None]) | (
        event_target == candidates[:, 1, None]
    )
    effective_terminal_mass = torch.where(
        terminal_edge, event_terminal_mass, torch.ones_like(event_terminal_mass)
    )
    state[rows, source, 0] = 0.0
    payload[rows, source, 0] = 1.0
    deficit[rows, source, 0] = 0.0
    effective_hops = min(int(max_hops), token_count)
    for _ in range(effective_hops):
        updated = state.clone()
        updated_payload = payload.clone()
        updated_deficit = deficit.clone()
        for relation_id in range(1, len(RELATION_NAMES)):
            relation_mask = event_admissible & (event_relation == relation_id)
            for register in range(REGISTER_CARDINALITY):
                origin = torch.gather(state[..., register], 1, event_source)
                edge_cost = origin + event_action
                edge_cost = torch.where(relation_mask, edge_cost, infinity)
                incoming = event_action.new_full((batch, token_count), 1.0e6)
                incoming.scatter_reduce_(
                    1, event_target, edge_cost, reduce="amin", include_self=True
                )
                origin_payload = torch.gather(payload[..., register], 1, event_source)
                edge_payload = origin_payload * event_transmission * effective_terminal_mass
                edge_payload = torch.where(
                    relation_mask, edge_payload, torch.zeros_like(edge_payload)
                )
                incoming_payload = event_action.new_zeros((batch, token_count))
                incoming_payload.scatter_reduce_(
                    1, event_target, edge_payload, reduce="amax", include_self=True
                )
                origin_deficit = torch.gather(deficit[..., register], 1, event_source)
                edge_deficit = origin_deficit + event_boundary_deficit
                edge_deficit = torch.where(relation_mask, edge_deficit, infinity)
                incoming_deficit = event_action.new_full((batch, token_count), 1.0e6)
                incoming_deficit.scatter_reduce_(
                    1, event_target, edge_deficit, reduce="amin", include_self=True
                )
                target_register = apply_relation(register, relation_id)
                updated[..., target_register] = torch.minimum(
                    updated[..., target_register], incoming
                )
                updated_payload[..., target_register] = torch.maximum(
                    updated_payload[..., target_register], incoming_payload
                )
                updated_deficit[..., target_register] = torch.minimum(
                    updated_deficit[..., target_register], incoming_deficit
                )
        if bool(
            torch.equal(updated, state)
            and torch.equal(updated_payload, payload)
            and torch.equal(updated_deficit, deficit)
        ):
            break
        state = updated
        payload = updated_payload
        deficit = updated_deficit

    candidate_action = torch.stack(
        (
            state[rows, candidates[:, 0], boundaries[:, 0]],
            state[rows, candidates[:, 1], boundaries[:, 1]],
        ),
        dim=-1,
    )
    candidate_payload = torch.stack(
        (
            payload[rows, candidates[:, 0], boundaries[:, 0]],
            payload[rows, candidates[:, 1], boundaries[:, 1]],
        ),
        dim=-1,
    )
    candidate_deficit = torch.stack(
        (
            deficit[rows, candidates[:, 0], boundaries[:, 0]],
            deficit[rows, candidates[:, 1], boundaries[:, 1]],
        ),
        dim=-1,
    )
    supported = candidate_action < 5.0e5
    supported = supported & (candidate_payload >= float(terminal_mass_floor))
    supported = supported & (candidate_deficit <= float(boundary_deficit_ceiling))
    answer = torch.full((batch,), 2, dtype=torch.long, device=source.device)
    failure = torch.full((batch,), 2, dtype=torch.long, device=source.device)
    only_a = supported[:, 0] & ~supported[:, 1]
    only_b = supported[:, 1] & ~supported[:, 0]
    both = supported.all(dim=-1)
    separated = (candidate_action[:, 0] - candidate_action[:, 1]).abs() >= float(action_margin)
    choose_a = only_a | (both & separated & (candidate_action[:, 0] < candidate_action[:, 1]))
    choose_b = only_b | (both & separated & (candidate_action[:, 1] < candidate_action[:, 0]))
    answer[choose_a] = 0
    answer[choose_b] = 1
    failure[choose_a | choose_b] = 0
    failure[~supported.any(dim=-1)] = 1
    selected = torch.where(
        answer < 2,
        torch.gather(candidate_action, 1, answer.clamp(max=1).unsqueeze(-1)).squeeze(-1),
        infinity,
    )
    return HardBoundarySolution(
        answer_ids=answer,
        candidate_supported=supported,
        candidate_action=candidate_action,
        candidate_payload_mass=candidate_payload,
        candidate_boundary_deficit=candidate_deficit,
        selected_action=selected,
        failure_code=failure,
    )


def hard_solutions_from_outputs(
    outputs: Mapping[str, torch.Tensor],
    config: ERGT43Config,
) -> tuple[HardBoundarySolution, HardBoundarySolution]:
    event_relation_full = torch.cat(
        (
            (1.0 - outputs["event_presence"]).unsqueeze(-1),
            outputs["event_relation_probability"],
        ),
        dim=-1,
    )
    event_relation = event_relation_full.argmax(dim=-1)
    event_present = (outputs["event_presence"] >= config.event_presence_floor) & (
        event_relation > 0
    )
    role_complete = outputs["role_contract_complete"].bool()
    physics_admissible = event_present & (outputs["event_world_gate"] > 0.0)
    physics_admissible = physics_admissible & (outputs["event_cone"] >= 0.5)
    physics_admissible = physics_admissible & role_complete
    source = outputs["source_logits"].argmax(dim=-1)
    candidates = outputs["candidate_logits"].argmax(dim=-1)
    boundaries = outputs["boundary_logits"].argmax(dim=-1)
    common = {
        "event_source": outputs["event_source_index"],
        "event_target": outputs["event_target_index"],
        "event_relation": event_relation,
        "source": source,
        "candidates": candidates,
        "boundaries": boundaries,
        "token_count": outputs["source_logits"].size(-1),
        "max_hops": config.max_hops,
        "action_margin": config.action_margin,
        "terminal_mass_floor": config.terminal_mass_floor,
        "boundary_deficit_ceiling": config.boundary_deficit_ceiling,
    }
    native = hard_min_plus_typed_boundary_solve(
        event_action=outputs["event_hard_action"],
        event_transmission=outputs["event_measured_transmission_pair"],
        event_boundary_deficit=outputs["event_measured_boundary_deficit_pair"],
        event_terminal_mass=outputs["event_measured_terminal_mass_pair"],
        event_admissible=physics_admissible,
        **common,
    )
    generic = hard_min_plus_typed_boundary_solve(
        event_action=torch.ones_like(outputs["event_hard_action"]),
        event_transmission=torch.ones_like(outputs["event_transmission_pair"]),
        event_boundary_deficit=torch.zeros_like(outputs["event_boundary_deficit_pair"]),
        event_terminal_mass=torch.ones_like(outputs["event_terminal_mass_pair"]),
        event_admissible=event_present,
        **common,
    )
    return native, generic


def oracle_solutions_from_batch(
    batch: ERGT43Batch,
    config: ERGT43Config,
) -> tuple[HardBoundarySolution, HardBoundarySolution]:
    """Audit that physical conditions, not topology, identify the gold candidate."""

    supervision = batch.base.supervision
    token_count = batch.base.token_ids.size(1)
    event_source = supervision.event_source_positions.clamp_min(0)
    event_target = supervision.event_target_positions.clamp_min(0)
    event_relation = supervision.relation_anchor_labels
    event_present = batch.physical.event_mask & (event_relation > 0)
    admissible = event_present & (batch.physical.cone_admissible >= 0.5)
    common = {
        "event_source": event_source,
        "event_target": event_target,
        "event_relation": event_relation,
        "source": supervision.source_positions,
        "candidates": supervision.candidate_positions,
        "boundaries": supervision.boundary_labels,
        "token_count": token_count,
        "max_hops": config.max_hops,
        "action_margin": config.action_margin,
        "terminal_mass_floor": config.terminal_mass_floor,
        "boundary_deficit_ceiling": config.boundary_deficit_ceiling,
    }
    native = hard_min_plus_typed_boundary_solve(
        event_action=batch.physical.action_cost,
        event_transmission=batch.physical.transmission,
        event_boundary_deficit=batch.physical.boundary_deficit,
        event_terminal_mass=batch.physical.terminal_capacity,
        event_admissible=admissible,
        **common,
    )
    generic = hard_min_plus_typed_boundary_solve(
        event_action=torch.ones_like(batch.physical.action_cost),
        event_transmission=torch.ones_like(batch.physical.transmission),
        event_boundary_deficit=torch.zeros_like(batch.physical.boundary_deficit),
        event_terminal_mass=torch.ones_like(batch.physical.terminal_capacity),
        event_admissible=event_present,
        **common,
    )
    return native, generic


class NativeGeometricBoundaryModel(nn.Module):
    """Raw-input ERGT field with a direct native boundary-value answer path."""

    uses_standard_qk_attention = False
    uses_discrete_graph_compiler = False
    uses_generic_graph_executor = False
    has_direct_answer_head = False
    has_soft_answer_fallback = False
    target_solver_is_hard_min_plus = True
    uses_identity_fibre_metric_binding = True
    uses_identity_fibre_quotient_binding = True
    uses_gauge_fixed_world_action = True
    uses_registered_local_action_cell = True
    world_action_is_subcell_correction = True
    uses_conserved_identity_initial_condition = True
    uses_adaptive_event_capacity = True
    uses_hard_omission_overflow_gate = True
    uses_hard_first_causal_event_fibres = True
    closure_horizon_coupled_to_answer_horizon = True
    uses_length_balanced_chain_binding_loss = True
    closes_event_source_over_authorized_role_worlds = True
    uses_product_manifold_geodesic_closure = True
    uses_multiscale_causal_backbone = True
    uses_padding_independent_multiscale_support = True
    uses_worst_chain_support_margin = True
    uses_world_lens_transitions = True
    derives_identity_inside_model = True
    public_input_is_raw_tokens_only = True
    curvature_observer_only = True
    spectrum_observer_only = True
    governing_path = (
        "raw_token_serialization_to_internal_semantics_and_conserved_identity_"
        "to_identity_conditioned_psi_to_identity_fibre_quotient_"
        "to_world_metrics_to_hard_first_causal_event_fibres_"
        "to_token_world_product_manifold_to_multistep_geodesic_closure_"
        "to_finite_speed_cone_to_candidate_conditioned_min_plus_to_typed_transport_"
        "to_terminal_boundary"
    )

    def __init__(self, config: ERGT43Config) -> None:
        super().__init__()
        self.config = config
        substrate_config = config.ergt42_config()
        raw_contract = config.raw_input_contract or RawTokenInputContract.direct_semantic(
            config.vocab_size
        )
        self.raw_input_adapter = RawTokenInputAdapter(raw_contract)
        self.substrate = PhysicsNativeSubstrate(substrate_config)
        # This module induces continuous typed event fields.  It is not the
        # discrete ``compile_geometric_programs`` target path removed in ERGT-43.
        self.field_inducer = ProductManifoldFieldInducer(substrate_config)
        self.identity_binder = IdentityFibreMetricBinder()
        self.physical_conditions = LocalPhysicalConditionField(config)

    @staticmethod
    def _substrate_intervention(intervention: str) -> str:
        if intervention in {
            "no_phi",
            "no_event_backreaction",
            "no_memory_geometry",
            "shuffled_geometry",
            "random_geometry",
        } or intervention.startswith(("drop_world_", "only_world_")):
            return intervention
        return "full"

    def _forward_from_internal_state(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        identity_fibre: torch.Tensor,
        intervention: str,
        intervention_seed: int,
        observe_spectrum: bool,
    ) -> dict[str, torch.Tensor]:
        if intervention not in INTERVENTIONS:
            raise ERGT43Error(f"unregistered intervention: {intervention}")
        state = self.substrate(
            token_ids,
            attention_mask,
            identity_fibre=identity_fibre,
            intervention=self._substrate_intervention(intervention),
            intervention_seed=intervention_seed,
            # ERGT-43 observes the complete selected-event graph below.  The
            # older prefix-limited token observer remains available to ERGT-42
            # but is deliberately not part of this solver path.
            observe_spectrum=False,
        )
        induced = self.field_inducer(state, attention_mask)
        induced["relation_authorized_worlds"] = self.field_inducer.role_world_contract[1]
        induced = padding_independent_multiscale_support(
            state,
            induced,
            attention_mask,
            self.config,
        )
        induced = {
            **induced,
            **self.identity_binder(state, induced, attention_mask),
        }
        physical = self.physical_conditions(state["psi"], attention_mask)
        outputs = {**state, **induced, **physical}
        outputs["internal_semantic_token_ids"] = token_ids
        outputs["attention_mask"] = attention_mask
        outputs["relation_authorized_worlds"] = self.field_inducer.role_world_contract[1]
        outputs["base_world_usage_entropy"] = state["world_usage_entropy"]
        outputs["base_world_geometry_diversity"] = state["world_geometry_diversity"]
        outputs["world_usage_entropy"] = induced["program_world_usage_entropy"]
        outputs["world_geometry_diversity"] = induced["program_world_geometry_diversity"]
        selected = _selected_event_fields(outputs, self.config, intervention=intervention)
        outputs.update(selected)
        if observe_spectrum:
            outputs.update(
                event_graph_spectral_observer(
                    outputs,
                    outputs["selected_event_indices"],
                    outputs["event_presence"],
                    event_presence_floor=self.config.event_presence_floor,
                    eps=self.config.eps,
                )
            )
        else:
            zero = outputs["event_presence"].new_zeros(outputs["event_presence"].size(0))
            outputs.update(
                {
                    "event_spectral_entropy": zero,
                    "event_spectral_effective_rank": zero,
                    "event_spectral_gap": zero,
                    "event_spectral_world_diversity": zero,
                }
            )
        logits, candidate_cost = soft_min_plus_typed_boundary_solve(
            outputs,
            max_hops=self.config.max_hops,
            temperature=self.config.softmin_temperature,
            transmission_floor=self.config.transmission_floor,
            boundary_deficit_ceiling=self.config.boundary_deficit_ceiling,
            terminal_mass_floor=self.config.terminal_mass_floor,
            eps=self.config.eps,
        )
        outputs["native_answer_logits"] = logits
        outputs["candidate_soft_action"] = candidate_cost
        outputs["architecture_attention_free"] = logits.new_tensor(1.0)
        outputs["old_discrete_compiler_disabled"] = logits.new_tensor(1.0)
        outputs["old_generic_executor_disabled"] = logits.new_tensor(1.0)
        outputs["typed_conservation_residual"] = (
            (outputs["event_relation_probability"].sum(dim=-1) - outputs["event_presence"])
            .abs()
            .max()
        )
        valid_transport = attention_mask[:, None, :].to(outputs["world_transport"].dtype)
        outputs["payload_conservation_residual"] = (
            (outputs["world_transport"].sum(dim=-1) - 1.0).abs() * valid_transport
        ).max()
        return outputs

    def _forward_impl(
        self,
        raw_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        intervention: str,
        intervention_seed: int,
        observe_spectrum: bool,
    ) -> dict[str, torch.Tensor]:
        token_ids, identity_fibre = self.raw_input_adapter(raw_token_ids, attention_mask)
        outputs = self._forward_from_internal_state(
            token_ids,
            attention_mask,
            identity_fibre=identity_fibre,
            intervention=intervention,
            intervention_seed=intervention_seed,
            observe_spectrum=observe_spectrum,
        )
        outputs["raw_token_ids"] = raw_token_ids
        return outputs

    def forward(
        self,
        raw_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        observe_spectrum: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self._forward_impl(
            raw_token_ids,
            attention_mask,
            intervention="full",
            intervention_seed=0,
            observe_spectrum=observe_spectrum,
        )

    def forward_with_intervention(
        self,
        raw_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        intervention: str,
        intervention_seed: int = 0,
        observe_spectrum: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self._forward_impl(
            raw_token_ids,
            attention_mask,
            intervention=intervention,
            intervention_seed=intervention_seed,
            observe_spectrum=observe_spectrum,
        )


def registered_measurement_margin_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: ERGT43Batch,
    config: ERGT43Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep every continuous field safely inside its registered hard cell."""

    selected = outputs["selected_event_indices"]
    mask = batch.physical.event_mask & batch.base.attention_mask
    selected_mask = _gather_tokens(mask, selected).bool()
    if not bool(selected_mask.any()):
        zero = outputs["native_answer_logits"].new_tensor(0.0)
        return zero, {
            "action": zero,
            "cone": zero,
            "transport": zero,
            "boundary": zero,
            "terminal": zero,
        }

    action_target = _gather_tokens(batch.physical.action_cost, selected)
    cone_target = _gather_tokens(batch.physical.cone_admissible, selected) >= 0.5
    transmission_target = _gather_tokens(batch.physical.transmission, selected)
    boundary_target = _gather_tokens(batch.physical.boundary_deficit, selected)
    terminal_target = _gather_tokens(batch.physical.terminal_capacity, selected)

    action_error = (outputs["event_continuous_hard_action"] - action_target).abs()
    local_action_error = (outputs["event_local_action"] - action_target).abs()
    action_violation = F.relu(action_error - 0.20 * float(config.action_measurement_quantum))
    local_action_violation = F.relu(
        local_action_error - 0.20 * float(config.action_measurement_quantum)
    )
    cone_violation = torch.where(
        cone_target,
        F.relu(0.75 - outputs["event_cone"]),
        F.relu(outputs["event_cone"] - 0.25),
    )

    transport_high = 0.5 * (1.0 + float(config.transmission_floor))
    transport_low = 0.5 * (float(config.transmission_floor) + float(config.attenuated_mass_level))
    transport_violation = torch.where(
        transmission_target >= float(config.transmission_floor),
        F.relu(transport_high - outputs["event_transmission_pair"]),
        F.relu(outputs["event_transmission_pair"] - transport_low),
    )

    boundary_high = 1.5 * float(config.boundary_deficit_ceiling)
    boundary_low = 0.5 * float(config.boundary_deficit_ceiling)
    boundary_violation = torch.where(
        boundary_target > float(config.boundary_deficit_ceiling),
        F.relu(boundary_high - outputs["event_boundary_deficit_pair"]),
        F.relu(outputs["event_boundary_deficit_pair"] - boundary_low),
    )

    terminal_high = 0.5 * (1.0 + float(config.terminal_mass_floor))
    terminal_low = 0.5 * (float(config.terminal_mass_floor) + float(config.attenuated_mass_level))
    terminal_violation = torch.where(
        terminal_target >= float(config.terminal_mass_floor),
        F.relu(terminal_high - outputs["event_terminal_mass_pair"]),
        F.relu(outputs["event_terminal_mass_pair"] - terminal_low),
    )

    def robust_mean(values: torch.Tensor) -> torch.Tensor:
        selected_values = values[selected_mask]
        return 0.5 * (selected_values.mean() + selected_values.amax())

    parts = {
        "action": robust_mean(action_violation),
        "local_action_cell": robust_mean(local_action_violation),
        "cone": robust_mean(cone_violation),
        "transport": robust_mean(transport_violation),
        "boundary": robust_mean(boundary_violation),
        "terminal": robust_mean(terminal_violation),
    }
    return torch.stack(tuple(parts.values())).mean(), parts


def registered_event_chain_binding_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: ERGT43Batch,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize the worst local fibre in each chain, independent of chain length."""

    valid = batch.base.attention_mask
    event_mask = batch.physical.event_mask & valid
    zero = outputs["native_answer_logits"].new_tensor(0.0)
    if not bool(event_mask.any()):
        return zero, {
            "positive_fibre": zero,
            "null_fibre": zero,
            "worst_positive_fibre": zero,
            "worst_null_fibre": zero,
        }

    relation_target = batch.base.supervision.relation_anchor_labels
    source_target = batch.base.supervision.event_source_positions.clamp_min(0)
    target_target = batch.base.supervision.event_target_positions.clamp_min(0)
    relation_nll = F.cross_entropy(
        outputs["event_relation_logits"].transpose(1, 2),
        relation_target,
        reduction="none",
    )
    source_nll = F.cross_entropy(
        outputs["event_source_logits"].transpose(1, 2),
        source_target,
        reduction="none",
    )
    target_nll = F.cross_entropy(
        outputs["event_target_logits"].transpose(1, 2),
        target_target,
        reduction="none",
    )
    positive_fibre_nll = (relation_nll + source_nll + target_nll) / 3.0
    null_fibre_nll = -torch.log_softmax(outputs["event_relation_logits"], dim=-1)[..., 0]
    null_mask = valid & ~event_mask

    def row_mean_and_max(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = mask.to(values.dtype)
        row_mean = (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        row_max = values.masked_fill(~mask, -torch.inf).amax(dim=1)
        row_max = torch.where(mask.any(dim=1), row_max, torch.zeros_like(row_max))
        return row_mean.mean(), row_max.mean()

    positive_mean, positive_worst = row_mean_and_max(positive_fibre_nll, event_mask)
    null_mean, null_worst = row_mean_and_max(null_fibre_nll, null_mask)
    positive_loss = 0.5 * (positive_mean + positive_worst)
    null_loss = 0.5 * (null_mean + null_worst)
    total = 0.5 * (positive_loss + null_loss)
    return total, {
        "positive_fibre": positive_loss,
        "null_fibre": null_loss,
        "worst_positive_fibre": positive_worst,
        "worst_null_fibre": null_worst,
    }


def _dyadic_path_edges(source: int, target: int) -> tuple[tuple[int, int], ...]:
    """Decompose a token displacement into the mesh offsets used by closure."""

    current = int(source)
    destination = int(target)
    edges: list[tuple[int, int]] = []
    while current != destination:
        delta = destination - current
        step = 1 << (abs(delta).bit_length() - 1)
        next_node = current + step if delta > 0 else current - step
        edges.append((current, next_node))
        current = next_node
    return tuple(edges)


def registered_multiscale_chain_support_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: ERGT43Batch,
    config: ERGT43Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep the worst gold event fibre inside the learned causal mesh margin.

    The forward solver receives a label-free valid-domain support floor. This
    teacher-phase term separately asks the underlying learned metric to recover
    that floor on its own. It is evaluated on the canonical dyadic path rather
    than a direct source-target edge, so it trains geodesic connectivity without
    turning the event labels into an executable graph.
    """

    zero = outputs["native_answer_logits"].new_tensor(0.0)
    event_mask = batch.physical.event_mask & batch.base.attention_mask
    event_positions = event_mask.nonzero(as_tuple=False)
    if event_positions.numel() == 0:
        return zero, {
            "mean_event_support": zero,
            "worst_event_support": zero,
            "capacity_margin": zero,
            "cone_margin": zero,
        }

    sources = batch.base.supervision.event_source_positions[event_mask].detach().cpu().tolist()
    targets = batch.base.supervision.event_target_positions[event_mask].detach().cpu().tolist()
    event_rows = event_positions[:, 0].detach().cpu().tolist()
    edge_rows: list[int] = []
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_events: list[int] = []
    retained_event_rows: list[int] = []
    for _event_id, (row, source, target) in enumerate(
        zip(event_rows, sources, targets, strict=True)
    ):
        path = _dyadic_path_edges(int(source), int(target))
        if not path:
            continue
        retained_event_rows.append(int(row))
        retained_event_id = len(retained_event_rows) - 1
        for left, right in path:
            edge_rows.append(int(row))
            edge_sources.append(left)
            edge_targets.append(right)
            edge_events.append(retained_event_id)
    if not edge_rows:
        return zero, {
            "mean_event_support": zero,
            "worst_event_support": zero,
            "capacity_margin": zero,
            "cone_margin": zero,
        }

    device = outputs["native_answer_logits"].device
    row_index = torch.tensor(edge_rows, dtype=torch.long, device=device)
    source_index = torch.tensor(edge_sources, dtype=torch.long, device=device)
    target_index = torch.tensor(edge_targets, dtype=torch.long, device=device)
    event_index = torch.tensor(edge_events, dtype=torch.long, device=device)
    raw_weight = outputs.get(
        "direct_program_world_edge_weight", outputs["program_world_edge_weight"]
    )[row_index, :, source_index, target_index]
    raw_length = outputs.get(
        "direct_program_world_edge_length", outputs["program_world_edge_length"]
    )[row_index, :, source_index, target_index]
    cone_budget = outputs["world_cone_budget"][row_index, :, source_index, target_index]
    active = outputs["active_worlds"].bool()
    if active.dim() == 1:
        active = active.unsqueeze(0).expand(batch.base.token_ids.size(0), -1)
    authorized = outputs["relation_authorized_worlds"].bool()
    if authorized.dim() == 1:
        authorized = authorized.unsqueeze(0).expand_as(active)
    valid_world = (active & authorized)[row_index]

    capacity_violation = F.relu(float(config.geodesic_backbone_support_floor) - raw_weight)
    cone_violation = F.relu(raw_length - cone_budget + float(config.geodesic_backbone_cone_margin))
    world_violation = capacity_violation + cone_violation
    infinity = torch.finfo(world_violation.dtype).max
    edge_violation = world_violation.masked_fill(~valid_world, infinity).amin(dim=-1)
    edge_capacity = capacity_violation.masked_fill(~valid_world, infinity).amin(dim=-1)
    edge_cone = cone_violation.masked_fill(~valid_world, infinity).amin(dim=-1)

    event_count = len(retained_event_rows)
    event_violation = edge_violation.new_zeros(event_count)
    event_capacity = edge_violation.new_zeros(event_count)
    event_cone = edge_violation.new_zeros(event_count)
    event_violation.scatter_reduce_(0, event_index, edge_violation, reduce="amax")
    event_capacity.scatter_reduce_(0, event_index, edge_capacity, reduce="amax")
    event_cone.scatter_reduce_(0, event_index, edge_cone, reduce="amax")

    retained_rows = torch.tensor(retained_event_rows, dtype=torch.long, device=device)
    row_losses: list[torch.Tensor] = []
    row_capacity: list[torch.Tensor] = []
    row_cone: list[torch.Tensor] = []
    for row in range(batch.base.token_ids.size(0)):
        row_mask = retained_rows == row
        if not bool(row_mask.any()):
            continue
        values = event_violation[row_mask]
        row_losses.append(0.5 * (values.mean() + values.amax()))
        row_capacity.append(event_capacity[row_mask].amax())
        row_cone.append(event_cone[row_mask].amax())
    total = torch.stack(row_losses).mean() if row_losses else zero
    return total, {
        "mean_event_support": event_violation.mean(),
        "worst_event_support": event_violation.amax(),
        "capacity_margin": torch.stack(row_capacity).mean() if row_capacity else zero,
        "cone_margin": torch.stack(row_cone).mean() if row_cone else zero,
    }


def native_geometric_training_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: ERGT43Batch,
    *,
    config: ERGT43Config,
    teacher_weight: float,
    answer_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train field formation, then verify the same answer path at zero teacher."""

    _, base_parts = physics_native_training_loss(
        outputs,
        batch.base,
        teacher_weight=teacher_weight,
        answer_weight=answer_weight,
    )
    mask = batch.physical.event_mask & batch.base.attention_mask
    if bool(mask.any()):
        action_loss = F.smooth_l1_loss(
            outputs["event_action_cost"][mask], batch.physical.action_cost[mask]
        )
        cone_loss = F.binary_cross_entropy_with_logits(
            outputs["event_cone_logit"][mask], batch.physical.cone_admissible[mask]
        )
        transmission_loss = F.binary_cross_entropy(
            outputs["event_transmission"][mask].clamp(1.0e-5, 1.0 - 1.0e-5),
            batch.physical.transmission[mask],
        )
        deficit_loss = F.smooth_l1_loss(
            outputs["event_boundary_deficit"][mask], batch.physical.boundary_deficit[mask]
        )
        terminal_loss = F.binary_cross_entropy(
            outputs["event_terminal_mass"][mask].clamp(1.0e-5, 1.0 - 1.0e-5),
            batch.physical.terminal_capacity[mask],
        )
    else:
        zero = outputs["native_answer_logits"].new_tensor(0.0)
        action_loss = cone_loss = transmission_loss = deficit_loss = terminal_loss = zero
    selected = outputs["selected_event_indices"]
    selected_mask = _gather_tokens(mask, selected).bool()
    if bool(selected_mask.any()):
        selected_action_target = _gather_tokens(batch.physical.action_cost, selected)
        effective_action_loss = F.smooth_l1_loss(
            outputs["event_continuous_hard_action"][selected_mask],
            selected_action_target[selected_mask],
        )
        path_violation = F.relu(
            float(config.geodesic_path_gate_floor)
            - outputs["event_world_closure_soft_gate"][selected_mask]
        )
        event_world_availability_loss = 0.5 * (path_violation.mean() + path_violation.amax())
        per_example_path_gate = (
            outputs["event_world_closure_soft_gate"]
            .masked_fill(
                ~selected_mask,
                1.0,
            )
            .amin(dim=1)
        )
        event_chain_coverage_loss = F.relu(
            float(config.geodesic_path_gate_floor) - per_example_path_gate
        ).mean()
    else:
        effective_action_loss = outputs["native_answer_logits"].new_tensor(0.0)
        event_world_availability_loss = outputs["native_answer_logits"].new_tensor(0.0)
        event_chain_coverage_loss = outputs["native_answer_logits"].new_tensor(0.0)
    physical_teacher = (
        action_loss + cone_loss + transmission_loss + deficit_loss + terminal_loss
    ) / 5.0
    event_chain_binding_loss, event_chain_binding_parts = registered_event_chain_binding_loss(
        outputs, batch
    )
    multiscale_chain_support_loss, multiscale_chain_support_parts = (
        registered_multiscale_chain_support_loss(outputs, batch, config)
    )
    causal_field_binding = (
        effective_action_loss
        + event_world_availability_loss
        + event_chain_coverage_loss
        + event_chain_binding_loss
        + multiscale_chain_support_loss
    ) / 5.0
    measurement_margin_loss, measurement_margin_parts = registered_measurement_margin_loss(
        outputs, batch, config
    )
    unsupervised_physics = (
        base_parts["bulk_boundary_loss"]
        + base_parts["conservation_loss"]
        + base_parts["world_entropy_loss"]
        + base_parts["role_routing_loss"]
        + base_parts["geometry_diversity_loss"]
    ) / 5.0
    total = (
        float(answer_weight) * base_parts["answer_loss"]
        + 2.0 * float(teacher_weight) * base_parts["structured_teacher_loss"]
        + 2.0 * float(teacher_weight) * physical_teacher
        + 2.0 * float(teacher_weight) * causal_field_binding
        + 2.0 * float(teacher_weight) * measurement_margin_loss
        + 0.5 * unsupervised_physics
    )
    parts = dict(base_parts)
    parts.update(
        {
            "total_loss": total,
            "physical_teacher_loss": physical_teacher,
            "physical_action_loss": action_loss,
            "physical_cone_loss": cone_loss,
            "physical_transmission_loss": transmission_loss,
            "physical_boundary_deficit_loss": deficit_loss,
            "physical_terminal_mass_loss": terminal_loss,
            "physical_effective_action_loss": effective_action_loss,
            "event_world_availability_loss": event_world_availability_loss,
            "event_chain_coverage_loss": event_chain_coverage_loss,
            "event_chain_binding_loss": event_chain_binding_loss,
            "multiscale_chain_support_loss": multiscale_chain_support_loss,
            "causal_field_binding_loss": causal_field_binding,
            "registered_measurement_margin_loss": measurement_margin_loss,
            **{
                f"registered_{name}_margin_loss": value
                for name, value in measurement_margin_parts.items()
            },
            **{
                f"event_chain_{name}_loss": value
                for name, value in event_chain_binding_parts.items()
            },
            **{
                f"multiscale_chain_{name}_loss": value
                for name, value in multiscale_chain_support_parts.items()
            },
            "unsupervised_physics_loss": unsupervised_physics,
        }
    )
    return total, parts


def architecture_contract(model: NativeGeometricBoundaryModel) -> dict[str, bool]:
    modules = tuple(model.modules())
    forward_parameters = set(inspect.signature(model.forward).parameters)
    return {
        "no_multihead_attention": not any(
            isinstance(module, nn.MultiheadAttention) for module in modules
        ),
        "no_transformer_layer": not any(
            isinstance(module, (nn.TransformerEncoder, nn.TransformerEncoderLayer))
            for module in modules
        ),
        "no_direct_answer_head": not any(
            "answer_head" in name for name, _ in model.named_modules()
        ),
        "old_discrete_graph_compiler_disabled": model.uses_discrete_graph_compiler is False,
        "old_generic_graph_executor_disabled": model.uses_generic_graph_executor is False,
        "raw_token_only_public_input": (
            model.public_input_is_raw_tokens_only is True
            and "raw_token_ids" in forward_parameters
            and "identity_fibre" not in forward_parameters
            and "token_ids" not in forward_parameters
        ),
        "identity_derived_inside_ergt": (
            model.derives_identity_inside_model is True
            and isinstance(model.raw_input_adapter, RawTokenInputAdapter)
        ),
        "hard_min_plus_target_solver": model.target_solver_is_hard_min_plus is True,
        "identity_fibre_metric_binding": (
            model.uses_identity_fibre_metric_binding is True
            and isinstance(model.identity_binder, IdentityFibreMetricBinder)
        ),
        "identity_fibre_quotient_binding": (model.uses_identity_fibre_quotient_binding is True),
        "position_independent_query_boundary_channel": hasattr(
            model.field_inducer, "query_boundary_from_semantic_seed"
        ) and model.field_inducer.query_boundary_from_semantic_seed is True,
        "gauge_fixed_world_action": model.uses_gauge_fixed_world_action is True,
        "registered_local_action_cell_before_world_residual": (
            model.uses_registered_local_action_cell is True
            and model.world_action_is_subcell_correction is True
            and model.config.world_action_scale < 0.5 * model.config.action_measurement_quantum
        ),
        "conserved_identity_initial_condition": (
            model.uses_conserved_identity_initial_condition is True
            and hasattr(model.substrate, "external_identity_seed")
        ),
        "adaptive_event_capacity": model.uses_adaptive_event_capacity is True,
        "hard_omission_overflow_gate": (
            model.uses_hard_omission_overflow_gate is True
            and isinstance(model.field_inducer, ProductManifoldFieldInducer)
        ),
        "hard_first_causal_event_fibres": (
            model.uses_hard_first_causal_event_fibres is True
            and isinstance(model.field_inducer, ProductManifoldFieldInducer)
        ),
        "closure_horizon_coupled_to_answer_horizon": (
            model.closure_horizon_coupled_to_answer_horizon is True
            and model.config.registered_geodesic_closure_steps >= model.config.max_hops
        ),
        "length_balanced_chain_binding_loss": (
            model.uses_length_balanced_chain_binding_loss is True
        ),
        "registered_event_capacity_bound": isinstance(
            model.field_inducer, ProductManifoldFieldInducer
        ),
        "event_source_role_world_closure": (
            model.closes_event_source_over_authorized_role_worlds is True
            and model.config.close_event_source_over_role_worlds is True
        ),
        "product_manifold_geodesic_closure": (
            model.uses_product_manifold_geodesic_closure is True
            and model.config.registered_geodesic_closure_steps > 0
        ),
        "multiscale_causal_backbone": (
            model.uses_multiscale_causal_backbone is True
            and model.config.geodesic_backbone_levels > 0
        ),
        "padding_independent_multiscale_support": (
            model.uses_padding_independent_multiscale_support is True
            and model.config.geodesic_backbone_support_floor > 0.0
        ),
        "worst_chain_support_margin": (
            model.uses_worst_chain_support_margin is True
            and model.config.geodesic_backbone_cone_margin >= 0.0
        ),
        "world_lens_transitions": (
            model.uses_world_lens_transitions is True
            and model.config.world_lens_transition_cost >= 0.0
        ),
        "curvature_is_observer_only": model.curvature_observer_only is True,
        "spectrum_is_observer_only": model.spectrum_observer_only is True,
        "world_metric_is_active": hasattr(model.substrate, "world_log_scale"),
        "typed_conditions_are_explicit": isinstance(
            model.physical_conditions, LocalPhysicalConditionField
        ),
        "registered_measurement_before_hard_solver": (
            model.config.action_measurement_quantum > 0.0
            and model.config.attenuated_mass_level < model.config.transmission_floor
        ),
    }


__all__ = [
    "ERGT43Config",
    "ERGT43Error",
    "HardBoundarySolution",
    "IdentityFibreMetricBinder",
    "INTERVENTIONS",
    "LocalPhysicalConditionField",
    "NativeGeometricBoundaryModel",
    "ProductManifoldFieldInducer",
    "RawTokenInputAdapter",
    "SCHEMA_VERSION",
    "architecture_contract",
    "canonical_identity_fibre_mask",
    "event_graph_spectral_observer",
    "exact_identity_fibre_gate",
    "hard_min_plus_typed_boundary_solve",
    "hard_solutions_from_outputs",
    "gauge_fixed_world_action",
    "native_geometric_training_loss",
    "oracle_solutions_from_batch",
    "padding_independent_multiscale_support",
    "registered_event_selection_audit",
    "registered_event_chain_binding_loss",
    "registered_multiscale_chain_support_loss",
    "registered_hard_action",
    "registered_hard_measurements",
    "registered_measurement_margin_loss",
    "soft_min_plus_typed_boundary_solve",
    "valid_domain_multiscale_backbone_mask",
]
