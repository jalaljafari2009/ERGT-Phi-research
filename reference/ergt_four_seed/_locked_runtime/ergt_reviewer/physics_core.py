"""Physics-native field substrate used by the reviewer ERGT path.

Only the governing Psi/world geometry, continuous field induction, and
training losses are retained here. Historical discrete compilation and graph
execution code is intentionally absent from the reviewer package.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

from .data_schema import (
    QUERY_ROLE_NAMES,
    REGISTER_CARDINALITY,
    RELATION_NAMES,
    ERGT35Batch,
)
from .world_contract import (
    ROLE_WORLD_ASSIGNMENTS,
    TYPED_SLOT_NAMES,
    WORLD_NAMES,
)

SCHEMA_VERSION: Final = "ergt-reviewer-physics-native-core-v1"
INTERVENTIONS: Final[tuple[str, ...]] = (
    "full",
    "no_phi",
    "no_event_backreaction",
    "no_transport",
    "no_memory_geometry",
    "shuffled_geometry",
    "random_geometry",
    *(f"drop_world_{index}" for index in range(len(WORLD_NAMES))),
    *(f"only_world_{index}" for index in range(len(WORLD_NAMES))),
)


class ERGT42Error(ValueError):
    """Raised when a physics-native contract is violated."""

@dataclass(frozen=True)
class ERGT42Config:
    """Configuration for the complete ERGT-42 governing path."""

    vocab_size: int
    max_tokens: int
    hidden_dim: int = 48
    psi_rank: int = 16
    n_worlds: int = 8
    field_steps: int = 3
    sparse_top_k: int = 8
    max_program_hops: int = 32
    local_decay: float = 5.0
    bridge_decay: float = 18.0
    edge_memory_rate: float = 0.45
    field_momentum: float = 0.55
    transport_temperature: float = 0.72
    cone_velocity_floor: float = 1.25
    identity_fibre_dim: int = 64
    adaptive_event_capacity: bool = False
    event_slot_reserve_factor: int = 1
    event_selection_floor: float = 0.45
    close_event_source_over_role_worlds: bool = False
    eps: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.vocab_size <= 2 or self.max_tokens <= 1:
            raise ERGT42Error("vocab_size and max_tokens must define a real input space")
        if self.hidden_dim < 8 or self.psi_rank <= 0:
            raise ERGT42Error("hidden_dim and psi_rank must be positive")
        if self.n_worlds != len(WORLD_NAMES):
            raise ERGT42Error("ERGT-42 requires the complete eight-world field")
        if self.field_steps <= 0 or self.sparse_top_k <= 0:
            raise ERGT42Error("field_steps and sparse_top_k must be positive")
        if self.max_program_hops <= 0:
            raise ERGT42Error("max_program_hops must be positive")
        if self.identity_fibre_dim <= 0:
            raise ERGT42Error("identity_fibre_dim must be positive")
        if self.event_slot_reserve_factor <= 0:
            raise ERGT42Error("event_slot_reserve_factor must be positive")
        if not 0.0 < self.event_selection_floor < 1.0:
            raise ERGT42Error("event_selection_floor must lie in (0, 1)")
        if not 0.0 <= self.edge_memory_rate < 1.0:
            raise ERGT42Error("edge_memory_rate must be in [0, 1)")
        if not 0.0 <= self.field_momentum < 1.0:
            raise ERGT42Error("field_momentum must be in [0, 1)")

def _sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    even_dim = max(2, dim + dim % 2)
    scale = torch.exp(
        torch.arange(0, even_dim, 2, dtype=torch.float32, device=device)
        * (-math.log(10_000.0) / even_dim)
    )
    encoding = torch.zeros(length, even_dim, device=device)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale)
    return encoding[:, :dim]

def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)

def _normalized_entropy(probabilities: torch.Tensor, valid_count: torch.Tensor) -> torch.Tensor:
    # Keep masked zero-probability cells exactly neutral.  Clamping the
    # probabilities themselves assigned a tiny entropy mass to every padding
    # cell, so the same valid sequence acquired a different world potential
    # when collated to a longer tensor.
    entropy = -(probabilities * probabilities.clamp_min(1.0e-8).log()).sum(dim=-1)
    denominator = valid_count.clamp_min(2).to(probabilities.dtype).log()
    return torch.where(valid_count > 1, entropy / denominator, torch.ones_like(entropy))

class PhysicsNativeSubstrate(nn.Module):
    """Evolve Psi into sparse world metrics and transport without QK attention."""

    def __init__(self, config: ERGT42Config) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        worlds = config.n_worlds
        self.token_embedding = nn.Embedding(config.vocab_size, hidden)
        self.identity_seed = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.external_identity_seed = nn.Sequential(
            nn.LayerNorm(config.identity_fibre_dim),
            nn.Linear(config.identity_fibre_dim, hidden, bias=False),
        )
        self.identity_initial_condition_gain_raw = nn.Parameter(torch.tensor(0.0))
        self.psi_seed = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.coherence_left = nn.Linear(hidden, config.psi_rank, bias=False)
        self.coherence_right = nn.Linear(hidden, config.psi_rank, bias=False)
        self.reconstruct_left = nn.Linear(hidden, config.psi_rank, bias=False)
        self.reconstruct_right = nn.Linear(hidden, config.psi_rank, bias=False)
        self.stability_head = nn.Linear(hidden, 1)
        self.velocity_head = nn.Linear(hidden, 1)
        self.world_value = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(worlds)])
        self.field_update = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.field_norm = nn.LayerNorm(hidden)
        self.bulk_decoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.world_log_scale = nn.Parameter(torch.zeros(worlds))
        self.world_bias = nn.Parameter(torch.zeros(worlds))

    def _pair_observables(
        self,
        psi: torch.Tensor,
        identity: torch.Tensor,
        attention_mask: torch.Tensor,
        bulk_deficit: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, tokens, _ = psi.shape
        valid = attention_mask[:, :, None] & attention_mask[:, None, :]
        diagonal = torch.eye(tokens, dtype=torch.bool, device=psi.device).unsqueeze(0)
        off_diagonal = valid & ~diagonal

        density = psi.square().mean(dim=-1)
        density = density / density.amax(dim=1, keepdim=True).clamp_min(self.config.eps)
        coherence_left = F.normalize(
            self.coherence_left(psi), dim=-1, eps=self.config.eps
        )
        coherence_right = F.normalize(
            self.coherence_right(psi), dim=-1, eps=self.config.eps
        )
        coherence = (
            torch.einsum("bid,bjd->bij", coherence_left, coherence_right) + 1.0
        ) * 0.5
        coherence = coherence.clamp(self.config.eps, 1.0)

        reconstruction_left = F.normalize(
            self.reconstruct_left(psi), dim=-1, eps=self.config.eps
        )
        reconstruction_right = F.normalize(
            self.reconstruct_right(psi), dim=-1, eps=self.config.eps
        )
        reconstruction = torch.sigmoid(
            2.0
            * torch.einsum(
                "bid,bjd->bij", reconstruction_left, reconstruction_right
            )
        )
        identity_distance = (identity[:, :, None, :] - identity[:, None, :, :]).square().sum(dim=-1)
        identity_overlap = torch.exp(-2.0 * identity_distance)

        positions = torch.arange(tokens, device=psi.device)
        separation = (positions[:, None] - positions[None, :]).abs().to(psi.dtype)
        locality = torch.exp(-separation / self.config.local_decay).unsqueeze(0)
        bridge = 1.0 - torch.exp(-separation / self.config.bridge_decay).unsqueeze(0)

        stability = torch.sigmoid(self.stability_head(psi).squeeze(-1))
        pair_stability = torch.sqrt(
            stability[:, :, None].clamp_min(self.config.eps)
            * stability[:, None, :].clamp_min(self.config.eps)
        )
        density_delta = (density[:, :, None] - density[:, None, :]).abs()
        gradient_fit = torch.exp(-density_delta)

        entropy_logits = coherence.masked_fill(~off_diagonal, -1.0e4)
        entropy_distribution = torch.softmax(entropy_logits, dim=-1)
        entropy_distribution = entropy_distribution * off_diagonal.to(
            entropy_distribution.dtype
        )
        entropy = _normalized_entropy(entropy_distribution, off_diagonal.sum(dim=-1))
        order = (1.0 - entropy).clamp(self.config.eps, 1.0)
        pair_order = torch.sqrt(order[:, :, None] * order[:, None, :]).clamp_min(self.config.eps)

        masked_coherence = coherence.masked_fill(~off_diagonal, 0.0)
        novelty = (1.0 - masked_coherence.amax(dim=-1)).clamp(self.config.eps, 1.0)
        pair_novelty = novelty[:, None, :].expand(batch, tokens, tokens)
        density_flow = torch.sigmoid(4.0 * (density[:, :, None] - density[:, None, :]))
        boundary_reconstructibility = torch.exp(
            -0.5 * (bulk_deficit[:, :, None] + bulk_deficit[:, None, :])
        )

        return {
            "valid": valid,
            "off_diagonal": off_diagonal,
            "density": density,
            "coherence": coherence,
            "reconstruction": reconstruction,
            "identity_overlap": identity_overlap,
            "identity_distance": identity_distance,
            "locality": locality.expand(batch, -1, -1),
            "bridge": bridge.expand(batch, -1, -1),
            "stability": pair_stability,
            "gradient_fit": gradient_fit,
            "order": pair_order,
            "novelty": pair_novelty,
            "density_flow": density_flow,
            "boundary_reconstructibility": boundary_reconstructibility,
            "density_delta": density_delta,
        }

    def _world_potentials(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        locality = obs["locality"]
        coherence = obs["coherence"]
        reconstruction = obs["reconstruction"]
        stability = obs["stability"]
        bridge = obs["bridge"]
        density_delta = obs["density_delta"]
        potentials = torch.stack(
            (
                locality * obs["gradient_fit"] * stability,
                torch.sqrt(coherence * obs["identity_overlap"]).clamp_min(self.config.eps)
                * obs["order"],
                reconstruction * obs["boundary_reconstructibility"],
                stability * obs["identity_overlap"] * (0.5 + 0.5 * locality),
                bridge * torch.maximum(coherence, obs["identity_overlap"]) * reconstruction,
                density_delta * stability,
                obs["novelty"] * (0.5 + 0.5 * bridge),
                obs["density_flow"] * (0.5 + 0.5 * coherence),
            ),
            dim=1,
        )
        scale = self.world_log_scale.exp().view(1, -1, 1, 1)
        bias = self.world_bias.view(1, -1, 1, 1)
        return torch.sigmoid(scale * torch.logit(potentials.clamp(1.0e-5, 1.0 - 1.0e-5)) + bias)

    def _phi_gate(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        factors = torch.stack(
            (
                obs["coherence"],
                obs["reconstruction"],
                obs["stability"],
                obs["gradient_fit"],
                obs["order"],
                obs["boundary_reconstructibility"],
            ),
            dim=0,
        ).clamp_min(self.config.eps)
        return torch.exp(factors.log().mean(dim=0))

    def _sparse_mask(self, scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        tokens = scores.size(-1)
        k = min(self.config.sparse_top_k, max(1, tokens - 1))
        masked = scores.masked_fill(~valid[:, None, :, :], -1.0)
        indices = masked.topk(k=k, dim=-1).indices
        hard = torch.zeros_like(scores).scatter_(-1, indices, 1.0)
        hard = hard * valid[:, None, :, :].to(scores.dtype)
        soft = scores * valid[:, None, :, :].to(scores.dtype)
        return hard + soft - soft.detach()

    def _bulk_boundary_deficit(
        self,
        psi: torch.Tensor,
        transport: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_transport = transport.mean(dim=1)
        one_shell = torch.bmm(mean_transport, psi)
        two_reach = torch.bmm(mean_transport, mean_transport)
        two_shell = torch.bmm(two_reach, psi)
        reconstructed_one = self.bulk_decoder(one_shell)
        reconstructed_two = self.bulk_decoder(two_shell)
        deficit = 0.5 * (
            (reconstructed_one - psi).square().mean(dim=-1)
            + (reconstructed_two - psi).square().mean(dim=-1)
        )
        deficit = deficit * attention_mask.to(deficit.dtype)
        bulk_loss = deficit.sum() / attention_mask.sum().clamp_min(1)
        return deficit, bulk_loss

    @staticmethod
    def _intervene(
        weights: torch.Tensor,
        *,
        intervention: str,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = torch.ones(weights.size(1), dtype=torch.bool, device=weights.device)
        if intervention == "full" or intervention in {
            "no_phi",
            "no_event_backreaction",
            "no_transport",
            "no_memory_geometry",
        }:
            return weights, active
        if intervention == "shuffled_geometry":
            generator = torch.Generator(device=weights.device)
            generator.manual_seed(int(seed))
            permutation = torch.randperm(
                weights.size(-1), generator=generator, device=weights.device
            )
            return weights.index_select(-1, permutation), active
        if intervention == "random_geometry":
            generator = torch.Generator(device=weights.device)
            generator.manual_seed(int(seed))
            random = torch.rand(weights.shape, generator=generator, device=weights.device)
            return random * (weights > 0).to(weights.dtype), active
        if intervention.startswith("drop_world_"):
            world = int(intervention.rsplit("_", 1)[1])
            active[world] = False
        elif intervention.startswith("only_world_"):
            world = int(intervention.rsplit("_", 1)[1])
            active[:] = False
            active[world] = True
        else:
            raise ERGT42Error(f"unknown ERGT-42 intervention: {intervention}")
        return weights * active.view(1, -1, 1, 1).to(weights.dtype), active

    def spectral_observer(
        self,
        edge_weight: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Observe world spectra without allowing spectrum to drive the answer."""

        batch_limit = min(2, edge_weight.size(0))
        token_limit = min(64, edge_weight.size(-1))
        entropies: list[torch.Tensor] = []
        ranks: list[torch.Tensor] = []
        gaps: list[torch.Tensor] = []
        signatures: list[torch.Tensor] = []
        for row in range(batch_limit):
            valid_tokens = min(int(attention_mask[row].sum().item()), token_limit)
            if valid_tokens < 2:
                continue
            for world in range(edge_weight.size(1)):
                adjacency = edge_weight[row, world, :valid_tokens, :valid_tokens].detach()
                adjacency = 0.5 * (adjacency + adjacency.transpose(0, 1))
                degree = adjacency.sum(dim=-1)
                laplacian = torch.diag(degree) - adjacency
                eigenvalues = torch.linalg.eigvalsh(laplacian).clamp_min(0.0)
                distribution = eigenvalues / eigenvalues.sum().clamp_min(self.config.eps)
                entropy = -(distribution.clamp_min(self.config.eps).log() * distribution).sum()
                entropy = entropy / math.log(float(max(2, valid_tokens)))
                entropies.append(entropy)
                ranks.append(entropy.mul(math.log(float(max(2, valid_tokens)))).exp())
                gaps.append(eigenvalues[1] if eigenvalues.numel() > 1 else eigenvalues[0])
                signatures.append(F.pad(eigenvalues[:16], (0, max(0, 16 - eigenvalues.numel()))))
        if not entropies:
            zero = edge_weight.new_tensor(0.0)
            return {
                "spectral_entropy": zero,
                "spectral_effective_rank": zero,
                "spectral_gap": zero,
                "spectral_world_diversity": zero,
            }
        signature = torch.stack(signatures)
        signature = F.normalize(signature, dim=-1, eps=self.config.eps)
        diversity = 1.0 - torch.mm(signature, signature.transpose(0, 1)).mean()
        return {
            "spectral_entropy": torch.stack(entropies).mean(),
            "spectral_effective_rank": torch.stack(ranks).mean(),
            "spectral_gap": torch.stack(gaps).mean(),
            "spectral_world_diversity": diversity,
        }

    def curvature_observer(
        self,
        sparse_weight: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Observe a differentiable Forman-style bottleneck proxy.

        The learned worlds are directed and weighted, so this is deliberately
        reported as a proxy rather than as exact Riemann or Ollivier curvature.
        Symmetrization makes the local degree term gauge-independent; the graph
        gradient records how sharply that local curvature changes across an
        edge. Neither quantity enters transport or the answer in ERGT-43 V12.
        """

        valid = attention_mask[:, None, :, None] & attention_mask[:, None, None, :]
        off_diagonal = ~torch.eye(
            sparse_weight.size(-1), device=sparse_weight.device, dtype=torch.bool
        )[None, None]
        valid = valid & off_diagonal
        symmetric = 0.5 * (sparse_weight + sparse_weight.transpose(-1, -2))
        symmetric = symmetric * valid.to(symmetric.dtype)
        scale = symmetric.amax(dim=(-1, -2), keepdim=True).clamp_min(self.config.eps)
        adjacency = symmetric / scale
        degree = adjacency.sum(dim=-1)
        raw_curvature = 4.0 - degree.unsqueeze(-1) - degree.unsqueeze(-2)
        curvature = torch.tanh(raw_curvature / 4.0) * adjacency
        node_denominator = adjacency.sum(dim=-1).clamp_min(self.config.eps)
        node_curvature = curvature.sum(dim=-1) / node_denominator
        curvature_gradient = (
            node_curvature.unsqueeze(-2) - node_curvature.unsqueeze(-1)
        ) * adjacency
        active = adjacency > self.config.eps
        active_count = active.sum().clamp_min(1).to(adjacency.dtype)
        negative_fraction = ((raw_curvature < 0.0) & active).sum().to(adjacency.dtype)
        negative_fraction = negative_fraction / active_count
        return {
            "world_forman_curvature_proxy": curvature,
            "world_curvature_graph_gradient": curvature_gradient,
            "curvature_proxy_mean": curvature.sum() / adjacency.sum().clamp_min(self.config.eps),
            "curvature_proxy_negative_fraction": negative_fraction,
            "curvature_gradient_mean_abs": curvature_gradient.abs().sum()
            / adjacency.sum().clamp_min(self.config.eps),
        }

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        identity_fibre: torch.Tensor | None = None,
        intervention: str = "full",
        intervention_seed: int = 0,
        observe_spectrum: bool = False,
    ) -> dict[str, torch.Tensor]:
        if token_ids.dim() != 2 or attention_mask.shape != token_ids.shape:
            raise ERGT42Error("token_ids and attention_mask must have matching [B, T] shape")
        if token_ids.size(1) > self.config.max_tokens:
            raise ERGT42Error("input exceeds configured positional capacity")
        position = _sinusoidal_positions(
            token_ids.size(1), self.config.hidden_dim, token_ids.device
        )
        token_seed = self.token_embedding(token_ids)
        # Identity is an initial-condition charge: geometry may move and update
        # payload state, but repeated symbols must remain bindable after transport.
        if identity_fibre is None:
            identity = self.identity_seed(token_seed)
            identity_initial_condition = torch.zeros_like(token_seed)
        else:
            if identity_fibre.dim() != 3 or identity_fibre.shape[:2] != token_ids.shape:
                raise ERGT42Error("identity_fibre must have shape [B, T, D]")
            if identity_fibre.size(-1) != self.config.identity_fibre_dim:
                raise ERGT42Error(
                    "identity_fibre width does not match configured conserved charge"
                )
            identity = identity_fibre.to(device=token_ids.device, dtype=token_seed.dtype)
            identity_initial_condition = self.external_identity_seed(identity)
        identity = F.normalize(identity, dim=-1, eps=self.config.eps)
        identity = identity * attention_mask.unsqueeze(-1).to(identity.dtype)
        identity_gain = 0.75 + 0.50 * torch.sigmoid(self.identity_initial_condition_gain_raw)
        identity_initial_condition = identity_initial_condition * attention_mask.unsqueeze(-1).to(
            token_seed.dtype
        )
        normalized_condition = F.normalize(
            identity_initial_condition, dim=-1, eps=self.config.eps
        )
        condition_distance = (
            normalized_condition[:, :, None, :] - normalized_condition[:, None, :, :]
        ).square().sum(dim=-1)
        exact_identity_distance = (
            identity[:, :, None, :] - identity[:, None, :, :]
        ).square().sum(dim=-1)
        valid_pair = attention_mask[:, :, None] & attention_mask[:, None, :]
        distinct_identity = valid_pair & (exact_identity_distance > 1.0e-5)
        distinct_count = distinct_identity.sum().clamp_min(1)
        identity_condition_collision_rate = (
            distinct_identity & (condition_distance <= 1.0e-6)
        ).sum().to(token_seed.dtype) / distinct_count.to(token_seed.dtype)
        identity_condition_min_separation = condition_distance.masked_fill(
            ~distinct_identity, float("inf")
        ).amin()
        identity_condition_min_separation = torch.where(
            distinct_identity.any(),
            identity_condition_min_separation,
            condition_distance.new_tensor(0.0),
        )
        psi = self.psi_seed(
            token_seed + position.unsqueeze(0) + identity_gain * identity_initial_condition
        )
        psi = psi * attention_mask.unsqueeze(-1).to(psi.dtype)
        # Operator identity is an internal charge of the initial Psi state. It
        # may be carried to geometric endpoints, but the evolving metric must
        # not transmute plus_one, plus_two, or hold into one another.
        typed_payload_charge = F.normalize(psi, dim=-1, eps=self.config.eps)
        typed_payload_charge = typed_payload_charge * attention_mask.unsqueeze(-1).to(psi.dtype)
        momentum = torch.zeros_like(psi)
        edge_memory: torch.Tensor | None = None
        bulk_deficit = torch.zeros_like(token_ids, dtype=psi.dtype)
        bulk_losses: list[torch.Tensor] = []
        final: dict[str, torch.Tensor] = {}

        for _ in range(self.config.field_steps):
            obs = self._pair_observables(psi, identity, attention_mask, bulk_deficit)
            phi = self._phi_gate(obs)
            if intervention == "no_phi":
                phi = torch.ones_like(phi)
            potential = self._world_potentials(obs)
            candidate_weight = potential * phi.unsqueeze(1)
            if edge_memory is None or intervention == "no_memory_geometry":
                edge_memory = candidate_weight
            else:
                rate = self.config.edge_memory_rate
                edge_memory = rate * edge_memory + (1.0 - rate) * candidate_weight
            edge_memory = edge_memory * obs["off_diagonal"][:, None].to(psi.dtype)
            edge_memory, active_worlds = self._intervene(
                edge_memory,
                intervention=intervention,
                seed=intervention_seed,
            )
            sparse_gate = self._sparse_mask(edge_memory, obs["off_diagonal"])
            sparse_weight = edge_memory * sparse_gate
            edge_length = -torch.log(edge_memory.clamp_min(self.config.eps))
            edge_length = edge_length.masked_fill(~obs["off_diagonal"][:, None], 30.0)

            velocity = self.config.cone_velocity_floor + 2.0 * torch.sigmoid(
                self.velocity_head(psi).squeeze(-1)
            )
            positions = torch.arange(token_ids.size(1), device=psi.device)
            separation = (positions[:, None] - positions[None, :]).abs().to(psi.dtype)
            # Proper time is an intrinsic per-example quantity.  Normalizing
            # by the padded tensor width made cone budgets depend on how an
            # otherwise identical example happened to be batched at audit
            # time.  The valid-token extent removes that external gauge.
            valid_extent = (
                attention_mask.sum(dim=1).sub(1).clamp_min(1).to(psi.dtype).view(-1, 1, 1)
            )
            proper_time = 1.0 + separation.unsqueeze(0) / valid_extent
            cone_budget = velocity[:, None, :, None] * proper_time[:, None]
            cone = edge_length <= cone_budget
            admissible = (sparse_gate.detach() > 0.5) & cone & obs["off_diagonal"][:, None]
            action = (
                edge_length
                + 0.75 * (1.0 - obs["boundary_reconstructibility"][:, None])
                + 0.50 * (1.0 - obs["stability"][:, None])
                + (~cone).to(psi.dtype) * 8.0
            )

            diagonal = torch.eye(token_ids.size(1), dtype=torch.bool, device=psi.device).view(
                1, 1, token_ids.size(1), token_ids.size(1)
            )
            identity_transport = diagonal & attention_mask[:, None, :, None]
            transport_mask = admissible | identity_transport
            transport_logits = -action / self.config.transport_temperature
            transport_logits = transport_logits.masked_fill(~transport_mask, -1.0e4)
            transport = torch.softmax(transport_logits, dim=-1)
            program_admissible = admissible
            if intervention == "no_transport":
                transport = identity_transport.to(psi.dtype).expand(
                    token_ids.size(0), self.config.n_worlds, -1, -1
                )
                program_admissible = torch.zeros_like(admissible)
            transported_worlds = torch.stack(
                [
                    torch.bmm(transport[:, world], self.world_value[world](psi))
                    for world in range(self.config.n_worlds)
                ],
                dim=1,
            )
            active_count = active_worlds.sum().clamp_min(1).to(psi.dtype)
            message = (transported_worlds * active_worlds.view(1, -1, 1, 1).to(psi.dtype)).sum(
                dim=1
            ) / active_count
            force = self.field_update(torch.cat((psi, message), dim=-1))
            momentum = (
                self.config.field_momentum * momentum + (1.0 - self.config.field_momentum) * force
            )
            psi = self.field_norm(psi + momentum)
            psi = psi * attention_mask.unsqueeze(-1).to(psi.dtype)
            bulk_deficit, bulk_loss = self._bulk_boundary_deficit(psi, transport, attention_mask)
            bulk_losses.append(bulk_loss)
            final = {
                "psi": psi,
                "semantic_token_seed": token_seed,
                "identity": identity,
                "identity_initial_condition": identity_initial_condition,
                "identity_initial_condition_gain": identity_gain,
                "identity_initial_condition_collision_rate": (
                    identity_condition_collision_rate
                ),
                "identity_initial_condition_min_separation": (
                    identity_condition_min_separation
                ),
                "typed_payload_charge": typed_payload_charge,
                "phi": phi,
                "world_potential": potential,
                "world_edge_weight": edge_memory,
                "world_sparse_weight": sparse_weight,
                "world_edge_length": edge_length,
                "world_action": action,
                "world_cone": cone.to(psi.dtype),
                "world_cone_budget": cone_budget,
                "world_program_mask": program_admissible,
                "world_transport": transport,
                "transported_worlds": transported_worlds,
                "active_worlds": active_worlds,
                "bulk_boundary_deficit": bulk_deficit,
                "bulk_boundary_loss": torch.stack(bulk_losses).mean(),
                "density": obs["density"],
                "coherence": obs["coherence"],
                "stability": obs["stability"],
                "reconstructibility": obs["boundary_reconstructibility"],
                "identity_overlap": obs["identity_overlap"],
                "identity_distance": obs["identity_distance"],
                "program_transport_enabled": psi.new_tensor(float(intervention != "no_transport")),
                "event_backreaction_enabled": psi.new_tensor(
                    float(intervention != "no_event_backreaction")
                ),
            }

        valid_edges = final["world_program_mask"].to(psi.dtype)
        edge_mass = (final["world_edge_weight"] * valid_edges).sum(dim=(-1, -2))
        edge_mass = edge_mass / valid_edges.sum(dim=(-1, -2)).clamp_min(1.0)
        usage = edge_mass.mean(dim=0)
        usage_probability = usage / usage.sum().clamp_min(self.config.eps)
        final["world_usage"] = usage_probability
        final["world_usage_entropy"] = -(
            usage_probability.clamp_min(self.config.eps)
            * usage_probability.clamp_min(self.config.eps).log()
        ).sum() / math.log(float(self.config.n_worlds))
        flattened = F.normalize(
            final["world_edge_weight"].flatten(start_dim=2), dim=-1, eps=self.config.eps
        )
        cosine = torch.einsum("bwd,bvd->bwv", flattened, flattened)
        triangle = torch.triu(
            torch.ones(
                self.config.n_worlds,
                self.config.n_worlds,
                dtype=torch.bool,
                device=psi.device,
            ),
            diagonal=1,
        )
        final["world_geometry_diversity"] = (1.0 - cosine[:, triangle].mean()).clamp(0.0, 1.0)
        # Curvature is a frozen observer.  Detaching here guarantees that it
        # cannot become a hidden training path even if downstream reporting
        # retains the returned tensors.
        with torch.no_grad():
            final.update(
                self.curvature_observer(
                    final["world_sparse_weight"].detach(),
                    attention_mask,
                )
            )
        if observe_spectrum:
            final.update(self.spectral_observer(final["world_edge_weight"], attention_mask))
        else:
            zero = psi.new_tensor(0.0)
            final.update(
                {
                    "spectral_entropy": zero,
                    "spectral_effective_rank": zero,
                    "spectral_gap": zero,
                    "spectral_world_diversity": zero,
                }
            )
        return final

class RolePreservingGeometricFieldInducer(nn.Module):
    """Compile typed slots from separated world fields, never from token equality."""

    def __init__(self, config: ERGT42Config) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.config = config
        self.world_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(config.n_worlds)
            ]
        )
        self.route_context = nn.Linear(hidden, len(TYPED_SLOT_NAMES) * config.n_worlds)
        self.role_world_prior = nn.Parameter(torch.zeros(len(TYPED_SLOT_NAMES), config.n_worlds))
        contract = torch.zeros(len(TYPED_SLOT_NAMES), config.n_worlds, dtype=torch.bool)
        for role_index, role_name in enumerate(TYPED_SLOT_NAMES):
            contract[role_index, list(ROLE_WORLD_ASSIGNMENTS[role_name])] = True
        self.register_buffer("role_world_contract", contract, persistent=True)
        self.entity_head = nn.Linear(hidden, 1)
        self.query_role_head = nn.Linear(hidden, len(QUERY_ROLE_NAMES))
        self.query_boundary_from_semantic_seed = True
        self.relation_source = nn.Linear(hidden, hidden, bias=False)
        self.relation_target = nn.Linear(hidden, hidden, bias=False)
        self.geometry_pair = nn.Linear(3, hidden, bias=False)
        self.edge_presence_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.event_relation_head = nn.Linear(hidden, len(RELATION_NAMES))
        self.event_source_query = nn.Linear(hidden, hidden, bias=False)
        self.event_target_query = nn.Linear(hidden, hidden, bias=False)
        self.event_endpoint_key = nn.Linear(hidden, hidden, bias=False)
        self.endpoint_coherence_gain = nn.Parameter(torch.tensor(3.0))
        self.geodesic_binding_gain = nn.Parameter(torch.tensor(4.0))
        self.event_backreaction_gain = nn.Parameter(torch.tensor(1.5))
        self.pointer_state = nn.Linear(hidden, hidden, bias=False)
        self.query_pointer_query = nn.Linear(hidden, hidden, bias=False)
        self.boundary_head = nn.Linear(hidden, REGISTER_CARDINALITY)
        self.boundary_token_head = nn.Linear(hidden, REGISTER_CARDINALITY)

    def _role_states(
        self,
        state: Mapping[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        psi = state["psi"]
        worlds = state["transported_worlds"]
        adapted = torch.stack(
            [self.world_adapters[index](worlds[:, index]) for index in range(self.config.n_worlds)],
            dim=1,
        )
        context = _masked_mean(psi, attention_mask, dim=1)
        route_logits = self.route_context(context).view(
            psi.size(0), len(TYPED_SLOT_NAMES), self.config.n_worlds
        )
        route_logits = route_logits + self.role_world_prior.unsqueeze(0)
        route_logits = route_logits.masked_fill(~self.role_world_contract.unsqueeze(0), -1.0e4)
        active = state["active_worlds"].view(1, 1, -1)
        route_logits = route_logits.masked_fill(~active, -1.0e4)
        route = torch.softmax(route_logits, dim=-1)
        availability = (self.role_world_contract & state["active_worlds"].unsqueeze(0)).any(dim=-1)
        route = route * availability.view(1, -1, 1).to(route.dtype)
        route_entropy_by_role = -(route.clamp_min(self.config.eps).log() * route).sum(dim=-1)
        authorized_count = self.role_world_contract.sum(dim=-1).clamp_min(2)
        route_entropy_by_role = route_entropy_by_role / authorized_count.to(
            route.dtype
        ).log().unsqueeze(0)
        role_world_usage = route.mean(dim=0)
        functional_world_usage = role_world_usage.mean(dim=0)
        functional_usage_probability = (
            functional_world_usage / functional_world_usage.sum().clamp_min(self.config.eps)
        )
        functional_world_usage_entropy = -(
            functional_usage_probability.clamp_min(self.config.eps)
            * functional_usage_probability.clamp_min(self.config.eps).log()
        ).sum() / math.log(float(self.config.n_worlds))
        role_states = {
            role: torch.einsum("bw,bwth->bth", route[:, index], adapted)
            for index, role in enumerate(TYPED_SLOT_NAMES)
        }
        routing_metrics = torch.stack(
            (
                route_entropy_by_role.mean(),
                functional_world_usage.min(),
                functional_world_usage.max(),
                functional_world_usage_entropy,
            )
        )
        return role_states, route, availability, routing_metrics, role_world_usage

    @staticmethod
    def _role_context(
        hidden: torch.Tensor,
        role_probability: torch.Tensor,
        role_id: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        weight = role_probability[..., role_id] * attention_mask.to(hidden.dtype)
        return torch.einsum("bt,bth->bh", weight, hidden) / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-6)

    @staticmethod
    def _oriented_neighbor_fields(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return left/right one-step field stencils without inspecting token ids."""

        left = torch.zeros_like(hidden)
        right = torch.zeros_like(hidden)
        left[:, 1:] = hidden[:, :-1]
        right[:, :-1] = hidden[:, 1:]
        return left, right

    def _event_slot_count(self, event_strength: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Reserve enough slots that predicted events cannot evict one another."""

        event_count = event_strength.size(1)
        active_count = (event_strength >= self.config.event_selection_floor).sum(dim=1)
        if not self.config.adaptive_event_capacity:
            return min(event_count, max(32, 4 * self.config.sparse_top_k)), active_count
        predicted_peak = int(active_count.detach().amax().cpu().item())
        reserved = self.config.event_slot_reserve_factor * predicted_peak
        event_slots = min(
            event_count,
            max(32, 4 * self.config.sparse_top_k, reserved),
        )
        return event_slots, active_count

    def _select_event_indices(
        self,
        event_probability: torch.Tensor,
        event_strength: torch.Tensor,
        event_slots: int,
    ) -> torch.Tensor:
        """Select event slots; subclasses may impose a stronger native contract."""

        del event_probability
        return event_strength.topk(k=event_slots, dim=1).indices

    def forward(
        self,
        state: Mapping[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        (
            role_states,
            routes,
            availability,
            routing_metrics,
            role_world_usage,
        ) = self._role_states(state, attention_mask)
        entity_logits = self.entity_head(role_states["entity"]).squeeze(-1)
        # Query markers are explicit raw symbols, not latent graph structure.
        # This position-independent semantic residual prevents a marker learned
        # at short horizons from becoming an absolute-position shortcut.
        query_role_logits = self.query_role_head(state["semantic_token_seed"])
        query_probability = torch.softmax(query_role_logits, dim=-1)

        relation_hidden = role_states["relation"]
        pointer_hidden = role_states["pointer"]
        left_field, right_field = self._oriented_neighbor_fields(pointer_hidden)
        event_relation_logits = self.event_relation_head(state["typed_payload_charge"])
        # Padding is outside the physical domain and must carry exactly zero
        # positive-relation charge.  Leaving the linear-head bias active on
        # padded cells let adaptive event selection populate spare slots with
        # non-events; those cells then shifted the per-example action gauge
        # even though every registered hard prediction remained unchanged.
        valid_event = attention_mask.unsqueeze(-1)
        event_relation_logits = torch.cat(
            (
                event_relation_logits[..., :1].masked_fill(~valid_event, 1.0e4),
                event_relation_logits[..., 1:].masked_fill(~valid_event, -1.0e4),
            ),
            dim=-1,
        )
        endpoint_key = self.event_endpoint_key(role_states["entity"])
        normalized_endpoint = F.normalize(role_states["entity"], dim=-1, eps=self.config.eps)
        source_coherence = torch.einsum(
            "beh,bih->bei",
            F.normalize(left_field, dim=-1, eps=self.config.eps),
            normalized_endpoint,
        )
        target_coherence = torch.einsum(
            "beh,bih->bei",
            F.normalize(right_field, dim=-1, eps=self.config.eps),
            normalized_endpoint,
        )
        coherence_gain = F.softplus(self.endpoint_coherence_gain)
        geodesic_gain = F.softplus(self.geodesic_binding_gain)
        base_minimum_action, base_minimum_world = state["world_action"].min(dim=1)
        pointer_role = TYPED_SLOT_NAMES.index("pointer")
        pointer_action = torch.einsum(
            "bw,bwij->bij", routes[:, pointer_role], state["world_action"]
        )
        source_action = torch.roll(pointer_action, shifts=1, dims=1)
        target_action = torch.roll(pointer_action, shifts=-1, dims=1)
        source_action[:, 0] = 30.0
        target_action[:, -1] = 30.0
        event_source_logits = (
            torch.einsum("beh,bih->bei", self.event_source_query(left_field), endpoint_key)
            / math.sqrt(float(self.config.hidden_dim))
            + coherence_gain * source_coherence
            - geodesic_gain * source_action
        )
        event_target_logits = (
            torch.einsum("beh,bih->bei", self.event_target_query(right_field), endpoint_key)
            / math.sqrt(float(self.config.hidden_dim))
            + coherence_gain * target_coherence
            - geodesic_gain * target_action
        )
        event_source_logits = event_source_logits + 2.0 * entity_logits[:, None, :]
        event_target_logits = event_target_logits + 2.0 * entity_logits[:, None, :]
        event_source_logits = event_source_logits.masked_fill(~attention_mask[:, None, :], -1.0e4)
        event_target_logits = event_target_logits.masked_fill(~attention_mask[:, None, :], -1.0e4)
        event_probability = torch.softmax(event_relation_logits, dim=-1)
        event_source_probability = torch.softmax(event_source_logits, dim=-1)
        event_target_probability = torch.softmax(event_target_logits, dim=-1)
        event_count = event_probability.size(1)
        event_strength = event_probability[..., 1:].sum(dim=-1)
        event_slots, predicted_event_count = self._event_slot_count(event_strength)
        valid_event_peak = int(attention_mask.sum(dim=1).amax().detach().cpu().item())
        if valid_event_peak <= 64:
            # Preserve the small-data contract without keying behavior to the
            # padded tensor width.  A compact 55-token example and the same
            # example padded to 255 cells must reserve the same 55 slots.
            event_slots = min(event_count, valid_event_peak)
            selected_event_indices = self._select_event_indices(
                event_probability,
                event_strength,
                event_slots,
            )
            endpoint_gather = selected_event_indices.unsqueeze(-1).expand(
                -1, -1, event_source_probability.size(-1)
            )
            relation_gather = selected_event_indices.unsqueeze(-1).expand(
                -1, -1, len(RELATION_NAMES) - 1
            )
            selected_event_source = torch.gather(
                event_source_probability, 1, endpoint_gather
            )
            selected_event_target = torch.gather(
                event_target_probability, 1, endpoint_gather
            )
            selected_event_relation = torch.gather(
                event_probability[..., 1:], 1, relation_gather
            )
        else:
            selected_event_indices = self._select_event_indices(
                event_probability,
                event_strength,
                event_slots,
            )
            endpoint_gather = selected_event_indices.unsqueeze(-1).expand(
                -1, -1, event_source_probability.size(-1)
            )
            relation_gather = selected_event_indices.unsqueeze(-1).expand(
                -1, -1, len(RELATION_NAMES) - 1
            )
            selected_event_source = torch.gather(event_source_probability, 1, endpoint_gather)
            selected_event_target = torch.gather(event_target_probability, 1, endpoint_gather)
            selected_event_relation = torch.gather(event_probability[..., 1:], 1, relation_gather)
        event_pair_probability = torch.einsum(
            "bki,bkr,bkj->bijr",
            selected_event_source,
            selected_event_relation,
            selected_event_target,
        ).clamp(max=1.0)
        event_pair_normalized = event_pair_probability / event_pair_probability.amax(
            dim=(1, 2, 3), keepdim=True
        ).clamp_min(self.config.eps)
        event_pair_mass = event_pair_probability.sum(dim=-1)
        event_pair_presence = event_pair_mass / event_pair_mass.amax(
            dim=(1, 2), keepdim=True
        ).clamp_min(self.config.eps)
        typed_operator_payload = event_pair_probability / event_pair_mass.unsqueeze(-1).clamp_min(
            self.config.eps
        )
        valid_pair = attention_mask[:, :, None] & attention_mask[:, None, :]
        diagonal = torch.eye(
            attention_mask.size(1), dtype=torch.bool, device=attention_mask.device
        ).unsqueeze(0)
        off_diagonal = valid_pair & ~diagonal

        # Predicted typed events act as a source term. They bend the authorized
        # world metrics before sparsity and the finite-speed cone are applied;
        # they never bypass those geometric constraints by directly OR-ing an
        # edge into the executable program.
        relation_role = TYPED_SLOT_NAMES.index("relation")
        relation_routes = routes[:, relation_role]
        coupling = torch.sigmoid(self.event_backreaction_gain) * state["event_backreaction_enabled"]
        if self.config.close_event_source_over_role_worlds:
            relation_world_drive = (
                self.role_world_contract[relation_role]
                & state["active_worlds"]
            ).to(event_pair_presence.dtype)
            relation_world_drive = relation_world_drive.unsqueeze(0).expand(
                event_pair_presence.size(0), -1
            )
        else:
            relation_world_drive = relation_routes
        event_stress = (
            coupling
            * relation_world_drive[:, :, None, None]
            * event_pair_presence[:, None]
            * state["active_worlds"].view(1, -1, 1, 1).to(event_pair_presence.dtype)
        )
        base_world_weight = state["world_edge_weight"]
        backreacted_world_weight = 1.0 - (1.0 - base_world_weight) * (1.0 - event_stress)
        program_world_weight = torch.where(
            state["event_backreaction_enabled"].bool(),
            backreacted_world_weight,
            base_world_weight,
        )
        program_world_weight = program_world_weight * off_diagonal[:, None].to(
            program_world_weight.dtype
        )
        token_count = program_world_weight.size(-1)
        sparse_k = min(self.config.sparse_top_k, max(1, token_count - 1))
        sparse_indices = (
            program_world_weight.masked_fill(~off_diagonal[:, None], -1.0)
            .topk(k=sparse_k, dim=-1)
            .indices
        )
        program_sparse_mask = torch.zeros_like(program_world_weight, dtype=torch.bool).scatter_(
            -1, sparse_indices, True
        )
        program_sparse_mask = program_sparse_mask & off_diagonal[:, None]
        program_world_length = -torch.log(program_world_weight.clamp_min(self.config.eps))
        program_world_length = program_world_length.masked_fill(~off_diagonal[:, None], 30.0)
        program_world_cone = program_world_length <= state["world_cone_budget"]
        program_world_action = (
            program_world_length
            + 0.75 * (1.0 - state["reconstructibility"][:, None])
            + 0.50 * (1.0 - state["stability"][:, None])
            + (~program_world_cone).to(program_world_length.dtype) * 8.0
        )
        transport_enabled = state["program_transport_enabled"].bool()
        program_world_mask = (
            program_sparse_mask
            & program_world_cone
            & state["active_worlds"].view(1, -1, 1, 1)
            & transport_enabled
        )
        minimum_action, minimum_world = program_world_action.min(dim=1)
        source = self.relation_source(relation_hidden)[:, :, None, :]
        target = self.relation_target(relation_hidden)[:, None, :, :]
        edge_evidence = program_world_weight.amax(dim=1)
        reconstructibility = state["reconstructibility"]
        geometry_features = torch.stack(
            (
                edge_evidence,
                torch.exp(-minimum_action.clamp(max=20.0)),
                reconstructibility,
            ),
            dim=-1,
        )
        relation_pair = torch.tanh(source + target + self.geometry_pair(geometry_features))
        pair_presence_residual = self.edge_presence_head(relation_pair).squeeze(-1)
        relation_presence_logit = (
            pair_presence_residual
            + 1.5 * torch.logit(edge_evidence.clamp(1.0e-4, 1.0 - 1.0e-4))
            + torch.logit(event_pair_presence.clamp(1.0e-4, 1.0 - 1.0e-4))
        )
        # Geometry decides whether a world-line is admissible. The relative
        # positive logits are solely the conserved typed payload transported
        # from a relation event, so Phi cannot silently change operator type.
        relation_logits = torch.cat(
            (
                -relation_presence_logit.unsqueeze(-1),
                relation_presence_logit.unsqueeze(-1)
                + typed_operator_payload.clamp_min(self.config.eps).log(),
            ),
            dim=-1,
        )
        relation_logits[..., 1:] = relation_logits[..., 1:].masked_fill(
            (~valid_pair | diagonal).unsqueeze(-1), -1.0e4
        )

        pointer_key = self.pointer_state(pointer_hidden)
        query_pointer_logits = torch.einsum(
            "bqh,bth->bqt", self.query_pointer_query(right_field), pointer_key
        ) / math.sqrt(float(self.config.hidden_dim))
        query_pointer_coherence = torch.einsum(
            "bqh,bth->bqt",
            F.normalize(right_field, dim=-1, eps=self.config.eps),
            normalized_endpoint,
        )
        query_action = torch.roll(pointer_action, shifts=-1, dims=1)
        query_action[:, -1] = 30.0
        query_pointer_logits = (
            query_pointer_logits
            + coherence_gain * query_pointer_coherence
            - geodesic_gain * query_action
            + 2.0 * entity_logits[:, None, :]
        )
        pointer_values = []
        for role_id in (1, 2, 4):
            weight = query_probability[..., role_id] * attention_mask.to(query_probability.dtype)
            pointer_values.append(
                torch.einsum("bq,bqt->bt", weight, query_pointer_logits)
                / weight.sum(dim=1, keepdim=True).clamp_min(self.config.eps)
            )
        pointer_logits = torch.stack(pointer_values, dim=1)
        pointer_logits = pointer_logits.masked_fill(~attention_mask[:, None, :], -1.0e4)

        boundary_hidden = role_states["boundary"]
        _, boundary_right_field = self._oriented_neighbor_fields(boundary_hidden)
        boundary_token_logits = self.boundary_token_head(boundary_right_field)
        boundary_contexts = [
            self._role_context(boundary_hidden, query_probability, role_id, attention_mask)
            for role_id in (3, 5)
        ]
        boundary_values = []
        for role_id, context in zip((3, 5), boundary_contexts, strict=True):
            weight = query_probability[..., role_id] * attention_mask.to(query_probability.dtype)
            token_value = torch.einsum("bt,btr->br", weight, boundary_token_logits) / weight.sum(
                dim=1, keepdim=True
            ).clamp_min(self.config.eps)
            boundary_values.append(token_value + self.boundary_head(context))
        boundary_logits = torch.stack(boundary_values, dim=1)
        program_mask = program_world_mask.any(dim=1) & off_diagonal
        sparse_edge_evidence = (
            program_world_weight * program_sparse_mask.to(program_world_weight.dtype)
        ).amax(dim=1)
        executable_pair = (relation_logits.argmax(dim=-1) > 0) & program_mask
        executable_world_mask = program_world_mask & executable_pair[:, None]
        program_edge_mass = (
            program_world_weight * executable_world_mask.to(program_world_weight.dtype)
        ).sum(dim=(-1, -2))
        program_edge_mass = program_edge_mass / executable_world_mask.sum(dim=(-1, -2)).clamp_min(
            1.0
        )
        program_usage = program_edge_mass.mean(dim=0)
        program_usage_probability = program_usage / program_usage.sum().clamp_min(self.config.eps)
        program_world_usage_entropy = -(
            program_usage_probability.clamp_min(self.config.eps)
            * program_usage_probability.clamp_min(self.config.eps).log()
        ).sum() / math.log(float(self.config.n_worlds))
        flattened_program_geometry = F.normalize(
            program_world_weight.flatten(start_dim=2), dim=-1, eps=self.config.eps
        )
        program_cosine = torch.einsum(
            "bwd,bvd->bwv", flattened_program_geometry, flattened_program_geometry
        )
        triangle = torch.triu(
            torch.ones(
                self.config.n_worlds,
                self.config.n_worlds,
                dtype=torch.bool,
                device=program_world_weight.device,
            ),
            diagonal=1,
        )
        program_world_geometry_diversity = (1.0 - program_cosine[:, triangle].mean()).clamp(
            0.0, 1.0
        )
        complete = availability.all()
        if not bool(complete):
            relation_logits = relation_logits.clone()
            relation_logits[..., :] = -1.0e4
            relation_logits[..., 0] = 1.0e4

        return {
            "entity_logits": entity_logits,
            "query_role_logits": query_role_logits,
            "pointer_logits": pointer_logits,
            "query_pointer_logits": query_pointer_logits,
            "source_logits": pointer_logits[:, 0],
            "candidate_logits": pointer_logits[:, 1:],
            "boundary_logits": boundary_logits,
            "relation_logits": relation_logits,
            "event_relation_logits": event_relation_logits,
            "event_source_logits": event_source_logits,
            "event_target_logits": event_target_logits,
            "event_pair_probability": event_pair_probability,
            "event_pair_normalized": event_pair_normalized,
            "event_pair_presence": event_pair_presence,
            "typed_operator_payload": typed_operator_payload,
            "relation_presence_logit": relation_presence_logit,
            "selected_event_indices": selected_event_indices,
            "predicted_event_count": predicted_event_count,
            "selected_event_capacity": relation_logits.new_tensor(event_slots),
            "event_selection_overflow": (
                predicted_event_count - event_slots
            ).clamp_min(0),
            "boundary_token_logits": boundary_token_logits,
            "minimum_action": minimum_action,
            "minimum_action_world": minimum_world,
            "base_minimum_action": base_minimum_action,
            "base_minimum_action_world": base_minimum_world,
            "pointer_geodesic_action": pointer_action,
            "event_backreaction_strength": coupling,
            "event_source_role_closure_enabled": relation_logits.new_tensor(
                float(self.config.close_event_source_over_role_worlds)
            ),
            "event_metric_backreaction_mean": (program_world_weight - base_world_weight)
            .abs()
            .mean(),
            "program_world_edge_weight": program_world_weight,
            "program_world_edge_length": program_world_length,
            "program_world_action": program_world_action,
            "program_world_sparse_mask": program_sparse_mask,
            "program_world_cone": program_world_cone,
            "program_world_mask": program_world_mask,
            "program_world_usage": program_usage_probability,
            "program_world_usage_entropy": program_world_usage_entropy,
            "program_world_geometry_diversity": program_world_geometry_diversity,
            "edge_evidence": edge_evidence,
            "sparse_edge_evidence": sparse_edge_evidence,
            "program_mask": program_mask,
            "role_routes": routes,
            "role_world_usage": role_world_usage,
            "role_route_entropy": routing_metrics[0],
            "functional_world_usage_min": routing_metrics[1],
            "functional_world_usage_max": routing_metrics[2],
            "functional_world_usage_entropy": routing_metrics[3],
            "role_availability": availability,
            "role_contract_complete": relation_logits.new_tensor(float(bool(complete))),
            "geodesic_binding_strength": geodesic_gain,
        }

def physics_native_training_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: ERGT35Batch,
    *,
    teacher_weight: float,
    answer_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train fields and program slots while fading all graph-sidecar teaching."""

    supervision = batch.supervision
    valid = batch.attention_mask
    entity_targets = supervision.entity_labels[valid]
    entity_positive_weight = (
        (entity_targets.numel() - entity_targets.sum()) / entity_targets.sum().clamp_min(1.0)
    ).clamp(1.0, 8.0)
    entity_loss = F.binary_cross_entropy_with_logits(
        outputs["entity_logits"][valid],
        entity_targets,
        pos_weight=entity_positive_weight,
    )
    query_class_weight = outputs["query_role_logits"].new_tensor((0.15, 1.0, 1.0, 1.0, 1.0, 1.0))
    query_loss = F.cross_entropy(
        outputs["query_role_logits"][valid],
        supervision.query_role_labels[valid],
        weight=query_class_weight,
    )
    pair_mask = supervision.entity_pair_mask & valid[:, :, None] & valid[:, None, :]
    relation_class_weight = outputs["relation_logits"].new_tensor((0.15, 1.0, 1.0, 1.0))
    relation_loss = F.cross_entropy(
        outputs["relation_logits"][pair_mask],
        supervision.relation_labels[pair_mask],
        weight=relation_class_weight,
    )
    event_relation_loss = F.cross_entropy(
        outputs["event_relation_logits"][valid],
        supervision.relation_anchor_labels[valid],
        weight=relation_class_weight,
    )
    event_mask = supervision.relation_event_mask & valid
    if bool(event_mask.any()):
        event_source_loss = F.cross_entropy(
            outputs["event_source_logits"][event_mask],
            supervision.event_source_positions[event_mask],
        )
        event_target_loss = F.cross_entropy(
            outputs["event_target_logits"][event_mask],
            supervision.event_target_positions[event_mask],
        )
    else:
        event_source_loss = relation_loss.new_tensor(0.0)
        event_target_loss = relation_loss.new_tensor(0.0)
    pointer_training_logits = outputs.get("pointer_training_logits", outputs["pointer_logits"])
    source_loss = F.cross_entropy(
        pointer_training_logits[:, 0], supervision.source_positions
    )
    candidate_loss = 0.5 * (
        F.cross_entropy(
            pointer_training_logits[:, 1], supervision.candidate_positions[:, 0]
        )
        + F.cross_entropy(
            pointer_training_logits[:, 2], supervision.candidate_positions[:, 1]
        )
    )
    query_pointer_mask = (
        (supervision.query_role_labels == 1)
        | (supervision.query_role_labels == 2)
        | (supervision.query_role_labels == 4)
    )
    query_pointer_targets = torch.zeros_like(supervision.query_role_labels)
    query_pointer_targets = torch.where(
        supervision.query_role_labels == 1,
        supervision.source_positions[:, None].expand_as(query_pointer_targets),
        query_pointer_targets,
    )
    query_pointer_targets = torch.where(
        supervision.query_role_labels == 2,
        supervision.candidate_positions[:, 0, None].expand_as(query_pointer_targets),
        query_pointer_targets,
    )
    query_pointer_targets = torch.where(
        supervision.query_role_labels == 4,
        supervision.candidate_positions[:, 1, None].expand_as(query_pointer_targets),
        query_pointer_targets,
    )
    query_pointer_loss = F.cross_entropy(
        outputs["query_pointer_logits"][query_pointer_mask],
        query_pointer_targets[query_pointer_mask],
    )
    boundary_loss = 0.5 * (
        F.cross_entropy(outputs["boundary_logits"][:, 0], supervision.boundary_labels[:, 0])
        + F.cross_entropy(outputs["boundary_logits"][:, 1], supervision.boundary_labels[:, 1])
    )
    boundary_token_mask = supervision.query_boundary_value_labels >= 0
    boundary_token_loss = F.cross_entropy(
        outputs["boundary_token_logits"][boundary_token_mask],
        supervision.query_boundary_value_labels[boundary_token_mask],
    )
    edge_target = (supervision.relation_labels[pair_mask] > 0).to(outputs["edge_evidence"].dtype)
    edge_positive_weight = (
        (edge_target.numel() - edge_target.sum()) / edge_target.sum().clamp_min(1.0)
    ).clamp(1.0, 8.0)
    geometry_edge_loss = F.binary_cross_entropy(
        outputs["edge_evidence"][pair_mask].clamp(1.0e-5, 1.0 - 1.0e-5),
        edge_target,
        weight=torch.where(
            edge_target > 0,
            edge_positive_weight.expand_as(edge_target),
            torch.ones_like(edge_target),
        ),
    )
    answer_loss = F.cross_entropy(outputs["native_answer_logits"], supervision.answer_labels)
    structured = (
        entity_loss
        + query_loss
        + relation_loss
        + event_relation_loss
        + event_source_loss
        + event_target_loss
        + source_loss
        + candidate_loss
        + query_pointer_loss
        + boundary_loss
        + boundary_token_loss
        + geometry_edge_loss
    ) / 12.0

    relation_probability = torch.softmax(outputs["relation_logits"], dim=-1)[..., 1:].sum(-1)
    true_edge = supervision.relation_labels > 0
    false_edge = pair_mask & ~true_edge
    true_action = outputs["minimum_action"][true_edge]
    false_action = outputs["minimum_action"][false_edge]
    if true_action.numel() and false_action.numel():
        action_margin_loss = F.relu(
            true_action.mean() - false_action.mean() + true_action.new_tensor(0.35)
        )
    else:
        action_margin_loss = answer_loss.new_tensor(0.0)
    conservation_loss = (
        (
            (outputs["world_transport"].sum(dim=-1) - 1.0).abs()
            * valid[:, None, :].to(outputs["world_transport"].dtype)
        ).sum()
        / valid.sum().clamp_min(1)
        / outputs["world_transport"].size(1)
    )
    world_entropy_loss = F.relu(
        outputs["world_usage_entropy"].new_tensor(0.45) - outputs["world_usage_entropy"]
    ).square()
    role_routing_loss = (
        F.relu(
            outputs["role_route_entropy"].new_tensor(0.45) - outputs["role_route_entropy"]
        ).square()
        + F.relu(
            outputs["functional_world_usage_max"]
            - outputs["functional_world_usage_max"].new_tensor(0.45)
        ).square()
        + F.relu(
            outputs["functional_world_usage_min"].new_tensor(0.015)
            - outputs["functional_world_usage_min"]
        ).square()
    ) / 3.0
    geometry_diversity_loss = F.relu(
        outputs["world_geometry_diversity"].new_tensor(0.12) - outputs["world_geometry_diversity"]
    ).square()
    physics = (
        outputs["bulk_boundary_loss"]
        + action_margin_loss
        + conservation_loss
        + world_entropy_loss
        + role_routing_loss
        + geometry_diversity_loss
    ) / 6.0
    total = (
        float(answer_weight) * answer_loss
        + 2.0 * float(teacher_weight) * structured
        + 0.5 * physics
    )
    parts = {
        "total_loss": total,
        "answer_loss": answer_loss,
        "answer_weight": total.new_tensor(float(answer_weight)),
        "structured_teacher_loss": structured,
        "entity_loss": entity_loss,
        "query_loss": query_loss,
        "relation_loss": relation_loss,
        "event_relation_loss": event_relation_loss,
        "event_source_loss": event_source_loss,
        "event_target_loss": event_target_loss,
        "source_loss": source_loss,
        "candidate_loss": candidate_loss,
        "query_pointer_loss": query_pointer_loss,
        "boundary_loss": boundary_loss,
        "boundary_token_loss": boundary_token_loss,
        "geometry_edge_loss": geometry_edge_loss,
        "bulk_boundary_loss": outputs["bulk_boundary_loss"],
        "action_margin_loss": action_margin_loss,
        "conservation_loss": conservation_loss,
        "world_entropy_loss": world_entropy_loss,
        "role_routing_loss": role_routing_loss,
        "geometry_diversity_loss": geometry_diversity_loss,
        "relation_probability_mean": relation_probability[pair_mask].mean(),
        "teacher_weight": total.new_tensor(float(teacher_weight)),
    }
    return total, parts


__all__ = [
    "ERGT42Config",
    "ERGT42Error",
    "INTERVENTIONS",
    "PhysicsNativeSubstrate",
    "RolePreservingGeometricFieldInducer",
    "SCHEMA_VERSION",
    "physics_native_training_loss",
]
