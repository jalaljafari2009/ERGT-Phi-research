"""Sparse min-plus closure on the ERGT token-by-world product manifold."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint


class ProductManifoldError(ValueError):
    """Raised when a product-manifold closure contract is malformed."""


@dataclass(frozen=True)
class ProductManifoldClosure:
    """Event endpoint measurements after sparse multi-step geodesic closure."""

    soft_action: torch.Tensor
    hard_action: torch.Tensor
    continuous_hard_action: torch.Tensor
    soft_gate: torch.Tensor
    hard_gate: torch.Tensor
    path_hops: torch.Tensor
    graph_edge_count: torch.Tensor
    steps_executed: int
    fixed_point_reached: bool


def _active_world_batch(active_worlds: torch.Tensor, batch_size: int) -> torch.Tensor:
    if active_worlds.dim() == 1:
        return active_worlds.unsqueeze(0).expand(batch_size, -1)
    if active_worlds.dim() == 2 and active_worlds.size(0) == batch_size:
        return active_worlds
    raise ProductManifoldError("active_worlds must have shape [W] or [B, W]")


def _authorized_world_batch(
    authorized_worlds: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if authorized_worlds.dim() == 1:
        return authorized_worlds.unsqueeze(0).expand(batch_size, -1)
    if authorized_worlds.dim() == 2 and authorized_worlds.size(0) == batch_size:
        return authorized_worlds
    raise ProductManifoldError("authorized_worlds must have shape [W] or [B, W]")


def _backbone_offsets(token_count: int, levels: int) -> tuple[int, ...]:
    offsets: list[int] = []
    for level in range(max(0, int(levels))):
        offset = 1 << level
        if offset >= token_count:
            break
        offsets.extend((-offset, offset))
    return tuple(offsets)


def _sparse_product_edges(
    *,
    world_action: torch.Tensor,
    world_weight: torch.Tensor,
    world_sparse_mask: torch.Tensor,
    world_cone: torch.Tensor,
    attention_mask: torch.Tensor,
    active_authorized_worlds: torch.Tensor,
    sparse_top_k: int,
    backbone_levels: int,
    include_backbone: bool,
    ignore_cone: bool,
    infinity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target indices, costs, capacities, and validity for each world node."""

    batch, worlds, token_count, target_count = world_action.shape
    if token_count != target_count:
        raise ProductManifoldError("world tensors must be square over tokens")
    expected = (batch, worlds, token_count, token_count)
    if (
        world_weight.shape != expected
        or world_sparse_mask.shape != expected
        or world_cone.shape != expected
    ):
        raise ProductManifoldError("world action, weight, sparse mask, and cone must align")
    if attention_mask.shape != (batch, token_count):
        raise ProductManifoldError("attention_mask must align with world token axes")

    valid_pair = attention_mask[:, None, :, None] & attention_mask[:, None, None, :]
    sparse_valid = world_sparse_mask.bool() & valid_pair
    if not ignore_cone:
        sparse_valid = sparse_valid & world_cone.bool()
    sparse_valid = sparse_valid & active_authorized_worlds[:, :, None, None]

    sparse_k = min(max(1, int(sparse_top_k)), token_count)
    masked_action = world_action.masked_fill(~sparse_valid, float(infinity))
    sparse_cost, sparse_target = masked_action.topk(
        k=sparse_k,
        dim=-1,
        largest=False,
    )
    sparse_is_valid = sparse_cost < 0.5 * float(infinity)
    sparse_capacity = torch.gather(world_weight, -1, sparse_target).clamp(0.0, 1.0)
    sparse_capacity = sparse_capacity * sparse_is_valid.to(sparse_capacity.dtype)

    targets = [sparse_target]
    costs = [sparse_cost]
    capacities = [sparse_capacity]
    validities = [sparse_is_valid]

    offsets = _backbone_offsets(token_count, backbone_levels) if include_backbone else ()
    if offsets:
        source = torch.arange(token_count, device=world_action.device)[:, None]
        raw_target = source + torch.tensor(offsets, device=world_action.device)[None, :]
        in_range = (raw_target >= 0) & (raw_target < token_count)
        backbone_target = raw_target.clamp(0, token_count - 1)
        backbone_target = backbone_target.view(1, 1, token_count, -1).expand(batch, worlds, -1, -1)
        backbone_cost = torch.gather(world_action, -1, backbone_target)
        backbone_capacity = torch.gather(world_weight, -1, backbone_target).clamp(0.0, 1.0)
        target_valid = torch.gather(
            attention_mask[:, None, :].expand(-1, worlds, -1),
            2,
            backbone_target.reshape(batch, worlds, -1),
        ).reshape_as(backbone_target)
        backbone_valid = (
            in_range.view(1, 1, token_count, -1)
            & attention_mask[:, None, :, None]
            & target_valid
            & active_authorized_worlds[:, :, None, None]
        )
        if not ignore_cone:
            backbone_valid = backbone_valid & torch.gather(world_cone.bool(), -1, backbone_target)
        backbone_cost = backbone_cost.masked_fill(~backbone_valid, float(infinity))
        backbone_capacity = backbone_capacity * backbone_valid.to(backbone_capacity.dtype)
        targets.append(backbone_target)
        costs.append(backbone_cost)
        capacities.append(backbone_capacity)
        validities.append(backbone_valid)

    return (
        torch.cat(targets, dim=-1),
        torch.cat(costs, dim=-1),
        torch.cat(capacities, dim=-1),
        torch.cat(validities, dim=-1),
    )


def _scatter_reduce_nodes(
    values: torch.Tensor,
    targets: torch.Tensor,
    *,
    token_count: int,
    reduce: str,
    fill_value: float,
) -> torch.Tensor:
    batch, events = values.shape[:2]
    worlds = targets.size(1)
    world_offset = torch.arange(worlds, device=targets.device).view(1, worlds, 1, 1) * token_count
    product_targets = (targets + world_offset).reshape(batch, -1)
    flattened_values = values.reshape(batch, events, -1)
    output = values.new_full(
        (batch, events, worlds * token_count),
        float(fill_value),
    )
    output.scatter_reduce_(
        2,
        product_targets[:, None, :].expand(-1, events, -1),
        flattened_values,
        reduce=reduce,
        include_self=True,
    )
    return output.view(batch, events, worlds, token_count)


def product_manifold_geodesic_closure(
    *,
    world_action: torch.Tensor,
    world_weight: torch.Tensor,
    world_sparse_mask: torch.Tensor,
    world_cone: torch.Tensor,
    attention_mask: torch.Tensor,
    active_worlds: torch.Tensor,
    authorized_worlds: torch.Tensor,
    role_route: torch.Tensor,
    event_source: torch.Tensor,
    event_target: torch.Tensor,
    max_steps: int,
    sparse_top_k: int,
    backbone_levels: int,
    world_transition_cost: float,
    temperature: float,
    include_backbone: bool = True,
    allow_world_transitions: bool = True,
    ignore_cone: bool = False,
    stop_at_fixed_point: bool = True,
    fixed_point_tolerance: float = 1.0e-7,
    eps: float = 1.0e-6,
) -> ProductManifoldClosure:
    """Solve event endpoint paths on a sparse token-by-world fibre product.

    Hard support is transitive reachability, not direct endpoint adjacency.
    Action evolves in the min-plus semiring, while soft support uses the
    max-min widest-path semiring so support does not decay merely with length.
    """

    if world_action.dim() != 4:
        raise ProductManifoldError("world_action must have shape [B, W, T, T]")
    batch, worlds, token_count, _ = world_action.shape
    if event_source.shape != event_target.shape or event_source.dim() != 2:
        raise ProductManifoldError("event endpoints must have shape [B, E]")
    if event_source.size(0) != batch:
        raise ProductManifoldError("event endpoint batch must match world fields")
    if role_route.shape != (batch, worlds):
        raise ProductManifoldError("role_route must have shape [B, W]")
    if max_steps <= 0 or world_transition_cost < 0.0 or temperature <= 0.0:
        raise ProductManifoldError("closure steps, transition cost, and temperature are invalid")
    if fixed_point_tolerance < 0.0:
        raise ProductManifoldError("fixed-point tolerance must be non-negative")

    active = _active_world_batch(active_worlds.bool(), batch)
    authorized = _authorized_world_batch(authorized_worlds.bool(), batch)
    active_authorized = active & authorized
    # A world ablation may remove every lens authorized for this role.  That
    # is a valid fail-closed state: distances remain infinite and no answer is
    # supported.  Raising here would make the causal intervention itself
    # unevaluable.

    infinity = 1.0e6
    edge_target, edge_cost, edge_capacity, edge_valid = _sparse_product_edges(
        world_action=world_action,
        world_weight=world_weight,
        world_sparse_mask=world_sparse_mask,
        world_cone=world_cone,
        attention_mask=attention_mask.bool(),
        active_authorized_worlds=active_authorized,
        sparse_top_k=sparse_top_k,
        backbone_levels=backbone_levels,
        include_backbone=include_backbone,
        ignore_cone=ignore_cone,
        infinity=infinity,
    )
    edge_cost = edge_cost.masked_fill(~edge_valid, infinity)
    edge_capacity = edge_capacity * edge_valid.to(edge_capacity.dtype)

    events = event_source.size(1)
    event_source = event_source.clamp(0, token_count - 1)
    event_target = event_target.clamp(0, token_count - 1)
    source_index = event_source[:, :, None, None].expand(-1, -1, worlds, 1)
    initial_distance = world_action.new_full(
        (batch, events, worlds, token_count),
        infinity,
    )
    initial_distance.scatter_(
        3,
        source_index,
        torch.where(
            active_authorized[:, None, :, None],
            torch.zeros_like(source_index, dtype=world_action.dtype),
            torch.full_like(source_index, infinity, dtype=world_action.dtype),
        ),
    )

    normalized_route = role_route * active_authorized.to(role_route.dtype)
    normalized_route = normalized_route / normalized_route.sum(dim=-1, keepdim=True).clamp_min(
        float(eps)
    )
    reach = world_action.new_zeros((batch, events, worlds, token_count))
    reach.scatter_(
        3,
        source_index,
        normalized_route[:, None, :, None].expand(-1, events, -1, -1),
    )
    distance = initial_distance
    path_hops = torch.full(
        (batch, events),
        -1,
        dtype=torch.long,
        device=world_action.device,
    )

    transition_cost = world_action.new_full((worlds, worlds), float(world_transition_cost))
    transition_cost.fill_diagonal_(0.0)
    transition_allowed = active_authorized[:, :, None] & active_authorized[:, None, :]
    eye = torch.eye(worlds, dtype=torch.bool, device=world_action.device)
    transition_allowed = transition_allowed & ~eye[None]
    transition_capacity = torch.exp(-transition_cost).clamp(0.0, 1.0)

    def relax_step(
        current_distance: torch.Tensor,
        current_reach: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_distance = current_distance.unsqueeze(-1) + edge_cost[:, None]
        incoming_distance = _scatter_reduce_nodes(
            edge_distance,
            edge_target,
            token_count=token_count,
            reduce="amin",
            fill_value=infinity,
        )
        next_distance = torch.minimum(current_distance, incoming_distance)

        edge_reach = torch.minimum(current_reach.unsqueeze(-1), edge_capacity[:, None])
        incoming_reach = _scatter_reduce_nodes(
            edge_reach,
            edge_target,
            token_count=token_count,
            reduce="amax",
            fill_value=0.0,
        )
        next_reach = torch.maximum(current_reach, incoming_reach)

        if allow_world_transitions:
            transition_distance = next_distance.unsqueeze(3) + transition_cost.view(
                1, 1, worlds, worlds, 1
            )
            transition_distance = transition_distance.masked_fill(
                ~transition_allowed[:, None, :, :, None],
                infinity,
            )
            next_distance = torch.minimum(
                next_distance,
                transition_distance.amin(dim=2),
            )
            transition_reach = torch.minimum(
                next_reach.unsqueeze(3),
                transition_capacity.view(1, 1, worlds, worlds, 1),
            )
            transition_reach = transition_reach.masked_fill(
                ~transition_allowed[:, None, :, :, None],
                0.0,
            )
            next_reach = torch.maximum(
                next_reach,
                transition_reach.amax(dim=2),
            )

        return next_distance, next_reach

    checkpoint_steps = torch.is_grad_enabled() and (
        world_action.requires_grad or world_weight.requires_grad or role_route.requires_grad
    )
    steps_executed = 0
    fixed_point_reached = False
    for step in range(int(max_steps)):
        if checkpoint_steps:
            next_distance, next_reach = checkpoint(
                relax_step,
                distance,
                reach,
                use_reentrant=False,
            )
        else:
            next_distance, next_reach = relax_step(distance, reach)
        if stop_at_fixed_point and not checkpoint_steps:
            distance_improved = bool(
                (next_distance < distance - float(fixed_point_tolerance)).any().item()
            )
            reach_improved = bool((next_reach > reach + float(fixed_point_tolerance)).any().item())
            fixed_point_reached = not (distance_improved or reach_improved)
        distance, reach = next_distance, next_reach
        steps_executed = step + 1
        target_index = event_target[:, :, None, None].expand(-1, -1, worlds, 1)
        reached = torch.gather(distance, 3, target_index).squeeze(-1).amin(dim=-1)
        newly_reached = (path_hops < 0) & (reached < 0.5 * infinity)
        path_hops = torch.where(
            newly_reached,
            torch.full_like(path_hops, step + 1),
            path_hops,
        )
        if fixed_point_reached:
            break

    target_index = event_target[:, :, None, None].expand(-1, -1, worlds, 1)
    target_distance = torch.gather(distance, 3, target_index).squeeze(-1)
    target_reach = torch.gather(reach, 3, target_index).squeeze(-1)
    hard_action = target_distance.amin(dim=-1)
    hard_gate = hard_action < 0.5 * infinity
    route_log = normalized_route[:, None, :].clamp_min(float(eps)).log()
    soft_action = -float(temperature) * torch.logsumexp(
        -target_distance.clamp(max=100.0) / float(temperature) + route_log,
        dim=-1,
    )
    continuous_hard_action = soft_action + (hard_action - soft_action).detach()
    soft_gate = target_reach.amax(dim=-1)
    path_hops = torch.where(
        hard_gate,
        path_hops,
        torch.full_like(path_hops, int(max_steps) + 1),
    )
    graph_edge_count = edge_valid.sum(dim=(-1, -2, -3)).to(world_action.dtype)
    return ProductManifoldClosure(
        soft_action=soft_action,
        hard_action=hard_action,
        continuous_hard_action=continuous_hard_action,
        soft_gate=soft_gate,
        hard_gate=hard_gate,
        path_hops=path_hops,
        graph_edge_count=graph_edge_count,
        steps_executed=steps_executed,
        fixed_point_reached=fixed_point_reached,
    )


def analytic_product_manifold_oracle_audit(
    *,
    path_hops: int = 64,
    world_transition_cost: float = 0.15,
) -> dict[str, float | bool]:
    """Prove long closure, transition necessity, and cone rejection without learning."""

    if path_hops < 4:
        raise ProductManifoldError("oracle audit requires at least four path hops")
    token_count = path_hops + 1
    worlds = 2
    action = torch.ones((1, worlds, token_count, token_count))
    weight = torch.zeros_like(action)
    sparse = torch.zeros_like(action, dtype=torch.bool)
    cone = torch.zeros_like(action, dtype=torch.bool)
    midpoint = path_hops // 2
    for source in range(midpoint):
        sparse[0, 0, source, source + 1] = True
        cone[0, 0, source, source + 1] = True
        weight[0, 0, source, source + 1] = 1.0
    for source in range(midpoint, path_hops):
        sparse[0, 1, source, source + 1] = True
        cone[0, 1, source, source + 1] = True
        weight[0, 1, source, source + 1] = 1.0
    common = {
        "world_action": action,
        "world_weight": weight,
        "world_sparse_mask": sparse,
        "attention_mask": torch.ones((1, token_count), dtype=torch.bool),
        "active_worlds": torch.ones(worlds, dtype=torch.bool),
        "authorized_worlds": torch.ones(worlds, dtype=torch.bool),
        "role_route": torch.full((1, worlds), 0.5),
        "event_source": torch.tensor([[0]]),
        "event_target": torch.tensor([[path_hops]]),
        "max_steps": path_hops,
        "sparse_top_k": 2,
        "backbone_levels": 0,
        "world_transition_cost": world_transition_cost,
        "temperature": 0.25,
        "include_backbone": False,
    }
    full = product_manifold_geodesic_closure(world_cone=cone, **common)
    no_transition = product_manifold_geodesic_closure(
        world_cone=cone,
        allow_world_transitions=False,
        **common,
    )
    blocked_cone = cone.clone()
    blocked_cone[0, 1, midpoint, midpoint + 1] = False
    blocked = product_manifold_geodesic_closure(world_cone=blocked_cone, **common)
    expected_action = float(path_hops) + float(world_transition_cost)
    return {
        "long_path_closed": bool(full.hard_gate.item()),
        "long_path_hops": float(full.path_hops.item()),
        "long_path_action": float(full.hard_action.item()),
        "long_path_action_exact": abs(float(full.hard_action.item()) - expected_action) < 1.0e-5,
        "world_transition_necessary": not bool(no_transition.hard_gate.item()),
        "cone_break_rejected": not bool(blocked.hard_gate.item()),
        "contract_pass": bool(
            full.hard_gate.item()
            and abs(float(full.hard_action.item()) - expected_action) < 1.0e-5
            and not no_transition.hard_gate.item()
            and not blocked.hard_gate.item()
        ),
    }


__all__ = [
    "ProductManifoldClosure",
    "ProductManifoldError",
    "analytic_product_manifold_oracle_audit",
    "product_manifold_geodesic_closure",
]
