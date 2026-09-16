"""Matched-topology data for native geometric boundary adjudication.

Every counterfactual pair has the same nodes, directed topology, typed
operators, hop count, and token count.  Both candidates are topologically and
operator-wise reachable.  The answer changes only when physical edge or
terminal conditions are exchanged between the two paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Final

import torch

from .data_schema import (
    ANSWER_TO_ID,
    RELATION_NAMES,
    RELATION_SURFACES,
    DynamicTypedEdge,
    ERGT35Batch,
    ERGT35Example,
    TrainOnlyTokenizer,
    apply_relation,
    collate_examples,
)

SCHEMA_VERSION: Final = "ergt43-matched-topology-data-v4"
IDENTITY_FIBRE_BITS: Final = 64
SCENARIOS: Final[tuple[str, ...]] = (
    "geodesic_action",
    "finite_speed_cone",
    "boundary_deficit",
    "payload_transport",
    "terminal_mass",
    "memory_source_condition",
    "combined_long_horizon",
)


class ERGT43DataError(ValueError):
    """Raised when a matched-topology or counterfactual contract is broken."""


@lru_cache(maxsize=131_072)
def raw_token_fingerprint_id(token: str) -> int:
    value = int.from_bytes(
        hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(),
        byteorder="big",
        signed=False,
    )
    return value if value < (1 << 63) else value - (1 << 64)


@dataclass(frozen=True)
class RawTokenInputContract:
    known_raw_token_ids: tuple[int, ...]
    known_semantic_token_ids: tuple[int, ...]
    semantic_pad_id: int
    semantic_unk_id: int
    semantic_vocabulary_size: int
    semantic_known_token_count: int
    semantic_hash_bucket_count: int

    def __post_init__(self) -> None:
        if len(self.known_raw_token_ids) != len(self.known_semantic_token_ids):
            raise ERGT43DataError("raw-token and semantic lookup tables must align")
        if tuple(sorted(self.known_raw_token_ids)) != self.known_raw_token_ids:
            raise ERGT43DataError("raw-token lookup keys must be sorted")
        if len(set(self.known_raw_token_ids)) != len(self.known_raw_token_ids):
            raise ERGT43DataError("raw-token lookup contains a 64-bit collision")
        if self.semantic_vocabulary_size != (
            self.semantic_known_token_count + self.semantic_hash_bucket_count
        ):
            raise ERGT43DataError("semantic vocabulary size does not match its contract")

    @classmethod
    def from_tokenizer(cls, tokenizer: TrainOnlyTokenizer) -> RawTokenInputContract:
        entries = sorted(
            (raw_token_fingerprint_id(token), int(token_id))
            for token, token_id in tokenizer.token_to_id.items()
        )
        return cls(
            known_raw_token_ids=tuple(raw_id for raw_id, _ in entries),
            known_semantic_token_ids=tuple(token_id for _, token_id in entries),
            semantic_pad_id=tokenizer.pad_id,
            semantic_unk_id=tokenizer.unk_id,
            semantic_vocabulary_size=tokenizer.vocab_size,
            semantic_known_token_count=len(tokenizer.token_to_id),
            semantic_hash_bucket_count=tokenizer.hash_bucket_count,
        )

    def to_json_record(self) -> dict[str, Any]:
        return {
            "type": "raw_blake2b64_to_internal_train_only_semantics",
            "known_raw_token_ids": list(self.known_raw_token_ids),
            "known_semantic_token_ids": list(self.known_semantic_token_ids),
            "semantic_pad_id": self.semantic_pad_id,
            "semantic_unk_id": self.semantic_unk_id,
            "semantic_vocabulary_size": self.semantic_vocabulary_size,
            "semantic_known_token_count": self.semantic_known_token_count,
            "semantic_hash_bucket_count": self.semantic_hash_bucket_count,
        }


@dataclass(frozen=True)
class PhysicalEdgeCondition:
    source_position: int
    target_position: int
    relation_id: int
    event_anchor_position: int
    action_cost: float
    cone_admissible: bool
    transmission: float
    boundary_deficit: float
    terminal_capacity: float

    def __post_init__(self) -> None:
        if self.source_position < 0 or self.target_position < 0:
            raise ERGT43DataError("physical edge endpoints must be non-negative")
        if self.relation_id <= 0 or self.relation_id >= len(RELATION_NAMES):
            raise ERGT43DataError("physical edge requires a registered typed operator")
        if self.event_anchor_position < 0:
            raise ERGT43DataError("physical event anchor must be non-negative")
        if self.action_cost <= 0.0:
            raise ERGT43DataError("action cost must be positive")
        if not 0.0 < self.transmission <= 1.0:
            raise ERGT43DataError("transmission must be in (0, 1]")
        if self.boundary_deficit < 0.0:
            raise ERGT43DataError("boundary deficit must be non-negative")
        if not 0.0 < self.terminal_capacity <= 1.0:
            raise ERGT43DataError("terminal capacity must be in (0, 1]")


@dataclass(frozen=True)
class MatchedTopologyExample:
    base: ERGT35Example
    physical_edges: tuple[PhysicalEdgeCondition, ...]
    pair_id: str
    scenario: str
    counterfactual_variant: int
    topology_sha256: str
    operator_sha256: str

    @property
    def tokens(self) -> tuple[str, ...]:
        return self.base.tokens

    @property
    def example_id(self) -> str:
        return self.base.example_id

    @property
    def answer_id(self) -> int:
        return self.base.answer_id

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ERGT43DataError(f"unregistered matched-topology scenario: {self.scenario}")
        if self.counterfactual_variant not in (0, 1):
            raise ERGT43DataError("counterfactual variant must be 0 or 1")
        if len(self.physical_edges) != len(self.base.edges):
            raise ERGT43DataError("each typed edge requires one physical condition")
        typed = {
            (edge.source_position, edge.target_position, edge.relation_id)
            for edge in self.base.edges
        }
        physical = {
            (edge.source_position, edge.target_position, edge.relation_id)
            for edge in self.physical_edges
        }
        if typed != physical:
            raise ERGT43DataError("physical and typed edges must have identical support")

    def to_manifest_record(self) -> dict[str, Any]:
        record = self.base.to_manifest_record()
        record["matched_topology"] = {
            "schema_version": SCHEMA_VERSION,
            "pair_id": self.pair_id,
            "scenario": self.scenario,
            "counterfactual_variant": self.counterfactual_variant,
            "topology_sha256": self.topology_sha256,
            "operator_sha256": self.operator_sha256,
            "physical_edges": [
                {
                    "source_position": edge.source_position,
                    "target_position": edge.target_position,
                    "relation_id": edge.relation_id,
                    "event_anchor_position": edge.event_anchor_position,
                    "action_cost": edge.action_cost,
                    "cone_admissible": edge.cone_admissible,
                    "transmission": edge.transmission,
                    "boundary_deficit": edge.boundary_deficit,
                    "terminal_capacity": edge.terminal_capacity,
                }
                for edge in self.physical_edges
            ],
        }
        return record


@dataclass(frozen=True)
class ERGT43PhysicalSupervision:
    event_mask: torch.Tensor
    action_cost: torch.Tensor
    cone_admissible: torch.Tensor
    transmission: torch.Tensor
    boundary_deficit: torch.Tensor
    terminal_capacity: torch.Tensor

    def to(self, device: torch.device | str) -> ERGT43PhysicalSupervision:
        return replace(
            self,
            event_mask=self.event_mask.to(device),
            action_cost=self.action_cost.to(device),
            cone_admissible=self.cone_admissible.to(device),
            transmission=self.transmission.to(device),
            boundary_deficit=self.boundary_deficit.to(device),
            terminal_capacity=self.terminal_capacity.to(device),
        )


@dataclass(frozen=True)
class ERGT43Batch:
    base: ERGT35Batch
    physical: ERGT43PhysicalSupervision
    raw_token_ids: torch.Tensor
    examples: tuple[MatchedTopologyExample, ...]

    def model_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "raw_token_ids": self.raw_token_ids,
            "attention_mask": self.base.attention_mask,
        }

    def to(self, device: torch.device | str) -> ERGT43Batch:
        return replace(
            self,
            base=self.base.to(device),
            physical=self.physical.to(device),
            raw_token_ids=self.raw_token_ids.to(device),
        )


@lru_cache(maxsize=131_072)
def _identity_fingerprint(token: str) -> bytes:
    return hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()


@lru_cache(maxsize=131_072)
def _identity_fibre_vector(token: str) -> tuple[float, ...]:
    """Return a train/test-independent identity charge for one raw symbol."""

    value = int.from_bytes(_identity_fingerprint(token), byteorder="big", signed=False)
    scale = 1.0 / math.sqrt(float(IDENTITY_FIBRE_BITS))
    return tuple(scale if (value >> bit) & 1 else -scale for bit in range(IDENTITY_FIBRE_BITS))


def reference_identity_fibre_tensor(
    examples: Sequence[MatchedTopologyExample],
    *,
    pad_to_tokens: int,
) -> torch.Tensor:
    reference = torch.zeros(
        (len(examples), pad_to_tokens, IDENTITY_FIBRE_BITS), dtype=torch.float32
    )
    for row, example in enumerate(examples):
        reference[row, : len(example.tokens)] = torch.tensor(
            [_identity_fibre_vector(token) for token in example.tokens], dtype=torch.float32
        )
    return reference


def identity_collision_audit(
    examples: Sequence[MatchedTopologyExample],
    tokenizer: TrainOnlyTokenizer,
) -> dict[str, Any]:
    """Audit semantic hash collisions separately from the identity fibre."""

    by_hop: dict[int, dict[str, int]] = {}
    identity_any = 0
    identity_candidate = 0
    semantic_any = 0
    semantic_candidate = 0
    for example in examples:
        tokens = example.base.tokens
        entity_tokens = [tokens[position] for position in example.base.entity_positions]
        candidate_tokens = [tokens[position] for position in example.base.candidate_positions]
        fingerprints = [_identity_fingerprint(token) for token in entity_tokens]
        candidate_fingerprints = [_identity_fingerprint(token) for token in candidate_tokens]
        encoded = tokenizer.encode(tokens)
        semantic_ids = [encoded[position] for position in example.base.entity_positions]
        candidate_semantic_ids = [encoded[position] for position in example.base.candidate_positions]

        identity_any_hit = len(set(fingerprints)) != len(fingerprints)
        identity_candidate_hit = any(
            fingerprints.count(fingerprint) > 1 for fingerprint in candidate_fingerprints
        )
        semantic_any_hit = len(set(semantic_ids)) != len(semantic_ids)
        semantic_candidate_hit = any(
            semantic_ids.count(token_id) > 1 for token_id in candidate_semantic_ids
        )
        identity_any += int(identity_any_hit)
        identity_candidate += int(identity_candidate_hit)
        semantic_any += int(semantic_any_hit)
        semantic_candidate += int(semantic_candidate_hit)

        hops = int(example.base.metadata["path_hops"])
        row = by_hop.setdefault(
            hops,
            {
                "examples": 0,
                "identity_any_collision_examples": 0,
                "identity_candidate_collision_examples": 0,
                "semantic_any_collision_examples": 0,
                "semantic_candidate_collision_examples": 0,
            },
        )
        row["examples"] += 1
        row["identity_any_collision_examples"] += int(identity_any_hit)
        row["identity_candidate_collision_examples"] += int(identity_candidate_hit)
        row["semantic_any_collision_examples"] += int(semantic_any_hit)
        row["semantic_candidate_collision_examples"] += int(semantic_candidate_hit)

    count = len(examples)
    denominator = max(1, count)
    return {
        "examples": count,
        "identity_fibre_bits": IDENTITY_FIBRE_BITS,
        "identity_any_collision_examples": identity_any,
        "identity_candidate_collision_examples": identity_candidate,
        "identity_any_collision_rate": identity_any / denominator,
        "identity_candidate_collision_rate": identity_candidate / denominator,
        "semantic_any_collision_examples": semantic_any,
        "semantic_candidate_collision_examples": semantic_candidate,
        "semantic_any_collision_rate": semantic_any / denominator,
        "semantic_candidate_collision_rate": semantic_candidate / denominator,
        "by_hop": {str(hops): values for hops, values in sorted(by_hop.items())},
    }


def _hash_structure(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_register(relations: Sequence[int]) -> int:
    register = 0
    for relation in relations:
        register = apply_relation(register, int(relation))
    return register


def _edge_values(
    scenario: str,
    *,
    degraded: bool,
    hop: int,
    path_hops: int,
) -> tuple[float, bool, float, float, float]:
    action = 1.0
    cone = True
    transmission = 1.0
    deficit = 0.0
    terminal = 1.0
    if not degraded:
        return action, cone, transmission, deficit, terminal
    terminal_hop = hop == path_hops - 1
    middle_hop = hop == max(0, path_hops // 2)
    first_hop = hop == 0
    if scenario == "geodesic_action":
        action = 3.0
    elif scenario == "finite_speed_cone" and middle_hop:
        cone = False
    elif scenario == "boundary_deficit" and terminal_hop:
        deficit = 2.0
    elif scenario == "payload_transport" and middle_hop:
        transmission = 0.25
    elif scenario == "terminal_mass" and terminal_hop:
        terminal = 0.25
    elif scenario == "memory_source_condition" and first_hop:
        transmission = 0.25
    elif scenario == "combined_long_horizon":
        action = 2.0
        if middle_hop:
            cone = False
            transmission = 0.25
        if terminal_hop:
            deficit = 2.0
            terminal = 0.25
    return action, cone, transmission, deficit, terminal


def _condition_tokens(
    action: float,
    cone: bool,
    transmission: float,
    deficit: float,
    terminal: float,
) -> tuple[str, ...]:
    return (
        f"action_{int(round(action))}",
        "cone_open" if cone else "cone_closed",
        f"transport_{int(round(100 * transmission)):03d}",
        f"deficit_{int(round(deficit))}",
        f"terminal_{int(round(100 * terminal)):03d}",
    )


def _make_pair(
    pair_index: int,
    *,
    local_pair_index: int,
    scenario: str,
    path_hops: int,
    split: str,
    rng: random.Random,
    node_label_pool_size: int,
) -> tuple[MatchedTopologyExample, MatchedTopologyExample]:
    if path_hops < 1:
        raise ERGT43DataError("matched topology requires at least one hop")
    node_count = 1 + 2 * path_hops
    if node_label_pool_size < node_count:
        raise ERGT43DataError(
            "node label pool must cover every node in the longest requested graph"
        )
    # Cover the complete finite identity vocabulary during short-hop training,
    # while randomizing the correspondence between a label and a graph role.
    start = (int(local_pair_index) * 17) % int(node_label_pool_size)
    labels = [
        (start + offset) % int(node_label_pool_size) for offset in range(node_count)
    ]
    rng.shuffle(labels)
    names = tuple(f"g{pair_index:06d}_n{label}" for label in labels)
    canonical_positions = tuple(range(1, node_count + 1))
    source_slot = 0
    path_a = (source_slot, *range(1, path_hops), path_hops)
    path_b = (
        source_slot,
        *range(path_hops + 1, 2 * path_hops),
        2 * path_hops,
    )
    relation_sequence = tuple(
        rng.randrange(1, len(RELATION_NAMES)) for _ in range(path_hops)
    )
    terminal_register = _path_register(relation_sequence)
    topology_payload = {
        "node_count": node_count,
        "paths": [list(range(path_hops + 1)), list(range(path_hops + 1))],
        "hop_count": path_hops,
    }
    operator_payload = {"relations": list(relation_sequence) * 2}
    topology_sha = _hash_structure(topology_payload)
    operator_sha = _hash_structure(operator_payload)
    pair_id = f"ergt43_pair_{pair_index:05d}"
    fact_order = [
        (path_id, hop)
        for path_id in range(2)
        for hop in range(path_hops)
    ]
    rng.shuffle(fact_order)

    examples: list[MatchedTopologyExample] = []
    for variant in (0, 1):
        degraded_path = variant
        tokens: list[str] = ["nodes", *names, ";", "facts"]
        typed_edges: list[DynamicTypedEdge] = []
        physical_edges: list[PhysicalEdgeCondition] = []
        paths = (path_a, path_b)
        for path_id, hop in fact_order:
            path = paths[path_id]
            source_slot_i = path[hop]
            target_slot_i = path[hop + 1]
            relation_id = relation_sequence[hop]
            action, cone, transmission, deficit, terminal = _edge_values(
                scenario,
                degraded=path_id == degraded_path,
                hop=hop,
                path_hops=path_hops,
            )
            source_position = canonical_positions[source_slot_i]
            target_position = canonical_positions[target_slot_i]
            tokens.append(names[source_slot_i])
            event_anchor = len(tokens)
            tokens.extend(
                [
                    RELATION_SURFACES[relation_id],
                    names[target_slot_i],
                    *_condition_tokens(action, cone, transmission, deficit, terminal),
                    ";",
                ]
            )
            typed_edges.append(DynamicTypedEdge(source_position, target_position, relation_id))
            physical_edges.append(
                PhysicalEdgeCondition(
                    source_position=source_position,
                    target_position=target_position,
                    relation_id=relation_id,
                    event_anchor_position=event_anchor,
                    action_cost=action,
                    cone_admissible=cone,
                    transmission=transmission,
                    boundary_deficit=deficit,
                    terminal_capacity=terminal,
                )
            )
        candidate_positions = (
            canonical_positions[path_a[-1]],
            canonical_positions[path_b[-1]],
        )
        tokens.extend(
            [
                "query",
                "source",
                names[source_slot],
                "candidate_a",
                names[path_a[-1]],
                "boundary_a",
                f"state_{terminal_register}",
                "candidate_b",
                names[path_b[-1]],
                "boundary_b",
                f"state_{terminal_register}",
            ]
        )
        answer_id = ANSWER_TO_ID["candidate_b" if degraded_path == 0 else "candidate_a"]
        base = ERGT35Example(
            example_id=f"{pair_id}_cf{variant}",
            split=split,
            raw_text=" ".join(tokens),
            tokens=tuple(tokens),
            entity_positions=canonical_positions,
            edges=tuple(typed_edges),
            source_position=canonical_positions[source_slot],
            candidate_positions=candidate_positions,
            boundary_labels=(terminal_register, terminal_register),
            answer_id=answer_id,
            candidate_registers=(terminal_register, terminal_register),
            metadata={
                "schema_version": SCHEMA_VERSION,
                "paper_evidence": True,
                "matched_topology": True,
                "both_candidates_topologically_supported": True,
                "scenario": scenario,
                "pair_id": pair_id,
                "counterfactual_variant": variant,
                "path_hops": path_hops,
                "label_balancing_nonce": rng.randrange(1 << 30),
                "scenario_hop_assignment": "orthogonal_grid",
                "node_role_labels_randomized": True,
                "fact_order_randomized_pair_consistent": True,
                "node_label_pool_size": int(node_label_pool_size),
            },
        )
        examples.append(
            MatchedTopologyExample(
                base=base,
                physical_edges=tuple(physical_edges),
                pair_id=pair_id,
                scenario=scenario,
                counterfactual_variant=variant,
                topology_sha256=topology_sha,
                operator_sha256=operator_sha,
            )
        )
    if examples[0].topology_sha256 != examples[1].topology_sha256:
        raise ERGT43DataError("counterfactual topology hashes must match")
    if examples[0].operator_sha256 != examples[1].operator_sha256:
        raise ERGT43DataError("counterfactual operator hashes must match")
    if examples[0].answer_id == examples[1].answer_id:
        raise ERGT43DataError("counterfactual labels must flip")
    if len(examples[0].tokens) != len(examples[1].tokens):
        raise ERGT43DataError("counterfactual token counts must match")
    return examples[0], examples[1]


def build_matched_topology_examples(
    *,
    pair_count: int,
    seed: int,
    split: str,
    min_hops: int = 2,
    max_hops: int = 8,
    scenarios: Sequence[str] = SCENARIOS,
    node_label_pool_size: int = 97,
) -> tuple[MatchedTopologyExample, ...]:
    """Build balanced counterfactual pairs with topology-conditioned label entropy 1 bit."""

    if pair_count <= 0 or min_hops < 1 or max_hops < min_hops:
        raise ERGT43DataError("invalid pair count or hop range")
    if node_label_pool_size < 1 + 2 * max_hops:
        raise ERGT43DataError("node label pool is too small for the requested hop range")
    normalized = tuple(str(item) for item in scenarios)
    if not normalized or any(item not in SCENARIOS for item in normalized):
        raise ERGT43DataError("scenario list contains an unregistered value")
    rng = random.Random(int(seed))
    examples: list[MatchedTopologyExample] = []
    for pair_index in range(pair_count):
        scenario = normalized[pair_index % len(normalized)]
        hops = min_hops + (
            (pair_index // len(normalized)) % (max_hops - min_hops + 1)
        )
        examples.extend(
            _make_pair(
                pair_index + seed * 10_000,
                local_pair_index=pair_index,
                scenario=scenario,
                path_hops=hops,
                split=split,
                rng=rng,
                node_label_pool_size=node_label_pool_size,
            )
        )
    rng.shuffle(examples)
    return tuple(examples)


def collate_matched_topology_examples(
    examples: Sequence[MatchedTopologyExample],
    tokenizer: TrainOnlyTokenizer,
    *,
    pad_to_tokens: int | None = None,
) -> ERGT43Batch:
    if not examples:
        raise ERGT43DataError("cannot collate an empty ERGT-43 batch")
    base = collate_examples(
        tuple(example.base for example in examples),
        tokenizer,
        pad_to_tokens=pad_to_tokens,
    )
    batch_size, token_count = base.token_ids.shape
    raw_token_ids = torch.zeros((batch_size, token_count), dtype=torch.int64)
    event_mask = torch.zeros((batch_size, token_count), dtype=torch.bool)
    action = torch.zeros((batch_size, token_count), dtype=torch.float32)
    cone = torch.ones((batch_size, token_count), dtype=torch.float32)
    transmission = torch.ones((batch_size, token_count), dtype=torch.float32)
    deficit = torch.zeros((batch_size, token_count), dtype=torch.float32)
    terminal = torch.ones((batch_size, token_count), dtype=torch.float32)
    for row, example in enumerate(examples):
        raw_token_ids[row, : len(example.tokens)] = torch.tensor(
            [raw_token_fingerprint_id(token) for token in example.tokens], dtype=torch.int64
        )
        for edge in example.physical_edges:
            anchor = edge.event_anchor_position
            if anchor >= token_count:
                raise ERGT43DataError("physical event anchor exceeds padded token capacity")
            event_mask[row, anchor] = True
            action[row, anchor] = float(edge.action_cost)
            cone[row, anchor] = float(edge.cone_admissible)
            transmission[row, anchor] = float(edge.transmission)
            deficit[row, anchor] = float(edge.boundary_deficit)
            terminal[row, anchor] = float(edge.terminal_capacity)
    return ERGT43Batch(
        base=base,
        physical=ERGT43PhysicalSupervision(
            event_mask=event_mask,
            action_cost=action,
            cone_admissible=cone,
            transmission=transmission,
            boundary_deficit=deficit,
            terminal_capacity=terminal,
        ),
        raw_token_ids=raw_token_ids,
        examples=tuple(examples),
    )


def manifest_hash(examples: Iterable[MatchedTopologyExample]) -> str:
    records = [example.to_manifest_record() for example in examples]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ERGT43Batch",
    "ERGT43DataError",
    "ERGT43PhysicalSupervision",
    "IDENTITY_FIBRE_BITS",
    "MatchedTopologyExample",
    "PhysicalEdgeCondition",
    "RawTokenInputContract",
    "SCENARIOS",
    "SCHEMA_VERSION",
    "build_matched_topology_examples",
    "collate_matched_topology_examples",
    "identity_collision_audit",
    "manifest_hash",
    "raw_token_fingerprint_id",
    "reference_identity_fibre_tensor",
]
