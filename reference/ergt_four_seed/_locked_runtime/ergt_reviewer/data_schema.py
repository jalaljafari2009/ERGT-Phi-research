"""Data contracts for the ERGT-35 end-to-end engineering gate.

The primary model input deliberately contains only token ids and a padding
mask.  Gold spans, graph edges, pointers, boundaries, and answers live in a
separate supervision sidecar and cannot enter an M0-M3 forward call.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

ANSWER_NAMES = ("candidate_a", "candidate_b", "abstain")
ANSWER_TO_ID = {name: index for index, name in enumerate(ANSWER_NAMES)}

RELATION_NAMES = ("none", "plus_one", "plus_two", "hold")
RELATION_TO_ID = {name: index for index, name in enumerate(RELATION_NAMES)}
RELATION_DELTAS = (0, 1, 2, 0)
RELATION_SURFACES = {1: "raises", 2: "shifts", 3: "keeps"}
RELATION_SURFACE_TO_ID = {surface: relation for relation, surface in RELATION_SURFACES.items()}
QUERY_ROLE_NAMES = ("none", "source", "candidate_a", "boundary_a", "candidate_b", "boundary_b")
QUERY_ROLE_TO_ID = {name: index for index, name in enumerate(QUERY_ROLE_NAMES)}
REGISTER_CARDINALITY = 3

PRIMARY_FORWARD_KEYS = frozenset({"token_ids", "attention_mask"})
FORBIDDEN_FORWARD_KEYS = frozenset(
    {
        "answer_labels",
        "boundary_labels",
        "candidate_positions",
        "edge_labels",
        "entity_labels",
        "entity_positions",
        "gold_graph",
        "operator_labels",
        "event_source_positions",
        "event_target_positions",
        "relation_anchor_labels",
        "relation_event_mask",
        "query_role_labels",
        "query_boundary_value_labels",
        "register_labels",
        "source_positions",
        "target_positions",
    }
)


class ERGT35DataError(ValueError):
    """Raised when an ERGT-35 data or leakage contract is invalid."""


@dataclass(frozen=True)
class DynamicTypedEdge:
    """One typed directed edge over example-local token pointers."""

    source_position: int
    target_position: int
    relation_id: int

    def __post_init__(self) -> None:
        if self.source_position < 0 or self.target_position < 0:
            raise ERGT35DataError("edge token pointers must be non-negative")
        if self.relation_id <= 0 or self.relation_id >= len(RELATION_NAMES):
            raise ERGT35DataError("edge relation_id must name a non-null relation")


@dataclass(frozen=True)
class ERGT35Example:
    """One raw-text example plus a supervision-only structured sidecar."""

    example_id: str
    split: str
    raw_text: str
    tokens: tuple[str, ...]
    entity_positions: tuple[int, ...]
    edges: tuple[DynamicTypedEdge, ...]
    source_position: int
    candidate_positions: tuple[int, int]
    boundary_labels: tuple[int, int]
    answer_id: int
    candidate_registers: tuple[int, int]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        token_count = len(self.tokens)
        if not self.example_id:
            raise ERGT35DataError("example_id must be non-empty")
        if not self.tokens or self.raw_text != " ".join(self.tokens):
            raise ERGT35DataError("raw_text must be the whitespace join of tokens")
        positions = self.entity_positions
        if tuple(sorted(set(positions))) != positions:
            raise ERGT35DataError("entity_positions must be sorted and unique")
        if any(position < 0 or position >= token_count for position in positions):
            raise ERGT35DataError("entity position is outside the token sequence")
        if self.source_position not in positions:
            raise ERGT35DataError("source pointer must reference an entity position")
        if len(self.candidate_positions) != 2 or any(
            position not in positions for position in self.candidate_positions
        ):
            raise ERGT35DataError("candidate pointers must reference local entities")
        if len(set(self.candidate_positions)) != 2:
            raise ERGT35DataError("candidate pointers must be distinct")
        if any(
            edge.source_position not in positions or edge.target_position not in positions
            for edge in self.edges
        ):
            raise ERGT35DataError("edge endpoints must reference local entities")
        if any(label < 0 or label >= REGISTER_CARDINALITY for label in self.boundary_labels):
            raise ERGT35DataError("boundary label is outside the register space")
        if self.answer_id < 0 or self.answer_id >= len(ANSWER_NAMES):
            raise ERGT35DataError("answer_id is outside the answer vocabulary")
        if any(value < -1 or value >= REGISTER_CARDINALITY for value in self.candidate_registers):
            raise ERGT35DataError("candidate register must be -1 or a valid state")

    def to_manifest_record(self) -> dict[str, Any]:
        """Return a serializable record with supervision clearly namespaced."""

        return {
            "example_id": self.example_id,
            "split": self.split,
            "raw_text": self.raw_text,
            "supervision": {
                "entity_positions": list(self.entity_positions),
                "edges": [
                    {
                        "source_position": edge.source_position,
                        "target_position": edge.target_position,
                        "relation_id": edge.relation_id,
                    }
                    for edge in self.edges
                ],
                "source_position": self.source_position,
                "candidate_positions": list(self.candidate_positions),
                "boundary_labels": list(self.boundary_labels),
                "answer_id": self.answer_id,
                "candidate_registers": list(self.candidate_registers),
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TrainOnlyTokenizer:
    """Minimal whitespace tokenizer whose vocabulary is fitted on training text."""

    token_to_id: Mapping[str, int]
    hash_bucket_count: int = 0
    minimum_document_frequency: int = 1

    @classmethod
    def fit(
        cls,
        examples: Iterable[ERGT35Example],
        *,
        hash_bucket_count: int = 0,
        minimum_document_frequency: int = 1,
    ) -> TrainOnlyTokenizer:
        if hash_bucket_count < 0:
            raise ERGT35DataError("hash_bucket_count must be non-negative")
        if minimum_document_frequency < 1:
            raise ERGT35DataError("minimum_document_frequency must be positive")
        document_frequency: Counter[str] = Counter()
        for example in examples:
            document_frequency.update(set(example.tokens))
        vocabulary = sorted(
            token
            for token, frequency in document_frequency.items()
            if frequency >= int(minimum_document_frequency)
        )
        mapping = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        mapping.update({token: index + 2 for index, token in enumerate(vocabulary)})
        return cls(
            token_to_id=mapping,
            hash_bucket_count=int(hash_bucket_count),
            minimum_document_frequency=int(minimum_document_frequency),
        )

    @property
    def pad_id(self) -> int:
        return int(self.token_to_id[PAD_TOKEN])

    @property
    def unk_id(self) -> int:
        return int(self.token_to_id[UNK_TOKEN])

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id) + self.hash_bucket_count

    def encode(self, tokens: Sequence[str]) -> list[int]:
        encoded: list[int] = []
        for token in tokens:
            known = self.token_to_id.get(token)
            if known is not None:
                encoded.append(int(known))
            elif self.hash_bucket_count > 0:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest, byteorder="big") % self.hash_bucket_count
                encoded.append(len(self.token_to_id) + bucket)
            else:
                encoded.append(self.unk_id)
        return encoded

    def to_json_record(self) -> dict[str, Any]:
        return {
            "type": "whitespace_train_only_with_fixed_hash_oov",
            "token_to_id": dict(self.token_to_id),
            "hash_bucket_count": self.hash_bucket_count,
            "minimum_document_frequency": self.minimum_document_frequency,
        }


@dataclass(frozen=True)
class ERGT35Supervision:
    """Gold tensors kept outside the primary model-input dictionary."""

    entity_labels: torch.Tensor
    entity_pair_mask: torch.Tensor
    relation_labels: torch.Tensor
    relation_anchor_labels: torch.Tensor
    relation_event_mask: torch.Tensor
    event_source_positions: torch.Tensor
    event_target_positions: torch.Tensor
    query_role_labels: torch.Tensor
    query_boundary_value_labels: torch.Tensor
    source_positions: torch.Tensor
    candidate_positions: torch.Tensor
    boundary_labels: torch.Tensor
    answer_labels: torch.Tensor

    def to(self, device: torch.device | str) -> ERGT35Supervision:
        return replace(
            self,
            entity_labels=self.entity_labels.to(device),
            entity_pair_mask=self.entity_pair_mask.to(device),
            relation_labels=self.relation_labels.to(device),
            relation_anchor_labels=self.relation_anchor_labels.to(device),
            relation_event_mask=self.relation_event_mask.to(device),
            event_source_positions=self.event_source_positions.to(device),
            event_target_positions=self.event_target_positions.to(device),
            query_role_labels=self.query_role_labels.to(device),
            query_boundary_value_labels=self.query_boundary_value_labels.to(device),
            source_positions=self.source_positions.to(device),
            candidate_positions=self.candidate_positions.to(device),
            boundary_labels=self.boundary_labels.to(device),
            answer_labels=self.answer_labels.to(device),
        )


@dataclass(frozen=True)
class ERGT35Batch:
    """Tensorized examples with a strict primary-input/supervision split."""

    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    supervision: ERGT35Supervision
    examples: tuple[ERGT35Example, ...]

    def model_inputs(self) -> dict[str, torch.Tensor]:
        inputs = {"token_ids": self.token_ids, "attention_mask": self.attention_mask}
        assert_primary_forward_inputs(inputs)
        return inputs

    def to(self, device: torch.device | str) -> ERGT35Batch:
        return replace(
            self,
            token_ids=self.token_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            supervision=self.supervision.to(device),
        )


def assert_primary_forward_inputs(inputs: Mapping[str, Any]) -> None:
    """Fail if any supervision or non-pre-registered field reaches a model."""

    keys = frozenset(inputs)
    forbidden = sorted(keys & FORBIDDEN_FORWARD_KEYS)
    unexpected = sorted(keys - PRIMARY_FORWARD_KEYS)
    missing = sorted(PRIMARY_FORWARD_KEYS - keys)
    if forbidden:
        raise ERGT35DataError(f"gold fields entered primary forward inputs: {forbidden}")
    if unexpected:
        raise ERGT35DataError(f"unexpected primary forward inputs: {unexpected}")
    if missing:
        raise ERGT35DataError(f"missing primary forward inputs: {missing}")


def collate_examples(
    examples: Sequence[ERGT35Example],
    tokenizer: TrainOnlyTokenizer,
    *,
    pad_to_tokens: int | None = None,
) -> ERGT35Batch:
    """Create a padded batch while preserving dynamic token-pointer targets."""

    if not examples:
        raise ERGT35DataError("cannot collate an empty example sequence")
    observed_max_tokens = max(len(example.tokens) for example in examples)
    if pad_to_tokens is not None and int(pad_to_tokens) < observed_max_tokens:
        raise ERGT35DataError(
            "pad_to_tokens cannot be smaller than the longest example in the batch"
        )
    max_tokens = observed_max_tokens if pad_to_tokens is None else int(pad_to_tokens)
    batch_size = len(examples)
    token_ids = torch.full((batch_size, max_tokens), tokenizer.pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    entity_labels = torch.zeros((batch_size, max_tokens), dtype=torch.float32)
    entity_pair_mask = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    relation_labels = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.long)
    relation_anchor_labels = torch.zeros((batch_size, max_tokens), dtype=torch.long)
    relation_event_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    event_source_positions = torch.full((batch_size, max_tokens), -1, dtype=torch.long)
    event_target_positions = torch.full((batch_size, max_tokens), -1, dtype=torch.long)
    query_role_labels = torch.zeros((batch_size, max_tokens), dtype=torch.long)
    query_boundary_value_labels = torch.full((batch_size, max_tokens), -1, dtype=torch.long)
    source_positions = torch.zeros(batch_size, dtype=torch.long)
    candidate_positions = torch.zeros((batch_size, 2), dtype=torch.long)
    boundary_labels = torch.zeros((batch_size, 2), dtype=torch.long)
    answer_labels = torch.zeros(batch_size, dtype=torch.long)

    for row, example in enumerate(examples):
        encoded = tokenizer.encode(example.tokens)
        length = len(encoded)
        token_ids[row, :length] = torch.tensor(encoded, dtype=torch.long)
        attention_mask[row, :length] = True
        positions = torch.tensor(example.entity_positions, dtype=torch.long)
        entity_labels[row, positions] = 1.0
        entity_pair_mask[row, positions[:, None], positions[None, :]] = True
        for edge in example.edges:
            relation_labels[row, edge.source_position, edge.target_position] = edge.relation_id
        canonical_by_name = {
            example.tokens[position]: position for position in example.entity_positions
        }
        observed_events: list[tuple[int, int, int]] = []
        for token_position, token in enumerate(example.tokens):
            relation_id = RELATION_SURFACE_TO_ID.get(token)
            if relation_id is None:
                continue
            if token_position == 0 or token_position + 1 >= length:
                raise ERGT35DataError("relation event is missing an adjacent endpoint mention")
            source = canonical_by_name.get(example.tokens[token_position - 1])
            target = canonical_by_name.get(example.tokens[token_position + 1])
            if source is None or target is None:
                raise ERGT35DataError("relation event endpoints do not bind to canonical entities")
            relation_anchor_labels[row, token_position] = relation_id
            relation_event_mask[row, token_position] = True
            event_source_positions[row, token_position] = source
            event_target_positions[row, token_position] = target
            observed_events.append((source, target, relation_id))
        expected_events = [
            (edge.source_position, edge.target_position, edge.relation_id)
            for edge in example.edges
        ]
        if sorted(observed_events) != sorted(expected_events):
            raise ERGT35DataError("raw relation events do not match the supervision graph")
        query_values = {
            "source": None,
            "candidate_a": None,
            "boundary_a": int(example.boundary_labels[0]),
            "candidate_b": None,
            "boundary_b": int(example.boundary_labels[1]),
        }
        for token_position, token in enumerate(example.tokens):
            role_id = QUERY_ROLE_TO_ID.get(token)
            if role_id is None or role_id == 0:
                continue
            query_role_labels[row, token_position] = role_id
            boundary_value = query_values[token]
            if boundary_value is not None:
                query_boundary_value_labels[row, token_position] = boundary_value
        source_positions[row] = example.source_position
        candidate_positions[row] = torch.tensor(example.candidate_positions, dtype=torch.long)
        boundary_labels[row] = torch.tensor(example.boundary_labels, dtype=torch.long)
        answer_labels[row] = example.answer_id

    supervision = ERGT35Supervision(
        entity_labels=entity_labels,
        entity_pair_mask=entity_pair_mask,
        relation_labels=relation_labels,
        relation_anchor_labels=relation_anchor_labels,
        relation_event_mask=relation_event_mask,
        event_source_positions=event_source_positions,
        event_target_positions=event_target_positions,
        query_role_labels=query_role_labels,
        query_boundary_value_labels=query_boundary_value_labels,
        source_positions=source_positions,
        candidate_positions=candidate_positions,
        boundary_labels=boundary_labels,
        answer_labels=answer_labels,
    )
    return ERGT35Batch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        supervision=supervision,
        examples=tuple(examples),
    )


def apply_relation(register: int, relation_id: int) -> int:
    """Apply the pre-registered typed operator to a finite register."""

    if register < 0 or register >= REGISTER_CARDINALITY:
        raise ERGT35DataError("register is outside the finite state space")
    if relation_id <= 0 or relation_id >= len(RELATION_DELTAS):
        raise ERGT35DataError("relation_id does not identify an executable operator")
    return (register + RELATION_DELTAS[relation_id]) % REGISTER_CARDINALITY
