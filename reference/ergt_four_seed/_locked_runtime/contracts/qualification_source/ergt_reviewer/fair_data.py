"""Shared raw-input protocol for the direct Transformer and native ERGT.

The historical generator used graph-specific entity spellings.  A train-only
word vocabulary would map every unseen entity to ``<unk>`` while ERGT retained
an exact identity fibre.  This module removes that confound: graph-local
entities are serialized as ``node_0``, ``node_1``, ... for both models, using
a fixed collision-free protocol vocabulary.  Labels and structured sidecars
remain supervision/audit data and never enter either forward call.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from .data_schema import PAD_TOKEN, UNK_TOKEN, TrainOnlyTokenizer
from .matched_data import (
    SCENARIOS,
    MatchedTopologyExample,
    PhysicalEdgeCondition,
    build_matched_topology_examples,
    raw_token_fingerprint_id,
)

_ENTITY_PATTERN = re.compile(r"^g\d+_n(\d+)$")
_NOISE_TOKENS = tuple(f"noise_{index:02d}" for index in range(16))
_MECHANISM_TUNING_SCENARIOS = tuple(
    scenario for scenario in SCENARIOS if scenario != "combined_long_horizon"
)


@dataclass(frozen=True)
class DatasetBundle:
    train: tuple[MatchedTopologyExample, ...]
    tuning: tuple[MatchedTopologyExample, ...]
    mechanism_tuning: tuple[MatchedTopologyExample, ...]
    validation: tuple[MatchedTopologyExample, ...]
    hop_cohorts: Mapping[str, tuple[MatchedTopologyExample, ...]]
    context_cohorts: Mapping[str, tuple[MatchedTopologyExample, ...]]
    unsupported_cohorts: Mapping[str, tuple[MatchedTopologyExample, ...]]
    tokenizer: TrainOnlyTokenizer


def _canonical_token(token: str) -> str:
    match = _ENTITY_PATTERN.match(token)
    return f"node_{int(match.group(1))}" if match else token


def canonicalize_examples(
    examples: Iterable[MatchedTopologyExample],
) -> tuple[MatchedTopologyExample, ...]:
    """Replace graph-instance names with shared compositional local symbols."""

    canonical: list[MatchedTopologyExample] = []
    for example in examples:
        tokens = tuple(_canonical_token(token) for token in example.tokens)
        metadata = {
            **dict(example.base.metadata),
            "shared_collision_free_serialization": True,
            "entity_serialization": "randomized_finite_node_label",
        }
        base = replace(
            example.base,
            raw_text=" ".join(tokens),
            tokens=tokens,
            metadata=metadata,
        )
        canonical.append(replace(example, base=base))
    return tuple(canonical)


def add_context_distractors(
    examples: Sequence[MatchedTopologyExample],
    target_tokens: int,
) -> tuple[MatchedTopologyExample, ...]:
    """Append label-independent distractors without changing any gold pointer."""

    padded: list[MatchedTopologyExample] = []
    for row, example in enumerate(examples):
        if len(example.tokens) > target_tokens:
            raise ValueError(
                f"example {example.example_id} has {len(example.tokens)} tokens, "
                f"which exceeds context target {target_tokens}"
            )
        needed = target_tokens - len(example.tokens)
        distractors = tuple(
            _NOISE_TOKENS[(row * 7 + index) % len(_NOISE_TOKENS)] for index in range(needed)
        )
        tokens = (*example.tokens, *distractors)
        metadata = {
            **dict(example.base.metadata),
            "context_target_tokens": int(target_tokens),
            "distractor_tokens": int(needed),
            "distractors_are_label_independent": True,
        }
        base = replace(
            example.base,
            raw_text=" ".join(tokens),
            tokens=tuple(tokens),
            metadata=metadata,
        )
        padded.append(replace(example, base=base))
    return tuple(padded)


def append_training_distractors(
    examples: Sequence[MatchedTopologyExample],
    distractor_count: int,
) -> tuple[MatchedTopologyExample, ...]:
    """Expose every registered noise symbol without changing graph supervision."""

    if distractor_count <= 0:
        return tuple(examples)
    augmented: list[MatchedTopologyExample] = []
    for row, example in enumerate(examples):
        distractors = tuple(
            _NOISE_TOKENS[(row * 7 + index) % len(_NOISE_TOKENS)]
            for index in range(distractor_count)
        )
        tokens = (*example.tokens, *distractors)
        base = replace(
            example.base,
            raw_text=" ".join(tokens),
            tokens=tuple(tokens),
            metadata={
                **dict(example.base.metadata),
                "training_noise_exposure": int(distractor_count),
                "training_noise_is_label_independent": True,
            },
        )
        augmented.append(replace(example, base=base))
    return tuple(augmented)


def make_unsupported(
    examples: Sequence[MatchedTopologyExample],
) -> tuple[MatchedTopologyExample, ...]:
    """Make both candidates physically inadmissible while preserving topology."""

    unsupported: list[MatchedTopologyExample] = []
    for example in examples:
        candidate_positions = set(example.base.candidate_positions)
        tokens = list(example.tokens)
        physical: list[PhysicalEdgeCondition] = []
        for edge in example.physical_edges:
            if edge.target_position in candidate_positions:
                edge = replace(edge, terminal_capacity=0.25)
                terminal_token_position = edge.event_anchor_position + 6
                tokens[terminal_token_position] = "terminal_025"
            physical.append(edge)
        metadata = {
            **dict(example.base.metadata),
            "unsupported_answer": True,
            "both_candidates_terminally_inadmissible": True,
        }
        base = replace(
            example.base,
            raw_text=" ".join(tokens),
            tokens=tuple(tokens),
            answer_id=2,
            candidate_registers=(-1, -1),
            metadata=metadata,
        )
        unsupported.append(
            replace(
                example,
                base=base,
                physical_edges=tuple(physical),
                pair_id=f"{example.pair_id}_unsupported",
            )
        )
    return tuple(unsupported)


def protocol_tokenizer(node_label_pool_size: int = 97) -> TrainOnlyTokenizer:
    """Return the preregistered finite vocabulary shared by both models."""

    vocabulary = {
        "nodes",
        "facts",
        ";",
        "query",
        "source",
        "candidate_a",
        "candidate_b",
        "boundary_a",
        "boundary_b",
        "raises",
        "shifts",
        "keeps",
        "action_1",
        "action_2",
        "action_3",
        "cone_open",
        "cone_closed",
        "transport_025",
        "transport_100",
        "deficit_0",
        "deficit_2",
        "terminal_025",
        "terminal_100",
        "state_0",
        "state_1",
        "state_2",
        *_NOISE_TOKENS,
        *(f"node_{index}" for index in range(node_label_pool_size)),
    }
    mapping = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    mapping.update({token: index + 2 for index, token in enumerate(sorted(vocabulary))})
    return TrainOnlyTokenizer(token_to_id=mapping)


def assert_shared_raw_input_contract(
    cohorts: Iterable[Sequence[MatchedTopologyExample]],
    tokenizer: TrainOnlyTokenizer,
    *,
    training_examples: Sequence[MatchedTopologyExample] | None = None,
) -> dict[str, int | float | bool | list[int] | list[str]]:
    materialized = tuple(tuple(cohort) for cohort in cohorts)
    examples = tuple(example for cohort in materialized for example in cohort)
    training = tuple(training_examples or (materialized[0] if materialized else ()))
    unknown = sum(
        token_id == tokenizer.unk_id
        for example in examples
        for token_id in tokenizer.encode(example.tokens)
    )
    topology_pairs: dict[str, set[str]] = {}
    operator_pairs: dict[str, set[str]] = {}
    raw_tokens = {token for example in examples for token in example.tokens}
    raw_fingerprints = {raw_token_fingerprint_id(token) for token in raw_tokens}
    for example in examples:
        topology_pairs.setdefault(example.pair_id, set()).add(example.topology_sha256)
        operator_pairs.setdefault(example.pair_id, set()).add(example.operator_sha256)
    training_token_ids = {
        token_id for example in training for token_id in tokenizer.encode(example.tokens)
    }
    evaluation = tuple(example for example in examples if example not in training)
    evaluation_token_ids = {
        token_id
        for example in evaluation
        for token_id in tokenizer.encode(example.tokens)
        if token_id != tokenizer.pad_id
    }
    unexposed_ids = sorted(evaluation_token_ids - training_token_ids)
    train_combinations = {
        (example.scenario, int(example.base.metadata["path_hops"]))
        for example in training
        if not bool(example.base.metadata.get("unsupported_answer", False))
    }
    train_scenarios = {scenario for scenario, _ in train_combinations}
    train_hops = {hops for _, hops in train_combinations}
    expected_combinations = len(train_scenarios) * len(train_hops)
    raw_hashes = [
        hashlib.sha256(example.base.raw_text.encode("utf-8")).hexdigest() for example in examples
    ]
    supported_raw_hashes = [
        hashlib.sha256(example.base.raw_text.encode("utf-8")).hexdigest()
        for example in examples
        if not bool(example.base.metadata.get("unsupported_answer", False))
    ]
    train_hashes = {
        hashlib.sha256(example.base.raw_text.encode("utf-8")).hexdigest() for example in training
    }
    evaluation_hashes = {
        hashlib.sha256(example.base.raw_text.encode("utf-8")).hexdigest() for example in evaluation
    }
    token_frequency = Counter(
        token_id
        for example in training
        for token_id in tokenizer.encode(example.tokens)
        if token_id != tokenizer.pad_id
    )
    node_ids = {
        tokenizer.token_to_id[token] for token in tokenizer.token_to_id if token.startswith("node_")
    }
    noise_ids = {
        tokenizer.token_to_id[token]
        for token in tokenizer.token_to_id
        if token.startswith("noise_")
    }
    duplicate_count = len(supported_raw_hashes) - len(set(supported_raw_hashes))
    intentional_unsupported_duplicates = len(raw_hashes) - len(set(raw_hashes)) - duplicate_count
    orthogonal_coverage = len(train_combinations) / max(1, expected_combinations)
    fingerprint_contract = len(raw_fingerprints) == len(raw_tokens)
    topology_contract = all(len(values) == 1 for values in topology_pairs.values())
    operator_contract = all(len(values) == 1 for values in operator_pairs.values())
    contract_pass = (
        unknown == 0
        and fingerprint_contract
        and topology_contract
        and operator_contract
        and not unexposed_ids
        and not (train_hashes & evaluation_hashes)
        and duplicate_count == 0
        and orthogonal_coverage == 1.0
        and all(token_frequency[token_id] > 0 for token_id in node_ids | noise_ids)
    )
    return {
        "examples": len(examples),
        "unknown_token_count": unknown,
        "unknown_token_free": unknown == 0,
        "raw_fingerprint_collision_free": fingerprint_contract,
        "external_identity_sidecar_used": False,
        "topology_hash_pair_contract": topology_contract,
        "operator_hash_pair_contract": operator_contract,
        "evaluation_unexposed_semantic_token_ids": unexposed_ids,
        "all_evaluation_semantic_rows_train_exposed": not unexposed_ids,
        "all_node_rows_train_exposed": all(token_frequency[token_id] > 0 for token_id in node_ids),
        "all_noise_rows_train_exposed": all(
            token_frequency[token_id] > 0 for token_id in noise_ids
        ),
        "train_evaluation_raw_overlap_count": len(train_hashes & evaluation_hashes),
        "exact_raw_duplicate_count": duplicate_count,
        "intentional_unsupported_duplicate_count": intentional_unsupported_duplicates,
        "scenario_hop_combinations_observed": len(train_combinations),
        "scenario_hop_combinations_expected": expected_combinations,
        "scenario_hop_orthogonal_coverage": orthogonal_coverage,
        "scenario_hop_orthogonal": orthogonal_coverage == 1.0,
        "data_contract_pass": contract_pass,
    }


def audit_label_contract(
    cohorts: Mapping[str, Sequence[MatchedTopologyExample]],
) -> dict[str, object]:
    """Audit labels independently of model predictions or scientific gates."""

    cohort_rows: list[dict[str, object]] = []
    all_pair_flip = True
    all_cell_balance = True
    all_unsupported = True
    all_physical_tokens = True
    total_counts: Counter[int] = Counter()
    for cohort_name, examples in cohorts.items():
        counts = Counter(int(example.answer_id) for example in examples)
        total_counts.update(counts)
        grouped: dict[str, list[MatchedTopologyExample]] = {}
        cells: dict[tuple[str, int], Counter[int]] = {}
        physical_alignment = True
        for example in examples:
            grouped.setdefault(example.pair_id, []).append(example)
            if example.answer_id != 2:
                key = (example.scenario, int(example.base.metadata["path_hops"]))
                cells.setdefault(key, Counter())[int(example.answer_id)] += 1
            for edge in example.physical_edges:
                anchor = int(edge.event_anchor_position)
                expected = (
                    f"action_{int(round(edge.action_cost))}",
                    "cone_open" if edge.cone_admissible else "cone_closed",
                    f"transport_{int(round(100 * edge.transmission)):03d}",
                    f"deficit_{int(round(edge.boundary_deficit))}",
                    f"terminal_{int(round(100 * edge.terminal_capacity)):03d}",
                )
                observed = tuple(example.tokens[anchor + offset] for offset in range(2, 7))
                physical_alignment = physical_alignment and observed == expected
        pair_flip = True
        unsupported_contract = True
        for values in grouped.values():
            unsupported = all(int(value.answer_id) == 2 for value in values)
            if unsupported:
                unsupported_contract = unsupported_contract and len(values) == 2
                continue
            pair_flip = pair_flip and (
                len(values) == 2
                and {int(value.counterfactual_variant) for value in values} == {0, 1}
                and {int(value.answer_id) for value in values} == {0, 1}
                and all(
                    int(value.answer_id) == (1 if int(value.counterfactual_variant) == 0 else 0)
                    for value in values
                )
                and len({value.topology_sha256 for value in values}) == 1
                and len({value.operator_sha256 for value in values}) == 1
            )
        cell_balance = all(
            cell.get(0, 0) == cell.get(1, 0) and cell.get(0, 0) > 0 for cell in cells.values()
        )
        all_pair_flip = all_pair_flip and pair_flip
        all_cell_balance = all_cell_balance and cell_balance
        all_unsupported = all_unsupported and unsupported_contract
        all_physical_tokens = all_physical_tokens and physical_alignment
        cohort_rows.append(
            {
                "cohort": cohort_name,
                "examples": len(examples),
                "candidate_a_labels": counts.get(0, 0),
                "candidate_b_labels": counts.get(1, 0),
                "abstain_labels": counts.get(2, 0),
                "counterfactual_flip_exact": pair_flip,
                "scenario_hop_label_balance": cell_balance,
                "unsupported_abstain_exact": unsupported_contract,
                "physical_tokens_match_supervision": physical_alignment,
            }
        )
    contract_pass = (
        all_pair_flip
        and all_cell_balance
        and all_unsupported
        and all_physical_tokens
        and total_counts.get(0, 0) == total_counts.get(1, 0)
    )
    return {
        "schema_version": "ergt-reviewer-label-contract-v1",
        "answer_label_mapping": {
            "candidate_a": 0,
            "candidate_b": 1,
            "abstain": 2,
        },
        "training_label_families": [
            "answer",
            "entity",
            "query_role",
            "relation_anchor",
            "event_source",
            "event_target",
            "boundary_value",
            "action",
            "cone",
            "transport",
            "boundary_deficit",
            "terminal_mass",
        ],
        "labels_are_training_or_audit_only": True,
        "labels_enter_primary_model_inputs": False,
        "counterfactual_flip_exact": all_pair_flip,
        "scenario_hop_label_balance": all_cell_balance,
        "unsupported_abstain_exact": all_unsupported,
        "physical_tokens_match_supervision": all_physical_tokens,
        "candidate_label_counts_equal": total_counts.get(0, 0) == total_counts.get(1, 0),
        "aggregate_label_counts": {
            "candidate_a": total_counts.get(0, 0),
            "candidate_b": total_counts.get(1, 0),
            "abstain": total_counts.get(2, 0),
        },
        "cohorts": cohort_rows,
        "label_contract_pass": contract_pass,
    }


def _cohort(
    *,
    pair_count: int,
    seed: int,
    split: str,
    min_hops: int,
    max_hops: int,
    node_label_pool_size: int,
    scenarios: Sequence[str] = SCENARIOS,
) -> tuple[MatchedTopologyExample, ...]:
    canonical = canonicalize_examples(
        build_matched_topology_examples(
            pair_count=pair_count,
            seed=seed,
            split=split,
            min_hops=min_hops,
            max_hops=max_hops,
            scenarios=scenarios,
            node_label_pool_size=node_label_pool_size,
        )
    )
    namespaced: list[MatchedTopologyExample] = []
    for example in canonical:
        pair_id = f"{split}:{example.pair_id}"
        base = replace(
            example.base,
            example_id=f"{split}:{example.base.example_id}",
            metadata={**dict(example.base.metadata), "pair_id": pair_id},
        )
        namespaced.append(replace(example, base=base, pair_id=pair_id))
    return tuple(namespaced)


def _unique_cohort(
    *,
    pair_count: int,
    seed: int,
    split: str,
    min_hops: int,
    max_hops: int,
    node_label_pool_size: int,
    forbidden_raw_texts: set[str],
    scenarios: Sequence[str] = SCENARIOS,
) -> tuple[MatchedTopologyExample, ...]:
    """Build complete pairs that are raw-text disjoint from earlier cohorts."""

    accepted: list[MatchedTopologyExample] = []
    accepted_pairs = 0
    attempt = 0
    local_raw_texts: set[str] = set()
    while accepted_pairs < pair_count:
        remaining = pair_count - accepted_pairs
        candidates = _cohort(
            pair_count=max(remaining * 2, 24),
            seed=seed + attempt * 1_000_003,
            split=split,
            min_hops=min_hops,
            max_hops=max_hops,
            node_label_pool_size=node_label_pool_size,
            scenarios=scenarios,
        )
        grouped: dict[str, list[MatchedTopologyExample]] = {}
        for example in candidates:
            grouped.setdefault(example.pair_id, []).append(example)
        for pair in grouped.values():
            raw_texts = {example.base.raw_text for example in pair}
            if len(pair) != 2 or len(raw_texts) != 2:
                continue
            if raw_texts & forbidden_raw_texts or raw_texts & local_raw_texts:
                continue
            accepted.extend(pair)
            local_raw_texts.update(raw_texts)
            accepted_pairs += 1
            if accepted_pairs == pair_count:
                break
        attempt += 1
        if attempt > 100:
            raise RuntimeError(f"could not build {pair_count} unique pairs for {split}")
    forbidden_raw_texts.update(local_raw_texts)
    return tuple(accepted)


def _balanced_training_cohort(
    *,
    pair_counts_by_hop: Mapping[str, object],
    seed: int,
    split: str,
    node_label_pool_size: int,
    forbidden_raw_texts: set[str],
) -> tuple[MatchedTopologyExample, ...]:
    """Build an explicit scenario-balanced hop mixture for shared training."""

    normalized = {
        int(hops): int(pair_count)
        for hops, pair_count in pair_counts_by_hop.items()
    }
    if not normalized or any(hops < 1 or pair_count <= 0 for hops, pair_count in normalized.items()):
        raise ValueError("training_hop_pair_counts must contain positive hop/count entries")
    if any(pair_count % len(SCENARIOS) for pair_count in normalized.values()):
        raise ValueError(
            "each training_hop_pair_counts value must be divisible by the scenario count"
        )
    examples: list[MatchedTopologyExample] = []
    for hops, pair_count in sorted(normalized.items()):
        examples.extend(
            _unique_cohort(
                pair_count=pair_count,
                seed=seed + 10_007 * hops,
                split=f"{split}_hop_{hops}",
                min_hops=hops,
                max_hops=hops,
                node_label_pool_size=node_label_pool_size,
                forbidden_raw_texts=forbidden_raw_texts,
            )
        )
    random.Random(seed + 90_001).shuffle(examples)
    return tuple(examples)


def partition_mechanism_tuning_shards(
    examples: Sequence[MatchedTopologyExample],
) -> dict[str, tuple[MatchedTopologyExample, ...]]:
    """Recover independently generated selection shards from their split names."""

    grouped: dict[str, list[MatchedTopologyExample]] = {}
    for example in examples:
        match = re.search(r"mechanism_tuning_shard_(\d+)", example.base.split)
        shard = f"shard_{int(match.group(1))}" if match else "shard_0"
        grouped.setdefault(shard, []).append(example)
    return {name: tuple(values) for name, values in sorted(grouped.items())}


def build_dataset_bundle(
    config: Mapping[str, object],
    seed: int,
    *,
    forbidden_raw_texts: set[str] | None = None,
) -> DatasetBundle:
    """Build disjoint train/tune/validation and preregistered test panels."""

    train_pairs = int(config["train_pairs"])
    validation_pairs = int(config["validation_pairs"])
    tuning_pairs = int(config.get("tuning_pairs", max(4, validation_pairs // 2)))
    tuning_unsupported_pairs = int(
        config.get("tuning_unsupported_pairs", max(2, validation_pairs // 6))
    )
    pairs_per_hop = int(config["pairs_per_hop"])
    context_pairs = int(config["context_pairs"])
    train_min_hops = int(config["train_min_hops"])
    train_max_hops = int(config["train_max_hops"])
    node_label_pool_size = int(config.get("node_label_pool_size", 97))
    training_distractor_count = int(config.get("training_distractor_tokens", 16))
    train_unsupported_pairs = int(config.get("train_unsupported_pairs", max(2, train_pairs // 8)))

    seen_supported_raw = forbidden_raw_texts if forbidden_raw_texts is not None else set()

    def unique_cohort(**kwargs: object) -> tuple[MatchedTopologyExample, ...]:
        return _unique_cohort(
            forbidden_raw_texts=seen_supported_raw,
            node_label_pool_size=node_label_pool_size,
            **kwargs,
        )

    training_hop_pair_counts = config.get("training_hop_pair_counts")
    if isinstance(training_hop_pair_counts, Mapping):
        train = _balanced_training_cohort(
            pair_counts_by_hop=training_hop_pair_counts,
            seed=seed,
            split="train",
            node_label_pool_size=node_label_pool_size,
            forbidden_raw_texts=seen_supported_raw,
        )
        if len(train) != 2 * train_pairs:
            raise ValueError(
                "training_hop_pair_counts must sum to the configured train_pairs"
            )
    else:
        train = unique_cohort(
            pair_count=train_pairs,
            seed=seed,
            split="train",
            min_hops=train_min_hops,
            max_hops=train_max_hops,
        )
    unsupported_train = make_unsupported(
        unique_cohort(
            pair_count=train_unsupported_pairs,
            seed=seed + 3,
            split="train_unsupported",
            min_hops=train_min_hops,
            max_hops=train_max_hops,
        )
    )
    train = append_training_distractors((*train, *unsupported_train), training_distractor_count)
    id_hops = int(config.get("id_hops", train_min_hops))
    tuning_supported = unique_cohort(
        pair_count=tuning_pairs,
        seed=seed + 101,
        split="tuning",
        min_hops=id_hops,
        max_hops=id_hops,
    )
    tuning_unsupported = make_unsupported(
        unique_cohort(
            pair_count=tuning_unsupported_pairs,
            seed=seed + 151,
            split="tuning_unsupported",
            min_hops=id_hops,
            max_hops=id_hops,
        )
    )
    tuning = (*tuning_supported, *tuning_unsupported)
    mechanism_tuning_pairs_per_scenario = int(
        config.get("mechanism_tuning_pairs_per_scenario", 2)
    )
    mechanism_tuning_hops = tuple(
        int(value) for value in config.get("mechanism_tuning_hops", (8, 16))
    )
    mechanism_tuning_shard_count = int(config.get("mechanism_tuning_shard_count", 1))
    if mechanism_tuning_shard_count < 1:
        raise ValueError("mechanism_tuning_shard_count must be positive")
    mechanism_tuning = tuple(
        example
        for shard_index in range(mechanism_tuning_shard_count)
        for hops in mechanism_tuning_hops
        for scenario_index, scenario in enumerate(_MECHANISM_TUNING_SCENARIOS)
        for example in unique_cohort(
            pair_count=mechanism_tuning_pairs_per_scenario,
            seed=seed + 1700 + 100_003 * shard_index + 101 * hops + scenario_index,
            split=(
                f"mechanism_tuning_shard_{shard_index}_hop_{hops}_{scenario}"
            ),
            min_hops=hops,
            max_hops=hops,
            scenarios=(scenario,),
        )
    )
    validation = unique_cohort(
        pair_count=validation_pairs,
        seed=seed + 211,
        split="validation",
        min_hops=id_hops,
        max_hops=id_hops,
    )

    hop_cohorts = {
        f"hop_{hops}": unique_cohort(
            pair_count=pairs_per_hop,
            seed=seed + 1000 + hops,
            split=f"test_hop_{hops}",
            min_hops=hops,
            max_hops=hops,
        )
        for hops in tuple(int(value) for value in config["test_hops"])
    }

    context_cohorts: dict[str, tuple[MatchedTopologyExample, ...]] = {}
    for target in tuple(int(value) for value in config["context_lengths"]):
        base = unique_cohort(
            pair_count=context_pairs,
            seed=seed + 2000 + target,
            split=f"context_{target}",
            min_hops=4,
            max_hops=4,
        )
        context_cohorts[f"context_{target}_hop_4"] = add_context_distractors(base, target)
    for target in tuple(int(value) for value in config.get("interaction_context_lengths", ())):
        base = unique_cohort(
            pair_count=context_pairs,
            seed=seed + 3000 + target,
            split=f"context_{target}_hop_24",
            min_hops=24,
            max_hops=24,
        )
        context_cohorts[f"context_{target}_hop_24"] = add_context_distractors(base, target)

    unsupported_cohorts = {
        f"unsupported_hop_{hops}": make_unsupported(
            unique_cohort(
                pair_count=max(4, pairs_per_hop // 2),
                seed=seed + 4000 + hops,
                split=f"unsupported_hop_{hops}",
                min_hops=hops,
                max_hops=hops,
            )
        )
        for hops in tuple(int(value) for value in config["unsupported_hops"])
    }
    tokenizer = protocol_tokenizer(node_label_pool_size=node_label_pool_size)
    return DatasetBundle(
        train=tuple(train),
        tuning=tuning,
        mechanism_tuning=mechanism_tuning,
        validation=validation,
        hop_cohorts=hop_cohorts,
        context_cohorts=context_cohorts,
        unsupported_cohorts=unsupported_cohorts,
        tokenizer=tokenizer,
    )


__all__ = [
    "DatasetBundle",
    "add_context_distractors",
    "append_training_distractors",
    "audit_label_contract",
    "assert_shared_raw_input_contract",
    "build_dataset_bundle",
    "canonicalize_examples",
    "make_unsupported",
    "partition_mechanism_tuning_shards",
    "protocol_tokenizer",
]
