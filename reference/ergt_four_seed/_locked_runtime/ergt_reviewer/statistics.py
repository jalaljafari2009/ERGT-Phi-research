"""Small dependency-free statistical tests for paired confirmatory outputs."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    discordant = int(b + c)
    if discordant == 0:
        return 1.0
    tail = min(int(b), int(c))
    probability = sum(
        math.comb(discordant, value) * (0.5**discordant)
        for value in range(tail + 1)
    )
    return min(1.0, 2.0 * probability)


def paired_comparison(
    native_rows: Sequence[Mapping[str, Any]],
    transformer_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 2_000,
    seed: int = 1337,
    replicate_key: str | None = None,
) -> dict[str, float]:
    def row_key(row: Mapping[str, Any]) -> str:
        replicate = str(row.get(replicate_key, "single")) if replicate_key else "single"
        return f"{replicate}:{row['example_id']}"

    native = {row_key(row): row for row in native_rows}
    transformer = {row_key(row): row for row in transformer_rows}
    identifiers = sorted(native.keys() & transformer.keys())
    if not identifiers:
        raise ValueError("paired comparison has no shared example identifiers")
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in identifiers:
        replicate = str(native[key].get(replicate_key, "single")) if replicate_key else "single"
        groups[(replicate, str(native[key]["pair_id"]))].append(key)
    group_ids = tuple(sorted(groups))
    group_differences = {
        group: sum(
            float(bool(native[key]["correct"]))
            - float(bool(transformer[key]["correct"]))
            for key in groups[group]
        )
        / max(1, len(groups[group]))
        for group in group_ids
    }
    native_pair_exact = {
        group: len(groups[group]) >= 1
        and all(bool(native[key]["correct"]) for key in groups[group])
        for group in group_ids
    }
    transformer_pair_exact = {
        group: len(groups[group]) >= 1
        and all(bool(transformer[key]["correct"]) for key in groups[group])
        for group in group_ids
    }
    b = sum(
        native_pair_exact[group] and not transformer_pair_exact[group]
        for group in group_ids
    )
    c = sum(
        transformer_pair_exact[group] and not native_pair_exact[group]
        for group in group_ids
    )
    groups_by_replicate: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for group in group_ids:
        groups_by_replicate[group[0]].append(group)
    replicate_ids = tuple(sorted(groups_by_replicate))
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(max(1, int(bootstrap_samples))):
        sampled_replicates = [rng.choice(replicate_ids) for _ in replicate_ids]
        sampled_groups: list[tuple[str, str]] = []
        for replicate in sampled_replicates:
            available = groups_by_replicate[replicate]
            sampled_groups.extend(rng.choice(available) for _ in available)
        differences.append(
            sum(group_differences[group] for group in sampled_groups)
            / max(1, len(sampled_groups))
        )
    ordered = sorted(differences)
    lower = ordered[int(0.025 * (len(ordered) - 1))]
    upper = ordered[int(0.975 * (len(ordered) - 1))]
    observed = sum(group_differences.values()) / max(1, len(group_differences))
    return {
        "paired_examples": float(len(identifiers)),
        "paired_counterfactual_clusters": float(len(group_ids)),
        "independent_data_replicates": float(len(replicate_ids)),
        "native_minus_transformer_accuracy": observed,
        "bootstrap_ci_low": lower,
        "bootstrap_ci_high": upper,
        "mcnemar_native_only_correct": float(b),
        "mcnemar_transformer_only_correct": float(c),
        "mcnemar_exact_pvalue": exact_mcnemar_pvalue(b, c),
        "statistical_unit_is_counterfactual_pair": 1.0,
        "hierarchical_data_seed_bootstrap": float(replicate_key is not None),
    }


def linear_slope(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    x_mean = sum(x for x, _ in points) / len(points)
    y_mean = sum(y for _, y in points) / len(points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0.0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator


__all__ = ["exact_mcnemar_pvalue", "linear_slope", "paired_comparison"]
