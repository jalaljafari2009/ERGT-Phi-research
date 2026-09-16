"""V9 repaired confirmation with independent ERGT and baseline statuses."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .suite_v9 import PACKAGE_ROOT, run_reviewer_suite
from .v8_confirmation import (
    PACKAGED_QUALIFICATION_MANIFEST,
    validate_qualification_manifest,
)

SCHEMA_VERSION = "ergt-geometric-long-horizon-study-v1"
LOCK_SCHEMA_VERSION = "ergt-reviewer-v9-execution-lock-v1"
EXECUTION_LOCK = PACKAGE_ROOT / "manifests" / "v9_execution_lock.json"
PROFILE_BY_MODE = {
    "single_seed_validation": "repair_pilot_v9",
    "four_seed_confirmation": "paper_final_v9",
    # Backward-compatible aliases for previously distributed notebooks.
    "repair_pilot": "repair_pilot_v9",
    "paper_full": "paper_final_v9",
}

DEVELOPMENT_OPTIMIZATION_SEEDS = frozenset((1337, 2027, 7331, 9173, 10429, 12011))
DEVELOPMENT_DATA_SEEDS = frozenset((4103, 5209, 6317, 14033, 15149, 16301))


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_source_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def validate_v9_execution_lock(
    profile: str,
    qualification_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    lock = _load_json(EXECUTION_LOCK)
    record = dict(lock)
    observed_lock_hash = str(record.pop("lock_sha256", ""))
    expected_lock_hash = _payload_sha256(record)
    expected_files = dict(record.get("files") or {})
    observed_files = {
        relative: _canonical_source_sha256(PACKAGE_ROOT / relative) for relative in expected_files
    }
    mismatched_files = sorted(
        relative
        for relative, expected in expected_files.items()
        if observed_files.get(relative) != expected
    )
    selected_manifest_hash = str(qualification_manifest.get("manifest_sha256", ""))
    checks = {
        "schema_matches": record.get("schema_version") == LOCK_SCHEMA_VERSION,
        "lock_is_immutable": bool(record.get("immutable", False)),
        "lock_hash_matches": bool(observed_lock_hash) and observed_lock_hash == expected_lock_hash,
        "locked_files_are_unchanged": not mismatched_files,
        "profile_is_locked": profile in set(record.get("profiles") or ()),
        "qualification_manifest_is_exact": selected_manifest_hash
        == record.get("qualification_manifest_sha256"),
        "baseline_failure_is_nonblocking_for_ergt": bool(
            dict(record.get("claim_policy") or {}).get(
                "baseline_failure_is_nonblocking_for_ergt", False
            )
        ),
        "one_hop_id_is_not_native_freeze_condition": bool(
            dict(record.get("checkpoint_contract") or {}).get("one_hop_id_is_reporting_only", False)
        ),
        "multihop_readiness_required": bool(
            dict(record.get("checkpoint_contract") or {}).get(
                "independent_2_8_hop_readiness", False
            )
        ),
        "minimum_exposure_required": int(
            dict(record.get("checkpoint_contract") or {}).get("minimum_pair_group_exposure", 0)
        )
        >= 8,
        "four_seed_results_are_retained": bool(
            dict(record.get("claim_policy") or {}).get("all_four_seed_results_retained", False)
        ),
        "matched_convergence_requires_three_seeds": int(
            dict(record.get("claim_policy") or {}).get(
                "matched_convergence_minimum_seed_count", 0
            )
        )
        == 3,
        "matched_seed_selection_uses_id_only": bool(
            dict(record.get("claim_policy") or {}).get(
                "matched_convergence_seed_selection_uses_id_only", False
            )
        ),
    }
    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "pass": all(checks.values()),
        "checks": checks,
        "observed_lock_sha256": observed_lock_hash,
        "recomputed_lock_sha256": expected_lock_hash,
        "expected_source_sha256": expected_files,
        "observed_source_sha256": observed_files,
        "mismatched_files": mismatched_files,
        "checkpoint_contract": dict(record.get("checkpoint_contract") or {}),
        "claim_policy": dict(record.get("claim_policy") or {}),
    }


def v9_seed_audit(
    config: Mapping[str, Any],
    qualification_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    optimization_seeds = {int(value) for value in config["seeds"]}
    data_seeds = {int(value) for value in config["data_seeds"]}
    developmental = bool(config.get("development_repair_pilot", False))
    checks = {
        "optimization_and_data_seeds_disjoint": optimization_seeds.isdisjoint(data_seeds),
        "qualification_optimization_seed_excluded": int(
            qualification_manifest.get("optimization_seed", -1)
        )
        not in optimization_seeds,
        "qualification_data_seed_excluded": int(qualification_manifest.get("data_seed", -1))
        not in data_seeds,
    }
    if not developmental:
        checks.update(
            {
                "four_optimization_seeds": len(optimization_seeds) == 4,
                "four_data_seeds": len(data_seeds) == 4,
                "optimization_seeds_fresh": optimization_seeds.isdisjoint(
                    DEVELOPMENT_OPTIMIZATION_SEEDS
                ),
                "data_seeds_fresh": data_seeds.isdisjoint(DEVELOPMENT_DATA_SEEDS),
            }
        )
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "development_repair_pilot": developmental,
        "development_seed_reuse_is_not_confirmatory": developmental,
        "optimization_seeds": sorted(optimization_seeds),
        "data_seeds": sorted(data_seeds),
    }


def run_v9_confirmation(
    *,
    mode: str = "single_seed_validation",
    qualification_manifest_path: str | Path | None = None,
    device: str | torch.device | None = "auto",
    output_root: str | Path | None = None,
    run_id: str | None = None,
    resume: bool = True,
    copy_outputs_to_downloads: bool = True,
) -> dict[str, Any]:
    if mode not in PROFILE_BY_MODE:
        raise ValueError(f"mode must be one of {sorted(PROFILE_BY_MODE)}")
    profile = PROFILE_BY_MODE[mode]
    manifest_path = Path(qualification_manifest_path or PACKAGED_QUALIFICATION_MANIFEST)
    manifest = _load_json(manifest_path)
    qualification_audit = validate_qualification_manifest(manifest)
    if not bool(qualification_audit["pass"]):
        failed = [name for name, passed in qualification_audit["checks"].items() if not passed]
        raise RuntimeError("V9 baseline qualification manifest is invalid: " + ", ".join(failed))

    config = _load_json(PACKAGE_ROOT / "configs" / f"{profile}.json")
    lock_audit = validate_v9_execution_lock(profile, manifest)
    if not bool(lock_audit["pass"]):
        failed = [name for name, passed in lock_audit["checks"].items() if not passed]
        raise RuntimeError("V9 execution lock is invalid: " + ", ".join(failed))
    seed_audit = v9_seed_audit(config, manifest)
    if not bool(seed_audit["pass"]):
        failed = [name for name, passed in seed_audit["checks"].items() if not passed]
        raise RuntimeError("V9 seed audit failed: " + ", ".join(failed))

    selected = dict(manifest.get("selected_candidate") or {})
    recipe = dict(selected.get("recipe") or {})
    return run_reviewer_suite(
        profile=profile,
        device=device,
        output_root=output_root,
        run_id=run_id or f"ergt_geometric_long_horizon_{mode}",
        resume=resume,
        copy_outputs_to_downloads=copy_outputs_to_downloads,
        schema_version=SCHEMA_VERSION,
        config_overrides={
            "qualified_transformer_recipe": recipe,
            "qualification_manifest_audit": qualification_audit,
            "fresh_seed_audit": seed_audit,
            "final_execution_lock_audit": lock_audit,
            "qualification_manifest_source": str(manifest_path),
            "v9_mode": mode,
        },
        baseline_qualification_manifest=manifest,
    )


def run_geometric_reasoning_study(**kwargs: Any) -> dict[str, Any]:
    """Publication-facing entry point for the locked ERGT/Transformer study."""

    return run_v9_confirmation(**kwargs)


__all__ = [
    "EXECUTION_LOCK",
    "LOCK_SCHEMA_VERSION",
    "PROFILE_BY_MODE",
    "SCHEMA_VERSION",
    "run_geometric_reasoning_study",
    "run_v9_confirmation",
    "v9_seed_audit",
    "validate_v9_execution_lock",
]
