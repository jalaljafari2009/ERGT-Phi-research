"""Validate the selected Transformer recipe without shipping development code."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_SOURCE_ROOT = PACKAGE_ROOT / "contracts" / "qualification_source"
QUALIFICATION_SCHEMA_VERSION = "ergt-reviewer-baseline-qualification-v8.1"
PACKAGED_QUALIFICATION_MANIFEST = (
    PACKAGE_ROOT / "manifests" / "v8_1_selected_baseline_manifest.json"
)
QUALIFICATION_SOURCE_FILES = (
    "ergt_reviewer/baseline.py",
    "ergt_reviewer/baseline_qualification_v8.py",
    "ergt_reviewer/fair_data.py",
    "ergt_reviewer/training.py",
    "configs/baseline_qualification_v8.json",
)


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_qualification_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(manifest)
    observed_hash = str(record.pop("manifest_sha256", ""))
    expected_hash = _payload_sha256(record)
    selected = dict(record.get("selected_candidate") or {})
    recipe = dict(selected.get("recipe") or {})
    metrics = dict(selected.get("qualification_metrics") or {})
    parameter_audit = dict(selected.get("parameter_audit") or {})
    forbidden = set(record.get("forbidden_components") or ())
    registered_hashes = dict(record.get("qualification_source_sha256") or {})
    current_hashes = {
        relative: hashlib.sha256(
            (QUALIFICATION_SOURCE_ROOT / relative).read_bytes()
        ).hexdigest()
        for relative in QUALIFICATION_SOURCE_FILES
    }
    required_forbidden = {
        "compiler", "generic_executor", "ERGT_graph", "geometry_side_input",
        "final_horizon_selection",
    }
    checks = {
        "schema_matches": record.get("schema_version") == QUALIFICATION_SCHEMA_VERSION,
        "protocol_revision_matches": record.get("protocol_revision")
        == "v8.1_v7_warmup_restoration",
        "qualification_passed": bool(record.get("qualification_pass", False)),
        "selection_is_development_only": bool(record.get("selection_is_development_only", False)),
        "final_panels_remained_closed": int(record.get("final_12_32_hop_panels_evaluated", -1)) == 0,
        "manifest_hash_matches": bool(observed_hash) and observed_hash == expected_hash,
        "selected_recipe_present": bool(recipe),
        "selected_candidate_matches_recipe": bool(selected.get("candidate_id"))
        and selected.get("candidate_id") == recipe.get("candidate_id"),
        "selected_loss_mode_registered": recipe.get("loss_mode") == "matched_auxiliary",
        "selected_depth_registered": int(recipe.get("n_layers", 0)) == 4,
        "selected_batch_size_registered": int(recipe.get("batch_size", 0)) == 16,
        "selected_warmup_batch_size_registered": int(recipe.get("warmup_batch_size", 0)) == 8,
        "selected_warmup_budget_registered": int(recipe.get("warmup_steps", 0)) == 1400,
        "selected_warmup_stream_registered": int(recipe.get("warmup_sampling_seed_offset", 0)) == 701,
        "selected_curriculum_registered": [
            int(stage.get("max_hop", 0)) for stage in recipe.get("curriculum_stages", ())
        ] == [2, 4, 8],
        "forbidden_components_complete": required_forbidden <= forbidden,
        "fresh_final_profile_registered": record.get("fresh_final_profile") == "paper_final_v8",
        "selected_train_id_converged": float(metrics.get("train_id_supported_accuracy", 0.0)) >= 0.90,
        "selected_tuning_converged": float(metrics.get("tuning_supported_accuracy", 0.0)) >= 0.90
        and float(metrics.get("tuning_pair_exact", 0.0)) >= 0.90,
        "selected_raw_disjoint_validation_converged":
        float(metrics.get("validation_supported_accuracy", 0.0)) >= 0.90
        and float(metrics.get("validation_pair_exact", 0.0)) >= 0.90
        and float(metrics.get("validation_unsupported_error_rate", 1.0)) <= 0.10,
        "selected_inference_parameters_matched": bool(parameter_audit.get("within_ten_percent", False))
        and float(parameter_audit.get("inference_active_relative_parameter_difference", 1.0)) <= 0.10,
        "final_profile_protocol_revision_matches": _load_json(
            QUALIFICATION_SOURCE_ROOT / "configs" / "paper_final_v8.json"
        ).get("baseline_qualification_protocol_revision") == record.get("protocol_revision"),
        "qualification_source_is_unchanged": registered_hashes == current_hashes,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "observed_manifest_sha256": observed_hash,
        "recomputed_manifest_sha256": expected_hash,
        "selected_recipe": recipe,
        "registered_source_sha256": registered_hashes,
        "current_source_sha256": current_hashes,
        "optimization_seed": int(record.get("optimization_seed", -1)),
        "data_seed": int(record.get("data_seed", -1)),
    }


def run_v8_final_confirmation(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("V8 development execution is not distributed; V9 uses only its locked recipe.")


__all__ = ["PACKAGED_QUALIFICATION_MANIFEST", "run_v8_final_confirmation", "validate_qualification_manifest"]
