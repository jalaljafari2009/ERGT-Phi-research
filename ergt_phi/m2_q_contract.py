"""Load and validate the frozen M2-Q v2 architecture contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA = "ergt-phi-m2-q-anchor-config-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_m2_q_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Reject contract drift that would invalidate the approved M2-Q v2 design."""
    _require(config.get("schema") == SCHEMA, "unknown M2-Q contract schema")
    _require(config.get("revision") == "m2-q-v2", "unexpected M2-Q revision")
    _require(config.get("stage") == "M2" and config.get("research_track") == "A",
             "M2-Q v2 belongs to track A / M2")

    public = config["public_input_contract"]
    _require(public["inputs"] == ["raw_token_ids", "attention_mask"],
             "public input contract must remain raw tokens plus mask")
    _require(public["gold_or_answer_inputs_allowed"] is False,
             "gold or answer inputs are forbidden")

    bypass = config["disabled_bypass"]
    _require(bypass["applies_to"] == ["public_forward", "integrated_active_forward"],
             "zero-coupling bypass scope must cover both native forward entrypoints")
    _require(bypass["checked_before_prepass"] is True, "bypass must precede prepass")
    _require(bypass["reference_prepass_calls"] == 0 and bypass["proposal_probe_calls"] == 0,
             "disabled path must not run the prepass or probe")
    _require(bypass["extra_rng_consumption"] is False,
             "disabled path must not consume extra RNG")

    prepass = config["reference_prepass"]
    _require(prepass["model_role"] == "independent_frozen_context_provider",
             "reference prepass must use an independent frozen provider")
    _require(prepass["evaluation_mode"] is True and prepass["parameters_require_grad"] is False,
             "reference provider must be eval and frozen")
    _require(prepass["optimizer_present"] is False and prepass["load_strict"] is True,
             "reference provider must have no optimizer and load strictly")
    _require(prepass["maximum_calls_per_forward"] == 1 and
             prepass["proposal_probe_calls_per_prepass"] == 1,
             "v2 permits one prepass and one proposal probe per forward")
    _require(prepass["stops_before_candidate_answer_solving"] is True and
             prepass["raw_inputs_only"] is True and prepass["proposal_tensors_detached"] is True,
             "prepass information boundary changed")

    anchor = config["anchor"]
    _require(anchor["builds_per_forward"] == 1 and anchor["fixed_across_outer_steps"] is True,
             "anchor must be built once and fixed")
    _require(anchor["rebuild_from_active_q_forbidden"] is True,
             "active Q must not rebuild the v2 anchor")
    _require(anchor["gold_used_only_for_loss_after_raw_feature_cache"] is True,
             "gold may only index the post-cache training loss")

    active = config["active_path"]
    _require(active["starts_from_fresh_native_state"] is True and
             active["initial_lagged_q"] == "empty",
             "active path must start fresh with empty lagged Q")
    _require(active["prepass_q_written_to_lagged_memory"] is False,
             "prepass Q cannot seed lagged memory")
    _require(active["first_outer_step_is_native"] is True and
             active["active_step_proposal_consumed_at"] == "next_outer_step",
             "lagged causal schedule changed")
    _require(active["active_model_and_reference_model_share_mutable_state"] is False,
             "active and reference models cannot share mutable state")

    scope = config["m2_shadow_scope"]
    _require(scope["entrypoint"] == "explicit_offline_calibration_runner_not_public_forward",
             "M2 shadow measurements must use an explicit offline entrypoint")
    _require(scope["may_observe_prepass_features_while_coupling_is_zero"] is True and
             scope["native_return_value_replaced_or_modified"] is False,
             "offline M2 may observe shadow features but cannot change native output")
    _require(scope["phase_mode"] == "shadow" and scope["coupling"] == 0.0 and scope["nu"] == 0.0,
             "M2-Q v2 is a zero-coupling linear shadow calibration")
    _require(scope["native_geometry_activated"] is False and
             scope["native_answer_head_used"] is False,
             "M2 cannot activate native geometry or answer execution")

    readiness = config["calibration"]["readiness"]
    expected = {
        "balanced_accuracy_min": 0.7,
        "per_class_recall_min": 0.5,
        "ce_reduction_min": 0.05,
        "offset_separation_radians_min": 1.0,
        "phase_resultant_max": 0.98,
        "consecutive_windows": 2,
    }
    _require(readiness == expected, "registered M2 readiness thresholds changed")
    _require(config["calibration"]["selection"].startswith("freeze_first_checkpoint"),
             "first qualified checkpoint must be frozen")

    resources = config["resources"]
    _require(resources["reference_prepasses_per_forward"] == 1 and
             resources["proposal_probes_per_prepass"] == 1,
             "prepass cost accounting changed")
    _require(resources["include_prepass_cache_and_probe_in_reported_cost"] is True,
             "complete prepass cost must be reported")
    _require(resources["maximum_runtime_minutes"] is None and
             resources["maximum_runtime_must_be_pinned_in_experiment_protocol"] is True,
             "runtime cap belongs in the preregistered experiment protocol")

    claims = config["claims"]
    _require(not any(claims.values()), "contract creation cannot make a scientific or phase claim")
    return config


def load_m2_q_contract(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("M2-Q contract must be a JSON object")
    return validate_m2_q_contract(config)
