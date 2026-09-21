from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ergt_phi.m2_q_contract import load_m2_q_contract, validate_m2_q_contract


ROOT = Path(__file__).resolve().parents[1]


def test_registered_m2_q_contract_is_valid() -> None:
    config = load_m2_q_contract(ROOT / "configs" / "m2_q_v2.json")
    assert config["status"] == "contract_frozen_implementation_pending"


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("disabled_bypass", "checked_before_prepass", False, "bypass must precede prepass"),
        ("active_path", "prepass_q_written_to_lagged_memory", True,
         "prepass Q cannot seed lagged memory"),
        ("m2_shadow_scope", "native_return_value_replaced_or_modified", True,
         "offline M2 may observe shadow features"),
    ],
)
def test_contract_rejects_causal_or_zero_effect_drift(
    section: str, field: str, value: object, message: str
) -> None:
    config = deepcopy(load_m2_q_contract(ROOT / "configs" / "m2_q_v2.json"))
    config[section][field] = value
    with pytest.raises(ValueError, match=message):
        validate_m2_q_contract(config)
