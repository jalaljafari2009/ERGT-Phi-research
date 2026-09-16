import inspect
import json
from pathlib import Path

from ergt_four_seed.runtime import LOCKED_RUNTIME_ROOT, activate_locked_runtime

activate_locked_runtime()
from ergt_reviewer.training_v9 import _train_native_v9


def test_frozen_checkpoint_contract_is_locked_and_implemented():
    lock = json.loads((LOCKED_RUNTIME_ROOT / "manifests/v9_execution_lock.json").read_text())
    contract = lock["checkpoint_contract"]
    assert contract["immediate_freeze_after_readiness"] is True
    assert contract["zero_teacher_optimizer_steps"] == 0
    assert contract["fallback_to_unselected_final_state_for_verdict"] is False
    assert contract["final_12_32_hop_panels_used_for_selection"] is False
    source = inspect.getsource(_train_native_v9)
    assert "teacher = 0.0 if field_ready else 1.0" in source
    assert "if teacher > 0.0:" in source
    assert source.index("if teacher > 0.0:") < source.index("optimizer.step()")
    assert '"final_12_32_hop_evaluations_during_training": 0' in source
    assert 'evaluation_state_status = "unselected_final_training_state_diagnostic_only"' in source
