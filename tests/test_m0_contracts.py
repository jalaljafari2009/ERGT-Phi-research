import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from ergt_phi.runtime import ROOT, REFERENCE
from ergt_phi.fixtures import example_batch, model_config, seed_all
from ergt_phi.native_steps import SteppedNative
from ergt_phi.training import update_native
from ergt_reviewer.native_solver import NativeGeometricBoundaryModel
from ergt_reviewer.suite_v9 import _native_runtime_invariance_audit


def test_reference_files_remain_byte_identical():
    manifest = json.loads((ROOT / "manifests/reference.json").read_text())
    original = Path(manifest["original_root"])
    for relative, expected in manifest["files"].items():
        assert hashlib.sha256((REFERENCE / relative).read_bytes()).hexdigest() == expected
        if original.exists():
            assert hashlib.sha256((original / relative).read_bytes()).hexdigest() == expected


def test_native_batch_and_padding_invariance():
    seed_all()
    batch, tokenizer = example_batch(hops=2)
    original = NativeGeometricBoundaryModel(model_config(tokenizer))
    stepped = SteppedNative(original.config)
    stepped.load_state_dict(original.state_dict())
    for model in (original, stepped):
        result = _native_runtime_invariance_audit(model, batch.examples, tokenizer,
            device=torch.device("cpu"), maximum_tokens=128)
        assert result["contract_pass"], result


def test_geometry_preview_does_not_mutate_old_field():
    seed_all()
    batch, tokenizer = example_batch()
    model = SteppedNative(model_config(tokenizer))
    semantic, identity = model.raw_input_adapter(**batch.model_inputs())
    state = model.substrate.seed_native_state(semantic, batch.base.attention_mask, identity_fibre=identity)
    for _ in range(2):
        before = {key: value.detach().clone() for key, value in state.items() if isinstance(value, torch.Tensor)}
        losses_before = list(state["bulk_losses"])
        observation = model.substrate.observe_native_state(state)
        preview = model.substrate.preview_native_geometry(state, observation)
        for key, expected in before.items():
            assert torch.equal(expected, state[key]), key
        next_state = model.substrate.advance_native_field(state, preview)
        assert len(state["bulk_losses"]) == len(losses_before)
        for key, expected in before.items():
            assert torch.equal(expected, state[key]), key
        state = next_state


def test_freeze_prevents_any_optimizer_or_rng_change():
    batch, tokenizer = example_batch()
    model = SteppedNative(model_config(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0025)
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    progress = {"frozen": True, "global_step": 100, "data_cursor": 200}
    with pytest.raises(RuntimeError, match="after final freeze"):
        update_native(model, optimizer, batch, progress=progress)
    for key, expected in before.items():
        assert torch.equal(expected, model.state_dict()[key])
    assert not optimizer.state
    assert torch.equal(rng, torch.get_rng_state())
    assert progress == {"frozen": True, "global_step": 100, "data_cursor": 200}
