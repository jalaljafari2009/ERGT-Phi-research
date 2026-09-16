import copy
import os
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
import pytest
import torch
from ergt_phi.checkpoint import capture, restore, save, load
from ergt_phi.fixtures import example_batch, model_config, seed_all, training_step
from ergt_phi.native_steps import SteppedNative


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_resume_native_in_new_process(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/resume_worker.py"
    for mode in ("continuous", "first", "resume"):
        completed = subprocess.run([sys.executable, "-B", str(script), mode, str(tmp_path)],
            capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=180)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    equal(torch.load(tmp_path / "continuous.pt", weights_only=False),
        torch.load(tmp_path / "resume.pt", weights_only=False))


def test_complete_rollback_restores_state_and_rng(tmp_path):
    seed_all()
    batch, tokenizer = example_batch()
    model = SteppedNative(model_config(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0025)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.9)
    training_step(model, optimizer, batch, scheduler)
    contract = {"stage": "M0", "source": "fixture-v1"}
    progress = {"global_step": 1, "data_cursor": 2, "stage": "M0", "frozen": False}
    checkpoint = save(tmp_path / "state.pt", model, optimizer,
        contract=contract, progress=progress, scheduler=scheduler)
    expected_draws = (random.random(), np.random.rand(), torch.rand(3))
    training_step(model, optimizer, batch, scheduler)
    model.eval()
    next(model.parameters()).requires_grad_(False)
    restored = load(tmp_path / "state.pt", model, optimizer, contract=contract, scheduler=scheduler)
    equal(capture(model, optimizer, contract=contract, progress=restored, scheduler=scheduler), checkpoint)
    equal((random.random(), np.random.rand(), torch.rand(3)), expected_draws)


def test_incompatible_contract_rejected_before_model_mutation():
    batch, tokenizer = example_batch()
    model = SteppedNative(model_config(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0025)
    state = capture(model, optimizer, contract={"source": "v1"}, progress={"global_step": 0})
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="contract mismatch"):
        restore(state, model, optimizer, contract={"source": "v2"})
    equal(before, model.state_dict())
