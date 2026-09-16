import copy
import tempfile
from pathlib import Path

import torch

from ergt_four_seed.runtime import activate_locked_runtime

activate_locked_runtime()
from ergt_reviewer.training_v9 import _load_training_state, _save_training_state


def _step(model, optimizer, inputs, targets):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()


def test_interrupted_resume_matches_continuous_execution():
    torch.manual_seed(91)
    template = torch.nn.Linear(3, 2)
    initial = copy.deepcopy(template.state_dict())
    inputs = torch.tensor([[0.25, -0.5, 1.0], [1.5, 0.0, -0.75]])
    targets = torch.tensor([[0.1, 0.9], [0.3, -0.2]])

    continuous = torch.nn.Linear(3, 2)
    continuous.load_state_dict(initial)
    continuous_optimizer = torch.optim.AdamW(continuous.parameters(), lr=0.01)
    _step(continuous, continuous_optimizer, inputs, targets)
    _step(continuous, continuous_optimizer, inputs, targets)

    interrupted = torch.nn.Linear(3, 2)
    interrupted.load_state_dict(initial)
    interrupted_optimizer = torch.optim.AdamW(interrupted.parameters(), lr=0.01)
    _step(interrupted, interrupted_optimizer, inputs, targets)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "state.pt"
        _save_training_state(
            checkpoint, model=interrupted, optimizer=interrupted_optimizer, step=1,
            best_state=None, best_score=0.0, curves=[{"step": 1}], config={"test": True},
            run_hash="resume-contract", training_state={"field_ready": False},
        )
        resumed = torch.nn.Linear(3, 2)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
        loaded = _load_training_state(
            checkpoint, model=resumed, optimizer=resumed_optimizer,
            run_hash="resume-contract", device=torch.device("cpu"),
        )
        assert loaded[0] == 1
        _step(resumed, resumed_optimizer, inputs, targets)

    for name, value in continuous.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0.0, atol=0.0)
