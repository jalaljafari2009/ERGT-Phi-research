"""Atomic, contract-bound complete research checkpoints for trusted local runs."""
from pathlib import Path
import copy
import hashlib
import json
import random
import numpy as np
import torch


def fingerprint(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA RNG topology changed")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def capture(model, optimizer, *, contract, progress, scheduler=None, scaler=None):
    return copy.deepcopy({
        "schema": "ergt-phi-m0-full-state-v1", "contract": contract,
        "contract_sha256": fingerprint(contract), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "requires_grad": {k: p.requires_grad for k, p in model.named_parameters()},
        "module_training": {k: m.training for k, m in model.named_modules()},
        "rng": rng_state(), "progress": progress,
    })


def save(path, model, optimizer, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = capture(model, optimizer, **kwargs)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return state


def restore(state, model, optimizer, *, contract, scheduler=None, scaler=None):
    # Validate the full contract before any mutation. Restore only trusted checkpoints.
    if state.get("schema") != "ergt-phi-m0-full-state-v1":
        raise ValueError("unknown checkpoint schema")
    if state["contract_sha256"] != fingerprint(state["contract"]) or state["contract_sha256"] != fingerprint(contract):
        raise ValueError("checkpoint contract mismatch")
    for name, object_ in (("scheduler", scheduler), ("scaler", scaler)):
        if (state[name] is None) != (object_ is None):
            raise ValueError(f"checkpoint {name} mismatch")
    parameters = dict(model.named_parameters())
    if set(parameters) != set(state["requires_grad"]):
        raise ValueError("checkpoint parameter names mismatch")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    optimizer.zero_grad(set_to_none=True)  # checkpoints are at completed optimizer-step boundaries
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    for name, parameter in parameters.items():
        parameter.requires_grad_(state["requires_grad"][name])
    for name, module in model.named_modules():
        module.training = state["module_training"][name]
    restore_rng(state["rng"])
    return copy.deepcopy(state["progress"])


def load(path, model, optimizer, **kwargs):
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    return restore(state, model, optimizer, **kwargs)
