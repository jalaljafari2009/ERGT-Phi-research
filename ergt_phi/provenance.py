"""Bind research training checkpoints to code, data, model and runtime policy."""
from dataclasses import asdict
import hashlib
import platform
import torch
from .runtime import ROOT
from .model import PhaseConfig


def source_hashes():
    paths = sorted((ROOT / "ergt_phi").glob("*.py"))
    paths += sorted((ROOT / "scripts").glob("*.py"))
    paths += sorted((ROOT / "reference/ergt_four_seed").rglob("*.py"))
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def training_contract(model, optimizer, *, data_fingerprint, stage="M0", scheduler=None):
    base = getattr(model, "baseline", model)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {
        "schema": "ergt-phi-training-contract-v1", "source_sha256": source_hashes(),
        "data_fingerprint": data_fingerprint, "model_config": asdict(base.config),
        "phase_config": asdict(getattr(model, "phase", PhaseConfig())), "stage": stage,
        "python": platform.python_version(), "torch": torch.__version__,
        "device": str(next(model.parameters()).device), "dtype": str(next(model.parameters()).dtype),
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "optimizer": type(optimizer).__qualname__,
        "optimizer_parameter_groups": [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups],
        "optimizer_initial_options": [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups],
        "scheduler": type(scheduler).__qualname__ if scheduler is not None else None,
        "native_loss_policy": {"teacher_weight": 1.0, "answer_weight": .15, "clip_grad_norm": 1.0},
        "checkpoint_boundary": "after complete optimizer step; no gradient accumulation",
    }
