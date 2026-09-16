"""Research optimizer boundary with an explicit final-freeze guard."""
from .runtime import activate_reference
activate_reference()
import torch
from ergt_reviewer.native_solver import native_geometric_training_loss


def update_native(model, optimizer, batch, *, progress, scheduler=None):
    if progress.get("frozen", False):
        raise RuntimeError("Optimizer updates after final freeze are forbidden")
    model.train()
    config = model.baseline.config if hasattr(model, "baseline") else model.config
    outputs = model(**batch.model_inputs())
    loss, parts = native_geometric_training_loss(outputs, batch,
        config=config, teacher_weight=1.0, answer_weight=.15)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    progress["global_step"] = int(progress.get("global_step", 0)) + 1
    progress["data_cursor"] = int(progress.get("data_cursor", 0)) + len(batch.examples)
    return loss.detach(), parts
