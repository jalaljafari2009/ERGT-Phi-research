"""Small real-ERGT fixtures for engineering tests, never paper evidence."""
import random
import numpy as np
import torch
from .runtime import activate_reference
activate_reference()
from ergt_reviewer.fair_data_v9 import canonicalize_examples, protocol_tokenizer, make_unsupported
from ergt_reviewer.matched_data import build_matched_topology_examples, RawTokenInputContract, collate_matched_topology_examples
from ergt_reviewer.native_solver import ERGT43Config, NativeGeometricBoundaryModel, native_geometric_training_loss


def seed_all(seed=83):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def example_batch(hops=1, unsupported=False, pad=None):
    tokenizer = protocol_tokenizer(node_label_pool_size=17)
    examples = canonicalize_examples(build_matched_topology_examples(
        pair_count=1, seed=9041, split="m0_engineering", min_hops=hops,
        max_hops=hops, scenarios=("geodesic_action",), node_label_pool_size=17))
    if unsupported:
        examples = make_unsupported(examples)
    return collate_matched_topology_examples(examples, tokenizer, pad_to_tokens=pad), tokenizer


def model_config(tokenizer, paper=False):
    if paper:
        return ERGT43Config(vocab_size=tokenizer.vocab_size, max_tokens=768,
            raw_input_contract=RawTokenInputContract.from_tokenizer(tokenizer),
            hidden_dim=56, psi_rank=20, field_steps=3, sparse_top_k=20,
            max_hops=36, geodesic_closure_steps=36, geodesic_backbone_levels=10)
    return ERGT43Config(vocab_size=tokenizer.vocab_size, max_tokens=128,
        raw_input_contract=RawTokenInputContract.from_tokenizer(tokenizer),
        hidden_dim=8, psi_rank=4, field_steps=3, sparse_top_k=4,
        max_hops=3, geodesic_closure_steps=3, geodesic_backbone_levels=4)


def training_step(model, optimizer, batch, scheduler=None):
    model.train()
    outputs = model(**batch.model_inputs())
    config = model.baseline.config if hasattr(model, "baseline") else model.config
    loss, _ = native_geometric_training_loss(outputs, batch, config=config,
        teacher_weight=1.0, answer_weight=0.15)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.detach()
