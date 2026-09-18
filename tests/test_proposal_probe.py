import torch
from ergt_phi.proposal_probe import cache_native_proposal_pairs
from ergt_phi.fixtures import example_batch,model_config
from ergt_phi.native_steps import SteppedNative


def test_proposal_features_are_post_forward_loss_indexed_only():
    batch,tok=example_batch()
    model=SteppedNative(model_config(tok)).eval();model.requires_grad_(False)
    records=cache_native_proposal_pairs(model,batch.examples,tok)
    row=records[0]
    assert row['pair_features'].shape[0]==len(row['labels'])
    assert row['pair_features'].shape[1]==11
    assert not row['pair_features'].requires_grad
