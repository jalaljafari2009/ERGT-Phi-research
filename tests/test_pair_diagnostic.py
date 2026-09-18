import torch
import pytest
from ergt_phi.pair_diagnostic import PairDiagnosticConfig,PairRelationNetwork


def test_pair_features_are_ordered_and_interaction_aware():
    model=PairRelationNetwork(3,PairDiagnosticConfig(hidden_dim=8))
    x=torch.arange(2*4*3,dtype=torch.float32).reshape(2,4,3)
    events=torch.tensor([[0,1,2],[1,0,3]])
    f=model.pair_features(x,events)
    assert f.shape==(2,12)
    torch.testing.assert_close(f[0,:3],x[0,1]); torch.testing.assert_close(f[0,3:6],x[0,2])
    torch.testing.assert_close(f[0,6:9],(x[0,1]-x[0,2]).abs()); torch.testing.assert_close(f[0,9:],x[0,1]*x[0,2])
    assert not torch.equal(f[0],f[1])


def test_pair_head_rejects_invalid_events_and_is_loss_only():
    model=PairRelationNetwork(3)
    x=torch.zeros(1,4,3)
    with pytest.raises(ValueError): model(x,torch.tensor([[0,0,4]]))
    assert not hasattr(model,'native_answer_logits')


def test_pair_head_can_backpropagate():
    model=PairRelationNetwork(3)
    x=torch.randn(1,4,3); e=torch.tensor([[0,1,2]]); y=torch.tensor([1])
    loss=torch.nn.functional.cross_entropy(model(x,e),y); loss.backward()
    assert any(p.grad is not None for p in model.parameters())
