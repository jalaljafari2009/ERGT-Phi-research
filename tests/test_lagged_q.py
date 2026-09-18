import torch
import pytest
from ergt_phi.lagged_q import build_sparse_q_with_presence,LaggedQ,native_proposal_q


def test_null_class_presence_is_consumed_once_and_pi_normalizes():
    source=torch.tensor([[4.,0.,-1.],[0.,4.,-1.]])
    target=torch.tensor([[0.,4.,-1.],[4.,0.,-1.]])
    relation=torch.tensor([[0.,8.,-4.,-4.],[0.,-4.,8.,-4.]])
    edges=torch.tensor([[0,1],[1,0]])
    q,mass,pi=build_sparse_q_with_presence(source,target,relation,edges)
    assert bool((mass>0).all())
    torch.testing.assert_close(pi.sum(-1),torch.ones(2),atol=1e-6,rtol=0)
    # First edge selects relation 1 and second relation 2; null mass is not squared.
    assert int(pi[0].argmax())==0 and int(pi[1].argmax())==1
    assert float(mass.max())<1


def test_lagged_q_starts_empty_detaches_and_commits_after_step():
    q=LaggedQ(); first=q.consume(device=torch.device('cpu'),dtype=torch.float64)
    assert first.shape==(0,0)
    candidate=torch.tensor([[.2,.8]],requires_grad=True)
    with pytest.raises(ValueError): q.commit(candidate)
    q.commit(candidate.detach())
    consumed=q.consume(); assert not consumed.requires_grad
    consumed[0,0]=9.; assert q.previous[0,0] != 9.


def test_q_empty_edges_and_invalid_domain():
    source=torch.zeros(2,3);target=torch.zeros(2,3);relation=torch.zeros(2,4)
    empty=torch.empty((2,0),dtype=torch.long)
    q,mass,pi=build_sparse_q_with_presence(source,target,relation,empty)
    assert q.shape==(0,3) and mass.shape==(0,) and pi.shape==(0,3)
    with pytest.raises(ValueError): build_sparse_q_with_presence(source,target,relation,torch.tensor([[0],[3]]))
    with pytest.raises(ValueError): build_sparse_q_with_presence(source,target,relation,torch.tensor([[0],[0]]))


def test_native_proposal_adapter_uses_event_slot_contract_and_mask():
    proposal={
        'event_source_logits': torch.tensor([[[5.,0.,-2.],[0.,5.,-2.],[0.,0.,0.]]]),
        'event_target_logits': torch.tensor([[[0.,5.,-2.],[5.,0.,-2.],[0.,0.,0.]]]),
        'event_relation_logits': torch.tensor([[[0.,7.,-3.,-3.],[0.,-3.,7.,-3.],[0.,0.,0.,0.]]]),
        # This field is intentionally irrelevant: the adapter must use the
        # typed proposal tensors above and never answer/gold outputs.
        'native_answer_logits': torch.tensor([[99., -99.]]),
    }
    edges=torch.tensor([[0,1],[1,0]],dtype=torch.long)
    q,mass,pi=native_proposal_q(proposal,edges,attention_mask=torch.tensor([[True,True,False]]))
    assert q.shape==(2,3) and bool((mass>0).all())
    torch.testing.assert_close(pi.sum(-1),torch.ones(2),atol=1e-6,rtol=0)
    assert int(pi[0].argmax())==0 and int(pi[1].argmax())==1
