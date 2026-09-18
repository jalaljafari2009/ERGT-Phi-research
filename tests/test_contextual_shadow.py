"""Acceptance tests for the M2 contextual-anchor revision."""
import copy
import torch
from ergt_phi.contextual_shadow import ContextualShadowModel, native_field_history
from ergt_phi.shadow_phase import CalibrationConfig, PhaseAnchorNetwork
from ergt_phi.fixtures import example_batch, model_config
from ergt_phi.native_steps import SteppedNative
from ergt_reviewer.matched_data import raw_token_fingerprint_id
from ergt_reviewer.data_schema import RELATION_SURFACES


def exact(a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): exact(x,y)
    else: assert a==b


def make():
    batch,tok=example_batch(hops=2)
    base=SteppedNative(model_config(tok)).eval()
    phase=PhaseAnchorNetwork(base.config.hidden_dim*2,base.config.n_worlds,3,CalibrationConfig(hidden_dim=16))
    shadow=ContextualShadowModel(base,phase)
    return batch,shadow


def test_context_is_frozen_raw_only_and_has_native_field_steps():
    batch,shadow=make()
    states=native_field_history(shadow.baseline,**batch.model_inputs())
    assert len(states)==shadow.baseline.config.field_steps+1
    assert states[0].shape[-1]*2==shadow.phase.network[1].in_features
    assert all(not p.requires_grad for p in shadow.baseline.parameters())
    assert not shadow.baseline.training


def test_relation_change_reaches_endpoint_context_and_anchor():
    batch,shadow=make()
    edge=batch.examples[0].physical_edges[0]
    changed=batch.raw_token_ids.clone()
    changed[0,edge.event_anchor_position]=raw_token_fingerprint_id(RELATION_SURFACES[edge.relation_id%3+1])
    left=shadow.context_features(**batch.model_inputs())
    right=shadow.context_features(changed,batch.base.attention_mask)
    anchor_left=shadow.anchors(**batch.model_inputs())
    anchor_right=shadow.anchors(changed,batch.base.attention_mask)
    assert not torch.equal(changed,batch.raw_token_ids)
    assert float((left[:,[edge.source_position,edge.target_position]]-right[:,[edge.source_position,edge.target_position]]).abs().max())>1e-8
    assert float((anchor_left[:,[edge.source_position,edge.target_position]]-anchor_right[:,[edge.source_position,edge.target_position]]).abs().max().detach())>1e-8


def test_contextual_shadow_does_not_change_native_answer_or_weights():
    batch,shadow=make()
    before=copy.deepcopy(shadow.baseline.state_dict())
    with torch.no_grad():
        native=shadow.baseline(**batch.model_inputs())
        observed=shadow(**batch.model_inputs())
    exact(native,observed)
    exact(before,shadow.baseline.state_dict())
    assert all(p.grad is None for p in shadow.baseline.parameters())


def test_context_features_are_batch_and_padding_independent():
    batch,shadow=make()
    with torch.no_grad():
        expected=shadow.context_features(**batch.model_inputs())
        one=shadow.context_features(batch.raw_token_ids[:1],batch.base.attention_mask[:1])
        padded_ids=torch.cat((batch.raw_token_ids[:1],torch.zeros(1,8,dtype=torch.long)),dim=1)
        padded_mask=torch.cat((batch.base.attention_mask[:1],torch.zeros(1,8,dtype=torch.bool)),dim=1)
        padded=shadow.context_features(padded_ids,padded_mask)
    torch.testing.assert_close(expected[:1],one,atol=1e-6,rtol=0)
    torch.testing.assert_close(expected[:1],padded[:,:expected.shape[1]],atol=2e-6,rtol=0)
    assert torch.equal(padded[:,expected.shape[1]:],torch.zeros_like(padded[:,expected.shape[1]:]))
