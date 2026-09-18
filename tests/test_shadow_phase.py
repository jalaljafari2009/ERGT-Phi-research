"""M2 engineering acceptance, independent of calibration qualification."""
import copy
import inspect
import math
from pathlib import Path
import subprocess
import sys
import pytest
import torch
import numpy as np
from dataclasses import replace
from ergt_phi.shadow_phase import (CalibrationConfig,PhaseAnchorNetwork,ShadowPhaseModel,
    relation_logits,phase_loss,calibration_step,readiness)
from ergt_phi.shadow_data import pair_partition
from ergt_phi.fixtures import example_batch,model_config
from ergt_phi.native_steps import SteppedNative


def exact(a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: exact(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): exact(x,y)
    else: assert a==b


def fixture():
    p=PhaseAnchorNetwork(4,2,3,CalibrationConfig(hidden_dim=8))
    g=torch.Generator().manual_seed(44)
    tensors=(torch.randn(2,5,4,generator=g),torch.ones(2,5,dtype=torch.bool),
             torch.tensor([[0,0,1],[0,2,3],[1,1,2],[1,3,4]]),torch.tensor([0,1,2,0]))
    return p,tensors


def test_separate_seed_preserves_rng_and_initial_code():
    before=torch.get_rng_state().clone()
    p,_=fixture()
    assert torch.equal(before,torch.get_rng_state())
    other,_=fixture(); exact(p.state_dict(),other.state_dict())
    for row in p.offsets: torch.testing.assert_close(row,torch.arange(3)*2*math.pi/3)


def test_formula_range_and_padding_batch_invariance():
    p,(psi,mask,_,_)=fixture()
    out=p(psi,mask)
    torch.testing.assert_close(out,(math.pi*torch.tanh(p.network(psi))).transpose(1,2),rtol=0,atol=0)
    assert bool((out.abs()<=math.pi).all())
    for i in range(2): torch.testing.assert_close(out[i],p(psi[i:i+1],mask[i:i+1])[0],atol=1e-6,rtol=0)
    padded=p(torch.cat((psi,torch.zeros(2,3,4)),dim=1),torch.cat((mask,torch.zeros(2,3,dtype=torch.bool)),dim=1))
    torch.testing.assert_close(out,padded[:,:,:5],atol=1e-6,rtol=0)
    assert torch.equal(padded[:,:,5:],torch.zeros_like(padded[:,:,5:]))


def test_relation_loss_formula_gauge_and_gradients():
    p,tensors=fixture()
    psi,mask,events,labels=tensors
    anchor=p(psi,mask)
    logits=relation_logits(anchor,p.offsets,events,p.config.temperature)
    expected=[]
    for b,i,j in events:
        expected.append(((1+torch.cos(anchor[b,:,j,None]-anchor[b,:,i,None]-p.offsets))/2).mean(0)/p.config.temperature)
    torch.testing.assert_close(logits,torch.stack(expected))
    shifted=relation_logits(anchor+2,p.offsets,events,p.config.temperature)
    torch.testing.assert_close(logits,shifted,atol=1e-6,rtol=1e-6)
    loss,_=phase_loss(p,*tensors); loss.backward()
    assert p.offset_raw.grad is not None and bool((p.offset_raw.grad.abs()>0).any())
    assert any(x.grad is not None and bool((x.grad.abs()>0).any()) for x in p.network.parameters())


def test_offsets_bounded_and_noncollapsed():
    p,_=fixture()
    with torch.no_grad(): p.offset_raw.copy_(torch.tensor([[100.,-100.,100.],[-100.,100.,-100.]]))
    assert bool(((p.offsets-p.initial_offsets).abs()<=p.config.offset_radius+1e-6).all())
    difference=p.offsets[:,:,None]-p.offsets[:,None,:]
    distance=torch.atan2(difference.sin(),difference.cos()).abs()
    assert float(distance[:,~torch.eye(3,dtype=torch.bool)].min().detach())>1.5


def test_native_output_rng_weights_and_training_freeze():
    batch,tok=example_batch()
    baseline=SteppedNative(model_config(tok)).eval()
    saved=copy.deepcopy(baseline.state_dict())
    with torch.no_grad(): expected=baseline(**batch.model_inputs())
    before=torch.get_rng_state().clone()
    shadow=ShadowPhaseModel(baseline,PhaseAnchorNetwork(baseline.config.hidden_dim,baseline.config.n_worlds,3))
    shadow.train()
    assert not shadow.baseline.training
    with torch.no_grad(): actual=shadow(**batch.model_inputs())
    exact(expected,actual)
    anchors=shadow.anchors(**batch.model_inputs())
    anchors.square().mean().backward()
    assert all(not p.requires_grad and p.grad is None for p in baseline.parameters())
    exact(saved,baseline.state_dict())
    assert torch.equal(before,torch.get_rng_state())
    assert set(inspect.signature(shadow.forward).parameters)=={'raw_token_ids','attention_mask','observe_spectrum'}


def test_remote_relation_is_invisible_at_canonical_endpoint_anchors():
    from ergt_reviewer.data_schema import RELATION_SURFACES
    from ergt_reviewer.matched_data import raw_token_fingerprint_id
    batch,tok=example_batch()
    base=SteppedNative(model_config(tok)).eval()
    shadow=ShadowPhaseModel(base,PhaseAnchorNetwork(base.config.hidden_dim,base.config.n_worlds,3))
    edge=batch.examples[0].physical_edges[0]
    changed=batch.raw_token_ids.clone()
    changed[0,edge.event_anchor_position]=raw_token_fingerprint_id(RELATION_SURFACES[edge.relation_id%3+1])
    left=shadow.anchors(**batch.model_inputs())
    right=shadow.anchors(changed,batch.base.attention_mask)
    assert not torch.equal(changed,batch.raw_token_ids)
    exact(left[0,:,[edge.source_position,edge.target_position]],right[0,:,[edge.source_position,edge.target_position]])


def test_pair_partition_never_splits_counterfactual_pair():
    batch,_=example_batch()
    examples=[]
    for i in range(10):
        for example in batch.examples: examples.append(replace(example,pair_id=f'pair_{i}'))
    fit,monitor=pair_partition(examples,14)
    assert {examples[i].pair_id for i in fit}.isdisjoint({examples[i].pair_id for i in monitor})
    assert pair_partition(examples,14)==(fit,monitor)
    assert sorted(fit+monitor)==list(range(20))


def test_atomic_update_rolls_back_numerical_failure(monkeypatch):
    phase,tensors=fixture()
    opt=torch.optim.AdamW(phase.parameters(),lr=.002)
    progress={'step':0,'cursor':0}
    contract={'stage':'test'}
    calibration_step(phase,opt,tensors,contract=contract,progress=progress)
    state,optim,p0,rng=copy.deepcopy(phase.state_dict()),copy.deepcopy(opt.state_dict()),dict(progress),torch.get_rng_state().clone()
    def bad_loss(*args):
        torch.rand(4)
        return torch.tensor(float('nan')),None
    monkeypatch.setattr('ergt_phi.shadow_phase.phase_loss',bad_loss)
    with pytest.raises(FloatingPointError): calibration_step(phase,opt,tensors,contract=contract,progress=progress)
    exact(state,phase.state_dict()); exact(optim,opt.state_dict()); assert progress==p0
    assert torch.equal(rng,torch.get_rng_state())


def test_freeze_rejects_update_before_state_or_rng_changes():
    phase,tensors=fixture(); opt=torch.optim.AdamW(phase.parameters())
    phase.freeze(); state=copy.deepcopy(phase.state_dict()); rng=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError,match='frozen'):
        calibration_step(phase,opt,tensors,contract={},progress={})
    exact(state,phase.state_dict()); assert torch.equal(rng,torch.get_rng_state())


def test_readiness_checks_all_classes_and_collapse():
    cfg=CalibrationConfig()
    metrics={'class_counts':[10,10,10],'balanced_accuracy':.9,'per_class_recall':[.9,.9,.9],
             'ce':.4,'minimum_offset_separation':2.,'phase_resultant':.5}
    assert readiness(metrics,1.,cfg)[0]
    for name,value in [('class_counts',[0,10,10]),('balanced_accuracy',.4),('per_class_recall',[0.,1.,1.]),('phase_resultant',1.),('ce',1.),('minimum_offset_separation',.1)]:
        assert not readiness({**metrics,name:value},1.,cfg)[0]


@pytest.mark.parametrize('kwargs',[{'epochs':0},{'batch_size':0},{'temperature':0},{'offset_radius':-1},{'learning_rate':float('nan')},{'required_windows':0},{'readiness_balanced_accuracy':1.1}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError): CalibrationConfig(**kwargs)


def test_invalid_inputs_and_empty_supervision():
    p,tensors=fixture()
    with pytest.raises(ValueError): p(tensors[0]*float('nan'),tensors[1])
    with pytest.raises(ValueError): PhaseAnchorNetwork(4,2,3,CalibrationConfig(offset_radius=2))
    with pytest.raises(ValueError): phase_loss(p,tensors[0],tensors[1],tensors[2][:0],tensors[3][:0])


def test_shadow_resume_in_fresh_process(tmp_path):
    root=Path(__file__).resolve().parents[1]
    worker=root/'scripts/shadow_resume_worker.py'
    for mode in ('continuous','first','resume'):
        subprocess.run([sys.executable,'-B',str(worker),mode,str(tmp_path)],cwd=root,check=True,timeout=120)
    a=torch.load(tmp_path/'continuous.pt',map_location='cpu',weights_only=False)
    b=torch.load(tmp_path/'resumed.pt',map_location='cpu',weights_only=False)
    for key in ('model','optimizer','progress','requires_grad','module_training','contract'):
        exact(a[key],b[key])
    exact(a['rng']['torch_cpu'],b['rng']['torch_cpu'])
    exact(a['rng']['python'],b['rng']['python'])
    assert a['rng']['numpy'][0]==b['rng']['numpy'][0]
    np.testing.assert_array_equal(a['rng']['numpy'][1],b['rng']['numpy'][1])
    exact(a['rng']['numpy'][2:],b['rng']['numpy'][2:])
