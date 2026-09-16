"""M1 mathematical checks on independent float64 toy problems."""
from dataclasses import replace
import math
import pytest
import torch
from ergt_phi.phase_core import (
    KernelConfig, SparseContext, prepare_patch, scatter_patch, incident_sum,
    reward, reward_gradients, response, energy, stability_certificate,
    mirror_step, solve_phase_mirror, solve_phase_euclidean_audit,
)


def fixture(dtype=torch.float64):
    anchor = torch.tensor([-.2,.1,.3,-.1],dtype=dtype)
    base = torch.tensor([.3,.6,.45,.7],dtype=dtype)
    edges = torch.tensor([[0,1,2,3],[1,2,3,0]])
    evidence = torch.tensor([[.3,.0,.1],[.1,.4,.2],[.2,.1,.3],[.0,.2,.1]],dtype=dtype)
    offsets = torch.arange(3,dtype=dtype)*(2*math.pi/3)
    return prepare_patch(anchor,base,edges,evidence,offsets)[0]


@pytest.mark.parametrize('nu', [0.,.3,1.])
@pytest.mark.parametrize('n', [2,3,5])
def test_response_derivatives_and_zero_limit(nu,n):
    cfg = KernelConfig(nu=nu,n=n)
    x = torch.tensor([0.,1e-8,.2,.7,1.],dtype=torch.float64,requires_grad=True)
    g,gp,gpp = response(x,cfg)
    numerical_first = torch.autograd.grad(g.sum(),x,create_graph=True)[0]
    torch.testing.assert_close(gp,numerical_first,atol=1e-12,rtol=1e-12)
    if nu:
        numerical_second = torch.autograd.grad(numerical_first.sum(),x)[0]
        torch.testing.assert_close(gpp,numerical_second,atol=1e-11,rtol=1e-11)
    assert g[0] == 0 and gp[0] == 1 and gpp[0] == 0


@pytest.mark.parametrize('nu', [0.,.2,1.])
def test_explicit_reward_and_energy_gradients(nu):
    ctx = fixture()
    cfg = KernelConfig(nu=nu,coupling=.001)
    theta = (ctx.anchor+.04).requires_grad_()
    a = (ctx.base+.01).requires_grad_()
    expected = torch.autograd.grad(reward(theta,a,ctx,cfg),(theta,a))
    observed = reward_gradients(theta,a,ctx,cfg)
    for x,y in zip(expected,observed): torch.testing.assert_close(x,y,atol=1e-13,rtol=1e-13)
    ef = torch.autograd.grad(energy(theta,a,ctx,cfg),(theta,a))
    torch.testing.assert_close(ef[0],cfg.mu*(theta-ctx.anchor)-cfg.coupling*observed[0])
    torch.testing.assert_close(ef[1],cfg.tau_a*(torch.logit(a)-torch.logit(ctx.base))-cfg.coupling*observed[1])
    # The incidence gradient conserves the sum of phase components.
    torch.testing.assert_close(observed[0].sum(),theta.new_zeros(()),atol=1e-15,rtol=0)


@pytest.mark.parametrize('nu,coupling',[(0.,.02),(.3,.002)])
def test_unrolled_solver_finite_difference(nu,coupling):
    ctx = fixture()
    cfg = KernelConfig(nu=nu,coupling=coupling,steps=7)
    def function(anchor,base):
        out = solve_phase_mirror(replace(ctx,anchor=anchor,base=base),cfg)
        return torch.cat((out.theta,out.adhesion))
    assert torch.autograd.gradcheck(function,(ctx.anchor.clone().requires_grad_(),ctx.base.clone().requires_grad_()),eps=1e-6,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('nu,coupling',[(0.,.02),(.2,.002)])
def test_hessian_observation_and_contraction(nu,coupling):
    ctx = fixture()
    cfg = KernelConfig(nu=nu,coupling=coupling)
    cert = stability_certificate(ctx,cfg)
    for phase_shift in (-.2,0.,.2):
        y = torch.cat((ctx.anchor+phase_shift*torch.arange(4),ctx.base))
        h = torch.autograd.functional.hessian(lambda z: reward(z[:4],z[4:],ctx,cfg),y)
        assert float(torch.linalg.eigvalsh(h).abs().max()) <= cert.reward_hessian_bound
    theta,v = ctx.anchor,torch.logit(ctx.base)
    other_t = theta+torch.tensor([.1,-.03,.07,.02])
    other_v = v+torch.tensor([.02,.01,-.04,.01])
    left = mirror_step(theta,v,ctx,cfg,cert)
    right = mirror_step(other_t,other_v,ctx,cfg,cert)
    before = max(torch.linalg.vector_norm(theta-other_t),torch.linalg.vector_norm(v-other_v))
    after = max(torch.linalg.vector_norm(left[0]-right[0]),torch.linalg.vector_norm(left[1]-right[1]))
    assert after <= cert.contraction_bound*before


@pytest.mark.parametrize('nu,coupling,c0',[(0.,.02,.5),(.2,.002,.5),(0.,.02,100.)])
def test_independent_euclidean_prox_agrees_at_convergence(nu,coupling,c0):
    ctx = fixture()
    cfg = KernelConfig(nu=nu,coupling=coupling,c0=c0,steps=90)
    mirror = solve_phase_mirror(ctx,cfg)
    theta,a,_ = solve_phase_euclidean_audit(ctx,cfg)
    torch.testing.assert_close(mirror.theta,theta,atol=1e-10,rtol=0)
    torch.testing.assert_close(mirror.adhesion,a,atol=1e-10,rtol=0)
    assert energy(mirror.theta,mirror.adhesion,ctx,cfg) <= energy(ctx.anchor,ctx.base,ctx,cfg)+1e-12


def test_analytic_residual_bounds_error_against_longer_solve():
    ctx = fixture()
    short = solve_phase_mirror(ctx,KernelConfig(steps=3))
    long = solve_phase_mirror(ctx,KernelConfig(steps=100))
    error = torch.maximum(torch.linalg.vector_norm(short.theta-long.theta),torch.linalg.vector_norm(short.logits-long.logits))
    assert error <= short.error_bound_ignoring_roundoff+1e-14
    assert short.certificate.ignores_roundoff and short.certificate.scope == 'fixed_context_only'


def test_fixed_clock_jacobi_and_diagnostic_not_committed():
    ctx,cfg = fixture(),KernelConfig(steps=1)
    cert = stability_certificate(ctx,cfg)
    v0 = torch.logit(ctx.base)
    gt,ga = reward_gradients(ctx.anchor,ctx.base,ctx,cfg)
    expected_t = (ctx.anchor+cert.eta*cfg.mu*ctx.anchor+cert.eta*cfg.coupling*gt)/(1+cert.eta*cfg.mu)
    expected_v = (v0+cert.eta*cfg.tau_a*v0+cert.eta*cfg.coupling*ga)/(1+cert.eta*cfg.tau_a)
    out = solve_phase_mirror(ctx,cfg)
    torch.testing.assert_close(out.theta,expected_t,atol=1e-15,rtol=0)
    torch.testing.assert_close(out.logits,expected_v,atol=1e-15,rtol=0)
    assert out.iterations == 1
    assert out.residual > 0


def test_empty_and_zero_coupling_are_exact_neutral_and_rng_free():
    ctx = fixture()
    empty = replace(ctx,base=ctx.base[:0],edge_index=ctx.edge_index[:,:0],mixture=ctx.mixture[:0],weight=ctx.weight[:0])
    rng = torch.get_rng_state().clone()
    for context,cfg in [(empty,KernelConfig()),(ctx,KernelConfig(coupling=0))]:
        out = solve_phase_mirror(context,cfg,previous_theta=context.anchor+3)
        assert torch.equal(out.theta,context.anchor)
        assert torch.equal(out.adhesion,context.base)
        assert out.iterations == 0 and out.residual == 0
    assert torch.equal(rng,torch.get_rng_state())


def test_zero_evidence_unsupported_base_self_padding_and_literal_scatter():
    anchor = torch.zeros(4,dtype=torch.float64)
    base = torch.tensor([0.,1.,.4,.5,.6,.7],dtype=torch.float64)
    edges = torch.tensor([[0,0,0,1,1,2],[1,2,2,2,1,3]])
    evidence = torch.tensor([[1.,0.],[1.,0.],[0.,0.],[.5,0.],[1.,0.],[1.,0.]],dtype=torch.float64,requires_grad=True)
    ctx,kept = prepare_patch(anchor,base,edges,evidence,anchor[:2],eligible=torch.tensor([True,True,True,True,True,False]))
    assert kept.tolist() == [3]
    assert ctx.mixture.tolist() == [[1.,0.]]
    assert not ctx.weight.requires_grad and not ctx.mixture.requires_grad
    assert ctx.weight.item() == pytest.approx(.5/1.5)
    result = scatter_patch(base,kept,solve_phase_mirror(ctx).adhesion)
    assert torch.equal(result[[0,1,2,4,5]],base[[0,1,2,4,5]])
    zero,kept = prepare_patch(anchor,base,edges,torch.zeros_like(evidence),anchor[:2])
    assert zero.base.numel() == 0
    assert torch.equal(scatter_patch(base,kept,solve_phase_mirror(zero).adhesion),base)


def test_gauge_edge_permutation_padding_and_block_independence():
    ctx = fixture()
    out = solve_phase_mirror(ctx)
    shifted = solve_phase_mirror(replace(ctx,anchor=ctx.anchor+8*math.pi))
    torch.testing.assert_close(shifted.adhesion,out.adhesion,atol=1e-14,rtol=0)
    torch.testing.assert_close(shifted.theta,out.theta+8*math.pi,atol=1e-13,rtol=0)
    permutation = torch.tensor([3,1,0,2])
    permuted = solve_phase_mirror(replace(ctx,base=ctx.base[permutation],edge_index=ctx.edge_index[:,permutation],mixture=ctx.mixture[permutation],weight=ctx.weight[permutation]))
    torch.testing.assert_close(permuted.theta,out.theta,atol=1e-14,rtol=0)
    torch.testing.assert_close(permuted.adhesion,out.adhesion[permutation],atol=1e-14,rtol=0)
    padded = solve_phase_mirror(replace(ctx,anchor=torch.cat((ctx.anchor,ctx.anchor.new_tensor([9.,-8.])))))
    torch.testing.assert_close(padded.theta[:4],out.theta,atol=0,rtol=0)
    torch.testing.assert_close(padded.adhesion,out.adhesion,atol=0,rtol=0)
    # Reordering independent example/world solves cannot share a normalizer/state.
    other = replace(ctx,anchor=-ctx.anchor)
    forward = [solve_phase_mirror(x) for x in (ctx,other)]
    reverse = [solve_phase_mirror(x) for x in (other,ctx)]
    for x,y in zip(forward,reversed(reverse)):
        torch.testing.assert_close(x.adhesion,y.adhesion,atol=0,rtol=0)


def test_weight_normalization_counts_both_endpoints():
    ctx = fixture()
    prepared,_ = prepare_patch(ctx.anchor,ctx.base,ctx.edge_index,torch.full_like(ctx.mixture,100.),ctx.offsets)
    degrees = incident_sum(prepared.edge_index,prepared.weight,4)
    torch.testing.assert_close(degrees,torch.ones_like(degrees),atol=1e-15,rtol=0)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_trust_region_and_local_edge_length_bound(dtype):
    ctx = fixture(dtype)
    cfg = KernelConfig(c0=100.)
    result = solve_phase_mirror(ctx,cfg,previous_theta=ctx.anchor+20)
    tol = 2e-6 if dtype == torch.float32 else 1e-14
    assert bool(((result.theta-ctx.anchor).abs() <= cfg.phase_radius+tol).all())
    assert bool(((result.logits-torch.logit(ctx.base)).abs() <= cfg.logit_radius+tol).all())
    assert bool(((torch.log(result.adhesion)-torch.log(ctx.base)).abs() <= cfg.logit_radius+tol).all())
    omega = torch.tensor([.2,.4,.6,.8],dtype=dtype)
    old = torch.tensor([.8,.1,.3,.5],dtype=dtype)
    for beta in (0.,.7):
        preview = beta*old+(1-beta)*ctx.base*omega
        changed = beta*old+(1-beta)*result.adhesion*omega
        lp = -torch.log(preview.clamp_min(1e-6))
        lc = -torch.log(changed.clamp_min(1e-6))
        assert bool((lc >= 0).all())
        assert bool(((lc-lp).abs() <= cfg.logit_radius+tol).all())


@pytest.mark.parametrize('kwargs',[{'mu':0},{'tau_a':-1},{'n':1},{'n':2.5},{'steps':0},{'c':0},{'nu':-1},{'coupling':float('nan')},{'epsilon_a':.5},{'sigma':1},{'phase_radius':-1}])
def test_config_domains(kwargs):
    with pytest.raises(ValueError): KernelConfig(**kwargs)


def test_context_domains_and_guard_reject_before_iteration(monkeypatch):
    ctx = fixture()
    for broken in [replace(ctx,base=ctx.base*0),replace(ctx,weight=ctx.weight*10),replace(ctx,mixture=ctx.mixture*0),replace(ctx,anchor=ctx.anchor*float('nan')),replace(ctx,edge_index=ctx.edge_index+20),replace(ctx,mixture=ctx.mixture.clone().requires_grad_())]:
        with pytest.raises(ValueError): solve_phase_mirror(broken)
    def forbidden(*args): raise AssertionError('iteration ran despite invalid certificate')
    monkeypatch.setattr('ergt_phi.phase_core.mirror_step',forbidden)
    with pytest.raises(ValueError,match='contraction guard'):
        solve_phase_mirror(ctx,KernelConfig(coupling=1))


def test_no_input_mutation_or_solver_rng_use():
    ctx = fixture()
    before = [x.clone() for x in (ctx.anchor,ctx.base,ctx.edge_index,ctx.mixture,ctx.offsets,ctx.weight)]
    rng = torch.get_rng_state().clone()
    solve_phase_mirror(ctx)
    for x,y in zip(before,(ctx.anchor,ctx.base,ctx.edge_index,ctx.mixture,ctx.offsets,ctx.weight)):
        assert torch.equal(x,y)
    assert torch.equal(rng,torch.get_rng_state())


def test_unrepresentable_response_rejected():
    with pytest.raises(ValueError,match='representable'):
        response(torch.zeros(1),KernelConfig(nu=1,c=1e-100))


def test_context_gradients_detach_evidence_but_retain_anchor_and_base():
    original = fixture()
    anchor = original.anchor.clone().requires_grad_()
    base = original.base.clone().requires_grad_()
    evidence = original.mixture.clone().requires_grad_()
    ctx,_ = prepare_patch(anchor,base,original.edge_index,evidence,original.offsets)
    out = solve_phase_mirror(ctx)
    gradients = torch.autograd.grad(out.adhesion.sum(),(anchor,base,evidence),allow_unused=True)
    assert gradients[0] is not None and bool((gradients[0].abs() > 0).any())
    assert gradients[1] is not None and bool((gradients[1].abs() > 0).any())
    assert gradients[2] is None


def test_zero_radii_and_guard_ceiling():
    ctx = fixture()
    result = solve_phase_mirror(ctx,KernelConfig(phase_radius=0,logit_radius=0))
    torch.testing.assert_close(result.theta,ctx.anchor,atol=0,rtol=0)
    torch.testing.assert_close(result.logits,torch.logit(ctx.base),atol=0,rtol=0)
    ceiling = stability_certificate(ctx,KernelConfig()).coupling_ceiling
    with pytest.raises(ValueError,match='contraction guard'):
        solve_phase_mirror(ctx,KernelConfig(coupling=ceiling*1.001))
    assert stability_certificate(ctx,KernelConfig(coupling=ceiling*.999)).contraction_bound < 1


def test_invalid_numerical_domain():
    with pytest.raises(ValueError,match='representable'):
        solve_phase_mirror(fixture(torch.float32),KernelConfig(epsilon_a=1e-12))
    with pytest.raises(ValueError,match='domain'):
        response(torch.tensor([-.1]),KernelConfig())
    ctx = fixture()
    with pytest.raises(ValueError,match='duplicate'):
        solve_phase_mirror(replace(ctx,edge_index=ctx.edge_index[:,[0,0,2,3]]))
    with pytest.raises(ValueError,match='warm start'):
        solve_phase_mirror(ctx,previous_theta=torch.zeros(7,dtype=ctx.anchor.dtype))
