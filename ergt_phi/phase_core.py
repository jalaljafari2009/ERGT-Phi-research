"""Independent sparse inner problem, specification sections 6--10.

Each context represents ONE example and ONE world. No native geometry, answer
selection or phase calibration is performed here. Certificates ignore roundoff.
"""
from dataclasses import dataclass
import math
import torch
from torch import Tensor


@dataclass(frozen=True)
class KernelConfig:
    mu: float = 1.0
    tau_a: float = .25
    coupling: float = .02
    nu: float = 0.0
    n: int = 2
    c: float = .5
    c0: float = .5  # Explicit research choice: the specification leaves c0 open.
    phase_radius: float = .35
    logit_radius: float = .10
    epsilon_a: float = 1e-4
    epsilon_q: float = 1e-12
    steps: int = 24
    sigma: float = .5

    def __post_init__(self):
        for name in ('mu','tau_a','coupling','nu','c','c0','phase_radius',
                     'logit_radius','epsilon_a','epsilon_q','sigma'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if min(self.mu,self.tau_a,self.c) <= 0:
            raise ValueError('mu, tau_a and c must be positive')
        if min(self.coupling,self.nu,self.phase_radius,self.logit_radius,self.epsilon_q) < 0:
            raise ValueError('coupling, nu, radii and epsilon_q must be nonnegative')
        if not 0 < self.epsilon_a < .5 or not 0 < self.sigma < 1:
            raise ValueError('invalid epsilon_a or sigma')
        if type(self.n) is not int or self.n < 2 or type(self.steps) is not int or self.steps < 1:
            raise ValueError('integer n >= 2 and steps >= 1 required')


@dataclass(frozen=True)
class SparseContext:
    anchor: Tensor          # [N], gradients allowed
    base: Tensor            # [E], gradients allowed
    edge_index: Tensor      # [2,E], source then target
    mixture: Tensor         # [E,R], detached
    offsets: Tensor         # [R], fixed throughout a solve
    weight: Tensor          # [E], detached, max incident degree <= 1


@dataclass(frozen=True)
class Certificate:
    dstar: float
    reward_hessian_bound: float
    coupling_ceiling: float
    strong_convexity_margin: float
    eta: float
    contraction_bound: float
    ignores_roundoff: bool = True
    scope: str = 'fixed_context_only'


@dataclass(frozen=True)
class SolveResult:
    theta: Tensor
    adhesion: Tensor
    logits: Tensor
    iterations: int
    residual: Tensor
    error_bound_ignoring_roundoff: Tensor
    certificate: Certificate


def _finite(*values):
    if any(not bool(torch.isfinite(x).all()) for x in values):
        raise ValueError('nonfinite input or numerical result')


def incident_sum(edge_index, values, node_count):
    """Both endpoint occurrences count, even for oppositely directed pairs."""
    return values.new_zeros(node_count).index_add(0,edge_index[0],values).index_add(0,edge_index[1],values)


def validate(ctx, cfg):
    a,b,e,p,beta,w = ctx.anchor,ctx.base,ctx.edge_index,ctx.mixture,ctx.offsets,ctx.weight
    if a.ndim != 1 or a.numel() == 0 or b.ndim != 1 or beta.ndim != 1 or beta.numel() == 0:
        raise ValueError('nonempty 1D anchor/offsets and 1D base required')
    if e.shape != (2,b.numel()) or p.shape != (b.numel(),beta.numel()) or w.shape != b.shape:
        raise ValueError('invalid sparse shapes')
    if e.dtype != torch.long or a.dtype not in (torch.float32,torch.float64):
        raise ValueError('int64 indices and float32/float64 continuous state required')
    if cfg.epsilon_a < torch.finfo(a.dtype).eps:
        raise ValueError('epsilon_a must be representable away from both 0 and 1')
    if any(x.dtype != a.dtype or x.device != a.device for x in (b,p,beta,w)) or e.device != a.device:
        raise ValueError('all tensors must share device and floating dtype')
    _finite(a,b,p,beta,w)
    if e.numel() and (int(e.min()) < 0 or int(e.max()) >= a.numel()):
        raise ValueError('edge index outside node domain')
    if bool((e[0] == e[1]).any()) or torch.unique(e,dim=1).shape[1] != b.numel():
        raise ValueError('self or duplicate edges are not a valid patch')
    if bool(((b < cfg.epsilon_a) | (b > 1-cfg.epsilon_a)).any()):
        raise ValueError('base outside eligible interval; filter, never clamp it')
    tol = 32*torch.finfo(a.dtype).eps
    if bool((p < 0).any()) or not torch.allclose(p.sum(-1),torch.ones_like(b),atol=tol,rtol=0):
        raise ValueError('mixture must be nonnegative and sum to one per edge')
    if bool((w < 0).any()) or bool((incident_sum(e,w,a.numel()) > 1+tol).any()):
        raise ValueError('invalid incident-weight normalization')
    if p.requires_grad or w.requires_grad:
        raise ValueError('semantic mixtures and weights must be detached')


def prepare_patch(anchor, base, edge_index, evidence, offsets, cfg=KernelConfig(), *, eligible=None):
    """Filter an already selected preview candidate list, without dense Q.

    The future native adapter must supply padding/domain/top-k eligibility.
    Returns the sparse context AND retained candidate positions for scattering.
    Evidence already contains presence mass; it is never multiplied a second time.
    """
    if edge_index.shape != (2,base.numel()) or evidence.shape != (base.numel(),offsets.numel()):
        raise ValueError('candidate shape mismatch')
    if anchor.ndim != 1 or anchor.numel() == 0 or base.ndim != 1 or offsets.ndim != 1:
        raise ValueError('invalid candidate dimensions')
    if edge_index.dtype != torch.long or edge_index.device != anchor.device:
        raise ValueError('invalid candidate indices')
    if any(x.dtype != anchor.dtype or x.device != anchor.device for x in (base,evidence,offsets)):
        raise ValueError('candidate dtype/device mismatch')
    _finite(anchor,base,evidence,offsets)
    if bool((evidence < 0).any()): raise ValueError('negative proposal evidence')
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= anchor.numel()):
        raise ValueError('candidate endpoint outside node domain')
    if eligible is None: eligible = torch.ones_like(base,dtype=torch.bool)
    if eligible.shape != base.shape or eligible.dtype != torch.bool or eligible.device != base.device:
        raise ValueError('invalid eligibility mask')
    q = evidence.detach().sum(-1)
    _finite(q)
    mask = eligible & (edge_index[0] != edge_index[1]) & (q > cfg.epsilon_q) & (base >= cfg.epsilon_a) & (base <= 1-cfg.epsilon_a)
    kept = mask.nonzero(as_tuple=True)[0]
    edges = edge_index[:,kept]
    mass = q[kept]
    mixture = evidence.detach()[kept]/mass[:,None]
    raw = mass/(1+mass)
    scale = incident_sum(edges,raw,anchor.numel()).max().clamp_min(1)
    ctx = SparseContext(anchor,base[kept],edges,mixture,offsets,raw/scale)
    validate(ctx,cfg)
    return ctx,kept


def scatter_patch(base, kept, adhesion):
    """Fresh candidate vector; all cells outside the patch remain literal base."""
    return base.index_copy(0,kept,adhesion)


def response(x, cfg):
    """G, G', G'' without inverse powers of x; valid at x=0."""
    _finite(x)
    if bool(((x < 0) | (x > 1)).any()):
        raise ValueError('response domain is [0,1]')
    if cfg.nu == 0:
        return x,torch.ones_like(x),torch.zeros_like(x)
    xn = x.pow(cfg.n)
    cn = x.new_tensor(cfg.c).pow(cfg.n)
    if not bool(torch.isfinite(cn)) or not bool(cn > 0):
        raise ValueError('c**n is not representable in the chosen dtype')
    denom = xn+cn
    h = xn/denom
    chi = cfg.nu*cfg.n*xn*cn/denom.square()
    g = x*(1+cfg.nu*h)
    gp = 1+cfg.nu*h+chi
    gpp = cfg.nu*cfg.n*cn*x.pow(cfg.n-1)*((cfg.n+1)*cn-(cfg.n-1)*xn)/denom.pow(3)
    _finite(g,gp,gpp)
    return g,gp,gpp


def _interactions(theta, adhesion, ctx):
    delta = theta[ctx.edge_index[1],None]-theta[ctx.edge_index[0],None]-ctx.offsets[None,:]
    return delta,(1+torch.cos(delta))/2


def reward(theta, adhesion, ctx, cfg):
    _,kappa = _interactions(theta,adhesion,ctx)
    g,_,_ = response(adhesion[:,None]*kappa,cfg)
    return (ctx.weight*((ctx.mixture*g).sum(-1)-cfg.c0*adhesion)).sum()


def reward_gradients(theta, adhesion, ctx, cfg):
    delta,kappa = _interactions(theta,adhesion,ctx)
    _,gp,_ = response(adhesion[:,None]*kappa,cfg)
    u = -.5*ctx.weight*adhesion*(ctx.mixture*gp*torch.sin(delta)).sum(-1)
    gt = torch.zeros_like(theta).index_add(0,ctx.edge_index[0],-u).index_add(0,ctx.edge_index[1],u)
    ga = ctx.weight*((ctx.mixture*kappa*gp).sum(-1)-cfg.c0)
    return gt,ga


def energy(theta, adhesion, ctx, cfg):
    kl = adhesion*(torch.log(adhesion)-torch.log(ctx.base))+(1-adhesion)*(torch.log1p(-adhesion)-torch.log1p(-ctx.base))
    return cfg.mu/2*(theta-ctx.anchor).square().sum()+cfg.tau_a*kl.sum()-cfg.coupling*reward(theta,adhesion,ctx,cfg)


def stability_certificate(ctx, cfg):
    validate(ctx,cfg)
    dstar = max(float(incident_sum(ctx.edge_index,ctx.weight,ctx.anchor.numel()).max()), float(ctx.weight.max()) if ctx.weight.numel() else 0.)
    m1 = 1+cfg.nu*(1+cfg.n/4)
    m2 = cfg.nu*cfg.n*(cfg.n+1)/cfg.c
    cr = dstar*(1.5*m2+2*m1)
    mb = min(cfg.mu,cfg.tau_a)
    if not math.isfinite(cr): raise ValueError('nonfinite stability bound')
    if 1.25*cfg.coupling*cr > cfg.sigma*mb:
        raise ValueError('analytic contraction guard failed; lower coupling or nonlinear gain')
    eta = 1/mb
    q = (1+eta*1.25*cfg.coupling*cr)/(1+eta*mb)
    return Certificate(dstar,cr,cfg.sigma*mb/(1.25*cr) if cr else math.inf,
                       min(cfg.mu,4*cfg.tau_a)-cfg.coupling*cr,eta,q)


def _bounds(ctx,cfg):
    v0 = torch.logit(ctx.base)
    floor = math.log(cfg.epsilon_a)-math.log1p(-cfg.epsilon_a)
    return v0,torch.clamp(v0-cfg.logit_radius,min=floor),torch.clamp(v0+cfg.logit_radius,max=-floor)


def mirror_step(theta, logits, ctx, cfg, cert):
    v0,lo,hi = _bounds(ctx,cfg)
    gt,ga = reward_gradients(theta,torch.sigmoid(logits),ctx,cfg)
    next_theta = (theta+cert.eta*cfg.mu*ctx.anchor+cert.eta*cfg.coupling*gt)/(1+cert.eta*cfg.mu)
    next_v = (logits+cert.eta*cfg.tau_a*v0+cert.eta*cfg.coupling*ga)/(1+cert.eta*cfg.tau_a)
    return torch.clamp(next_theta,ctx.anchor-cfg.phase_radius,ctx.anchor+cfg.phase_radius),torch.clamp(next_v,lo,hi)


def solve_phase_mirror(ctx, cfg=KernelConfig(), previous_theta=None):
    cert = stability_certificate(ctx,cfg)
    v0,_,_ = _bounds(ctx,cfg)
    if previous_theta is not None:
        if previous_theta.shape != ctx.anchor.shape or previous_theta.dtype != ctx.anchor.dtype or previous_theta.device != ctx.anchor.device:
            raise ValueError('invalid warm start')
        _finite(previous_theta)
    if ctx.base.numel() == 0 or cfg.coupling == 0:
        zero = ctx.anchor.new_zeros(())
        return SolveResult(ctx.anchor,ctx.base,v0,0,zero,zero,cert)
    theta = ctx.anchor if previous_theta is None else torch.clamp(previous_theta,ctx.anchor-cfg.phase_radius,ctx.anchor+cfg.phase_radius)
    v = v0
    for _ in range(cfg.steps):
        theta,v = mirror_step(theta,v,ctx,cfg,cert)
    _finite(theta,v)
    # Diagnostic map evaluation is detached and never committed as a K+1 update.
    with torch.no_grad():
        nt,nv = mirror_step(theta,v,ctx,cfg,cert)
        residual = torch.maximum(torch.linalg.vector_norm(nt-theta),torch.linalg.vector_norm(nv-v))
        _finite(residual)
    return SolveResult(theta,torch.sigmoid(v),v,cfg.steps,residual,residual/(1-cert.contraction_bound),cert)


@torch.no_grad()
def solve_phase_euclidean_audit(ctx, cfg=KernelConfig(), *, max_steps=2000, tolerance=1e-12):
    """Independent Euclidean adhesion prox using bracketed monotone roots.

    Validation-only solver; no autograd, no native execution and no production use.
    """
    cert = stability_certificate(ctx,cfg)
    if max_steps < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('invalid audit iteration budget or tolerance')
    v0,lo,hi = _bounds(ctx,cfg)
    theta,a = ctx.anchor.clone(),ctx.base.clone()
    if a.numel() == 0 or cfg.coupling == 0: return theta,a,0
    eta = min(1.,1/max(cfg.coupling*cert.reward_hessian_bound,1e-30))
    alo,ahi = torch.sigmoid(lo),torch.sigmoid(hi)
    for step in range(max_steps):
        gt,ga = reward_gradients(theta,a,ctx,cfg)
        nt = torch.clamp((theta+eta*cfg.mu*ctx.anchor+eta*cfg.coupling*gt)/(1+eta*cfg.mu),ctx.anchor-cfg.phase_radius,ctx.anchor+cfg.phase_radius)
        z = a+eta*cfg.coupling*ga
        lower,upper = alo.clone(),ahi.clone()
        for _ in range(64):
            mid = (lower+upper)/2
            f = mid+eta*cfg.tau_a*(torch.logit(mid)-v0)-z
            lower = torch.where(f < 0,mid,lower)
            upper = torch.where(f >= 0,mid,upper)
        na = (lower+upper)/2
        change = max(float((nt-theta).abs().max()),float((na-a).abs().max()))
        theta,a = nt,na
        if change < tolerance: return theta,a,step+1
    raise RuntimeError('audit solver did not converge within its budget')
