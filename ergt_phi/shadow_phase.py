"""M2 shadow anchors: structured training loss, no governing native insertion."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .checkpoint import capture, restore


@dataclass(frozen=True)
class CalibrationConfig:
    hidden_dim: int = 64
    temperature: float = .2
    offset_radius: float = .25
    init_seed: int = 18092026
    shuffle_seed: int = 19092026
    split_seed: int = 20092026
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = .002
    weight_decay: float = .0001
    offset_penalty: float = .001
    readiness_balanced_accuracy: float = .70
    readiness_min_recall: float = .50
    readiness_ce_reduction: float = .05
    minimum_offset_separation: float = 1.0
    maximum_phase_resultant: float = .98
    required_windows: int = 2

    def __post_init__(self):
        for name in ('hidden_dim','epochs','batch_size','required_windows'):
            value = getattr(self,name)
            if type(value) is not int or value <= 0: raise ValueError(f'invalid {name}')
        for name in ('temperature','learning_rate'):
            value = getattr(self,name)
            if not math.isfinite(value) or value <= 0: raise ValueError(f'invalid {name}')
        for name in ('offset_radius','weight_decay','offset_penalty','minimum_offset_separation'):
            value = getattr(self,name)
            if not math.isfinite(value) or value < 0: raise ValueError(f'invalid {name}')
        for name in ('readiness_balanced_accuracy','readiness_min_recall','readiness_ce_reduction','maximum_phase_resultant'):
            value = getattr(self,name)
            if not math.isfinite(value) or not 0 <= value <= 1: raise ValueError(f'invalid {name}')


class PhaseAnchorNetwork(nn.Module):
    """pi*tanh(W2*SiLU(W1*LN(Psi0))), independently seeded on CPU."""
    def __init__(self, input_dim, worlds, relations, config=CalibrationConfig()):
        super().__init__()
        if min(input_dim,worlds,relations) < 1: raise ValueError('positive dimensions required')
        if relations > 1 and 2*config.offset_radius >= 2*math.pi/relations:
            raise ValueError('offset displacement permits code collapse')
        self.config = config
        self.worlds,self.relations = worlds,relations
        # CPU-only initialization; CUDA generators are neither used nor reseeded.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.init_seed)
            self.network = nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,config.hidden_dim),nn.SiLU(),nn.Linear(config.hidden_dim,worlds))
        code = torch.arange(relations,dtype=torch.float32)*2*math.pi/relations
        self.register_buffer('initial_offsets',code.unsqueeze(0).repeat(worlds,1))
        self.offset_raw = nn.Parameter(torch.zeros(worlds,relations))

    @property
    def offsets(self):
        return self.initial_offsets+self.config.offset_radius*torch.tanh(self.offset_raw)

    def forward(self, psi0, attention_mask):
        if psi0.ndim != 3 or attention_mask.shape != psi0.shape[:2] or attention_mask.dtype != torch.bool:
            raise ValueError('expected Psi0 [B,N,D] and boolean mask [B,N]')
        if not bool(torch.isfinite(psi0).all()): raise ValueError('nonfinite Psi0')
        anchors = math.pi*torch.tanh(self.network(psi0))
        return (anchors*attention_mask.unsqueeze(-1)).transpose(1,2)

    def freeze(self):
        self.eval()
        self.requires_grad_(False)


class ShadowPhaseModel(nn.Module):
    """Owns a frozen stepped baseline. Gold endpoints never enter its forward."""
    def __init__(self, baseline, phase):
        super().__init__()
        self.baseline,self.phase = baseline,phase
        self.baseline.requires_grad_(False)
        self.baseline.eval()

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        return self

    @torch.no_grad()
    def initial_field(self, raw_token_ids, attention_mask):
        semantic,identity = self.baseline.raw_input_adapter(raw_token_ids,attention_mask)
        state = self.baseline.substrate.seed_native_state(semantic,attention_mask,identity_fibre=identity)
        return state['psi'].detach()

    def anchors(self, raw_token_ids, attention_mask):
        return self.phase(self.initial_field(raw_token_ids,attention_mask),attention_mask)

    def forward(self, raw_token_ids, attention_mask, *, observe_spectrum=False):
        # Native output object and exact governing execution are unchanged.
        return self.baseline(raw_token_ids,attention_mask,observe_spectrum=observe_spectrum)


def relation_logits(anchors, offsets, event_indices, temperature):
    """LOSS/DIAGNOSTIC ONLY: event_indices is [batch, source, target], not topology input."""
    if event_indices.ndim != 2 or event_indices.shape[1] != 3 or event_indices.dtype != torch.long:
        raise ValueError('event indices must have shape [events,3] and dtype int64')
    if not math.isfinite(temperature) or temperature <= 0: raise ValueError('invalid temperature')
    if anchors.ndim != 3 or offsets.ndim != 2 or offsets.shape[0] != anchors.shape[1]:
        raise ValueError('invalid anchor or offset shapes')
    if event_indices.numel():
        b,i,j = event_indices.unbind(-1)
        if bool((event_indices < 0).any()) or bool((b >= anchors.shape[0]).any()) or bool((i >= anchors.shape[2]).any()) or bool((j >= anchors.shape[2]).any()):
            raise ValueError('event index outside phase domain')
        delta = anchors[b,:,j]-anchors[b,:,i]
    else:
        delta = anchors.new_empty((0,anchors.shape[1]))
    kappa = (1+torch.cos(delta[:,:,None]-offsets[None,:,:]))/2
    return kappa.mean(dim=1)/temperature


def phase_loss(phase, psi0, attention_mask, events, labels):
    if not labels.numel(): raise ValueError('empty relation supervision')
    anchors = phase(psi0,attention_mask)
    logits = relation_logits(anchors,phase.offsets,events,phase.config.temperature)
    ce = F.cross_entropy(logits,labels)
    penalty = (phase.offsets-phase.initial_offsets).square().mean()
    return ce+phase.config.offset_penalty*penalty,logits


def calibration_step(phase, optimizer, tensors, *, contract, progress):
    """Atomic update; rollback all research state on nonfinite/numerical failure."""
    if progress.get('frozen') or not any(p.requires_grad for p in phase.parameters()):
        raise RuntimeError('calibration is frozen')
    before = capture(phase,optimizer,contract=contract,progress=progress)
    try:
        optimizer.zero_grad(set_to_none=True)
        loss,_ = phase_loss(phase,*tensors)
        if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite phase loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(phase.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in phase.parameters()):
            raise FloatingPointError('nonfinite phase parameter')
    except Exception:
        progress.clear()
        progress.update(restore(before,phase,optimizer,contract=contract))
        raise
    progress['step'] = progress.get('step',0)+1
    return float(loss.detach()),float(norm)


def readiness(metrics, initial_ce, cfg):
    checks = {
        'all_relation_classes_present': all(n > 0 for n in metrics['class_counts']),
        'balanced_accuracy': metrics['balanced_accuracy'] >= cfg.readiness_balanced_accuracy,
        'minimum_class_recall': min(metrics['per_class_recall']) >= cfg.readiness_min_recall,
        'ce_reduction': metrics['ce'] <= initial_ce*(1-cfg.readiness_ce_reduction),
        'offset_separation': metrics['minimum_offset_separation'] >= cfg.minimum_offset_separation,
        'phase_not_constant': metrics['phase_resultant'] <= cfg.maximum_phase_resultant,
    }
    return all(checks.values()),checks
