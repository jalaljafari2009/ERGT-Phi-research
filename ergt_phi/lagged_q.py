"""Sparse, lagged native proposal context for the future phase patch."""
from dataclasses import dataclass
import torch


def _masked_softmax(logits, valid):
    if logits.ndim != 2 or valid.ndim != 1 or logits.shape[1] != valid.numel():
        raise ValueError('logits/mask shape mismatch')
    if valid.dtype != torch.bool or not bool(valid.any()):
        raise ValueError('proposal domain must contain a valid token')
    masked=logits.masked_fill(~valid[None,:],float('-inf'))
    return torch.softmax(masked,dim=-1)


def build_sparse_q(source_logits,target_logits,relation_logits,edge_index,*,valid_slot=None,node_mask=None):
    """Build Q_e,r on requested candidate pairs, with detached proposal context.

    relation_logits includes the null class in column zero. Its non-null softmax
    mass is event presence and is consumed once; no second presence multiplication.
    """
    if source_logits.ndim != 2 or target_logits.shape != source_logits.shape or relation_logits.ndim != 2 or relation_logits.shape[0] != source_logits.shape[0] or relation_logits.shape[1] < 2:
        raise ValueError('proposal slot shapes mismatch')
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long or edge_index.device != source_logits.device:
        raise ValueError('edge_index must be [2,E] int64 on the proposal device')
    u,n=source_logits.shape
    if node_mask is None: node_mask=torch.ones(n,dtype=torch.bool,device=source_logits.device)
    if valid_slot is None: valid_slot=torch.ones(u,dtype=torch.bool,device=source_logits.device)
    if node_mask.shape != (n,) or valid_slot.shape != (u,) or node_mask.dtype != torch.bool or valid_slot.dtype != torch.bool:
        raise ValueError('invalid node or slot mask')
    if edge_index.numel() and (int(edge_index.min())<0 or int(edge_index.max())>=n): raise ValueError('edge outside node domain')
    if bool((edge_index[0]==edge_index[1]).any()): raise ValueError('self edge in phase patch')
    q,_,_=build_sparse_q_with_presence(source_logits,target_logits,relation_logits,edge_index,valid_slot=valid_slot,node_mask=node_mask)
    return q


def build_sparse_q_with_presence(source_logits,target_logits,relation_logits,edge_index,*,valid_slot=None,node_mask=None):
    """Return (Q, q_mass, pi) for diagnostics; Q already includes event presence once."""
    if source_logits.ndim != 2 or target_logits.shape != source_logits.shape:
        raise ValueError('source/target logits must both be [U,N]')
    if relation_logits.ndim != 2 or relation_logits.shape[0] != source_logits.shape[0] or relation_logits.shape[1] < 2:
        raise ValueError('relation logits must be [U,R+1]')
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long or edge_index.device != source_logits.device:
        raise ValueError('edge_index must be [2,E] int64 on the proposal device')
    u,n=source_logits.shape
    if valid_slot is None: valid_slot=torch.ones(u,dtype=torch.bool,device=source_logits.device)
    if node_mask is None: node_mask=torch.ones(n,dtype=torch.bool,device=source_logits.device)
    if valid_slot.shape != (u,) or node_mask.shape != (n,) or valid_slot.dtype != torch.bool or node_mask.dtype != torch.bool:
        raise ValueError('invalid node or slot mask')
    if valid_slot.device != source_logits.device or node_mask.device != source_logits.device:
        raise ValueError('node and slot masks must share the proposal device')
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= n):
        raise ValueError('edge outside node domain')
    if edge_index.numel() and bool((edge_index[0] == edge_index[1]).any()):
        raise ValueError('self edge in phase patch')
    psrc=_masked_softmax(source_logits,node_mask)*valid_slot[:,None]
    ptgt=_masked_softmax(target_logits,node_mask)*valid_slot[:,None]
    full=torch.softmax(relation_logits,dim=-1); nonnull=full[:,1:]*valid_slot[:,None]
    source=psrc[:,edge_index[0]]; target=ptgt[:,edge_index[1]]
    q=torch.einsum('ue,ur->er',source*target,nonnull).detach()
    mass=q.sum(-1); pi=torch.where(mass[:,None]>0,q/mass[:,None],torch.zeros_like(q)).detach()
    return q,mass,pi


@dataclass
class LaggedQ:
    """Q^0 is empty; commit happens only after an outer native step completes."""
    previous: torch.Tensor | None = None

    def consume(self, *, device=None, dtype=None):
        if self.previous is None:
            if device is None: device=torch.device('cpu')
            if dtype is None: dtype=torch.float32
            return torch.empty((0,0),device=device,dtype=dtype)
        return self.previous.detach().clone()

    def commit(self,q_next):
        if not isinstance(q_next,torch.Tensor) or q_next.ndim != 2:
            raise ValueError('q_next must be sparse [E,R] tensor')
        if not bool(torch.isfinite(q_next).all()) or q_next.requires_grad:
            raise ValueError('Q must be finite and detached before commit')
        self.previous=q_next.detach().clone()


def relation_context_from_slots(source_logits,target_logits,relation_logits,edge_index,*,node_mask=None,valid_slot=None):
    """Typed adapter for one example/world's ProbeNative slot tensors."""
    return build_sparse_q_with_presence(source_logits,target_logits,relation_logits,edge_index,node_mask=node_mask,valid_slot=valid_slot)


def native_proposal_q(proposal, edge_index, *, attention_mask=None, batch_index=0):
    """Extract one example's detached sparse Q from ``ProbeNative`` output.

    ``ProbeNative`` names the three slot tensors ``event_source_logits``,
    ``event_target_logits`` and ``event_relation_logits``.  The phase patch
    consumes only these proposal tensors; answer selection and gold labels are
    deliberately absent from this adapter.
    """
    required = ('event_source_logits', 'event_target_logits', 'event_relation_logits')
    missing = [key for key in required if key not in proposal]
    if missing:
        raise ValueError(f'missing native proposal fields: {missing}')

    def one(value, name):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'{name} must be a tensor')
        if value.ndim == 0:
            raise ValueError(f'{name} must have a batch or slot dimension')
        if value.ndim >= 3:
            if value.shape[0] <= batch_index:
                raise ValueError('batch_index outside proposal batch')
            return value[batch_index]
        return value

    source = one(proposal['event_source_logits'], 'event_source_logits')
    target = one(proposal['event_target_logits'], 'event_target_logits')
    relation = one(proposal['event_relation_logits'], 'event_relation_logits')
    if source.ndim != 2 or target.ndim != 2 or relation.ndim != 2:
        raise ValueError('native proposal slot tensors must be batched [B,U,N]/[B,U,R+1]')
    n = source.shape[1]
    if attention_mask is None:
        node_mask = torch.ones(n, dtype=torch.bool, device=source.device)
    else:
        mask = attention_mask
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError('attention_mask must be boolean')
        if mask.ndim == 2:
            if mask.shape[0] <= batch_index:
                raise ValueError('batch_index outside attention mask batch')
            node_mask = mask[batch_index]
        elif mask.ndim == 1:
            node_mask = mask
        else:
            raise ValueError('attention_mask must be [B,N] or [N]')
        if node_mask.shape != (n,) or node_mask.device != source.device:
            raise ValueError('attention mask does not match native proposal domain')
    return relation_context_from_slots(
        source, target, relation, edge_index,
        node_mask=node_mask, valid_slot=node_mask,
    )


def select_native_patch(edge_evidence, attention_mask, *, top_k=32):
    """Select a preview-only sparse candidate list without gold topology.

    The native edge evidence determines the candidate order.  Q mass is checked
    later by ``prepare_patch``; this function only applies padding and self-edge
    exclusions and returns the corresponding native base values.
    """
    if edge_evidence.ndim != 2 or attention_mask.ndim != 1:
        raise ValueError('edge evidence/mask must be [N,N] and [N]')
    if edge_evidence.shape != (attention_mask.numel(), attention_mask.numel()):
        raise ValueError('edge evidence and mask domain mismatch')
    if attention_mask.dtype != torch.bool or attention_mask.device != edge_evidence.device:
        raise ValueError('attention mask must be boolean on the proposal device')
    if type(top_k) is not int or top_k <= 0:
        raise ValueError('top_k must be a positive integer')
    valid = attention_mask[:, None] & attention_mask[None, :]
    valid = valid & ~torch.eye(attention_mask.numel(), dtype=torch.bool, device=edge_evidence.device)
    scores = edge_evidence.detach().masked_fill(~valid, float('-inf'))
    flat = scores.flatten()
    count = min(top_k, int(valid.sum()))
    if count == 0:
        return edge_evidence.new_empty((2, 0), dtype=torch.long), edge_evidence.new_empty((0,))
    values, positions = torch.topk(flat, count)
    keep = torch.isfinite(values)
    positions = positions[keep]
    edges = torch.stack((positions // attention_mask.numel(), positions % attention_mask.numel()))
    return edges.to(dtype=torch.long), edge_evidence.detach()[edges[0], edges[1]]
