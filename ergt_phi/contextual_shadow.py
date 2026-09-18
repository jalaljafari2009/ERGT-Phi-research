"""Explicit M2 anchor revision: fixed native context, still no governing feedback."""
import torch
from .shadow_phase import ShadowPhaseModel
from .shadow_data import collate_matched_topology_examples


@torch.no_grad()
def native_field_history(baseline, raw_token_ids, attention_mask):
    """Seed + completed native field states; no gold sidecars, probes or answers."""
    if baseline.training or any(p.requires_grad for p in baseline.parameters()):
        raise ValueError('context provider must be frozen and in evaluation mode')
    semantic,identity=baseline.raw_input_adapter(raw_token_ids,attention_mask)
    state=baseline.substrate.seed_native_state(semantic,attention_mask,identity_fibre=identity)
    states=[state['psi'].detach()]
    for _ in range(baseline.config.field_steps):
        obs=baseline.substrate.observe_native_state(state)
        geometry=baseline.substrate.preview_native_geometry(state,obs)
        state=baseline.substrate.advance_native_field(state,geometry)
        states.append(state['psi'].detach())
    return states


class ContextualShadowModel(ShadowPhaseModel):
    """Anchor input = concat(Psi0, detached final native Psi), fixed per raw input."""
    @torch.no_grad()
    def context_features(self, raw_token_ids, attention_mask):
        states=native_field_history(self.baseline,raw_token_ids,attention_mask)
        return torch.cat((states[0],states[-1]),dim=-1)

    def anchors(self, raw_token_ids, attention_mask):
        return self.phase(self.context_features(raw_token_ids,attention_mask),attention_mask)


@torch.no_grad()
def cache_context_fields(shadow,examples,tokenizer):
    records=[]
    for index,example in enumerate(examples):
        batch=collate_matched_topology_examples((example,),tokenizer)
        features=shadow.context_features(**batch.model_inputs())[0].cpu()
        # Gold structure indexes the LOSS only, after raw-only feature extraction.
        events=torch.tensor([[e.source_position,e.target_position] for e in example.base.edges],dtype=torch.long)
        labels=torch.tensor([e.relation_id-1 for e in example.base.edges],dtype=torch.long)
        records.append({'psi':features,'events':events,'labels':labels,'example_id':example.example_id,'pair_id':example.pair_id})
        if (index+1)%50==0: print(f'Cached native context: {index+1}/{len(examples)}',flush=True)
    return records
