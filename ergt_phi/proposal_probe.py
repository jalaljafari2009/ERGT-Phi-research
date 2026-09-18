"""Native proposal features for an information-only M2 diagnostic."""
import torch
from .contextual_shadow import native_field_history
from .shadow_data import collate_matched_topology_examples


@torch.no_grad()
def cache_native_proposal_pairs(baseline,examples,tokenizer):
    if baseline.training or any(p.requires_grad for p in baseline.parameters()):
        raise ValueError('proposal provider must be frozen and in evaluation mode')
    records=[]
    for index,example in enumerate(examples):
        batch=collate_matched_topology_examples((example,),tokenizer)
        semantic,identity=baseline.raw_input_adapter(**batch.model_inputs())
        state=baseline.substrate(semantic,batch.base.attention_mask,identity_fibre=identity)
        proposal=baseline.probe_native_proposals(state,batch.base.attention_mask)
        # Full pair proposals are computed from raw tokens and native fields.
        # Gold endpoints index the loss record only, after this forward pass.
        pair=torch.cat((proposal['relation_logits'][0,:,:],proposal['event_pair_normalized'][0],
                        proposal['event_pair_probability'][0],proposal['edge_evidence'][0,:,:,None]),dim=-1).cpu()
        events=torch.tensor([[e.source_position,e.target_position] for e in example.base.edges],dtype=torch.long)
        labels=torch.tensor([e.relation_id-1 for e in example.base.edges],dtype=torch.long)
        rows=torch.stack([pair[i,j] for i,j in events.tolist()])
        records.append({'pair_features':rows,'events':events,'labels':labels,'example_id':example.example_id,'pair_id':example.pair_id})
        if (index+1)%50==0: print(f'Cached proposal pairs: {index+1}/{len(examples)}',flush=True)
    return records
