"""Regenerate ONLY the accepted reference's training cohort and partition by pair."""
import hashlib
import json
import random
from collections import defaultdict
import torch
from .runtime import activate_reference
activate_reference()
from ergt_reviewer.fair_data_v9 import (_balanced_training_cohort, _unique_cohort,
    make_unsupported, append_training_distractors, protocol_tokenizer)
from ergt_reviewer.matched_data import manifest_hash, collate_matched_topology_examples


def training_only(run_root):
    protocol = json.loads((run_root/'protocol.json').read_text())
    cfg = protocol['config']
    seed = int(cfg['data_seeds'][0])
    seen = set()
    supported = _balanced_training_cohort(pair_counts_by_hop=cfg['training_hop_pair_counts'],
        seed=seed,split='train',node_label_pool_size=cfg['node_label_pool_size'],forbidden_raw_texts=seen)
    unsupported = make_unsupported(_unique_cohort(pair_count=cfg['train_unsupported_pairs'],
        seed=seed+3,split='train_unsupported',min_hops=cfg['train_min_hops'],max_hops=cfg['train_max_hops'],
        node_label_pool_size=cfg['node_label_pool_size'],forbidden_raw_texts=seen))
    examples = append_training_distractors((*supported,*unsupported),cfg['training_distractor_tokens'])
    expected = next(iter(json.loads((run_root/'data_manifest.json').read_text()).values()))['train_sha256']
    if manifest_hash(examples) != expected: raise ValueError('regenerated training cohort hash mismatch')
    return examples,protocol_tokenizer(cfg['node_label_pool_size']),expected


def pair_partition(examples, seed):
    groups = defaultdict(list)
    for i,x in enumerate(examples): groups[x.pair_id].append(i)
    # Pair-level split, stratified by original train split (hop/scenario) when possible.
    strata = defaultdict(list)
    for pair,ids in groups.items(): strata[examples[ids[0]].base.split].append(pair)
    rng = random.Random(seed)
    fit,monitor = [],[]
    for key in sorted(strata):
        pairs = sorted(strata[key]); rng.shuffle(pairs)
        count = max(1,len(pairs)//5) if len(pairs)>1 else 0
        for index,pair in enumerate(pairs):
            (monitor if index<count else fit).extend(groups[pair])
    assert set(fit).isdisjoint(monitor) and len(fit)+len(monitor)==len(examples)
    if not fit or not monitor: raise ValueError('training cohort too small for calibration partition')
    return sorted(fit),sorted(monitor)


@torch.no_grad()
def cache_training_fields(shadow, examples, tokenizer):
    records = []
    for index,example in enumerate(examples):
        batch = collate_matched_topology_examples((example,),tokenizer)
        psi = shadow.initial_field(**batch.model_inputs())[0].cpu()
        events = torch.tensor([[e.source_position,e.target_position] for e in example.base.edges],dtype=torch.long)
        labels = torch.tensor([e.relation_id-1 for e in example.base.edges],dtype=torch.long)
        records.append({'psi':psi,'events':events,'labels':labels,'example_id':example.example_id,'pair_id':example.pair_id})
        if (index+1)%100 == 0: print(f'Cached Psi0: {index+1}/{len(examples)}',flush=True)
    return records


def pack_records(records, indices):
    subset = [records[i] for i in indices]
    n = max(r['psi'].shape[0] for r in subset)
    psi = subset[0]['psi'].new_zeros((len(subset),n,subset[0]['psi'].shape[1]))
    mask = torch.zeros((len(subset),n),dtype=torch.bool)
    events,labels = [],[]
    for b,r in enumerate(subset):
        length = r['psi'].shape[0]
        psi[b,:length] = r['psi']; mask[b,:length] = True
        events.append(torch.cat((torch.full((len(r['events']),1),b,dtype=torch.long),r['events']),dim=1))
        labels.append(r['labels'])
    return psi,mask,torch.cat(events),torch.cat(labels)


def feature_conflicts(records):
    """Observed ambiguity at identical Psi0 endpoints; not a generalization metric."""
    counts = defaultdict(lambda:[0,0,0])
    total = 0
    for r in records:
        for (i,j),label in zip(r['events'].tolist(),r['labels'].tolist()):
            key = hashlib.sha256(torch.cat((r['psi'][i],r['psi'][j])).numpy().tobytes()).hexdigest()
            counts[key][label]+=1; total+=1
    conflicting = [x for x in counts.values() if sum(v>0 for v in x)>1]
    return {'events':total,'distinct_endpoint_features':len(counts),'conflicting_feature_groups':len(conflicting),
            'events_in_conflicting_groups':sum(sum(x) for x in conflicting),
            'empirical_feature_memorization_accuracy_ceiling':sum(max(x) for x in counts.values())/total}
