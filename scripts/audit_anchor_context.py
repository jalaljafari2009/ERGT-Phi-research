"""Check whether changing a remote relation reaches native endpoint field states."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from ergt_phi.native_steps import SteppedNative
from ergt_phi.contextual_shadow import native_field_history
from ergt_phi.shadow_data import training_only,pair_partition
from ergt_phi.shadow_phase import CalibrationConfig
from ergt_reviewer.native_solver import ERGT43Config
from ergt_reviewer.matched_data import RawTokenInputContract,collate_matched_topology_examples,raw_token_fingerprint_id
from ergt_reviewer.data_schema import RELATION_SURFACES


def load_reference():
    run=ROOT/'runs/imported_m0/m0_single_seed_reference'
    path=run/'opt_12011_data_16301/native_ergt_training.pt'
    accepted=json.loads((ROOT/'manifests/trained_m0_audit.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==accepted['checkpoint_sha256']
    state=torch.load(path,map_location='cpu',weights_only=True)
    config=dict(state['config']);config['raw_input_contract']=RawTokenInputContract(**config['raw_input_contract'])
    baseline=SteppedNative(ERGT43Config(**config)).eval()
    baseline.load_state_dict(state['best_state'],strict=True);baseline.requires_grad_(False)
    return run,path,baseline


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    run,cp,baseline=load_reference()
    examples,tokenizer,_=training_only(run)
    fit,_=pair_partition(examples,CalibrationConfig().split_seed)
    selected={}
    for index in fit:
        example=examples[index]
        hops=len(example.base.edges)//2
        selected.setdefault(hops,index)
    protocol={'scope':'fit_partition_only','examples_by_hop':selected,
              'edits':'first event, each of the two other non-null relation words',
              'observable':'maximum absolute difference at canonical source/target Psi',
              'threshold':1e-6,'pass_fraction':.9,'candidate':'final_completed_native_field',
              'claims':'Sensitivity only; not proof of decodable or sufficient relation information.'}
    out=ROOT/'runs/m2_context';out.mkdir(parents=True,exist_ok=True)
    (out/'sensitivity_protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for hops,index in sorted(selected.items()):
        example=examples[index];edge=example.physical_edges[0]
        batch=collate_matched_topology_examples((example,),tokenizer)
        original=native_field_history(baseline,**batch.model_inputs())
        for relation in (r for r in (1,2,3) if r!=edge.relation_id):
            raw=batch.raw_token_ids.clone()
            raw[0,edge.event_anchor_position]=raw_token_fingerprint_id(RELATION_SURFACES[relation])
            changed=native_field_history(baseline,raw,batch.base.attention_mask)
            delta=[float((a[0,[edge.source_position,edge.target_position]]-b[0,[edge.source_position,edge.target_position]]).abs().max()) for a,b in zip(original,changed)]
            rows.append({'hops':hops,'example_id':example.example_id,'old_relation':edge.relation_id,'new_relation':relation,'endpoint_max_abs_by_step':delta})
            print(json.dumps(rows[-1]),flush=True)
    fraction=sum(row['endpoint_max_abs_by_step'][-1]>protocol['threshold'] for row in rows)/len(rows)
    result={'initial_unchanged':all(r['endpoint_max_abs_by_step'][0]==0 for r in rows),
            'final_sensitive_fraction':fraction,'candidate_pass':fraction>=protocol['pass_fraction'],
            'selected_native_step':baseline.config.field_steps,'rows':rows,
            'base_checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('ergt_phi/contextual_shadow.py','scripts/audit_anchor_context.py')},
            'protocol':protocol}
    (ROOT/'manifests/anchor_context_audit.json').write_text(json.dumps(result,indent=2))
    print('CONTEXT SENSITIVITY: '+str(result['candidate_pass']),flush=True)


if __name__=='__main__':main()
