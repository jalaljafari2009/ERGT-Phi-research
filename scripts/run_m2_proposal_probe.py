"""Measure label-free native proposal information before considering M3."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ergt_phi.research_paths import legacy_input, workspace_path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from ergt_phi.shadow_data import training_only,pair_partition
from ergt_phi.proposal_probe import cache_native_proposal_pairs
from ergt_phi.information_probe import history_pair_features,train_probe
from ergt_phi.native_steps import SteppedNative
from ergt_reviewer.native_solver import ERGT43Config
from ergt_reviewer.matched_data import RawTokenInputContract


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=ROOT/'runs/m2_proposal';out.mkdir(parents=True,exist_ok=True);run=legacy_input(ROOT, "runs/imported_m0/m0_single_seed_reference")
    state=torch.load(run/'opt_12011_data_16301/native_ergt_training.pt',map_location='cpu',weights_only=True);c=dict(state['config']);c['raw_input_contract']=RawTokenInputContract(**c['raw_input_contract'])
    base=SteppedNative(ERGT43Config(**c)).eval();base.load_state_dict(state['best_state']);base.requires_grad_(False)
    examples,tokenizer,train_hash=training_only(run);fit,monitor=pair_partition(examples,20092026)
    print('Caching native proposal matrices...',flush=True);records=cache_native_proposal_pairs(base,examples,tokenizer)
    train_x,train_y=history_pair_features(records,fit,'final');monitor_x,monitor_y=history_pair_features(records,monitor,'final')
    results={'proposal_true':train_probe(train_x,train_y,monitor_x,monitor_y,seed=25092026,epochs=40,shuffle=False),'proposal_shuffled':train_probe(train_x,train_y,monitor_x,monitor_y,seed=25092027,epochs=40,shuffle=True)}
    protocol={'stage':'M2_native_proposal_information_probe','train_sha256':train_hash,'fit_examples':len(fit),'monitor_examples':len(monitor),'feature_dim':train_x.shape[1],'epochs':40,'uses_gold_only_for_post_forward_indexing':True,'native_answer_head_used':False,'source_sha256':{f:digest(ROOT/f) for f in ['ergt_phi/proposal_probe.py','ergt_phi/information_probe.py','scripts/run_m2_proposal_probe.py']}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));result={'status':'completed_proposal_probe','M2_complete':False,'M3_authorized_by_results':False,'results':results,'protocol_sha256':digest(out/'protocol.json'),'interpretation':'label-free native proposals are measured before phase activation'}
    (out/'result.json').write_text(json.dumps(result,indent=2));(workspace_path(ROOT, "manifests/m2_proposal.json")).write_text(json.dumps(result,indent=2));print(json.dumps({k:v['final_monitor'] for k,v in results.items()},indent=2))


if __name__=='__main__':main()
