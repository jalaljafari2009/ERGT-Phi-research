"""Test whether native field context contains relation information before M3."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ergt_phi.research_paths import legacy_input, workspace_path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch
from ergt_phi.shadow_phase import PhaseAnchorNetwork,ShadowPhaseModel
from ergt_phi.shadow_data import training_only,pair_partition,cache_training_fields
from ergt_phi.contextual_shadow import ContextualShadowModel,cache_context_histories
from ergt_phi.information_probe import history_pair_features,train_probe
from ergt_phi.native_steps import SteppedNative
from ergt_reviewer.native_solver import ERGT43Config
from ergt_reviewer.matched_data import RawTokenInputContract


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out=ROOT/'runs/m2_information'; out.mkdir(parents=True,exist_ok=True); run=legacy_input(ROOT, "runs/imported_m0/m0_single_seed_reference")
    state=torch.load(run/'opt_12011_data_16301/native_ergt_training.pt',map_location='cpu',weights_only=True); config=dict(state['config']); config['raw_input_contract']=RawTokenInputContract(**config['raw_input_contract'])
    base=SteppedNative(ERGT43Config(**config)).eval(); base.load_state_dict(state['best_state']); base.requires_grad_(False)
    examples,tokenizer,train_hash=training_only(run); fit,monitor=pair_partition(examples,20092026)
    print('Caching field history...',flush=True)
    contextual=ContextualShadowModel(base,PhaseAnchorNetwork(base.config.hidden_dim*2,base.config.n_worlds,3)); history=cache_context_histories(contextual,examples,tokenizer)
    token=cache_training_fields(ShadowPhaseModel(base,PhaseAnchorNetwork(base.config.hidden_dim,base.config.n_worlds,3)),examples,tokenizer)
    results={}; started=time.perf_counter()
    for name,records,mode in (('psi0',token,'final'),('native_final',history,'final'),('native_history',history,'all')):
        train_x,train_y=history_pair_features(records,fit,mode); monitor_x,monitor_y=history_pair_features(records,monitor,mode)
        print(f'Probe {name}: features={train_x.shape[1]} events={len(train_y)}',flush=True)
        results[name+'_true']=train_probe(train_x,train_y,monitor_x,monitor_y,seed=24092026+len(results),shuffle=False)
        results[name+'_shuffled']=train_probe(train_x,train_y,monitor_x,monitor_y,seed=24092026+len(results),shuffle=True)
    protocol={'stage':'M2_context_information_probe','train_sha256':train_hash,'fit_examples':len(fit),'monitor_examples':len(monitor),'feature_modes':['psi0','native_final','native_history'],'probe':'LayerNorm+linear relation classifier','epochs':40,'native_integration':False,'source_sha256':{f:digest(ROOT/f) for f in ['ergt_phi/information_probe.py','ergt_phi/contextual_shadow.py','scripts/run_m2_information_probe.py']}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)); result={'status':'completed_information_probe','M2_complete':False,'M3_authorized_by_results':False,'results':results,'protocol_sha256':digest(out/'protocol.json'),'elapsed_seconds':time.perf_counter()-started,'interpretation':'probe measures accessible relation information; it is not native accuracy or a promotion gate'}
    (out/'result.json').write_text(json.dumps(result,indent=2)); (workspace_path(ROOT, "manifests/m2_information.json")).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v['final_monitor']['balanced_accuracy'] for k,v in results.items()},indent=2))


if __name__=='__main__': main()
