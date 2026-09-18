"""Compare token, contextual-token and pair endpoint shadow representations."""
from pathlib import Path
import sys,json,hashlib,random,time
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from ergt_phi.shadow_phase import CalibrationConfig,PhaseAnchorNetwork,ShadowPhaseModel,calibration_step
from ergt_phi.shadow_data import training_only,pair_partition,cache_training_fields,pack_records
from ergt_phi.contextual_shadow import ContextualShadowModel,cache_context_fields
from ergt_phi.pair_diagnostic import PairRelationNetwork,PairDiagnosticConfig,pair_loss
from ergt_phi.native_steps import SteppedNative
from ergt_reviewer.native_solver import ERGT43Config
from ergt_reviewer.matched_data import RawTokenInputContract


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


@torch.no_grad()
def metrics_phase(model,records,indices):
    model.eval(); confusion=torch.zeros(3,3,dtype=torch.long); ce=0.; total=0
    for start in range(0,len(indices),model.config.batch_size):
        psi,mask,events,labels=pack_records(records,indices[start:start+model.config.batch_size])
        from ergt_phi.shadow_phase import relation_logits
        logits=relation_logits(model(psi,mask),model.offsets,events,model.config.temperature)
        ce+=float(F.cross_entropy(logits,labels,reduction='sum')); total+=labels.numel()
        pred=logits.argmax(-1); confusion+=torch.bincount(labels*3+pred,minlength=9).reshape(3,3)
    counts=confusion.sum(1); recall=confusion.diag().double()/counts.clamp_min(1)
    return {'ce':ce/total,'accuracy':float(confusion.diag().sum())/total,'balanced_accuracy':float(recall.mean()),'per_class_recall':recall.tolist(),'class_counts':counts.tolist(),'confusion_matrix':confusion.tolist(),'events':total}


@torch.no_grad()
def metrics_pair(model,records,indices):
    model.eval(); confusion=torch.zeros(3,3,dtype=torch.long); ce=0.; total=0
    for start in range(0,len(indices),model.config.batch_size):
        features,mask,events,labels=pack_records(records,indices[start:start+model.config.batch_size])
        logits=model(features,events); ce+=float(F.cross_entropy(logits,labels,reduction='sum')); total+=labels.numel()
        pred=logits.argmax(-1); confusion+=torch.bincount(labels*3+pred,minlength=9).reshape(3,3)
    counts=confusion.sum(1); recall=confusion.diag().double()/counts.clamp_min(1)
    return {'ce':ce/total,'accuracy':float(confusion.diag().sum())/total,'balanced_accuracy':float(recall.mean()),'per_class_recall':recall.tolist(),'class_counts':counts.tolist(),'confusion_matrix':confusion.tolist(),'events':total}


def run_phase(records,fit,monitor,source,shuffled):
    cfg=CalibrationConfig(); model=PhaseAnchorNetwork(records[0]['psi'].shape[-1],8,3,cfg); opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    initial=metrics_phase(model,records,monitor); progress={'step':0,'frozen':False}
    contract={'stage':'M2_diagnostic_phase','source':source,'shuffled_labels':shuffled,'config':asdict(cfg)}
    for epoch in range(cfg.epochs):
        order=list(fit); random.Random(cfg.shuffle_seed+epoch).shuffle(order)
        for start in range(0,len(order),cfg.batch_size):
            ids=order[start:start+cfg.batch_size]; tensors=list(pack_records(records,ids))
            if shuffled:
                g=torch.Generator().manual_seed(cfg.shuffle_seed+epoch*1000+start); tensors[3]=tensors[3][torch.randperm(tensors[3].numel(),generator=g)]
            calibration_step(model,opt,tuple(tensors),contract=contract,progress=progress)
    return {'initial_monitor':initial,'final_fit':metrics_phase(model,records,fit),'final_monitor':metrics_phase(model,records,monitor),'steps':progress['step']}


def run_pair(records,fit,monitor,shuffled):
    cfg=PairDiagnosticConfig(); model=PairRelationNetwork(records[0]['psi'].shape[-1],cfg); opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    initial=metrics_pair(model,records,monitor)
    for epoch in range(cfg.epochs):
        order=list(fit); random.Random(41000+epoch).shuffle(order)
        for start in range(0,len(order),cfg.batch_size):
            features,mask,events,labels=pack_records(records,order[start:start+cfg.batch_size])
            if shuffled:
                g=torch.Generator().manual_seed(42000+epoch*1000+start); labels=labels[torch.randperm(labels.numel(),generator=g)]
            opt.zero_grad(set_to_none=True); loss=pair_loss(model,features,events,labels); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
    return {'initial_monitor':initial,'final_fit':metrics_pair(model,records,fit),'final_monitor':metrics_pair(model,records,monitor),'steps':cfg.epochs*((len(fit)+cfg.batch_size-1)//cfg.batch_size)}


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out=ROOT/'runs/m2_diagnostic'; out.mkdir(parents=True,exist_ok=True); run=ROOT/'runs/imported_m0/m0_single_seed_reference'
    state=torch.load(run/'opt_12011_data_16301/native_ergt_training.pt',map_location='cpu',weights_only=True); c=dict(state['config']); c['raw_input_contract']=RawTokenInputContract(**c['raw_input_contract'])
    base=SteppedNative(ERGT43Config(**c)).eval(); base.load_state_dict(state['best_state']); base.requires_grad_(False)
    examples,tokenizer,train_hash=training_only(run); fit,monitor=pair_partition(examples,20092026)
    print('Caching token and contextual features...',flush=True); token_records=cache_training_fields(ShadowPhaseModel(base,PhaseAnchorNetwork(base.config.hidden_dim,base.config.n_worlds,3)),examples,tokenizer)
    context_phase=PhaseAnchorNetwork(base.config.hidden_dim*2,base.config.n_worlds,3); contextual=ContextualShadowModel(base,context_phase); context_records=cache_context_fields(contextual,examples,tokenizer)
    results={}; started=time.perf_counter()
    for name,records in (('token',token_records),('contextual',context_records)):
        for shuffled in (False,True):
            key=name+('_shuffled' if shuffled else '_true'); print('Training',key,flush=True); results[key]=run_phase(records,fit,monitor,name,shuffled)
    for shuffled in (False,True):
        key='pair'+('_shuffled' if shuffled else '_true'); print('Training',key,flush=True); results[key]=run_pair(context_records,fit,monitor,shuffled)
    protocol={'stage':'M2_representation_diagnostic','data_train_sha256':train_hash,'fit_count':len(fit),'monitor_count':len(monitor),'epochs':20,'batch_size':16,'shuffle_controls':True,'native_integration':False,'answer_claim':False,'source_sha256':{f:digest(ROOT/f) for f in ['ergt_phi/shadow_phase.py','ergt_phi/shadow_data.py','ergt_phi/contextual_shadow.py','ergt_phi/pair_diagnostic.py','scripts/run_m2_diagnostic.py']}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));
    result={'status':'completed_diagnostic','M2_complete':False,'M3_authorized_by_results':False,'results':results,'elapsed_seconds':time.perf_counter()-started,'protocol_sha256':digest(out/'protocol.json'),'interpretation':{'true_pair_vs_context_required':'compare monitor balanced accuracy','shuffled_control':'should remain near chance; it is not a qualification gate','pair_head_is_loss_only':True}}
    (out/'result.json').write_text(json.dumps(result,indent=2)); (ROOT/'manifests/m2_diagnostic.json').write_text(json.dumps(result,indent=2)); print(json.dumps({k:v['final_monitor']['balanced_accuracy'] for k,v in results.items()},indent=2))


if __name__=='__main__': main()
