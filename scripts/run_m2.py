"""Run the preregistered CPU shadow-calibration experiment on M0 training data only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ergt_phi.research_paths import legacy_input, workspace_path
import json
import hashlib
import random
import time
from dataclasses import asdict
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from ergt_phi.shadow_phase import CalibrationConfig,PhaseAnchorNetwork,calibration_step,readiness
from ergt_phi.shadow_phase import ShadowPhaseModel
from ergt_phi.contextual_shadow import ContextualShadowModel,cache_context_fields
from ergt_phi.shadow_data import training_only,pair_partition,pack_records,feature_conflicts
from ergt_phi.shadow_metrics import evaluate_calibration
from ergt_phi.native_steps import SteppedNative
from ergt_phi.checkpoint import capture,restore,save
from ergt_reviewer.native_solver import ERGT43Config,hard_solutions_from_outputs
from ergt_reviewer.matched_data import RawTokenInputContract,collate_matched_topology_examples
from ergt_reviewer.data_schema import RELATION_SURFACES


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: exact(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): exact(x,y)
    else: assert a==b


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    cfg = CalibrationConfig()
    out = ROOT/'runs/m2'; out.mkdir(parents=True,exist_ok=True)
    run = legacy_input(ROOT, "runs/imported_m0/m0_single_seed_reference")
    cp = run/'opt_12011_data_16301/native_ergt_training.pt'
    accepted = json.loads((legacy_input(ROOT, "manifests/trained_m0_audit.json")).read_text())
    if digest(cp)!=accepted['checkpoint_sha256'] or accepted['status']!='passed':
        raise ValueError('accepted M0 checkpoint required')
    state = torch.load(cp,map_location='cpu',weights_only=True)
    model_cfg = dict(state['config']); model_cfg['raw_input_contract']=RawTokenInputContract(**model_cfg['raw_input_contract'])
    base = SteppedNative(ERGT43Config(**model_cfg)).eval()
    base.load_state_dict(state['best_state'],strict=True)
    baseline_state = {k:v.clone() for k,v in base.state_dict().items()}
    # Revised M2 anchor: frozen native final-field context concatenated with Psi0.
    # The context pass uses raw tokens only, never labels or gold edges.
    phase = PhaseAnchorNetwork(base.config.hidden_dim * 2,base.config.n_worlds,3,cfg)
    shadow = ContextualShadowModel(base,phase)
    optimizer = torch.optim.AdamW(phase.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    examples,tokenizer,train_hash = training_only(run)
    fit,monitor = pair_partition(examples,cfg.split_seed)
    files = ['ergt_phi/shadow_phase.py','ergt_phi/shadow_data.py','ergt_phi/shadow_metrics.py',
             'ergt_phi/contextual_shadow.py','ergt_phi/checkpoint.py','ergt_phi/native_steps.py','scripts/run_m2.py']
    contract = {'stage':'M2_shadow_contextual_anchor_revision','config':asdict(cfg),'base_checkpoint_sha256':digest(cp),
                'source_sha256':{f:digest(ROOT/f) for f in files},'train_sha256':train_hash,
                'fit_example_ids':[examples[i].example_id for i in fit],
                'monitor_example_ids':[examples[i].example_id for i in monitor],
                'split_policy':'whole_pair_stratified_split_inside_original_training_cohort',
                'device':'cpu','torch':torch.__version__,'threads':1,'phase_mode':'shadow',
                'anchor_revision':'concat(Psi0,stop_gradient(native_final_field_context))',
                'coupling':0.,'nu':0.,'inner_iterations':0,'feedback_mode':'none',
                'active_parameter_groups':['phase.network','phase.offset_raw'],
                'optimizer':'AdamW','final_horizon_examples_used':0,
                'native_tuning_validation_readiness_examples_used':0,
                'threshold_provenance':'new_research_preregistration_not_fixed_by_mathematical_spec'}
    # Write protocol and thresholds BEFORE observing calibration performance.
    (out/'protocol.json').write_text(json.dumps(contract,indent=2))
    (workspace_path(ROOT, "manifests/m2_status.json")).write_text(json.dumps({'status':'running','M2_complete':False,'M3_authorized_by_results':False}))
    print(f'Training hash verified. Fit={len(fit)}, monitor={len(monitor)}; caching frozen native context.',flush=True)
    records = cache_context_fields(shadow,examples,tokenizer)
    ambiguity = feature_conflicts(records)
    (out/'feature_conflicts.json').write_text(json.dumps(ambiguity,indent=2))
    # Construct a raw-input counterexample by changing one remote relation word.
    # Endpoints and their initial local features are unchanged; no answer label claimed.
    first = min(range(len(examples)),key=lambda i:len(examples[i].tokens))
    sample = examples[first]
    batch = collate_matched_topology_examples((sample,),tokenizer)
    edge = sample.physical_edges[0]
    changed = batch.raw_token_ids.clone()
    from ergt_reviewer.matched_data import raw_token_fingerprint_id
    changed[0,edge.event_anchor_position] = raw_token_fingerprint_id(RELATION_SURFACES[edge.relation_id%3+1])
    with torch.no_grad():
        initial = shadow.anchors(**batch.model_inputs())
        counter = shadow.anchors(changed,batch.base.attention_mask)
        # The revised context must now carry a relation-token change to the endpoints.
        endpoint_context_delta = float((shadow.context_features(**batch.model_inputs())[:,[edge.source_position,edge.target_position]]-
            shadow.context_features(changed,batch.base.attention_mask)[:,[edge.source_position,edge.target_position]]).abs().max())
        assert endpoint_context_delta > 1e-8
        original_output = base(**batch.model_inputs())
    initial_monitor = evaluate_calibration(phase,records,monitor)
    initial_fit = evaluate_calibration(phase,records,fit)
    progress = {'step':0,'epoch':0,'cursor':0,'order':[],'frozen':False,'readiness_streak':0}
    initial_state = capture(phase,optimizer,contract=contract,progress=progress)
    save(out/'initial_full_state.pt',phase,optimizer,contract=contract,progress=progress)
    curves = []; qualified=False; started=time.perf_counter()
    for epoch in range(cfg.epochs):
        order = list(fit); random.Random(cfg.shuffle_seed+epoch).shuffle(order)
        progress.update(epoch=epoch,order=order,cursor=0)
        phase.train()
        for start in range(0,len(order),cfg.batch_size):
            calibration_step(phase,optimizer,pack_records(records,order[start:start+cfg.batch_size]),contract=contract,progress=progress)
            progress['cursor']=min(start+cfg.batch_size,len(order))
        measured = evaluate_calibration(phase,records,monitor)
        fit_metrics = evaluate_calibration(phase,records,fit)
        passed,checks = readiness(measured,initial_monitor['ce'],cfg)
        progress['readiness_streak'] = progress['readiness_streak']+1 if passed else 0
        progress['epoch']=epoch+1
        row = {'epoch':epoch+1,'step':progress['step'],'fit':fit_metrics,'monitor':measured,'checks':checks,'ready':passed,'streak':progress['readiness_streak']}
        curves.append(row)
        (out/'curves.json').write_text(json.dumps(curves,indent=2))
        save(out/'latest_full_state.pt',phase,optimizer,contract=contract,progress=progress)
        print(json.dumps({'epoch':epoch+1,'fit_accuracy':fit_metrics['accuracy'],'monitor_balanced_accuracy':measured['balanced_accuracy'],'monitor_ce':measured['ce'],'ready':passed}),flush=True)
        if progress['readiness_streak'] >= cfg.required_windows:
            qualified=True; break
    # Verify trained shadow never changed the native outputs or weights.
    with torch.no_grad():
        exact(original_output,shadow(**batch.model_inputs()))
        expected = [asdict(x) for x in hard_solutions_from_outputs(original_output,base.config)]
        observed = [asdict(x) for x in hard_solutions_from_outputs(shadow(**batch.model_inputs()),base.config)]
        exact(expected,observed)
        trained_anchors = shadow.anchors(**batch.model_inputs())
        changed_anchors = shadow.anchors(changed,batch.base.attention_mask)
        endpoint_anchor_delta = float((trained_anchors[:,:,[edge.source_position,edge.target_position]]-
            changed_anchors[:,:,[edge.source_position,edge.target_position]]).abs().max())
        assert endpoint_anchor_delta > 1e-8
    exact(base.state_dict(),baseline_state)
    assert all(not p.requires_grad and p.grad is None for p in base.parameters())
    counterexample = {'remote_relation_token_changed':True,'contextual_endpoint_features_changed':True,
                      'contextual_endpoint_anchors_changed':True,'endpoint_context_delta':endpoint_context_delta,
                      'endpoint_anchor_delta':endpoint_anchor_delta,
                      'interpretation':'The frozen native context carries the remote relation change to the canonical endpoints.'}
    if qualified:
        phase.freeze(); progress['frozen']=True
        save(out/'selected_full_state.pt',phase,optimizer,contract=contract,progress=progress)
    else:
        save(out/'rejected_full_state.pt',phase,optimizer,contract=contract,progress=progress)
        progress = restore(initial_state,phase,optimizer,contract=contract)
        exact(phase.state_dict(),initial_state['model']); exact(optimizer.state_dict(),initial_state['optimizer'])
        phase.freeze(); progress['frozen']=True
        save(out/'restored_initial_full_state.pt',phase,optimizer,contract=contract,progress=progress)
    result = {'status':'qualified' if qualified else 'calibration_not_qualified','implementation_complete':True,
              'experiment_completed':True,'M2_complete':qualified,'M3_authorized_by_results':qualified,
              'initial_fit':initial_fit,'initial_monitor':initial_monitor,'final_fit':curves[-1]['fit'],
              'final_monitor':curves[-1]['monitor'],'final_checks':curves[-1]['checks'],
              'completed_epochs':len(curves),'completed_optimizer_steps':curves[-1]['step'],
              'calibration_seconds':time.perf_counter()-started,'trained_native_weights_unchanged':True,
              'native_output_and_hard_decision_parity':True,'remote_relation_counterexample':counterexample,
              'feature_conflicts':ambiguity,'offsets_frozen':True,'rollback_to_initial_on_failure':not qualified,
              'protocol_sha256':digest(out/'protocol.json'),'train_sha256':train_hash,
              'fit_examples':len(fit),'monitor_examples':len(monitor),'final_horizon_examples_used':0,
              'model_answer_improvement_claimed':False,'checkpoint_resume_contract':'full_research_state_at_step_boundaries'}
    (workspace_path(ROOT, "manifests/m2_status.json")).write_text(json.dumps(result,indent=2))
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print('M2 EXPERIMENT COMPLETED: '+result['status'],flush=True)


if __name__=='__main__': main()
