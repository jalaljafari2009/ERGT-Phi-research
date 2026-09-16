"""Audit imported selected weights on CPU; never train or overwrite the reference."""
from pathlib import Path
import sys
import json
import hashlib
import copy
from dataclasses import asdict
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ergt_phi.fixtures import seed_all, training_step
from ergt_phi.native_steps import SteppedNative
from ergt_phi.model import ResearchModel, PhaseConfig
from ergt_reviewer.native_solver import NativeGeometricBoundaryModel, ERGT43Config, hard_solutions_from_outputs
from ergt_reviewer.matched_data import RawTokenInputContract, build_matched_topology_examples, collate_matched_topology_examples
from ergt_reviewer.fair_data_v9 import protocol_tokenizer, canonicalize_examples, make_unsupported

def exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=False)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x,y in zip(a,b): exact(x,y)
    else: assert a == b, (a,b)

def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed_all()
    run = ROOT / 'runs/imported_m0/m0_single_seed_reference'
    checkpoint = run / 'opt_12011_data_16301/native_ergt_training.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    field = torch.load(checkpoint.with_name('native_ergt_training_field_step_700.pt'), map_location='cpu', weights_only=True)
    training = state['training_state']
    assert training['training_complete'] and training['field_ready'] and training['mechanism_ready']
    assert training['selected_step'] == field['step'] == 700
    assert training['optimizer_steps'] == 700 and training['zero_teacher_optimizer_steps'] == 0
    exact(state['best_state'], field['model_state'])
    exact(state['model_state'], state['best_state'])
    cfg = dict(state['config'])
    cfg['raw_input_contract'] = RawTokenInputContract(**cfg['raw_input_contract'])
    cfg = ERGT43Config(**cfg)
    tokenizer = protocol_tokenizer(97)
    assert RawTokenInputContract.from_tokenizer(tokenizer) == cfg.raw_input_contract
    original = NativeGeometricBoundaryModel(cfg).eval()
    stepped = SteppedNative(cfg).eval()
    for model in (original, stepped): model.load_state_dict(state['best_state'], strict=True)
    wrappers = [ResearchModel(original, p) for p in (PhaseConfig(), PhaseConfig(phase_mode='disabled', coupling=.02), PhaseConfig(phase_mode='coupled', coupling=0))]
    rows = []
    scenarios = ['geodesic_action','finite_speed_cone','payload_transport','terminal_mass','boundary_deficit','memory_source_condition','combined_long_horizon']
    cases = [(2,s,False) for s in scenarios] + [(h,'geodesic_action',False) for h in (1,8,32)] + [(2,'geodesic_action',True)]
    for hops, scenario, unsupported in cases:
        examples = canonicalize_examples(build_matched_topology_examples(pair_count=1, seed=9041, split='m0_trained_parity', min_hops=hops, max_hops=hops, scenarios=(scenario,), node_label_pool_size=97))
        if unsupported: examples = make_unsupported(examples)
        batch = collate_matched_topology_examples(examples,tokenizer)
        with torch.no_grad():
            a = original(**batch.model_inputs())
            b = stepped(**batch.model_inputs())
            exact(a,b)
            exact([asdict(x) for x in hard_solutions_from_outputs(a,cfg)], [asdict(x) for x in hard_solutions_from_outputs(b,cfg)])
            for wrapper in wrappers: exact(a,wrapper(**batch.model_inputs()))
        rows.append(dict(hops=hops,scenario=scenario,unsupported=unsupported,examples=len(examples),pass_exact=True))
        print(json.dumps(rows[-1]),flush=True)
        if hops == 1: short_batch = batch
    with torch.no_grad():
        for intervention in ['no_phi','no_memory_geometry','no_transport','shuffled_geometry','random_geometry','only_world_0','no_action','no_boundary','no_terminal_mass','no_cone']:
            kwargs = dict(**short_batch.model_inputs(), intervention=intervention, intervention_seed=917)
            exact(original.forward_with_intervention(**kwargs),stepped.forward_with_intervention(**kwargs))
    # Disposable copies: the selected trained state is never updated.
    left,right = copy.deepcopy(original),copy.deepcopy(stepped)
    oa,ob = [torch.optim.AdamW(m.parameters(),lr=.0025) for m in (left,right)]
    rng = torch.get_rng_state().clone()
    la = training_step(left,oa,short_batch)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    lb = training_step(right,ob,short_batch)
    exact(after,torch.get_rng_state()); exact(la,lb)
    exact(left.state_dict(),right.state_dict()); exact(oa.state_dict(),ob.state_dict())
    for a,b in zip(left.parameters(),right.parameters()): exact(a.grad,b.grad)
    exact(original.state_dict(),state['best_state'])
    lock = json.loads((run/'locked_native_core_audit.json').read_text())
    for name,digest in lock['expected_sha256'].items():
        source = ROOT/'reference/ergt_four_seed/_locked_runtime'/name
        assert hashlib.sha256(source.read_bytes().replace(b'\r\n',b'\n')).hexdigest() == digest
    result = dict(status='passed', scope='single_seed_development_reference', device='cpu', torch=torch.__version__, selected_step=700, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), selected_equals_saved_model_and_field=True, canonical_source_hashes_match=True, cases=rows, intervention_cases=10, bypass_configurations=3, gradient_optimizer_rng_exact=True, original_weights_unchanged=True, rtol=0, atol=0, cross_device_bitwise_claim=False, four_seed_reproduction=False, scientific_claim_status='open')
    (ROOT/'manifests/trained_m0_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('TRAINED M0 AUDIT PASSED',flush=True)

if __name__ == '__main__': main()
