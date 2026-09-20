"""Audit the native-proposal Q path without activating native geometry.

This is a shadow-only causal-scheduling check.  It builds sparse Q from a
completed frozen native snapshot, selects candidates from preview evidence, runs
the independent phase solver for diagnostics, and proves that the baseline
answer and parameters remain bitwise unchanged.
"""
from dataclasses import asdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ergt_phi.research_paths import legacy_input, workspace_path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from ergt_phi.lagged_q import native_proposal_q, select_native_patch
from ergt_phi.phase_core import KernelConfig, prepare_patch, solve_phase_mirror
from ergt_phi.shadow_data import training_only
from ergt_phi.shadow_phase import CalibrationConfig, PhaseAnchorNetwork
from ergt_phi.native_steps import SteppedNative
from ergt_reviewer.native_solver import ERGT43Config
from ergt_reviewer.matched_data import RawTokenInputContract, collate_matched_topology_examples


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            exact(a, b)
    else:
        assert left == right


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run = legacy_input(ROOT, "runs/imported_m0/m0_single_seed_reference")
    checkpoint = run / 'opt_12011_data_16301/native_ergt_training.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = dict(state['config'])
    config['raw_input_contract'] = RawTokenInputContract(**config['raw_input_contract'])
    baseline = SteppedNative(ERGT43Config(**config)).eval()
    baseline.load_state_dict(state['best_state'], strict=True)
    baseline.requires_grad_(False)
    before_weights = {key: value.detach().clone() for key, value in baseline.state_dict().items()}

    phase = PhaseAnchorNetwork(
        baseline.config.hidden_dim, baseline.config.n_worlds, 3,
        CalibrationConfig(init_seed=21092026),
    ).eval()
    kernel = KernelConfig(coupling=.02, nu=0.0, steps=24)
    examples, tokenizer, train_hash = training_only(run)
    # A bounded audit cohort catches scheduling and mask errors without tuning
    # any threshold on a held-out answer horizon.
    audit_examples = examples[:24]
    rows = []

    with torch.no_grad():
        first_batch = collate_matched_topology_examples((audit_examples[0],), tokenizer)
        baseline_before = baseline(**first_batch.model_inputs())
        for index, example in enumerate(audit_examples):
            batch = collate_matched_topology_examples((example,), tokenizer)
            semantic, identity = baseline.raw_input_adapter(**batch.model_inputs())
            seed = baseline.substrate.seed_native_state(
                semantic, batch.base.attention_mask, identity_fibre=identity,
            )
            state_after_native = baseline.substrate(
                semantic, batch.base.attention_mask, identity_fibre=identity,
            )
            proposal = baseline.probe_native_proposals(
                state_after_native, batch.base.attention_mask,
            )
            mask = batch.base.attention_mask[0]
            edge_index, base = select_native_patch(
                proposal['edge_evidence'][0], mask, top_k=32,
            )
            q, q_mass, pi = native_proposal_q(
                proposal, edge_index, attention_mask=batch.base.attention_mask,
            )
            supported = q_mass > kernel.epsilon_q
            pi_error = float((pi[supported].sum(-1) - 1).abs().max()) if bool(supported.any()) else 0.0
            anchors = phase(seed['psi'], batch.base.attention_mask)[0]
            world_rows = []
            for world in range(anchors.shape[0]):
                context, kept = prepare_patch(
                    anchors[world], base, edge_index, q,
                    phase.offsets[world], kernel,
                )
                solved = solve_phase_mirror(context, kernel)
                world_rows.append({
                    'world': world,
                    'candidate_edges': int(edge_index.shape[1]),
                    'kept_edges': int(kept.numel()),
                    'phase_iterations': int(solved.iterations),
                    'residual': float(solved.residual),
                })
            rows.append({
                'example_id': example.example_id,
                'candidate_edges': int(edge_index.shape[1]),
                'supported_edges': int(supported.sum()),
                'q_mass_max': float(q_mass.max()) if q_mass.numel() else 0.0,
                'pi_normalization_error': pi_error,
                'q_detached': not q.requires_grad and not pi.requires_grad,
                'worlds': world_rows,
            })

        # The shadow Q/phase path must not mutate the frozen native execution.
        baseline_after = baseline(**first_batch.model_inputs())
        exact(baseline_before, baseline_after)
    exact(baseline.state_dict(), before_weights)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in baseline.parameters())

    result = {
        'status': 'completed_q_shadow_zero_effect',
        'M2_complete': False,
        'M3_authorized_by_results': False,
        'native_geometry_activated': False,
        'answer_head_called_by_shadow_branch': False,
        'zero_effect_native_output_parity': True,
        'zero_effect_native_weights_parity': True,
        'audit_examples': len(rows),
        'rows': rows,
        'all_q_detached': all(row['q_detached'] for row in rows),
        'max_pi_normalization_error': max((row['pi_normalization_error'] for row in rows), default=0.0),
        'train_sha256': train_hash,
        'checkpoint_sha256': digest(checkpoint),
        'source_sha256': {
            'ergt_phi/lagged_q.py': digest(ROOT / 'ergt_phi/lagged_q.py'),
            'ergt_phi/phase_core.py': digest(ROOT / 'ergt_phi/phase_core.py'),
            'scripts/run_m2_q_shadow.py': digest(ROOT / 'scripts/run_m2_q_shadow.py'),
        },
        'interpretation': 'Q is consumed in a detached shadow phase solve; native geometry and answer execution remain unchanged.',
    }
    out = ROOT / 'runs/m2_q_shadow'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    (workspace_path(ROOT, "manifests/m2_q_shadow.json")).write_text(json.dumps(result, indent=2))
    print(json.dumps({
        'status': result['status'],
        'audit_examples': result['audit_examples'],
        'max_pi_normalization_error': result['max_pi_normalization_error'],
        'zero_effect_native_output_parity': result['zero_effect_native_output_parity'],
        'zero_effect_native_weights_parity': result['zero_effect_native_weights_parity'],
        'M3_authorized_by_results': result['M3_authorized_by_results'],
    }, indent=2))


if __name__ == '__main__':
    main()
