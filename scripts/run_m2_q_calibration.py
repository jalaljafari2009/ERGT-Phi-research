"""Run a preregistered Q-conditioned M2 shadow calibration.

This is an explicit M2 protocol revision: detached, label-free native Q context
is appended to Psi0 before the shadow phase anchor network.  It never enters
native geometry or answer execution.  Gold endpoints are used only after raw
features have been cached, for the existing phase relation loss.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from ergt_phi.checkpoint import capture, restore
from ergt_phi.lagged_q import native_proposal_q
from ergt_phi.native_steps import SteppedNative
from ergt_phi.shadow_data import pair_partition, pack_records, training_only
from ergt_phi.shadow_metrics import evaluate_calibration
from ergt_phi.shadow_phase import (
    CalibrationConfig,
    PhaseAnchorNetwork,
    calibration_step,
    readiness,
)
from ergt_reviewer.matched_data import RawTokenInputContract, collate_matched_topology_examples
from ergt_reviewer.native_solver import ERGT43Config


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def cache_q_records(baseline, examples, tokenizer):
    """Cache Psi0 plus node-wise source/target Q marginals from raw input only."""
    records = []
    for index, example in enumerate(examples):
        batch = collate_matched_topology_examples((example,), tokenizer)
        semantic, identity = baseline.raw_input_adapter(**batch.model_inputs())
        seed = baseline.substrate.seed_native_state(
            semantic, batch.base.attention_mask, identity_fibre=identity,
        )
        state = baseline.substrate(
            semantic, batch.base.attention_mask, identity_fibre=identity,
        )
        proposal = baseline.probe_native_proposals(state, batch.base.attention_mask)
        mask = batch.base.attention_mask[0]
        n = mask.numel()
        valid = mask[:, None] & mask[None, :]
        valid &= ~torch.eye(n, dtype=torch.bool)
        edge_index = torch.nonzero(valid, as_tuple=False).t().contiguous()
        q, _, _ = native_proposal_q(
            proposal, edge_index, attention_mask=batch.base.attention_mask,
        )
        relations = q.shape[1]
        source = q.new_zeros((n, relations)).index_add(0, edge_index[0], q)
        target = q.new_zeros((n, relations)).index_add(0, edge_index[1], q)
        # log1p keeps high-mass native proposals finite while preserving zeros.
        q_context = torch.log1p(torch.cat((source, target), dim=-1))
        features = torch.cat((seed['psi'][0], q_context), dim=-1).cpu()
        events = torch.tensor(
            [[e.source_position, e.target_position] for e in example.base.edges],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [e.relation_id - 1 for e in example.base.edges], dtype=torch.long,
        )
        records.append({
            'psi': features,
            'events': events,
            'labels': labels,
            'example_id': example.example_id,
            'pair_id': example.pair_id,
        })
        if (index + 1) % 50 == 0:
            print(f'Cached Q-conditioned features: {index + 1}/{len(examples)}', flush=True)
    return records


def batch_tensors(records, indices, *, shuffle=False, seed=0):
    psi, mask, events, labels = pack_records(records, indices)
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        labels = labels[torch.randperm(labels.numel(), generator=generator)]
    return psi, mask, events, labels


def train_shadow(records, fit, monitor, cfg, *, worlds, seed, shuffle=False):
    input_dim = records[0]['psi'].shape[1]
    phase = PhaseAnchorNetwork(input_dim, worlds, 3, cfg)
    optimizer = torch.optim.AdamW(
        phase.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    contract = {'stage': 'M2_Q_conditioned_shadow_calibration', 'shuffle': shuffle}
    progress = {'step': 0, 'epoch': 0, 'cursor': 0, 'order': [], 'frozen': False}
    initial = evaluate_calibration(phase, records, monitor)
    curves = []
    qualified = False
    streak = 0
    for epoch in range(cfg.epochs):
        order = list(fit)
        random.Random(seed + epoch).shuffle(order)
        progress.update(epoch=epoch, order=order, cursor=0)
        phase.train()
        for start in range(0, len(order), cfg.batch_size):
            batch_indices = order[start:start + cfg.batch_size]
            tensors = batch_tensors(
                records, batch_indices, shuffle=shuffle,
                seed=seed + 100000 + epoch * 1000 + start,
            )
            calibration_step(
                phase, optimizer, tensors, contract=contract, progress=progress,
            )
            progress['cursor'] = min(start + cfg.batch_size, len(order))
        measured = evaluate_calibration(phase, records, monitor)
        fit_metrics = evaluate_calibration(phase, records, fit)
        passed, checks = readiness(measured, initial['ce'], cfg)
        streak = streak + 1 if passed else 0
        row = {
            'epoch': epoch + 1,
            'step': progress['step'],
            'fit': fit_metrics,
            'monitor': measured,
            'checks': checks,
            'ready': passed,
            'streak': streak,
        }
        curves.append(row)
        print(json.dumps({
            'shuffle': shuffle,
            'epoch': epoch + 1,
            'fit_balanced_accuracy': fit_metrics['balanced_accuracy'],
            'monitor_balanced_accuracy': measured['balanced_accuracy'],
            'ready': passed,
        }), flush=True)
        if streak >= cfg.required_windows:
            qualified = True
    return phase, optimizer, initial, curves, qualified


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    run = ROOT / 'runs/imported_m0/m0_single_seed_reference'
    checkpoint = run / 'opt_12011_data_16301/native_ergt_training.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model_config = dict(state['config'])
    model_config['raw_input_contract'] = RawTokenInputContract(**model_config['raw_input_contract'])
    baseline = SteppedNative(ERGT43Config(**model_config)).eval()
    baseline.load_state_dict(state['best_state'], strict=True)
    baseline.requires_grad_(False)
    baseline_state = {key: value.detach().clone() for key, value in baseline.state_dict().items()}
    examples, tokenizer, train_hash = training_only(run)
    fit, monitor = pair_partition(examples, 20092026)
    cfg = CalibrationConfig(
        init_seed=22092026,
        shuffle_seed=23092026,
        split_seed=20092026,
        epochs=20,
    )
    out = ROOT / 'runs/m2_q_calibration'
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        'stage': 'M2_Q_conditioned_shadow_calibration',
        'revision': 'append_detached_native_Q_node_marginals_to_Psi0',
        'config': asdict(cfg),
        'fit_examples': len(fit),
        'monitor_examples': len(monitor),
        'train_sha256': train_hash,
        'checkpoint_sha256': digest(checkpoint),
        'gold_edges_used_only_after_feature_cache': True,
        'native_geometry_activated': False,
        'native_answer_head_used': False,
        'source_sha256': {
            'ergt_phi/lagged_q.py': digest(ROOT / 'ergt_phi/lagged_q.py'),
            'ergt_phi/shadow_phase.py': digest(ROOT / 'ergt_phi/shadow_phase.py'),
            'ergt_phi/shadow_metrics.py': digest(ROOT / 'ergt_phi/shadow_metrics.py'),
            'scripts/run_m2_q_calibration.py': digest(ROOT / 'scripts/run_m2_q_calibration.py'),
        },
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(f'Caching Q-conditioned features. Fit={len(fit)}, monitor={len(monitor)}', flush=True)
    records = cache_q_records(baseline, examples, tokenizer)
    real_phase, real_optimizer, initial, real_curves, real_qualified = train_shadow(
        records, fit, monitor, cfg, worlds=baseline.config.n_worlds,
        seed=cfg.shuffle_seed, shuffle=False,
    )
    shuffled_phase, shuffled_optimizer, shuffled_initial, shuffled_curves, _ = train_shadow(
        records, fit, monitor, cfg, worlds=baseline.config.n_worlds,
        seed=cfg.shuffle_seed + 1, shuffle=True,
    )
    # Native parity is checked after both shadow optimizers have run.
    first_batch = collate_matched_topology_examples((examples[0],), tokenizer)
    with torch.no_grad():
        before = baseline(**first_batch.model_inputs())
        after = baseline(**first_batch.model_inputs())
    # The two calls above are intentionally adjacent and deterministic; exact
    # parity catches accidental baseline mutation during feature extraction.
    def exact(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                exact(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                exact(x, y)
        else:
            assert a == b
    exact(before, after)
    exact(baseline.state_dict(), baseline_state)
    assert all(not p.requires_grad and p.grad is None for p in baseline.parameters())
    final = real_curves[-1]
    result = {
        'status': 'q_calibration_qualified' if real_qualified else 'q_calibration_not_qualified',
        'M2_complete': bool(real_qualified),
        'M3_authorized_by_results': False,
        'protocol_sha256': digest(out / 'protocol.json'),
        'initial_monitor': initial,
        'final_fit': final['fit'],
        'final_monitor': final['monitor'],
        'final_checks': final['checks'],
        'real_curves': real_curves,
        'shuffled_initial_monitor': shuffled_initial,
        'shuffled_final_monitor': shuffled_curves[-1]['monitor'],
        'shuffled_curves': shuffled_curves,
        'trained_native_weights_unchanged': True,
        'native_output_parity': True,
        'native_geometry_activated': False,
        'native_answer_head_used': False,
        'calibration_seconds': time.perf_counter() - started,
        'fit_examples': len(fit),
        'monitor_examples': len(monitor),
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    (ROOT / 'manifests/m2_q_calibration.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({
        'status': result['status'],
        'monitor_balanced_accuracy': result['final_monitor']['balanced_accuracy'],
        'shuffled_monitor_balanced_accuracy': result['shuffled_final_monitor']['balanced_accuracy'],
        'M2_complete': result['M2_complete'],
        'M3_authorized_by_results': result['M3_authorized_by_results'],
    }, indent=2))


if __name__ == '__main__':
    main()
