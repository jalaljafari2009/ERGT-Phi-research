import inspect
import json
from collections import Counter

from ergt_four_seed.runtime import LOCKED_RUNTIME_ROOT, activate_locked_runtime

activate_locked_runtime()
from ergt_reviewer.fair_data_v9 import build_dataset_bundle
from ergt_reviewer.suite_v9 import _v9_run_status
from ergt_reviewer.training_v9 import _epoch_grouped_sample, _ordered_pair_groups


def test_final_training_distribution_and_pair_sampler():
    config = json.loads(
        (LOCKED_RUNTIME_ROOT / "configs/paper_final_v9.json").read_text(encoding="utf-8")
    )
    data = build_dataset_bundle(config, int(config["data_seeds"][0]))
    supported = [
        example for example in data.train
        if not bool(example.base.metadata.get("unsupported_answer", False))
    ]
    by_hop = Counter(int(example.base.metadata["path_hops"]) for example in supported)
    assert by_hop == Counter({hop: 56 for hop in range(1, 9)})
    readiness_hops = {
        int(example.base.metadata["path_hops"])
        for example in data.native_readiness
        if not bool(example.base.metadata.get("unsupported_answer", False))
    }
    assert readiness_hops == set(range(2, 9))

    groups = _ordered_pair_groups(data.train)
    groups_per_batch = int(config["batch_size"]) // 2
    batches = (len(groups) + groups_per_batch - 1) // groups_per_batch
    seen = []
    for sample_index in range(batches):
        _, pair_ids = _epoch_grouped_sample(
            groups, int(config["batch_size"]), sample_index, 17117
        )
        seen.extend(pair_ids)
    first_epoch = seen[: len(groups)]
    assert len(first_epoch) == len(set(first_epoch)) == len(groups)


def test_baseline_status_is_nonblocking_and_observers_are_detached():
    signature = inspect.signature(_v9_run_status)
    assert "baseline" not in signature.parameters
    assert _v9_run_status(
        verdict_enabled=True,
        integrity_pass=True,
        native_claim_bundle_pass=True,
        observed_paired_advantage_pass=True,
    ) == "passed"
    native_source = (
        LOCKED_RUNTIME_ROOT / "ergt_reviewer/native_solver.py"
    ).read_text(encoding="utf-8")
    physics_source = (
        LOCKED_RUNTIME_ROOT / "ergt_reviewer/physics_core.py"
    ).read_text(encoding="utf-8")
    assert "spectral_runtime" not in native_source
    assert "spectral_runtime" not in physics_source
