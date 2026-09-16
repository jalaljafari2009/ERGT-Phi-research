"""Persist reproducible UNTRAINED numerical evidence, separate from scientific results."""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import asdict
import torch
from ergt_phi.fixtures import example_batch, model_config, seed_all
from ergt_phi.native_steps import SteppedNative
from ergt_phi.runtime import ROOT
from ergt_phi.checkpoint import save
from ergt_phi.provenance import training_contract
from ergt_reviewer.native_solver import NativeGeometricBoundaryModel, hard_solutions_from_outputs
from ergt_reviewer.matched_data import manifest_hash


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed_all()
    batch, tokenizer = example_batch(hops=2)
    model = NativeGeometricBoundaryModel(model_config(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0025)
    initial_path = ROOT / "runs/m0/untrained_initial_full_state.pt"
    save(initial_path, model, optimizer,
        contract=training_contract(model, optimizer, data_fingerprint=manifest_hash(batch.examples)),
        progress={"global_step": 0, "data_cursor": 0, "stage": "M0", "frozen": False,
            "status": "untrained_engineering_only", "initial_seed": 83})
    stepped = SteppedNative(model.config)
    stepped.load_state_dict(model.state_dict())
    with torch.no_grad():
        outputs = model(**batch.model_inputs())
        observed = stepped(**batch.model_inputs())
        for key in outputs:
            if isinstance(outputs[key], torch.Tensor):
                torch.testing.assert_close(outputs[key], observed[key], rtol=0, atol=0)
        semantics, identity = stepped.raw_input_adapter(**batch.model_inputs())
        trace = []
        stepped.substrate(semantics, batch.base.attention_mask, identity_fibre=identity, trace=trace)
        solution, _ = hard_solutions_from_outputs(outputs, model.config)
    target = ROOT / "runs/m0/untrained_fixture.pt"
    torch.save({"status": "untrained_engineering_only", "seed": 83, "config": asdict(model.config),
        "model_state": model.state_dict(), "inputs": batch.model_inputs(), "outputs": outputs,
        "field_trace": trace, "hard_solution": asdict(solution),
        "examples": [example.to_manifest_record() for example in batch.examples]}, target)
    (ROOT / "manifests/engineering_fixture.json").write_text(json.dumps({
        "status": "untrained_engineering_only", "path": str(target.relative_to(ROOT)),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "field_snapshots": len(trace), "compared_output_fields": len(outputs),
        "forward_exact": True, "trained_accuracy_claim": False,
        "test_cpu_threads": 1, "deterministic_algorithms": True,
        "initial_full_state_path": initial_path.relative_to(ROOT).as_posix(),
        "initial_full_state_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
