"""Real ERGT optimizer steps in separate processes to audit exact resume."""
import argparse
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from ergt_phi.fixtures import example_batch, model_config, seed_all, training_step
from ergt_phi.native_steps import SteppedNative
from ergt_phi.checkpoint import load, save, capture
from ergt_phi.provenance import training_contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["continuous", "first", "resume"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed_all(183)
    batch, tokenizer = example_batch()
    model = SteppedNative(model_config(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0025)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    from ergt_reviewer.matched_data import manifest_hash
    dataset = tuple(example for selection in range(3) for example in example_batch(hops=2 if selection == 1 else 1, unsupported=selection == 2)[0].examples)
    contract = training_contract(model, optimizer, data_fingerprint=manifest_hash(dataset), scheduler=scheduler)
    progress = {"global_step": 0, "data_cursor": 0, "selected_examples": [], "stage": "M0", "frozen": False}
    path = args.directory / "interrupted.pt"
    if args.mode == "resume":
        seed_all(997)  # must be overwritten by the checkpoint, not reseeded to the original
        progress = load(path, model, optimizer, contract=contract, scheduler=scheduler)
    target = 1 if args.mode == "first" else 3
    while progress["global_step"] < target:
        # Exercise every RNG and a data cursor in the actual optimizer sequence.
        draws = (random.random(), float(np.random.rand()), float(torch.rand(())))
        selection = int(sum(draws) * 10) % 3
        batch, _ = example_batch(hops=2 if selection == 1 else 1, unsupported=selection == 2)
        progress["selected_examples"].append({"draws": draws, "selection": selection})
        progress["data_cursor"] += len(batch.examples)
        training_step(model, optimizer, batch, scheduler)
        progress["global_step"] += 1
    if args.mode == "first":
        save(path, model, optimizer, contract=contract, progress=progress, scheduler=scheduler)
    else:
        result = capture(model, optimizer, contract=contract, progress=progress, scheduler=scheduler)
        result["next_rng_draws"] = (random.random(), float(np.random.rand()), torch.rand(3))
        torch.save(result, args.directory / f"{args.mode}.pt")


if __name__ == "__main__":
    main()
