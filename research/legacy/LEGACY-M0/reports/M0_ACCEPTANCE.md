# M0 acceptance contract

## M0-A engineering scope

The reference is copied byte-for-byte from its registered manifest. No reference
source or scientific configuration is edited. Both copies are checked after work.
Tests use identical CPU float32 weights/inputs, one thread, deterministic PyTorch.
All original-versus-refactor comparisons require rtol=atol=0. Batch/padding audits
retain the original native contract (exact discrete fields, 1e-6 continuous tolerance).

Required evidence:

1. Original and copied manifests pass; all 80 registered cohorts regenerate.
2. All 12 original tests pass, including complete untrained smoke/report/bundle.
3. Disabled/zero-coupling dispatch preserves native loss, gradients, weights,
   AdamW state and torch RNG; wrapper construction consumes no random state.
4. Seed and all three outer-step snapshots match the original implementation.
5. Full forward tensors and both hard solution objects match, including representative
   interventions, unsupported examples, padding and the registered paper architecture.
6. Proposal probe returns local evidence before product closure/answer solving.
7. Preview and next-step computation do not mutate the supplied old field state.
8. A real small ERGT model resumes in a fresh process with exact model/optimizer/
   scheduler/RNG/data-cursor state; rollback restores frozen flags and module modes.
9. A changed checkpoint contract is rejected before model mutation.
10. Final-freeze guard rejects optimizer updates before any model or RNG changes.

This is finite fixture coverage, not a proof over every possible input. Baseline
invariance on trained states remains part of M0-B and later qualification.

## M0-B scientific reference scope

Run the unchanged registered CUDA entrypoint. Retain source/config/data hashes,
environment, native checkpoint, selection/readiness and stability evidence. A
single-seed validation is a development reference; four-seed reproduction is a
separate stronger claim. Do not relabel the untrained fixture as a trained checkpoint.

The original launcher preserves original checkpoint behavior. The research complete-
state format is tested separately and does not retroactively make old snapshots
complete RNG snapshots. No GPU-to-CPU or cross-version bitwise identity is claimed.

## Not implemented at M0

No phase anchors, Q feedback, sparse phase energy, mirror solver, nonlinear response,
active phase training, coherent subspace or compute reduction. Future active mode
requests fail explicitly rather than silently returning the baseline.
