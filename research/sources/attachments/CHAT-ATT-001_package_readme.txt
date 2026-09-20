# ERGT Four-Seed Reproduction Package

This archive is the self-contained executable companion to the manuscript. It
reproduces the locked four-seed comparison between the attention-free native
ERGT geometric reasoner and the strengthened direct Transformer baseline. It
does not require Git, the original development repository, prior checkpoints,
or prior result files.

## What is included

- the byte-locked V9 scientific runtime and the byte-locked native geometric core;
- the exact four optimization seeds and four disjoint data seeds;
- the selected strengthened direct-Transformer recipe and its qualification hashes;
- complete canonical raw-input records and cohort/tokenizer fingerprints;
- checkpoint, non-regression, intervention, and claim contracts;
- one fixed `Run all` Colab notebook;
- integrity, clean-import, data-parity, resume, and frozen-checkpoint tests;
- manuscript source and PDF.

No historical notebooks, development transcripts, previous outputs, trained
checkpoints, or precomputed scientific results are included. The JSONL file in
`data/` is a registered input record, not an answer cache.

The manuscript also cites the public archival repository snapshot used for the
paper record. This ZIP is the execution-only standalone release: it preserves
the same locked final scientific path while intentionally omitting archived
transcripts and previously generated evidence.

## Colab execution

1. Open the public repository at `https://github.com/jalaljafari2009/ERGT-paper`,
   choose `Code -> Download ZIP`, and extract the downloaded
   `ERGT-paper-main.zip`.
2. Open `notebook/ERGT_Attention_Free_Geometric_Reasoning_Study.ipynb` from that
   repository in Colab.
3. In `Runtime -> Change runtime type`, select a GPU. A historical Colab runtime
   version is not required.
4. Choose `Run all`. When prompted, upload exactly the inner file
   `ERGT-paper-main/ERGT_FourSeed_Reproduction.zip` from the extracted folder.
   Do not upload the outer `ERGT-paper-main.zip` downloaded from GitHub.
5. Do not edit a seed, threshold, profile, model recipe, or checkpoint setting.
   The notebook exposes no scientific knob; all values come from locked files.
6. Output is written under
   `MyDrive/ERGT_FourSeed_Reproduction/runs/ergt_four_seed_confirmation` and a
   downloadable bundle is also created by the runtime.

The full run trains both models independently for all four registered seed
pairs and can take substantial GPU time. `resume=True` is fixed in the notebook;
compatible interrupted runs resume from their own checkpoints. A changed code,
config, data, seed, or protocol hash is rejected.

## Scientific separation

The ERGT path starts from raw tokens and uses internal Psi/identity state,
world-conditioned geometry, product-state geodesic closure, finite-speed
constraints, typed payload transport, terminal/boundary measurements, and a
fail-closed answer rule. It uses no Transformer attention, compiler, generic
executor, provided ERGT graph, or geometry side input at inference.

The baseline is a direct Transformer. It receives the same raw tokens and
matched training label families, but no ERGT geometry, graph, compiler, or
executor. Baseline convergence is reported independently per seed. A baseline
miss is retained as a scientific result and cannot invalidate ERGT's own
convergence, stability, mechanism, or long-horizon panels.

Curvature and Laplacian-spectrum quantities are detached, fail-open observers.
They are recorded for analysis but cannot affect loss, checkpoint freezing,
interventions, or answers.

## Integrity and tests

From the extracted package root:

```bash
python -m pytest -q tests
python tests/run_acceptance.py
```

The acceptance suite checks release hashes, imports from a clean outside
directory, regenerates every registered cohort, compares interrupted and
uninterrupted optimizer state, audits the freeze contract, and runs a complete
untrained end-to-end smoke path through data, both models, interventions,
observers, reports, and bundle creation. The optional single-seed GPU acceptance is:

```bash
python tests/run_one_seed_acceptance.py --output /path/to/output
```

After the notebook finishes, validate its run directory with:

```bash
python tests/validate_completed_run.py /path/to/ergt_four_seed_confirmation
```

## Reproducibility boundary

The reference run used Google Colab runtime `2026.04`, Python 3.12.13, NumPy
2.0.2, and PyTorch 2.10.0. Exact version equality is recorded but is not an
execution gate. The notebook accepts current compatible Colab environments,
requires Python 3.10 or newer, PyTorch 2.0 or newer, and a CUDA GPU, and installs
`numpy`, `pandas`, or `torch>=2.0` only when a required package is missing or too
old. The exact observed package, CUDA, and GPU versions are saved with each run.
The package guarantees the same protocol, source/config/data hashes, contracts,
and registered tolerances; it does not claim bit-for-bit equality across GPU and
CUDA stacks.
