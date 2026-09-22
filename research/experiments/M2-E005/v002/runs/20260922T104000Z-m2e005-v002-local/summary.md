# Run 20260922T104000Z-m2e005-v002-local

Experiment: `M2-E005/v002`; stage `M2`; kind `research`.

Hypothesis: Preregistered scenario interventions on the frozen native model will create nonzero, finite variation in answer-margin and event/path outcomes in both locked development windows. Capacity-matched readouts trained only on historical M0 fit can then measure how much answer/path sensitivity is present in raw, Q, no-phase and frozen-phase representations without using fresh targets for training or claiming Phase superiority.

Execution: `execution_completed`, return code `0`.
Started: 2026-09-22T10:39:02.360112+00:00; finished: 2026-09-22T11:27:59.270039+00:00.

| Preregistered gate | Observed | Expected | Status |
|---|---|---|---|
| challenge_complete | `true` | eq `true` | pass |
| challenge_support_validated | `false` | eq `true` | fail |
| window_1_failures | `192` | ge `4` | pass |
| window_2_failures | `192` | ge `4` | pass |
| window_1_successes | `60` | ge `4` | pass |
| window_2_successes | `60` | ge `4` | pass |
| window_1_chain_degradation | `0` | ge `4` | fail |
| window_2_chain_degradation | `0` | ge `4` | fail |
| window_1_margin_variation | `1.385415130840576` | ge `1e-06` | pass |
| window_2_margin_variation | `1.4364381095175371` | ge `1e-06` | pass |
| fresh_targets_not_used | `false` | eq `false` | pass |
| fresh_lock_match | `true` | eq `true` | pass |
| raw_text_disjoint | `true` | eq `true` | pass |
| pair_disjoint | `true` | eq `true` | pass |
| windows_pair_disjoint | `true` | eq `true` | pass |
| fit_monitor_disjoint | `true` | eq `true` | pass |
| historical_training_only | `true` | eq `true` | pass |
| capacity_matched | `true` | eq `true` | pass |
| reference_unchanged | `true` | eq `true` | pass |
| phase_unchanged | `true` | eq `true` | pass |
| input_files_unchanged | `true` | eq `true` | pass |
| m8_closed | `false` | eq `false` | pass |
| m3_closed | `false` | eq `false` | pass |
| no_phase_superiority_claim | `false` | eq `false` | pass |

Missing required artifacts: [].

Eligible for explicit review as pass: `False`.

This is a deterministic evidence summary, not a scientific conclusion. Execution success does not prove the hypothesis. Record interpretation, limitations, alternatives, and the next action in a linked decision.

- [Original run manifest](run.json)
- [Artifact locations and hashes](artifacts.json)
- [Machine gate evaluation](evaluation.json)
- Reviews: `reviews/*.json`; linked architecture decisions: `research/decisions/`.
