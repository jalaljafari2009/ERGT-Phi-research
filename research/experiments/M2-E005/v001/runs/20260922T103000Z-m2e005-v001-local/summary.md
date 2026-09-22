# Run 20260922T103000Z-m2e005-v001-local

Experiment: `M2-E005/v001`; stage `M2`; kind `research`.

Hypothesis: Preregistered scenario interventions on the frozen native model will create nonzero, finite variation in answer-margin and event/path outcomes in both locked development windows. Capacity-matched readouts trained only on historical M0 fit can then measure how much answer/path sensitivity is present in raw, Q, no-phase and frozen-phase representations without using fresh targets for training or claiming Phase superiority.

Execution: `execution_failed`, return code `1`.
Started: 2026-09-22T10:30:24.481491+00:00; finished: 2026-09-22T10:30:49.282265+00:00.

| Preregistered gate | Observed | Expected | Status |
|---|---|---|---|
| challenge_complete | `null` | eq `true` | missing |
| challenge_support_validated | `null` | eq `true` | missing |
| window_1_failures | `null` | ge `4` | missing |
| window_2_failures | `null` | ge `4` | missing |
| window_1_successes | `null` | ge `4` | missing |
| window_2_successes | `null` | ge `4` | missing |
| window_1_chain_degradation | `null` | ge `4` | missing |
| window_2_chain_degradation | `null` | ge `4` | missing |
| window_1_margin_variation | `null` | ge `1e-06` | missing |
| window_2_margin_variation | `null` | ge `1e-06` | missing |
| fresh_targets_not_used | `null` | eq `false` | missing |
| fresh_lock_match | `null` | eq `true` | missing |
| raw_text_disjoint | `null` | eq `true` | missing |
| pair_disjoint | `null` | eq `true` | missing |
| windows_pair_disjoint | `null` | eq `true` | missing |
| fit_monitor_disjoint | `null` | eq `true` | missing |
| historical_training_only | `null` | eq `true` | missing |
| capacity_matched | `null` | eq `true` | missing |
| reference_unchanged | `null` | eq `true` | missing |
| phase_unchanged | `null` | eq `true` | missing |
| input_files_unchanged | `null` | eq `true` | missing |
| m8_closed | `null` | eq `false` | missing |
| m3_closed | `null` | eq `false` | missing |
| no_phase_superiority_claim | `null` | eq `false` | missing |

Missing required artifacts: ['runs/m2_answer_path_challenge/fresh_challenge_rows.jsonl', 'runs/m2_answer_path_challenge/historical_fit_challenge_rows.jsonl', 'runs/m2_answer_path_challenge/input_integrity.json', 'runs/m2_answer_path_challenge/m2_answer_path_handoff.json', 'runs/m2_answer_path_challenge/readout_checkpoints.pt', 'runs/m2_answer_path_challenge/readout_predictions.jsonl', 'runs/m2_answer_path_challenge/result.json'].

Eligible for explicit review as pass: `False`.

This is a deterministic evidence summary, not a scientific conclusion. Execution success does not prove the hypothesis. Record interpretation, limitations, alternatives, and the next action in a linked decision.

- [Original run manifest](run.json)
- [Artifact locations and hashes](artifacts.json)
- [Machine gate evaluation](evaluation.json)
- Reviews: `reviews/*.json`; linked architecture decisions: `research/decisions/`.
