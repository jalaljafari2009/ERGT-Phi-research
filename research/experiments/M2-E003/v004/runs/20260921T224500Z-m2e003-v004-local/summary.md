# Run 20260921T224500Z-m2e003-v004-local

Experiment: `M2-E003/v004`; stage `M2`; kind `research`.

Hypothesis: The frozen M2-Q v2 checkpoint retains the preregistered auxiliary relation-calibration gates in two raw-text- and pair-disjoint fresh-development windows; Q-only and capacity-matched no-phase controls distinguish proposal access from phase representation without using fresh labels for training or selection.

Execution: `execution_completed`, return code `0`.
Started: 2026-09-21T22:43:11.569065+00:00; finished: 2026-09-21T22:49:57.195127+00:00.

| Preregistered gate | Observed | Expected | Status |
|---|---|---|---|
| fresh_confirmation_complete | `true` | eq `true` | pass |
| fresh_lock_match | `true` | eq `true` | pass |
| raw_text_disjoint | `true` | eq `true` | pass |
| counterfactual_pair_disjoint | `true` | eq `true` | pass |
| confirmation_windows_pair_disjoint | `true` | eq `true` | pass |
| window_1_all_phase_checks | `true` | eq `true` | pass |
| window_1_balanced_accuracy | `0.780028253661042` | ge `0.7` | pass |
| window_1_relation_0_recall | `0.8154761904761905` | ge `0.5` | pass |
| window_1_relation_1_recall | `0.7971204188481675` | ge `0.5` | pass |
| window_1_relation_2_recall | `0.7274881516587678` | ge `0.5` | pass |
| window_2_all_phase_checks | `true` | eq `true` | pass |
| window_2_balanced_accuracy | `0.7711597113342311` | ge `0.7` | pass |
| window_2_relation_0_recall | `0.8198757763975155` | ge `0.5` | pass |
| window_2_relation_1_recall | `0.7305825242718447` | ge `0.5` | pass |
| window_2_relation_2_recall | `0.7630208333333334` | ge `0.5` | pass |
| controls_executed | `true` | eq `true` | pass |
| fresh_labels_not_used | `false` | eq `false` | pass |
| selected_checkpoint_frozen | `true` | eq `true` | pass |
| selected_checkpoint_unchanged | `true` | eq `true` | pass |
| reference_checkpoint_unchanged | `true` | eq `true` | pass |
| m8_final_horizons_closed | `false` | eq `false` | pass |
| m3_remains_closed | `false` | eq `false` | pass |

Missing required artifacts: [].

Eligible for explicit review as pass: `True`.

This is a deterministic evidence summary, not a scientific conclusion. Execution success does not prove the hypothesis. Record interpretation, limitations, alternatives, and the next action in a linked decision.

- [Original run manifest](run.json)
- [Artifact locations and hashes](artifacts.json)
- [Machine gate evaluation](evaluation.json)
- Reviews: `reviews/*.json`; linked architecture decisions: `research/decisions/`.
