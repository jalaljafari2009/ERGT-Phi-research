# Run 20260921T221000Z-m2e003-local

Experiment: `M2-E003/v001`; stage `M2`; kind `research`.

Hypothesis: The frozen M2-Q v2 checkpoint retains the preregistered auxiliary relation-calibration gates in two raw-text- and pair-disjoint fresh-development windows; Q-only and capacity-matched no-phase controls distinguish proposal access from phase representation without using fresh labels for training or selection.

Execution: `execution_failed`, return code `1`.
Started: 2026-09-21T22:06:08.521733+00:00; finished: 2026-09-21T22:06:26.693955+00:00.

| Preregistered gate | Observed | Expected | Status |
|---|---|---|---|
| fresh_confirmation_complete | `null` | eq `true` | missing |
| fresh_lock_match | `null` | eq `true` | missing |
| raw_text_disjoint | `null` | eq `true` | missing |
| counterfactual_pair_disjoint | `null` | eq `true` | missing |
| confirmation_windows_pair_disjoint | `null` | eq `true` | missing |
| window_1_all_phase_checks | `null` | eq `true` | missing |
| window_1_balanced_accuracy | `null` | ge `0.7` | missing |
| window_1_relation_0_recall | `null` | ge `0.5` | missing |
| window_1_relation_1_recall | `null` | ge `0.5` | missing |
| window_1_relation_2_recall | `null` | ge `0.5` | missing |
| window_2_all_phase_checks | `null` | eq `true` | missing |
| window_2_balanced_accuracy | `null` | ge `0.7` | missing |
| window_2_relation_0_recall | `null` | ge `0.5` | missing |
| window_2_relation_1_recall | `null` | ge `0.5` | missing |
| window_2_relation_2_recall | `null` | ge `0.5` | missing |
| controls_executed | `null` | eq `true` | missing |
| fresh_labels_not_used | `null` | eq `false` | missing |
| selected_checkpoint_frozen | `null` | eq `true` | missing |
| selected_checkpoint_unchanged | `null` | eq `true` | missing |
| reference_checkpoint_unchanged | `null` | eq `true` | missing |
| m8_final_horizons_closed | `null` | eq `false` | missing |
| m3_remains_closed | `null` | eq `false` | missing |

Missing required artifacts: ['runs/m2_fresh_confirmation/control_checkpoints.pt', 'runs/m2_fresh_confirmation/event_predictions.jsonl', 'runs/m2_fresh_confirmation/fresh_development_audit.json', 'runs/m2_fresh_confirmation/fresh_feature_cache_manifest.json', 'runs/m2_fresh_confirmation/m2_fresh_handoff.json', 'runs/m2_fresh_confirmation/result.json'].

Eligible for explicit review as pass: `False`.

This is a deterministic evidence summary, not a scientific conclusion. Execution success does not prove the hypothesis. Record interpretation, limitations, alternatives, and the next action in a linked decision.

- [Original run manifest](run.json)
- [Artifact locations and hashes](artifacts.json)
- [Machine gate evaluation](evaluation.json)
- Reviews: `reviews/*.json`; linked architecture decisions: `research/decisions/`.
