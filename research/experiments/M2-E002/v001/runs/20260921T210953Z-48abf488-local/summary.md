# Run 20260921T210953Z-48abf488-local

Experiment: `M2-E002/v001`; stage `M2`; kind `research`.

Hypothesis: Under the frozen Q-conditioned anchor contract, the first two-window qualifying state can be saved immediately, reloaded with identical monitor metrics, and resumed exactly in a fresh process while the native reference remains unchanged.

Execution: `execution_completed`, return code `0`.
Started: 2026-09-21T21:11:42.402717+00:00; finished: 2026-09-21T21:17:10.869802+00:00.

| Preregistered gate | Observed | Expected | Status |
|---|---|---|---|
| handoff_complete | `true` | eq `true` | pass |
| first_qualified_state_selected | `true` | eq `true` | pass |
| balanced_accuracy | `0.7555267392741171` | ge `0.7` | pass |
| relation_0_recall | `0.8202247191011236` | ge `0.5` | pass |
| relation_1_recall | `0.7228260869565217` | ge `0.5` | pass |
| relation_2_recall | `0.7235294117647059` | ge `0.5` | pass |
| ce_reduction | `true` | eq `true` | pass |
| offset_separation | `true` | eq `true` | pass |
| phase_not_constant | `true` | eq `true` | pass |
| two_consecutive_windows | `2` | ge `2` | pass |
| selected_checkpoint_frozen | `true` | eq `true` | pass |
| fresh_process_reload | `true` | eq `true` | pass |
| reload_metric_exact | `true` | eq `true` | pass |
| fresh_process_resume | `true` | eq `true` | pass |
| native_output_exact | `true` | eq `true` | pass |
| native_decision_exact | `true` | eq `true` | pass |
| reference_unchanged | `true` | eq `true` | pass |
| shuffled_control_executed | `true` | eq `true` | pass |
| m3_remains_closed | `false` | eq `false` | pass |

Missing required artifacts: [].

Eligible for explicit review as pass: `True`.

This is a deterministic evidence summary, not a scientific conclusion. Execution success does not prove the hypothesis. Record interpretation, limitations, alternatives, and the next action in a linked decision.

- [Original run manifest](run.json)
- [Artifact locations and hashes](artifacts.json)
- [Machine gate evaluation](evaluation.json)
- Reviews: `reviews/*.json`; linked architecture decisions: `research/decisions/`.
