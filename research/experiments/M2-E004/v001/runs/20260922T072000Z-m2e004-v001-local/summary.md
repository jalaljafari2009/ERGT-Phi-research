# Run 20260922T072000Z-m2e004-v001-local

Experiment: `M2-E004/v001`; stage `M2`; kind `research`.

Hypothesis: If the auxiliary relation target is too direct, a frozen Q-only readout will lose substantial balanced accuracy when the relation channels are zeroed or cyclically permuted, while raw-only access and literal input-token exposure will explain why perfect relation decoding does not establish answer/path improvement. A frozen native answer/path panel is the more relevant target for the next M2 design.

Execution: `execution_completed`, return code `0`.
Started: 2026-09-22T07:10:02.535100+00:00; finished: 2026-09-22T07:30:12.190078+00:00.

| Preregistered gate | Observed | Expected | Status |
|---|---|---|---|
| audit_complete | `true` | eq `true` | pass |
| q_zero_executed | `true` | eq `true` | pass |
| q_permutation_executed | `true` | eq `true` | pass |
| raw_only_executed | `true` | eq `true` | pass |
| answer_path_panel_executed | `true` | eq `true` | pass |
| fresh_labels_not_used | `false` | eq `false` | pass |
| fresh_lock_match | `true` | eq `true` | pass |
| raw_text_disjoint | `true` | eq `true` | pass |
| pair_disjoint | `true` | eq `true` | pass |
| windows_pair_disjoint | `true` | eq `true` | pass |
| reference_unchanged | `true` | eq `true` | pass |
| phase_unchanged | `true` | eq `true` | pass |
| q_control_unchanged | `true` | eq `true` | pass |
| no_phase_control_unchanged | `true` | eq `true` | pass |
| input_files_unchanged | `true` | eq `true` | pass |
| parent_window_1_reproduced | `true` | eq `true` | pass |
| parent_window_2_reproduced | `true` | eq `true` | pass |
| m8_closed | `false` | eq `false` | pass |
| m3_closed | `false` | eq `false` | pass |
| no_phase_superiority_claim | `false` | eq `false` | pass |

Missing required artifacts: [].

Eligible for explicit review as pass: `True`.

This is a deterministic evidence summary, not a scientific conclusion. Execution success does not prove the hypothesis. Record interpretation, limitations, alternatives, and the next action in a linked decision.

- [Original run manifest](run.json)
- [Artifact locations and hashes](artifacts.json)
- [Machine gate evaluation](evaluation.json)
- Reviews: `reviews/*.json`; linked architecture decisions: `research/decisions/`.
