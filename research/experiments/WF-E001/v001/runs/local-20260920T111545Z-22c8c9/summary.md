# Run local-20260920T111545Z-22c8c9

Experiment: `WF-E001/v001`; stage `workflow`; kind `infrastructure`.

Hypothesis: A deterministic CPU fixture can be packaged, executed, imported and reviewed with exact provenance and all declared artifacts.

Execution: `execution_completed`, return code `0`.
Started: 2026-09-20T11:15:46.631912+00:00; finished: 2026-09-20T11:15:50.694874+00:00.

| Preregistered gate | Observed | Expected | Status |
|---|---|---|---|
| arithmetic | `true` | eq `true` | pass |
| deterministic | `true` | eq `true` | pass |
| scope_is_infrastructure | `false` | eq `false` | pass |

Missing required artifacts: [].

Eligible for explicit review as pass: `True`.

This is a deterministic evidence summary, not a scientific conclusion. Execution success does not prove the hypothesis. Record interpretation, limitations, alternatives, and the next action in a linked decision.

- [Original run manifest](run.json)
- [Artifact locations and hashes](artifacts.json)
- [Machine gate evaluation](evaluation.json)
- Reviews: `reviews/*.json`; linked architecture decisions: `research/decisions/`.
