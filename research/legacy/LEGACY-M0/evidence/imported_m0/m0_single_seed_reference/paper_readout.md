# ERGT Attention-Free Geometric Reasoning Study

- Schema: `ergt-geometric-long-horizon-study-v1`
- Protocol: `single_seed_validation`
- Primary models: Direct Transformer Baseline and ERGT Geometric Model.
- Input: the same collision-free raw serialization for both models.
- Shared training: balanced 1--8 hop raw examples for both models.
- ID convergence: one-hop train subset plus disjoint one-hop validation.
- Direct Transformer recipe: locked by a bounded development-only qualification stage; final 12--32-hop panels were not used for selection.
- Native checkpoint selection: strict independent 2--8-hop readiness, minimum deterministic exposure, immediate freeze, then two disjoint mechanism shards; final interventions remain held out.
- Frozen horizon comparison: 4--32 hops; the registered endpoint is 32 hops.
- ERGT inference: compiler-free and free of a generic graph executor.
- Curvature and spectrum: observer-only; neither changes training nor answers.

## Claim Matrix

| Claim | Status | Gate | Scope |
|---|---|---|---|
| attention_free_native_geometric_answer_path | supported | `architecture_and_shared_input` | Static source and forward-contract audit. |
| native_ergt_converges_in_distribution | supported | `native_ergt_converged_id` | ERGT one-hop train subset and disjoint one-hop validation; baseline-independent. |
| qualified_direct_transformer_baseline_with_disclosed_supervision | supported | `qualified_baseline_and_disclosed_training_supervision` | A bounded development-only search selected the direct Transformer recipe; no labels, compiler, executor, or geometry enter inference. |
| strengthened_direct_transformer_convergence_by_seed | reported | `per_seed_convergence_readout` | Every seed and both one-hop ID accuracies are reported numerically; this readout does not determine run status or ERGT convergence. |
| multihop_exposure_locked_field_freeze_and_mechanism_selection | supported | `native_multihop_exposure_locked_checkpoint` | Two strict 2--8-hop readiness windows plus minimum exposure precede freeze; no optimizer update follows it. |
| cumulative_native_stability_non_regression | supported | `native_stability_capability_catalog` | Every registered S01--S14 invariant is explicitly evaluated per seed. |
| native_ergt_absolute_long_horizon_accuracy_to_32_hops | open | `native_ergt_absolute_32_hop_floor` | Absolute ERGT endpoint; independent of baseline convergence. |
| observed_paired_advantage_over_strengthened_direct_transformer | supported | `observed_paired_32_hop_advantage` | Paired effect, hierarchical interval, and exact paired test as observed. |
| conditional_matched_convergence_comparison | reported | `conditional_matched_convergence_readout` | The ID-qualified seed count and its frozen 32-hop endpoint are reported numerically. The registered three-seed criterion controls manuscript wording only and never marks the notebook run as failed. |
| causal_dependence_on_registered_geometric_mechanisms | supported | `registered_causal_interventions` | Same-checkpoint action/cone/transport/boundary/terminal/memory interventions. |
| functional_multiworld_dependence | supported | `functional_multiworld_dependence` | No claim of uniform or independent semantic specialization. |
| bounded_unsupported_answer_rejection | supported | `unsupported_answer_rejection` | Controlled unsupported candidates, not open-domain hallucination. |
| native_ergt_bounded_claim_bundle | open | `native_ergt_claim_bundle` | Native convergence, absolute 32-hop floor, mechanisms, worlds, and rejection. |
| curvature_and_spectrum_observers | supported | `observer_only_source_contract` | Descriptive only; no causal claim. |
| universal_transformer_superiority | not_claimed | `out_of_scope` | No universal comparison is asserted. |
| language_modeling_or_perplexity_dominance | not_claimed | `out_of_scope` | This package is not a language-model benchmark. |
| compute_efficiency_superiority | not_claimed | `resource_disclosure_only` | Runtime and memory are disclosed without an efficiency claim. |

## Registered Endpoint

- ERGT Geometric Model: `0.786`
- Direct Transformer Baseline: `0.482`
- ERGT minus Direct Transformer: `0.304`
- ERGT ID convergence: `True`
- Direct Transformer convergence by seed: `converged_0_of_1_seeds`
- Observed paired 32-hop comparison: `supported`
- Conditional matched-convergence readout: `conditional_endpoint_reported_below_registered_seed_count` (0/1 registered seeds)

## Gate Results

| Gate | Passed | Detail |
|---|---:|---|
| locked_native_core_unchanged | True | Immutable V21-derived native-core SHA-256 audit. |
| architecture_and_shared_input | True | Direct Transformer and native ERGT receive the same raw serialization. |
| data_protocol_integrity | True | No train/evaluation or cross-seed raw overlap. |
| locked_baseline_qualification_manifest | True | The direct Transformer recipe was selected by the bounded development-only qualification stage without evaluating final 12--32-hop panels. |
| immutable_final_execution_lock | True | The V21/V6 native path, final config, evaluator, data path, selected baseline recipe, and checkpoint policy match their registered SHA-256 lock. |
| fresh_confirmatory_seed_separation | True | Optimization and data seeds are disjoint from qualification and all V3--V6 development seeds. |
| qualified_baseline_and_disclosed_training_supervision | True | The selected direct Transformer uses the locked qualified loss mode, raw tokens only, and no compiler, executor, or ERGT geometry. |
| native_multihop_exposure_locked_checkpoint | True | Strict independent 2--8-hop readiness, complete V21 chain/closure/overflow checks, deterministic coverage, and minimum exposure precede native freeze. |
| native_stability_capability_catalog | True | All cumulative S01--S14 architecture, training, invariance, mechanism, multiworld, identity, and observer invariants are evaluated per seed. |
| direct_transformer_checkpoint_selection | False | The strengthened direct Transformer checkpoint is selected without final panels; a miss is reported for that baseline arm and is not an ERGT veto. |
| decoupled_frozen_mechanism_checkpoint_selection | True | All six mechanisms pass independently on both frozen selection shards. |
| heldout_validation_not_used_for_selection | True | Final validation and causal panels are evaluation-only. |
| parameter_count_match | True | Inference-active parameters differ by at most ten percent. |
| native_ergt_converged_id | True | ERGT clears its registered train/validation ID floor on every seed. |
| native_ergt_absolute_32_hop_floor | False | ERGT independently clears the registered absolute 32-hop accuracy floor. |
| observed_paired_32_hop_advantage | True | Observed paired difference is positive, its hierarchical interval excludes zero, and the exact paired test is significant. |
| registered_large_32_hop_margin | False | The observed mean difference clears the preregistered large-effect margin. |
| registered_causal_interventions | True | Every held-out scenario is evaluable and clears the targeted-drop floor. |
| functional_multiworld_dependence | True | No single world retains the complete program. |
| unsupported_answer_rejection | True | Native unsupported error remains below the registered ceiling. |
| native_ergt_claim_bundle | False | Native checkpoint stability, ERGT convergence, absolute 32-hop accuracy, mechanisms, multiworld dependence, and rejection pass together. |

## Failed Rows

No failed rows.

## Open Scientific Rows

| Category | Gate/scenario | Reason | Value | Required |
|---|---|---|---:|---:|
| open_scientific_gate | direct_transformer_checkpoint_selection | The strengthened direct Transformer checkpoint is selected without final panels; a miss is reported for that baseline arm and is not an ERGT veto. | False | False |
| open_scientific_gate | native_ergt_absolute_32_hop_floor | ERGT independently clears the registered absolute 32-hop accuracy floor. | False | False |
| open_scientific_gate | registered_large_32_hop_margin | The observed mean difference clears the preregistered large-effect margin. | False | False |
| open_scientific_gate | native_ergt_claim_bundle | Native checkpoint stability, ERGT convergence, absolute 32-hop accuracy, mechanisms, multiworld dependence, and rejection pass together. | False | False |

## Final Decision

- Status: `completed_with_open_claims`
- Interpretation: The four-seed study reports ERGT convergence, direct-Transformer convergence by seed, the all-seed paired comparison, and the preregistered matched-convergence subset separately. A baseline miss is retained as a result and never converts an ERGT result or completed all-seed comparison into a failed execution.

No universal Transformer superiority, language-modeling dominance, or compute-efficiency claim is made.
