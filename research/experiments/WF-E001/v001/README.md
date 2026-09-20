# Validate the versioned notebook and result-evidence lifecycle

Experiment: `WF-E001/v001` · stage `workflow` · kind `infrastructure`.

Hypothesis: A deterministic CPU fixture can be packaged, executed, imported and reviewed with exact provenance and all declared artifacts.

Goal reference: research/decisions/ADR-0001.md; docs/OPERATIONAL_ROADMAP_FA.md workflow contract

- [Preregistered protocol](protocol.json)
- Notebook: `experiment.ipynb` (generate with the workflow CLI).
- Immutable release lock: `package.json` (created when packaged).
- Imported evidence: `runs/<run_id>/run.json`, `artifacts.json`, `summary.md`.
- Reviewed decisions: `runs/<run_id>/reviews/*.json`, linked to research/decisions.

An execution success is not a scientific pass. Review gates and limitations before proceeding.
