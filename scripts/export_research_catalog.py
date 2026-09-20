"""Export all registered runs for paper preparation, without promoting claims."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_tools.workflow import (  # noqa: E402
    WorkflowError, _verified_import, load_registry, revision_path, sha256_file,
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def export_legacy(root: Path, output: Path) -> None:
    """Export recorded historical values, retaining their audited scope."""
    index = read(root / "research/legacy/index.json")
    lines = ["# Historical experiment catalog", "",
             "Original reports, metrics, code provenance and notebook associations were consolidated. "
             "These records are historical evidence, not new executions or independent confirmation. "
             "Read each dossier's limitations before using numbers in a paper.", "",
             "| Dossier | Stage | Audited scope | Original report | Recorded metrics |",
             "|---|---|---|---|---|"]
    rows = []
    selected = {"balanced_accuracy", "accuracy", "tests", "failures", "errors", "skipped",
                "max_pi_normalization_error", "scientific_claim_status", "four_seed_reproduction"}
    def leaves(value, pointer=""):
        if isinstance(value, dict):
            for name, child in value.items():
                if "curves" in name or name in {"rows", "source_sha256", "artifact_sha256"}:
                    continue
                escaped = str(name).replace("~", "~0").replace("/", "~1")
                path = pointer + "/" + escaped
                if name in selected and not isinstance(child, (dict, list)):
                    yield path, child
                elif isinstance(child, (dict, list)):
                    yield from leaves(child, path)
        elif isinstance(value, list):
            for number, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    yield from leaves(child, pointer + "/" + str(number))
    for item in index["entries"]:
        folder = Path(item["root"]).relative_to("research").as_posix()
        report = Path(item["report"]["path"]).relative_to("research").as_posix()
        manifest = item.get("manifest")
        metrics = "No independent run manifest"
        if manifest:
            source = root / manifest["path"]
            if sha256_file(source) != manifest["sha256"]:
                raise ValueError(f"Historical manifest hash changed: {source}")
            values = list(leaves(read(source)))
            for pointer, value in values:
                rows.append({"legacy_id": item["id"], "stage": item["stage"], "scope": item["scope"],
                             "metric_pointer": pointer, "recorded_value": json.dumps(value),
                             "artifact": manifest["path"], "artifact_sha256": manifest["sha256"],
                             "new_execution": False, "independent_confirmation": False})
            metrics = f"{len(values)} recorded values; [manifest](../{Path(manifest['path']).relative_to('research').as_posix()})"
        if item["id"] == "LEGACY-M0":
            inventory = read(root / item["inventory"]["path"])
            original = next(row for row in inventory["entries"] if row["old_path"].endswith("/m0_single_seed_reference/final_verdict.json"))
            verdict_path = original["tracked_evidence"]
            if sha256_file(root / verdict_path) != original["sha256"]:
                raise ValueError("Historical M0 verdict changed")
            verdict = read(root / verdict_path)
            for key in ("native_32_hop_accuracy_mean", "direct_transformer_32_hop_accuracy_mean",
                        "native_minus_transformer_32_hop", "native_ergt_claim_status"):
                rows.append({"legacy_id": item["id"], "stage": item["stage"], "scope": "single_seed_development_reference_claims_open",
                             "metric_pointer": "/" + key, "recorded_value": json.dumps(verdict[key]),
                             "artifact": verdict_path, "artifact_sha256": original["sha256"],
                             "new_execution": False, "independent_confirmation": False})
            metrics += f" + 4 [Colab verdict values](../{Path(verdict_path).relative_to('research').as_posix()})"
        lines.append(f"| [{item['id']}](../{folder}/README.md) | {item['stage']} | {item['scope']} | "
                     f"[report](../{report}) | {metrics} |")
    (output / "LEGACY_CATALOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "legacy_evidence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["legacy_id", "metric_pointer", "recorded_value"])
        writer.writeheader()
        writer.writerows(rows)


def export(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    rows, lines = [], [
        "# Registered run catalog", "",
        "Generated by `python -B scripts/export_research_catalog.py`. All registered runs are included; "
        "this catalog does not select successful runs or infer scientific phase acceptance.", "",
        "Historical results and all branches are in [LEGACY_CATALOG.md](LEGACY_CATALOG.md) "
        "and [legacy_evidence.csv](legacy_evidence.csv). "
        "Use [CLAIMS.md](../CLAIMS.md) for claim scope and limitations.", "",
        "| Experiment / run | Kind | Execution | Recorded review | Current evidence check | Reports |",
        "|---|---|---|---|---|---|",
    ]
    registry = load_registry(root)
    for experiment_id, entry in sorted(registry["experiments"].items()):
        for revision in sorted(entry["revisions"]):
            directory = revision_path(root, experiment_id, revision)
            protocol = read(directory / "protocol.json")
            for run_dir in sorted((directory / "runs").glob("*")):
                if not (run_dir / "run.json").is_file():
                    continue
                manifest = read(run_dir / "run.json")
                evaluation = read(run_dir / "evaluation.json")
                reviews = sorted((run_dir / "reviews").glob("*.json"))
                review_path = reviews[-1] if reviews else None
                review = read(review_path) if review_path else {}
                try:
                    _, _, verified = _verified_import(root, directory, run_dir.name)
                    integrity = "verified" if verified == evaluation else "evaluation_changed"
                    if review:
                        decision_path = (root / review["decision_path"]).resolve()
                        if (not decision_path.is_relative_to(root) or not decision_path.is_file()
                                or sha256_file(decision_path) != review["decision_sha256"]):
                            integrity = "decision_changed_or_missing"
                except (WorkflowError, OSError, ValueError, KeyError) as exc:
                    integrity = "unavailable_or_changed: " + str(exc)
                relative = run_dir.relative_to(root / "research").as_posix()
                reports = f"[summary](../{relative}/summary.md)"
                if (run_dir / "interpretation.md").is_file():
                    reports += f" · [interpretation](../{relative}/interpretation.md)"
                if review_path:
                    reports += f" · [review](../{relative}/reviews/{review_path.name})"
                lines.append("| " + " | ".join(cell(x) for x in (
                    f"{experiment_id}/{revision}/{run_dir.name}", protocol["kind"],
                    manifest["status"], review.get("decision", "pending"), integrity,
                )) + " | " + reports + " |")
                for gate in evaluation["gates"] or [{}]:
                    rows.append({
                        "experiment_id": experiment_id, "revision": revision,
                        "run_id": run_dir.name, "stage": protocol["stage"], "kind": protocol["kind"],
                        "execution_status": manifest["status"], "returncode": manifest["returncode"],
                        "colab_detected": manifest.get("environment", {}).get("colab_detected", "unknown"),
                        "gate": gate.get("name", ""), "observed": json.dumps(gate.get("observed")),
                        "operator": gate.get("op", ""), "threshold": json.dumps(gate.get("value")),
                        "gate_status": gate.get("status", ""), "review": review.get("decision", "pending"),
                        "decision_id": review.get("decision_id", ""), "integrity": integrity,
                        "source_commit": manifest["source_commit"],
                        "package_sha256": manifest["package_sha256"],
                        "protocol_sha256": manifest["protocol_sha256"],
                        "notebook_sha256": manifest["notebook_sha256"],
                        "run_path": run_dir.relative_to(root).as_posix(),
                        "scientific_phase_promoted": False,
                    })
    output = root / "research/paper"
    output.mkdir(parents=True, exist_ok=True)
    export_legacy(root, output)
    catalog, evidence = output / "RUN_CATALOG.md", output / "evidence.csv"
    catalog.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with evidence.open("w", encoding="utf-8", newline="") as stream:
        fields = list(rows[0]) if rows else ["experiment_id", "revision", "run_id", "gate", "observed"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return catalog, evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    for path in export(parser.parse_args().root):
        print(path)
