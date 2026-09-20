"""Lifecycle tests use synthetic data; they are not scientific model evidence."""

import hashlib
import json
from pathlib import Path
import stat
import zipfile

import pytest

from research_tools.workflow import (
    WorkflowError, create_experiment, evaluate_run, import_bundle, load_registry,
    main, review_run, revise_experiment, sha256_file, validate_protocol,
    workflow_status,
)


def protocol(**changes):
    value = {
        "schema": "ergt-phi-protocol-v1", "experiment_id": "WF-E999",
        "revision": "v001", "stage": "workflow", "kind": "infrastructure",
        "title": "Synthetic workflow contract", "hypothesis": "Verified gates remain distinct from execution",
        "goal_reference": "research/WORKFLOW.md", "parent": None,
        "command": ["{python}", "scripts/synthetic.py"],
        "output_paths": ["runs/synthetic"],
        "required_artifacts": ["runs/synthetic/metrics.json"], "inputs": [],
        "gates": [{"name": "accuracy", "artifact": "runs/synthetic/metrics.json",
                   "pointer": "/accuracy", "op": "ge", "value": 0.7}],
        "created_at": "2026-09-20T00:00:00+00:00",
    }
    value.update(changes)
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def release(tmp_path, proto=None):
    directory = create_experiment(tmp_path, proto or protocol())
    (directory / "experiment.ipynb").write_text('{"cells": []}\n', encoding="utf-8")
    lock = {"schema": "ergt-phi-package-v1", "protocol_sha256": sha256_file(directory / "protocol.json"),
            "notebook_sha256": sha256_file(directory / "experiment.ipynb"),
            "package_sha256": "1" * 64, "source_commit": "a" * 40}
    write_json(directory / "package.json", lock)
    return directory, lock


def bundle(tmp_path, directory, lock, *, run_id="run-001", accuracy=0.8,
           payload=None, manifest_changes=None, extra_entries=None, name="results.zip"):
    proto = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    payload = payload if payload is not None else {
        "runs/synthetic/metrics.json": json.dumps({"accuracy": accuracy}).encode("utf-8"),
        "runner/stdout.log": b"execution finished\n",
    }
    manifest = {
        "schema": "ergt-phi-run-v1", "run_id": run_id, "experiment_id": proto["experiment_id"],
        "revision": proto["revision"],
        **{key: lock[key] for key in ("protocol_sha256", "notebook_sha256", "package_sha256", "source_commit")},
        "command": proto["command"], "started_at": "2026-09-20T00:00:00Z",
        "finished_at": "2026-09-20T00:00:01Z", "returncode": 0,
        "status": "execution_completed", "environment": {"python": "synthetic"},
        "artifacts": [{"path": path, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                      for path, data in payload.items()],
        "missing_required_artifacts": sorted(set(proto["required_artifacts"]) - set(payload)),
        "error": None,
    }
    manifest.update(manifest_changes or {})
    target = tmp_path / name
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("RUN_MANIFEST.json", json.dumps(manifest))
        for path, data in payload.items():
            archive.writestr("payload/" + path, data)
        for path, data in (extra_entries or {}).items():
            archive.writestr(path, data)
    return target


def adr(root, identifier="ADR-9999"):
    file = root / "research" / "decisions" / f"{identifier}.md"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("Synthetic review: gate met; infrastructure only; next run scientific protocol.\n", encoding="utf-8")


def test_create_revise_keeps_parent_and_never_overwrites(tmp_path):
    first = create_experiment(tmp_path, protocol())
    original = (first / "protocol.json").read_bytes()
    with pytest.raises(WorkflowError, match="already registered"):
        create_experiment(tmp_path, protocol())
    with pytest.raises(WorkflowError, match="revision_reason"):
        revise_experiment(tmp_path, "WF-E999", "v001", protocol())
    second = revise_experiment(tmp_path, "WF-E999", "v001", protocol(revision_reason="Change the control"))
    parent = json.loads((second / "protocol.json").read_text(encoding="utf-8"))["parent"]
    assert second.name == "v002"
    assert parent == {"experiment_id": "WF-E999", "revision": "v001"}
    assert (first / "protocol.json").read_bytes() == original
    assert load_registry(tmp_path)["experiments"]["WF-E999"]["revisions"]["v001"] == "experiments/WF-E999/v001"


@pytest.mark.parametrize("unsafe", ["../outside", "/absolute", "C:/escape", "a\\escape", "foo/../bar", "NUL.json"])
def test_protocol_rejects_unsafe_artifact_paths(unsafe):
    with pytest.raises(WorkflowError):
        validate_protocol(protocol(output_paths=[unsafe]))


def test_end_to_end_import_review_records_infrastructure_scope(tmp_path):
    directory, lock = release(tmp_path)
    source = bundle(tmp_path, directory, lock)
    imported = import_bundle(tmp_path, source)
    evaluation = json.loads((imported / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["eligible_for_pass"] is True
    assert evaluation["scientific_phase_pass"] is False
    assert import_bundle(tmp_path, source) == imported
    status = workflow_status(tmp_path)
    assert status["experiments"][0]["state"] == "awaiting_review"
    adr(tmp_path)
    review = review_run(tmp_path, "WF-E999", "v001", "run-001", "pass",
                        "Synthetic gate passed", "ADR-9999", "Prepare scientific protocol")
    content = json.loads(review.read_text(encoding="utf-8"))
    assert content["phase_promoted"] is False
    assert content["kind"] == "infrastructure"
    assert workflow_status(tmp_path)["experiments"][0]["state"] == "passed_infrastructure"
    assert review_run(tmp_path, "WF-E999", "v001", "run-001", "pass",
                      "Synthetic gate passed", "ADR-9999", "Prepare scientific protocol") == review
    assert len(list(review.parent.glob("*.json"))) == 1
    assert "not a scientific conclusion" in (imported / "summary.md").read_text(encoding="utf-8")


def test_success_returncode_cannot_pass_failed_gate(tmp_path):
    directory, lock = release(tmp_path)
    imported = import_bundle(tmp_path, bundle(tmp_path, directory, lock, accuracy=0.2))
    evaluation = json.loads((imported / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["execution_success"] is True
    assert evaluation["eligible_for_pass"] is False
    adr(tmp_path)
    with pytest.raises(WorkflowError, match="Cannot pass"):
        review_run(tmp_path, "WF-E999", "v001", "run-001", "pass", "bad gate", "ADR-9999", "next")
    review_run(tmp_path, "WF-E999", "v001", "run-001", "revise", "control failed", "ADR-9999", "Fix it")
    assert any("successor" in note for note in workflow_status(tmp_path)["reminders"])


def test_no_gates_never_passes_and_missing_checkpoint_blocks_acceptance(tmp_path):
    directory, lock = release(tmp_path, protocol(gates=[], required_artifacts=[
        "runs/synthetic/metrics.json", "runs/synthetic/model.pt"]))
    imported = import_bundle(tmp_path, bundle(tmp_path, directory, lock))
    evaluation = json.loads((imported / "evaluation.json").read_text(encoding="utf-8"))
    assert not evaluation["all_gates_pass"]
    assert not evaluation["eligible_for_pass"]
    assert evaluation["missing_required_artifacts"] == ["runs/synthetic/model.pt"]


def test_missing_pointer_nan_and_boolean_are_not_numeric_success(tmp_path):
    artifact = tmp_path / "metrics.json"
    proto = protocol()
    manifest = {"status": "execution_completed", "returncode": 0}
    for data in ('{"other":0.8}', '{"accuracy":NaN}', '{"accuracy":true}'):
        artifact.write_text(data, encoding="utf-8")
        result = evaluate_run(proto, manifest, {"runs/synthetic/metrics.json": artifact})
        assert not result["eligible_for_pass"]
        assert result["gates"][0]["status"] == "invalid"


@pytest.mark.parametrize("field,value", [
    ("protocol_sha256", "2" * 64), ("notebook_sha256", "2" * 64),
    ("package_sha256", "2" * 64), ("source_commit", "b" * 40),
    ("command", ["{python}", "different.py"]),
])
def test_wrong_provenance_is_rejected_before_any_import(tmp_path, field, value):
    directory, lock = release(tmp_path)
    source = bundle(tmp_path, directory, lock, manifest_changes={field: value})
    with pytest.raises(WorkflowError):
        import_bundle(tmp_path, source)
    assert not (directory / "runs").exists()


@pytest.mark.parametrize("entry", ["../escape.txt", "payload/../../escape", "payload/C:/escape", "undeclared.txt"])
def test_zip_traversal_and_unlisted_files_rejected(tmp_path, entry):
    directory, lock = release(tmp_path)
    source = bundle(tmp_path, directory, lock, extra_entries={entry: b"bad"})
    with pytest.raises(WorkflowError):
        import_bundle(tmp_path, source)
    assert not (directory / "runs").exists()


def test_zip_symlink_and_duplicate_rejected(tmp_path):
    directory, lock = release(tmp_path)
    source = bundle(tmp_path, directory, lock)
    with zipfile.ZipFile(source, "a") as archive:
        info = zipfile.ZipInfo("payload/link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "../../outside")
    with pytest.raises(WorkflowError, match="Unsafe"):
        import_bundle(tmp_path, source)
    source = bundle(tmp_path, directory, lock)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(source, "a") as archive:
            archive.writestr("RUN_MANIFEST.json", "{}")
    with pytest.raises(WorkflowError, match="Duplicate"):
        import_bundle(tmp_path, source)


def test_hash_failure_and_conflicting_run_leave_prior_evidence_intact(tmp_path):
    directory, lock = release(tmp_path)
    source = bundle(tmp_path, directory, lock)
    imported = import_bundle(tmp_path, source)
    original = (imported / "run.json").read_bytes()
    changed = bundle(tmp_path, directory, lock, accuracy=0.9, name="changed.zip")
    with pytest.raises(WorkflowError, match="different evidence"):
        import_bundle(tmp_path, changed)
    assert (imported / "run.json").read_bytes() == original
    bad = bundle(tmp_path, directory, lock, run_id="run-bad", manifest_changes={"artifacts": [
        {"path": "runs/synthetic/metrics.json", "sha256": "9" * 64, "bytes": 17},
        {"path": "runner/stdout.log", "sha256": hashlib.sha256(b"execution finished\n").hexdigest(), "bytes": 19},
    ]}, name="bad.zip")
    with pytest.raises(WorkflowError, match="SHA-256"):
        import_bundle(tmp_path, bad)
    assert not (directory / "runs" / "run-bad").exists()


def test_mutated_released_protocol_or_imported_artifact_blocks_review(tmp_path):
    directory, lock = release(tmp_path)
    source = bundle(tmp_path, directory, lock)
    imported = import_bundle(tmp_path, source)
    adr(tmp_path)
    metric = imported / "files" / "runs" / "synthetic" / "metrics.json"
    metric.write_text('{"accuracy":1.0}', encoding="utf-8")
    with pytest.raises(WorkflowError, match="changed or missing"):
        review_run(tmp_path, "WF-E999", "v001", "run-001", "pass", "x", "ADR-9999", "next")
    (directory / "protocol.json").write_text(json.dumps(protocol(title="Changed after release")), encoding="utf-8")
    with pytest.raises(WorkflowError, match="Released protocol.json changed"):
        import_bundle(tmp_path, source)


def test_review_amendment_requires_new_decision_and_preserves_history(tmp_path):
    directory, lock = release(tmp_path)
    import_bundle(tmp_path, bundle(tmp_path, directory, lock))
    adr(tmp_path)
    first = review_run(tmp_path, "WF-E999", "v001", "run-001", "inconclusive", "Need control", "ADR-9999", "Add control")
    with pytest.raises(WorkflowError, match="new linked decision"):
        review_run(tmp_path, "WF-E999", "v001", "run-001", "pass", "Control observed", "ADR-9999", "Next")
    adr(tmp_path, "ADR-10000")
    second = review_run(tmp_path, "WF-E999", "v001", "run-001", "pass", "Control observed", "ADR-10000", "Next")
    assert first.is_file() and second.name == "002.json"
    assert json.loads(second.read_text(encoding="utf-8"))["supersedes"] == "001.json"


def test_weights_stay_outside_tracked_run_and_backup_reminder_can_be_resolved(tmp_path):
    proto = protocol(required_artifacts=["runs/synthetic/metrics.json", "runs/synthetic/model.pt"])
    directory, lock = release(tmp_path, proto)
    source = bundle(tmp_path, directory, lock, payload={
        "runs/synthetic/metrics.json": b'{"accuracy":0.9}', "runs/synthetic/model.pt": b"checkpoint\x00bytes",
    })
    imported = import_bundle(tmp_path, source)
    index = json.loads((imported / "artifacts.json").read_text(encoding="utf-8"))
    weight = next(item for item in index["artifacts"] if item["path"].endswith(".pt"))
    assert not weight["tracked"]
    assert weight["local_path"].startswith("research/artifacts/")
    assert (tmp_path / weight["local_path"]).is_file()
    assert any("only local" in reminder for reminder in workflow_status(tmp_path)["reminders"])
    import_bundle(tmp_path, source, source_uri="https://drive.google.com/example-archive")
    assert not any("only local" in reminder for reminder in workflow_status(tmp_path)["reminders"])
    assert (imported / "locations" / "001.json").exists()


def test_cli_status_is_model_independent(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "status", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["cloud_execution_observed"] is False
    assert result["automatic_phase_promotion"] is False
    assert result["reminders"]


def test_status_detects_missing_artifact_and_forgotten_interpretation(tmp_path):
    directory, lock = release(tmp_path)
    imported = import_bundle(tmp_path, bundle(tmp_path, directory, lock))
    assert any("interpretation.md" in note for note in workflow_status(tmp_path)["reminders"])
    (imported / "files" / "runs" / "synthetic" / "metrics.json").unlink()
    status = workflow_status(tmp_path)
    assert status["experiments"][0]["state"] == "integrity_error"
    assert "missing" in status["experiments"][0]["error"]


def test_run_manifest_tampering_is_not_a_new_success(tmp_path):
    directory, lock = release(tmp_path)
    imported = import_bundle(tmp_path, bundle(tmp_path, directory, lock))
    manifest = json.loads((imported / "run.json").read_text(encoding="utf-8"))
    manifest["source_commit"] = "f" * 40
    write_json(imported / "run.json", manifest)
    adr(tmp_path)
    with pytest.raises(WorkflowError, match="manifest changed"):
        review_run(tmp_path, "WF-E999", "v001", "run-001", "pass", "x", "ADR-9999", "next")
    assert workflow_status(tmp_path)["experiments"][0]["state"] == "integrity_error"
