"""Meaningful round trips for frozen notebooks, package safety, and failed runs."""
from pathlib import Path
import json
import sys
import zipfile

import pytest

from research_tools.notebooks import build_source_package, generate_notebook
from research_tools.runner import run_protocol, sha256
from research_tools.workflow import create_experiment, import_bundle


def prepare(tmp_path, monkeypatch, *, code=None, inputs=None):
    root = tmp_path / "repository"
    root.mkdir()
    scripts = root / "scripts"
    scripts.mkdir()
    if code is None:
        code = "from pathlib import Path\nPath('runs/example').mkdir(parents=True)\nPath('runs/example/metrics.json').write_text('{\"ok\": true}')\nprint('experiment finished')\n"
    (scripts / "example.py").write_text(code, encoding="utf-8")
    protocol = {
        "schema": "ergt-phi-protocol-v1", "experiment_id": "WF-E099", "revision": "v001",
        "stage": "workflow", "kind": "infrastructure", "title": "Workflow test",
        "hypothesis": "Evidence can return to its registered revision",
        "goal_reference": "infrastructure only", "parent": None,
        "command": ["{python}", "-B", "scripts/example.py"],
        "inputs": inputs or [], "output_paths": ["runs/example"],
        "required_artifacts": ["runs/example/metrics.json"],
        "gates": [{"name": "ok", "artifact": "runs/example/metrics.json", "pointer": "/ok", "op": "eq", "value": True}],
        "created_at": "2026-09-20T00:00:00+00:00",
    }
    revision = create_experiment(root, protocol)
    monkeypatch.setattr("research_tools.notebooks.source_commit", lambda _: "a" * 40)
    return root, revision


def release_and_extract(root, revision, tmp_path):
    generate_notebook(root, revision)
    package = build_source_package(root, revision)
    extracted = tmp_path / "colab_workspace"
    with zipfile.ZipFile(package) as archive:
        archive.extractall(extracted)
    return package, extracted, extracted / revision.relative_to(root)


def read_bundle(bundle):
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("RUN_MANIFEST.json"))
        for item in manifest["artifacts"]:
            import hashlib
            data = archive.read("payload/" + item["path"])
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
            assert len(data) == item["bytes"]
        assert set(archive.namelist()) == {"RUN_MANIFEST.json"} | {
            "payload/" + item["path"] for item in manifest["artifacts"]
        }
        return manifest, archive.read("payload/runner/stdout_stderr.log").decode()


def test_notebook_explains_and_compiles_without_execution(tmp_path, monkeypatch):
    root, revision = prepare(tmp_path, monkeypatch)
    notebook_path = generate_notebook(root, revision)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["ergt_phi"]["protocol_sha256"] == sha256(revision / "protocol.json")
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            compile("".join(item["source"]), "notebook", "exec")
            assert item["execution_count"] is None and item["outputs"] == []
    text = notebook_path.read_text(encoding="utf-8")
    assert "package.json" in text and "SHA" in text and "hypothesis" in text
    assert not (root / "runs").exists()
    assert generate_notebook(root, revision) == notebook_path
    protocol = json.loads((revision / "protocol.json").read_text())
    protocol["hypothesis"] = "A changed hypothesis needs a new revision"
    (revision / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(FileExistsError):
        generate_notebook(root, revision)


def test_package_excludes_weights_secrets_and_other_experiments(tmp_path, monkeypatch):
    root, revision = prepare(tmp_path, monkeypatch)
    for name in ("scripts/.env", "scripts/private.pem", "scripts/weights.pt", "scripts/secrets/token.json", "runs/previous.json", "research/inbox/data.txt", ".venv/private.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private")
    generate_notebook(root, revision)
    package = build_source_package(root, revision)
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        assert "scripts/example.py" in names
        assert not any("private" in name or "weights.pt" in name or ".env" in name or "runs/" in name or "inbox" in name or "token.json" in name for name in names)
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        assert names == set(manifest["files"]) | {"PACKAGE_MANIFEST.json"}
    assert build_source_package(root, revision) == package
    original_hash = sha256(package)
    package.unlink()
    assert sha256(build_source_package(root, revision)) == original_hash
    package.unlink()
    (root / "scripts/example.py").write_text("print('changed')")
    with pytest.raises(ValueError, match="Cannot restore archived source"):
        build_source_package(root, revision)


def test_success_roundtrip_records_real_outputs_and_log(tmp_path, monkeypatch):
    root, revision = prepare(tmp_path, monkeypatch)
    package, extracted, remote_revision = release_and_extract(root, revision, tmp_path)
    output = tmp_path / "results/success.zip"
    result = run_protocol(extracted, remote_revision, output, run_id="run-success", package_sha256=sha256(package))
    manifest, log = read_bundle(output)
    assert manifest == result
    assert manifest["status"] == "execution_completed" and manifest["returncode"] == 0
    assert manifest["missing_required_artifacts"] == []
    assert manifest["command"][0] == "{python}" and manifest["executed_command"][0] == sys.executable
    assert manifest["package_sha256"] == sha256(package)
    assert "experiment finished" in log
    imported = import_bundle(root, output, source_uri="local-test://success")
    assert json.loads((imported / "evaluation.json").read_text())["eligible_for_pass"] is True
    with pytest.raises(FileExistsError):
        run_protocol(extracted, remote_revision, output, run_id="run-success", package_sha256=sha256(package))


def test_subprocess_failure_still_exports_partial_artifacts(tmp_path, monkeypatch):
    code = "from pathlib import Path\nPath('runs/example').mkdir(parents=True)\nPath('runs/example/partial.txt').write_text('checkpoint not reached')\nprint('failure detail', flush=True)\nraise SystemExit(7)\n"
    root, revision = prepare(tmp_path, monkeypatch, code=code)
    package, extracted, remote_revision = release_and_extract(root, revision, tmp_path)
    output = tmp_path / "failed.zip"
    run_protocol(extracted, remote_revision, output, run_id="failed", package_sha256=sha256(package))
    manifest, log = read_bundle(output)
    assert manifest["status"] == "execution_failed" and manifest["returncode"] == 7
    assert manifest["missing_required_artifacts"] == ["runs/example/metrics.json"]
    assert "failure detail" in log
    assert "runs/example/partial.txt" in {item["path"] for item in manifest["artifacts"]}
    imported = import_bundle(root, output, source_uri="local-test://failure")
    assert json.loads((imported / "evaluation.json").read_text())["eligible_for_pass"] is False


def test_source_tampering_and_stale_outputs_cannot_be_new_evidence(tmp_path, monkeypatch):
    root, revision = prepare(tmp_path, monkeypatch)
    package, extracted, remote_revision = release_and_extract(root, revision, tmp_path)
    stale = extracted / "runs/example/metrics.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"ok": true}')
    output = tmp_path / "stale.zip"
    run_protocol(extracted, remote_revision, output, run_id="stale", package_sha256=sha256(package))
    manifest, _ = read_bundle(output)
    assert manifest["status"] == "preflight_failed"
    assert "Stale output" in manifest["error"]
    assert all(item["path"].startswith("runner/") for item in manifest["artifacts"])
    (extracted / "scripts/example.py").write_text("print('changed source')")
    tampered = tmp_path / "tampered.zip"
    run_protocol(extracted, remote_revision, tampered, run_id="tampered", package_sha256=sha256(package))
    manifest, _ = read_bundle(tampered)
    assert manifest["status"] == "preflight_failed"
    assert "Source integrity failed" in manifest["error"]


def test_input_sha_is_required_before_execution(tmp_path, monkeypatch):
    root, revision = prepare(tmp_path, monkeypatch, inputs=[{"path": "runs/input/model.pt", "sha256": "0" * 64, "uri": "drive://registered"}])
    package, extracted, remote_revision = release_and_extract(root, revision, tmp_path)
    output = tmp_path / "missing.zip"
    run_protocol(extracted, remote_revision, output, run_id="missing", package_sha256=sha256(package))
    manifest, _ = read_bundle(output)
    assert manifest["status"] == "preflight_failed"
    assert "Missing input" in manifest["error"]
    assert not (extracted / "runs/example").exists()
    imported = import_bundle(root, output, source_uri="local-test://missing-input")
    assert json.loads((imported / "evaluation.json").read_text())["eligible_for_pass"] is False


def test_missing_executable_is_importable_failed_evidence(tmp_path, monkeypatch):
    root, revision = prepare(tmp_path, monkeypatch)
    protocol_path = revision / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["command"] = ["ergt_nonexistent_executable_98d5"]
    protocol_path.write_text(json.dumps(protocol))
    package, extracted, remote_revision = release_and_extract(root, revision, tmp_path)
    output = tmp_path / "missing-executable.zip"
    result = run_protocol(extracted, remote_revision, output, run_id="no-executable", package_sha256=sha256(package))
    assert result["status"] == "execution_failed" and result["returncode"] is None
    imported = import_bundle(root, output, source_uri="local-test://no-executable")
    assert json.loads((imported / "evaluation.json").read_text())["eligible_for_pass"] is False


def test_timeout_exports_diagnostic_bundle(tmp_path, monkeypatch):
    root, revision = prepare(tmp_path, monkeypatch, code="import time\nprint('started', flush=True)\ntime.sleep(30)\n")
    protocol_path = revision / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["timeout_seconds"] = 0.3
    protocol_path.write_text(json.dumps(protocol))
    package, extracted, remote_revision = release_and_extract(root, revision, tmp_path)
    output = tmp_path / "timeout.zip"
    result = run_protocol(extracted, remote_revision, output, run_id="timeout", package_sha256=sha256(package))
    assert result["status"] == "execution_failed" and "TimeoutError" in result["error"]
    imported = import_bundle(root, output, source_uri="local-test://timeout")
    assert json.loads((imported / "evaluation.json").read_text())["eligible_for_pass"] is False
