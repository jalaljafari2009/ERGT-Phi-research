"""Migration must preserve checkpoint identity and isolate fresh outputs."""
import json

import pytest

from ergt_phi.research_paths import current_input, legacy_input, workspace_path


def mapping(root, *, files=None, directories=None):
    target = root / "research/legacy/path_map.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"schema": "ergt-phi-legacy-path-map-v1",
                                  "files": files or {}, "directories": directories or {}}), encoding="utf-8")


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_archived_checkpoint_wins_over_old_scratch_and_directory_prefix(tmp_path):
    archived = write(tmp_path, "research/artifacts/legacy/M0/run/model.pt", "accepted checkpoint")
    write(tmp_path, "runs/imported_m0/run/model.pt", "unreviewed replacement")
    mapping(tmp_path, directories={"runs": "research/artifacts/general",
                                   "runs/imported_m0": "research/artifacts/legacy/M0"})
    assert legacy_input(tmp_path, "runs/imported_m0/run/model.pt") == archived
    archived.unlink()
    with pytest.raises(FileNotFoundError, match="Required evidence is unavailable"):
        legacy_input(tmp_path, "runs/imported_m0/run/model.pt")


def test_exact_file_mapping_overrides_directory_mapping(tmp_path):
    archived = write(tmp_path, "research/legacy/M0/manifests/reference.json", "registered")
    mapping(tmp_path, files={"manifests/reference.json": archived.relative_to(tmp_path).as_posix()},
            directories={"manifests": "research/artifacts/unused"})
    assert legacy_input(tmp_path, "manifests/reference.json") == archived


def test_generated_manifest_does_not_mutate_history(tmp_path):
    archived = write(tmp_path, "research/legacy/M2/result.json", "failed historical attempt")
    mapping(tmp_path, files={"manifests/m2_status.json": archived.relative_to(tmp_path).as_posix()})
    fresh = workspace_path(tmp_path, "manifests/m2_status.json")
    fresh.write_text("running new attempt", encoding="utf-8")
    assert archived.read_text(encoding="utf-8") == "failed historical attempt"
    assert legacy_input(tmp_path, "manifests/m2_status.json") == archived
    assert current_input(tmp_path, "manifests/m2_status.json") == fresh


def test_unmapped_input_can_use_original_location(tmp_path):
    original = write(tmp_path, "runs/new/protocol.json", "new protocol")
    assert legacy_input(tmp_path, "runs/new/protocol.json") == original


@pytest.mark.parametrize("bad", ["../outside", "/absolute", "C:/outside", "runs/../../outside", "runs\\outside"])
def test_path_traversal_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        legacy_input(tmp_path, bad)
    with pytest.raises(ValueError):
        workspace_path(tmp_path, bad)


def test_unsafe_mapping_target_rejected(tmp_path):
    mapping(tmp_path, files={"manifests/reference.json": "../outside"})
    with pytest.raises(ValueError):
        legacy_input(tmp_path, "manifests/reference.json")


def test_missing_map_does_not_change_canonical_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        legacy_input(tmp_path, "manifests/reference.json")


def test_current_input_resolves_scratch_before_archived_predecessor(tmp_path):
    archived = write(tmp_path, "research/artifacts/legacy/M2/run/result.json", "old")
    mapping(tmp_path, directories={"runs/m2": "research/artifacts/legacy/M2/run"})
    assert current_input(tmp_path, "runs/m2/result.json") == archived
    fresh = write(tmp_path, "runs/m2/result.json", "new")
    assert current_input(tmp_path, "runs/m2/result.json") == fresh
