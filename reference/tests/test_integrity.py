import json

from ergt_four_seed.integrity import verify_integrity
from ergt_four_seed.runtime import activate_locked_runtime


def test_release_integrity():
    result = verify_integrity()
    assert result["pass"]
    assert result["verified_file_count"] > 25


def test_internal_scientific_locks():
    activate_locked_runtime()
    from ergt_reviewer.suite_v9 import _locked_native_core_audit
    from ergt_reviewer.v8_confirmation import (
        PACKAGED_QUALIFICATION_MANIFEST,
        validate_qualification_manifest,
    )
    from ergt_reviewer.v9_confirmation import validate_v9_execution_lock

    manifest = json.loads(PACKAGED_QUALIFICATION_MANIFEST.read_text(encoding="utf-8"))
    qualification = validate_qualification_manifest(manifest)
    execution = validate_v9_execution_lock("paper_final_v9", manifest)
    native_core = _locked_native_core_audit()
    assert qualification["pass"]
    assert execution["pass"]
    assert execution["mismatched_files"] == []
    assert native_core["locked_native_core_unchanged"]
