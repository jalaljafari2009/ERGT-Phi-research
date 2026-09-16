"""Regenerate and verify every registered raw-input cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .runtime import activate_locked_runtime


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _cohorts(bundle: Any) -> dict[str, tuple[Any, ...]]:
    return {
        "train": bundle.train,
        "tuning": bundle.tuning,
        "native_readiness": bundle.native_readiness,
        "mechanism_tuning": bundle.mechanism_tuning,
        "validation": bundle.validation,
        **bundle.hop_cohorts,
        **bundle.context_cohorts,
        **bundle.unsupported_cohorts,
    }


def verify_registered_data() -> dict[str, Any]:
    activate_locked_runtime()
    from ergt_reviewer.fair_data_v9 import build_dataset_bundle
    from ergt_reviewer.matched_data import manifest_hash

    package_root = Path(__file__).resolve().parents[1]
    runtime_root = Path(__file__).resolve().parent / "_locked_runtime"
    config = json.loads(
        (runtime_root / "configs" / "paper_final_v9.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (package_root / "data" / "cohort_fingerprints.json").read_text(encoding="utf-8")
    )
    shared_raw_texts: set[str] = set()
    observed: dict[str, Any] = {}
    mismatches: list[str] = []
    for data_seed in config["data_seeds"]:
        bundle = build_dataset_bundle(
            config, int(data_seed), forbidden_raw_texts=shared_raw_texts
        )
        seed_record: dict[str, Any] = {"cohorts": {}}
        for name, examples in _cohorts(bundle).items():
            record = {"count": len(examples), "manifest_sha256": manifest_hash(examples)}
            seed_record["cohorts"][name] = record
            expected = registry["seeds"][str(data_seed)]["cohorts"].get(name)
            if expected != record:
                mismatches.append(f"{data_seed}:{name}")
        tokenizer_record = bundle.tokenizer.to_json_record()
        seed_record["tokenizer_sha256"] = hashlib.sha256(_canonical(tokenizer_record)).hexdigest()
        if seed_record["tokenizer_sha256"] != registry["seeds"][str(data_seed)]["tokenizer_sha256"]:
            mismatches.append(f"{data_seed}:tokenizer")
        observed[str(data_seed)] = seed_record
    canonical_path = package_root / "data" / "canonical_registered_inputs.jsonl"
    canonical_hash = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    if canonical_hash != registry["canonical_registered_inputs_sha256"]:
        mismatches.append("canonical_registered_inputs.jsonl")
    if mismatches:
        raise RuntimeError("registered data parity failure: " + ", ".join(mismatches))
    return {
        "pass": True,
        "data_seeds": list(config["data_seeds"]),
        "cohort_count": sum(len(value["cohorts"]) for value in observed.values()),
        "canonical_registered_inputs_sha256": canonical_hash,
    }
