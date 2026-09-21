"""Validate the M2-Q v2 addendum/config without running a scientific experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ergt_phi.m2_q_contract import load_m2_q_contract
from scripts.check_spec_lock import verify_spec_lock


def main() -> None:
    verify_spec_lock(ROOT)
    path = ROOT / "configs" / "m2_q_v2.json"
    config = load_m2_q_contract(path)
    addendum = ROOT / config["specification"]["addendum"]
    if not addendum.is_file():
        raise FileNotFoundError(f"missing M2-Q addendum: {addendum}")
    result = {
        "pass": True,
        "schema": config["schema"],
        "revision": config["revision"],
        "status": config["status"],
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "addendum_sha256": hashlib.sha256(addendum.read_bytes()).hexdigest(),
        "scientific_run_performed": False,
        "m3_authorized": False,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
