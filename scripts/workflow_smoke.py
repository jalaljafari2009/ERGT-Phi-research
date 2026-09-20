"""Deterministic infrastructure fixture, never a scientific ERGT result.

Used by the versioned notebook workflow to exercise execution, export, import
and preregistered gates without GPUs, torch or training data.
"""
from pathlib import Path
import json


def main():
    output = Path("runs/workflow_smoke")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "scope": "infrastructure_smoke_only",
        "scientific_evidence": False,
        "checks": {"arithmetic": sum([1, 2, 3]) == 6, "deterministic": True},
        "M2_complete": False,
        "M3_authorized_by_results": False,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
