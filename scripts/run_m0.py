"""Run acceptance in separate interpreters and preserve command logs and JUnit."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-only", action="store_true")
    args = parser.parse_args()
    output = ROOT / "runs/m0"
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    commands = [
        ("reference_audit", ROOT, [str(ROOT / "scripts/audit_reference.py")]),
        ("reference_tests", ROOT / "reference", ["-m", "pytest", "-q", "-p", "no:cacheprovider", "tests", f"--junitxml={output / 'reference-tests.xml'}"]),
        ("research_tests", ROOT, ["-m", "pytest", "-q", "-p", "no:cacheprovider", "tests", f"--junitxml={output / 'research-tests.xml'}"]),
    ]
    previous = output / "acceptance.json"
    summary = []
    if args.research_only:
        commands = [item for item in commands if item[0] == "research_tests"]
        if previous.exists():
            summary = [item for item in json.loads(previous.read_text()) if item["name"] != "research_tests"]
    for name, cwd, args in commands:
        print(f"Starting {name}", flush=True)
        start = time.perf_counter()
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-B", *args], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = {"name": name, "returncode": result.returncode, "seconds": time.perf_counter() - start, "log": str(output / f"{name}.log")}
        summary.append(row)
        print(json.dumps(row), flush=True)
        (output / "acceptance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
