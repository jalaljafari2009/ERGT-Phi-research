"""Run all inexpensive standalone acceptance checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "pytest", "-q", str(root / "tests")]
    raise SystemExit(subprocess.run(command, cwd=root, check=False).returncode)
