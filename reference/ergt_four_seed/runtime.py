"""Load the byte-locked scientific runtime from this package only."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LOCKED_RUNTIME_ROOT = Path(__file__).resolve().parent / "_locked_runtime"


def activate_locked_runtime() -> Path:
    runtime = str(LOCKED_RUNTIME_ROOT)
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    return LOCKED_RUNTIME_ROOT
