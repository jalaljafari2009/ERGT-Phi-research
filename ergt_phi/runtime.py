"""Resolve the pinned reference, rejecting ambiguous runtime imports."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference"
LOCKED_RUNTIME = REFERENCE / "ergt_four_seed/_locked_runtime"


def activate_reference():
    for name in ("ergt_four_seed", "ergt_reviewer"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            if REFERENCE not in Path(module.__file__).resolve().parents:
                raise RuntimeError(f"Ambiguous reference import: {name} from {module.__file__}")
    for path in (REFERENCE, LOCKED_RUNTIME):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return LOCKED_RUNTIME
