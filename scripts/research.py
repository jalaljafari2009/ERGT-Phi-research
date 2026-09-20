"""Run the local research workflow without loading the model or PyTorch."""

from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    from scripts.check_spec_lock import verify_spec_lock
    from research_tools.workflow import main

    # Match the workflow's --root override: protect the checkout being acted on.
    preflight = argparse.ArgumentParser(add_help=False)
    preflight.add_argument("--root", type=Path, default=ROOT)
    target, _ = preflight.parse_known_args()
    try:
        verify_spec_lock(target.root)
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"Specification lock failed: {exc}") from exc
    raise SystemExit(main(root=ROOT))
