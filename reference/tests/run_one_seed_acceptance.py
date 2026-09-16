"""Optional GPU acceptance for the locked one-seed validation profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from ergt_four_seed.environment import verify_environment
    from ergt_four_seed.integrity import verify_integrity
    from ergt_four_seed.data_registry import verify_registered_data
    from ergt_four_seed.runtime import activate_locked_runtime

    verify_integrity(root)
    verify_environment(strict=True)
    verify_registered_data()
    activate_locked_runtime()
    from ergt_reviewer.v9_confirmation import run_geometric_reasoning_study
    result = run_geometric_reasoning_study(
        mode="single_seed_validation", device="cuda", output_root=Path(args.output),
        run_id="ergt_single_seed_acceptance", resume=True,
        copy_outputs_to_downloads=False,
    )
    print(result["summary_path"])


if __name__ == "__main__":
    main()
