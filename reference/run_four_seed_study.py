"""Command-line entry point for the immutable final protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from ergt_four_seed import run_four_seed_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    result = run_four_seed_study(
        device=args.device,
        output_root=Path(args.output),
        resume=True,
        copy_outputs_to_downloads=False,
        strict_environment=True,
    )
    print(result["summary_path"])


if __name__ == "__main__":
    main()
