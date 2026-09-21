"""Build the preregistered M2 fresh-development identity lock without model evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ergt_phi.m2_fresh import build_fresh_development, confirmation_windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="configs/m2_fresh_v1.json")
    parser.add_argument(
        "--reference-metadata",
        default="research/legacy/LEGACY-M0/evidence/imported_m0/m0_single_seed_reference",
    )
    parser.add_argument(
        "--output", default="research/plans/M2_E003_FRESH_DEVELOPMENT_LOCK.json",
    )
    args = parser.parse_args()
    contract = json.loads((ROOT / args.contract).read_text(encoding="utf-8"))
    fresh = contract["fresh_development"]
    examples, _, audit = build_fresh_development(
        ROOT / args.reference_metadata,
        data_seed=int(fresh["data_seed"]),
        pair_counts_by_hop=fresh["pair_counts_by_hop"],
        unsupported_pairs=int(fresh["unsupported_pairs"]),
    )
    first, second = confirmation_windows(examples, seed=int(fresh["window_seed"]))
    first_pairs = sorted({examples[index].pair_id for index in first})
    second_pairs = sorted({examples[index].pair_id for index in second})
    document = {
        **audit,
        "contract": args.contract.replace("\\", "/"),
        "window_seed": int(fresh["window_seed"]),
        "windows": {
            "window_1": {
                "example_ids": [examples[index].example_id for index in first],
                "pair_ids": first_pairs,
            },
            "window_2": {
                "example_ids": [examples[index].example_id for index in second],
                "pair_ids": second_pairs,
            },
        },
        "window_pair_disjoint": not bool(set(first_pairs) & set(second_pairs)),
        "results_observed_when_lock_created": False,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "fresh_cohort_sha256": audit["fresh_cohort_sha256"],
        "examples": audit["examples"],
        "pairs": audit["pairs"],
        "window_examples": [len(first), len(second)],
        "raw_text_disjoint": audit["raw_text_disjoint_from_m0_training"],
        "pair_disjoint": audit["counterfactual_pair_disjoint_from_m0_training"],
    }, indent=2))


if __name__ == "__main__":
    main()
