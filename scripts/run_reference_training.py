"""Launch the unchanged registered protocol on CUDA; never substitute CPU silently."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ergt_phi.research_paths import workspace_path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ergt_phi.runtime import ROOT, REFERENCE, activate_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "four"), default="single")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/reference_training")
    args = parser.parse_args()
    activate_reference()
    import torch
    from ergt_four_seed.integrity import verify_integrity
    from ergt_four_seed.data_registry import verify_registered_data
    verify_integrity(REFERENCE)
    if not torch.cuda.is_available():
        (workspace_path(ROOT, "manifests/reference_training_status.json")).write_text(json.dumps({
            "status": "pending_cuda", "mode": args.mode, "trained_checkpoint_created": False,
            "reason": "CUDA unavailable in the current execution environment",
        }, indent=2), encoding="utf-8")
        raise SystemExit("M0-B pending: CUDA is unavailable. Run this unchanged script on a compatible CUDA host or use the reference Colab notebook. No trained checkpoint was produced.")
    verify_registered_data()
    if args.mode == "four":
        from ergt_four_seed import run_four_seed_study
        result = run_four_seed_study(device="cuda", output_root=args.output, resume=True, copy_outputs_to_downloads=False)
    else:
        from ergt_reviewer.v9_confirmation import run_geometric_reasoning_study
        result = run_geometric_reasoning_study(mode="single_seed_validation", device="cuda", output_root=args.output,
            run_id="m0_single_seed_reference", resume=True, copy_outputs_to_downloads=False)
    verdict_path = Path(result["run_root"]) / "final_verdict.json"
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    (workspace_path(ROOT, "manifests/reference_training_status.json")).write_text(json.dumps({
        "status": "execution_completed", "mode": args.mode,
        "run_root": str(result["run_root"]), "summary_path": str(result["summary_path"]),
        "native_claim_status": verdict.get("native_ergt_claim_status"),
        "execution_integrity_pass": verdict.get("execution_integrity_pass"),
        "note": "Inspect checkpoint selection and native readiness before admitting a trained reference.",
    }, indent=2), encoding="utf-8")
    print(result["summary_path"])


if __name__ == "__main__":
    main()
