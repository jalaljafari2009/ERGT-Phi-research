"""Read-only verification of both source trees, data and observed environment."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ergt_phi.runtime import ROOT, REFERENCE, activate_reference


def main():
    activate_reference()
    import torch
    from ergt_four_seed.integrity import verify_integrity
    from ergt_four_seed.data_registry import verify_registered_data
    from ergt_four_seed.environment import verify_environment
    original = ROOT.parent / "ERGT-paper-main"
    record = {
        "original_integrity": verify_integrity(original) if original.is_dir() else {"available": False, "reason": "Original desktop tree is not present on this host"},
        "reference_integrity": verify_integrity(REFERENCE),
        "data_parity": verify_registered_data(),
        "environment": verify_environment(),
        "runtime_settings": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cpu_threads": torch.get_num_threads(), "dtype": str(torch.get_default_dtype()),
            "cuda_available": torch.cuda.is_available(), "torch_cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
        "python_executable": sys.executable, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "pytest", "dulwich")},
    }
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        record["nvidia_smi"] = {"returncode": gpu.returncode, "stdout": gpu.stdout, "stderr": gpu.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        record["nvidia_smi"] = {"available": False, "error": str(error)}
    (ROOT / "manifests/environment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (ROOT / "requirements-m0.lock.txt").write_text(packages.stdout, encoding="utf-8")
    config = json.loads((REFERENCE / "configs/immutable_final_configuration.json").read_text())
    recipe = json.loads((REFERENCE / "contracts/selected_transformer_manifest.json").read_text())["selected_candidate"]["recipe"]
    (ROOT / "manifests/resolved_paper_protocol.json").write_text(json.dumps({
        "native_and_data": config, "transformer_recipe": recipe,
        "native_answer_weight": .15, "native_teacher_weight_during_updates": 1.0,
        "transformer_total_registered_steps": recipe["warmup_steps"] + sum(s["steps"] for s in recipe["curriculum_stages"]),
        "note": "Official entrypoint also attaches qualification and execution audits. This is a descriptive export, not an alternate execution config.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"integrity": True, "cohort_count": record["data_parity"]["cohort_count"], "cuda": torch.cuda.is_available(), "report": "manifests/environment.json"}))


if __name__ == "__main__":
    main()
