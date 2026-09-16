"""Runtime compatibility and reference-environment reporting."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


MINIMUM_PYTHON = (3, 10)
MINIMUM_PYTORCH = (2, 0)
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "torch": "torch>=2.0",
}


def _numeric_version(value: str, width: int = 3) -> tuple[int, ...]:
    parts = [int(part) for part in re.findall(r"\d+", value)[:width]]
    return tuple(parts + [0] * (width - len(parts)))


def ensure_required_packages(*, install_missing: bool = True) -> dict[str, Any]:
    """Install only dependencies that are absent or below a required API floor."""

    if sys.version_info[:2] < MINIMUM_PYTHON:
        raise RuntimeError(
            "Python 3.10 or newer is required; select a current Google Colab runtime"
        )
    requested: dict[str, str] = {}
    for module_name, requirement in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            requested[module_name] = requirement
    if "torch" not in requested:
        torch_version = importlib.metadata.version("torch")
        if _numeric_version(torch_version, 2) < MINIMUM_PYTORCH:
            requested["torch"] = REQUIRED_PACKAGES["torch"]
    if requested and install_missing:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                *requested.values(),
            ],
            check=True,
        )
        importlib.invalidate_caches()
    unresolved = [
        module_name
        for module_name in REQUIRED_PACKAGES
        if importlib.util.find_spec(module_name) is None
    ]
    if unresolved:
        raise RuntimeError("required Python packages are unavailable: " + ", ".join(unresolved))
    torch_version = importlib.metadata.version("torch")
    if _numeric_version(torch_version, 2) < MINIMUM_PYTORCH:
        raise RuntimeError("PyTorch 2.0 or newer is required")
    return {
        "pass": True,
        "installed_or_upgraded": sorted(requested) if install_missing else [],
        "required_packages": dict(REQUIRED_PACKAGES),
        "minimum_python": ".".join(map(str, MINIMUM_PYTHON)),
        "minimum_pytorch": ".".join(map(str, MINIMUM_PYTORCH)),
    }


def verify_environment(strict: bool = True) -> dict[str, Any]:
    missing_core = [
        module_name
        for module_name in ("numpy", "torch")
        if importlib.util.find_spec(module_name) is None
    ]
    if missing_core:
        raise RuntimeError("required core packages are unavailable: " + ", ".join(missing_core))
    import numpy
    import torch

    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "environment.lock.json").read_text(encoding="utf-8"))
    pandas_version = (
        importlib.metadata.version("pandas")
        if importlib.util.find_spec("pandas") is not None
        else None
    )
    observed = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "torch_release": torch.__version__.split("+", 1)[0],
        "pandas": pandas_version,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    reference_checks = {
        "python": observed["python"] == lock["python"],
        "numpy": observed["numpy"] == lock["numpy"],
        "torch": observed["torch_release"] == lock["pytorch"],
        "pandas": observed["pandas"] == lock["pandas"],
    }
    compatibility_checks = {
        "python_api_floor": sys.version_info[:2] >= MINIMUM_PYTHON,
        "pytorch_api_floor": _numeric_version(observed["torch_release"], 2) >= MINIMUM_PYTORCH,
        "required_core_packages_importable": not missing_core,
    }
    compatible = all(compatibility_checks.values())
    if strict and not compatible:
        failed = [name for name, passed in compatibility_checks.items() if not passed]
        raise RuntimeError(
            "runtime lacks a required execution capability: " + ", ".join(failed)
        )
    return {
        "pass": compatible,
        "checks": compatibility_checks,
        "reference_match": all(reference_checks.values()),
        "reference_checks": reference_checks,
        "reference_match_required": False,
        "notebook_packages_installed_if_missing": dict(REQUIRED_PACKAGES),
        "observed": observed,
        "lock": lock,
    }
