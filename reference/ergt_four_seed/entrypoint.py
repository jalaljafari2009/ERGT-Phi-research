"""Single publication-facing execution path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data_registry import verify_registered_data
from .environment import verify_environment
from .integrity import verify_integrity
from .runtime import activate_locked_runtime


def run_four_seed_study(
    *,
    device: str = "cuda",
    output_root: str | Path,
    run_id: str = "ergt_four_seed_confirmation",
    resume: bool = True,
    copy_outputs_to_downloads: bool = True,
    strict_environment: bool = True,
) -> dict[str, Any]:
    """Run the immutable four-seed protocol after all preflight checks."""

    integrity = verify_integrity()
    environment = verify_environment(strict=strict_environment)
    if device == "cuda" and not environment["observed"]["cuda_available"]:
        raise RuntimeError("CUDA was requested but no GPU is available")
    data_parity = verify_registered_data()
    activate_locked_runtime()
    from ergt_reviewer.v9_confirmation import run_geometric_reasoning_study

    result = run_geometric_reasoning_study(
        mode="four_seed_confirmation",
        device=device,
        output_root=output_root,
        run_id=run_id,
        resume=resume,
        copy_outputs_to_downloads=copy_outputs_to_downloads,
    )
    result["release_integrity"] = integrity
    result["release_environment"] = environment
    result["registered_data_parity"] = data_parity
    return result
