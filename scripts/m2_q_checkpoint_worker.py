"""Fresh-process resume/reload worker for the registered M2-Q v2 runner."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from ergt_phi.checkpoint import load, save
from ergt_phi.shadow_metrics import evaluate_calibration
from ergt_phi.shadow_phase import CalibrationConfig, PhaseAnchorNetwork, calibration_step


def main() -> None:
    if len(sys.argv) != 5 or sys.argv[1] not in {"resume", "reload"}:
        raise SystemExit(
            "usage: m2_q_checkpoint_worker.py resume|reload CHECKPOINT PAYLOAD OUTPUT"
        )
    mode, checkpoint, payload_path, output = (
        sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    config = CalibrationConfig(**payload["config"])
    phase = PhaseAnchorNetwork(
        payload["input_dim"], payload["worlds"], payload["relations"], config,
    )
    optimizer = torch.optim.AdamW(
        phase.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    progress = load(
        checkpoint, phase, optimizer,
        contract=payload["checkpoint_contract"],
    )
    if mode == "resume":
        calibration_step(
            phase, optimizer, payload["batch"],
            contract=payload["checkpoint_contract"], progress=progress,
        )
        progress["cursor"] += int(payload["cursor_increment"])
        save(
            output, phase, optimizer,
            contract=payload["checkpoint_contract"], progress=progress,
        )
        return

    frozen = bool(progress.get("frozen"))
    parameters_frozen = all(not parameter.requires_grad for parameter in phase.parameters())
    modules_eval = all(not module.training for module in phase.modules())
    if not (frozen and parameters_frozen and modules_eval):
        raise RuntimeError("selected checkpoint did not restore as frozen/eval")
    metrics = evaluate_calibration(
        phase, payload["records"], payload["monitor_indices"],
    )
    result = {
        "pass": True,
        "fresh_process": True,
        "progress_frozen": frozen,
        "parameters_frozen": parameters_frozen,
        "modules_eval": modules_eval,
        "selected_epoch": progress.get("selected_epoch"),
        "selected_step": progress.get("selected_step"),
        "monitor": metrics,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
