"""Compatibility gate; future active phase modes are deliberately unavailable."""
from dataclasses import dataclass
import math
from torch import nn


@dataclass(frozen=True)
class PhaseConfig:
    phase_mode: str = "disabled"
    coupling: float = 0.0
    mask_mode: str = "frozen"
    feedback_mode: str = "lagged"
    nu: float = 0.0
    inner_steps: int = 24

    def __post_init__(self):
        if self.phase_mode not in {"disabled", "shadow", "coupled"}:
            raise ValueError("unknown phase_mode")
        if not math.isfinite(self.coupling) or self.coupling < 0:
            raise ValueError("coupling must be finite and nonnegative")
        if self.mask_mode != "frozen" or self.feedback_mode != "lagged":
            raise ValueError("M0 only reserves frozen masks and lagged feedback")
        if not math.isfinite(self.nu) or self.nu < 0 or self.inner_steps <= 0:
            raise ValueError("invalid future phase configuration")


class ResearchModel(nn.Module):
    def __init__(self, baseline, phase=PhaseConfig()):
        super().__init__()
        self.baseline = baseline
        self.phase = phase

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith("_") or name == "baseline":
                raise
            # Preserve the native evaluator/auditor's public configuration and flags.
            # Do not register modules a second time or shadow nn.Module internals.
            return getattr(super().__getattr__("baseline"), name)

    def forward(self, raw_token_ids, attention_mask, *, observe_spectrum=False):
        if self.phase.phase_mode == "disabled" or self.phase.coupling == 0:
            return self.baseline(raw_token_ids, attention_mask, observe_spectrum=observe_spectrum)
        raise NotImplementedError("Phase activation starts after M0 acceptance; no silent fallback")

    def forward_with_intervention(self, raw_token_ids, attention_mask, *, intervention, intervention_seed=0, observe_spectrum=False):
        if self.phase.phase_mode == "disabled" or self.phase.coupling == 0:
            return self.baseline.forward_with_intervention(raw_token_ids, attention_mask,
                intervention=intervention, intervention_seed=intervention_seed, observe_spectrum=observe_spectrum)
        raise NotImplementedError("Phase interventions are not implemented in M0")
