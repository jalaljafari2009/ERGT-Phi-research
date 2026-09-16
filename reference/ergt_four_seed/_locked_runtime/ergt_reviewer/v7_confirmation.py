"""Compatibility name retained without historical runtime code."""

def _not_distributed(*args, **kwargs):
    raise RuntimeError(
        "Historical development qualification is intentionally excluded. "
        "Use ergt_four_seed.run_four_seed_study for the locked final protocol."
    )

run_v7_final_confirmation = _not_distributed
