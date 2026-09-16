"""Fixed eight-world contract used by the native ERGT path.

This small module replaces the historical experiment dependency graph.  The
reviewer package needs only the registered world names and role permissions;
it does not import any former comparison arm.
"""

from __future__ import annotations

WORLD_NAMES = (
    "causal_flow",
    "semantic_manifold",
    "predictive_reconstruction",
    "memory_curvature",
    "long_bridge",
    "contrast_boundary",
    "novelty_surprise",
    "density_source_sink",
)

TYPED_SLOT_NAMES = ("entity", "relation", "pointer", "query", "boundary")

ROLE_WORLD_ASSIGNMENTS = {
    "entity": (3, 7),
    "relation": (1, 2, 4, 5),
    "pointer": (0, 3, 4, 7),
    "query": (1, 5, 6, 7),
    "boundary": (5, 6, 7),
}

__all__ = ["ROLE_WORLD_ASSIGNMENTS", "TYPED_SLOT_NAMES", "WORLD_NAMES"]
