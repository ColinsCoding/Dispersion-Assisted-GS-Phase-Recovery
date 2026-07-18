"""Optics stage: ray/beam (ABCD) matrix optics."""
from optics.abcd import (
    free_space,
    propagate_q,
    q_at_waist,
    thin_lens,
    width_from_q,
)

__all__ = ["free_space", "thin_lens", "propagate_q", "q_at_waist", "width_from_q"]
