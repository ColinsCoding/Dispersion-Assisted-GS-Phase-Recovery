"""
bead_comparison.py -- Part 1+2 of the SEALS morphology research spec:
reproduce the two-bead comparison from the SEALS paper (7.32um and 9.94um
polystyrene beads), then sweep particle diameter more broadly.

SIMULATED/SYNTHETIC: every trace here comes from the Mie forward model
already validated elsewhere in this repo (_seals_physics.py / measurement.py,
cross-checked against the original MATLAB mie-2.m -- see
projects/seals/README.md), not from a real instrument. This module extracts
FEATURES from that model's output; it does not measure or classify a real
particle. Polystyrene beads are not cells -- see ../morphology/__init__.py.
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.seals.inverse.seals_to_tdgsa import seals_intensity_trace  # noqa: E402

# The SEALS paper's two proof-of-principle bead diameters (meters).
BEAD_A_DIAMETER_M = 7.32e-6
BEAD_B_DIAMETER_M = 9.94e-6


def bead_trace(diameter_m, params_override=None):
    """
    Run the validated SEALS/Mie forward model at a given particle diameter,
    holding every other optical/instrument parameter at its default. Thin
    wrapper around seals_to_tdgsa.seals_intensity_trace -- no physics is
    reimplemented here.

    Returns (lamvec, theta_deg, mie_fields), same as seals_intensity_trace.
    """
    if diameter_m <= 0:
        raise ValueError(f"diameter_m={diameter_m} must be positive")
    p = {"dia": diameter_m}
    if params_override:
        p.update(params_override)
    return seals_intensity_trace(p)


def extract_features(theta_deg, I):
    """
    Extract simple, physically-interpretable features from an intensity-
    vs-angle trace (Part 1's requested feature set):
      n_lobes         -- count of local maxima above 1% of the peak
                          (ignores noise-floor-scale wiggle, not a real
                          detection threshold -- see LIMITATION in the
                          notebook)
      lobe_spacing_deg -- mean angular spacing between consecutive lobes
      peak_intensity   -- max(I)
      integrated_intensity -- trapezoidal integral of I over theta
      centroid_deg     -- intensity-weighted mean angle
      variance_deg2    -- intensity-weighted angular spread about the centroid

    NumPy only (no scipy.signal) for portability across this repo's Python
    environments.
    """
    I = np.asarray(I, dtype=float)
    theta_deg = np.asarray(theta_deg, dtype=float)
    if I.shape != theta_deg.shape:
        raise ValueError(f"I shape {I.shape} != theta_deg shape {theta_deg.shape}")
    if np.any(I < 0):
        raise ValueError("I must be non-negative (it's an intensity)")

    # SEALS's wavelength -> angle mapping comes out monotonically DECREASING
    # in theta (see SEALS.m / seals_to_tdgsa.seals_intensity_trace) -- sort
    # to ascending angle first, so integrated_intensity (a physical area,
    # must be >= 0) and lobe_spacing_deg (a physical distance, must be >= 0)
    # aren't silently sign-flipped by np.trapezoid/np.diff on a
    # decreasing x-array.
    order = np.argsort(theta_deg)
    theta_deg, I = theta_deg[order], I[order]

    threshold = 0.01 * I.max()
    is_peak = np.zeros_like(I, dtype=bool)
    is_peak[1:-1] = (I[1:-1] > I[:-2]) & (I[1:-1] > I[2:]) & (I[1:-1] > threshold)
    peak_angles = theta_deg[is_peak]
    n_lobes = int(is_peak.sum())
    lobe_spacing_deg = float(np.mean(np.diff(peak_angles))) if n_lobes >= 2 else float('nan')

    peak_intensity = float(I.max())
    integrated_intensity = float(np.trapezoid(I, theta_deg))
    weight = I / I.sum()
    centroid_deg = float(np.sum(theta_deg * weight))
    variance_deg2 = float(np.sum(weight * (theta_deg - centroid_deg) ** 2))

    return {
        "n_lobes": n_lobes, "lobe_spacing_deg": lobe_spacing_deg,
        "peak_intensity": peak_intensity, "integrated_intensity": integrated_intensity,
        "centroid_deg": centroid_deg, "variance_deg2": variance_deg2,
    }


def compare_two_beads(dia_a_m=BEAD_A_DIAMETER_M, dia_b_m=BEAD_B_DIAMETER_M):
    """
    Part 1: reproduce the SEALS paper's two-bead comparison. Returns both
    traces (wavelength + angle domain), normalized versions, the difference
    trace, and extracted features for each.
    """
    lam_a, theta_a, mie_a = bead_trace(dia_a_m)
    lam_b, theta_b, mie_b = bead_trace(dia_b_m)
    # The SEALS grating geometry (wavelength -> angle mapping) does not
    # depend on particle diameter -- only the Mie intensity at each angle
    # does. Assert this so a future change that breaks that assumption is
    # caught loudly, rather than silently comparing traces on mismatched
    # angle axes.
    np.testing.assert_allclose(theta_a, theta_b)

    I_a, I_b = mie_a.I_p, mie_b.I_p
    I_a_norm, I_b_norm = I_a / I_a.max(), I_b / I_b.max()

    return {
        "diameter_a_um": dia_a_m * 1e6, "diameter_b_um": dia_b_m * 1e6,
        "lamvec": lam_a, "theta_deg": theta_a,
        "I_a": I_a, "I_b": I_b,
        "I_a_norm": I_a_norm, "I_b_norm": I_b_norm, "diff_norm": I_a_norm - I_b_norm,
        "features_a": extract_features(theta_a, I_a), "features_b": extract_features(theta_a, I_b),
    }


def diameter_sweep(diameters_um=(5, 6, 7.32, 8, 9.94, 11, 12)):
    """
    Part 2: sweep particle diameter, holding every other optical parameter
    fixed. Returns (theta_deg, {diameter_um: I}, features_df) -- everything
    needed for the overlay plot, heatmap, and feature table Part 2 asks for.
    """
    diameters_um = list(diameters_um)
    if len(diameters_um) < 2:
        raise ValueError("need at least 2 diameters for a sweep")

    theta_deg = None
    traces = {}
    rows = []
    for dia_um in diameters_um:
        lam, theta, mie = bead_trace(dia_um * 1e-6)
        if theta_deg is None:
            theta_deg = theta
        else:
            np.testing.assert_allclose(theta, theta_deg)
        traces[dia_um] = mie.I_p
        rows.append({"diameter_um": dia_um, **extract_features(theta, mie.I_p)})

    return theta_deg, traces, pd.DataFrame(rows)
