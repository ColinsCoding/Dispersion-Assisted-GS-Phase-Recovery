"""
bench_calculator.py -- quick physical <-> normalized dispersion conversions
for planning/interpreting a REAL TD-GSA bench measurement (oscilloscope +
dispersive fiber), not just simulated data.

Reuses the exact dimensional-analysis derivation already built and SymPy-
verified in notebooks/phase_retrieval.ipynb (Section "Dimensional analysis:
GS convergence threshold") -- this module does not re-derive anything, it
extracts that notebook's formulas into a tested, reusable, importable form
so they're usable at the bench, not just inside one notebook cell.

Physics, in the order the formulas below implement it:
  1. A signal of bandwidth B [GHz] passed through a fiber with dispersion
     D_phys [ps/nm] accumulates a quadratic phase whose value at the edge
     of that bandwidth is phi(B) = alpha * D_phys * B^2  [rad], where
     alpha = pi * lambda0^2 / c  (lambda0 = probe wavelength, nm; c in
     km/s so alpha comes out in nm*ps/GHz^2).
  2. dgs.gs_core's normalized dispersion parameter is
     D_norm = (alpha/pi) * D_phys * fs^2   (fs = sample rate, GHz).
  3. GS's alternating-projections algorithm needs phi(B) >= 5*pi (see
     dgs.gs_core's own |D_norm|>=5000 rule of thumb, and the SymPy
     derivation this module is extracted from) to reliably converge, i.e.
     |D_phys|_min = 5*pi / (alpha * B^2).
  4. For a fiber with per-km dispersion D_per_km [ps/nm/km] (SMF-28 at
     C-band: ~17 ps/nm/km), the physical length needed is
     L = D_phys / D_per_km  [km].

Run: py -3.12 -m dgs.bench_calculator
"""
import numpy as np

SMF28_D_PER_KM = 17.0   # ps/nm/km, standard single-mode fiber at C-band (~1550 nm)
C_KM_S = 3e5             # km/s -- matches the notebook derivation's rounding


def alpha_ghz(lambda0_nm=1550.0, c_km_s=C_KM_S):
    """alpha = pi * lambda0^2 / c, in nm*ps/GHz^2 -- the physical-units
    dispersion coefficient this whole module is built on."""
    if lambda0_nm <= 0:
        raise ValueError(f"lambda0_nm={lambda0_nm} must be positive")
    return np.pi * lambda0_nm ** 2 / c_km_s * 1e-6


def normalize_dispersion(D_phys_ps_nm, fs_GHz, lambda0_nm=1550.0):
    """Physical dispersion [ps/nm] + sample rate [GHz] -> dgs.gs_core's
    normalized D. This is what retrieve_phase's D1/D2 arguments expect."""
    if fs_GHz <= 0:
        raise ValueError(f"fs_GHz={fs_GHz} must be positive")
    return (alpha_ghz(lambda0_nm) / np.pi) * D_phys_ps_nm * fs_GHz ** 2


def denormalize_dispersion(D_norm, fs_GHz, lambda0_nm=1550.0):
    """Inverse of normalize_dispersion: normalized D -> physical D [ps/nm]
    at a given sample rate."""
    if fs_GHz <= 0:
        raise ValueError(f"fs_GHz={fs_GHz} must be positive")
    return D_norm * np.pi / (alpha_ghz(lambda0_nm) * fs_GHz ** 2)


def min_physical_dispersion(bandwidth_GHz, lambda0_nm=1550.0):
    """|D_phys|_min [ps/nm] for GS to reliably converge on a signal of the
    given bandwidth [GHz] -- the physical-units form of dgs.gs_core's
    |D_norm| >= 5000 rule of thumb (phi(B) >= 5*pi)."""
    if bandwidth_GHz <= 0:
        raise ValueError(f"bandwidth_GHz={bandwidth_GHz} must be positive")
    return 5 * np.pi / (alpha_ghz(lambda0_nm) * bandwidth_GHz ** 2)


def fiber_length_km(D_phys_ps_nm, D_per_km=SMF28_D_PER_KM):
    """Physical dispersion [ps/nm] -> fiber length [km] needed, for a fiber
    with the given per-km dispersion (default: SMF-28 at C-band)."""
    if D_per_km == 0:
        raise ValueError("D_per_km must be nonzero")
    return D_phys_ps_nm / D_per_km


def bench_plan(bandwidth_GHz, D_per_km=SMF28_D_PER_KM, lambda0_nm=1550.0, margin=1.5):
    """
    One-shot bench-planning summary: given the signal bandwidth you expect
    to measure, and a fiber's per-km dispersion, returns the minimum
    physical dispersion needed for convergence, a MARGINED target (default
    1.5x the minimum -- comfortably above the threshold, not right at the
    edge of it), and the fiber length that target requires.

    Returns a dict: bandwidth_GHz, D_phys_min_ps_nm, D_phys_target_ps_nm,
    length_min_km, length_target_km, D_per_km, lambda0_nm.
    """
    if margin < 1.0:
        raise ValueError(f"margin={margin} should be >= 1.0 (below 1.0 targets UNDER the convergence threshold)")
    D_min = min_physical_dispersion(bandwidth_GHz, lambda0_nm)
    D_target = D_min * margin
    return {
        "bandwidth_GHz": bandwidth_GHz,
        "D_phys_min_ps_nm": D_min,
        "D_phys_target_ps_nm": D_target,
        "length_min_km": fiber_length_km(D_min, D_per_km),
        "length_target_km": fiber_length_km(D_target, D_per_km),
        "D_per_km": D_per_km,
        "lambda0_nm": lambda0_nm,
    }


if __name__ == "__main__":
    print(f"alpha = {alpha_ghz():.4e} nm*ps/GHz^2  (lambda0=1550 nm)\n")

    print("Physical -> normalized D, at a few sample rates (D_phys=-695 ps/nm):")
    for fs in [10, 50, 100, 200]:
        D_norm = normalize_dispersion(-695.0, fs)
        print(f"  fs={fs:>4} GHz  ->  D_norm = {D_norm:>10.1f}")
    print()

    print("Bench plan: minimum + margined fiber length, by expected signal bandwidth:")
    print(f"  {'B (GHz)':>10}  {'D_min (ps/nm)':>15}  {'D_target (ps/nm)':>18}  {'L_min (km)':>12}  {'L_target (km)':>14}")
    for B in [1, 10, 100, 500, 1000]:
        p = bench_plan(B)
        print(f"  {B:>10}  {p['D_phys_min_ps_nm']:>15.1f}  {p['D_phys_target_ps_nm']:>18.1f}  "
              f"{p['length_min_km']:>12.3f}  {p['length_target_km']:>14.3f}")
