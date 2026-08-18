"""bessel_modes.py -- the analytic LP01 (fundamental) mode of a circular
step-index waveguide, for checking the finite-difference solver
(modes.py) against a known closed-form solution on the ONE geometry
(circle) where cylindrical symmetry makes an exact scalar solution
possible.

Both this analytic solution and laplacian.py/modes.py solve the SAME
scalar Helmholtz equation -- this checks the finite-difference
IMPLEMENTATION against exact math, not the scalar model against real
(vector) physics (that separate, larger approximation is already flagged
in the notebook's Section 1 LIMITATION and is not what this module tests).

Standard weakly-guided-mode (LPmn) formulation (Snyder & Love; Okamoto,
"Fundamentals of Optical Waveguides"): inside the core, R(r)=J_m(u r/a);
outside, R(r)=K_m(w r/a) (modified Bessel, decaying). Continuity of R and
R' at r=a gives the characteristic equation solved below for m=0
(LP0n family); this module solves for the LOWEST root (LP01, fundamental).
"""
import numpy as np
from scipy.special import jv, kv, jn_zeros
from scipy.optimize import brentq


def v_number(n_core: float, n_clad: float, wavelength_um: float, radius_um: float) -> float:
    """The normalized frequency V = k0*a*sqrt(n_core^2-n_clad^2). Determines
    how many LP modes a circular step-index core supports (V<2.405: single
    mode; larger V: progressively more)."""
    if n_core <= n_clad:
        raise ValueError("n_core must exceed n_clad for a guiding structure")
    if wavelength_um <= 0 or radius_um <= 0:
        raise ValueError("wavelength_um and radius_um must be positive")
    k0 = 2 * np.pi / wavelength_um
    return k0 * radius_um * np.sqrt(n_core ** 2 - n_clad ** 2)


def _lp01_residual(u: float, V: float) -> float:
    w = np.sqrt(V ** 2 - u ** 2)
    return u * jv(1, u) / jv(0, u) - w * kv(1, w) / kv(0, w)


def solve_lp01_u(V: float) -> float:
    """Root of the m=0 characteristic equation u*J1(u)/J0(u)=w*K1(w)/K0(w)
    (w=sqrt(V^2-u^2)) in the LP01 branch 0<u<min(V, first zero of J0).
    Scanned on a fine grid for the first sign change, then refined with
    brentq -- robust to J0's pole inside the search interval."""
    if V <= 0:
        raise ValueError("V must be positive")
    u_max = min(V, jn_zeros(0, 1)[0]) - 1e-6
    if u_max <= 0:
        raise ValueError(f"V={V} too small to support a guided LP01 mode (need V > 0)")
    u_scan = np.linspace(1e-3, u_max, 4000)
    residuals = _lp01_residual(u_scan, V)
    sign_changes = np.where(np.diff(np.sign(residuals)) != 0)[0]
    if len(sign_changes) == 0:
        raise RuntimeError(f"no LP01 root found for V={V} -- check core/cladding contrast")
    i = sign_changes[0]
    return brentq(_lp01_residual, u_scan[i], u_scan[i + 1], args=(V,))


def lp01_effective_index(n_core: float, n_clad: float, wavelength_um: float, radius_um: float) -> dict:
    """Solve for the analytic LP01 mode. Returns a dict with n_eff, u, w, V."""
    V = v_number(n_core, n_clad, wavelength_um, radius_um)
    u = solve_lp01_u(V)
    w = np.sqrt(V ** 2 - u ** 2)
    k0 = 2 * np.pi / wavelength_um
    beta_sq = k0 ** 2 * n_core ** 2 - (u / radius_um) ** 2
    n_eff = np.sqrt(beta_sq) / k0
    return {"n_eff": n_eff, "u": u, "w": w, "V": V}


def lp01_radial_profile(r: np.ndarray, u: float, w: float, radius_um: float) -> np.ndarray:
    """The LP01 radial field profile R(r), continuous at r=radius_um,
    peak-normalized to R(0)=1 (a SHAPE comparison, not a power-normalized
    comparison -- modes.py's FD fields use a different normalization
    convention, see Section 6)."""
    r = np.asarray(r, dtype=float)
    if np.any(r < 0):
        raise ValueError("r must be non-negative")
    if radius_um <= 0:
        raise ValueError("radius_um must be positive")
    core = r <= radius_um
    R = np.empty_like(r)
    R[core] = jv(0, u * r[core] / radius_um)
    boundary_value = jv(0, u)  # J0(u), the core solution's value at r=a
    clad_scale = boundary_value / kv(0, w)  # match cladding solution to it at r=a
    R[~core] = clad_scale * kv(0, w * r[~core] / radius_um)
    return R
