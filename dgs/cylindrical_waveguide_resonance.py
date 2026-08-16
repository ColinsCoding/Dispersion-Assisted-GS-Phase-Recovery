"""cylindrical_waveguide_resonance.py -- why cylindrical geometry makes
Bessel functions physically inevitable, and how a cylindrical
cavity/waveguide only strongly couples to input radiation near its
discrete resonant frequencies.

DERIVATION SKETCH (the reason this module exists, not just a formula
lookup): the Helmholtz equation nabla^2(psi) + k^2*psi = 0 in cylindrical
coordinates (r, phi, z) separates as psi = R(r)*Phi(phi)*Z(z). The radial
factor R(r) satisfies BESSEL'S EQUATION,
    r^2 R'' + r R' + (k_c^2 r^2 - m^2) R = 0,
whose solution regular at r=0 (finite field on the axis -- a hard physical
requirement, not a convenience) is R(r) = J_m(k_c*r). This is the SAME
J_m already wired into live_cas_ollama_win.py's Bessel worked example --
here it's not an abstract special function, it's literally "the radial
shape of a field confined to a cylinder."

BOUNDARY CONDITIONS pick out which k_c values are allowed (perfectly
conducting circular waveguide/cavity, the classic solvable case):
  TM modes (E_z = 0 at r=a):        J_m(k_c*a) = 0   -> k_c = j_{m,n}/a
  TE modes (dE_z/dr = 0 at r=a):    J_m'(k_c*a) = 0  -> k_c = j'_{m,n}/a
where j_{m,n} is the n-th positive zero of J_m (scipy.special.jn_zeros)
and j'_{m,n} is the n-th positive zero of J_m' (scipy.special.jnp_zeros).

RESONANCE: a cavity (both ends closed, length L) only supports standing
waves at discrete f_{mnp}; a waveguide (open, propagating along z) only
PROPAGATES above its cutoff f_c -- below cutoff the wave is evanescent
(exponentially decaying, not radiating). Driving either with INPUT
RADIATION at a frequency near an eigenfrequency produces a resonance peak
(Lorentzian in amplitude-squared, set by the cavity's Q factor); far from
resonance, coupling is weak and mostly reflects.

Scope: the exactly-solvable perfectly-conducting-wall case (classic
microwave-engineering circular waveguide/cavity). Optical fiber LP modes
are the "soft wall" generalization (continuity of field/derivative across
a core-cladding INDEX step rather than a hard E=0 wall) -- a harder,
transcendental root-finding problem, left for a future module rather than
folded in here.

Requires scipy (available on py-3.13 in this repo; see
environment_scipy_available memory note).
"""

from __future__ import annotations
import numpy as np
from scipy import special

C_LIGHT = 299792458.0   # m/s


def _validate_mode_indices(m: int, n: int) -> None:
    if m < 0:
        raise ValueError(f"m must be >= 0, got {m}")
    if n < 1:
        raise ValueError(f"n must be >= 1 (n-th zero, 1-indexed), got {n}")


def _validate_boundary(boundary: str) -> None:
    if boundary not in ("TM", "TE"):
        raise ValueError(f"boundary must be 'TM' or 'TE', got {boundary!r}")


# ── 1. Radial wavenumber: which k_c values the boundary condition allows ────

def radial_wavenumber(m: int, n: int, a: float, boundary: str = "TM") -> float:
    """k_c = j_{m,n}/a (TM: J_m(k_c*a)=0) or j'_{m,n}/a (TE: J_m'(k_c*a)=0).
    `a` is the waveguide/cavity radius (m)."""
    _validate_mode_indices(m, n)
    _validate_boundary(boundary)
    if a <= 0:
        raise ValueError(f"a (radius) must be > 0, got {a}")
    if boundary == "TM":
        zero = special.jn_zeros(m, n)[-1]
    else:
        zero = special.jnp_zeros(m, n)[-1]
    return float(zero) / a


def verify_boundary_condition(m: int, n: int, a: float, boundary: str = "TM",
                              tol: float = 1e-9) -> bool:
    """CHECKED, not assumed: the k_c returned by radial_wavenumber must
    actually make J_m(k_c*a) (TM) or J_m'(k_c*a) (TE) vanish -- confirms
    scipy's root, not just trusts the function name."""
    k_c = radial_wavenumber(m, n, a, boundary)
    if boundary == "TM":
        residual = special.jv(m, k_c * a)
    else:
        residual = special.jvp(m, k_c * a, 1)
    if abs(residual) > tol:
        label = f"J_{m}({k_c*a:.6f})" if boundary == "TM" else f"J_{m}'({k_c*a:.6f})"
        raise AssertionError(f"boundary condition not satisfied: {label} = {residual:.2e}, expected ~0")
    return True


# ── 2. Radial field profile: the physical shape, not just the eigenvalue ────

def radial_mode_profile(m: int, n: int, a: float, r: np.ndarray, boundary: str = "TM") -> np.ndarray:
    """R(r) = J_m(k_c*r) for 0 <= r <= a -- the actual field shape across
    the cross-section, using the k_c the boundary condition selects."""
    r = np.asarray(r, dtype=float)
    if np.any(r < 0) or np.any(r > a * (1 + 1e-9)):
        raise ValueError(f"r must lie in [0, a={a}]")
    k_c = radial_wavenumber(m, n, a, boundary)
    return special.jv(m, k_c * r)


# ── 3. Cutoff frequency (waveguide) and resonant frequency (cavity) ─────────

def waveguide_cutoff_frequency(m: int, n: int, a: float, boundary: str = "TM",
                               c: float = C_LIGHT) -> float:
    """f_c = c*k_c/(2*pi). Below f_c the mode is evanescent (cannot
    propagate down the guide); above f_c it propagates with a real beta."""
    k_c = radial_wavenumber(m, n, a, boundary)
    return c * k_c / (2 * np.pi)


def waveguide_propagation_constant(f: float, m: int, n: int, a: float,
                                   boundary: str = "TM", c: float = C_LIGHT):
    """beta = sqrt((2*pi*f/c)^2 - k_c^2). Returns a REAL beta (propagating)
    if f > f_c, or a negative-imaginary-magnitude marker via a complex
    return (evanescent decay rate) if f < f_c -- both are physical, but the
    caller must check which regime it got (np.iscomplex)."""
    if f <= 0:
        raise ValueError(f"f must be > 0, got {f}")
    k_c = radial_wavenumber(m, n, a, boundary)
    k = 2 * np.pi * f / c
    return np.sqrt(complex(k**2 - k_c**2))


def cavity_resonant_frequency(m: int, n: int, p: int, a: float, L: float,
                              boundary: str = "TM", c: float = C_LIGHT) -> float:
    """f_{mnp} = c/(2*pi) * sqrt(k_c^2 + (p*pi/L)^2) -- a CLOSED cavity
    (both ends capped) additionally quantizes the axial direction into a
    standing wave, p = 0, 1, 2, ... half-wavelengths along L."""
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}")
    if p < 0:
        raise ValueError(f"p must be >= 0, got {p}")
    k_c = radial_wavenumber(m, n, a, boundary)
    k_z = p * np.pi / L
    return c / (2 * np.pi) * np.sqrt(k_c**2 + k_z**2)


def dominant_mode_cutoff(a: float, c: float = C_LIGHT) -> dict:
    """The lowest-cutoff (dominant) propagating mode in a circular
    waveguide is TE11 -- CHECKED here against TM01 and TE21 rather than
    quoted from a microwave-engineering table, since j'_{1,1} < j_{0,1}
    is not obvious without computing both."""
    candidates = {
        "TE11": waveguide_cutoff_frequency(1, 1, a, "TE", c),
        "TM01": waveguide_cutoff_frequency(0, 1, a, "TM", c),
        "TE21": waveguide_cutoff_frequency(2, 1, a, "TE", c),
        "TM11": waveguide_cutoff_frequency(1, 1, a, "TM", c),
    }
    dominant = min(candidates, key=candidates.get)
    if dominant != "TE11":
        raise AssertionError(f"expected TE11 to be dominant, got {dominant}: {candidates}")
    return {"dominant": dominant, "cutoffs_Hz": candidates}


# ── 4. Driven resonance: coupling strength vs. input radiation frequency ────

def driven_resonance_response(f: np.ndarray, f0: float, Q: float) -> np.ndarray:
    """Normalized power response of a driven damped resonator (the
    standard Lorentzian-in-frequency lineshape):
        |A(f)|^2 = 1 / [ (1 - (f/f0)^2)^2 + (f/(f0*Q))^2 ]
    NOTE: the peak of this curve is NOT exactly at f=f0 for finite Q (a
    common assumption that's wrong) -- see resonance_peak_frequency. What
    IS exactly true at f=f0 is |A(f0)|^2 = Q^2 (verified in
    verify_resonance_peak), which is a different, weaker statement than
    "f0 is the peak.\""""
    f = np.asarray(f, dtype=float)
    if f0 <= 0:
        raise ValueError(f"f0 must be > 0, got {f0}")
    if Q <= 0:
        raise ValueError(f"Q must be > 0, got {Q}")
    detuning = 1 - (f / f0)**2
    damping = f / (f0 * Q)
    return 1.0 / (detuning**2 + damping**2)


def resonance_peak_frequency(f0: float, Q: float) -> float:
    """The TRUE peak of driven_resonance_response, found by setting
    d|A|^2/df = 0: f_peak = f0*sqrt(1 - 1/(2*Q^2)) -- strictly BELOW f0 for
    any finite Q, converging to f0 only as Q -> infinity. Raises ValueError
    if Q <= 1/sqrt(2): below that, the "peak" (in f>0) doesn't exist as an
    interior maximum -- the response decreases monotonically from f=0."""
    if Q <= 1.0 / np.sqrt(2):
        raise ValueError(f"Q={Q} <= 1/sqrt(2) -- no interior resonance peak exists "
                         f"(response is monotonically decreasing in f)")
    return f0 * np.sqrt(1 - 1 / (2 * Q**2))


def verify_resonance_peak(f0: float, Q: float, span: float = 0.5, n_pts: int = 200_001) -> bool:
    """CHECKED, both halves: (1) the response at EXACTLY f0 equals Q^2
    (an exact identity, independent of the peak-shift subtlety), and (2)
    the numerically-located peak matches resonance_peak_frequency's
    formula f0*sqrt(1-1/(2Q^2)) -- NOT f0 itself, correcting the common
    assumption that amplitude resonance sits exactly at the natural
    frequency."""
    exact_at_f0 = driven_resonance_response(np.array([f0]), f0, Q)[0]
    if abs(exact_at_f0 - Q**2) / Q**2 > 1e-9:
        raise AssertionError(f"response at f0: {exact_at_f0:.6g} != Q^2={Q**2:.6g}")

    f = np.linspace(f0 * (1 - span), f0 * (1 + span), n_pts)
    response = driven_resonance_response(f, f0, Q)
    peak_idx = int(np.argmax(response))
    numeric_peak_f = f[peak_idx]
    predicted_peak_f = resonance_peak_frequency(f0, Q)
    grid_spacing = f[1] - f[0]
    if abs(numeric_peak_f - predicted_peak_f) > 2 * grid_spacing:
        raise AssertionError(f"numeric peak at f={numeric_peak_f:.6g}, "
                             f"formula predicts f={predicted_peak_f:.6g}")
    return True


if __name__ == "__main__":
    a = 0.01   # 1 cm radius circular waveguide

    print("=== 1. Radial wavenumber & boundary condition (checked, not assumed) ===")
    for (m, n, boundary) in [(0, 1, "TM"), (1, 1, "TE")]:
        k_c = radial_wavenumber(m, n, a, boundary)
        ok = verify_boundary_condition(m, n, a, boundary)
        print(f"  {boundary}{m}{n}: k_c = {k_c:.4f} rad/m, boundary condition verified: {ok}")

    print("\n=== 2. Dominant mode: TE11 has the lowest cutoff (checked against TM01, TE21, TM11) ===")
    dom = dominant_mode_cutoff(a)
    for name, fc in dom["cutoffs_Hz"].items():
        marker = "  <-- dominant" if name == dom["dominant"] else ""
        print(f"  {name}: f_c = {fc/1e9:.4f} GHz{marker}")

    print("\n=== 3. Cavity resonant frequencies f_mnp (radial + axial quantization) ===")
    L = 0.03   # 3 cm cavity length
    for p in (0, 1, 2):
        f_mnp = cavity_resonant_frequency(1, 1, p, a, L, "TE")
        print(f"  TE11p, p={p}: f = {f_mnp/1e9:.4f} GHz")

    print("\n=== 4. Driven resonance: coupling to input radiation, peak NEAR (not exactly at) f0 ===")
    f0 = waveguide_cutoff_frequency(1, 1, a, "TE")
    for Q in (5, 50, 500):
        ok = verify_resonance_peak(f0, Q)
        f_peak = resonance_peak_frequency(f0, Q)
        print(f"  Q={Q:>4d}: response(f0)=Q^2={Q**2} exactly; true peak at "
              f"f={f_peak/1e9:.4f} GHz (f0={f0/1e9:.4f} GHz); verified: {ok}")

    print("\nA cylinder confines fields into Bessel-function radial profiles because")
    print("that's the ONLY solution to the Helmholtz equation finite on the axis --")
    print("resonance/cutoff frequencies follow directly from where the boundary")
    print("condition forces that profile to vanish (TM) or flatten (TE) at r=a.")
