"""dispersion_calculus_problems.py -- a photonics calculus problem set built on
H(f) = exp(i*pi*D*f^2), the dispersion kernel used throughout this repo
(dgs.gs_core, dgs.dispersion_integrals, dgs.dispersive_fourier).

Three problems, each posed, solved symbolically (SymPy), and checked
numerically. Problem 1 is cross-referenced against dgs.dispersion_integrals'
already-verified result rather than re-deriving it from an unchecked
independent guess; Problems 2 and 3 are new here.

  Problem 1 -- Impulse response (a Fresnel/Gaussian integral)
    Integral[H(f)*exp(2*pi*I*f*t), (f,-oo,oo)] = ?
    Solved by completing the square in the exponent. Cross-checked against
    dgs.dispersion_integrals.impulse_response (independently verified there
    against a direct Riemann-sum Fourier integral).

  Problem 2 -- Group delay is linear in frequency
    tau_g(f) = -1/(2*pi) * d/df[phi(f)],  phi(f) = pi*D*f^2
    Solved by direct differentiation. Checked numerically against
    np.gradient of the unwrapped phase of a sampled H(f) array. This is
    *why* a dispersive element maps frequency to time -- the time-stretch
    mechanism the whole repo is built around.

  Problem 3 -- All-pass / energy conservation
    |H(f)|^2 = 1 for all f  (dispersion is phase-only, no amplitude loss)
    Solved via H(f)*conjugate(H(f)) = exp(i*pi*D*f^2)*exp(-i*pi*D*f^2) = 1.
    This is *why* GS phase retrieval is well-posed here: propagation
    through H(f) loses no energy, so recovering the phase alone is enough
    to determine the field completely.
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict

from dgs.dispersion_integrals import impulse_response as _impulse_response_ref


# ── Problem 1: impulse response (Fresnel/Gaussian integral) ──────────────────

def problem1_statement() -> str:
    return ("Evaluate  h(t) = Integral[ exp(i*pi*D*f^2) * exp(2*pi*i*f*t), "
            "(f, -oo, oo) ]  for real D != 0.")


def problem1_solve_symbolic() -> sp.Expr:
    """Solve Problem 1 by completing the square:
        i*pi*D*f^2 + 2*pi*i*f*t = i*pi*D*(f + t/D)^2 - i*pi*t^2/D
    which turns the integral into exp(-i*pi*t^2/D) times a plain Fresnel/
    Gaussian integral over the shifted variable u=f+t/D.
    """
    Dpos = sp.Symbol("D", positive=True)
    u, t = sp.symbols("u t", real=True)
    fresnel = sp.integrate(sp.exp(sp.I * sp.pi * Dpos * u**2), (u, -sp.oo, sp.oo))
    h_t = sp.exp(-sp.I * sp.pi * t**2 / Dpos) * sp.simplify(fresnel)
    return sp.simplify(h_t)


def problem1_verify(D: float, t_vals=(0.0, 0.5, 1.3, -2.0)) -> Dict:
    """Cross-check this module's from-scratch derivation against
    dgs.dispersion_integrals.impulse_response (already independently
    verified there against a Riemann-sum Fourier integral), so Problem 1
    isn't just checked against itself."""
    if D == 0:
        raise ValueError("D=0: no dispersion -- h(t) is a delta function, "
                          "not the Gaussian/Fresnel kernel solved here")
    t_vals = np.asarray(t_vals, dtype=float)
    h_this_module = (1.0 / np.sqrt(abs(D))) * np.exp(1j * np.sign(D) * np.pi / 4) \
        * np.exp(-1j * np.pi * t_vals**2 / D)
    h_reference = _impulse_response_ref(D, t_vals)
    max_abs_diff = float(np.max(np.abs(h_this_module - h_reference)))
    return {"D": D, "t_vals": t_vals.tolist(), "max_abs_diff_vs_dispersion_integrals": max_abs_diff}


# ── Problem 2: group delay is linear in frequency (NEW) ───────────────────────

def problem2_statement() -> str:
    return ("Given phi(f) = pi*D*f^2 (the phase of H(f)=exp(i*pi*D*f^2)), "
            "find the group delay tau_g(f) = -1/(2*pi) * d(phi)/df, and show "
            "it is linear in f.")


def problem2_solve_symbolic() -> sp.Expr:
    """Solve Problem 2 by direct differentiation of phi(f)=pi*D*f^2."""
    D, f = sp.symbols("D f", real=True)
    phi = sp.pi * D * f**2
    tau_g = -sp.diff(phi, f) / (2 * sp.pi)
    return sp.simplify(tau_g)  # -D*f


def problem2_verify(D: float, F: float = 5.0, n: int = 4001) -> Dict:
    """Numeric check: unwrap the phase of a sampled H(f) array and compare
    its negative-scaled derivative against the closed form tau_g(f)=-D*f."""
    if F <= 0:
        raise ValueError(f"F={F}: half-bandwidth must be positive")
    if n < 3:
        raise ValueError(f"n={n}: need at least 3 samples to take a derivative")
    f = np.linspace(-F, F, n)
    df = f[1] - f[0]
    phi = np.unwrap(np.angle(np.exp(1j * np.pi * D * f**2)))
    tau_g_numeric = -np.gradient(phi, df) / (2 * np.pi)
    tau_g_analytic = -D * f
    # endpoints of np.gradient are one-sided and noisier; compare the interior
    max_abs_err = float(np.max(np.abs(tau_g_numeric[5:-5] - tau_g_analytic[5:-5])))
    return {"D": D, "F": F, "n": n, "max_abs_err": max_abs_err}


# ── Problem 3: all-pass / energy conservation (NEW) ───────────────────────────

def problem3_statement() -> str:
    return "Show that |H(f)|^2 = 1 for all real f, D -- dispersion is phase-only."


def problem3_solve_symbolic() -> sp.Expr:
    """Solve Problem 3: H(f)*conjugate(H(f)) simplifies to 1 for real D, f,
    since exp(i*theta)*exp(-i*theta)=1 for any real theta."""
    D, f = sp.symbols("D f", real=True)
    H = sp.exp(sp.I * sp.pi * D * f**2)
    mag_sq = sp.simplify(H * sp.conjugate(H))
    return mag_sq  # -> 1


def problem3_verify(D: float, F: float = 50.0, n: int = 2001) -> Dict:
    """Numeric check: |H(f)| must equal 1 to floating-point precision at
    every sampled frequency, for arbitrary D (including D=0)."""
    f = np.linspace(-F, F, n)
    H = np.exp(1j * np.pi * D * f**2)
    max_abs_dev_from_1 = float(np.max(np.abs(np.abs(H) - 1.0)))
    return {"D": D, "F": F, "n": n, "max_abs_dev_from_1": max_abs_dev_from_1}


# ── CLI ───────────────────────────────────────────────────────────────────────
def print_problem_set() -> None:
    print("=" * 72)
    print("  DISPERSION CALCULUS PROBLEM SET  --  H(f) = exp(i*pi*D*f^2)")
    print("=" * 72)

    print("\nProblem 1:", problem1_statement())
    print("Solution:  h(t) =", problem1_solve_symbolic())
    for D in [5.0, -5.0, 12.3]:
        v = problem1_verify(D)
        print(f"  D={D:6.2f}  max_abs_diff vs dgs.dispersion_integrals = "
              f"{v['max_abs_diff_vs_dispersion_integrals']:.2e}")

    print("\nProblem 2:", problem2_statement())
    print("Solution:  tau_g(f) =", problem2_solve_symbolic())
    for D in [5.0, -5.0, 20.0]:
        v = problem2_verify(D)
        print(f"  D={D:6.2f}  max_abs_err (numeric gradient vs -D*f) = {v['max_abs_err']:.2e}")

    print("\nProblem 3:", problem3_statement())
    print("Solution:  |H(f)|^2 =", problem3_solve_symbolic())
    for D in [0.0, 5.0, -600.0]:
        v = problem3_verify(D)
        print(f"  D={D:8.1f}  max |H(f)|-1 deviation = {v['max_abs_dev_from_1']:.2e}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    print_problem_set()
