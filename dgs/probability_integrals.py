"""Gaussian probability integrals I_n = integral_0^inf x^n * exp(-lambda*x^2) dx,
n=0..5 (Appendix B1 of the standard modern-physics reference table), plus the
even/odd extension to the full real line. This IS the same integral family
behind every Gaussian-pulse energy/normalization calculation in this repo
(dgs.jalali_lab_calculus_problems' completing-the-square section, the
epsilon-delta convergence argument for a QM wavefunction's normalization) --
here as a clean, closed-form, CAS-verified reference table rather than
re-derived each time it's needed.

NumPy only at call time; SymPy used once, at import, to VERIFY every closed
form against direct symbolic integration (not just transcribed from the table).
"""

import numpy as np
import sympy as sp


def _verify_table():
    """Symbolically integrate x^n*exp(-lambda*x^2) from 0 to inf for n=0..5
    and check each against the closed forms below -- runs once at import
    time so a typo in the table would fail loudly, not silently."""
    x, lam = sp.symbols("x lambda", positive=True)
    closed_forms = {
        0: sp.Rational(1, 2) * sp.sqrt(sp.pi) * lam**sp.Rational(-1, 2),
        1: sp.Rational(1, 2) * lam**-1,
        2: sp.Rational(1, 4) * sp.sqrt(sp.pi) * lam**sp.Rational(-3, 2),
        3: sp.Rational(1, 2) * lam**-2,
        4: sp.Rational(3, 8) * sp.sqrt(sp.pi) * lam**sp.Rational(-5, 2),
        5: lam**-3,
    }
    for n, claimed in closed_forms.items():
        integrated = sp.integrate(x**n * sp.exp(-lam * x**2), (x, 0, sp.oo))
        if sp.simplify(integrated - claimed) != 0:
            raise AssertionError(f"I_{n} closed form does not match direct integration: "
                                  f"{integrated} vs {claimed}")
    return closed_forms


_CLOSED_FORMS = _verify_table()   # raises at import time if the table is wrong


def I_n(n, lam):
    """I_n(lambda) = integral_0^inf x^n * exp(-lambda*x^2) dx, n=0..5,
    verified symbolically against direct integration at module import."""
    if n not in _CLOSED_FORMS:
        raise ValueError(f"n must be in 0..5 (table range), got {n}")
    if np.any(np.asarray(lam) <= 0):
        raise ValueError("lambda must be positive")
    lam = np.asarray(lam, dtype=float)
    expr = _CLOSED_FORMS[n]
    f = sp.lambdify(sp.Symbol("lambda", positive=True), expr, "numpy")
    return f(lam)


def full_line_integral(n, lam):
    """integral_{-inf}^{+inf} x^n * exp(-lambda*x^2) dx: 2*I_n(lambda) for
    even n (the integrand is even), exactly 0 for odd n (the integrand is
    odd, symmetric cancellation -- no numerical approximation involved)."""
    if n % 2 == 1:
        return np.zeros_like(np.asarray(lam, dtype=float)) if np.ndim(lam) else 0.0
    return 2.0 * I_n(n, lam)


def gaussian_normalization_constant(lam):
    """The n=0 case is exactly the normalization integral for a Gaussian
    wavefunction psi(x)=exp(-lambda*x^2): integral|psi|^2 dx form falls out
    of I_0, e.g. normalizing psi=A*exp(-lambda*x^2) requires
    A^2 * full_line_integral(0, 2*lambda) = 1."""
    return 1.0 / np.sqrt(full_line_integral(0, 2 * lam))


if __name__ == "__main__":
    lam_val = 2.0
    print(f"Gaussian integral table at lambda={lam_val}:")
    for n in range(6):
        print(f"  I_{n} = {I_n(n, lam_val):.6f}   "
              f"full-line = {full_line_integral(n, lam_val):.6f}")

    A = gaussian_normalization_constant(lam_val)
    print(f"\nnormalization constant A for psi=A*exp(-{lam_val}*x^2): A = {A:.6f}")
    check = A**2 * full_line_integral(0, 2*lam_val)
    print(f"check: A^2 * full_line_integral(0, 2*lambda) = {check:.6f} (expect 1.0)")
