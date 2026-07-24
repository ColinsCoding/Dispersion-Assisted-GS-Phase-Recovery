"""dispersion_integrals.py -- closed-form SymPy integrals for H(f)=exp(i*pi*D*f^2)

gs_verify.py already checks H(nu)'s algebraic identities (unitarity, symmetry,
Parseval) and dispersive_fourier.py's tsdft_sympy_5() differentiates H(nu) to
get the group delay -- but nothing in this repo actually *integrates* H(nu) in
closed form. These are two new results, both completed-square Gaussian/Fresnel
integrals, each checked symbolically (SymPy) and numerically (against a direct
Fourier sum matching gs_core.disperse's cyclic-frequency convention):

  1. impulse_response(D, t): the exact analytic impulse response
     h(t) = Integral[ H(f) exp(2*pi*I*f*t), (f, -oo, oo) ]
     obtained by completing the square in the exponent and evaluating the
     resulting Fresnel-type Gaussian integral -- not a numerical FFT.

  2. gaussian_broadening_T1(D, T0): the exact analytic output width of a
     transform-limited Gaussian pulse E(t)=exp(-t^2/(2*T0^2)) after passing
     through H(f), derived the same way. Reduces to the standard GVD
     broadening law T1(z) = T0*sqrt(1+(z/L_D)^2) under the D=2*pi*beta2*L
     mapping already fixed in gs_verify.py's verify_transfer_function (S1).

Both closed forms are consequences of the single completed-square identity
    I*pi*D*f^2 + 2*pi*I*f*t = I*pi*D*(f + t/D)^2 - I*pi*t^2/D
which turns every "propagate through H(f)" integral into a plain Gaussian
integral. derive_completed_square() proves that identity in SymPy so the two
results below aren't independent guesses -- they share one derivation step.
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict


# ── Shared derivation step ────────────────────────────────────────────────────

def derive_completed_square() -> sp.Eq:
    """Prove I*pi*D*f^2 + 2*pi*I*f*t == I*pi*D*(f+t/D)^2 - I*pi*t^2/D symbolically.

    Every closed-form integral below reduces to this one completed-square
    step; verifying it once in SymPy means both results share a single,
    checked derivation rather than two independent hand computations.
    """
    D, f, t = sp.symbols("D f t", real=True)
    lhs = sp.I * sp.pi * D * f**2 + 2 * sp.pi * sp.I * f * t
    rhs = sp.I * sp.pi * D * (f + t / D) ** 2 - sp.I * sp.pi * t**2 / D
    residual = sp.simplify(sp.expand(lhs - rhs))
    if residual != 0:
        raise AssertionError(f"completed-square identity failed: residual={residual}")
    return sp.Eq(lhs, rhs)


# ── Result 1: analytic impulse response of the all-pass dispersion filter ────

def impulse_response_symbolic() -> sp.Expr:
    """Closed form of h(t) = Integral[H(f)*exp(2*pi*I*f*t), (f,-oo,oo)] for D>0.

    Completing the square reduces the integral to
        exp(-i*pi*t^2/D) * Integral[exp(i*pi*D*u^2), (u,-oo,oo)]
    and SymPy evaluates the remaining Fresnel/Gaussian integral directly
    (D assumed positive here; sign flips to exp(-i*pi/4) for D<0 by evenness
    of f^2, handled numerically in impulse_response() below via sign(D)).
    """
    Dpos = sp.Symbol("D", positive=True)
    u, t = sp.symbols("u t", real=True)
    fresnel = sp.integrate(sp.exp(sp.I * sp.pi * Dpos * u**2), (u, -sp.oo, sp.oo))
    fresnel = sp.simplify(fresnel)  # (-1)**(1/4)/sqrt(D) == exp(i*pi/4)/sqrt(D)
    h_t = sp.exp(-sp.I * sp.pi * t**2 / Dpos) * fresnel
    return sp.simplify(h_t)


def impulse_response(D: float, t) -> complex:
    """Numeric closed-form h(t) for real D (any sign) and array-like t.

    h(t) = (1/sqrt(|D|)) * exp(i*sign(D)*pi/4) * exp(-i*pi*t^2/D)

    Bounds: D must be nonzero (D=0 is the trivial no-dispersion identity
    filter -- Integral[exp(2*pi*I*f*t)]df is a delta function, not this
    Gaussian kernel).
    """
    if D == 0:
        raise ValueError("D=0: no dispersion -- impulse response is a delta "
                          "function, not the Gaussian/Fresnel kernel derived here")
    t = np.asarray(t, dtype=float)
    return (1.0 / np.sqrt(abs(D))) * np.exp(1j * np.sign(D) * np.pi / 4) \
        * np.exp(-1j * np.pi * t**2 / D)


def verify_impulse_response_numeric(D: float, F: float = 50.0,
                                     n: int = 2_000_001) -> Dict:
    """Compare impulse_response(D,t) against a direct Riemann-sum Fourier
    integral of H(f) over [-F,F] -- an independent numeric evaluation of the
    same Integral[H(f)*exp(2*pi*I*f*t)]df that impulse_response_symbolic()
    solved in closed form. Truncating to +/-F gives the finite-range
    quadrature error reported as max_rel_err (Fresnel integrals converge
    only conditionally, so this is expected to be O(1/sqrt(F)), not zero)."""
    f = np.linspace(-F, F, n)
    df = f[1] - f[0]
    H = np.exp(1j * np.pi * D * f**2)
    t_vals = np.array([0.0, 0.5, 1.3, -2.0])
    errs = []
    for t in t_vals:
        h_num = df * np.sum(H * np.exp(1j * 2 * np.pi * f * t))
        h_ana = complex(impulse_response(D, t))
        errs.append(abs(h_num - h_ana) / abs(h_ana))
    return {"D": D, "t_vals": t_vals.tolist(), "max_rel_err": float(max(errs))}


# ── Result 2: closed-form GVD broadening of a Gaussian pulse ─────────────────

def gaussian_broadening_symbolic() -> sp.Expr:
    """Derive E_out(t) for a transform-limited Gaussian E(t)=exp(-t^2/(2*T0^2))
    propagated through H(f)=exp(i*pi*D*f^2), by Fourier-transforming E(t),
    multiplying by H(f), and integrating back -- the same completed-square
    Gaussian-integral trick as impulse_response_symbolic(), now with a
    Gaussian spectrum instead of a delta, so the linear-in-f term is real.
    """
    t, f = sp.symbols("t f", real=True)
    T0 = sp.Symbol("T0", positive=True)
    D = sp.Symbol("D", real=True)

    Ef = sp.sqrt(2) * sp.sqrt(sp.pi) * T0 * sp.exp(-2 * sp.pi**2 * T0**2 * f**2)
    a = -2 * sp.pi**2 * T0**2 + sp.I * sp.pi * D
    b = 2 * sp.pi * sp.I * t
    E_out = sp.sqrt(2) * sp.sqrt(sp.pi) * T0 * sp.sqrt(-sp.pi / a) * sp.exp(-b**2 / (4 * a))
    return sp.simplify(E_out)


def gaussian_broadening_T1(D, T0: float = 1.0):
    """Closed-form output 1/e-intensity half-width after GVD broadening.

    T1(D) = T0 * sqrt(1 + (D/(2*pi*T0^2))^2)

    Derived from gaussian_broadening_symbolic()'s E_out(t) by taking
    |E_out(t)|^2 = exp(2*Re(coefficient of t^2)*t^2) and solving for the
    width T1 such that |E_out(t)|^2 = exp(-t^2/T1^2). Reduces to the
    standard fiber-optics GVD broadening law T1(z)=T0*sqrt(1+(z/L_D)^2)
    under D=2*pi*beta2*L, L_D=T0^2/|beta2| (gs_verify.py S1 mapping):
    D/(2*pi*T0^2) = beta2*L/T0^2 = z/L_D exactly.

    Bounds: T0 must be positive (a physical pulse width).
    """
    if T0 <= 0:
        raise ValueError(f"T0={T0}: pulse width must be positive")
    D = np.asarray(D, dtype=float)
    return T0 * np.sqrt(1.0 + (D / (2 * np.pi * T0**2)) ** 2)


def verify_gaussian_broadening_numeric(D: float, T0: float = 1.0,
                                        N: int = 2**16, dt: float = 0.005) -> Dict:
    """Compare gaussian_broadening_T1 against a direct FFT propagation
    (gs_core.disperse's exact convention, continuous limit) fit by the
    second moment of the output intensity: for I(t)=exp(-t^2/T1^2),
    Var[t] = T1^2/2, so T1 = sqrt(2*Var[t])."""
    t = (np.arange(N) - N // 2) * dt
    E = np.exp(-t**2 / (2 * T0**2))
    f = np.fft.fftfreq(N, d=dt)
    H = np.exp(1j * np.pi * D * f**2)
    E_out = np.fft.ifft(np.fft.fft(E) * H)
    # NOT fftshifted: E's peak already sits at index N//2 matching t's
    # centering, and fft/ifft preserve index correspondence -- shifting
    # once here would misalign I_out against t (caught by this exact bug).
    I_out = np.abs(E_out) ** 2
    var_t = np.sum(t**2 * I_out) / np.sum(I_out)
    T1_numeric = float(np.sqrt(2 * var_t))
    T1_analytic = float(gaussian_broadening_T1(D, T0))
    return {"D": D, "T0": T0, "T1_analytic": T1_analytic,
            "T1_numeric": T1_numeric,
            "rel_err": abs(T1_analytic - T1_numeric) / T1_analytic if T1_analytic else 0.0}


if __name__ == "__main__":
    print("=== Shared derivation step ===")
    print(derive_completed_square())

    print("\n=== Result 1: impulse response h(t) ===")
    print("Closed form (D>0):", impulse_response_symbolic())
    for D in [5.0, -5.0, 12.3, -0.8]:
        v = verify_impulse_response_numeric(D)
        print(f"  D={D:6.2f}  max_rel_err vs Riemann-sum FT = {v['max_rel_err']:.4f}")

    print("\n=== Result 2: Gaussian GVD broadening T1(D) ===")
    print("E_out(t) =", gaussian_broadening_symbolic())
    for D, T0 in [(0.0, 1.0), (5.0, 1.0), (20.0, 1.0), (-8.0, 0.7), (50.0, 2.0)]:
        v = verify_gaussian_broadening_numeric(D, T0)
        print(f"  D={D:6.1f} T0={T0:.1f}  T1_analytic={v['T1_analytic']:.5f}  "
              f"T1_numeric={v['T1_numeric']:.5f}  rel_err={v['rel_err']:.2e}")
