"""Numerical calculus -- the deterministic versions you design with.

Symbolic calculus (SymPy) tells you the exact rule; numerical calculus is what
actually runs in an instrument, a controller, or a solver, where you only have
*samples* of a signal, not a formula. This module is the core toolkit:

  * finite-difference derivatives  -- velocity, acceleration from position
    (the power rule, recovered numerically: d/dx x^n -> n x^{n-1})
  * numerical integration (quadrature) -- trapezoid & Simpson, the "integrals of
    approximation functions": fit a line/parabola to the samples and integrate that
  * Taylor approximation from sampled derivatives -- local polynomial models

Each carries its error order so you know how the result improves as you refine the
grid: central difference and trapezoid are O(h^2), Simpson is O(h^4). The closing
demo uses these to verify Griffiths' flux rule numerically: d/dt of Phi = B h x(t)
equals -B h v, which is the motional emf. NumPy only. Education.
"""

import numpy as np


# ── differentiation: rates of change from samples ───────────────────
def derivative(f, x, h=1e-5):
    """f'(x) by the CENTRAL difference (f(x+h)-f(x-h))/2h -- error O(h^2).
    Central beats forward because the two one-sided errors cancel to leading order."""
    return (f(x + h) - f(x - h)) / (2 * h)


def second_derivative(f, x, h=1e-4):
    """f''(x) by the central second difference (f(x+h)-2f(x)+f(x-h))/h^2, O(h^2)."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


def gradient(y, t):
    """Numerical derivative of SAMPLED data y(t) -- central in the interior, one-sided
    at the ends (np.gradient). This is how you get velocity from a position track."""
    return np.gradient(np.asarray(y, float), np.asarray(t, float))


# kinematics: the position -> velocity -> acceleration chain (each is a derivative)
def velocity(position, t):
    """v(t) = dx/dt from a sampled position track."""
    return gradient(position, t)


def acceleration(position, t):
    """a(t) = d^2x/dt^2 -- the derivative applied twice."""
    return gradient(gradient(position, t), t)


# ── integration: accumulated totals (quadrature) ────────────────────
def trapezoid(y, x):
    """Integral of sampled y over x by the trapezoid rule: approximate the curve by
    straight segments between samples and sum their areas. Error O(h^2)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def simpson(y, x):
    """Integral by Simpson's rule: fit a PARABOLA across each pair of intervals and
    integrate that -- error O(h^4), far better than trapezoid for smooth y. Requires
    an even number of intervals (odd number of equally spaced points)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    n = len(x) - 1
    if n % 2 != 0:
        raise ValueError("Simpson needs an even number of intervals (odd # of points)")
    h = (x[-1] - x[0]) / n
    s = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2])
    return float(h / 3 * s)


def cumulative_integral(y, x):
    """Running integral F(x) = integral_{x0}^{x} y dt (trapezoid). The numerical
    Fundamental Theorem: differentiate this and you get y back."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    F = np.zeros_like(y)
    F[1:] = np.cumsum((y[1:] + y[:-1]) / 2 * np.diff(x))
    return F


# ── Taylor approximation from numerical derivatives ─────────────────
def taylor_coefficients(f, a, order, h=1e-3):
    """Taylor coefficients f^{(k)}(a)/k! at point a, derivatives taken numerically.
    These are the weights of the local polynomial model of f near a."""
    from math import factorial
    coeffs = [f(a)]
    for k in range(1, order + 1):
        # k-th derivative via the central finite-difference stencil
        deriv = sum((-1) ** i * _binom(k, i) * f(a + (k / 2 - i) * h)
                    for i in range(k + 1)) / h ** k
        coeffs.append(deriv / factorial(k))
    return coeffs


def taylor_approx(f, a, order, x, h=1e-3):
    """Evaluate the order-`order` Taylor polynomial of f about a at point(s) x."""
    coeffs = taylor_coefficients(f, a, order, h)
    x = np.asarray(x, float)
    return sum(c * (x - a) ** k for k, c in enumerate(coeffs))


def _binom(n, k):
    from math import comb
    return comb(n, k)


# ── the repo/Griffiths connection: verify the flux rule numerically ──
def motional_flux_rate(B, h, x_of_t, t):
    """Griffiths Fig 7.10/7.13: a rectangular loop (height h) in field B with its
    edge at x(t). The flux is Phi(t) = B h x(t); this returns dPhi/dt computed
    NUMERICALLY from samples. For x(t)=x0 - v t it equals -B h v, and the emf is
    eps = -dPhi/dt = +B h v -- the motional emf, recovered with finite differences."""
    Phi = B * h * np.asarray(x_of_t, float)
    return gradient(Phi, t)


if __name__ == "__main__":
    # 1. power rule, numerically: d/dx x^3 = 3x^2  at x=2 -> 12
    print("power rule  d/dx x^3 at x=2 :", round(derivative(lambda x: x**3, 2.0), 6), "(exact 12)")
    # 2. integration: integral_0^pi sin = 2, Simpson should nail it
    xs = np.linspace(0, np.pi, 101)
    print("integral_0^pi sin x: trapezoid =", round(trapezoid(np.sin(xs), xs), 6),
          " Simpson =", round(simpson(np.sin(xs), xs), 8), "(exact 2)")
    # 3. flux rule: x(t) = 10 - 3t, B=0.5, h=2  ->  dPhi/dt = -B h v = -3
    t = np.linspace(0, 1, 200)
    dphi = motional_flux_rate(0.5, 2.0, 10 - 3 * t, t)
    print("flux rule  dPhi/dt :", round(float(dphi.mean()), 4),
          " => emf = -dPhi/dt =", round(-float(dphi.mean()), 4), "(B h v = 0.5*2*3 = 3)")
