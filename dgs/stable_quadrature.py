"""Numerically stable quadrature and summation for photonics computations.

Optical calculations lean on integrals that are easy to state and hard to evaluate: the Fresnel
diffraction integrals C(x), S(x) have a fast-oscillating integrand cos/sin(pi t^2 / 2) that defeats
fixed-step rules at large x, and chained attenuation / partition sums span many orders of magnitude and
lose precision to overflow and catastrophic cancellation.

This module collects the numerically stable primitives:

- ``gauss_legendre``      -- Gauss-Legendre quadrature, exact for polynomials up to degree 2n-1.
- ``adaptive_simpson``    -- adaptive Simpson quadrature that refines where the integrand oscillates.
- ``fresnel_C``/``fresnel_S`` -- Fresnel integrals via adaptive quadrature (validated against scipy).
- ``knife_edge_intensity``-- straight-edge (knife-edge) Fresnel diffraction pattern; the famous
  quarter-intensity at the geometric shadow boundary.
- ``logsumexp``           -- log-sum-exp without overflow (Beer-Lambert / Boltzmann chains).
- ``kahan_sum``           -- compensated summation, accurate where naive accumulation cancels.

All functions validate their inputs and raise ValueError with a clear message on bad arguments.
Pure NumPy; runs on py -3.13.
"""
from __future__ import annotations
import numpy as np

__all__ = [
    "gauss_legendre", "adaptive_simpson", "fresnel_C", "fresnel_S",
    "knife_edge_intensity", "logsumexp", "kahan_sum",
]


def gauss_legendre(f, a, b, n=64):
    """Integrate a vectorized ``f`` over ``[a, b]`` with ``n``-point Gauss-Legendre quadrature.

    Exact (to rounding) for polynomials of degree <= 2n-1. ``f`` must accept a NumPy array of nodes.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    a, b = float(a), float(b)
    if b == a:
        return 0.0
    x, w = np.polynomial.legendre.leggauss(n)          # nodes/weights on [-1, 1]
    t = 0.5 * (b - a) * x + 0.5 * (b + a)               # affine map to [a, b]
    return float(0.5 * (b - a) * np.sum(w * f(t)))


def adaptive_simpson(f, a, b, tol=1e-11, max_depth=60):
    """Adaptive Simpson quadrature of a scalar-callable ``f`` on ``[a, b]`` to absolute ``tol``.

    Recursively bisects, spending points only where the integrand curves or oscillates -- the robust
    choice for the Fresnel integrand cos/sin(pi t^2 / 2).
    """
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    a, b = float(a), float(b)
    if b == a:
        return 0.0

    def _simpson(fa, fm, fb, lo, hi):
        return (hi - lo) / 6.0 * (fa + 4.0 * fm + fb)

    def _recurse(lo, hi, fa, fm, fb, whole, tol, depth):
        mid = 0.5 * (lo + hi)
        lmid, rmid = 0.5 * (lo + mid), 0.5 * (mid + hi)
        flm, frm = float(f(lmid)), float(f(rmid))
        left = _simpson(fa, flm, fm, lo, mid)
        right = _simpson(fm, frm, fb, mid, hi)
        if depth <= 0 or abs(left + right - whole) <= 15.0 * tol:
            return left + right + (left + right - whole) / 15.0   # Richardson-extrapolated
        return (_recurse(lo, mid, fa, flm, fm, left, 0.5 * tol, depth - 1)
                + _recurse(mid, hi, fm, frm, fb, right, 0.5 * tol, depth - 1))

    fa, fb = float(f(a)), float(f(b))
    mid = 0.5 * (a + b)
    fm = float(f(mid))
    whole = _simpson(fa, fm, fb, a, b)
    return _recurse(a, b, fa, fm, fb, whole, tol, max_depth)


def fresnel_C(x, tol=1e-11):
    """Fresnel cosine integral C(x) = int_0^x cos(pi t^2 / 2) dt (odd in x)."""
    x = float(x)
    if x < 0:
        return -fresnel_C(-x, tol)
    return adaptive_simpson(lambda t: np.cos(np.pi * t * t / 2.0), 0.0, x, tol)


def fresnel_S(x, tol=1e-11):
    """Fresnel sine integral S(x) = int_0^x sin(pi t^2 / 2) dt (odd in x)."""
    x = float(x)
    if x < 0:
        return -fresnel_S(-x, tol)
    return adaptive_simpson(lambda t: np.sin(np.pi * t * t / 2.0), 0.0, x, tol)


def knife_edge_intensity(v, tol=1e-11):
    """Normalized intensity I/I0 of straight-edge (knife-edge) Fresnel diffraction.

    ``v`` is the dimensionless Fresnel parameter (v > 0 lit side, v < 0 geometric shadow):
        I/I0 = 1/2 [ (1/2 + C(v))^2 + (1/2 + S(v))^2 ].
    At the shadow boundary v = 0 this is exactly 1/4; deep in the light it approaches 1 with fringes.
    """
    C = fresnel_C(v, tol)
    S = fresnel_S(v, tol)
    return 0.5 * ((0.5 + C) ** 2 + (0.5 + S) ** 2)


def logsumexp(a, axis=None):
    """Stable log(sum(exp(a))): subtract the max so exp never overflows / underflows to a wrong answer."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        raise ValueError("logsumexp of an empty array is undefined")
    amax = np.max(a, axis=axis, keepdims=True)
    amax_safe = np.where(np.isfinite(amax), amax, 0.0)   # guard -inf columns
    out = np.log(np.sum(np.exp(a - amax_safe), axis=axis, keepdims=True)) + amax_safe
    return np.squeeze(out) if axis is not None else out.item()


def kahan_sum(xs):
    """Compensated (Kahan) summation: tracks the low-order bits lost in each add.

    Accurate where naive left-to-right summation cancels catastrophically (widely varying magnitudes).
    """
    total = 0.0
    comp = 0.0                                            # running compensation for lost low-order bits
    for x in xs:
        y = float(x) - comp
        t = total + y
        comp = (t - total) - y
        total = t
    return total
