"""
calc2_primer.py -- Calculus 2 primer built around what's actually useful for
a Computer Engineering degree: change-of-base logarithms (binary log for
algorithm complexity and bits of information), exponential growth/decay,
related rates via the Pythagorean theorem, and second derivatives
(concavity, and the circuits analog of position->velocity->acceleration:
charge -> current -> dI/dt).

Reuses dgs.numerical_methods.derivative/second_derivative for the numerical
calculus pieces rather than reimplementing finite differences.
"""
import numpy as np

from dgs.numerical_methods import derivative, second_derivative


# ── Exponentials & logarithms ─────────────────────────────────────────────

def change_of_base(x, base):
    """log_base(x) = ln(x) / ln(base) -- the change-of-base formula. Every
    standard math library ships ln (natural log) and often log10, but
    rarely an arbitrary-base log directly -- this is the formula that gets
    you any base from just one."""
    if x <= 0:
        raise ValueError(f"x={x} must be positive (log undefined otherwise)")
    if base <= 0 or base == 1:
        raise ValueError(f"base={base} must be positive and != 1")
    return np.log(x) / np.log(base)


def log2_via_change_of_base(x):
    """log2(x), built from change_of_base -- cross-checked against NumPy's
    own np.log2 in tests/test_calc2_primer.py."""
    return change_of_base(x, 2)


# ── Why this matters in Computer Engineering specifically ─────────────────

def bits_of_information(n_outcomes):
    """log2(n_outcomes) -- bits needed to distinguish n_outcomes equally-
    likely possibilities. bits_of_information(256) = 8: exactly what a
    byte encodes."""
    if n_outcomes <= 0:
        raise ValueError(f"n_outcomes={n_outcomes} must be positive")
    return change_of_base(n_outcomes, 2)


def binary_search_steps(n_items):
    """ceil(log2(n_items)) -- worst-case comparisons for binary search over
    n_items, the textbook O(log n) example. Halving n each step IS a
    base-2 log relationship: how many times can you halve n before you're
    down to 1?"""
    if n_items <= 0:
        raise ValueError(f"n_items={n_items} must be positive")
    return int(np.ceil(log2_via_change_of_base(n_items)))


# ── Exponential growth/decay ───────────────────────────────────────────────

def exponential(t, A0, k):
    """A(t) = A0 * exp(k*t) -- growth (k>0) or decay (k<0). The general
    solution to dA/dt = k*A: rate of change proportional to current value,
    the defining property of exponentials."""
    return A0 * np.exp(k * np.asarray(t, dtype=float))


def half_life_to_rate(half_life):
    """k such that A0*exp(k*t_half) = A0/2  ->  k = -ln(2)/t_half."""
    if half_life <= 0:
        raise ValueError(f"half_life={half_life} must be positive")
    return -np.log(2) / half_life


# ── Related rates: the classic Pythagorean (ladder) problem ───────────────

def ladder_related_rate(x, y, L, dx_dt):
    """Ladder of fixed length L leans against a wall: base at distance x
    from the wall, top at height y, x^2+y^2=L^2. Given how fast the base
    slides out (dx_dt), find how fast the top slides down (dy_dt).

    Differentiate x^2+y^2=L^2 w.r.t. t: 2x(dx/dt) + 2y(dy/dt) = 0
    -> dy/dt = -(x/y) * dx/dt.
    """
    if L <= 0:
        raise ValueError(f"L={L} must be positive")
    if not np.isclose(x ** 2 + y ** 2, L ** 2, rtol=1e-6):
        raise ValueError(f"(x,y)=({x},{y}) doesn't satisfy x^2+y^2=L^2 for L={L}")
    if y == 0:
        raise ValueError("y=0: ladder flat on the ground, dy/dt undefined (division by zero)")
    return -(x / y) * dx_dt


# ── Second derivatives: concavity, and the circuits analog ────────────────

def concavity(f, x, h=1e-4):
    """Sign of f''(x): concave up (+1), concave down (-1), or flat/
    inflection-like within numerical tolerance (0). Thin wrapper around
    dgs.numerical_methods.second_derivative."""
    f2 = second_derivative(f, x, h=h)
    tol = 1e-6
    if f2 > tol:
        return 1
    if f2 < -tol:
        return -1
    return 0


def current_and_its_rate(charge_fn, t, h_first=1e-5, h_second=1e-4):
    """Circuit analog of position->velocity->acceleration: given charge
    Q(t) [Coulombs], compute current I(t)=dQ/dt [Amps] and dI/dt [Amps/s]
    at time t -- the SAME second-derivative machinery concavity() uses
    above, applied to a circuits quantity instead of a position.

    Uses DIFFERENT default step sizes for the first- and second-derivative
    calls (matching dgs.numerical_methods.derivative/second_derivative's
    own defaults), not one shared h -- a central second difference divides
    by h^2, so a step size that's fine for a first derivative (e.g. 1e-6)
    is too small there: roundoff in f(x+h)-2f(x)+f(x-h) gets amplified by
    1/h^2 and dominates the result (see dgs.c_type_precision for the
    general machine-precision mechanism behind this)."""
    I = derivative(charge_fn, t, h=h_first)
    dI_dt = second_derivative(charge_fn, t, h=h_second)
    return I, dI_dt


if __name__ == "__main__":
    print("change-of-base logarithms in Computer Engineering:")
    print(f"  log2(8)   = {log2_via_change_of_base(8):.4f}  (should be 3)")
    print(f"  bits_of_information(256) = {bits_of_information(256):.4f}  (a byte: should be 8)")
    print(f"  binary_search_steps(1_000_000) = {binary_search_steps(1_000_000)}  (should be 20)")
    print()
    print("exponential decay:")
    k = half_life_to_rate(5730)   # carbon-14 half-life in years
    print(f"  carbon-14 half-life=5730 yr -> k={k:.6e} /yr")
    print(f"  A(5730)/A0 = {exponential(5730, 1.0, k):.4f}  (should be 0.5)")
    print()
    print("related rates -- ladder problem (3-4-5 triangle, x2 = 6-8-10):")
    dy_dt = ladder_related_rate(x=6, y=8, L=10, dx_dt=2)
    print(f"  base sliding out at 2 ft/s -> top sliding down at {dy_dt:.2f} ft/s")
    print()
    print("second derivatives -- concavity:")
    print(f"  x^2  at x=0: concavity={concavity(lambda x: x**2, 0.0)}  (concave up, +1)")
    print(f"  -x^2 at x=0: concavity={concavity(lambda x: -x**2, 0.0)}  (concave down, -1)")
    print()
    print("second derivatives -- circuits: Q(t)=t^2 (Coulombs) -> I(t), dI/dt")
    I, dI_dt = current_and_its_rate(lambda t: t ** 2, t=3.0)
    print(f"  at t=3s: I={I:.4f} A (should be 6), dI/dt={dI_dt:.4f} A/s (should be 2)")
