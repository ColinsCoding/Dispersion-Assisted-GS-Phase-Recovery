"""Continued fractions: the best way to approximate a real number by integers.

Any real number is a nest of integer reciprocals -- its continued fraction:
        x = a0 + 1/(a1 + 1/(a2 + 1/(a3 + ...))),   written [a0; a1, a2, a3, ...],
found by peeling off the integer part, inverting the remainder, and repeating. The
partial fractions you get by truncating -- the CONVERGENTS p_k/q_k -- are the BEST
rational approximations to x in a precise sense: no fraction with a smaller denominator
gets closer. That is why pi ~= 22/7 (error 0.04%) and pi ~= 355/113 (error 8e-8, the
approximation good to 6 decimals with a 3-digit denominator) are famous, and why gear
ratios and calendars are built from convergents.

The convergents follow a simple recurrence from the coefficients:
        p_k = a_k p_{k-1} + p_{k-2},   q_k = a_k q_{k-1} + q_{k-2},
and each convergent is closer than 1/q^2:  |x - p_k/q_k| < 1/q_k^2. Some numbers have
beautiful patterns -- sqrt(2) = [1; 2,2,2,...], the golden ratio = [1; 1,1,1,...] (its
convergents are ratios of consecutive Fibonacci numbers, which is why it is the "most
irrational" number, hardest to approximate).

Verified against pi's convergents (22/7, 355/113), sqrt(2), the golden ratio =
Fibonacci ratios, the 1/q^2 error bound, and Python's own Fraction.limit_denominator.
Pure stdlib (math, fractions); py-3.13.
"""

import math
from fractions import Fraction


def continued_fraction(x, n_terms=15, tol=1e-12):
    """The continued-fraction coefficients [a0, a1, ...] of x, up to n_terms (or
    until the remainder vanishes for a rational x). Peel the integer part, invert
    the fractional part, repeat."""
    if n_terms < 1:
        raise ValueError("n_terms must be >= 1")
    coeffs = []
    for _ in range(n_terms):
        a = math.floor(x)
        coeffs.append(a)
        frac = x - a
        if frac < tol:
            break
        x = 1.0 / frac
    return coeffs


def convergents(coeffs):
    """The convergents p_k/q_k of a continued fraction, as a list of (p, q) using
    p_k = a_k p_{k-1} + p_{k-2}, q_k = a_k q_{k-1} + q_{k-2}. Each is the best
    rational approximation for its denominator size."""
    if not coeffs:
        raise ValueError("need at least one coefficient")
    p_prev, p = 1, coeffs[0]
    q_prev, q = 0, 1
    out = [(p, q)]
    for a in coeffs[1:]:
        p, p_prev = a * p + p_prev, p
        q, q_prev = a * q + q_prev, q
        out.append((p, q))
    return out


def evaluate(coeffs):
    """Evaluate a finite continued fraction back to an exact Fraction (folds the
    nested reciprocals from the inside out)."""
    if not coeffs:
        raise ValueError("need at least one coefficient")
    value = Fraction(coeffs[-1])
    for a in reversed(coeffs[:-1]):
        value = a + 1 / value
    return value


def best_rational(x, max_denominator):
    """The best rational approximation p/q to x with q <= max_denominator: the
    convergent (or bounding semiconvergent) with the largest allowed denominator.
    Delegates to Fraction.limit_denominator, which is exactly this algorithm."""
    if max_denominator < 1:
        raise ValueError("max_denominator must be >= 1")
    fr = Fraction(x).limit_denominator(max_denominator)
    return fr.numerator, fr.denominator


def approximation_error(x, p, q):
    """|x - p/q|, the absolute error of a rational approximation."""
    if q == 0:
        raise ValueError("q must be nonzero")
    return abs(x - p / q)


if __name__ == "__main__":
    print("pi = [a0; a1, ...] =", continued_fraction(math.pi, 6))
    print("  convergents:")
    for p, q in convergents(continued_fraction(math.pi, 6))[:5]:
        print(f"    {p}/{q} = {p/q:.9f}  (error {approximation_error(math.pi, p, q):.2e}, "
              f"bound 1/q^2 = {1/q**2:.2e})")

    print("\nsqrt(2) =", continued_fraction(math.sqrt(2), 8),
          "-> convergents", convergents(continued_fraction(math.sqrt(2), 6)))
    phi = (1 + math.sqrt(5)) / 2
    print("golden ratio =", continued_fraction(phi, 8),
          "-> convergents (Fibonacci ratios)", convergents(continued_fraction(phi, 8))[:6])

    print("\nbest rational for pi with denominator <= 200:", best_rational(math.pi, 200))
    print("reconstruct [3;7,15,1] ->", evaluate([3, 7, 15, 1]), "=",
          float(evaluate([3, 7, 15, 1])))
