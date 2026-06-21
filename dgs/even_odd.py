"""Even and odd functions -- the symmetry that halves your work.

Every function splits UNIQUELY into an even part and an odd part:
    f(x) = f_e(x) + f_o(x),   f_e = [f(x)+f(-x)]/2,   f_o = [f(x)-f(-x)]/2,
with f_e(-x) = f_e(x) (mirror symmetric) and f_o(-x) = -f_o(x) (sign-flipping).

Why it matters:
  * INTEGRALS. Over a symmetric interval [-a, a] the ODD part integrates to ZERO,
    so int_{-a}^{a} f = 2 int_0^a f_e. Half the integral is free.
  * FOURIER. An even function has only cosine terms; an odd function only sines.
    That is exactly why the sine/cosine transforms exist.
  * IDENTITIES. e^x = cosh(x) + sinh(x) is just its even + odd parts.

Leans on dgs.numerical_methods for the integrals. NumPy. Education.
"""

import numpy as np
from dgs import numerical_methods as nm


def even_part(f, x):
    """The even component f_e(x) = [f(x) + f(-x)] / 2  (mirror-symmetric)."""
    x = np.asarray(x, float)
    return (f(x) + f(-x)) / 2


def odd_part(f, x):
    """The odd component f_o(x) = [f(x) - f(-x)] / 2  (flips sign under x -> -x)."""
    x = np.asarray(x, float)
    return (f(x) - f(-x)) / 2


def decompose(f, x):
    """Return (even_part, odd_part) sampled on x. Their sum is f(x) exactly."""
    return even_part(f, x), odd_part(f, x)


def is_even(f, x, tol=1e-9):
    """True if f(-x) == f(x) across the samples x (e.g. cos, x**2, |x|)."""
    x = np.asarray(x, float)
    return bool(np.max(np.abs(f(x) - f(-x))) < tol)


def is_odd(f, x, tol=1e-9):
    """True if f(-x) == -f(x) across the samples x (e.g. sin, x**3, x)."""
    x = np.asarray(x, float)
    return bool(np.max(np.abs(f(-x) + f(x))) < tol)


def symmetric_integral(f, a, n=4001):
    """int_{-a}^{a} f(x) dx. The odd part contributes nothing, so this equals the
    even part's integral doubled -- computed directly here for cross-checking."""
    x = np.linspace(-a, a, n)
    return nm.trapezoid(f(x), x)


def even_integral_shortcut(f, a, n=4001):
    """The free half: int_{-a}^{a} f = 2 int_0^a f_e(x) dx (odd part drops out)."""
    xh = np.linspace(0, a, n)
    return 2 * nm.trapezoid(even_part(f, xh), xh)


if __name__ == "__main__":
    x = np.linspace(-3, 3, 2001)
    # cos is even, sin is odd
    print("cos even?", is_even(np.cos, x), "  sin odd?", is_odd(np.sin, x))
    # e^x = cosh + sinh (its even + odd parts)
    e_even, e_odd = decompose(np.exp, x)
    print("e^x even part == cosh?", np.allclose(e_even, np.cosh(x)),
          "  odd part == sinh?", np.allclose(e_odd, np.sinh(x)))
    # symmetric integral of an odd function is zero; of e^x uses only the even part
    f = lambda t: t**3 + t           # purely odd
    print(f"int_-2^2 (x^3+x) dx = {symmetric_integral(f, 2.0):.2e}  (odd -> 0)")
    full = symmetric_integral(np.exp, 2.0)
    half = even_integral_shortcut(np.exp, 2.0)
    print(f"int_-2^2 e^x: direct {full:.4f}  vs  2*even-part {half:.4f}  "
          f"(exact 2 sinh 2 = {2*np.sinh(2):.4f})")
