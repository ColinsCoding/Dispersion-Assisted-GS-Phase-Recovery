"""The COMPUTER definition of a derivative: forward-mode automatic
differentiation via dual numbers, contrasted with the finite-difference
definition already in dgs.numerical_methods.derivative.

Three definitions of f'(x), all for the same function, all cross-checked:
  * MATH definition:    f'(x) = lim_{h->0} [f(x+h)-f(x)]/h  (an idealized limit)
  * PHYSICS definition: the instantaneous rate of change of a measured
                        quantity (dgs.numerical_methods.velocity/acceleration
                        applied to sampled position data)
  * COMPUTER definition: NEITHER of the above, exactly, is what a computer
                        can do. A computer can't take h->0 (finite precision)
                        and doesn't have a formula for f' unless you supply one.
                        Two real options:
                          (a) finite differences -- approximate the limit with
                              a SPECIFIC small h, trading truncation error
                              (h too big) against floating-point roundoff error
                              (h too small) -- there's a genuine sweet spot.
                          (b) automatic differentiation -- carry BOTH a value
                              and its derivative through every arithmetic
                              operation (a dual number a+b*eps, eps^2=0), so
                              the chain rule is applied mechanically and
                              EXACTLY, with no h and no step-size error at all.

This module implements (b), the dual-number approach, since (a) already
exists as dgs.numerical_methods.derivative.
"""

import numpy as np

from dgs.numerical_methods import derivative as finite_diff_derivative


class Dual:
    """A dual number a + b*eps with eps^2=0: `a` carries the function VALUE,
    `b` carries its DERIVATIVE, propagated automatically by the chain rule
    through every operator overload below. No step size, no h, no limit --
    just algebra with a nilpotent unit."""

    __slots__ = ("val", "deriv")

    def __init__(self, val, deriv=0.0):
        self.val = float(val)
        self.deriv = float(deriv)

    def _other(self, other):
        return other if isinstance(other, Dual) else Dual(other, 0.0)

    def __add__(self, other):
        o = self._other(other)
        return Dual(self.val + o.val, self.deriv + o.deriv)

    __radd__ = __add__

    def __sub__(self, other):
        o = self._other(other)
        return Dual(self.val - o.val, self.deriv - o.deriv)

    def __rsub__(self, other):
        return self._other(other) - self

    def __mul__(self, other):
        # product rule: d(uv) = u'v + uv'
        o = self._other(other)
        return Dual(self.val * o.val, self.deriv * o.val + self.val * o.deriv)

    __rmul__ = __mul__

    def __truediv__(self, other):
        # quotient rule: d(u/v) = (u'v - uv')/v^2
        o = self._other(other)
        if o.val == 0:
            raise ZeroDivisionError("dual number division by zero value")
        return Dual(self.val / o.val,
                    (self.deriv * o.val - self.val * o.deriv) / (o.val ** 2))

    def __rtruediv__(self, other):
        return self._other(other) / self

    def __pow__(self, n):
        # power rule: d(u^n) = n*u^(n-1)*u'  (n a real constant, not a Dual)
        if isinstance(n, Dual):
            raise TypeError("Dual ** Dual (variable exponent) not supported")
        return Dual(self.val ** n, n * self.val ** (n - 1) * self.deriv)

    def __neg__(self):
        return Dual(-self.val, -self.deriv)

    def __repr__(self):
        return f"Dual(val={self.val}, deriv={self.deriv})"


def dsin(x):
    """sin, dual-aware: d(sin(u)) = cos(u)*u'."""
    if isinstance(x, Dual):
        return Dual(np.sin(x.val), np.cos(x.val) * x.deriv)
    return np.sin(x)


def dcos(x):
    """cos, dual-aware: d(cos(u)) = -sin(u)*u'."""
    if isinstance(x, Dual):
        return Dual(np.cos(x.val), -np.sin(x.val) * x.deriv)
    return np.cos(x)


def dexp(x):
    """exp, dual-aware: d(exp(u)) = exp(u)*u'."""
    if isinstance(x, Dual):
        return Dual(np.exp(x.val), np.exp(x.val) * x.deriv)
    return np.exp(x)


def autodiff_derivative(f, x):
    """The COMPUTER (autodiff) derivative of f at x: seed a Dual with
    derivative=1 (dx/dx=1), evaluate f, read off the propagated derivative.
    Exact to machine precision -- no step size involved at all."""
    result = f(Dual(x, 1.0))
    return result.val, result.deriv


def finite_difference_error_sweep(f, f_prime_exact, x, hs):
    """The COMPUTER (finite-difference) derivative's actual error curve vs
    step size h: truncation error shrinks as h shrinks (good), but
    floating-point roundoff error GROWS as h shrinks (bad, since
    f(x+h)-f(x-h) becomes a difference of nearly-equal floats). Returns the
    absolute error at each h, exposing the real U-shaped trade-off."""
    hs = np.asarray(hs, dtype=float)
    exact = f_prime_exact(x)
    errors = np.array([
        abs(finite_diff_derivative(f, x, h=h) - exact) for h in hs
    ])
    return errors


if __name__ == "__main__":
    x0 = 1.0

    # sin(x) via all three "computer" routes, checked against the exact cos(x)
    f = dsin
    exact = np.cos(x0)

    val_ad, deriv_ad = autodiff_derivative(f, x0)
    print(f"f(x)=sin(x) at x={x0}")
    print(f"  exact f'(x)=cos(x)        = {exact:.15f}")
    print(f"  autodiff (dual number)    = {deriv_ad:.15f}  "
          f"(error = {abs(deriv_ad-exact):.2e})")

    deriv_fd = finite_diff_derivative(np.sin, x0, h=1e-5)
    print(f"  finite difference, h=1e-5 = {deriv_fd:.15f}  "
          f"(error = {abs(deriv_fd-exact):.2e})")

    hs = np.logspace(-1, -14, 30)
    errors = finite_difference_error_sweep(np.sin, np.cos, x0, hs)
    best_idx = np.argmin(errors)
    print(f"\n  finite-difference sweet spot: h={hs[best_idx]:.2e}, "
          f"error={errors[best_idx]:.2e}")
    print(f"  too small (h={hs[-1]:.0e}): error={errors[-1]:.2e}  <- roundoff dominates")
    print(f"  too big   (h={hs[0]:.0e}): error={errors[0]:.2e}  <- truncation dominates")
    print("\n  autodiff has NO sweet spot to find -- it's exact by construction.")
