"""Symbolic engine (SymPy).

Purpose:
    Wrap SymPy expressions in a small, single-responsibility container that can
    simplify, differentiate (gradient / Jacobian / Hessian), apply common-subexpression
    elimination, and lambdify -- the front end of the theory-to-code bridge.

Equations (Gaussian beam, Saleh & Teich Ch. 3):
    Rayleigh range     z_R = pi * w0**2 / lambda
    beam width         w(z) = w0 * sqrt(1 + (z / z_R)**2)
    Gouy phase         psi(z) = atan(z / z_R)

Assumptions:
    - Real-valued symbols; paraxial approximation for the beam relations.
Limitations:
    - `lambdify` backends limited to those SymPy supports (numpy, math).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import sympy as sp

__all__ = [
    "SymbolicExpression",
    "rayleigh_range",
    "gaussian_beam_width",
    "gouy_phase",
]


@dataclass(frozen=True)
class SymbolicExpression:
    """A named SymPy scalar expression together with its ordered free symbols.

    The ordered `symbols` tuple defines the call signature used by `lambdify` and by
    the C/Fortran/JS code generators, so downstream stages agree on argument order.
    """

    expr: sp.Expr
    symbols: tuple[sp.Symbol, ...]
    name: str = "f"

    def simplified(self) -> "SymbolicExpression":
        """Return a copy with a simplified expression."""
        return SymbolicExpression(sp.simplify(self.expr), self.symbols, self.name)

    def gradient(self) -> sp.Matrix:
        """Column vector of first partials d(expr)/d(symbol_i)."""
        return sp.Matrix([sp.diff(self.expr, s) for s in self.symbols])

    def hessian(self) -> sp.Matrix:
        """Symmetric matrix of second partials d^2(expr)/d(s_i)d(s_j)."""
        return sp.hessian(self.expr, self.symbols)

    def cse(self) -> tuple[list[tuple[sp.Symbol, sp.Expr]], sp.Expr]:
        """Common-subexpression elimination; returns (replacements, reduced_expr)."""
        replacements, reduced = sp.cse(self.expr, optimizations="basic")
        return replacements, reduced[0]

    def lambdify(self, backend: str = "numpy") -> Callable[..., object]:
        """Compile the expression to a fast numeric callable of `self.symbols`."""
        return sp.lambdify(self.symbols, self.expr, modules=backend)

    def evaluate(self, **values: float) -> float:
        """Substitute numeric values for symbols (by name) and return a float."""
        subs = {s: values[s.name] for s in self.symbols}
        return float(self.expr.subs(subs))


def jacobian(exprs: Sequence[sp.Expr], symbols: Sequence[sp.Symbol]) -> sp.Matrix:
    """Jacobian matrix of a vector of expressions with respect to `symbols`."""
    return sp.Matrix(list(exprs)).jacobian(sp.Matrix(list(symbols)))


def _beam_symbols() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    z = sp.Symbol("z", real=True)
    w0 = sp.Symbol("w0", positive=True)
    zR = sp.Symbol("zR", positive=True)
    return z, w0, zR


def rayleigh_range() -> SymbolicExpression:
    """z_R = pi * w0**2 / lambda."""
    w0 = sp.Symbol("w0", positive=True)
    lam = sp.Symbol("lam", positive=True)
    return SymbolicExpression(sp.pi * w0**2 / lam, (w0, lam), "rayleigh_range")


def gaussian_beam_width() -> SymbolicExpression:
    """w(z) = w0 * sqrt(1 + (z / z_R)**2)."""
    z, w0, zR = _beam_symbols()
    expr = w0 * sp.sqrt(1 + (z / zR) ** 2)
    return SymbolicExpression(expr, (z, w0, zR), "beam_width")


def gouy_phase() -> SymbolicExpression:
    """psi(z) = atan(z / z_R)."""
    z, _, zR = _beam_symbols()
    return SymbolicExpression(sp.atan(z / zR), (z, zR), "gouy_phase")
