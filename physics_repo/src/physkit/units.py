"""Object-oriented dimensional analysis over the seven SI base quantities.

A ``Dimension`` is a vector of integer (or rational) exponents over the base quantities
length (L), mass (M), time (T), electric current (I), temperature (K), amount (N), luminous
intensity (J). A ``Quantity`` pairs a numeric magnitude with a ``Dimension``. Multiplication adds
exponents, powers scale them, and addition requires equal dimensions -- so a dimensionally invalid
expression raises rather than returning a silently wrong number.

This is the checking layer used throughout the notebooks: before trusting a formula numerically, we
confirm both sides carry the same dimension.
"""

from __future__ import annotations
from fractions import Fraction

_BASE = ("L", "M", "T", "I", "K", "N", "J")


class Dimension:
    """A product of powers of the SI base quantities."""

    __slots__ = ("exp",)

    def __init__(self, **kwargs):
        self.exp = {b: Fraction(kwargs.get(b, 0)) for b in _BASE}

    # ---- algebra ----
    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(**{b: self.exp[b] + other.exp[b] for b in _BASE})

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(**{b: self.exp[b] - other.exp[b] for b in _BASE})

    def __pow__(self, p) -> "Dimension":
        p = Fraction(p)
        return Dimension(**{b: self.exp[b] * p for b in _BASE})

    def __eq__(self, other) -> bool:
        return isinstance(other, Dimension) and self.exp == other.exp

    def __hash__(self):
        return hash(tuple(self.exp[b] for b in _BASE))

    @property
    def is_dimensionless(self) -> bool:
        return all(v == 0 for v in self.exp.values())

    def __repr__(self) -> str:
        parts = [f"{b}^{self.exp[b]}" for b in _BASE if self.exp[b] != 0]
        return "dimensionless" if not parts else " ".join(parts)


# Base dimensions
DIMENSIONLESS = Dimension()
LENGTH = Dimension(L=1)
MASS = Dimension(M=1)
TIME = Dimension(T=1)
CURRENT = Dimension(I=1)
TEMPERATURE = Dimension(K=1)
AMOUNT = Dimension(N=1)
LUMINOUS = Dimension(J=1)

# Derived dimensions (built from the base ones, so definitions are auditable)
VELOCITY = LENGTH / TIME
ACCELERATION = VELOCITY / TIME
FORCE = MASS * ACCELERATION
ENERGY = FORCE * LENGTH
POWER = ENERGY / TIME
CHARGE = CURRENT * TIME
VOLTAGE = ENERGY / CHARGE
FREQUENCY = DIMENSIONLESS / TIME
WAVENUMBER = DIMENSIONLESS / LENGTH
ACTION = ENERGY * TIME
CAPACITANCE = CHARGE / VOLTAGE
E_FIELD = VOLTAGE / LENGTH


class Quantity:
    """A magnitude with a physical dimension (value assumed in coherent SI)."""

    __slots__ = ("value", "dim")

    def __init__(self, value: float, dim: Dimension = DIMENSIONLESS):
        self.value = float(value)
        self.dim = dim

    def __mul__(self, other):
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, self.dim * other.dim)
        return Quantity(self.value * other, self.dim)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, self.dim / other.dim)
        return Quantity(self.value / other, self.dim)

    def __pow__(self, p):
        return Quantity(self.value ** float(p), self.dim ** p)

    def __add__(self, other):
        if not isinstance(other, Quantity) or self.dim != other.dim:
            raise ValueError(f"cannot add incompatible dimensions {self.dim} and "
                             f"{getattr(other, 'dim', 'scalar')}")
        return Quantity(self.value + other.value, self.dim)

    def __sub__(self, other):
        if not isinstance(other, Quantity) or self.dim != other.dim:
            raise ValueError(f"cannot subtract incompatible dimensions {self.dim} and "
                             f"{getattr(other, 'dim', 'scalar')}")
        return Quantity(self.value - other.value, self.dim)

    def to_dimension(self, dim: Dimension) -> "Quantity":
        """Assert this quantity has the expected dimension; return it unchanged."""
        if self.dim != dim:
            raise ValueError(f"expected dimension {dim}, found {self.dim}")
        return self

    def __repr__(self):
        return f"Quantity({self.value:g}, {self.dim})"


def same_dimension(a: Quantity, b: Quantity) -> bool:
    """True if two quantities share a physical dimension (are addable / comparable)."""
    return a.dim == b.dim
