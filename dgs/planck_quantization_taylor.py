"""Integer multiples -> Planck's law -> Taylor series, plus a full adder.
(Knight Ch. 37.1 / 37.8; Round 2 question: red-hot iron at 720 nm via Wien's law.)

1. INTEGER MULTIPLES (Planck, 1900): a mode of frequency f may only hold energy
   E_n = n h f, n = 0, 1, 2, ... With Boltzmann weights e^{-n x}, x = hf/kT,
   two geometric series give the mean energy
       <E> = hf / (e^{x} - 1) = kT * x / (e^x - 1).
2. TAYLOR SERIES centred at x = 0 (low frequency):  x/(e^x - 1) = 1 - x/2 + x^2/12 - ...
   so <E> -> kT: classical equipartition and the Rayleigh-Jeans law, which blows up
   at high f (the ultraviolet catastrophe). Gross behaviour at large x: x e^{-x}
   (Wien's approximation) -- quantization kills the high-frequency modes.
3. WIEN'S LAW: T = b / lambda_peak (b from the Lambert-W root, dgs.wien_wave_502).
4. FULL ADDER in SymPy logic: s = a XOR b XOR c, c_out = majority(a, b, c),
   and integer arithmetic a + b + c = 2 c_out + s for all 8 inputs.

References (IEEE)
[1] M. Planck, "Ueber das Gesetz der Energieverteilung im Normalspectrum,"
    Ann. Phys., vol. 309, no. 3, pp. 553-563, 1901.
[2] R. D. Knight, Physics for Scientists and Engineers: A Strategic Approach with
    Modern Physics, 5th ed. Hoboken, NJ, USA: Pearson, 2022, ch. 37.
"""
import itertools

import numpy as np
import sympy as sp
from sympy.logic.boolalg import Xor, And, Or

from dgs.wien_wave_502 import wien_constant

x = sp.Symbol("x", positive=True)
n = sp.Symbol("n", integer=True, nonnegative=True)


# ── 1. mean energy from integer multiples ────────────────────────────

def _converged(expr):
    """Take the |r| < 1 branch of a Piecewise returned by summation."""
    return expr.args[0][0] if isinstance(expr, sp.Piecewise) else expr


def mean_energy_symbolic():
    """<E>/kT = (sum n x r^n) / (sum r^n), r = e^{-x} in (0, 1): two geometric series.

    SymPy cannot sum e^{-n x} directly (it does not know e^{-x} < 1), so sum in r,
    keep the convergent branch, then substitute r = e^{-x}."""
    r = sp.Symbol("r", positive=True)
    Z = _converged(sp.summation(r ** n, (n, 0, sp.oo)))
    S1 = _converged(sp.summation(n * r ** n, (n, 0, sp.oo)))
    ratio = sp.simplify((x * S1 / Z).subs(r, sp.exp(-x)))
    return sp.simplify(Z), sp.simplify(S1), ratio


def mean_energy_numeric(xval, nmax=2000):
    k = np.arange(nmax)
    w = np.exp(-k * xval)
    return float(np.sum(k * xval * w) / np.sum(w))


# ── 2. Taylor series and gross behaviour ─────────────────────────────

def taylor_low_frequency(order=6):
    return sp.series(x / (sp.exp(x) - 1), x, 0, order)


def large_x_ratio_limit():
    """lim_{x->oo} [x/(e^x - 1)] / [x e^{-x}] = 1: Wien's approximation."""
    return sp.limit((x / (sp.exp(x) - 1)) / (x * sp.exp(-x)), x, sp.oo)


def taylor_error(xval, order):
    """|exact - truncated series| at x for a series of the given order."""
    s = taylor_low_frequency(order).removeO()
    exact = xval / np.expm1(xval)
    return abs(exact - float(s.subs(x, xval)))


# ── 3. Wien ──────────────────────────────────────────────────────────

def wien_temperature(lam_nm):
    if lam_nm <= 0:
        raise ValueError("wavelength must be positive")
    return wien_constant() / (lam_nm * 1e-9)


def nearest_option(value, options):
    return min(options, key=lambda o: abs(o - value))


# ── 4. full adder ────────────────────────────────────────────────────

def full_adder_symbolic():
    a, b, c = sp.symbols("a b c")
    s = Xor(a, b, c)
    cout = Or(And(a, b), And(c, Xor(a, b)))
    return s, cout, (a, b, c)


def full_adder_truth_table():
    """Rows (a, b, c, s, cout, a+b+c, 2*cout+s) for all 8 inputs."""
    s, cout, (a, b, c) = full_adder_symbolic()
    rows = []
    for va, vb, vc in itertools.product((0, 1), repeat=3):
        sub = {a: bool(va), b: bool(vb), c: bool(vc)}
        sv, cv = int(bool(s.subs(sub))), int(bool(cout.subs(sub)))
        rows.append((va, vb, vc, sv, cv, va + vb + vc, 2 * cv + sv))
    return rows


def ripple_add(p, q, bits=8):
    """Add two non-negative integers with a chain of full adders (bit by bit)."""
    if p < 0 or q < 0 or p >= 2 ** bits or q >= 2 ** bits:
        raise ValueError("operands must fit in the given number of bits")
    carry, out = 0, 0
    for i in range(bits):
        a, b = (p >> i) & 1, (q >> i) & 1
        s = a ^ b ^ carry
        carry = (a & b) | (carry & (a ^ b))
        out |= s << i
    return out | (carry << bits)
