"""EEE 117 resistive-circuit toolkit: every method of the first unit, in SymPy.

Ohm's law and power, series and parallel, voltage and current dividers, the Wheatstone bridge,
delta-to-wye, node-voltage (``Circuit`` from dgs.field_circuits), mesh-current (``Mesh`` here),
ideal op-amps, and a binary R-2R ladder DAC. Everything works with numbers or SymPy symbols;
the matching Lean proofs are in lean/EEE117Circuits.lean.
"""
from __future__ import annotations

import sympy as sp

from dgs.field_circuits import Circuit


def _resistances(Rs):
    if not Rs:
        raise ValueError("give at least one resistance")
    out = [sp.sympify(R) for R in Rs]
    for R in out:
        if R.is_number and not R > 0:
            raise ValueError(f"resistances must be positive, got {R}")
    return out


# ---- Ohm's law and power ----------------------------------------------------------------------
def ohm(V=None, I=None, R=None):
    """Solve V = I R for whichever one of V, I, R is None. Returns dict with all three and P."""
    if sum(v is None for v in (V, I, R)) != 1:
        raise ValueError("give exactly two of V, I, R")
    if R is not None:
        _resistances((R,))
    V, I, R = (sp.sympify(v) if v is not None else None for v in (V, I, R))
    if V is None:
        V = I * R
    elif I is None:
        I = V / R
    else:
        R = V / I
    return {"V": sp.simplify(V), "I": sp.simplify(I), "R": sp.simplify(R), "P": sp.simplify(V * I)}


def power_forms(V, I, R):
    """P = V I = I^2 R = V^2 / R (all three, so you can see they agree when V = I R)."""
    return {"VI": V * I, "I2R": I**2 * R, "V2/R": V**2 / R}


# ---- series / parallel / dividers ---------------------------------------------------------------
def series(*Rs):
    return sp.simplify(sum(_resistances(Rs)))


def parallel(*Rs):
    return sp.simplify(1 / sum(1 / R for R in _resistances(Rs)))


def voltage_divider(Vs, R_out, *R_others):
    """Voltage across R_out when it is in series with R_others: Vs R_out / (R_out + sum R_others)."""
    _resistances((R_out,) + R_others)
    return sp.simplify(Vs * R_out / series(R_out, *R_others))


def current_divider(Is, R_branch, *R_others):
    """Current in R_branch when it is in parallel with R_others: Is (R_eq / R_branch)."""
    _resistances((R_branch,) + R_others)
    return sp.simplify(Is * parallel(R_branch, *R_others) / R_branch)


# ---- Wheatstone bridge -------------------------------------------------------------------------
def wheatstone_output(Vs, R1, R2, R3, R4):
    """Open-circuit bridge voltage V_a - V_b, with R1 top-left, R2 bottom-left, R3 top-right,
    R4 bottom-right: V_a = Vs R2/(R1+R2), V_b = Vs R4/(R3+R4). Zero when R1 R4 = R2 R3."""
    return sp.simplify(voltage_divider(Vs, R2, R1) - voltage_divider(Vs, R4, R3))


def wheatstone_unknown(R1, R2, R3):
    """Balanced bridge: the unknown R4 = R2 R3 / R1 (what a bridge measurement reads)."""
    _resistances((R1, R2, R3))
    return sp.simplify(sp.sympify(R2) * R3 / R1)


# ---- delta <-> wye -----------------------------------------------------------------------------
def delta_to_wye(R12, R23, R31):
    """Delta resistors between terminal pairs -> wye resistors to the center: R1 = R12 R31 / S."""
    R12, R23, R31 = _resistances((R12, R23, R31))
    S = R12 + R23 + R31
    return tuple(sp.simplify(x) for x in (R12 * R31 / S, R12 * R23 / S, R23 * R31 / S))


def wye_to_delta(R1, R2, R3):
    """Wye -> delta: R12 = (R1 R2 + R2 R3 + R3 R1) / R3, and so on."""
    R1, R2, R3 = _resistances((R1, R2, R3))
    N = R1 * R2 + R2 * R3 + R3 * R1
    return tuple(sp.simplify(x) for x in (N / R3, N / R1, N / R2))


# ---- mesh-current method ------------------------------------------------------------------------
class Mesh:
    """Mesh-current analysis for planar circuits with resistors and voltage sources.

    Every mesh current is clockwise. A resistor lies on one mesh (outer edge) or is shared by two.
    ``V(name, value, mesh)`` is a source whose voltage RISES in the clockwise direction of that
    mesh (use a negative value for one that drops). KVL for mesh k:
        sum over its resistors of R (I_k - I_neighbor) = sum of source rises in mesh k.
    (Current sources need supermeshes; use node-voltage ``Circuit`` for those.)
    """

    def __init__(self):
        self.resistors, self.sources = [], []

    def R(self, name, value, m1, m2=None):
        _resistances((value,))
        if m1 == m2:
            raise ValueError(f"{name}: a shared resistor needs two different meshes")
        self.resistors.append((name, sp.sympify(value), m1, m2))
        return self

    def V(self, name, value, mesh):
        self.sources.append((name, sp.sympify(value), mesh))
        return self

    @property
    def meshes(self):
        ms = []
        for _, _, m1, m2 in self.resistors:
            for m in (m1, m2):
                if m is not None and m not in ms:
                    ms.append(m)
        return ms

    def current(self, m):
        return sp.Symbol(f"I_{m}")

    def equations(self):
        eqs = []
        for k in self.meshes:
            drop = sp.Integer(0)
            for _, R, m1, m2 in self.resistors:
                if k in (m1, m2):
                    other = m2 if k == m1 else m1
                    drop += R * (self.current(k) - (self.current(other) if other is not None else 0))
            rise = sum((v for _, v, m in self.sources if m == k), sp.Integer(0))
            eqs.append(sp.Eq(sp.expand(drop), rise))
        return eqs

    def matrix(self):
        return sp.linear_eq_to_matrix(self.equations(), [self.current(m) for m in self.meshes])

    def solve(self):
        sol = sp.solve(self.equations(), [self.current(m) for m in self.meshes], dict=True)
        if len(sol) != 1:
            raise ValueError("no unique mesh solution")
        return {k: sp.simplify(v) for k, v in sol[0].items()}

    def branch_current(self, name):
        """Current in a resistor, in the clockwise direction of its first mesh."""
        sol = self.solve()
        for n, _, m1, m2 in self.resistors:
            if n == name:
                return sp.simplify(sol[self.current(m1)] - (sol[self.current(m2)] if m2 is not None else 0))
        raise KeyError(name)

    def paper(self):
        lines = ["1. Draw a clockwise current in each mesh: " + ", ".join(map(str, (self.current(m) for m in self.meshes))),
                 "2. KVL around each mesh (resistor drops = source rises):"]
        lines += [f"     {sp.sstr(e.lhs)} = {sp.sstr(e.rhs)}" for e in self.equations()]
        lines.append("3. Solve:")
        lines += [f"     {k} = {sp.sstr(v)}" for k, v in self.solve().items()]
        return "\n".join(lines)


# ---- op-amps -------------------------------------------------------------------------------------
def inverting_amp(Rin, Rf, Vin=sp.Symbol("V_s")):
    """Ideal inverting amplifier solved by node-voltage. Returns (Circuit, V_out)."""
    ckt = (Circuit().V("Vin", "in", "0", Vin).R("Rin", "in", "n", Rin).R("Rf", "n", "out", Rf)
           .opamp("U1", "0", "n", "out"))
    return ckt, ckt.solve()[sp.Symbol("V_out")]


def noninverting_amp(Rg, Rf, Vin=sp.Symbol("V_s")):
    ckt = (Circuit().V("Vin", "in", "0", Vin).R("Rg", "n", "0", Rg).R("Rf", "n", "out", Rf)
           .opamp("U1", "in", "n", "out"))
    return ckt, ckt.solve()[sp.Symbol("V_out")]


def summing_amp(Rs, Rf, Vins):
    """Inverting summer: V_out = -Rf sum(V_k / R_k)."""
    if len(Rs) != len(Vins):
        raise ValueError("need one resistor per input")
    ckt = Circuit()
    for k, (R, V) in enumerate(zip(Rs, Vins)):
        ckt.V(f"V{k}", f"in{k}", "0", V).R(f"R{k}", f"in{k}", "n", R)
    ckt.R("Rf", "n", "out", Rf).opamp("U1", "0", "n", "out")
    return ckt, ckt.solve()[sp.Symbol("V_out")]


# ---- binary R-2R ladder DAC ------------------------------------------------------------------------
def r2r_dac(bits, Vref=1, R=1):
    """R-2R ladder: ``bits`` is MSB-first, e.g. (1, 0, 1). Solved by node-voltage.

    Returns (Circuit, V_out). Ideal answer: V_out = Vref * code / 2^n (unloaded output).
    """
    n = len(bits)
    if n < 1 or any(b not in (0, 1) for b in bits):
        raise ValueError("bits must be a non-empty sequence of 0/1")
    ckt = Circuit().R("Rterm", "n0", "0", 2 * R)              # termination at the LSB end
    lsb_first = list(reversed(bits))
    for k, b in enumerate(lsb_first):
        ckt.V(f"B{k}", f"s{k}", "0", b * Vref).R(f"R2_{k}", f"s{k}", f"n{k}", 2 * R)
        if k < n - 1:
            ckt.R(f"R_{k}", f"n{k}", f"n{k + 1}", R)
    return ckt, ckt.solve()[sp.Symbol(f"V_n{n - 1}")]
