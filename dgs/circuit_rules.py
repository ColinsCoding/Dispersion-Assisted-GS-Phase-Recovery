"""Circuit theory as a rulebook you can check: three axioms, a symbolic solver, ~50 derived rules, safety.

In plain words: every linear circuit you will meet in a physics or EE course (resistors, capacitors,
inductors, coupled inductors, sources, ideal op-amps) follows from three things:

  1. KCL  charge is conserved, so the currents leaving any node add to zero;
  2. KVL  voltage is a potential, so the drops around any loop add to zero;
  3. element laws  v = R i,  i = C dv/dt,  v = L di/dt (+ M di_other/dt for coupled coils).

Everything else ("series resistors add", Thevenin, the RC time constant, op-amp gain, the
transformer turns ratio) is a theorem derived from those three. This module makes that literal:

  * Circuit: a netlist (R, L, C, V, I, K coupling, E controlled source, ideal OPAMP) and a symbolic
    Modified Nodal Analysis solver in the Laplace domain (SymPy). Laplace turns d/dt into s, so a
    capacitor is just an admittance sC and an inductor an impedance sL, and the whole circuit is one
    linear system A x = b. Solving it IS applying KCL + KVL + element laws to every node at once.
  * impedance(), thevenin(), to_time(): the derived quantities you usually want.
  * RULES: ~50 named rules (series/parallel, dividers, filters, Thevenin/Norton, superposition,
    Millman, Wheatstone, Y-Delta, reciprocity, Tellegen, RC/RL/RLC dynamics, energy, op-amps,
    coupled inductors and transformers). Each rule carries a plain statement, the formula, and a
    check() that builds a small circuit, solves it from the axioms, and confirms the formula
    symbolically. verify_all() runs the whole book. That is the honest version of "a thousand lines
    of rules": a few dozen theorems, each proven by the same three axioms rather than memorised.
  * Safety: safety_check() solves the DC operating point numerically and compares every part to
    simple limits (resistor power, capacitor voltage and stored energy, inductor switch-off
    kickback), and shock_assessment() compares a touch voltage with body-current bands.

Sign conventions (SPICE): a branch current is positive flowing from the element's first node,
through the element, to its second node. So a battery that delivers power has a negative current,
and sum(v_branch * i_branch) over every element is zero (Tellegen). Inductors put their dot on the
first node. Ground is "0" (also "gnd").

SAFETY NUMBERS ARE TEACHING DEFAULTS WRITTEN FROM MEMORY, not quotes from the standards:
extra-low voltage around 50 V AC rms / 120 V ripple-free DC (IEC 61140 / IEC 60364-4-41 style, dry
conditions; roughly half that when wet), body-current bands loosely after IEC 60479-1 (perception
~0.5 mA AC, let-go ~10 mA, ventricular fibrillation possible from a few tens of mA depending on
duration and path), body impedance hand to hand falling from ~3 kOhm at 25 V to under 1 kOhm at
high voltage (BODY_IMPEDANCE table). Check the current
standard and your lab's rules before trusting any limit here with a real person.

Connects to: dgs.spice (numeric MNA + RLC transients), dgs.ac_circuits (phasors, op-amps),
dgs.mutual_inductance (the M = k sqrt(L1 L2) bound used by the K element), dgs.logic_timing
(gate delays are RC delays: see rule "50 % delay = RC ln 2"), dgs.gates_to_internet.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import sympy as sp

s = sp.Symbol("s")
t = sp.Symbol("t", positive=True)
GROUND = {"0", "gnd", "GND"}
_KINDS = {"R", "L", "C", "V", "I", "K", "E", "OPAMP"}


# ======================================================================== the netlist + solver
@dataclass
class Element:
    kind: str
    name: str
    nodes: tuple
    value: object = None
    extra: dict = field(default_factory=dict)


class Circuit:
    """A netlist. Add parts with R/L/C/V/I/K/E/opamp, then solve() for every node voltage and current.

    Values may be numbers or SymPy expressions. Source values are Laplace-domain: a DC level V that
    switches on at t = 0 is V/s; a constant (already-on) DC source for the operating point is V.
    """

    def __init__(self, title=""):
        self.title = title
        self.elements: list[Element] = []

    # ---- building
    def _add(self, kind, name, nodes, value=None, **extra):
        if any(e.name == name for e in self.elements):
            raise ValueError(f"duplicate element name {name!r}")
        nodes = tuple(str(n) for n in nodes)
        if kind in {"R", "L", "C", "V", "I"} and nodes[0] == nodes[1]:
            raise ValueError(f"{name}: both terminals on node {nodes[0]!r} (element is shorted out)")
        if kind in {"R", "L", "C"}:
            v = sp.sympify(value)
            if v.is_number and not (v > 0):
                raise ValueError(f"{name}: {kind} value must be positive; got {value!r}")
        self.elements.append(Element(kind, name, nodes, None if value is None else sp.sympify(value), extra))
        return self

    def R(self, name, n1, n2, ohms):
        return self._add("R", name, (n1, n2), ohms)

    def L(self, name, n1, n2, henries):
        """Inductor; its dot (for K coupling) is on n1."""
        return self._add("L", name, (n1, n2), henries)

    def C(self, name, n1, n2, farads):
        return self._add("C", name, (n1, n2), farads)

    def V(self, name, n_plus, n_minus, volts):
        return self._add("V", name, (n_plus, n_minus), volts)

    def I(self, name, n_from, n_to, amps):
        """Current source: `amps` flows from n_from, through the source, into n_to."""
        return self._add("I", name, (n_from, n_to), amps)

    def K(self, name, L_a, L_b, k):
        """Couple two inductors already in the netlist: M = k sqrt(La Lb), 0 <= k <= 1."""
        kk = sp.sympify(k)
        if kk.is_number and not (0 <= kk <= 1):
            raise ValueError(f"{name}: coupling k must satisfy 0 <= k <= 1; got {k!r}")
        for ln in (L_a, L_b):
            if not any(e.kind == "L" and e.name == ln for e in self.elements):
                raise ValueError(f"{name}: no inductor named {ln!r} (add it before coupling it)")
        return self._add("K", name, (), kk, L_a=L_a, L_b=L_b)

    def E(self, name, n_plus, n_minus, c_plus, c_minus, gain):
        """Voltage-controlled voltage source: v(n+) - v(n-) = gain (v(c+) - v(c-))."""
        return self._add("E", name, (n_plus, n_minus, c_plus, c_minus), gain)

    def opamp(self, name, n_plus, n_minus, n_out):
        """Ideal op-amp (nullor): no input current, v+ = v-, output supplies whatever current is needed."""
        return self._add("OPAMP", name, (n_plus, n_minus, n_out))

    def copy(self):
        return copy.deepcopy(self)

    def node_names(self):
        seen = []
        for e in self.elements:
            for n in e.nodes:
                if n not in GROUND and n not in seen:
                    seen.append(n)
        return seen

    def element(self, name):
        for e in self.elements:
            if e.name == name:
                return e
        raise KeyError(name)

    def mutual(self, k_elem):
        La, Lb = self.element(k_elem.extra["L_a"]).value, self.element(k_elem.extra["L_b"]).value
        return k_elem.value * sp.sqrt(La * Lb)

    # ---- solving
    def solve(self, s_value=None, subs=None, simplify=True):
        """Modified Nodal Analysis. Returns a Solution with V(node) and I(element) for every part.

        s_value: substitute a value for s (0 gives the DC operating point; caps open, inductors short).
        subs:    dict of symbol -> value applied to element values before solving.
        """
        nodes = self.node_names()
        idx = {n: i for i, n in enumerate(nodes)}
        branch = {}                                        # elements that carry an unknown current
        for e in self.elements:
            if e.kind in {"V", "L", "E", "OPAMP"}:
                branch[e.name] = len(nodes) + len(branch)
        N = len(nodes) + len(branch)
        if N == 0:
            raise ValueError("empty circuit")
        A = sp.zeros(N, N)
        b = sp.zeros(N, 1)
        sv = s if s_value is None else sp.sympify(s_value)
        val = lambda x: sp.sympify(x).subs(subs) if subs else sp.sympify(x)
        ni = lambda n: None if n in GROUND else idx[n]

        def add(r, c, x):
            if r is not None and c is not None:
                A[r, c] += x

        for e in self.elements:
            if e.kind in {"R", "C"}:
                y = 1 / val(e.value) if e.kind == "R" else sv * val(e.value)
                p, q = ni(e.nodes[0]), ni(e.nodes[1])
                add(p, p, y); add(q, q, y); add(p, q, -y); add(q, p, -y)
            elif e.kind == "I":
                p, q = ni(e.nodes[0]), ni(e.nodes[1])
                if p is not None:
                    b[p] -= val(e.value)
                if q is not None:
                    b[q] += val(e.value)
            elif e.kind in {"V", "L", "E"}:
                j = branch[e.name]
                p, q = ni(e.nodes[0]), ni(e.nodes[1])
                add(p, j, 1); add(q, j, -1); add(j, p, 1); add(j, q, -1)
                if e.kind == "V":
                    b[j] = val(e.value)
                elif e.kind == "L":
                    A[j, j] -= sv * val(e.value)
                else:
                    add(j, ni(e.nodes[2]), -val(e.value)); add(j, ni(e.nodes[3]), val(e.value))
            elif e.kind == "OPAMP":
                j = branch[e.name]
                p, m, o = (ni(n) for n in e.nodes)
                add(o, j, 1)
                add(j, p, 1); add(j, m, -1)
        for e in self.elements:
            if e.kind == "K":
                ja, jb = branch[e.extra["L_a"]], branch[e.extra["L_b"]]
                M = val(self.mutual(e))
                A[ja, jb] -= sv * M
                A[jb, ja] -= sv * M
        unknowns = sp.symbols(f"x0:{N}")
        sol = sp.linsolve((A, b), *unknowns)
        if not sol:
            raise ValueError("circuit has no solution (e.g. two ideal sources fighting, or a loop of ideal V sources)")
        x = list(sol)[0]
        if any(xi.free_symbols & set(unknowns) for xi in x):
            raise ValueError("circuit is underdetermined: a node floats (no DC path to ground?) "
                             "or a current is undefined")
        f = (lambda z: sp.simplify(z)) if simplify else (lambda z: z)
        volts = {n: f(x[i]) for n, i in idx.items()}
        for g in GROUND:
            volts[g] = sp.Integer(0)
        cur = {}
        for e in self.elements:
            vb = volts[e.nodes[0]] - volts[e.nodes[1]] if e.kind not in {"K", "OPAMP"} else None
            if e.kind == "R":
                cur[e.name] = f(vb / val(e.value))
            elif e.kind == "C":
                cur[e.name] = f(sv * val(e.value) * vb)
            elif e.kind == "I":
                cur[e.name] = val(e.value)
            elif e.name in branch:
                cur[e.name] = f(x[branch[e.name]])
        return Solution(self, volts, cur)


class Solution:
    """Node voltages and branch currents from Circuit.solve()."""

    def __init__(self, circuit, volts, currents):
        self.circuit, self.volts, self.currents = circuit, volts, currents

    def V(self, node, ref="0"):
        return sp.simplify(self.volts[str(node)] - self.volts[str(ref)])

    def I(self, name):
        return self.currents[name]

    def branch_voltage(self, e):
        if e.kind == "OPAMP":
            return self.volts[e.nodes[2]]                 # output port to ground
        if e.kind == "E":
            return self.volts[e.nodes[0]] - self.volts[e.nodes[1]]
        return self.volts[e.nodes[0]] - self.volts[e.nodes[1]]

    def branches(self):
        """(element, v_branch, i_branch) for everything that carries current (K and op-amp inputs don't)."""
        out = []
        for e in self.circuit.elements:
            if e.kind == "K":
                continue
            out.append((e, self.branch_voltage(e), self.currents[e.name]))
        return out


def kcl_residuals(circuit, sol):
    """Sum of currents leaving each node (should be exactly 0 everywhere)."""
    res = {n: sp.Integer(0) for n in circuit.node_names()}
    for e in circuit.elements:
        if e.kind == "K":
            continue
        n_from, n_to = (e.nodes[2], "0") if e.kind == "OPAMP" else (e.nodes[0], e.nodes[1])
        i = sol.currents[e.name]
        if n_from in res:
            res[n_from] += i
        if n_to in res:
            res[n_to] -= i
    return {n: sp.simplify(r) for n, r in res.items()}


def tellegen_sum(sol):
    """sum(v_k i_k) over all branches; Tellegen's theorem says it is exactly 0."""
    return sp.simplify(sum(v * i for _, v, i in sol.branches()))


def _zero_sources(circuit):
    c = circuit.copy()
    for e in c.elements:
        if e.kind in {"V", "I"}:
            e.value = sp.Integer(0)
    return c


def impedance(circuit, a, b="0", s_value=None):
    """Impedance seen between nodes a and b with independent sources zeroed (V shorted, I opened)."""
    c = _zero_sources(circuit)
    name = "_Itest"
    while any(e.name == name for e in c.elements):
        name += "_"
    c.I(name, b, a, 1)                               # 1 A pushed into node a
    if not any(n in GROUND for e in c.elements for n in e.nodes):
        c.V(name + "_ref", b, "0", 0)                # give a floating network a reference (carries no current)
    sol = c.solve(s_value=s_value)
    return sp.simplify(sol.V(a, b))


def thevenin(circuit, a, b="0"):
    """(V_oc, Z_th) seen at terminals a-b."""
    return sp.simplify(circuit.solve().V(a, b)), impedance(circuit, a, b)


def to_time(F, var=s):
    """Inverse Laplace transform for t > 0 (Heaviside(t) -> 1)."""
    f = sp.inverse_laplace_transform(sp.together(F), var, t)
    return sp.simplify(f.subs(sp.Heaviside(t), 1))


# ======================================================================== the rulebook
@dataclass
class Rule:
    name: str
    category: str
    statement: str
    formula: str
    check: Callable[[], bool]

    def verify(self):
        return bool(self.check())


RULES: list[Rule] = []
CATEGORIES = ["axiom", "combination", "divider", "filter", "theorem", "dynamics", "op-amp",
              "coupled", "power"]


def rule(name, category, statement, formula):
    if category not in CATEGORIES:
        raise ValueError(category)

    def deco(fn):
        RULES.append(Rule(name, category, statement, formula, fn))
        return fn
    return deco


def get_rule(name):
    for r in RULES:
        if r.name == name:
            return r
    raise KeyError(name)


def verify_all():
    """Run every rule's check. Returns list of (name, category, ok)."""
    return [(r.name, r.category, r.verify()) for r in RULES]


def _same(a, b):
    d = sp.simplify(sp.together(sp.sympify(a) - sp.sympify(b)))
    return d == 0 or sp.simplify(sp.expand(d)) == 0


R1, R2, R3, R4, R5, RL, Rf, Rg = sp.symbols("R1 R2 R3 R4 R5 R_L R_f R_g", positive=True)
C1, C2, L1, L2, Va, Vb, Vc, I0, k = sp.symbols("C1 C2 L1 L2 V_a V_b V_c I_0 k", positive=True)
V0 = sp.Symbol("V_0", positive=True)


def _mixed():
    """A deliberately messy circuit: source, R, L, C and a coupled coil, used by the axiom checks.

    Part values are exact rationals and only s stays symbolic: KCL, Tellegen and reciprocity are
    statements about the network's structure, and with ten free symbols the exact rational-function
    answers grow too large to simplify in reasonable time.
    """
    c = Circuit("mixed")
    c.V("Vs", "in", "0", Va).R("R1", "in", "a", 3).L("L1", "a", "b", 2).C("C1", "b", "0", sp.Rational(1, 5))
    c.R("R2", "a", "0", 7).L("L2", "c", "0", 8).R("R3", "c", "b", 11).K("K1", "L1", "L2", sp.Rational(1, 2))
    return c


# ---------------------------------------------------------------- axioms
@rule("KCL", "axiom", "Charge is conserved: the currents leaving any node add to zero.", "sum_k i_k = 0")
def _kcl():
    c = _mixed()
    return all(r == 0 for r in kcl_residuals(c, c.solve()).values())


@rule("KVL", "axiom", "Around a loop the source voltage equals the sum of the drops.",
      "V = I (R + sL + 1/(sC))")
def _kvl():
    c = Circuit().V("Vs", "1", "0", Va).R("R1", "1", "2", R1).L("L1", "2", "3", L1).C("C1", "3", "0", C1)
    sol = c.solve()
    i = -sol.I("Vs")
    return _same(Va, i * (R1 + s * L1 + 1 / (s * C1)))


@rule("Ohm's law", "axiom", "A resistor's current is its voltage divided by its resistance.", "i = v / R")
def _ohm():
    sol = Circuit().V("Vs", "1", "0", Va).R("R1", "1", "0", R1).solve()
    return _same(sol.I("R1"), Va / R1)


@rule("Capacitor law", "axiom", "i = C dv/dt; in Laplace terms an impedance 1/(sC).", "Z_C = 1/(sC)")
def _cap():
    return _same(impedance(Circuit().C("C1", "a", "0", C1), "a"), 1 / (s * C1))


@rule("Inductor law", "axiom", "v = L di/dt; in Laplace terms an impedance sL.", "Z_L = sL")
def _ind():
    return _same(impedance(Circuit().L("L1", "a", "0", L1), "a"), s * L1)


@rule("Coupled inductors", "axiom",
      "Current changing in one coil induces M di/dt in the other, with M = k sqrt(L1 L2).",
      "v2 = s M i1,  M = k sqrt(L1 L2)")
def _coupled():
    c = Circuit().I("I1", "0", "a", I0).L("L1", "a", "0", L1).L("L2", "b", "0", L2).K("K", "L1", "L2", k)
    return _same(c.solve().V("b"), s * k * sp.sqrt(L1 * L2) * I0)


# ---------------------------------------------------------------- combinations
def _Z2(kind, v1, v2, series):
    c = Circuit()
    add = getattr(c, kind)
    if series:
        add(f"{kind}1", "a", "m", v1); add(f"{kind}2", "m", "0", v2)
    else:
        add(f"{kind}1", "a", "0", v1); add(f"{kind}2", "a", "0", v2)
    return impedance(c, "a")


@rule("Series resistors", "combination", "Resistors in series add.", "R = R1 + R2")
def _sr():
    return _same(_Z2("R", R1, R2, True), R1 + R2)


@rule("Parallel resistors", "combination", "Conductances in parallel add.", "R = R1 R2 / (R1 + R2)")
def _pr():
    return _same(_Z2("R", R1, R2, False), R1 * R2 / (R1 + R2))


@rule("Series inductors", "combination", "Uncoupled inductors in series add like resistors.", "L = L1 + L2")
def _sl():
    return _same(_Z2("L", L1, L2, True), s * (L1 + L2))


@rule("Parallel inductors", "combination", "Uncoupled inductors in parallel combine like resistors.",
      "L = L1 L2 / (L1 + L2)")
def _pl():
    return _same(_Z2("L", L1, L2, False), s * L1 * L2 / (L1 + L2))


@rule("Series capacitors", "combination", "Capacitors combine the opposite way: series like parallel R.",
      "C = C1 C2 / (C1 + C2)")
def _sc():
    return _same(_Z2("C", C1, C2, True), 1 / (s * C1 * C2 / (C1 + C2)))


@rule("Parallel capacitors", "combination", "Capacitors in parallel add (plate areas add).", "C = C1 + C2")
def _pc():
    return _same(_Z2("C", C1, C2, False), 1 / (s * (C1 + C2)))


@rule("Series aiding coils", "coupled", "Coupled coils in series, dots aligned: the mutual adds twice.",
      "L = L1 + L2 + 2M")
def _aid():
    c = Circuit().L("L1", "a", "m", L1).L("L2", "m", "0", L2).K("K", "L1", "L2", k)
    return _same(impedance(c, "a"), s * (L1 + L2 + 2 * k * sp.sqrt(L1 * L2)))


@rule("Series opposing coils", "coupled", "Flip one coil and the mutual subtracts twice; "
      "so M = (L_aid - L_opp)/4.", "L = L1 + L2 - 2M")
def _opp():
    c = Circuit().L("L1", "a", "m", L1).L("L2", "0", "m", L2).K("K", "L1", "L2", k)
    return _same(impedance(c, "a"), s * (L1 + L2 - 2 * k * sp.sqrt(L1 * L2)))


# ---------------------------------------------------------------- dividers and filters
@rule("Voltage divider", "divider", "Series resistors split a voltage in proportion to resistance.",
      "v_out = V R2 / (R1 + R2)")
def _vdiv():
    sol = Circuit().V("Vs", "in", "0", Va).R("R1", "in", "out", R1).R("R2", "out", "0", R2).solve()
    return _same(sol.V("out"), Va * R2 / (R1 + R2))


@rule("Current divider", "divider", "Parallel branches split a current in proportion to the OTHER "
      "branch's resistance.", "i_1 = I R2 / (R1 + R2)")
def _idiv():
    sol = Circuit().I("I", "0", "a", I0).R("R1", "a", "0", R1).R("R2", "a", "0", R2).solve()
    return _same(sol.I("R1"), I0 * R2 / (R1 + R2))


@rule("RC low-pass", "filter", "Series R, shunt C: passes slow signals, corner at 1/(2 pi RC).",
      "H(s) = 1 / (1 + sRC)")
def _lp():
    sol = Circuit().V("Vs", "in", "0", Va).R("R1", "in", "out", R1).C("C1", "out", "0", C1).solve()
    return _same(sol.V("out") / Va, 1 / (1 + s * R1 * C1))


@rule("RC high-pass", "filter", "Series C, shunt R: blocks DC, passes fast signals.",
      "H(s) = sRC / (1 + sRC)")
def _hp():
    sol = Circuit().V("Vs", "in", "0", Va).C("C1", "in", "out", C1).R("R1", "out", "0", R1).solve()
    return _same(sol.V("out") / Va, s * R1 * C1 / (1 + s * R1 * C1))


@rule("RL low-pass", "filter", "Series L, shunt R: the inductor chokes fast changes; tau = L/R.",
      "H(s) = R / (R + sL)")
def _rl():
    sol = Circuit().V("Vs", "in", "0", Va).L("L1", "in", "out", L1).R("R1", "out", "0", R1).solve()
    return _same(sol.V("out") / Va, R1 / (R1 + s * L1))


# ---------------------------------------------------------------- network theorems
def _thev_net():
    return Circuit().V("Vs", "in", "0", Va).R("R1", "in", "a", R1).R("R2", "a", "0", R2).C("C1", "a", "0", C1)


@rule("Thevenin", "theorem", "Any linear two-terminal network acts like one source V_oc in series with "
      "one impedance Z_th, for every load.", "v_load = V_oc Z_L / (Z_L + Z_th)")
def _thev():
    c = _thev_net()
    Voc, Zth = thevenin(c, "a")
    loaded = c.copy().R("RL", "a", "0", RL).solve().V("a")
    return _same(loaded, Voc * RL / (RL + Zth))


@rule("Norton", "theorem", "The same network is a current source I_sc in parallel with Z_th.",
      "I_sc = V_oc / Z_th")
def _norton():
    c = _thev_net()
    Voc, Zth = thevenin(c, "a")
    Isc = c.copy().V("Vshort", "a", "0", 0).solve().I("Vshort")   # current into + of the short = I_sc
    return _same(Isc, Voc / Zth)


@rule("Source transformation", "theorem", "V in series with R behaves exactly like V/R in parallel "
      "with R, as far as any load can tell.", "V, R (series)  <->  V/R, R (parallel)")
def _srct():
    a = Circuit().V("Vs", "in", "0", Va).R("R1", "in", "a", R1).R("RL", "a", "0", RL).solve().V("a")
    b = Circuit().I("Is", "0", "a", Va / R1).R("R1", "a", "0", R1).R("RL", "a", "0", RL).solve().V("a")
    return _same(a, b)


@rule("Superposition", "theorem", "With several independent sources, the response is the sum of the "
      "responses to each source alone (others zeroed).", "v = v|_(only Va) + v|_(only I0)")
def _sup():
    v = Circuit().V("Vs", "in", "0", Va).R("R1", "in", "n", R1).I("I", "0", "n", I0).R("R2", "n", "0", R2) \
        .solve().V("n")
    return _same(v, v.subs(I0, 0) + v.subs(Va, 0))


@rule("Maximum power transfer", "theorem", "A resistive load takes the most power when it equals the "
      "source resistance, and then gets V^2 / 4R.", "R_L = R_th,  P_max = V_th^2 / (4 R_th)")
def _mpt():
    sol = Circuit().V("Vs", "in", "0", Va).R("R1", "in", "a", R1).R("RL", "a", "0", RL).solve()
    P = sp.simplify(sol.V("a") * sol.I("RL"))
    best = sp.solve(sp.diff(P, RL), RL)
    return best == [R1] and _same(P.subs(RL, R1), Va ** 2 / (4 * R1))


@rule("Millman", "theorem", "Sources feeding one node through resistors: the node sits at the "
      "conductance-weighted average.", "v = sum(V_i/R_i) / sum(1/R_i)")
def _millman():
    c = Circuit().V("A", "pa", "0", Va).V("B", "pb", "0", Vb).V("Cc", "pc", "0", Vc)
    c.R("R1", "pa", "n", R1).R("R2", "pb", "n", R2).R("R3", "pc", "n", R3)
    return _same(c.solve().V("n"), (Va / R1 + Vb / R2 + Vc / R3) / (1 / R1 + 1 / R2 + 1 / R3))


@rule("Wheatstone balance", "theorem", "A bridge carries no current through its middle when the ratios "
      "match.", "R1 R4 = R2 R3  ->  i_5 = 0")
def _wheat():
    c = Circuit().V("Vs", "top", "0", Va).R("R1", "top", "a", R1).R("R2", "a", "0", R2)
    c.R("R3", "top", "b", R3).R("R4", "b", "0", R4).R("R5", "a", "b", R5)
    return _same(c.solve().I("R5").subs(R4, R2 * R3 / R1), 0)


@rule("Y-Delta", "theorem", "A triangle of resistors equals a star with R_a = R_ab R_ca / sum, etc.",
      "R_a = R_ab R_ca / (R_ab + R_bc + R_ca)")
def _ydelta():
    S = R1 + R2 + R3                             # R1 = R_ab, R2 = R_bc, R3 = R_ca
    D = Circuit().R("Rab", "a", "b", R1).R("Rbc", "b", "c", R2).R("Rca", "c", "a", R3)
    Y = Circuit().R("Ra", "a", "n", R1 * R3 / S).R("Rb", "b", "n", R1 * R2 / S).R("Rc", "c", "n", R2 * R3 / S)
    pairs = [("a", "b"), ("b", "c"), ("c", "a")]
    return all(_same(impedance(D, p, q), impedance(Y, p, q)) for p, q in pairs)


@rule("Reciprocity", "theorem", "In any network of R, L, C and coupled coils, the transfer impedance is "
      "the same both ways: drive port 1, read port 2, or the reverse.", "Z_21 = Z_12")
def _recip():
    def net():
        c = Circuit().R("R1", "p", "a", 3).L("L1", "a", "0", 2).C("C1", "a", "q", sp.Rational(1, 5))
        c.L("L2", "q", "0", 8).R("R2", "p", "0", 7).R("R3", "q", "0", 11).K("K", "L1", "L2", sp.Rational(1, 3))
        return c
    z21 = net().I("I", "0", "p", 1).solve().V("q")
    z12 = net().I("I", "0", "q", 1).solve().V("p")
    return _same(z21, z12)


@rule("Tellegen", "theorem", "Power balances exactly: summed over every branch (sources included), "
      "v times i is zero.", "sum_k v_k i_k = 0")
def _tellegen():
    return tellegen_sum(_mixed().solve()) == 0


# ---------------------------------------------------------------- dynamics and energy
@rule("RC step", "dynamics", "A capacitor charged through R approaches the source exponentially.",
      "v_C(t) = V (1 - exp(-t/RC))")
def _rcstep():
    v = Circuit().V("Vs", "in", "0", V0 / s).R("R1", "in", "out", R1).C("C1", "out", "0", C1).solve().V("out")
    return _same(to_time(v), V0 * (1 - sp.exp(-t / (R1 * C1))))


@rule("RL step", "dynamics", "Current in an inductor builds up exponentially with tau = L/R.",
      "i(t) = (V/R)(1 - exp(-R t / L))")
def _rlstep():
    i = Circuit().V("Vs", "in", "0", V0 / s).R("R1", "in", "a", R1).L("L1", "a", "0", L1).solve().I("L1")
    return _same(to_time(i), V0 / R1 * (1 - sp.exp(-R1 * t / L1)))


@rule("50 % delay", "dynamics", "An RC node crosses half-swing after RC ln 2 (the gate-delay formula).",
      "t_50 = RC ln 2")
def _t50():
    v = to_time(Circuit().V("Vs", "in", "0", V0 / s).R("R1", "in", "o", R1).C("C1", "o", "0", C1).solve().V("o"))
    sol = sp.solve(sp.Eq(v, V0 / 2), t)
    return len(sol) == 1 and _same(sol[0], R1 * C1 * sp.log(2))


def _char_poly(expr):
    """Monic denominator-in-s of a rational expression (the characteristic polynomial)."""
    num, den = sp.fraction(sp.together(expr))
    p = sp.Poly(sp.expand(den), s)
    return sp.Poly(sp.expand(den / p.LC()), s)


@rule("Series RLC", "dynamics", "Series R-L-C resonates at 1/sqrt(LC) with quality factor sqrt(L/C)/R.",
      "s^2 + (R/L) s + 1/(LC),  Q = sqrt(L/C)/R")
def _srlc():
    c = Circuit().R("R1", "a", "b", R1).L("L1", "b", "c", L1).C("C1", "c", "0", C1)
    p = _char_poly(1 / impedance(c, "a"))
    a1, a0 = p.coeff_monomial(s), p.coeff_monomial(1)
    w0 = sp.sqrt(a0)
    return p.degree() == 2 and _same(a1, R1 / L1) and _same(a0, 1 / (L1 * C1)) and \
        _same(w0 / a1, sp.sqrt(L1 / C1) / R1)


@rule("Parallel RLC", "dynamics", "Parallel R-L-C: same resonance, but Q = R sqrt(C/L) (bigger R, sharper).",
      "s^2 + s/(RC) + 1/(LC),  Q = R sqrt(C/L)")
def _prlc():
    c = Circuit().R("R1", "a", "0", R1).L("L1", "a", "0", L1).C("C1", "a", "0", C1)
    p = _char_poly(impedance(c, "a"))
    a1, a0 = p.coeff_monomial(s), p.coeff_monomial(1)
    return _same(a1, 1 / (R1 * C1)) and _same(a0, 1 / (L1 * C1)) and _same(sp.sqrt(a0) / a1, R1 * sp.sqrt(C1 / L1))


@rule("Critical damping", "dynamics", "A series RLC rings if R < 2 sqrt(L/C), crawls if larger; exactly "
      "2 sqrt(L/C) is the fastest settle with no overshoot.", "R_crit = 2 sqrt(L/C)")
def _crit():
    p = sp.Poly(s ** 2 + R1 / L1 * s + 1 / (L1 * C1), s)
    roots = sp.solve(sp.discriminant(p.as_expr(), s), R1)
    return roots == [2 * sp.sqrt(L1 / C1)] or _same(roots[0], 2 * sp.sqrt(L1 / C1))


@rule("Capacitor energy", "dynamics", "Charging C to V through any R stores CV^2/2 and burns another "
      "CV^2/2 in R; the source supplies CV^2.", "W_C = C V^2 / 2 = W_R")
def _ecap():
    i = to_time(Circuit().V("Vs", "in", "0", V0 / s).R("R1", "in", "o", R1).C("C1", "o", "0", C1).solve().I("R1"))
    W_R = sp.integrate(i ** 2 * R1, (t, 0, sp.oo))
    W_src = sp.integrate(V0 * i, (t, 0, sp.oo))
    return _same(W_R, C1 * V0 ** 2 / 2) and _same(W_src, C1 * V0 ** 2)


@rule("Inductor energy", "dynamics", "An inductor carrying I stores L I^2 / 2 in its field.", "W_L = L I^2 / 2")
def _eind():
    i = to_time(Circuit().V("Vs", "in", "0", V0 / s).R("R1", "in", "a", R1).L("L1", "a", "0", L1).solve().I("L1"))
    v = L1 * sp.diff(i, t)
    W = sp.integrate(v * i, (t, 0, sp.oo))
    return _same(W, L1 * (V0 / R1) ** 2 / 2)


@rule("Charge sharing", "dynamics", "Connect a charged C1 to an empty C2: charge is conserved, and "
      "C1 C2 V^2 / 2(C1 + C2) is lost in the wire no matter how small its R.",
      "v_f = C1 V / (C1 + C2),  W_lost = C1 C2 V^2 / (2 (C1 + C2))")
def _share():
    # a charged capacitor = uncharged C in series with a step source V/s (Laplace initial-condition model)
    c = Circuit().V("Vic", "p", "0", V0 / s).C("C1", "p", "a", C1).R("R1", "a", "b", R1).C("C2", "b", "0", C2)
    sol = c.solve()
    i = to_time(sol.I("R1"))
    lost = sp.integrate(i ** 2 * R1, (t, 0, sp.oo))
    vf = sp.limit(s * sol.V("b"), s, 0)
    return _same(lost, C1 * C2 * V0 ** 2 / (2 * (C1 + C2))) and _same(vf, C1 * V0 / (C1 + C2))


# ---------------------------------------------------------------- op-amps
@rule("Op-amp buffer", "op-amp", "Output tied to the minus input copies the plus input.", "v_out = v_in")
def _buf():
    sol = Circuit().V("Vs", "in", "0", Va).opamp("U", "in", "out", "out").R("RL", "out", "0", RL).solve()
    return _same(sol.V("out"), Va)


@rule("Inverting amplifier", "op-amp", "Virtual ground at the minus input: gain -Rf/Rin.",
      "v_out = -(R_f / R_1) v_in")
def _inv():
    sol = Circuit().V("Vs", "in", "0", Va).R("R1", "in", "n", R1).R("Rf", "n", "out", Rf).opamp("U", "0", "n", "out").solve()
    return _same(sol.V("out"), -Rf / R1 * Va)


@rule("Non-inverting amplifier", "op-amp", "Feedback divider sets gain 1 + Rf/Rg, input draws no current.",
      "v_out = (1 + R_f / R_g) v_in")
def _noninv():
    sol = Circuit().V("Vs", "in", "0", Va).R("Rg", "n", "0", Rg).R("Rf", "n", "out", Rf).opamp("U", "in", "n", "out").solve()
    return _same(sol.V("out"), (1 + Rf / Rg) * Va)


@rule("Summing amplifier", "op-amp", "Currents into the virtual ground add: an analog adder.",
      "v_out = -R_f (V_a/R_1 + V_b/R_2)")
def _sum():
    c = Circuit().V("A", "pa", "0", Va).V("B", "pb", "0", Vb).R("R1", "pa", "n", R1).R("R2", "pb", "n", R2)
    sol = c.R("Rf", "n", "out", Rf).opamp("U", "0", "n", "out").solve()
    return _same(sol.V("out"), -Rf * (Va / R1 + Vb / R2))


@rule("Difference amplifier", "op-amp", "Matched resistor pairs amplify only the difference.",
      "v_out = (R_f / R_1)(V_b - V_a)")
def _diff():
    c = Circuit().V("A", "pa", "0", Va).V("B", "pb", "0", Vb)
    c.R("R1", "pa", "n", R1).R("Rf", "n", "out", Rf).R("R1b", "pb", "p", R1).R("Rfb", "p", "0", Rf)
    return _same(c.opamp("U", "p", "n", "out").solve().V("out"), Rf / R1 * (Vb - Va))


@rule("Integrator", "op-amp", "Capacitor feedback integrates: v_out = -(1/RC) integral v_in dt.",
      "H(s) = -1 / (sRC)")
def _integ():
    sol = Circuit().V("Vs", "in", "0", Va).R("R1", "in", "n", R1).C("C1", "n", "out", C1).opamp("U", "0", "n", "out").solve()
    return _same(sol.V("out") / Va, -1 / (s * R1 * C1))


@rule("Differentiator", "op-amp", "Swap R and C and it differentiates: v_out = -RC dv_in/dt.",
      "H(s) = -sRC")
def _differ():
    sol = Circuit().V("Vs", "in", "0", Va).C("C1", "in", "n", C1).R("R1", "n", "out", R1).opamp("U", "0", "n", "out").solve()
    return _same(sol.V("out") / Va, -s * R1 * C1)


# ---------------------------------------------------------------- coupled coils and transformers
@rule("Mutual inductance bound", "coupled", "Stored energy can't be negative, so M^2 <= L1 L2, i.e. "
      "M = k sqrt(L1 L2) with 0 <= k <= 1 (see dgs.mutual_inductance).", "M = k sqrt(L1 L2), k <= 1")
def _mbound():
    from dgs.mutual_inductance import derive_energy_bound
    d = derive_energy_bound()
    Ls1, Ls2 = sp.symbols("L_1 L_2", positive=True)
    M = sp.Symbol("M", real=True)
    return _same(d["discriminant"], M ** 2 - Ls1 * Ls2)


def _xfmr(kval=k, load=RL):
    c = Circuit().V("Vs", "p", "0", Va).L("L1", "p", "0", L1).L("L2", "q", "0", L2).K("K", "L1", "L2", kval)
    return c.R("RL", "q", "0", load)


@rule("Reflected impedance", "coupled", "A load on the secondary shows up at the primary through M.",
      "Z_in = sL1 - s^2 M^2 / (sL2 + Z_L)")
def _refl():
    sol = _xfmr().solve()
    M = k * sp.sqrt(L1 * L2)
    return _same(Va / (-sol.I("Vs")), s * L1 - s ** 2 * M ** 2 / (s * L2 + RL))


@rule("Ideal transformer ratio", "coupled", "With perfect coupling (k = 1) the voltage ratio is "
      "sqrt(L2/L1) = N2/N1 for every load.", "v2 / v1 = sqrt(L2 / L1)")
def _ideal():
    sol = _xfmr(1).solve()
    return _same(sol.V("q") / Va, sp.sqrt(L2 / L1))


@rule("T-equivalent", "coupled", "Coupled coils with a common ground equal three plain inductors: "
      "L1 - M, L2 - M in series, M to ground.", "(L1 - M, M, L2 - M)")
def _teq():
    M = k * sp.sqrt(L1 * L2)
    coupled = _xfmr().solve()
    T = Circuit().V("Vs", "p", "0", Va).L("La", "p", "m", L1 - M).L("Lm", "m", "0", M).L("Lb", "m", "q", L2 - M)
    tsol = T.R("RL", "q", "0", RL).solve()
    return _same(coupled.I("Vs"), tsol.I("Vs")) and _same(coupled.V("q"), tsol.V("q"))


# ---------------------------------------------------------------- power
@rule("Resistor power", "power", "Power burned in a resistor: v i = i^2 R = v^2 / R (all three agree).",
      "P = v i = i^2 R = v^2 / R")
def _pres():
    sol = Circuit().V("Vs", "1", "0", Va).R("R1", "1", "2", R1).R("R2", "2", "0", R2).solve()
    v, i = sol.V("1", "2"), sol.I("R1")
    return _same(v * i, i ** 2 * R1) and _same(v * i, v ** 2 / R1)


@rule("Source power", "power", "The source supplies exactly what the resistors burn (energy conservation).",
      "-V i_V = sum i^2 R")
def _psrc():
    sol = Circuit().V("Vs", "1", "0", Va).R("R1", "1", "2", R1).R("R2", "2", "0", R2).R("R3", "2", "0", R3).solve()
    burned = sum(sol.I(n) ** 2 * r for n, r in (("R1", R1), ("R2", R2), ("R3", R3)))
    return _same(-Va * sol.I("Vs"), burned)


@rule("AC average power", "power", "With sinusoids the average power is (1/2) Re(V I*): only the "
      "in-phase part of the current does work.", "P = |V|^2 Re(Y) / 2")
def _pac():
    w = sp.Symbol("omega", positive=True)
    sol = Circuit().V("Vs", "1", "0", Va).R("R1", "1", "2", R1).L("L1", "2", "0", L1).solve(s_value=sp.I * w)
    i = -sol.I("Vs")
    P = sp.simplify(sp.re(sp.expand_complex(Va * sp.conjugate(i))) / 2)
    return _same(P, sp.simplify(sp.Abs(i) ** 2 * R1 / 2))


# ======================================================================== safety
SAFETY = {
    # teaching defaults, from memory -- see module docstring
    "elv_ac_rms": 50.0,          # V, dry conditions
    "elv_dc": 120.0,             # V, ripple-free, dry
    "elv_ac_rms_wet": 25.0,
    "elv_dc_wet": 60.0,
    "body_resistance": None,     # ohm; None -> voltage-dependent BODY_IMPEDANCE table below
    "resistor_rating_W": 0.25,
    "resistor_derating": 0.5,    # run at <= 50 % of rating
    "cap_voltage_derating": 0.8,  # run at <= 80 % of rated voltage
    "cap_energy_J": 1.0,         # above ~1 J a discharge can burn or startle badly (rule of thumb)
    "switch_rating_V": 60.0,     # default switch / transistor voltage rating for kickback checks
}

# Total body impedance, hand to hand, dry, large contact area, 50th percentile, loosely after the
# IEC 60479-1 table (FROM MEMORY): it falls as voltage rises because the skin breaks down.
BODY_IMPEDANCE = [(25, 3250), (50, 2500), (75, 2000), (100, 1725), (125, 1550), (220, 1225),
                  (700, 775), (1000, 775)]


def body_impedance(V):
    """Hand-to-hand body impedance (ohm) at touch voltage V, interpolated; flat outside 25 V..1 kV."""
    vs, zs = zip(*BODY_IMPEDANCE)
    return float(np.interp(abs(V), vs, zs))


AC_BANDS = [   # (upper limit in mA, label), simplified from IEC 60479-1 style zones (duration ignored)
    (0.5, "below perception"),
    (5.0, "perceptible, startling, usually harmless"),
    (10.0, "painful; near the let-go limit"),
    (30.0, "cannot let go; muscle contraction, breathing difficulty"),
    (math.inf, "ventricular fibrillation possible (depends on duration and path): DANGEROUS"),
]
DC_BANDS = [
    (2.0, "below perception"),
    (25.0, "perceptible to painful"),
    (100.0, "severe; fibrillation possible for long exposures"),
    (math.inf, "fibrillation likely: DANGEROUS"),
]


def shock_assessment(touch_voltage, ac=True, body_resistance=None, wet=False):
    """Body current for a touch voltage and which band it falls in. Returns dict."""
    V = abs(float(touch_voltage))
    Rb = body_resistance if body_resistance is not None else (SAFETY["body_resistance"] or body_impedance(V))
    if Rb <= 0:
        raise ValueError("body resistance must be positive")
    I_mA = 1e3 * V / Rb
    bands = AC_BANDS if ac else DC_BANDS
    label = next(lbl for lim, lbl in bands if I_mA < lim)
    limit = (SAFETY["elv_ac_rms_wet"] if wet else SAFETY["elv_ac_rms"]) if ac else \
        (SAFETY["elv_dc_wet"] if wet else SAFETY["elv_dc"])
    return {"V": V, "R_body": Rb, "I_mA": I_mA, "band": label, "elv_limit_V": limit, "within_elv": V <= limit}


def bleeder_time(C, R, V_start, V_safe=None):
    """Time for a bleeder R to discharge C from V_start to V_safe: RC ln(V_start / V_safe)."""
    V_safe = SAFETY["elv_dc_wet"] if V_safe is None else V_safe
    if not (C > 0 and R > 0 and V_start > 0 and V_safe > 0):
        raise ValueError("C, R, V_start, V_safe must be positive")
    return 0.0 if V_start <= V_safe else R * C * math.log(V_start / V_safe)


def inductor_kickback(L, I, t_off):
    """|v| = L I / t_off when a switch interrupts inductor current I in time t_off (why flyback diodes exist)."""
    if not (L > 0 and t_off > 0):
        raise ValueError("L and t_off must be positive")
    return abs(L * I / t_off)


def safety_check(circuit, values=None, ratings=None, switch_off_time=1e-6):
    """DC operating point + per-part limits. Returns list of finding dicts (part, quantity, value, limit, status, note).

    ratings: optional {element name: rating}, W for resistors, V for capacitors, V for the switch
    that may interrupt an inductor. Element values must be numeric after `values` substitution.
    """
    ratings = ratings or {}
    sol = circuit.solve(s_value=0, subs=values, simplify=False)
    num = lambda z: float(sp.N(sp.sympify(z).subs(values or {})))
    out = []

    def rec(part, qty, val, lim, ok, note):
        out.append({"part": part, "quantity": qty, "value": val, "limit": lim,
                    "status": "OK" if ok else "WARN", "note": note})

    for e in circuit.elements:
        if e.kind == "R":
            v = num(sol.branch_voltage(e)); P = v * v / num(e.value)
            lim = ratings.get(e.name, SAFETY["resistor_rating_W"]) * SAFETY["resistor_derating"]
            rec(e.name, "power (W)", P, lim, P <= lim, f"{P*1e3:.3g} mW vs derated limit {lim*1e3:.3g} mW")
        elif e.kind == "C":
            v = abs(num(sol.branch_voltage(e))); W = 0.5 * num(e.value) * v * v
            if e.name in ratings:
                lim = ratings[e.name] * SAFETY["cap_voltage_derating"]
                rec(e.name, "voltage (V)", v, lim, v <= lim, f"{v:.3g} V vs {lim:.3g} V (80 % of rating)")
            rec(e.name, "stored energy (J)", W, SAFETY["cap_energy_J"], W <= SAFETY["cap_energy_J"],
                f"{W:.3g} J; touch voltage {v:.3g} V " + ("(above DC ELV: add a bleeder)" if v > SAFETY["elv_dc"] else ""))
        elif e.kind == "L":
            i = abs(num(sol.I(e.name)))
            vk = inductor_kickback(num(e.value), i, switch_off_time)
            lim = ratings.get(e.name, SAFETY["switch_rating_V"])
            rec(e.name, "switch-off kickback (V)", vk, lim, vk <= lim,
                f"L I / t_off = {vk:.3g} V for t_off = {switch_off_time:g} s; "
                + ("fine" if vk <= lim else "add a flyback diode or snubber"))
        elif e.kind == "V":
            v = abs(num(e.value))
            ok = v <= SAFETY["elv_dc"]
            rec(e.name, "source voltage (V)", v, SAFETY["elv_dc"], ok,
                "extra-low voltage (dry)" if ok else "above ELV: insulation and interlocks required")
    return out
