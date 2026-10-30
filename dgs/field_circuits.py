"""Energy lives in the field, and lumped circuits are an approximation to Maxwell (Feynman-style).

Written in this repo's own words, following the topics of The Feynman Lectures on Physics Vol. II:
field energy and the Poynting vector (Ch. 27), AC circuits as an approximation (Ch. 22), and
the capacitor that turns into a cavity at high frequency (Ch. 23). Read the free online edition
at feynmanlectures.caltech.edu for his own wording.

1. Field energy: integrate u = eps0 E^2/2 + B^2/(2 mu0) and recover C V^2/2 and L I^2/2.
2. Poynting: energy flows into a resistor from the field OUTSIDE the wire, at exactly I^2 R;
   and into a charging capacitor through its open sides.
3. High frequency: a parallel-plate capacitor's field is E0 J0(w r / c); the first zero turns
   it into a resonant cavity. The lumped picture is the first term of a series.
4. The lumped-circuit assumptions, as a table: what Maxwell term each one drops and when it fails.
5. ``Circuit``: symbolic nodal analysis that writes the KCL equations you would write on paper,
   in the Laplace domain, and solves them with SymPy.
"""
from __future__ import annotations

import sympy as sp

eps0, mu0, c = sp.symbols("epsilon_0 mu_0 c", positive=True)
V, I, R, A, d, a, l, n, r, w, t = sp.symbols("V I R A d a ell n r omega t", positive=True)
s = sp.symbols("s")


# ---------------------------------------------------------------------------------------------
# 1. Field energy
# ---------------------------------------------------------------------------------------------
def capacitor_field_energy():
    """Parallel plates, area A, gap d, voltage V: integrate eps0 E^2 / 2 over the gap."""
    E = V / d
    u = eps0 * E**2 / 2
    U = u * A * d                              # uniform field fills volume A d
    C = eps0 * A / d
    return {"E": E, "u": u, "U_field": sp.simplify(U), "C": C, "U_circuit": C * V**2 / 2,
            "match": sp.simplify(U - C * V**2 / 2) == 0}


def solenoid_field_energy():
    """Long solenoid, n turns per length, area A, length ell: integrate B^2 / (2 mu0)."""
    B = mu0 * n * I
    u = B**2 / (2 * mu0)
    U = u * A * l
    L = mu0 * n**2 * A * l
    return {"B": B, "u": u, "U_field": sp.simplify(U), "L": L, "U_circuit": L * I**2 / 2,
            "match": sp.simplify(U - L * I**2 / 2) == 0}


# ---------------------------------------------------------------------------------------------
# 2. Poynting vector
# ---------------------------------------------------------------------------------------------
def wire_poynting():
    """A wire of radius a, length ell, resistance R, current I.

    Inside the wire E = I R / ell along the axis; at the surface B = mu0 I / (2 pi a) around it.
    S = E x B / mu0 points radially INWARD. Its flux through the surface 2 pi a ell is I^2 R:
    the heat comes in from the field outside, not along the wire.
    """
    E = I * R / l
    B = mu0 * I / (2 * sp.pi * a)
    S = E * B / mu0
    P_in = sp.simplify(S * 2 * sp.pi * a * l)
    return {"E": E, "B": B, "S": S, "P_in": P_in, "match": sp.simplify(P_in - I**2 * R) == 0}


def charging_capacitor_poynting():
    """Circular plates radius a, gap d, field E(t) rising.

    Ampere-Maxwell around the rim: 2 pi a B = mu0 eps0 pi a^2 dE/dt, so B = (a / 2c^2) dE/dt.
    S = E B / mu0 points inward through the open side (area 2 pi a d). Its flux equals the rate of
    change of the stored field energy (eps0 E^2 / 2) pi a^2 d.
    """
    Ef = sp.Function("E")(t)
    B = mu0 * eps0 * a * Ef.diff(t) / 2
    S = Ef * B / mu0
    P_in = S * 2 * sp.pi * a * d
    dUdt = sp.diff(eps0 * Ef**2 / 2 * sp.pi * a**2 * d, t)
    return {"B": B, "P_in": sp.simplify(P_in), "dU_dt": sp.simplify(dUdt),
            "match": sp.simplify(P_in - dUdt) == 0}


# ---------------------------------------------------------------------------------------------
# 3. Capacitor -> cavity
# ---------------------------------------------------------------------------------------------
def capacitor_field_series(order=3):
    """E(r)/E0 = J0(w r / c) as a power series: 1 - (w r / 2c)^2 + ... through r^(2 order).

    The lumped capacitor is the leading 1; each further term is a correction from the induced
    B field, then the E field it induces, and so on.
    """
    if not (isinstance(order, int) and order >= 1):
        raise ValueError("order must be an integer >= 1")
    x = w * r / c
    return sp.series(sp.besselj(0, x), r, 0, 2 * order + 1).removeO()


def first_bessel_zero():
    """First zero of J0, 2.40483..., found with SymPy's root finder."""
    return float(sp.nsolve(sp.besselj(0, sp.Symbol("x")), 2.4))


def cavity_resonance_hz(radius_m):
    """Lowest TM010 resonance of a pillbox (plates radius a): f = 2.405 c / (2 pi a)."""
    if not radius_m > 0:
        raise ValueError("radius_m must be positive")
    return first_bessel_zero() * 2.99792458e8 / (2 * 3.141592653589793 * radius_m)


# ---------------------------------------------------------------------------------------------
# 4. The lumped-circuit assumptions
# ---------------------------------------------------------------------------------------------
LUMPED_ASSUMPTIONS = [
    {"assumption": "No changing magnetic flux outside the elements",
     "maxwell": "curl E = -dB/dt", "dropped": "dB/dt outside inductors",
     "gives": "KVL: voltages around a loop add to zero; a unique V between two nodes",
     "fails_when": "loops pick up flux: ground loops, transformer coupling, crosstalk"},
    {"assumption": "No charge piling up inside wires or nodes",
     "maxwell": "div J = -d(rho)/dt", "dropped": "d(rho)/dt in the wires",
     "gives": "KCL: currents into a node add to zero",
     "fails_when": "parasitic capacitance to ground matters (fast edges, long traces)"},
    {"assumption": "Each element is much smaller than a wavelength",
     "maxwell": "wave equation, speed c", "dropped": "propagation delay",
     "gives": "every point of a wire has the same V at the same instant",
     "fails_when": "size > lambda/10: GHz boards, THz, long cables -> transmission lines"},
    {"assumption": "Each element's field is isolated inside it",
     "maxwell": "fields of different parts don't overlap", "dropped": "mutual coupling",
     "gives": "each element described by one law (V = IR, I = C dV/dt, V = L dI/dt)",
     "fails_when": "elements couple: add mutual inductance M or a parasitic C to the model"},
    {"assumption": "No radiation",
     "maxwell": "Poynting flux to infinity", "dropped": "radiated power",
     "gives": "all energy stays in R (heat), C (E field) and L (B field)",
     "fails_when": "the circuit is an antenna: size ~ lambda/4"},
]


# ---------------------------------------------------------------------------------------------
# 5. Symbolic nodal analysis
# ---------------------------------------------------------------------------------------------
class Circuit:
    """Nodal analysis in the s-domain, written the way you would on paper.

        ckt = Circuit()
        ckt.V("Vs", "in", "0", sp.Symbol("V_s"))
        ckt.R("R1", "in", "out", sp.Symbol("R"))
        ckt.C("C1", "out", "0", sp.Symbol("C"))
        ckt.transfer("out", "Vs")          # 1 / (C R s + 1)

    Node "0" is ground. Values may be numbers or SymPy symbols. Each element adds its current
    *leaving* a node to that node's KCL equation; voltage sources add an unknown current.
    """

    KINDS = ("R", "C", "L", "V", "I", "OA")

    def __init__(self):
        self.elements = []
        self.opamp_out = {}

    def _add(self, kind, name, a, b, value):
        if any(e[1] == name for e in self.elements):
            raise ValueError(f"duplicate element name {name!r}")
        if a == b:
            raise ValueError(f"{name}: both ends on node {a!r}")
        if isinstance(value, (int, float)) and kind in "RCL" and value <= 0:
            raise ValueError(f"{name}: {kind} must be positive, got {value}")
        self.elements.append((kind, name, str(a), str(b), sp.sympify(value)))
        return self

    def R(self, name, a, b, value):
        return self._add("R", name, a, b, value)

    def C(self, name, a, b, value):
        return self._add("C", name, a, b, value)

    def L(self, name, a, b, value):
        return self._add("L", name, a, b, value)

    def V(self, name, plus, minus, value):
        """Independent voltage source: V(plus) - V(minus) = value."""
        return self._add("V", name, plus, minus, value)

    def I(self, name, a, b, value):
        """Independent current source pushing ``value`` from node a, through itself, to node b."""
        return self._add("I", name, a, b, value)

    def opamp(self, name, plus, minus, out):
        """Ideal op-amp: no input current, V(plus) = V(minus) (needs negative feedback), and an
        unknown output current I_name pushed into node ``out`` (supplied by the rails)."""
        if str(out) == "0":
            raise ValueError(f"{name}: op-amp output cannot be ground")
        self._add("OA", name, plus, minus, 0)
        self.opamp_out[name] = str(out)
        return self

    @property
    def nodes(self):
        ns = []
        for kind, name, a, b, _ in self.elements:
            for x in (a, b) + ((self.opamp_out[name],) if kind == "OA" else ()):
                if x != "0" and x not in ns:
                    ns.append(x)
        if not any("0" in (e[2], e[3]) for e in self.elements):
            raise ValueError("no element touches ground node '0'")
        return ns

    def _v(self, node):
        return sp.Integer(0) if node == "0" else sp.Symbol(f"V_{node}")

    @staticmethod
    def admittance(kind, value):
        return {"R": 1 / value, "C": s * value, "L": 1 / (s * value)}[kind]

    def equations(self):
        """KCL at every non-ground node, plus one constraint per voltage source."""
        kcl = {nd: sp.Integer(0) for nd in self.nodes}
        clash = {self._v(nd) for nd in kcl} & set().union(*(e[4].free_symbols for e in self.elements))
        if clash:
            raise ValueError(f"element values use node-voltage names {sorted(map(str, clash))}; "
                             "rename the symbol or the node (e.g. source V_s on node 'in')")
        extra = []
        for kind, name, a, b, val in self.elements:
            va, vb = self._v(a), self._v(b)
            if kind in "RCL":
                i_ab = (va - vb) * self.admittance(kind, val)
            elif kind == "I":
                i_ab = val
            elif kind == "OA":
                extra.append(sp.Eq(va - vb, 0))
                kcl[self.opamp_out[name]] -= sp.Symbol(f"I_{name}")
                continue
            else:
                i_ab = sp.Symbol(f"I_{name}")
                extra.append(sp.Eq(va - vb, val))
            if a != "0":
                kcl[a] += i_ab
            if b != "0":
                kcl[b] -= i_ab
        return [sp.Eq(expr, 0) for expr in kcl.values()] + extra

    def unknowns(self):
        return [self._v(nd) for nd in self.nodes] + [sp.Symbol(f"I_{e[1]}") for e in self.elements if e[0] in ("V", "OA")]

    def matrix(self):
        """The paper form A x = b."""
        return sp.linear_eq_to_matrix(self.equations(), self.unknowns())

    def solve(self):
        sol = sp.solve(self.equations(), self.unknowns(), dict=True)
        if len(sol) != 1:
            raise ValueError("circuit has no unique solution (floating node or source loop?)")
        return {k: sp.simplify(v) for k, v in sol[0].items()}

    def transfer(self, out_node, source):
        """H(s) = V(out_node) / value of the named source."""
        src = [e for e in self.elements if e[1] == source]
        if not src:
            raise KeyError(f"no source named {source!r}")
        return sp.simplify(self.solve()[self._v(out_node)] / src[0][4])

    def element_table(self, subs=None):
        """Polarity reference, voltage, current and power for every element.

        Netlist convention (the passive sign convention):
          - an element written ``kind name a b value`` has its + reference at node a, - at node b;
          - v = V(a) - V(b), and the reference current i flows from a, THROUGH the element, to b,
            so it enters the + terminal;
          - p = v i is the power ABSORBED. p > 0: the element takes energy (a resistor, always);
            p < 0: it delivers energy (a source doing work).
        Tellegen's theorem: the p of all elements add to exactly 0 (energy is conserved).
        With capacitors or inductors the values are s-domain; substitute s or use resistive
        circuits for powers in watts.
        """
        sol = self.solve()
        subs = subs or {}
        rows = []
        for kind, name, a, b, val in self.elements:
            if kind == "OA":                      # shown as its output port: out (+) to ground (-)
                a, b = self.opamp_out[name], "0"
            v = (self._v(a) - self._v(b)).subs(sol)
            if kind == "OA":
                i = -sol[sp.Symbol(f"I_{name}")]  # I_name is pushed OUT of the + terminal
            elif kind in "RCL":
                i = v * self.admittance(kind, val)
            elif kind == "I":
                i = val
            else:
                i = sol[sp.Symbol(f"I_{name}")]
            v, i = sp.simplify(sp.sympify(v).subs(subs)), sp.simplify(sp.sympify(i).subs(subs))
            p = sp.simplify(v * i)
            role = "absorbs" if p.is_positive else "delivers" if p.is_negative else "depends on values"
            rows.append({"element": name, "kind": kind, "+ node": a, "- node": b, "v = V+ - V-": v,
                         "i (+ to - through it)": i, "p = v i (absorbed)": p, "role": role})
        return rows

    def tellegen(self, subs=None):
        """Sum of absorbed powers; 0 for every correctly solved circuit."""
        return sp.simplify(sum(r["p = v i (absorbed)"] for r in self.element_table(subs)))

    def paper(self):
        """The hand-derivation steps as text."""
        lines = ["1. Label every node; node 0 is ground: " + ", ".join(self.nodes),
                 "2. Unknowns: " + ", ".join(map(str, self.unknowns())),
                 "3. KCL (currents leaving each node = 0), and source constraints:"]
        lines += [f"     {sp.sstr(eq.lhs)} = {sp.sstr(eq.rhs)}" for eq in self.equations()]
        lines.append("4. Write as A x = b and solve (substitution, Cramer, or row reduction).")
        lines += [f"     {k} = {sp.sstr(v)}" for k, v in self.solve().items()]
        lines.append("5. Check: units, s -> 0 (capacitors open, inductors short), s -> oo.")
        return "\n".join(lines)


def example_circuits():
    """Classic circuits as (name, Circuit, output node, source)."""
    Rs, R1, R2, R3, R4, Cs, Ls, Vs = sp.symbols("R R_1 R_2 R_3 R_4 C L V_s", positive=True)
    div = Circuit().V("Vs", "in", "0", Vs).R("R1", "in", "out", R1).R("R2", "out", "0", R2)
    rc = Circuit().V("Vs", "in", "0", Vs).R("R1", "in", "out", Rs).C("C1", "out", "0", Cs)
    rlc = (Circuit().V("Vs", "in", "0", Vs).L("L1", "in", "a", Ls).C("C1", "a", "out", Cs)
           .R("R1", "out", "0", Rs))
    bridge = (Circuit().V("Vs", "top", "0", Vs).R("R1", "top", "a", R1).R("R2", "a", "0", R2)
              .R("R3", "top", "b", R3).R("R4", "b", "0", R4).R("Rg", "a", "b", Rs))
    return [("voltage divider", div, "out", "Vs"), ("RC low-pass", rc, "out", "Vs"),
            ("series RLC band-pass", rlc, "out", "Vs"), ("Wheatstone bridge (node a)", bridge, "a", "Vs")]
