"""Digital logic + power basics for CpE/EE coursework.

Sections
--------
1. Greek alphabet (Alpha to Omega) as SymPy symbols, with what each letter usually means.
2. The 7 basic gates, truth tables, and NAND universality (every gate built from NAND only).
3. Multiplexers: a 2^k:1 MUX, its Boolean expression, and any function implemented with one.
4. Karnaugh maps: Gray-code grids for 2-4 variables plus a minimal sum of products (SymPy SOPform),
   checked against the minterm list.
5. Power three ways (P = dW/dt, P = F.v, P = IV) and energy from geometry, mass, and charge,
   with E = P t solved for whichever of E, P, t is missing.
"""
from __future__ import annotations

from itertools import product

import sympy as sp
from sympy.logic.boolalg import SOPform, simplify_logic
from sympy.physics import units as u

# ---------------------------------------------------------------------------------------------
# 1. Greek alphabet
# ---------------------------------------------------------------------------------------------
_GREEK = [
    ("alpha", "angle; alpha particle; fine-structure constant"),
    ("beta", "v/c in relativity; beta decay; transistor current gain"),
    ("gamma", "Lorentz factor; gamma ray; photon"),
    ("delta", "small change; Dirac delta; skin depth"),
    ("epsilon", "permittivity epsilon_0; small quantity; emissivity"),
    ("zeta", "damping ratio in 2nd-order circuits"),
    ("eta", "efficiency"),
    ("theta", "angle; phase"),
    ("iota", "rarely used (index)"),
    ("kappa", "dielectric constant; thermal conductivity"),
    ("lambda", "wavelength; decay constant; eigenvalue"),
    ("mu", "permeability mu_0; micro-; magnetic moment; mobility"),
    ("nu", "frequency (physics); neutrino"),
    ("xi", "random variable; coherence length"),
    ("omicron", "rarely used (looks like o)"),
    ("pi", "3.14159...; pi bonds; product (upper case)"),
    ("rho", "charge or mass density; resistivity"),
    ("sigma", "conductivity; surface charge; std deviation; sum (upper case)"),
    ("tau", "time constant RC or L/R; torque; lifetime"),
    ("upsilon", "Upsilon meson (rare)"),
    ("phi", "phase; electric potential; magnetic flux (upper case); work function"),
    ("chi", "susceptibility; chi-squared"),
    ("psi", "wave function; Psi(x, t) upper case"),
    ("omega", "angular frequency 2 pi f; ohms Omega (upper case); solid angle"),
]


def greek_alphabet():
    """Return [(name, lower Symbol, upper Symbol, typical use)] from Alpha to Omega (24 letters).

    SymPy pretty-prints these names as the Greek glyphs. ``lambda`` is a Python keyword, so it
    is created with ``sp.Symbol('lambda')`` rather than a variable named lambda.
    """
    return [(n, sp.Symbol(n), sp.Symbol(n.capitalize()), use) for n, use in _GREEK]


# ---------------------------------------------------------------------------------------------
# 2. Gates
# ---------------------------------------------------------------------------------------------
def _bit(v):
    if v not in (0, 1, True, False):
        raise ValueError(f"logic inputs must be 0 or 1, got {v!r}")
    return int(v)


def NOT(a):
    return 1 - _bit(a)


def AND(a, b):
    return _bit(a) & _bit(b)


def OR(a, b):
    return _bit(a) | _bit(b)


def NAND(a, b):
    return NOT(AND(a, b))


def NOR(a, b):
    return NOT(OR(a, b))


def XOR(a, b):
    return _bit(a) ^ _bit(b)


def XNOR(a, b):
    return NOT(XOR(a, b))


GATES = {"NOT": NOT, "AND": AND, "OR": OR, "NAND": NAND, "NOR": NOR, "XOR": XOR, "XNOR": XNOR}

A, B = sp.symbols("A B")
GATE_EXPR = {
    "NOT": sp.Not(A), "AND": A & B, "OR": A | B, "NAND": sp.Nand(A, B),
    "NOR": sp.Nor(A, B), "XOR": sp.Xor(A, B), "XNOR": sp.Not(sp.Xor(A, B)),
}

# Every gate rebuilt from NAND only.
NAND_ONLY = {
    "NOT": lambda a, b=0: NAND(a, a),
    "AND": lambda a, b: NAND(NAND(a, b), NAND(a, b)),
    "OR": lambda a, b: NAND(NAND(a, a), NAND(b, b)),
    "NAND": lambda a, b: NAND(a, b),
    "NOR": lambda a, b: NAND(NAND(NAND(a, a), NAND(b, b)), NAND(NAND(a, a), NAND(b, b))),
    "XOR": lambda a, b: NAND(NAND(a, NAND(a, b)), NAND(b, NAND(a, b))),
    "XNOR": lambda a, b: NAND(NAND(NAND(a, NAND(a, b)), NAND(b, NAND(a, b))),
                              NAND(NAND(a, NAND(a, b)), NAND(b, NAND(a, b)))),
}
NAND_COUNT = {"NOT": 1, "AND": 2, "OR": 3, "NAND": 1, "NOR": 4, "XOR": 4, "XNOR": 5}


def truth_table(gate):
    """Rows (inputs..., output) for a gate name from GATES."""
    if gate not in GATES:
        raise KeyError(f"unknown gate {gate!r}; choose from {list(GATES)}")
    if gate == "NOT":
        return [(a, NOT(a)) for a in (0, 1)]
    return [(a, b, GATES[gate](a, b)) for a, b in product((0, 1), repeat=2)]


def check_nand_universality():
    """True if every NAND-only build matches the real gate on all inputs."""
    for g, f in GATES.items():
        for a, b in product((0, 1), repeat=2):
            want = f(a) if g == "NOT" else f(a, b)
            if NAND_ONLY[g](a, b) != want:
                return False
    return True


# ---------------------------------------------------------------------------------------------
# 3. Multiplexer
# ---------------------------------------------------------------------------------------------
def mux(select, inputs):
    """2^k:1 multiplexer. ``select`` is a tuple of k bits, MSB first; returns inputs[index]."""
    sel = [_bit(s) for s in select]
    if len(inputs) != 2 ** len(sel):
        raise ValueError(f"a {len(sel)}-select MUX needs {2 ** len(sel)} inputs, got {len(inputs)}")
    idx = int("".join(map(str, sel)), 2) if sel else 0
    return _bit(inputs[idx])


def mux_expression(k):
    """SymPy Boolean expression of a 2^k:1 MUX: OR over i of (select == i) AND D_i."""
    if k < 1:
        raise ValueError("k must be >= 1")
    S = tuple(sp.Symbol(f"S{j}") for j in range(k - 1, -1, -1))     # MSB first
    D = tuple(sp.Symbol(f"D{i}") for i in range(2 ** k))
    terms = []
    for i in range(2 ** k):
        bits = format(i, f"0{k}b")
        lits = [s if bit == "1" else sp.Not(s) for s, bit in zip(S, bits)]
        terms.append(sp.And(*lits, D[i]))
    return sp.Or(*terms), S, D


def function_via_mux(minterms, n_vars):
    """Data inputs that make a 2^n:1 MUX (variables on the selects) implement sum m(minterms)."""
    _check_minterms(minterms, n_vars)
    return [1 if i in set(minterms) else 0 for i in range(2 ** n_vars)]


# ---------------------------------------------------------------------------------------------
# 4. Karnaugh maps
# ---------------------------------------------------------------------------------------------
def gray(n):
    """n-bit Gray code as strings: neighbours differ in exactly one bit."""
    if n < 0:
        raise ValueError("n must be >= 0")
    return [format(i ^ (i >> 1), f"0{n}b") for i in range(2 ** n)] if n else [""]


def _check_minterms(minterms, n_vars):
    if not 1 <= n_vars <= 6:
        raise ValueError("n_vars must be 1..6")
    bad = [m for m in minterms if not (isinstance(m, int) and 0 <= m < 2 ** n_vars)]
    if bad:
        raise ValueError(f"minterms out of range 0..{2 ** n_vars - 1}: {bad}")


def kmap(minterms, variables, dontcares=()):
    """K-map for 2-4 variables.

    Returns (grid, row_labels, col_labels, sop) where grid cells are 1, 0 or 'X', rows use the
    first half of ``variables`` and columns the rest, both in Gray order, and ``sop`` is the
    minimal sum of products from SymPy's SOPform (Quine-McCluskey).
    """
    names = [str(v) for v in variables]
    n = len(names)
    if not 2 <= n <= 4:
        raise ValueError("kmap draws 2-4 variables")
    _check_minterms(list(minterms) + list(dontcares), n)
    if set(minterms) & set(dontcares):
        raise ValueError("a cell cannot be both a minterm and a don't-care")
    nr = n // 2
    rows, cols = gray(nr), gray(n - nr)
    grid = []
    for r in rows:
        line = []
        for c in cols:
            m = int(r + c, 2)
            line.append(1 if m in minterms else "X" if m in dontcares else 0)
        grid.append(line)
    syms = sp.symbols(" ".join(names))
    sop = SOPform(list(syms), [list(map(int, format(m, f"0{n}b"))) for m in minterms],
                  [list(map(int, format(m, f"0{n}b"))) for m in dontcares])
    row_lab = ["".join(names[:nr]) + "=" + r for r in rows]
    col_lab = ["".join(names[nr:]) + "=" + c for c in cols]
    return grid, row_lab, col_lab, sop


def verify_sop(expr, minterms, variables, dontcares=()):
    """True if ``expr`` is 1 on every minterm and 0 on every other non-don't-care input."""
    syms = [sp.Symbol(str(v)) for v in variables]
    n = len(syms)
    for m in range(2 ** n):
        if m in dontcares:
            continue
        bits = format(m, f"0{n}b")
        val = bool(expr.subs({s: bit == "1" for s, bit in zip(syms, bits)}))
        if val != (m in minterms):
            return False
    return True


def literal_count(expr):
    """Number of literals in a SOP, the usual 'cost' a K-map minimizes."""
    return sum(1 for a in sp.preorder_traversal(expr) if isinstance(a, sp.Symbol))


# ---------------------------------------------------------------------------------------------
# 5. Power and energy
# ---------------------------------------------------------------------------------------------
POWER_DEFINITIONS = [
    ("rate of doing work", "P = dW/dt", "any energy transfer: lifting, heating, a battery draining"),
    ("force times velocity", "P = F . v", "mechanics: a motor pulling a load, drag on a car"),
    ("current times voltage", "P = I V  (= I^2 R = V^2 / R for a resistor)", "circuits: EEE 117"),
]


def _positive(**kw):
    for k, v in kw.items():
        if v is not None and not v > 0:
            raise ValueError(f"{k} must be positive, got {v!r}")


def power_rate_of_work(W, t):
    """Average power W/t (J, s -> W)."""
    _positive(t=t)
    return W / t


def power_force_velocity(F, v, angle_deg=0.0):
    """P = F v cos(angle) (N, m/s -> W)."""
    import math
    return F * v * math.cos(math.radians(angle_deg))


def power_electrical(I=None, V=None, R=None):
    """P from any two of I (A), V (V), R (ohm): IV, I^2 R or V^2/R."""
    given = sum(x is not None for x in (I, V, R))
    if given != 2:
        raise ValueError("give exactly two of I, V, R")
    _positive(R=R)
    if R is None:
        return I * V
    if V is None:
        return I ** 2 * R
    return V ** 2 / R


def energy_power_time(E=None, P=None, t=None):
    """E = P t. Give exactly two (J, W, s); returns dict with all three."""
    if sum(x is not None for x in (E, P, t)) != 2:
        raise ValueError("give exactly two of E, P, t")
    _positive(t=t, P=P)
    if E is None:
        E = P * t
    elif P is None:
        P = E / t
    else:
        t = E / P
    return {"E_J": E, "P_W": P, "t_s": t}


EPS0 = 8.8541878128e-12   # F/m
K_E = 8.9875517923e9      # N m^2 / C^2
C_LIGHT = 2.99792458e8    # m/s
G_EARTH = 9.80665         # m/s^2


def energy_capacitor_geometry(area_m2, gap_m, voltage_V, kappa=1.0):
    """Parallel plates: C = kappa eps0 A / d, U = C V^2 / 2. Energy from geometry (J)."""
    _positive(area_m2=area_m2, gap_m=gap_m, kappa=kappa)
    C = kappa * EPS0 * area_m2 / gap_m
    return 0.5 * C * voltage_V ** 2


def energy_from_mass(m_kg, v=0.0, h=0.0, rest=False):
    """K = m v^2/2 plus U = m g h; rest=True adds E0 = m c^2 (J)."""
    _positive(m_kg=m_kg)
    return 0.5 * m_kg * v ** 2 + m_kg * G_EARTH * h + (m_kg * C_LIGHT ** 2 if rest else 0.0)


def energy_from_charge(q_C, V=None, q2_C=None, r_m=None):
    """U = q V through a potential difference, or U = k q1 q2 / r for two point charges (J)."""
    if V is not None and (q2_C is None and r_m is None):
        return q_C * V
    if V is None and q2_C is not None and r_m is not None:
        _positive(r_m=r_m)
        return K_E * q_C * q2_C / r_m
    raise ValueError("give V, or both q2_C and r_m")


def check_power_units():
    """SymPy unit check that all three definitions of power come out in watts."""
    J, s, N, m, A_, V_ = u.joule, u.second, u.newton, u.meter, u.ampere, u.volt
    forms = {"W/t": J / s, "F v": N * m / s, "I V": A_ * V_, "I^2 R": A_ ** 2 * u.ohm,
             "V^2/R": V_ ** 2 / u.ohm}
    return {k: sp.simplify(u.convert_to(v, u.watt) / u.watt) == 1 for k, v in forms.items()}
