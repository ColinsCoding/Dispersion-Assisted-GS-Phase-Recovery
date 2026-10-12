"""Why SymPy quantum-information calculations get slow -- and how to fix them.
Plus dimensional checks of QIS formulas and reversible logic as Boolean algebra.

1. SCALING. n qubits = vectors of length 2^n, gates as 2^n x 2^n matrices.
   Building the full circuit matrix symbolically costs ~ (2^n)^3 symbolic
   multiplications per matrix product, and every entry is an expression that
   grows with the number of symbolic angles -- then simplify() walks all of it.
   Measured, not guessed: time and expression size (count_ops) vs n.

2. FIXES (each measured):
   a. apply gates to the STATE, never build the full matrix: O(2^n) per gate
      instead of O(4^n)/O(8^n);
   b. keep symbols only where needed; lambdify the final expression and
      evaluate numerically;
   c. cse() -- common-subexpression elimination -- shrinks repeated work;
   d. targeted simplification (trigsimp / expand_trig) instead of simplify();
   e. past ~10 qubits, use NumPy (or a dedicated simulator) -- symbolic is for
      insight on small systems, not for scale.

3. UNITS: E = hbar omega, Rabi frequency Omega = d E / hbar, T2 and gate counts,
   checked with dgs.units_dimensions (exact dimension vectors) and
   sympy.physics.units (conversions).

4. LOGIC: CNOT and Toffoli as Boolean functions (sympy.logic) and as
   permutation matrices; Toffoli with target 1 is NAND, so reversible logic is
   universal; satisfiable() finds inputs that produce a wanted output.
"""
import itertools
import time

import numpy as np
import sympy as sp
from sympy.logic.boolalg import Xor, And, Not
from sympy.logic.inference import satisfiable

from dgs.units_dimensions import dim, dim_mul, dim_div, check_homogeneous, dim_str, DIMS

# ── gates ────────────────────────────────────────────────────────────

def ry_sym(theta):
    return sp.Matrix([[sp.cos(theta / 2), -sp.sin(theta / 2)], [sp.sin(theta / 2), sp.cos(theta / 2)]])


CNOT_SYM = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def _kron_sym(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = sp.kronecker_product(out, m)
    return out


def _cnot_chain_full(n):
    """Full 2^n matrix of CNOT(0,1) CNOT(1,2) ... CNOT(n-2,n-1), symbolic."""
    U = sp.eye(2 ** n)
    for k in range(n - 1):
        U = _kron_sym(sp.eye(2 ** k), CNOT_SYM, sp.eye(2 ** (n - k - 2))) * U
    return U


def circuit_full_matrix(n, thetas):
    """Naive way: build the whole unitary (RY on every qubit, then a CNOT chain)."""
    layer = _kron_sym(*[ry_sym(t) for t in thetas[:n]])
    return _cnot_chain_full(n) * layer


def circuit_full_state(n, thetas):
    """Naive: full matrix times |0...0>."""
    e0 = sp.zeros(2 ** n, 1)
    e0[0] = 1
    return circuit_full_matrix(n, thetas) * e0


def circuit_state_vector(n, thetas):
    """Better: product state from single-qubit rotations, then apply CNOTs as
    index permutations on the 2^n amplitudes (no 2^n x 2^n matrix at all)."""
    psi = [sp.Integer(1)]
    for t in thetas[:n]:
        a = [sp.cos(t / 2), sp.sin(t / 2)]               # RY(t)|0>
        psi = [p * q for p in psi for q in a]              # qubit 0 is the most significant bit
    psi = list(psi)
    for k in range(n - 1):                                 # CNOT(control k, target k+1)
        cbit, tbit = n - 1 - k, n - 2 - k
        new = psi[:]
        for idx in range(2 ** n):
            if (idx >> cbit) & 1:
                new[idx] = psi[idx ^ (1 << tbit)]
        psi = new
    return sp.Matrix(psi)


def time_call(f, *a, repeat=1):
    best = np.inf
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = f(*a)
        best = min(best, time.perf_counter() - t0)
    return best, out


def expression_size(M):
    return int(sum(sp.count_ops(e) for e in M))


def numeric_state(n, thetas):
    """Same circuit in NumPy (numbers, not symbols)."""
    psi = np.array([1.0])
    for t in thetas[:n]:
        psi = np.kron(psi, [np.cos(t / 2), np.sin(t / 2)])
    for k in range(n - 1):
        cbit, tbit = n - 1 - k, n - 2 - k
        idx = np.arange(2 ** n)
        sel = (idx >> cbit) & 1 == 1
        new = psi.copy()
        new[sel] = psi[idx[sel] ^ (1 << tbit)]
        psi = new
    return psi


# ── fixes: lambdify, cse, targeted simplify ──────────────────────────

def purity_expression(n, thetas):
    """A scalar QIS quantity: sum |amp|^4 (inverse participation), symbolic."""
    psi = circuit_state_vector(n, thetas)
    return sum(a ** 4 for a in psi)


def cse_savings(expr):
    """(ops before, ops after cse) -- repeated sub-expressions computed once."""
    reps, red = sp.cse(expr)
    after = sum(sp.count_ops(r[1]) for r in reps) + sum(sp.count_ops(e) for e in red)
    return int(sp.count_ops(expr)), int(after)


# ── units ────────────────────────────────────────────────────────────

QIS_DIMS = {
    "hbar": DIMS["action"], "omega": DIMS["frequency"], "energy": DIMS["energy"],
    "dipole": dim_mul(DIMS["charge"], DIMS["length"]),
    "E_field": dim_div(DIMS["voltage"], DIMS["length"]),
    "time": DIMS["time"],
}


def check_qis_units():
    """E = hbar omega; Omega_Rabi = d E / hbar has dimension of frequency;
    Omega t is dimensionless (a rotation angle)."""
    E = dim_mul(QIS_DIMS["hbar"], QIS_DIMS["omega"])
    check_homogeneous(hbar_omega=E, energy=QIS_DIMS["energy"])
    rabi = dim_div(dim_mul(QIS_DIMS["dipole"], QIS_DIMS["E_field"]), QIS_DIMS["hbar"])
    check_homogeneous(rabi=rabi, frequency=DIMS["frequency"])
    angle = dim_mul(rabi, QIS_DIMS["time"])
    return {"hbar*omega": dim_str(E), "Rabi d*E/hbar": dim_str(rabi), "Omega*t": dim_str(angle)}


def gates_within_coherence(T2_s, gate_time_s):
    """How many gates fit in one coherence time (dimensionless ratio)."""
    if T2_s <= 0 or gate_time_s <= 0:
        raise ValueError("times must be positive")
    return T2_s / gate_time_s


# ── logic ────────────────────────────────────────────────────────────

a, b, c = sp.symbols("a b c")


def cnot_logic():
    """CNOT: (a, b) -> (a, a XOR b)."""
    return (a, Xor(a, b))


def toffoli_logic():
    """Toffoli: (a, b, c) -> (a, b, c XOR (a AND b))."""
    return (a, b, Xor(c, And(a, b)))


def logic_to_permutation(outputs, inputs):
    """Truth table of a reversible gate as a permutation matrix (numpy int)."""
    n = len(inputs)
    P = np.zeros((2 ** n, 2 ** n), dtype=int)
    for bits in itertools.product((0, 1), repeat=n):
        sub = {s: bool(v) for s, v in zip(inputs, bits)}
        out = [int(bool(o.subs(sub))) for o in outputs]
        i = int("".join(map(str, bits)), 2)
        j = int("".join(map(str, out)), 2)
        P[j, i] = 1
    return P


def toffoli_is_nand():
    """With c = 1 the Toffoli target becomes NOT(a AND b): universal classical logic."""
    t = toffoli_logic()[2].subs(c, True)
    return sp.simplify_logic(t) == sp.simplify_logic(Not(And(a, b)))


def inputs_for_toffoli_output(target_value=True):
    """satisfiable(): all (a, b, c) giving the requested Toffoli target bit."""
    expr = toffoli_logic()[2] if target_value else Not(toffoli_logic()[2])
    return sorted([tuple(int(bool(m[s])) for s in (a, b, c)) for m in satisfiable(expr, all_models=True)])
