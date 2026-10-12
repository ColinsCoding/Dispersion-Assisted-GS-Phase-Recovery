"""Polarization IS a qubit: wave plates in Dirac notation (SymPy).

Dictionary (consistent with the CODE of dgs.jones_calculus):
    |H> = |0> = (1, 0)        |V> = |1> = (0, 1)
    |D> = (|H> + |V>)/sqrt2   |A> = (|H> - |V>)/sqrt2
    |+i> = (|H> + i|V>)/sqrt2 |-i> = (|H> - i|V>)/sqrt2
  Circular states are named by their Bloch/Poincare axis (S3 = +1 or -1) rather
  than "right/left": which one is "right-handed" depends on whether you look
  toward or away from the source, and textbooks disagree.

Retarder with fast axis at theta:  W(delta, theta) = R(-theta) diag(1, e^{i delta}) R(theta)
    QWP(0) = diag(1, i)  = the quantum PHASE gate S
    HWP(0) = diag(1, -1) = Pauli Z
    HWP(22.5 deg) = (1/sqrt2) [[1, 1], [1, -1]] = the HADAMARD gate
Stokes vector = Bloch vector:  (S1, S2, S3) = (<Z>, <X>, <Y>).
Malus's law = Born rule:  P = |<theta|psi>|^2 = cos^2(theta) for psi = H.
Calc 3: Pancharatnam-Berry phase of a closed geodesic loop on the sphere = +-Omega/2
(sign = direction of travel), Omega the enclosed solid angle (a surface integral).

NOTE: the docstring of dgs.jones_calculus says QWP = diag(1, -i) and calls
(1, -i)/sqrt2 the S3 = +1 pole; its code actually gives diag(1, +i) and S3 = -1
for that state. This module follows the code, and tests pin the behaviour.
"""
import numpy as np
import sympy as sp

th = sp.Symbol("theta", real=True)

KETS = {
    "H": sp.Matrix([1, 0]), "V": sp.Matrix([0, 1]),
    "D": sp.Matrix([1, 1]) / sp.sqrt(2), "A": sp.Matrix([1, -1]) / sp.sqrt(2),
    "+i": sp.Matrix([1, sp.I]) / sp.sqrt(2), "-i": sp.Matrix([1, -sp.I]) / sp.sqrt(2),
}
X = sp.Matrix([[0, 1], [1, 0]])
Y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
Z = sp.Matrix([[1, 0], [0, -1]])
S_GATE = sp.Matrix([[1, 0], [0, sp.I]])
HADAMARD = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)


def rotation(angle):
    return sp.Matrix([[sp.cos(angle), sp.sin(angle)], [-sp.sin(angle), sp.cos(angle)]])


def retarder(delta, angle=0):
    """W = R(-angle) diag(1, e^{i delta}) R(angle) -- same convention as jones_calculus code."""
    return sp.simplify(rotation(-angle) * sp.diag(1, sp.exp(sp.I * delta)) * rotation(angle))


def qwp(angle=0):
    return retarder(sp.pi / 2, angle)


def hwp(angle=0):
    return retarder(sp.pi, angle)


def braket(a, b):
    """<a|b> = a^dagger b."""
    return sp.simplify((a.H * b)[0])


def equal_up_to_phase(u, v):
    """True if u = e^{i phi} v for some phi (vectors or matrices)."""
    u, v = sp.Matrix(u), sp.Matrix(v)
    k = next(i for i in range(len(v)) if sp.simplify(v[i]) != 0)
    ratio = sp.simplify(u[k] / v[k])
    return bool(sp.simplify(sp.Abs(ratio) - 1) == 0 and sp.simplify(u - ratio * v) == sp.zeros(*u.shape))


def is_unitary(U):
    return sp.simplify(U.H * U - sp.eye(U.shape[0])) == sp.zeros(*U.shape)


def bloch_vector(psi):
    """(<X>, <Y>, <Z>) and the Stokes mapping (S1, S2, S3) = (<Z>, <X>, <Y>)."""
    ev = lambda P: sp.simplify((psi.H * P * psi)[0])
    x, y, z = ev(X), ev(Y), ev(Z)
    return {"bloch": (x, y, z), "stokes": (z, x, y)}


def malus_born(angle):
    """|<theta|H>|^2 with |theta> = cos(theta)|H> + sin(theta)|V>."""
    ket_t = sp.Matrix([sp.cos(angle), sp.sin(angle)])
    return sp.simplify(sp.Abs(braket(ket_t, KETS["H"])) ** 2)


# ── Calc 3: solid angle and the geometric (Pancharatnam-Berry) phase ─

def solid_angle_octant():
    """Surface integral of the octant x, y, z > 0 of the unit sphere: int sin(t) dt dp."""
    t, p = sp.symbols("t p", positive=True)
    return sp.integrate(sp.integrate(sp.sin(t), (t, 0, sp.pi / 2)), (p, 0, sp.pi / 2))


def pancharatnam_phase(states):
    """Geometric phase of the closed loop psi_1 -> ... -> psi_n -> psi_1:
    gamma = arg(<1|2><2|3>...<n|1>), gauge-invariant."""
    vs = [np.asarray(s, complex).ravel() for s in states]
    prod = 1.0 + 0j
    for i in range(len(vs)):
        prod *= np.vdot(vs[i], vs[(i + 1) % len(vs)])
    return float(np.angle(prod))


def numeric(name):
    return np.array(KETS[name].evalf(), dtype=complex).ravel()
