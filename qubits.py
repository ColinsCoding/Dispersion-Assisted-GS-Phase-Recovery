"""Qubits and quantum gates -- the reversible, unitary cousin of digital_logic.

A classical FSM (digital_logic.py) hops between definite states on each clock
edge; the transition can throw information away (AND is irreversible). A quantum
circuit is the same idea made *unitary*: the state is a complex amplitude vector
of length 2^n, and every gate is a reversible, norm-preserving rotation of it.
CNOT is literally a reversible XOR; H builds superposition; H then CNOT builds
entanglement (a Bell pair).

  state |psi> in C^(2^n),  gate U unitary,  |psi> -> U|psi>,  measure -> |amp|^2.

NumPy only (exact state-vector simulation, fine up to ~12 qubits). Education.
"""

import numpy as np

# single-qubit gates (2x2 unitaries)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)            # quantum NOT
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)   # superposition
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def ket(bitstring):
    """Computational basis state |bitstring>, e.g. ket('01') in a 2-qubit space."""
    if any(c not in "01" for c in bitstring) or not bitstring:
        raise ValueError("bitstring must be non-empty 0/1")
    n = len(bitstring)
    v = np.zeros(2**n, dtype=complex)
    v[int(bitstring, 2)] = 1.0
    return v


def _embed(U, q, n):
    """Embed a 1-qubit gate U acting on qubit q into the n-qubit space (q=0 MSB)."""
    op = np.array([[1]], dtype=complex)
    for i in range(n):
        op = np.kron(op, U if i == q else I2)
    return op


def apply_1q(state, U, q):
    """Apply single-qubit gate U to qubit q of an n-qubit state vector."""
    n = int(round(np.log2(len(state))))
    if not 0 <= q < n:
        raise ValueError("qubit index out of range")
    return _embed(U, q, n) @ state


def cnot(state, control, target):
    """Controlled-NOT: flip `target` iff `control` is |1> (a reversible XOR)."""
    n = int(round(np.log2(len(state))))
    if control == target or not (0 <= control < n and 0 <= target < n):
        raise ValueError("distinct, in-range control/target required")
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    term0 = term1 = np.array([[1]], dtype=complex)
    for i in range(n):
        term0 = np.kron(term0, P0 if i == control else I2)
        term1 = np.kron(term1, P1 if i == control else (X if i == target else I2))
    return (term0 + term1) @ state


def measure_probs(state):
    """Born rule: probability of each computational-basis outcome, |amp|^2."""
    p = np.abs(state)**2
    return p / p.sum()


def sample(state, shots=1000, rng=None):
    """Sample measurement outcomes; returns {bitstring: count}."""
    rng = rng or np.random.default_rng()
    n = int(round(np.log2(len(state))))
    idx = rng.choice(len(state), size=shots, p=measure_probs(state))
    out = {}
    for i in idx:
        out[format(i, f"0{n}b")] = out.get(format(i, f"0{n}b"), 0) + 1
    return out


def bell_state():
    """The Bell pair (|00>+|11>)/sqrt(2): H on qubit 0, then CNOT(0->1)."""
    return cnot(apply_1q(ket("00"), H, 0), 0, 1)


# ── the algebra tools: observables, expectation values, the Bloch vector ──
def is_hermitian(A):
    """A = A^dagger -- the matrices that are physical observables (real eigenvalues)."""
    A = np.asarray(A)
    return np.allclose(A, A.conj().T)


def is_unitary(U):
    """U U^dagger = I -- the reversible, norm-preserving gates."""
    U = np.asarray(U)
    return np.allclose(U @ U.conj().T, np.eye(len(U)))


def expectation(state, operator):
    """Expectation value <psi|A|psi>. Real when A is Hermitian (a measurable average).

    This is the inner product at the heart of quantum readout: the average value
    you'd measure for observable A over many copies of the state."""
    state = np.asarray(state, dtype=complex)
    return complex(np.vdot(state, np.asarray(operator, dtype=complex) @ state))


def bloch_vector(state):
    """(<X>, <Y>, <Z>) for a single qubit -- its point on the Bloch sphere.

    The complete linear-algebra description of a pure qubit; |0>->(0,0,1),
    |1>->(0,0,-1), |+>->(1,0,0). Length 1 for a pure state. This is what quantum
    state tomography measures -- civilian quantum sensing in three numbers."""
    if len(state) != 2:
        raise ValueError("bloch_vector is for a single qubit (length-2 state)")
    return np.array([expectation(state, X).real,
                     expectation(state, Y).real,
                     expectation(state, Z).real])


if __name__ == "__main__":
    psi = bell_state()
    print("Bell state amplitudes:", np.round(psi, 3))
    print("measure probabilities:", np.round(measure_probs(psi), 3))
    print("1000 shots:", sample(psi, 1000, np.random.default_rng(0)))
