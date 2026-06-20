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


# ── the quantum Fourier transform: the FFT on amplitudes ────────────
def controlled_phase(state, qa, qb, phi):
    """Apply phase e^{i phi} to the |..1..1..> component (qubits qa AND qb both 1).
    A diagonal, symmetric two-qubit gate -- the building block of the QFT."""
    n = int(round(np.log2(len(state))))
    idx = np.arange(len(state))
    both = ((idx >> (n - 1 - qa)) & 1) & ((idx >> (n - 1 - qb)) & 1)
    return state * np.where(both == 1, np.exp(1j * phi), 1.0)


def swap(state, a, b):
    """Swap two qubits (three CNOTs)."""
    return cnot(cnot(cnot(state, a, b), b, a), a, b)


def qft_matrix(n):
    """The QFT unitary on n qubits: W[j,k] = exp(2 pi i j k / N)/sqrt(N), N=2^n.
    It's the DFT matrix of fourier_tools, normalized to be unitary."""
    N = 2**n
    j, k = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return np.exp(2j * np.pi * j * k / N) / np.sqrt(N)


def qft(state):
    """Quantum Fourier transform via the matrix. Equals sqrt(N) * numpy.fft.ifft."""
    n = int(round(np.log2(len(state))))
    return qft_matrix(n) @ np.asarray(state, dtype=complex)


def inverse_qft(state):
    """Inverse QFT (the conjugate-transpose unitary)."""
    n = int(round(np.log2(len(state))))
    return qft_matrix(n).conj().T @ np.asarray(state, dtype=complex)


def qft_circuit(state):
    """QFT built as an actual gate circuit: per qubit, an H then controlled phase
    rotations from the lower qubits, finished by reversing the qubit order.

    Uses n Hadamards + n(n-1)/2 controlled-phase gates = O(n^2) gates -- vs the
    classical FFT's O(N log N) = O(n 2^n). Returns the same state as qft()."""
    s = np.asarray(state, dtype=complex).copy()
    n = int(round(np.log2(len(s))))
    for j in range(n):
        s = apply_1q(s, H, j)
        for k in range(j + 1, n):
            s = controlled_phase(s, j, k, 2 * np.pi / 2**(k - j + 1))
    for i in range(n // 2):
        s = swap(s, i, n - 1 - i)
    return s


# ── phase estimation: a quantum ADC that measures an eigen-phase ────
def phase_gate(angle):
    """Single-qubit phase gate diag(1, e^{i angle})."""
    return np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=complex)


def phase_estimation(phi, n_counting):
    """Quantum phase estimation: read the eigen-phase phi in [0,1) into n qubits.

    A unitary U with U|psi> = e^{2 pi i phi}|psi> kicks the phase back onto the
    counting register; after the inverse QFT the register holds (close to) the
    integer m = phi * 2^n. So this is an analog-to-digital converter for *phase*:
    a continuous phi goes in, its n-bit binary expansion comes out -- the quantum
    sibling of the carrier-less receiver's job (recover the phase) and of the
    lab ADC (digitize a continuous quantity). Returns the counting-register state.
    """
    if n_counting < 1:
        raise ValueError("n_counting >= 1")
    state = ket("0" * n_counting)
    for j in range(n_counting):                       # uniform superposition
        state = apply_1q(state, H, j)
    for j in range(n_counting):                       # phase kickback, qubit 0 = MSB
        state = apply_1q(state, phase_gate(2 * np.pi * phi * 2**(n_counting - 1 - j)), j)
    return inverse_qft(state)


def estimate_phase(phi, n_counting):
    """Run QPE and return (phi_hat, probability) for the most likely readout.
    phi_hat = m/2^n is exact when phi is an n-bit fraction."""
    probs = measure_probs(phase_estimation(phi, n_counting))
    m = int(np.argmax(probs))
    return m / 2**n_counting, float(probs[m])


if __name__ == "__main__":
    psi = bell_state()
    print("Bell state amplitudes:", np.round(psi, 3))
    print("measure probabilities:", np.round(measure_probs(psi), 3))
    print("1000 shots:", sample(psi, 1000, np.random.default_rng(0)))
