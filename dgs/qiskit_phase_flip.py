"""The phase-flip code in Qiskit: protecting the phase that dispersion moves around.

Dispersion is a KNOWN phase: e^{-i a k^2} on every frequency. Gerchberg-Saxton
(and the qubit version in projects/tdgsa_qis) undoes it because it is known.
Dephasing is an UNKNOWN phase: a random Z on each qubit. Nothing can undo it
after the fact, unless the information was spread over several qubits first.
That is the 3-qubit phase-flip code:

  * ENCODE   |psi> = a|0> + b|1>  ->  a|+++> + b|--->   (CNOT, CNOT, H H H)
  * NOISE    each data qubit: Z with probability p (Kraus sqrt(1-p) I, sqrt(p) Z)
  * DETECT   H H H turns Z errors into X errors; two ancillas record the
             parities q0+q1 and q1+q2 (the stabilizers X1X2, X2X3 of the code)
  * CORRECT  coherently: a multi-controlled X fires for each syndrome pattern.
             This is the deferred-measurement principle: controlled gates do the
             same job as "measure the ancillas, then fix", so the whole cycle is
             one unitary plus noise and can be simulated exactly.
  * DECODE   undo the CNOTs; qubit 0 holds the protected state.

Average fidelity over the six Pauli eigenstates is exact and closed-form:
    unencoded  1 - 2p/3        encoded  1 - 2 p_L / 3,   p_L = 3p^2 - 2p^3
(p_L is dgs.quantum_error_correction.logical_error(p, 3)). The code wins for
p < 1/2. A systematic (coherent) phase theta on every qubit is also corrected
to leading order: infidelity drops from ~theta^2 to ~theta^4.

Exact density-matrix simulation with qiskit.quantum_info. Education.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Kraus, Operator, Statevector, partial_trace, state_fidelity

N_DATA, N_ANC = 3, 2
PAULI_STATES = {
    "0": Statevector([1, 0]),
    "1": Statevector([0, 1]),
    "+": Statevector([1, 1] / np.sqrt(2)),
    "-": Statevector([1, -1] / np.sqrt(2)),
    "+i": Statevector([1, 1j] / np.sqrt(2)),
    "-i": Statevector([1, -1j] / np.sqrt(2)),
}


def encoder():
    """|psi>|00> on (q0, q1, q2) -> a|+++> + b|--->."""
    qc = QuantumCircuit(N_DATA + N_ANC, name="encode")
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.h([0, 1, 2])
    return qc


def syndrome_and_correct():
    """H frame, parity ancillas q3 = q0+q1, q4 = q1+q2, coherent correction, decode."""
    qc = QuantumCircuit(N_DATA + N_ANC, name="correct")
    qc.h([0, 1, 2])                       # Z errors -> X errors
    qc.cx(0, 3); qc.cx(1, 3)              # ancilla 3: parity of q0, q1
    qc.cx(1, 4); qc.cx(2, 4)              # ancilla 4: parity of q1, q2
    # ctrl_state is written q4 q3 (rightmost = first control)
    qc.mcx([3, 4], 0, ctrl_state="01")    # syndrome (1, 0): error on q0
    qc.mcx([3, 4], 1, ctrl_state="11")    # syndrome (1, 1): error on q1
    qc.mcx([3, 4], 2, ctrl_state="10")    # syndrome (0, 1): error on q2
    qc.cx(0, 2)                           # decode
    qc.cx(0, 1)
    return qc


def full_cycle_circuit():
    """Encoder, a placeholder barrier for the noise, then correction (for drawing)."""
    qc = encoder()
    qc.barrier(label="noise")
    return qc.compose(syndrome_and_correct())


def dephasing(p):
    """Phase-flip channel: Z with probability p."""
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    return Kraus([np.sqrt(1 - p) * np.eye(2), np.sqrt(p) * np.diag([1.0, -1.0])])


def coherent_phase(theta):
    """Systematic phase error exp(-i theta Z), like an uncompensated bit of dispersion."""
    return Operator(np.diag([np.exp(-1j * theta), np.exp(1j * theta)]))


def _prepare(psi):
    """psi on q0, |0> on q1..q4, as a 5-qubit density matrix (qubit 0 = least significant)."""
    zero = Statevector.from_label("0" * (N_DATA + N_ANC - 1))
    return DensityMatrix(zero.tensor(psi))


def protected_state(psi, error, encoded=True):
    """Output density matrix of the protected qubit after one noisy cycle.

    error: a channel/operator applied to each data qubit (Kraus or Operator).
    """
    if not encoded:
        return DensityMatrix(psi).evolve(error)
    dm = _prepare(psi).evolve(encoder())
    for q in range(N_DATA):
        dm = dm.evolve(error, qargs=[q])
    dm = dm.evolve(syndrome_and_correct())
    return partial_trace(dm, list(range(1, N_DATA + N_ANC)))


def average_fidelity(error, encoded=True):
    """Mean fidelity over the six Pauli eigenstates (equals the Haar average for qubit channels)."""
    return float(np.mean([state_fidelity(protected_state(s, error, encoded), s) for s in PAULI_STATES.values()]))


def average_fidelity_theory(p, encoded=True):
    p_L = 3 * p**2 - 2 * p**3
    return 1 - 2 * (p_L if encoded else p) / 3


def logical_error_rate_from_shots(p, shots, rng=None):
    """Prepare |0>, run a noisy encoded cycle, measure qubit 0 `shots` times.

    Each '1' is a logical failure. Returns (estimate, Wilson 95% interval).
    """
    rng = np.random.default_rng(rng)
    rho = protected_state(PAULI_STATES["0"], dephasing(p), encoded=True)
    p_fail = float(np.real(rho.data[1, 1]))
    k = rng.binomial(shots, p_fail)
    return k / shots, wilson_interval(k, shots)


def wilson_interval(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion (well-behaved at k = 0 or k = n)."""
    if n <= 0:
        raise ValueError("n must be positive")
    phat = k / n
    den = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / den
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / den
    lo = 0.0 if k == 0 else max(0.0, centre - half)       # exact endpoints; avoid 1e-18 rounding dust
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi
