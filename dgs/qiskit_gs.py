"""
Dispersion-assisted Gerchberg-Saxton phase retrieval, with the dispersion built as a Qiskit circuit.

In plain words
--------------
A photodetector records only intensity |E|^2, so the phase of the field E is lost. If the field
is also recorded after two different, known dispersions, Gerchberg-Saxton (GS) can recover the
phase: go back and forth between the two recordings, each time keeping the measured intensity
and updating the phase.

Dispersion is a quadratic phase in the frequency domain. In the convention of ``dgs/gs_core.py``:

    E_out = F^-1  diag(exp(i pi D nu^2))  F  E,        nu = k/N, the normalized frequency

With the N = 2^n samples of E stored as the amplitudes of an n-qubit state, the same operation is
a quantum circuit: inverse QFT, a diagonal phase gate, QFT. Measuring the qubits returns sample k
with probability |E_out[k]|^2 / ||E||^2, which is the normalized detector intensity. So:

* ``dispersion_circuit`` builds the dispersion as a Qiskit circuit (checked against gs_core),
* ``measure_intensity`` produces the two "detector" recordings by sampling that circuit with a
  finite number of shots, or exactly from the statevector,
* ``gs_retrieve`` runs the classical GS loop with the circuits' unitaries (``Operator``).

What this is not: a quantum speedup. Loading N amplitudes and reading N intensities each cost at
least order N operations, which cancels the QFT's advantage. This is a faithful simulation that
shows the optics and the quantum-circuit language are the same linear algebra.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, QFTGate, StatePreparation
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector


def spectral_phase(N: int, D: float) -> np.ndarray:
    """exp(i pi D nu^2) on NumPy's FFT bins, nu = fftfreq(N) (same as gs_core.disperse)."""
    nu = np.fft.fftfreq(N)
    return np.exp(1j * np.pi * D * nu**2)


def dispersion_numpy(E: np.ndarray, D: float) -> np.ndarray:
    """Reference model: the same dispersion with NumPy's FFT."""
    return np.fft.ifft(np.fft.fft(E) * spectral_phase(len(E), D))


def dispersion_circuit(n_qubits: int, D: float) -> QuantumCircuit:
    """Dispersion D as a circuit on n qubits: inverse QFT, diagonal phase, QFT.

    Qiskit's QFT uses exp(+2 pi i jk/N), the opposite sign from NumPy's forward FFT, so NumPy's
    F is Qiskit's inverse QFT: the order is QFT^dagger, then the phase, then QFT.
    """
    qc = QuantumCircuit(n_qubits, name=f'disp({D:g})')
    qc.append(QFTGate(n_qubits).inverse(), range(n_qubits))
    qc.append(DiagonalGate(list(spectral_phase(2**n_qubits, D))), range(n_qubits))
    qc.append(QFTGate(n_qubits), range(n_qubits))
    return qc


def smooth_phase_field(n_qubits: int, n_harmonics: int = 8, seed: int = 0):
    """Unit-amplitude test field exp(i phi(t)) with a smooth, band-limited random phase in [-pi, pi]
    (the same kind of signal gs_core uses). Returns (t, E, phi)."""
    N = 2**n_qubits
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N)
    phi = sum(rng.uniform(-1, 1) * np.sin(2 * np.pi * k * t + rng.uniform(0, 2 * np.pi))
              for k in range(1, n_harmonics + 1))
    phi = phi / np.max(np.abs(phi)) * np.pi
    return t, np.exp(1j * phi), phi


def measure_intensity(E: np.ndarray, disp: QuantumCircuit, shots: int | None = None, seed: int = 0) -> np.ndarray:
    """Detector intensity after dispersion, from the circuit.

    Loads E (normalized) into the qubits, applies the dispersion circuit, then reads the exact
    probabilities (shots=None) or samples the qubits `shots` times. The result is scaled back to
    the field's energy, so it can be compared with |E_out|^2.
    """
    n = disp.num_qubits
    norm2 = float(np.vdot(E, E).real)
    qc = QuantumCircuit(n)
    qc.append(StatePreparation(E / np.sqrt(norm2)), range(n))
    qc.compose(disp, inplace=True)
    if shots is None:
        probs = Statevector(qc).probabilities()
    else:
        qc.measure_all()
        counts = StatevectorSampler(seed=seed).run([qc], shots=shots).result()[0].data.meas.get_int_counts()
        probs = np.zeros(2**n)
        for idx, cnt in counts.items():
            probs[idx] = cnt / shots
    return probs * norm2


def gs_retrieve(I1, I2, U1, U2, n_iter: int = 100, unit_amplitude: bool = True):
    """Two-plane Gerchberg-Saxton with the dispersion unitaries U1, U2 (N x N arrays).

    Start: undo dispersion 1 on sqrt(I1) with zero phase (as gs_core does). Each iteration:
    go to plane 1, impose sqrt(I1), come back; go to plane 2, impose sqrt(I2), come back;
    if unit_amplitude, impose |E| = 1. Returns the recovered field and the per-iteration
    RMS intensity misfit at plane 2.
    """
    a1, a2 = np.sqrt(np.maximum(I1, 0)), np.sqrt(np.maximum(I2, 0))
    U1h, U2h = U1.conj().T, U2.conj().T
    E = U1h @ a1.astype(complex)
    misfit = []
    for _ in range(n_iter):
        for U, Uh, a in ((U1, U1h, a1), (U2, U2h, a2)):
            E = Uh @ (a * np.exp(1j * np.angle(U @ E)))
            if unit_amplitude:
                E = np.exp(1j * np.angle(E))
        misfit.append(float(np.sqrt(np.mean((np.abs(U2 @ E)**2 - a2**2)**2))))
    return E, np.array(misfit)


def align_rms(phi_rec: np.ndarray, phi_true: np.ndarray):
    """RMS phase error after removing what intensity data cannot see: a constant offset
    (and, for safety, a sign flip). Returns (rms, aligned phase)."""
    best = None
    for s in (1, -1):
        off = np.angle(np.sum(np.exp(1j * (phi_true - s * phi_rec))))
        aligned = np.angle(np.exp(1j * (s * phi_rec + off)))
        rms = float(np.sqrt(np.mean(np.angle(np.exp(1j * (phi_true - aligned)))**2)))
        if best is None or rms < best[0]:
            best = (rms, aligned)
    return best
