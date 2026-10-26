"""1D quantum mechanics IS dispersion-assisted phase retrieval (the Pauli problem).

In plain words: a free particle's wavefunction spreads exactly like a light pulse in a
dispersive fiber (Knight Ch. 40; dgs.diffraction_dispersion). A detector only records
the probability density |psi(x, t)|^2 -- the phase is lost, just as a photodiode only
records |E(t)|^2. Pauli asked (1933) whether position and momentum densities determine
psi. This repo's question is the same one with two TIMES instead: do |psi(x, t1)|^2 and
|psi(x, t2)|^2 determine psi(x, 0)? Dispersion-assisted Gerchberg-Saxton (dgs.gs_core)
answers it numerically: free evolution for time t is the dispersion operator
        H(nu) = exp(i pi D nu^2),   D = -2 pi hbar t / (m dx^2)
with nu in cycles per sample. So gs_core.retrieve_phase(I1, I2, D1, D2) recovers a
wavefunction from two snapshots of its probability density.

What cannot be recovered is exactly what quantum mechanics says is unobservable or
symmetric:
  * global phase e^{i alpha} psi: same physics (gauge);
  * the time-reversed twin conj(psi): its densities at +t equal psi's at -t, so two
    measurements on the SAME side of t = 0 tell them apart.

The qubit version makes the "two dispersions" idea concrete. A Z measurement of
cos(theta/2)|0> + e^{i phi} sin(theta/2)|1> never sees phi; measuring after a known unitary
(H gives the X basis) gives cos(phi) -- phi up to sign, the twin ambiguity; a second
unitary (the Y basis) fixes the sign. Known unitary + intensity measurement, twice: the
same recipe as D1, D2.

Computer-engineering view: each GS iteration is 4 FFTs + 4 elementwise multiplies + 2
magnitude replacements, O(N log N); the detector is an ADC, so the number of bits sets
the achievable phase accuracy (adc_bit_sweep).

Units: hbar = m = dx = 1 unless given. NumPy; Qiskit optional for the qubit cross-check.
"""

import time as _time

import numpy as np

from dgs import gs_core
from dgs.phase_retrieval_ambiguities import global_phase_align, phase_invariant_distance


# ---------------------------------------------------------------- the dictionary: time <-> dispersion

def D_from_time(t, hbar=1.0, m=1.0, dx=1.0):
    """Normalized dispersion that equals free evolution for time t: D = -2 pi hbar t / (m dx^2)."""
    if m <= 0 or dx <= 0 or hbar <= 0:
        raise ValueError("hbar, m, dx must be positive")
    return -2 * np.pi * hbar * np.asarray(t, float) / (m * dx ** 2)


def time_from_D(D, hbar=1.0, m=1.0, dx=1.0):
    """Inverse map: t = -D m dx^2 / (2 pi hbar)."""
    if m <= 0 or dx <= 0 or hbar <= 0:
        raise ValueError("hbar, m, dx must be positive")
    return -np.asarray(D, float) * m * dx ** 2 / (2 * np.pi * hbar)


def free_evolve(psi, t, hbar=1.0, m=1.0, dx=1.0):
    """psi(x, t) for a free particle, by the dispersion operator of gs_core (periodic grid)."""
    return gs_core.disperse(np.asarray(psi, complex), float(D_from_time(t, hbar, m, dx)))


def gaussian_packet(x, s0, k0=0.0, x0=0.0):
    """Normalized packet (pi s0^2)^{-1/4} exp(-(x-x0)^2/(2 s0^2)) exp(i k0 x)."""
    if s0 <= 0:
        raise ValueError(f"s0 must be positive, got {s0}")
    x = np.asarray(x, float)
    return (np.pi * s0 ** 2) ** -0.25 * np.exp(-(x - x0) ** 2 / (2 * s0 ** 2) + 1j * k0 * x)


def packet_width(s0, t, hbar=1.0, m=1.0):
    """Density 1/e half-width s(t) = s0 sqrt(1 + (hbar t / m s0^2)^2) (same law as a dispersed pulse)."""
    return s0 * np.sqrt(1 + (hbar * np.asarray(t, float) / (m * s0 ** 2)) ** 2)


def density_moments(x, psi, dx=1.0):
    """(norm, <x>, density half-width sqrt(2) * rms) of |psi|^2 on the grid."""
    P = np.abs(psi) ** 2
    nrm = P.sum() * dx
    mean = (P * x).sum() * dx / nrm
    var = (P * (x - mean) ** 2).sum() * dx / nrm
    return float(nrm), float(mean), float(np.sqrt(2 * var))


# ---------------------------------------------------------------- the Pauli problem with two times

def quantize(I, bits, full_scale=None):
    """Uniform unipolar ADC: round I / full_scale to 2^bits - 1 levels (bits=None: no quantization)."""
    I = np.asarray(I, float)
    if bits is None:
        return I.copy()
    if int(bits) != bits or bits < 1:
        raise ValueError(f"bits must be a positive integer, got {bits}")
    fs = float(I.max() if full_scale is None else full_scale)
    if fs <= 0:
        raise ValueError("full_scale must be positive")
    levels = 2 ** int(bits) - 1
    return np.round(np.clip(I / fs, 0, 1) * levels) / levels * fs


def pauli_measurements(psi0, t1, t2, hbar=1.0, m=1.0, dx=1.0, bits=None):
    """Probability densities |psi(x, t1)|^2 and |psi(x, t2)|^2, optionally through a b-bit ADC
    (a shared full scale for both, as one detector would use)."""
    if t1 == t2:
        raise ValueError("t1 == t2 gives no measurement diversity")
    I1 = np.abs(free_evolve(psi0, t1, hbar, m, dx)) ** 2
    I2 = np.abs(free_evolve(psi0, t2, hbar, m, dx)) ** 2
    fs = max(I1.max(), I2.max())
    return quantize(I1, bits, fs), quantize(I2, bits, fs)


def pauli_retrieve(I1, I2, t1, t2, n_iter=100, unit_amplitude=True, hbar=1.0, m=1.0, dx=1.0):
    """Recover psi(x, 0) (up to global phase) with dispersion-assisted GS (dgs.gs_core).

    unit_amplitude=True adds the prior |psi(x, 0)| = const (a phase-only state, like a
    constant-envelope optical signal); returns (psi_rec, errors)."""
    D1, D2 = float(D_from_time(t1, hbar, m, dx)), float(D_from_time(t2, hbar, m, dx))
    phi, errors = gs_core.retrieve_phase(I1, I2, D1, D2, n_iter=n_iter, unit_amplitude=unit_amplitude)
    return np.exp(1j * phi), errors


def relative_error(psi_true, psi_rec):
    """Global-phase-invariant relative error min_alpha ||psi - e^{i alpha} psi_rec|| / ||psi||."""
    psi_true = np.asarray(psi_true, complex)
    return phase_invariant_distance(psi_true, psi_rec) / float(np.linalg.norm(psi_true))


def phase_rms_error(psi_true, psi_rec):
    """RMS phase error (rad) after removing the global phase."""
    aligned = global_phase_align(psi_true, psi_rec)
    return float(np.sqrt(np.mean(np.angle(aligned * np.conj(psi_true)) ** 2)))


def phase_only_state(n=512, n_harmonics=16, seed=0):
    """Deterministic smooth phase-only wavefunction exp(i phi(x)), |phi| <= pi (the gs_core test signal)."""
    d = gs_core.make_qpsk_measurements(n_symbols=n // 8, sps=8, snr_db=300.0, rng_seed=seed)
    return np.exp(1j * d["phi_true"])


def time_reversal_check(psi, t, hbar=1.0, m=1.0, dx=1.0):
    """max | |conj(psi) evolved to t|^2 - |psi evolved to -t|^2 |: zero, the twin ambiguity's origin."""
    a = np.abs(free_evolve(np.conj(psi), t, hbar, m, dx)) ** 2
    b = np.abs(free_evolve(psi, -t, hbar, m, dx)) ** 2
    return float(np.max(np.abs(a - b)))


def measurement_mismatch(psi_candidate, I1, I2, t1, t2, hbar=1.0, m=1.0, dx=1.0):
    """RMS mismatch between a candidate's predicted densities and the measured ones."""
    J1, J2 = pauli_measurements(psi_candidate, t1, t2, hbar, m, dx)
    return float(np.sqrt(np.mean((J1 - I1) ** 2) + np.mean((J2 - I2) ** 2)))


# ---------------------------------------------------------------- qubit analog: basis changes = dispersions

def qubit_state(theta, phi):
    """cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>."""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], complex)


_H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
_SDG = np.diag([1, -1j])
BASIS_UNITARY = {"Z": np.eye(2, dtype=complex), "X": _H, "Y": _H @ _SDG}


def basis_probability0(psi, basis):
    """P(0) after the known unitary that rotates `basis` onto Z: the 'intensity after dispersion'."""
    if basis not in BASIS_UNITARY:
        raise ValueError(f"basis must be one of {list(BASIS_UNITARY)}")
    amp = BASIS_UNITARY[basis] @ np.asarray(psi, complex)
    return float(abs(amp[0]) ** 2)


def phase_from_x_only(p0x, theta=np.pi / 2):
    """phi from the X basis alone: cos(phi) = (2 p0x - 1)/sin(theta) -> returns BOTH candidates +-phi."""
    c = np.clip((2 * p0x - 1) / np.sin(theta), -1, 1)
    return float(np.arccos(c)), float(-np.arccos(c))


def phase_from_xy(p0x, p0y):
    """phi = atan2(<Y>, <X>) with <X> = 2 p0x - 1, <Y> = 2 p0y - 1: the sign is fixed."""
    return float(np.arctan2(2 * p0y - 1, 2 * p0x - 1))


def qiskit_expectations(theta, phi):   # pragma: no cover - optional dependency
    """(<X>, <Y>, <Z>) from qiskit's Statevector, or None if qiskit is not installed."""
    try:
        from qiskit.quantum_info import Statevector, SparsePauliOp
    except ImportError:
        return None
    sv = Statevector(qubit_state(theta, phi))
    return tuple(float(np.real(sv.expectation_value(SparsePauliOp(p)))) for p in "XYZ")


# ---------------------------------------------------------------- computer-engineering view

def gs_flops_per_iteration(n):
    """Operation-count ESTIMATE per GS iteration: 4 complex FFTs at 5 N log2 N flops each,
    4 complex multiplies by H (6 N each), 2 magnitude replacements (~10 N each: abs, sqrt,
    divide), 2 unit-amplitude normalizations (~6 N each). A model, not a measurement."""
    if n < 2:
        raise ValueError("n must be >= 2")
    return int(4 * 5 * n * np.log2(n) + 4 * 6 * n + 2 * 10 * n + 2 * 6 * n)


def time_gs_iterations(n=1024, n_iter=50, seed=0):
    """Measured wall time per GS iteration on this CPU (best of 3), for the given size."""
    psi = phase_only_state(n, seed=seed)
    I1, I2 = pauli_measurements(psi, 800.0, 920.0)
    best = np.inf
    for _ in range(3):
        t0 = _time.perf_counter()
        pauli_retrieve(I1, I2, 800.0, 920.0, n_iter=n_iter)
        best = min(best, _time.perf_counter() - t0)
    return best / n_iter


def adc_bit_sweep(bits_list=(4, 6, 8, 10, 12, None), n=512, t1=800.0, t2=920.0, n_iter=150, seed=0):
    """Phase RMS error of the retrieved wavefunction vs ADC resolution (None = ideal detector)."""
    psi = phase_only_state(n, seed=seed)
    out = []
    for b in bits_list:
        I1, I2 = pauli_measurements(psi, t1, t2, bits=b)
        rec, _ = pauli_retrieve(I1, I2, t1, t2, n_iter=n_iter)
        out.append((b, phase_rms_error(psi, rec), relative_error(psi, rec)))
    return out
