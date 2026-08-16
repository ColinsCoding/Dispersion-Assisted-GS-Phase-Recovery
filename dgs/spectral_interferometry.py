"""spectral_interferometry.py -- Hilbert-transform spectral fringe demodulation,
following Pu & Jalali, "Neural network enabled time stretch spectral regression,"
Opt. Express 29(13), 20786-20794 (2021) (UCLA, Jalali lab).

WHAT THE PAPER DOES, physically (Fig. 1 of the paper): a broadband pulse is split
into a signal arm and a reference arm (Mach-Zehnder interferometer); the reference
is delayed by tau (the "shear"). The two arms recombine and interfere, producing a
spectral fringe pattern

    S(omega) = |E_test(omega)|^2 + |E_ref(omega)|^2
               + 2 Re[ E_test(omega) * conj(E_ref(omega)) * exp(+i*omega*tau) ]

Recovering the signal's complex field E_test(omega) -- both magnitude and phase --
from S(omega) alone is the "spectral regression" problem. The paper's classical
baseline (section 2.4) is: filter out the slowly-varying background, then recover
the analytic signal via a HILBERT TRANSFORM, then strip the known tau-carrier.
This module implements exactly that classical baseline, reusing
dgs/causality.py's hilbert_transform -- the SAME FFT-based analytic-signal
machinery already in this repo, applied along the spectral (omega) axis instead
of a time axis (the math is identical; only which axis plays "time" changes).

THIS IS A DIFFERENT SETUP FROM dgs/nn_spectral_regression.py: that module compares
a trained neural network against ITERATIVE GS phase retrieval from TWO DISPERSED
INTENSITIES (this repo's core two-arm dispersion-diversity problem). This module
implements the paper's ACTUAL physical setup -- ONE interferogram from a
signal+delayed-reference interferometer -- which is a genuinely different
measurement, not a renaming of the same one. The paper's own neural-network
regression step (a 5-layer FC-NN trained on 6000 experimental interferograms,
directly on S(omega), outperforming this classical Hilbert baseline) is NOT
reimplemented here -- what IS implemented and tested is the classical baseline it
was compared against, plus the paper's quantization-noise robustness experiment
(Fig. 4: how the demodulator's accuracy degrades vs. digitizer ENOB), which is a
real, checkable claim independent of any trained model.

Eq. (1) of the paper (output vector length of the NN, n_output = 2|D|*delta_lambda*Fs)
is included here too, reusing dgs/photonic_vs_electronic_delay.py's already-verified
dispersion_induced_delay_spread_s (|D|*delta_lambda) rather than re-deriving it.
"""
from __future__ import annotations
import numpy as np
from typing import Dict

from dgs.causality import hilbert_transform
from dgs.photonic_vs_electronic_delay import dispersion_induced_delay_spread_s


# ── 1. Causal (minimum-phase) synthetic test profiles ──────────────────────

def minimum_phase_from_log_magnitude(log_magnitude: np.ndarray) -> np.ndarray:
    """Bode gain-phase relation: phase(omega) = -H[log_magnitude](omega), the
    Hilbert-transform pair between log-magnitude and phase for a causal,
    minimum-phase system. Uses the SAME hilbert_transform as dgs/causality.py's
    Re/Im susceptibility Kramers-Kronig pair, but this is a DIFFERENT pair of
    quantities (log-magnitude & phase, not Re & Im of chi) -- both are valid
    causality constraints from the same underlying math, not the same relation
    restated."""
    log_magnitude = np.asarray(log_magnitude, dtype=float)
    return -hilbert_transform(log_magnitude)


def random_causal_profile(n: int, rng: np.random.Generator, n_harmonics: int = 5,
                           log_mag_amplitude: float = 0.5):
    """A random, smooth, CAUSAL magnitude+phase spectrum pair -- matching the
    paper's method (section 2.1) of generating labeled training/test profiles
    that satisfy a Kramers-Kronig-type causality relation, realized here as the
    log-magnitude/phase Bode relation. Returns (magnitude, phase), both length n.
    """
    if n < 8:
        raise ValueError(f"n={n}: must be >= 8")
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics={n_harmonics}: must be >= 1")
    x = np.linspace(0.0, 1.0, n)
    log_mag = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        amp = rng.uniform(-1.0, 1.0) * log_mag_amplitude / k
        phase_k = rng.uniform(0.0, 2 * np.pi)
        log_mag += amp * np.sin(2 * np.pi * k * x + phase_k)
    magnitude = np.exp(log_mag)
    phase = minimum_phase_from_log_magnitude(log_mag)
    return magnitude, phase


# ── 2. The forward model: spectral interferogram ────────────────────────────

def spectral_interferogram(E_test: np.ndarray, E_ref: np.ndarray,
                            omega: np.ndarray, tau: float) -> np.ndarray:
    """S(omega) = |E_test|^2 + |E_ref|^2 + 2*Re[E_test * conj(E_ref) * exp(+i*omega*tau)],
    the spectral fringe pattern from a signal arm E_test interfering with a
    reference arm E_ref delayed by tau (the shear).

    SIGN CONVENTION, and why it matters: with np.fft.fft's exp(-i*2*pi*k*n/N)
    convention, a term exp(+i*omega*tau) is the one whose quefrency-domain peak
    lands in the POSITIVE (low, kept) FFT bins that hilbert_demodulate's
    analytic-signal construction preserves. Using exp(-i*omega*tau) here
    instead would put the cross term in the bins the Hilbert transform
    discards as "negative frequency," and hilbert_demodulate would silently
    recover conj(E_test) instead of E_test -- the exact conjugate-ambiguity
    failure mode dgs/nn_spectral_regression.py exists to fight, reintroduced
    here by a sign mismatch instead of by GS's inherent ambiguity. The two
    functions' signs are a matched pair; do not change one without the other.
    """
    E_test = np.asarray(E_test, dtype=complex)
    E_ref = np.asarray(E_ref, dtype=complex)
    omega = np.asarray(omega, dtype=float)
    if E_test.shape != E_ref.shape or E_test.shape != omega.shape:
        raise ValueError("E_test, E_ref, and omega must have the same shape")
    if tau == 0:
        raise ValueError("tau (shear delay) must be nonzero -- zero shear produces no fringes")
    cross = E_test * np.conj(E_ref) * np.exp(1j * omega * tau)
    return np.abs(E_test) ** 2 + np.abs(E_ref) ** 2 + 2.0 * np.real(cross)


def _lowpass_background(S: np.ndarray, cutoff_frac: float = 0.05) -> np.ndarray:
    """Estimate the slowly-varying background |E_test|^2+|E_ref|^2 by keeping only
    the lowest quefrency components of S (both DC and its FFT-wraparound mirror)."""
    n = len(S)
    spec = np.fft.fft(S)
    k = max(1, int(cutoff_frac * n / 2))
    mask = np.zeros(n)
    mask[:k] = 1.0
    mask[-k:] = 1.0
    return np.real(np.fft.ifft(spec * mask))


def valid_tau_range(omega: np.ndarray, background_cutoff_frac: float = 0.05):
    """The usable range of shear delay tau for THIS omega grid: too small a tau
    puts the fringe term's quefrency peak inside the band _lowpass_background
    removes as DC (fringe gets deleted along with the background); too large a
    tau pushes the peak past the Nyquist quefrency (n/2 bins) and it aliases
    back into the array, corrupting the demodulation. Returns (tau_min, tau_max);
    pick tau comfortably inside this range, not at its edges."""
    omega = np.asarray(omega, dtype=float)
    n = len(omega)
    if n < 8:
        raise ValueError(f"omega has {n} samples: need at least 8")
    domega = omega[1] - omega[0]
    bin_step = 2.0 * np.pi / (n * domega)
    k_min = max(1, int(background_cutoff_frac * n / 2)) + 1
    k_max = n // 2 - 1
    if k_max <= k_min:
        raise ValueError("omega grid too coarse: no valid tau range for this background_cutoff_frac")
    return k_min * bin_step, k_max * bin_step


# ── 3. The classical baseline: Hilbert-transform demodulation ──────────────

def hilbert_demodulate(S: np.ndarray, omega: np.ndarray, tau: float,
                        E_ref: np.ndarray | None = None,
                        background_cutoff_frac: float = 0.05) -> Dict:
    """Recover E_test(omega) from the interferogram S(omega): remove the
    slowly-varying background, form the analytic signal along the omega axis
    via dgs/causality.py's hilbert_transform (single-sideband filtering,
    matching the paper's section 2.4), then strip the known tau-carrier and
    divide out the known reference field. If E_ref is None, assumes a flat,
    unit-amplitude, zero-phase reference (E_ref=1), the common experimental
    choice when the reference arm is not itself the unknown."""
    S = np.asarray(S, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if S.shape != omega.shape:
        raise ValueError("S and omega must have the same shape")
    if tau == 0:
        raise ValueError("tau (shear delay) must be nonzero")
    background = _lowpass_background(S, background_cutoff_frac)
    S_ac = S - background
    analytic = S_ac + 1j * hilbert_transform(S_ac)
    demod = analytic * np.exp(-1j * omega * tau)
    if E_ref is None:
        E_ref = np.ones_like(omega, dtype=complex)
    E_ref = np.asarray(E_ref, dtype=complex)
    E_test_est = demod / (2.0 * np.conj(E_ref) + 1e-12)
    return {
        "E_test_est": E_test_est,
        "magnitude_est": np.abs(E_test_est),
        "phase_est": np.angle(E_test_est),
        "background": background,
    }


# ── 4. Digitizer quantization noise (paper's Fig. 4 robustness test) ───────

def quantize_enob(signal: np.ndarray, enob: float) -> np.ndarray:
    """Simulate an ADC with the given effective number of bits (ENOB):
    uniformly quantize signal to 2**enob levels spanning its actual range."""
    signal = np.asarray(signal, dtype=float)
    if enob <= 0:
        raise ValueError(f"enob={enob}: must be positive")
    lo, hi = signal.min(), signal.max()
    span = hi - lo
    if span == 0:
        return signal.copy()
    levels = 2.0 ** enob
    step = span / levels
    return lo + np.round((signal - lo) / step) * step


def demodulation_rmse_vs_enob(n_trials: int, enob_values, n: int = 256, tau: float = 100.0,
                               rng_seed: int = 0) -> Dict:
    """Reproduce the STRUCTURE of the paper's Fig. 4: sweep digitizer ENOB and
    report mean magnitude/phase RMSE of the classical Hilbert demodulator (NOT
    the paper's trained neural network -- see module docstring). Uses
    random_causal_profile for ground truth and a flat unit reference."""
    if n_trials < 1:
        raise ValueError(f"n_trials={n_trials}: must be >= 1")
    enob_values = np.asarray(enob_values, dtype=float)
    if enob_values.size < 1:
        raise ValueError("enob_values must be non-empty")
    rng = np.random.default_rng(rng_seed)
    omega = np.linspace(-1.0, 1.0, n)
    mag_rmse = np.zeros(enob_values.size)
    phase_rmse = np.zeros(enob_values.size)
    for _ in range(n_trials):
        magnitude, phase = random_causal_profile(n, rng)
        E_test = magnitude * np.exp(1j * phase)
        E_ref = np.ones(n, dtype=complex)
        S = spectral_interferogram(E_test, E_ref, omega, tau)
        for j, enob in enumerate(enob_values):
            S_q = quantize_enob(S, enob)
            result = hilbert_demodulate(S_q, omega, tau, E_ref=E_ref)
            mag_rmse[j] += float(np.sqrt(np.mean((result["magnitude_est"] - magnitude) ** 2)))
            offset = np.angle(np.mean(np.exp(1j * (phase - result["phase_est"]))))
            aligned = np.angle(np.exp(1j * (result["phase_est"] + offset - phase)))
            phase_rmse[j] += float(np.sqrt(np.mean(aligned ** 2)))
    return {
        "enob_values": enob_values,
        "mean_magnitude_rmse": mag_rmse / n_trials,
        "mean_phase_rmse_rad": phase_rmse / n_trials,
    }


# ── 5. Eq. (1): NN output vector length ─────────────────────────────────────

def spectral_regression_output_size(D_ps_per_nm: float, delta_lambda_nm: float, Fs_hz: float) -> float:
    """Eq. (1) of Pu & Jalali (2021): n_output = 2*|D|*delta_lambda*Fs, the
    length of the neural network's output vector (concatenated magnitude+phase
    spectra). Reuses dgs/photonic_vs_electronic_delay.py's already-verified
    dispersion_induced_delay_spread_s (= |D|*delta_lambda) instead of
    re-deriving the group-delay-spread formula."""
    if Fs_hz <= 0:
        raise ValueError("Fs_hz must be positive")
    delay_s = dispersion_induced_delay_spread_s(D_ps_per_nm, delta_lambda_nm)
    return 2.0 * delay_s * Fs_hz


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 256
    omega = np.linspace(-1.0, 1.0, n)
    tau_min, tau_max = valid_tau_range(omega)
    tau = 100.0
    print(f"valid tau range for this omega grid: [{tau_min:.1f}, {tau_max:.1f}] -- using tau={tau}")

    print("\n=== 1. Random causal test profile + interferogram + demodulation ===")
    magnitude, phase = random_causal_profile(n, rng)
    E_test = magnitude * np.exp(1j * phase)
    E_ref = np.ones(n, dtype=complex)
    S = spectral_interferogram(E_test, E_ref, omega, tau)
    result = hilbert_demodulate(S, omega, tau, E_ref=E_ref)
    mag_rmse = float(np.sqrt(np.mean((result["magnitude_est"] - magnitude) ** 2)))
    offset = np.angle(np.mean(np.exp(1j * (phase - result["phase_est"]))))
    aligned = np.angle(np.exp(1j * (result["phase_est"] + offset - phase)))
    phase_rmse_deg = float(np.degrees(np.sqrt(np.mean(aligned ** 2))))
    print(f"  magnitude RMSE: {mag_rmse:.4e}")
    print(f"  phase RMSE (aligned): {phase_rmse_deg:.4f} deg")

    print("\n=== 2. Quantization-noise robustness (paper's Fig. 4 structure) ===")
    sweep = demodulation_rmse_vs_enob(n_trials=20, enob_values=[2, 3, 4, 5, 6, 8, 10])
    for e, mrms, prms in zip(sweep["enob_values"], sweep["mean_magnitude_rmse"], sweep["mean_phase_rmse_rad"]):
        print(f"  ENOB={e:4.1f}  mag RMSE={mrms:.4e}  phase RMSE={np.degrees(prms):.3f} deg")

    print("\n=== 3. Eq. (1): NN output vector size ===")
    n_out = spectral_regression_output_size(D_ps_per_nm=-1000.0, delta_lambda_nm=20.0, Fs_hz=50e9)
    print(f"  n_output = 2*|D|*delta_lambda*Fs = {n_out:,.0f} samples "
          f"(D=-1000 ps/nm, delta_lambda=20nm, Fs=50 GSa/s)")
