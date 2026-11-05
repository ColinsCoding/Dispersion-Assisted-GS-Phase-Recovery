"""
phase_shifting.py — phase recovery WITH a reference laser (phase-shifting interferometry)

The other way to get phase from intensity-only detectors, for comparison with
dispersion-assisted Gerchberg-Saxton (gs_core.py).

Physics in three lines:
  Add a known reference wave R e^{i delta_k} to the unknown field E = |E| e^{i phi}.
  Each frame records I_k = |E|^2 + R^2 + 2 |E| R cos(phi - delta_k).
  With N >= 3 equally spaced steps delta_k = 2 pi k / N:
      sum_k I_k e^{i delta_k} = N |E| R e^{i phi}      (the other terms cancel)
  so phi = angle(sum_k I_k e^{i delta_k}) and |E| = |sum| / (N R). For N = 4 this
  is the textbook phi = atan2(I1 - I3, I0 - I2).

Why it differs from dispersion-assisted GS:
  * the reference makes each frame LINEAR in E (through the cross term), so the
    phase comes out in one closed-form step, with no iteration;
  * the reference fixes the absolute phase, so there is no constant-offset or
    sign ambiguity;
  * it costs a stable, coherent reference arm, and any error in the phase steps
    leaks straight into the answer (see step_error_rad);
  * GS needs no reference and only 2 frames, and pays for that by iterating.

Noise convention: Gaussian noise with the same absolute standard deviation as
gs_core.make_qpsk_measurements, sigma = sqrt(mean|E|^2 * 10^(-snr_db/10)), so the
two methods are compared at the same detector noise. Shot noise is not modeled.
The predicted phase noise is sigma / (sqrt(2N) |E| R) (see predicted_phase_rms).
"""

import numpy as np


def _check_steps(n_steps):
    if int(n_steps) != n_steps or n_steps < 3:
        raise ValueError(f"n_steps={n_steps}: need an integer >= 3 (with 2 steps cos and sin cannot be separated).")
    return int(n_steps)


def phase_steps(n_steps=4):
    """The reference phase steps delta_k = 2 pi k / N, k = 0..N-1."""
    n_steps = _check_steps(n_steps)
    return 2 * np.pi * np.arange(n_steps) / n_steps


def psi_frames(E, R=1.0, n_steps=4, snr_db=None, step_error_rad=0.0, rng_seed=0):
    """Simulate the N camera frames I_k = |E + R e^{i (delta_k + error)}|^2 (+ noise).

    step_error_rad is a miscalibration of each step, applied as delta_k * (1 + step_error_rad / (pi/2))
    (a piezo whose stroke is off by a fixed fraction, the classic PSI error source).
    Returns an array of shape (N, len(E)).
    """
    if R <= 0:
        raise ValueError("R (reference amplitude) must be > 0")
    E = np.asarray(E, complex)
    deltas = phase_steps(n_steps) * (1 + step_error_rad / (np.pi / 2))
    frames = np.abs(E[None, :] + R * np.exp(1j * deltas)[:, None]) ** 2
    if snr_db is not None:
        rng = np.random.default_rng(rng_seed)
        sigma = np.sqrt(np.mean(np.abs(E) ** 2) * 10 ** (-snr_db / 10))
        frames = np.maximum(frames + rng.normal(0, sigma, frames.shape), 0)
    return frames


def psi_retrieve(frames, R=1.0):
    """Closed-form N-step recovery. Returns (phi, amplitude)."""
    frames = np.asarray(frames, float)
    if frames.ndim != 2:
        raise ValueError("frames must have shape (n_steps, n_samples)")
    n = _check_steps(frames.shape[0])
    s = np.tensordot(np.exp(1j * phase_steps(n)), frames, axes=1)
    return np.angle(s), np.abs(s) / (n * R)


def predicted_phase_rms(snr_db, n_steps=4, amplitude=1.0, R=1.0):
    """Small-noise phase error of N-step PSI: sigma / (sqrt(2N) |E| R)."""
    sigma = np.sqrt(amplitude ** 2 * 10 ** (-snr_db / 10))
    return sigma / (np.sqrt(2 * _check_steps(n_steps)) * amplitude * R)


def wrapped_rms(phi, phi_true):
    """RMS of the wrapped difference, with NO offset or sign removed."""
    return float(np.sqrt(np.mean(np.angle(np.exp(1j * (phi - phi_true))) ** 2)))


def compare_with_gs(snr_db=25.0, n_steps=4, rng_seed=0, n_iter=60):
    """Recover the same phase both ways at the same detector noise.

    Uses the test phase of gs_core.make_qpsk_measurements (64 symbols x 8 samples,
    D1=-5000, D2=-5750). GS error is reported after removing the offset/sign
    ambiguity (as in the README); PSI error is raw, because the reference fixes them.
    """
    from dgs import gs_core as gs
    wp = gs.make_qpsk_measurements(n_symbols=64, sps=8, D1=-5000.0, D2=-5750.0,
                                   snr_db=snr_db, rng_seed=rng_seed)
    phi_t = wp["phi_true"]
    phi_gs, _ = gs.retrieve_phase(wp["I1"], wp["I2"], wp["D1"], wp["D2"], n_iter=n_iter)
    gs_rms = min(wrapped_rms(s * phi_gs + np.angle(np.sum(np.exp(1j * (phi_t - s * phi_gs)))), phi_t)
                 for s in (1, -1))
    phi_psi, _ = psi_retrieve(psi_frames(np.exp(1j * phi_t), n_steps=n_steps, snr_db=snr_db,
                                         rng_seed=rng_seed + 1))
    return {"gs_rms": gs_rms, "gs_frames": 2, "psi_rms": wrapped_rms(phi_psi, phi_t),
            "psi_frames": n_steps, "psi_predicted": float(predicted_phase_rms(snr_db, n_steps))}
