"""
gs_core.py — Temporal Gerchberg-Saxton phase retrieval
Implements Solli, Gupta, Jalali — Appl. Phys. Lett. 95, 231108 (2009)

Physics in three lines:
  E(t) is the optical field.  You can only measure I(t) = |E(t)|².
  Dispersion maps E(t) → E_d(t) via H(ω) = exp(i π D ν²)  in frequency.
  Two measurements at D1, D2 let GS recover the unknown phase φ(t).

Grade 7 version:
  Imagine light as a wave on a jump rope.  You can video the rope shaking
  (intensity) but not feel which way it twisted (phase).  Run the rope
  through two different springs (D1, D2) and record both videos.
  GS guesses the twist, predicts both videos, compares, and corrects until
  the prediction matches reality.
"""

import numpy as np
from sympy import symbols, exp, pi, I, simplify, latex, sqrt

# ── SymPy derivation of the dispersion transfer function ─────────────────────

def show_transfer_function():
    """
    Returns the SymPy expression H(ν) and its LaTeX string.

    Group-velocity dispersion (GVD) phase shift in fiber:
        φ_GVD = ½ β₂ L ω²          [β₂ in ps²/km, L in km]

    In terms of cyclic frequency ν = ω / 2π and dispersion D [ps/nm]:
        H(ν) = exp(i π D ν²)

    The near-field temporal waveform is then:
        E_dispersed(t) = F⁻¹{ F[E(t)] · H(ν) }

    where F denotes the Fourier transform.
    """
    nu, D = symbols('nu D', real=True)
    H = exp(I * pi * D * nu**2)
    return H, latex(H)


# ── Core physics primitives ───────────────────────────────────────────────────

def disperse(E, D):
    """
    Apply dispersion D to field E in normalized discrete frequency.

    H[k] = exp(i π D (k/N)²)    k = 0..N-1  (normalized ν ∈ [0,1))

    Parameters
    ----------
    E : complex array, length N
    D : float, dispersion parameter (dimensionless or in ps/nm — consistent
        with how I1, I2 were generated)

    Returns
    -------
    E_d : complex array — dispersed temporal field
    """
    N = len(E)
    nu = np.fft.fftfreq(N)                        # normalized: ν ∈ [-0.5, 0.5)
    H = np.exp(1j * np.pi * D * nu**2)
    return np.fft.ifft(np.fft.fft(E) * H)


def undisperse(E_d, D):
    """Remove dispersion D: apply conjugate H*."""
    N = len(E_d)
    nu = np.fft.fftfreq(N)
    H_conj = np.exp(-1j * np.pi * D * nu**2)
    return np.fft.ifft(np.fft.fft(E_d) * H_conj)


def apply_amplitude_constraint(E, I_measured):
    """
    Replace |E(t)| with sqrt(I_measured(t)), keep phase.
    This is the core GS projection: enforce the measured intensity.
    """
    amp = np.sqrt(np.maximum(I_measured, 0.0))
    return amp * np.exp(1j * np.angle(E))


# ── One full GS iteration (paper Fig. 2) ─────────────────────────────────────

def gs_iteration(E, I1, I2, D1, D2):
    """
    One forward + reverse pass of the temporal GS algorithm.

    Forward (D1 → D2):
      1. Apply D1 to current estimate E → E_d1
      2. Enforce I1 on |E_d1|         → E_d1_constrained
      3. Remove D1, apply D2          → E_d2
      4. Enforce I2 on |E_d2|         → E_d2_constrained

    Reverse (D2 → D1):
      5. Remove D2, apply D1          → E_d1_back
      6. Enforce I1                   → E updated

    Parameters
    ----------
    E  : complex array, current field estimate
    I1 : float array, measured intensity at dispersion D1
    I2 : float array, measured intensity at dispersion D2
    D1, D2 : float, dispersion parameters

    Returns
    -------
    E_new : complex array, updated field estimate
    """
    # Forward arm: D1 constraint
    E_d1 = disperse(E, D1)
    E_d1 = apply_amplitude_constraint(E_d1, I1)

    # Convert D1 → D2
    E_base = undisperse(E_d1, D1)          # remove D1
    E_d2   = disperse(E_base, D2)          # apply D2
    E_d2   = apply_amplitude_constraint(E_d2, I2)

    # Reverse arm: back to D1
    E_base2 = undisperse(E_d2, D2)
    E_d1b   = disperse(E_base2, D1)
    E_new   = apply_amplitude_constraint(E_d1b, I1)

    return E_new


# ── Main retrieval loop ───────────────────────────────────────────────────────

def retrieve_phase(I1, I2, D1, D2, n_iter=20):
    """
    Recover optical phase from two time-domain intensity measurements.

    Parameters
    ----------
    I1, I2 : float arrays — measured intensities at dispersions D1, D2
    D1, D2 : float — dispersion parameters (same units as used to generate I1, I2)
    n_iter  : int — number of GS iterations (paper uses 20; errors plateau ~15)

    Returns
    -------
    phi    : float array — recovered phase φ(t) in radians
    errors : list of float — RMS amplitude error per iteration (should decrease)
    """
    N = min(len(I1), len(I2))
    I1, I2 = I1[:N], I2[:N]

    # Initial guess: unit amplitude, zero phase
    E = np.sqrt(np.maximum(I1, 0)).astype(complex)

    errors = []
    for _ in range(n_iter):
        E = gs_iteration(E, I1, I2, D1, D2)
        # Track RMS error on arm 1 (paper Fig. 3)
        err = float(np.sqrt(np.mean(
            (np.abs(disperse(E, D1))**2 - I1)**2
        )))
        errors.append(err)

    return np.angle(E), errors


# ── Generate synthetic test data (QPSK optical comm signal) ──────────────────

def make_qpsk_measurements(n_symbols=64, sps=8, D1=-600.0, D2=-900.0,
                            snr_db=25.0, rng_seed=0):
    """
    Simulate the system in Fig. 5(a) of the paper with QPSK modulated light.

    Encoding: bits → QPSK symbols → pulse-shaped complex baseband
              → treat as E(t) → disperse at D1, D2 → square-law detect → I1, I2

    Returns
    -------
    dict with keys: I1, I2, phi_true, t, D1, D2
    """
    rng = np.random.default_rng(rng_seed)
    N = n_symbols * sps

    # QPSK: random bits → ±1/√2 ± j/√2 symbols → upsample
    bits = rng.integers(0, 2, 2 * n_symbols)
    symbols = ((2 * bits[0::2] - 1) + 1j * (2 * bits[1::2] - 1)) / np.sqrt(2)
    E_up = np.zeros(N, dtype=complex)
    E_up[::sps] = symbols

    # Simple rectangular pulse (replace with RRC for real comms)
    h = np.ones(sps) / np.sqrt(sps)
    E = np.convolve(E_up, h, mode='same')

    phi_true = np.angle(E)

    # Disperse and detect
    I1 = np.abs(disperse(E, D1))**2
    I2 = np.abs(disperse(E, D2))**2

    # Add noise
    noise_floor = np.mean(I1) * 10**(-snr_db / 10)
    I1 += rng.normal(0, np.sqrt(noise_floor), N)
    I2 += rng.normal(0, np.sqrt(noise_floor), N)
    I1 = np.maximum(I1, 0)
    I2 = np.maximum(I2, 0)

    t = np.arange(N)
    return {"I1": I1, "I2": I2, "phi_true": phi_true, "t": t, "D1": D1, "D2": D2}


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # SymPy transfer function
    H_sym, H_latex = show_transfer_function()
    print(f"Transfer function: H(nu) = {H_sym}")
    print(f"LaTeX: {H_latex}\n")

    # Generate data and recover phase
    data = make_qpsk_measurements(n_symbols=128, snr_db=30.0)
    phi_est, errors = retrieve_phase(
        data["I1"], data["I2"], data["D1"], data["D2"], n_iter=20
    )

    phi_true = data["phi_true"]
    # Phase error ignores global offset (wrap to [-π, π])
    delta = np.angle(np.exp(1j * (phi_est - phi_true)))
    rms = float(np.sqrt(np.mean(delta**2)))
    print(f"RMS phase error after 20 iterations: {rms:.4f} rad ({np.degrees(rms):.2f}°)")
    print(f"Final amplitude error: {errors[-1]:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(data["I1"][:200], label='I₁ (D₁ = −600)')
    axes[0].plot(data["I2"][:200], label='I₂ (D₂ = −900)', alpha=0.7)
    axes[0].set_title("Measured intensities"); axes[0].legend()
    axes[0].set_xlabel("Sample index")

    axes[1].plot(phi_true[:200], label='True φ(t)', lw=1.5)
    axes[1].plot(phi_est[:200],  label='Recovered φ(t)', ls='--')
    axes[1].set_title("Phase recovery"); axes[1].legend()
    axes[1].set_xlabel("Sample index"); axes[1].set_ylabel("Phase (rad)")

    axes[2].semilogy(errors, 'o-', color='crimson')
    axes[2].set_title(f"GS convergence  (RMS error = {rms:.3f} rad)")
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("Amplitude error (RMS)")

    plt.tight_layout()
    plt.savefig("gs_core_test.png", dpi=150)
    print("Saved gs_core_test.png")
