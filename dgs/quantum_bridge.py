"""Quantum bridge -- the QM under all three Jalali-lab projects.

Modern physics says light is a complex wave whose squared magnitude is what a detector
measures (the Born rule), and that idea is the common root of this lab's three
projects:

  1. DISPERSION-GS PHASE RETRIEVAL. A detector reads |psi|^2 and throws the phase away
     -- the quantum measurement problem. Phase retrieval is solving it with dispersive
     diversity. (born_rule, phase_ambiguity.)
  2. SEALS SCATTERING. Light-matter scattering is a sum over partial waves; the n-th
     carries orbital angular momentum n*hbar, and the spherical Bessel radial functions
     (dgs.bessel_linalg) are the angular-momentum eigenstates. (partial_wave_angular_momentum.)
  3. 90-DEG OPTICAL HYBRID. Coherent detection measures the complex amplitude (I/Q) by
     interfering with a reference; its floor is shot noise sqrt(N) -- the photon's
     particle nature. (shot_noise_snr.)

And one principle threads them: Fourier/time-bandwidth uncertainty (time_bandwidth_product),
the same hbar/2 limit Griffiths derives for position-momentum. NumPy. Education.
"""

import numpy as np

_HBAR = 1.054571817e-34


def born_rule(psi):
    """|psi|^2: a detector's intensity (optics) = the probability density (QM Born rule).
    It DISCARDS the phase of psi -- exactly why phase retrieval (and the quantum
    measurement problem) is hard: many fields share one intensity."""
    return np.abs(np.asarray(psi)) ** 2


def phase_ambiguity(intensity, phase_a, phase_b):
    """Two fields sqrt(I) e^{i phi_a} and sqrt(I) e^{i phi_b} have IDENTICAL Born-rule
    intensity but different phase -- intensity alone cannot choose the phase. Returns the
    two distinct complex fields (the ambiguity phase retrieval must break)."""
    amp = np.sqrt(np.asarray(intensity, float))
    return amp * np.exp(1j * np.asarray(phase_a)), amp * np.exp(1j * np.asarray(phase_b))


def partial_wave_angular_momentum(n, hbar=_HBAR):
    """Orbital angular momentum of the n-th scattering partial wave = n*hbar per photon.
    The SEALS Mie pattern is a sum over these quantum partial waves, whose radial parts
    are the spherical Bessel functions (angular-momentum eigenstates)."""
    return n * hbar


def shot_noise_snr(n_photons):
    """Quantum shot-noise-limited SNR = sqrt(N): detecting N photons has noise sqrt(N).
    That sqrt(N) floor is the photon's PARTICLE nature -- the quantum limit a coherent
    receiver (the 90-deg hybrid, or the GS receiver) works against. SNR improves only as
    sqrt(N), so 4x the light buys 2x the SNR."""
    return np.sqrt(np.asarray(n_photons, float))


def time_bandwidth_product(t, pulse):
    """Delta_t * Delta_omega from a pulse and its spectrum (intensity second moments).
    A Gaussian (minimum-uncertainty) pulse reaches ~1/2 -- the energy-time uncertainty,
    the same hbar/2 limit as position-momentum, and the reason ultrashort pulses are
    broadband. Returns the dimensionless product."""
    t = np.asarray(t, float)
    I = np.abs(pulse) ** 2
    I = I / I.sum()
    tbar = np.sum(t * I)
    dt = np.sqrt(np.sum((t - tbar) ** 2 * I))
    w = 2 * np.pi * np.fft.fftfreq(len(t), t[1] - t[0])
    P = np.abs(np.fft.fft(pulse)) ** 2
    P = P / P.sum()
    wbar = np.sum(w * P)
    dw = np.sqrt(np.sum((w - wbar) ** 2 * P))
    return float(dt * dw)


if __name__ == "__main__":
    # phase ambiguity: same intensity, different phase
    I = np.array([1.0, 4.0, 9.0])
    a, b = phase_ambiguity(I, [0, 0, 0], [0.3, -0.7, 1.1])
    print("same intensity?", np.allclose(born_rule(a), born_rule(b)),
          " different fields?", not np.allclose(a, b))
    # shot noise: SNR = sqrt(N)
    print("shot-noise SNR at 1e4 photons =", shot_noise_snr(1e4), "(= sqrt(N))")
    # time-bandwidth of a Gaussian -> ~0.5 (minimum uncertainty)
    t = np.linspace(-20, 20, 4096)
    g = np.exp(-t**2 / 2)
    print("Gaussian time-bandwidth Dt*Dw =", round(time_bandwidth_product(t, g), 4), "(min ~0.5)")
    # angular momentum of the 3rd partial wave
    print("3rd partial wave L =", partial_wave_angular_momentum(3), "J s = 3 hbar")
