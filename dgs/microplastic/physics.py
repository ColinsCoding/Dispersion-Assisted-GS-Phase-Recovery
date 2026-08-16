"""Month 1: Maxwell equations, Fourier integrals, complex numbers -- NumPy validated.

Before any particle or detector model, the foundation has to be nailed down: what
a complex refractive index *means* physically, how it propagates a plane wave,
and that our discrete FFT actually reproduces the continuous Fourier transform it
approximates. Every later month (dispersion, scattering, detector noise) builds
on the four relations below.

    n~(omega)    = n + i*kappa                      complex refractive index
    eps_r        = n~^2                              Maxwell, nonmagnetic medium
    k~           = n~ * omega / c                    complex wave number
    E(z,omega)   = E(0,omega) * exp(i*k~*z)           propagation

The imaginary part kappa produces exponential decay (absorption); the real part
n produces the phase delay (dispersion). alpha = 2*omega*kappa/c is the intensity
absorption coefficient, so |E(z)|^2 = |E(0)|^2 * exp(-alpha*z) -- Beer-Lambert
falls straight out of the complex exponential, not as a separate postulate.

NumPy only. Education.
"""

import numpy as np

C = 2.99792458e8          # speed of light [m/s]
EPS0 = 8.8541878128e-12   # vacuum permittivity [F/m]


# ── complex refractive index (Maxwell, nonmagnetic medium) ──────────────────
def complex_index(n, kappa):
    """n~(omega) = n + i*kappa. n sets phase velocity, kappa sets absorption."""
    return np.asarray(n, dtype=complex) + 1j * np.asarray(kappa, dtype=float)


def permittivity(n_tilde):
    """eps_r = n~^2 -- the Maxwell relation linking optical index to the material
    response Maxwell's equations actually see."""
    n_tilde = np.asarray(n_tilde, dtype=complex)
    return n_tilde ** 2


def wave_number(n_tilde, omega, c=C):
    """k~ = n~ * omega / c, the complex propagation constant."""
    return np.asarray(n_tilde, dtype=complex) * np.asarray(omega, dtype=float) / c


# ── propagation through a slab ───────────────────────────────────────────────
def propagate_field(E0, k_tilde, z):
    """E(z,omega) = E(0,omega) * exp(i*k~*z). Re(k~) rotates the phase (dispersion);
    Im(k~) shrinks the amplitude (absorption)."""
    return np.asarray(E0, dtype=complex) * np.exp(1j * np.asarray(k_tilde, dtype=complex) * z)


def absorption_coefficient(omega, kappa, c=C):
    """alpha = 2*omega*kappa/c: the *intensity* absorption coefficient (factor of
    2 because intensity ~ |E|^2, and |exp(i*k~*z)|^2 = exp(-2*Im(k~)*z))."""
    return 2.0 * np.asarray(omega, dtype=float) * np.asarray(kappa, dtype=float) / c


def beer_lambert_transmittance(alpha, z):
    """T = exp(-alpha*z): fraction of intensity surviving a path length z."""
    return np.exp(-np.asarray(alpha, dtype=float) * np.asarray(z, dtype=float))


# ── energy flow ───────────────────────────────────────────────────────────────
def time_averaged_poynting(E0_amplitude, n, eps0=EPS0, c=C):
    """Time-averaged Poynting magnitude for a plane wave in a lossless dielectric
    of real index n: <S> = 0.5 * n * eps0 * c * E0^2 (n=1 recovers the vacuum
    formula). This is intensity, W/m^2 -- integrate over area for power P."""
    return 0.5 * np.asarray(n, dtype=float) * eps0 * c * np.asarray(E0_amplitude, dtype=float) ** 2


# ── Fourier integrals (validated against the continuous transform) ──────────
def fourier_transform(t, Et):
    """Numerical approximation to E(omega) = integral E(t) exp(-i*omega*t) dt via
    FFT, with proper dt scaling and fftshift so the returned frequency axis is
    monotonically increasing. np.fft.fft implicitly assumes the first sample sits
    at t=0, so when the input array starts elsewhere (e.g. a pulse centred at
    t=0 sampled from t[0]=-50) we correct with the extra linear phase
    exp(-i*omega*t[0]) -- otherwise every omega!=0 bin is dominated by a spurious
    phase and only the omega=0 magnitude looks right. Returns (omega, E_omega)."""
    t = np.asarray(t, dtype=float)
    Et = np.asarray(Et, dtype=complex)
    dt = t[1] - t[0]
    N = len(t)
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    omega = 2 * np.pi * freq
    Ef = np.fft.fftshift(np.fft.fft(Et)) * dt * np.exp(-1j * omega * t[0])
    return omega, Ef


def inverse_fourier_transform(omega, Ef, t0=0.0):
    """Inverse of fourier_transform: E(t) = (1/2pi) integral E(omega) exp(i*omega*t)
    domega, via IFFT. t0 must match the t[0] used when Ef was produced by
    fourier_transform (it undoes that function's exp(-i*omega*t[0]) phase
    reference before the IFFT, then reports t on the same absolute time grid).
    Returns (t, E_t)."""
    omega = np.asarray(omega, dtype=float)
    Ef = np.asarray(Ef, dtype=complex)
    domega = omega[1] - omega[0]
    N = len(omega)
    dt = 2 * np.pi / (N * domega)
    t = t0 + np.arange(N) * dt
    Et = np.fft.ifft(np.fft.ifftshift(Ef * np.exp(1j * omega * t0))) * (N * domega) / (2 * np.pi)
    return t, Et


def gaussian_pulse(t, tau=1.0, omega0=0.0, t0=0.0):
    """Gaussian envelope * optional carrier: E(t) = exp(-(t-t0)^2/(2*tau^2)) *
    exp(i*omega0*(t-t0)). tau is the 1/e amplitude half-width."""
    t = np.asarray(t, dtype=float)
    return np.exp(-(t - t0) ** 2 / (2 * tau ** 2)) * np.exp(1j * omega0 * (t - t0))


def gaussian_pulse_ft_analytic(omega, tau=1.0, omega0=0.0, t0=0.0):
    """Closed-form FT of gaussian_pulse: a Gaussian of time-width tau transforms
    to a Gaussian of frequency-width 1/tau, centred at omega0 -- the
    time-bandwidth product this whole repo's dispersion kernel exploits.
    E(omega) = tau*sqrt(2*pi) * exp(-tau^2*(omega-omega0)^2/2) * exp(-i*omega*t0)."""
    omega = np.asarray(omega, dtype=float)
    return (tau * np.sqrt(2 * np.pi)
            * np.exp(-tau ** 2 * (omega - omega0) ** 2 / 2)
            * np.exp(-1j * omega * t0))


def parseval_check(t, Et, omega, Ef):
    """Parseval / Plancherel: integral |E(t)|^2 dt == (1/2pi) integral |E(omega)|^2
    domega. Returns (time_energy, freq_energy, relative_error) -- the FFT
    convention above is only trustworthy if this holds to numerical precision."""
    dt = t[1] - t[0]
    domega = omega[1] - omega[0]
    time_energy = np.sum(np.abs(Et) ** 2) * dt
    freq_energy = np.sum(np.abs(Ef) ** 2) * domega / (2 * np.pi)
    rel_error = abs(time_energy - freq_energy) / time_energy
    return time_energy, freq_energy, rel_error


# ── photon counting ───────────────────────────────────────────────────────────
def photon_rate(n_photons, duration_seconds):
    """Average photon arrival rate: n_photons / duration_seconds."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    return n_photons / duration_seconds


if __name__ == "__main__":
    n_tilde = complex_index(1.33, 0.01)
    omega = 2 * np.pi * 3e14  # ~1000 nm optical
    alpha = absorption_coefficient(omega, 0.01)
    print(f"n~ = {n_tilde}, eps_r = {permittivity(n_tilde)}")
    print(f"alpha = {alpha:.3e} 1/m, transmittance over 1 mm = {beer_lambert_transmittance(alpha, 1e-3):.4f}")
    print(f"photon rate: {photon_rate(1e9, 10):.3e} photons/s")
