"""The uncertainty principle, from ultrashort pulses to energy and time.

A pulse cannot be short in time AND narrow in spectrum: the rms widths obey
    Delta_t * Delta_omega >= 1/2,
saturated by a Gaussian (the transform-limited pulse). Multiply by hbar and it is
the energy-time uncertainty
    Delta_E * Delta_t >= hbar/2,   Delta_E = hbar * Delta_omega,
so a femtosecond pulse necessarily spans a broad band of photon energies. That
broadband content is exactly why dispersion matters in this repo: a wide spectrum
spreads in time as it propagates. A linear chirp keeps Delta_t fixed but widens
Delta_omega, raising the product above the 1/2 floor -- the signature of the chirped
pulses the receiver works with.

Widths are taken on the intensity (|field|^2 in time, |spectrum|^2 in frequency).
NumPy (FFT + rms). Education.
"""

import numpy as np

HBAR = 1.054571817e-34          # reduced Planck constant [J s]


def _trapz(y, x):
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)


def rms_width(grid, density):
    """Root-mean-square width of a distribution: sqrt(<(g-<g>)^2>), with `density`
    the (unnormalized) weight over `grid`. The standard 'sigma' of the intensity."""
    g = np.asarray(grid, float)
    w = np.asarray(density, float)
    w = w / _trapz(w, g)
    mean = _trapz(g * w, g)
    return float(np.sqrt(_trapz((g - mean) ** 2 * w, g)))


def spectrum(t, field):
    """Angular-frequency axis and complex spectrum of a time-domain field (centered)."""
    N = len(t)
    dt = t[1] - t[0]
    F = np.fft.fftshift(np.fft.fft(field))
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, dt))
    return omega, F


def time_bandwidth_product(t, field):
    """Delta_t * Delta_omega from the intensity rms widths. >= 1/2 for any pulse,
    = 1/2 for a transform-limited Gaussian."""
    dt_rms = rms_width(t, np.abs(field) ** 2)
    omega, F = spectrum(t, field)
    dw_rms = rms_width(omega, np.abs(F) ** 2)
    return dt_rms * dw_rms


def energy_time_product(t, field):
    """Delta_E * Delta_t = hbar * (Delta_t * Delta_omega) [joule-seconds]. >= hbar/2."""
    return HBAR * time_bandwidth_product(t, field)


def gaussian_pulse(t, tau, chirp=0.0):
    """A Gaussian envelope exp(-t^2/(2 tau^2)) with an optional linear chirp
    exp(i * chirp * t^2). Chirp leaves the intensity (Delta_t) unchanged but spreads
    the spectrum (Delta_omega), pushing the time-bandwidth product above 1/2."""
    t = np.asarray(t, float)
    return np.exp(-t ** 2 / (2 * tau ** 2)) * np.exp(1j * chirp * t ** 2)


if __name__ == "__main__":
    t = np.linspace(-40, 40, 8192)
    for label, field in [
        ("Gaussian (transform-limited)", gaussian_pulse(t, 2.0)),
        ("Gaussian, chirped",            gaussian_pulse(t, 2.0, chirp=0.3)),
        ("rectangular",                  (np.abs(t) < 2.0).astype(float)),
    ]:
        print(f"  {label:30s} Dt*Dw = {time_bandwidth_product(t, field):.3f}")
    # shorter pulse -> broader spectrum: halving tau doubles the rms bandwidth
    for tau in (4.0, 2.0, 1.0):
        omega, F = spectrum(t, gaussian_pulse(t, tau))
        print(f"  tau={tau}:  Delta_omega = {rms_width(omega, np.abs(F)**2):.4f}")
    print("\nGaussian is the minimum (0.5); chirp and hard edges raise it.")
