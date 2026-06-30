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


# ── quantum position-momentum wavepacket ─────────────────────────────────────
# The time-bandwidth product above is the *optical* face of the same inequality.
# Here is the QM face: σ_x σ_p ≥ ħ/2, measured numerically from a wavefunction.

def wavepacket(sigma_x: float = 1.0, k0: float = 5.0,
               N: int = 4096, x_range: float = 40.0):
    """Normalised Gaussian wavepacket on a uniform grid.

    ψ(x) = (2π σ_x²)^{-1/4} exp(-x²/4σ_x²) exp(ik₀x)

    Minimum-uncertainty state: σ_x σ_p = ħ/2 exactly (in natural units ħ=1).
    Returns (psi, x).
    """
    x = np.linspace(-x_range, x_range, N)
    dx = x[1] - x[0]
    psi = (2 * np.pi * sigma_x**2) ** (-0.25) * np.exp(-x**2 / (4 * sigma_x**2)) \
          * np.exp(1j * k0 * x)
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm, x


def double_peak_packet(sigma_x: float = 0.8, k0: float = 3.0,
                       N: int = 4096, x_range: float = 40.0):
    """Superposition of two Gaussians — NOT minimum uncertainty.

    Position spread is large (two lobes separated by 6σ),
    so σ_x σ_p >> ħ/2.
    """
    x = np.linspace(-x_range, x_range, N)
    dx = x[1] - x[0]
    g = lambda x0: np.exp(-(x - x0)**2 / (4 * sigma_x**2)) * np.exp(1j * k0 * x)
    psi = g(-3 * sigma_x) + g(3 * sigma_x)
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm, x


def position_momentum_uncertainties(psi, x):
    """Measure σ_x, σ_p, and σ_x σ_p / ħ from a wavefunction on grid x.

    Uses FFT for the momentum representation (ħ = 1 natural units).
    Returns (sigma_x, sigma_p, product_in_hbar_units).
    """
    dx = x[1] - x[0]
    N  = len(psi)
    prob_x = np.abs(psi)**2

    mean_x  = np.sum(x * prob_x) * dx
    mean_x2 = np.sum(x**2 * prob_x) * dx
    sigma_x_val = np.sqrt(max(mean_x2 - mean_x**2, 0.0))

    # momentum-space via FFT (ħ=1: p = ħk = k)
    psi_p = np.fft.fftshift(np.fft.fft(psi)) * dx / np.sqrt(2 * np.pi)
    p = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    dp = p[1] - p[0]

    prob_p = np.abs(psi_p)**2
    prob_p /= np.sum(prob_p) * dp   # re-normalise FFT

    mean_p  = np.sum(p * prob_p) * dp
    mean_p2 = np.sum(p**2 * prob_p) * dp
    sigma_p_val = np.sqrt(max(mean_p2 - mean_p**2, 0.0))

    return sigma_x_val, sigma_p_val, sigma_x_val * sigma_p_val  # ħ=1


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

    # ── position-momentum section ────────────────────────────────────────────
    print("\n-- Position-momentum uncertainty  (hbar = 1 natural units) --")
    print(f"{'State':<30}  {'sx':>7}  {'sp':>7}  {'sx*sp':>9}  {'>=hbar/2?'}")
    print("-" * 65)
    qm_cases = [
        ("Gaussian  s=0.5",   wavepacket(sigma_x=0.5, k0=3.0)),
        ("Gaussian  s=1.0",   wavepacket(sigma_x=1.0, k0=3.0)),
        ("Gaussian  s=2.0",   wavepacket(sigma_x=2.0, k0=3.0)),
        ("Double-peak",       double_peak_packet()),
    ]
    for label, (psi, x) in qm_cases:
        sx, sp, prod = position_momentum_uncertainties(psi, x)
        ok = "YES" if prod >= 0.499 else "NO"
        print(f"{label:<30}  {sx:7.4f}  {sp:7.4f}  {prod:9.4f}  {ok}")
    print("Theoretical minimum (Gaussian): sigma_x * sigma_p = hbar/2 = 0.5000")
