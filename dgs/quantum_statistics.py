"""Quantum statistics — and the photon noise the receiver actually sees.

The quantum century in one file. In 1900 Planck fixed the blackbody catastrophe
by quantizing energy; the occupation rules that followed split the world in two:

    Maxwell-Boltzmann   n = e^{-(E-mu)/kT}              (classical, distinguishable)
    Bose-Einstein       n = 1 / (e^{(E-mu)/kT} - 1)     (photons, phonons; bunch)
    Fermi-Dirac         n = 1 / (e^{(E-mu)/kT} + 1)      (electrons; Pauli, n <= 1)

Both quantum laws collapse to Maxwell-Boltzmann when states are sparsely filled.

Why this lives in *this* repo: the carrier-less detector counts photons, and the
*statistics of the light* set the noise. Coherent (laser) light is Poissonian
(Mandel Q = 0); thermal light is Bose-Einstein / super-Poissonian (Q = nbar),
i.e. noisier for the same mean — so the same Gerchberg-Saxton recovery is harder
under thermal illumination than under a laser. NumPy only. Education / metrology.
"""

import numpy as np

# SI constants (for Planck's law); occupation functions use natural units (kB arg)
H = 6.62607015e-34      # J s
C = 2.99792458e8        # m/s
KB = 1.380649e-23       # J/K


# ── 1. occupation numbers ────────────────────────────────────────────
def maxwell_boltzmann(E, mu, T, kB=1.0):
    """Classical mean occupancy e^{-(E-mu)/kT} (distinguishable particles)."""
    return np.exp(-(np.asarray(E, float) - mu) / (kB * T))


def bose_einstein(E, mu, T, kB=1.0):
    """Boson mean occupancy 1/(e^{(E-mu)/kT}-1); requires E > mu (else diverges)."""
    x = (np.asarray(E, float) - mu) / (kB * T)
    if np.any(x <= 0):
        raise ValueError("Bose-Einstein needs E > mu (occupancy diverges otherwise)")
    return 1.0 / np.expm1(x)


def fermi_dirac(E, mu, T, kB=1.0):
    """Fermion mean occupancy 1/(e^{(E-mu)/kT}+1); always in [0,1] (Pauli)."""
    x = (np.asarray(E, float) - mu) / (kB * T)
    return 1.0 / (np.exp(x) + 1.0)


# ── 2. Planck's law (Bose-Einstein for photons) ──────────────────────
def planck_spectral_radiance(nu, T):
    """Blackbody spectral radiance B(nu,T) = (2 h nu^3/c^2)/(e^{h nu/kT}-1) [W/m^2/sr/Hz]."""
    nu = np.asarray(nu, float)
    with np.errstate(over="ignore"):           # huge h*nu/kT -> inf -> radiance 0 (fine)
        return (2 * H * nu**3 / C**2) / np.expm1(H * nu / (KB * T))


def rayleigh_jeans(nu, T):
    """Classical low-frequency limit 2 nu^2 kT / c^2 — the 'ultraviolet catastrophe'."""
    nu = np.asarray(nu, float)
    return 2 * nu**2 * KB * T / C**2


def wien_peak_frequency(T):
    """Frequency of peak blackbody radiance: nu_max = 2.8214 kT/h (Wien displacement)."""
    return 2.8214393721 * KB * T / H


def wien_peak_wavelength(T):
    """Wavelength of peak radiance B_lambda: lambda_max = b/T, b = 2.8978e-3 m K.

    NOTE the Jacobian subtlety: the wavelength peak is NOT c/nu_max. B(nu) and
    B(lambda) are densities in different variables, so c/nu_max (~883 nm for the
    Sun, near-IR) differs from lambda_max (~502 nm, visible). Same spectrum, two
    legitimate 'peaks'.
    """
    return 2.897771955e-3 / T


# ── 3. photon-number statistics (the detector noise) ─────────────────
def poisson_pmf(n, nbar):
    """Coherent (laser) photon-count distribution: P(n)=nbar^n e^{-nbar}/n!."""
    n = np.asarray(n, int)
    if nbar < 0:
        raise ValueError("nbar must be >= 0")
    from math import lgamma
    logp = n * np.log(nbar + 1e-300) - nbar - np.array([lgamma(k + 1) for k in n])
    return np.exp(logp)


def thermal_pmf(n, nbar):
    """Single-mode thermal (Bose-Einstein) photon counts: P(n)=nbar^n/(1+nbar)^{n+1}.

    Same mean as the coherent case but variance nbar + nbar^2 — photon *bunching*,
    the extra (classical-intensity-fluctuation) noise on top of shot noise.
    """
    n = np.asarray(n, int)
    if nbar < 0:
        raise ValueError("nbar must be >= 0")
    return nbar**n / (1.0 + nbar) ** (n + 1)


def mandel_q(variance, mean):
    """Mandel Q = var/mean - 1.  Q=0 Poisson (coherent), Q>0 super-Poissonian
    (thermal/bunched), Q<0 sub-Poissonian (nonclassical, e.g. single photons)."""
    if mean <= 0:
        raise ValueError("mean must be > 0")
    return variance / mean - 1.0


if __name__ == "__main__":
    T = 5772.0  # the Sun's surface
    nu_peak = wien_peak_frequency(T)
    print(f"Sun T={T} K -> B(nu) peaks at {nu_peak/1e12:.1f} THz "
          f"(c/nu_max={C/nu_peak*1e9:.0f} nm, near-IR); "
          f"B(lambda) peaks at {wien_peak_wavelength(T)*1e9:.0f} nm (visible)")
    nbar = 5.0
    n = np.arange(0, 40)
    var_th = (thermal_pmf(n, nbar) * n**2).sum() - nbar**2
    print(f"nbar={nbar}: coherent Q={mandel_q(nbar, nbar):.2f}, "
          f"thermal Q={mandel_q(var_th, nbar):.2f}  (thermal light is noisier)")
