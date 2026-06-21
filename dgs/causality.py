"""Causality and conservation -- two pillars, and why dispersion implies absorption.

CAUSALITY. A physical response cannot precede its cause: the impulse response is
zero for t < 0. That single fact, through Fourier analysis, forces the real and
imaginary parts of any susceptibility chi(omega) to be **Kramers-Kronig pairs** --
Hilbert transforms of one another:

    Re chi(omega) = -H[ Im chi ](omega) ,    Im chi(omega) = +H[ Re chi ](omega).

So **dispersion (Re) and absorption (Im) are not independent** -- measure one, you
know the other. This is the deep reason a dispersive medium must also absorb, and
it underlies the transfer function this whole repo inverts.

CONSERVATION. Charge is locally conserved: it cannot vanish here and reappear
there, only flow. That is the continuity equation

    d(rho)/dt + div J = 0

(in 1-D, d(rho)/dt + d J/dx = 0). A current that satisfies it conserves charge to
numerical precision; one that does not is unphysical.

NumPy only (FFT-based Hilbert transform, finite differences). Education.
"""

import numpy as np
from dgs import numerical_methods as nm


# ── the Hilbert transform (FFT-based analytic signal) ───────────────
def hilbert_transform(x):
    """H[x]: the Hilbert transform, computed as the imaginary part of the analytic
    signal (double the positive frequencies, zero the negative, inverse-FFT). This
    is the operator that maps the real and imaginary parts of a causal response."""
    x = np.asarray(x, float)
    N = len(x)
    X = np.fft.fft(x)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return np.imag(np.fft.ifft(X * h))


# ── a causal model: the Lorentz oscillator ──────────────────────────
def lorentz_susceptibility(omega, omega0, gamma, strength=1.0):
    """Lorentz-oscillator susceptibility chi(omega) = strength/(omega0^2 - omega^2 -
    i*gamma*omega). Its imaginary part is the absorption line (peak at omega0); its
    real part is the dispersion (anomalous near resonance). Causal by construction."""
    omega = np.asarray(omega, float)
    return strength / (omega0**2 - omega**2 - 1j * gamma * omega)


# ── Kramers-Kronig: get one part from the other (causality) ─────────
def kramers_kronig_real(chi_imag):
    """Reconstruct the real part (dispersion) from the imaginary part (absorption).
    Re chi = -H[Im chi]. Requires a symmetric omega grid centred on 0."""
    return -hilbert_transform(chi_imag)


def kramers_kronig_imag(chi_real):
    """Reconstruct the imaginary part (absorption) from the real part (dispersion).
    Im chi = +H[Re chi]."""
    return hilbert_transform(chi_real)


# ── conservation of charge: the continuity equation ─────────────────
def continuity_residual(rho, J, x, t):
    """Local charge-conservation residual d(rho)/dt + dJ/dx over a 2-D grid
    rho[t, x], J[t, x]. Returns the residual field; |residual| ~ 0 means charge is
    conserved (it only flows, never appears or vanishes)."""
    rho, J = np.asarray(rho, float), np.asarray(J, float)
    drho_dt = np.gradient(rho, t, axis=0)
    dJ_dx = np.gradient(J, x, axis=1)
    return drho_dt + dJ_dx


def drifting_packet(x, t, v, width=1.0):
    """A Gaussian charge packet rho = exp(-((x-vt)/w)^2) drifting at speed v, with the
    convective current J = v*rho. This pair satisfies continuity exactly (it just
    moves), so it is the textbook conserved example."""
    X, T = np.meshgrid(x, t)
    rho = np.exp(-((X - v * T) / width) ** 2)
    return rho, v * rho


if __name__ == "__main__":
    # CAUSALITY: reconstruct the Lorentz dispersion from its absorption (and back)
    w = np.linspace(-60, 60, 12000)
    chi = lorentz_susceptibility(w, omega0=10.0, gamma=1.5)
    Re_kk = kramers_kronig_real(chi.imag)
    Im_kk = kramers_kronig_imag(chi.real)
    s = slice(3000, 9000)                      # interior, away from FFT edges
    print("KRAMERS-KRONIG (causality links dispersion <-> absorption):")
    print(f"  Re from Im: max err {np.max(np.abs(chi.real[s]-Re_kk[s])):.2e}  "
          f"(scale {np.max(np.abs(chi.real[s])):.3f})")
    print(f"  Im from Re: max err {np.max(np.abs(chi.imag[s]-Im_kk[s])):.2e}")

    # CONSERVATION: a drifting charge packet conserves charge (continuity ~ 0)
    x = np.linspace(-10, 10, 400)
    t = np.linspace(0, 4, 300)
    rho, J = drifting_packet(x, t, v=1.5)
    res = continuity_residual(rho, J, x, t)
    print(f"\nCONTINUITY  d(rho)/dt + dJ/dx : max |residual| = {np.max(np.abs(res[5:-5,5:-5])):.2e}"
          f"  (charge conserved -> ~0)")
