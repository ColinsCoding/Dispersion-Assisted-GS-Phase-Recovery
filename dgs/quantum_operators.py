"""Observables are operators: <x>, <p>, <T>, and how sharp or fuzzy they are.

In quantum mechanics you never get "the position" of a particle -- you get a wavefunction
psi(x), and each measurable quantity is an OPERATOR that acts on it:
        position  x-hat psi = x psi,
        momentum  p-hat psi = -i hbar d psi/dx,
        kinetic   T-hat = p-hat^2 / 2m = -(hbar^2/2m) d^2/dx^2.
The AVERAGE you would measure is the expectation value
        <A> = integral psi* (A-hat psi) dx      (psi normalized),
and because these operators are HERMITIAN, <A> always comes out REAL -- that is what makes
them observables. The SPREAD around the average is the uncertainty
        Delta A = sqrt(<A^2> - <A>^2),
zero when the state is SHARP (an eigenstate of A-hat -- a definite value) and positive when
it is FUZZY (a superposition). Position and momentum cannot both be sharp: the Heisenberg
relation Delta_x Delta_p >= hbar/2 always holds, with EQUALITY only for a Gaussian wave
packet (the minimum-uncertainty state).

This module evaluates all of it on a grid, using an FFT SPECTRAL derivative for the momentum
operator (exact for a band-limited psi). Verified on a Gaussian packet -- <x>=x0, <p>=hbar k0,
Delta_x=sigma, Delta_p=hbar/2sigma, and Delta_x Delta_p = hbar/2 exactly -- and a plane wave
(sharp momentum, Delta_p ~ 0). The same operators whose eigenvalues are the levels of
dgs.quantum_oscillator and dgs.finite_square_well. NumPy only; py-3.13.
"""

import numpy as np


def normalize(psi, x):
    """Scale psi so integral |psi|^2 dx = 1 -- a proper probability amplitude."""
    psi = np.asarray(psi, complex)
    norm = np.sqrt(np.trapezoid(np.abs(psi) ** 2, x))
    if norm == 0:
        raise ValueError("cannot normalize the zero wavefunction")
    return psi / norm


def _d_dx(psi, dx):
    """Spectral (FFT) first derivative -- exact for a periodic/band-limited psi."""
    n = len(psi)
    k = 2 * np.pi * np.fft.fftfreq(n, dx)
    return np.fft.ifft(1j * k * np.fft.fft(psi))


def expectation_position(psi, x):
    """<x> = integral x |psi|^2 dx, the mean position."""
    psi = normalize(psi, x)
    return float(np.real(np.trapezoid(np.conj(psi) * x * psi, x)))


def uncertainty_position(psi, x):
    """Delta_x = sqrt(<x^2> - <x>^2), the spread in position (0 = sharply localized)."""
    psi = normalize(psi, x)
    x2 = np.real(np.trapezoid(np.conj(psi) * x ** 2 * psi, x))
    return float(np.sqrt(max(x2 - expectation_position(psi, x) ** 2, 0.0)))


def expectation_momentum(psi, x, hbar=1.0):
    """<p> = integral psi* (-i hbar d psi/dx) dx -- real for any state (p-hat is
    Hermitian). For a wave e^{i k0 x} it returns hbar k0."""
    psi = normalize(psi, x)
    dx = x[1] - x[0]
    p_psi = -1j * hbar * _d_dx(psi, dx)
    return float(np.real(np.trapezoid(np.conj(psi) * p_psi, x)))


def momentum_squared(psi, x, hbar=1.0):
    """<p^2> = hbar^2 integral |d psi/dx|^2 dx (>= 0), by integration by parts."""
    psi = normalize(psi, x)
    dx = x[1] - x[0]
    dpsi = _d_dx(psi, dx)
    return float(hbar ** 2 * np.real(np.trapezoid(np.abs(dpsi) ** 2, x)))


def uncertainty_momentum(psi, x, hbar=1.0):
    """Delta_p = sqrt(<p^2> - <p>^2), the spread in momentum."""
    p = expectation_momentum(psi, x, hbar)
    p2 = momentum_squared(psi, x, hbar)
    return float(np.sqrt(max(p2 - p ** 2, 0.0)))


def kinetic_energy(psi, x, mass=1.0, hbar=1.0):
    """<T> = <p^2> / 2m -- the average kinetic energy, always >= 0."""
    if mass <= 0:
        raise ValueError("mass must be positive")
    return momentum_squared(psi, x, hbar) / (2 * mass)


def heisenberg_product(psi, x, hbar=1.0):
    """Delta_x Delta_p for the state -- always >= hbar/2, with equality only for a
    Gaussian. Returns the product."""
    return uncertainty_position(psi, x) * uncertainty_momentum(psi, x, hbar)


def gaussian_packet(x, x0=0.0, sigma=1.0, k0=0.0):
    """A minimum-uncertainty Gaussian wave packet centered at x0 with width sigma and
    mean wavenumber k0: (2 pi sigma^2)^{-1/4} e^{-(x-x0)^2/4 sigma^2} e^{i k0 x}."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    x = np.asarray(x, float)
    return (2 * np.pi * sigma ** 2) ** -0.25 * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2)) \
        * np.exp(1j * k0 * x)


if __name__ == "__main__":
    hbar = 1.0
    x = np.linspace(-40, 40, 8192)
    x0, sigma, k0, m = 3.0, 1.5, 2.0, 1.0
    psi = gaussian_packet(x, x0, sigma, k0)

    print("Gaussian packet (x0=3, sigma=1.5, k0=2):")
    print(f"  <x> = {expectation_position(psi, x):.4f} (= x0=3),  "
          f"Delta_x = {uncertainty_position(psi, x):.4f} (= sigma=1.5)")
    print(f"  <p> = {expectation_momentum(psi, x):.4f} (= hbar k0=2),  "
          f"Delta_p = {uncertainty_momentum(psi, x):.4f} (= hbar/2sigma={hbar/(2*sigma):.4f})")
    dxdp = heisenberg_product(psi, x)
    print(f"  Delta_x Delta_p = {dxdp:.4f}  (= hbar/2 = {hbar/2:.4f}, the minimum)")
    print(f"  <T> = <p^2>/2m = {kinetic_energy(psi, x, m):.4f} "
          f"(= (hbar^2 k0^2 + hbar^2/4sigma^2)/2m = "
          f"{(hbar**2*k0**2 + hbar**2/(4*sigma**2))/(2*m):.4f})")

    print("\nsharp vs fuzzy (Gaussians trade Delta_x for Delta_p at fixed product hbar/2):")
    wide = gaussian_packet(x, 0, 10.0, k0)      # spread out in x -> sharp in p
    print(f"  wide packet (sigma=10):   Delta_x={uncertainty_position(wide, x):6.3f} (fuzzy x), "
          f"Delta_p={uncertainty_momentum(wide, x):.4f} (SHARP momentum)")
    narrow = gaussian_packet(x, 0, 0.3, 0)      # localized in x -> fuzzy in p
    print(f"  narrow packet (sigma=0.3): Delta_x={uncertainty_position(narrow, x):6.3f} (sharp x),  "
          f"Delta_p={uncertainty_momentum(narrow, x):.4f} (fuzzy momentum)")
