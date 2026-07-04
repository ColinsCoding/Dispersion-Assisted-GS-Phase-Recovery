"""The 1D heat/diffusion equation, solved as a PDE transformed into
frequency space: dT/dt = alpha*d^2T/dx^2 becomes, after a spatial Fourier
transform, the ORDINARY differential equation
    dT_hat(k,t)/dt = -alpha*k^2*T_hat(k,t)
for each wavenumber k independently -- exactly the same kind of decoupling
dgs.band_theory gets from Bloch's theorem and dgs.lti_systems gets from an
RC filter's transfer function: turn a coupled/PDE problem into a set of
INDEPENDENT one-line ODEs, one per frequency, each with its own decay rate.

The solution per mode is T_hat(k,t) = T_hat(k,0)*exp(-alpha*k^2*t) -- a
Gaussian LOW-PASS FILTER in k-space (high spatial frequencies, i.e. sharp
features, die out fastest). This module solves the heat equation THREE
independent ways and cross-checks them against each other:
  1. spatial FFT -> per-mode exp(-alpha*k^2*t) decay -> inverse FFT
  2. explicit finite-difference (FTCS) time-stepping in real space
  3. the analytic self-similar Gaussian solution (variance grows linearly
     in time: sigma(t)^2 = sigma0^2 + 2*alpha*t)
"""

import numpy as np


def heat_kernel_gaussian(x, t, alpha, sigma0, amplitude=1.0):
    """Analytic solution for a Gaussian initial temperature profile under
    1D diffusion: a Gaussian STAYS Gaussian, with variance growing linearly
    in time, sigma(t)^2 = sigma0^2 + 2*alpha*t (amplitude renormalized so
    total heat integral(T dx) is conserved)."""
    if alpha <= 0 or sigma0 <= 0 or t < 0:
        raise ValueError("alpha and sigma0 must be positive, t must be non-negative")
    x = np.asarray(x, dtype=float)
    sigma_t2 = sigma0 ** 2 + 2 * alpha * t
    norm = amplitude * sigma0 / np.sqrt(sigma_t2)   # conserves integral(T dx) = amplitude*sigma0*sqrt(2*pi)
    return norm * np.exp(-x ** 2 / (2 * sigma_t2))


def fourier_mode_decay(k, alpha, t):
    """The frequency-space transfer function of the heat equation:
    T_hat(k,t)/T_hat(k,0) = exp(-alpha*k^2*t) -- a Gaussian low-pass
    filter, exactly analogous to dgs.lti_systems' RC transfer function
    but with the pole structure replaced by a Gaussian roll-off."""
    if alpha <= 0 or t < 0:
        raise ValueError("alpha must be positive, t must be non-negative")
    k = np.asarray(k, dtype=float)
    return np.exp(-alpha * k ** 2 * t)


def solve_heat_fourier(T0, x, alpha, t):
    """Solve the heat equation by spatial FFT: transform T0(x) to T_hat(k,0),
    multiply by the per-mode decay exp(-alpha*k^2*t), inverse-transform back
    to real space. This is the PDE-in-frequency-space method, done for real."""
    if alpha <= 0 or t < 0:
        raise ValueError("alpha must be positive, t must be non-negative")
    x = np.asarray(x, dtype=float)
    T0 = np.asarray(T0, dtype=float)
    n = len(x)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    T0_hat = np.fft.fft(T0)
    T_hat = T0_hat * fourier_mode_decay(k, alpha, t)
    return np.real(np.fft.ifft(T_hat))


def ftcs_stability_limit(dx, alpha):
    """The FTCS explicit finite-difference stability limit (CFL-like
    condition): dt <= dx^2/(2*alpha). Exceeding this makes the explicit
    scheme numerically unstable regardless of the true physics."""
    if dx <= 0 or alpha <= 0:
        raise ValueError("dx and alpha must be positive")
    return dx ** 2 / (2 * alpha)


def solve_heat_finite_difference(T0, dx, dt, alpha, n_steps):
    """Explicit FTCS (forward-time, centered-space) finite-difference
    solver for the 1D heat equation, periodic boundary conditions --
    a completely independent real-space method to cross-check the
    Fourier-space solution against."""
    if dx <= 0 or dt <= 0 or alpha <= 0:
        raise ValueError("dx, dt, alpha must be positive")
    r = alpha * dt / dx ** 2
    if r > 0.5:
        raise ValueError(f"FTCS unstable: alpha*dt/dx^2={r:.3f} > 0.5; "
                          f"reduce dt below {ftcs_stability_limit(dx, alpha):.3e}")
    T = np.asarray(T0, dtype=float).copy()
    for _ in range(n_steps):
        T = T + r * (np.roll(T, -1) - 2 * T + np.roll(T, 1))
    return T


if __name__ == "__main__":
    alpha = 0.01       # thermal diffusivity, arbitrary units
    sigma0 = 0.3
    L = 20.0
    n = 2048
    x = np.linspace(-L / 2, L / 2, n, endpoint=False)
    dx = x[1] - x[0]
    t_final = 5.0

    T0 = heat_kernel_gaussian(x, 0.0, alpha, sigma0)

    T_fourier = solve_heat_fourier(T0, x, alpha, t_final)
    T_analytic = heat_kernel_gaussian(x, t_final, alpha, sigma0)

    dt = 0.4 * ftcs_stability_limit(dx, alpha)
    n_steps = int(round(t_final / dt))
    T_fd = solve_heat_finite_difference(T0, dx, dt, alpha, n_steps)

    err_fourier = np.max(np.abs(T_fourier - T_analytic))
    err_fd = np.max(np.abs(T_fd - T_analytic))
    print(f"heat equation dT/dt = {alpha}*d2T/dx2, Gaussian initial condition (sigma0={sigma0})")
    print(f"after t={t_final}: analytic sigma(t) = {np.sqrt(sigma0**2 + 2*alpha*t_final):.4f}")
    print(f"Fourier-space method  max error vs analytic: {err_fourier:.3e}")
    print(f"finite-difference ({n_steps} steps, dt={dt:.2e})  max error vs analytic: {err_fd:.3e}")
    print()
    print("frequency-space picture: high-k (sharp/fast-varying) modes decay")
    print(f"fastest -- at k=10: decay factor = {fourier_mode_decay(10.0, alpha, t_final):.3e}; "
          f"at k=1: decay factor = {fourier_mode_decay(1.0, alpha, t_final):.3e}")
