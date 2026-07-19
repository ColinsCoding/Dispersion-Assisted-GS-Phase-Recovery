"""Griffiths QM, plotted -- four visuals tied to the three live Jalali-lab projects.

No scipy (py-3.13 has none); special functions are built by hand from their
defining recursions, which is also the more honest way to show *why* they look
the way they do.

  1. square_well / harmonic_oscillator eigenfunctions + a superposition's time
     evolution -- Re(psi), Im(psi), |psi|^2 side by side. The detector only ever
     sees the last panel; the first two carry the phase phase-retrieval recovers.
  2. uncertainty_vs_chirp -- Delta-t * Delta-omega across pulse_gen chirps, the
     same Fourier uncertainty Griffiths derives for x and p (Ch.3).
  3. legendre_envelope / spherical_harmonic_intensity (Y_l^0) next to a
     partial-wave angular weight (2l+1)|j_l(x)|^2 -- same special-function
     machinery, hydrogen orbital vs. scattering multipole (Ch.4 vs SEALS).
  4. hard_sphere_phase_shift + born_square_well_cross_section -- partial-wave
     phase shifts and the Born approximation (Ch.11), the two analytic handles
     scattering theory gives you before you need a full Mie code.
"""

import numpy as np

# -- 1. Square well / harmonic oscillator ---------------------------------------

def square_well_eigenfunction(n, x, L=1.0):
    """psi_n(x) = sqrt(2/L) sin(n pi x / L) for 0<=x<=L, else 0. n=1,2,3,..."""
    if n < 1:
        raise ValueError("n must be >= 1")
    psi = np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)
    return np.where((x >= 0) & (x <= L), psi, 0.0)


def square_well_energy(n, L=1.0, hbar=1.0, m=1.0):
    return (n * np.pi * hbar) ** 2 / (2 * m * L ** 2)


def _hermite(n, x):
    """Physicists' Hermite polynomial H_n(x) via H_{k+1}=2x H_k - 2k H_{k-1}."""
    H_prev, H_curr = np.ones_like(x), 2.0 * x
    if n == 0:
        return H_prev
    if n == 1:
        return H_curr
    for k in range(1, n):
        H_prev, H_curr = H_curr, 2 * x * H_curr - 2 * k * H_prev
    return H_curr


def harmonic_oscillator_eigenfunction(n, x, m=1.0, omega=1.0, hbar=1.0):
    """psi_n(x) for V=1/2 m omega^2 x^2, normalized."""
    if n < 0:
        raise ValueError("n must be >= 0")
    xi = np.sqrt(m * omega / hbar) * x
    norm = (m * omega / (np.pi * hbar)) ** 0.25 / np.sqrt(2.0 ** n * math_factorial(n))
    return norm * _hermite(n, xi) * np.exp(-xi ** 2 / 2)


def math_factorial(n):
    out = 1
    for k in range(2, n + 1):
        out *= k
    return out


def superposition_snapshot(coeffs, ns, x, t, L=1.0, hbar=1.0, m=1.0):
    """psi(x,t) = sum_n c_n psi_n(x) exp(-i E_n t/hbar) for the square well.

    Returns the complex array psi(x) at time t. `coeffs` need not be normalized;
    they are normalized internally so |psi|^2 integrates to 1.
    """
    coeffs = np.asarray(coeffs, dtype=complex)
    coeffs = coeffs / np.sqrt(np.sum(np.abs(coeffs) ** 2))
    psi = np.zeros_like(x, dtype=complex)
    for c, n in zip(coeffs, ns):
        E_n = square_well_energy(n, L=L, hbar=hbar, m=m)
        psi += c * square_well_eigenfunction(n, x, L=L) * np.exp(-1j * E_n * t / hbar)
    return psi


# -- 2. Uncertainty vs chirp ------------------------------------------------------

def pulse_time_freq_widths(t, x):
    """RMS width Delta-t of |x(t)|^2 and Delta-omega of |FFT(x)|^2 (energy-weighted)."""
    I = np.abs(x) ** 2
    I = I / np.sum(I)
    t_mean = np.sum(t * I)
    dt = np.sqrt(np.sum((t - t_mean) ** 2 * I))

    dt_step = t[1] - t[0]
    X = np.fft.fftshift(np.fft.fft(np.fft.fftshift(x)))
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(t), d=dt_step))
    S = np.abs(X) ** 2
    S = S / np.sum(S)
    w_mean = np.sum(omega * S)
    domega = np.sqrt(np.sum((omega - w_mean) ** 2 * S))
    return float(dt), float(domega)


def uncertainty_vs_chirp(chirps, **pulse_kwargs):
    """Delta-t * Delta-omega for each chirp, using dgs.pulse_gen.generate_pulse."""
    from dgs.pulse_gen import generate_pulse
    products = []
    for c in chirps:
        t, x, _, _ = generate_pulse(chirp=c, **pulse_kwargs)
        dt, domega = pulse_time_freq_widths(t, x)
        products.append(dt * domega)
    return np.array(products)


# -- 3. Legendre / spherical harmonics / partial-wave angular weight -------------

def legendre(l, x):
    """Legendre polynomial P_l(x) via Bonnet's recursion."""
    if l < 0:
        raise ValueError("l must be >= 0")
    P_prev, P_curr = np.ones_like(x), x.copy() if l >= 1 else np.ones_like(x)
    if l == 0:
        return P_prev
    if l == 1:
        return P_curr
    for n in range(1, l):
        P_prev, P_curr = P_curr, ((2 * n + 1) * x * P_curr - n * P_prev) / (n + 1)
    return P_curr


def spherical_harmonic_intensity_m0(l, theta):
    """|Y_l^0(theta)|^2 = (2l+1)/(4 pi) * P_l(cos theta)^2 -- axisymmetric orbital."""
    P = legendre(l, np.cos(theta))
    return (2 * l + 1) / (4 * np.pi) * P ** 2


def spherical_bessel_jl(l, x):
    """Spherical Bessel j_l(x) via j_0=sinc, j_1, and the 3-term recurrence.

    j_{l+1}(x) = (2l+1)/x * j_l(x) - j_{l-1}(x).
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        j0 = np.where(np.abs(x) < 1e-8, 1.0, np.sin(x) / x)
        if l == 0:
            return j0
        j1 = np.where(np.abs(x) < 1e-8, 0.0, np.sin(x) / x ** 2 - np.cos(x) / x)
        if l == 1:
            return j1
        j_prev, j_curr = j0, j1
        for n in range(1, l):
            j_next = (2 * n + 1) / np.where(x == 0, 1e-30, x) * j_curr - j_prev
            j_prev, j_curr = j_curr, j_next
        return j_curr


def partial_wave_angular_weight(l, x):
    """(2l+1) |j_l(x)|^2 -- the multipole's share of scattered intensity at size
    parameter x. A simplified stand-in for a full Mie partial-wave sum, useful
    for comparing *which* l dominates against the hydrogen orbital of the same l."""
    return (2 * l + 1) * spherical_bessel_jl(l, x) ** 2


# -- 4. Partial-wave phase shifts and Born approximation --------------------------

def spherical_neumann_yl(l, x):
    """Spherical Neumann (Bessel of the 2nd kind) y_l(x), same recursion as j_l."""
    x = np.asarray(x, dtype=float)
    y0 = -np.cos(x) / x
    if l == 0:
        return y0
    y1 = -np.cos(x) / x ** 2 - np.sin(x) / x
    if l == 1:
        return y1
    y_prev, y_curr = y0, y1
    for n in range(1, l):
        y_next = (2 * n + 1) / x * y_curr - y_prev
        y_prev, y_curr = y_curr, y_next
    return y_curr


def hard_sphere_phase_shift(l, ka):
    """tan(delta_l) = j_l(ka) / y_l(ka) for scattering off an impenetrable sphere
    of radius a (Griffiths Ch.11 partial-wave analysis); returns delta_l in
    radians, wrapped to (-pi/2, pi/2]."""
    ka = np.asarray(ka, dtype=float)
    jl = spherical_bessel_jl(l, ka)
    yl = spherical_neumann_yl(l, ka)
    return np.arctan2(jl, yl)


def born_square_well_cross_section(theta, k, V0, a, hbar=1.0, m=1.0):
    """dsigma/dOmega in the Born approximation for a spherical square well
    V(r) = -V0 for r<a, 0 outside (Griffiths eq. 11.36-ish).

    f(theta) = (2 m V0) / (hbar^2 q^3) * [sin(qa) - qa cos(qa)],   q = 2k sin(theta/2)
    """
    theta = np.asarray(theta, dtype=float)
    q = 2 * k * np.sin(theta / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(
            np.abs(q) < 1e-8,
            (2 * m * V0 * a ** 3) / (3 * hbar ** 2),     # q->0 limit
            (2 * m * V0) / (hbar ** 2 * q ** 3) * (np.sin(q * a) - q * a * np.cos(q * a)),
        )
    return f ** 2
