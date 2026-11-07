"""Diffraction and dispersion are one equation: space-time duality.

In plain words: a laser beam spreading sideways as it travels (diffraction) and a
short light pulse spreading in time as it goes down a fiber (dispersion) obey the
SAME math. Swap "transverse position x" for "time T" and "distance z / wavenumber k"
for "fiber dispersion beta2 * L", and every diffraction result becomes a dispersion
result. Fraunhofer far-field diffraction (the far-field pattern is the Fourier
transform of the aperture) becomes the dispersive Fourier transform behind Jalali's
photonic time-stretch: after enough dispersion, the pulse's time trace IS its
spectrum. The Talbot self-imaging of a grating becomes the temporal Talbot effect of
a pulse train. (Kolner 1994 gives the general theory of this duality.)

The shared equation. Both the paraxial wave equation and the pulse envelope equation
reduce to
    du/dp = -(i/2) d^2u/dx^2 ,       transfer function  H(nu) = exp(i p nu^2 / 2),
with one "accumulated quadratic phase" p:
    diffraction:  x = transverse position,  nu = k_x,     p = -z / k   (k = 2 pi / lambda)
    dispersion:   x = retarded time T,      nu = omega,   p = beta2 * L
This is the same H(omega) = exp(i beta2 L omega^2 / 2) used by
dgs.dispersive_fourier.gvd_transfer_function, so the two modules agree on signs.
The equation is also the free-particle Schrodinger equation (Knight Ch. 40), with
p = -hbar t / m (psi~ evolves as exp(-i hbar k^2 t / 2m)): wave-packet spreading is pulse broadening.

Closed forms implemented and checked:
  * Gaussian u0 = exp(-x^2 / (2 s0^2)) stays Gaussian with complex width
    sigma^2 = s0^2 - i p, intensity width s(p) = s0 sqrt(1 + (p / s0^2)^2).
    The "Rayleigh / dispersion length" is |p| = s0^2  ->  z_R = k s0^2, L_D = s0^2/|beta2|.
  * Far field (|p| >> size^2): stationary phase gives u(x) ~ U(nu = -x/p), the
    Fraunhofer / dispersive-Fourier-transform map.
  * Talbot: a field with period Lambda revives at p = Lambda^2 / pi, i.e.
    z_T = 2 Lambda^2 / lambda (spatial) and L_T = T^2 / (pi |beta2|) (temporal);
    at half that, it reappears shifted by Lambda/2 (because m^2 and m have the same parity).

Vectorization: quadratic_phase_propagate takes an array of p values and returns
every propagated field at once via NumPy broadcasting (one FFT call), which is how the
Talbot carpet is computed. The direct Fresnel convolution is provided as a pure-Python
loop and as a broadcast matrix product so the speedups can be timed. NumPy + SymPy.
Education.
"""

import numpy as np
import sympy as sp


def _check_grid(u, d):
    u = np.asarray(u, dtype=complex)
    if u.ndim != 1 or u.size < 2:
        raise ValueError("u must be a 1D array with at least 2 samples")
    if not np.isfinite(d) or d <= 0:
        raise ValueError(f"grid spacing d must be positive, got {d}")
    return u


def frequency_grid(n, d):
    """Angular frequency (or k_x) axis nu = 2 pi fftfreq(n, d) matching np.fft.fft ordering."""
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    return 2 * np.pi * np.fft.fftfreq(n, d)


def transfer_function(nu, p):
    """H(nu) = exp(i p nu^2 / 2): diffraction (p = -z/k) and dispersion (p = beta2 L) alike."""
    return np.exp(0.5j * np.asarray(p, float) * np.asarray(nu, float) ** 2)


def quadratic_phase_propagate(u, d, p):
    """Propagate a 1D field through accumulated quadratic phase p (FFT method, periodic).

    p may be a scalar (returns shape (n,)) or a 1D array of m values (returns shape
    (m, n), one row per p) -- the vectorized form used for Talbot carpets.
    """
    u = _check_grid(u, d)
    p_arr = np.asarray(p, float)
    if p_arr.ndim > 1 or not np.all(np.isfinite(p_arr)):
        raise ValueError("p must be a finite scalar or 1D array")
    nu = frequency_grid(u.size, d)
    H = transfer_function(nu, p_arr[..., None])          # broadcast: (m, n) or (n,)
    return np.fft.ifft(np.fft.fft(u) * H, axis=-1)


def diffraction_p(z, lam):
    """Accumulated quadratic phase for free-space paraxial diffraction: p = -z / k."""
    if lam <= 0:
        raise ValueError(f"wavelength must be positive, got {lam}")
    return -np.asarray(z, float) * lam / (2 * np.pi)


def dispersion_p(beta2, L):
    """Accumulated quadratic phase for a dispersive fiber: p = beta2 * L (GDD)."""
    if np.any(np.asarray(L) < 0):
        raise ValueError("fiber length L must be non-negative")
    return np.asarray(beta2, float) * np.asarray(L, float)


def gaussian_field(x, s0):
    """u0(x) = exp(-x^2 / (2 s0^2)); intensity exp(-x^2/s0^2)."""
    if s0 <= 0:
        raise ValueError(f"s0 must be positive, got {s0}")
    return np.exp(-np.asarray(x, float) ** 2 / (2 * s0 ** 2)).astype(complex)


def gaussian_exact(x, s0, p):
    """Exact solution for a Gaussian input: (s0/sigma) exp(-x^2/(2 sigma^2)), sigma^2 = s0^2 - i p."""
    if s0 <= 0:
        raise ValueError(f"s0 must be positive, got {s0}")
    sig2 = s0 ** 2 - 1j * np.asarray(p, float)
    x = np.asarray(x, float)
    return s0 / np.sqrt(sig2) * np.exp(-x ** 2 / (2 * sig2))


def gaussian_width(s0, p):
    """Intensity 1/e half-width after propagation: s(p) = s0 sqrt(1 + (p/s0^2)^2)."""
    if s0 <= 0:
        raise ValueError(f"s0 must be positive, got {s0}")
    return s0 * np.sqrt(1 + (np.asarray(p, float) / s0 ** 2) ** 2)


def characteristic_p(s0):
    """|p| at which the Gaussian width has grown by sqrt(2): p = s0^2 (Rayleigh / dispersion length)."""
    if s0 <= 0:
        raise ValueError(f"s0 must be positive, got {s0}")
    return s0 ** 2


def rms_width(x, u):
    """RMS width of |u|^2 along the last axis; for intensity exp(-x^2/s^2) this is s/sqrt(2)."""
    x = np.asarray(x, float)
    I = np.abs(np.asarray(u)) ** 2
    norm = I.sum(axis=-1)
    mean = (I * x).sum(axis=-1) / norm
    return np.sqrt((I * x ** 2).sum(axis=-1) / norm - mean ** 2)


def far_field_coordinate(nu, p):
    """Stationary-phase map: frequency nu lands at x = -p nu (Fraunhofer angle / time-stretch delay)."""
    return -np.asarray(p, float) * np.asarray(nu, float)


def far_field_prediction(u, d, x, p):
    """|u(x, p)|^2 predicted from the input spectrum alone: |U(-x/p)|^2 * d^2 / (2 pi |p|)."""
    u = _check_grid(u, d)
    if p == 0:
        raise ValueError("far field needs p != 0")
    nu = np.fft.fftshift(frequency_grid(u.size, d))
    S = np.abs(np.fft.fftshift(np.fft.fft(u))) ** 2
    return np.interp(-np.asarray(x, float) / p, nu, S) * d ** 2 / (2 * np.pi * abs(p))


def fresnel_number(size, p):
    """N_F = size^2 / |p|: >> 1 near field, << 1 far field (Fraunhofer / dispersive Fourier transform)."""
    if p == 0:
        raise ValueError("p must be non-zero")
    return size ** 2 / abs(p)


def talbot_p(period):
    """Revival quadratic phase p_T = Lambda^2 / pi for a field of period Lambda."""
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    return period ** 2 / np.pi


def talbot_length_diffraction(period, lam):
    """Spatial Talbot length z_T = 2 Lambda^2 / lambda."""
    if lam <= 0:
        raise ValueError(f"wavelength must be positive, got {lam}")
    return 2 * talbot_p(period) * np.pi / lam


def talbot_length_dispersion(period, beta2):
    """Temporal Talbot length L_T = T^2 / (pi |beta2|) for a pulse train of period T."""
    if beta2 == 0:
        raise ValueError("beta2 must be non-zero")
    return talbot_p(period) / abs(beta2)


def fresnel_kernel(x, p):
    """Impulse response of du/dp = -(i/2) u_xx: exp(-i x^2/(2p)) / sqrt(-2 pi i p)."""
    if p == 0:
        raise ValueError("p must be non-zero")
    return np.exp(-1j * np.asarray(x, float) ** 2 / (2 * p)) / np.sqrt(-2j * np.pi * p)


def fresnel_direct_loop(u, x, p):
    """Direct Fresnel convolution with Python loops: O(n^2) interpreted operations."""
    u = _check_grid(u, x[1] - x[0])
    d = float(x[1] - x[0])
    c = 1 / np.sqrt(-2j * np.pi * p)
    out = np.zeros(u.size, complex)
    xs = [float(v) for v in x]
    us = [complex(v) for v in u]
    for i in range(len(xs)):
        acc = 0j
        for j in range(len(xs)):
            dx = xs[i] - xs[j]
            acc += us[j] * complex(np.cos(dx * dx / (2 * p)), -np.sin(dx * dx / (2 * p)))
        out[i] = c * acc * d
    return out


def fresnel_direct_vectorized(u, x, p):
    """Same convolution, vectorized: build the (n, n) kernel matrix by broadcasting, one matvec."""
    u = _check_grid(u, x[1] - x[0])
    x = np.asarray(x, float)
    K = fresnel_kernel(x[:, None] - x[None, :], p)
    return K @ u * (x[1] - x[0])


def sympy_duality():
    """Symbolic results: PDE residual of the Gaussian solution, width, stationary point,
    Talbot condition, and the diffraction / dispersion substitutions."""
    x, nu = sp.symbols("x nu", real=True)
    p = sp.symbols("p", real=True)
    s0, k, z, lam, beta2, L, T0, Lam, m = sp.symbols(
        "s_0 k z lambda beta_2 L T_0 Lambda m", positive=True)
    sig2 = s0 ** 2 - sp.I * p
    u = s0 / sp.sqrt(sig2) * sp.exp(-x ** 2 / (2 * sig2))
    residual = sp.simplify(sp.diff(u, p) + sp.I / 2 * sp.diff(u, x, 2))
    # intensity exponent: |u|^2 ~ exp(-x^2 * 2 Re(1/(2 sigma^2))) = exp(-x^2 / s^2)
    inv_s2 = sp.simplify(sp.re(sp.expand_complex(1 / sig2)))
    s_of_p = sp.sqrt(sp.simplify(1 / inv_s2))
    phase = p * nu ** 2 / 2 + nu * x
    nu_star = sp.solve(sp.diff(phase, nu), nu)[0]
    # Talbot: every harmonic nu_m = 2 pi m / Lambda picks up a multiple of 2 pi
    p_T = sp.symbols("p_T", positive=True)
    talbot_phase = sp.simplify(p_T * (2 * sp.pi * m / Lam) ** 2 / 2)
    p_T_sol = sp.solve(sp.Eq(talbot_phase.subs(m, 1), 2 * sp.pi), p_T)[0]
    w_beam = s_of_p.subs(p, -z / k)
    w_pulse = s_of_p.subs({p: beta2 * L, s0: T0})
    return {
        "u": u, "residual": residual, "s_of_p": s_of_p, "nu_star": nu_star,
        "talbot_phase": talbot_phase, "p_T": p_T_sol,
        "z_T": sp.simplify((p_T_sol * k).subs(k, 2 * sp.pi / lam)),
        "w_beam": w_beam, "z_R": sp.solve(sp.Eq(s_of_p.subs(p, z / k) ** 2, 2 * s0 ** 2), z)[0],
        "T_pulse": w_pulse, "L_D": sp.solve(sp.Eq(w_pulse ** 2, 2 * T0 ** 2), L)[0],
        "symbols": dict(x=x, nu=nu, p=p, s0=s0, k=k, z=z, lam=lam, beta2=beta2, L=L,
                        T0=T0, Lam=Lam, m=m),
    }
