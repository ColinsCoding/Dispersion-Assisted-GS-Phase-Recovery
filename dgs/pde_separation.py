"""Separation of variables for PDEs in electrodynamics and remote sensing.

The method of separation of variables assumes the solution factors:
    Phi(r, theta, phi) = R(r) * Theta(theta) * Phi(phi)

Substituting into the PDE produces three ODEs, one per coordinate.
Each ODE introduces a separation constant; the boundary conditions quantise
these constants into discrete eigenvalues (quantum numbers l, m, n, ...).

This is the same mathematics as quantum mechanics: Schrodinger's equation in
spherical coordinates separates into the same radial equation as Laplace's
equation (with the effective potential shifted), and the same spherical harmonics
Y_l^m serve as angular eigenfunctions in both.

Coverage:
  -- Legendre polynomials P_l(x) and associated P_l^m(x)
  -- Spherical harmonics Y_l^m(theta, phi)
  -- Laplace equation in spherical coordinates (Griffiths Ch 3)
  -- Helmholtz equation (wave equation, time-harmonic EM)
  -- Cylindrical separation: Bessel functions J_n, N_n
  -- Multipole expansion of a charge distribution
  -- 4-cylinder-on-sphere geometry: symmetry breaking via azimuthal modes
  -- Dispersive Fourier Transform (Jalali lab): PDE view of time-stretch

Griffiths mapping:
  Ch 3 Sec 3.3  -- separation of variables (Laplace in spherical)
  Ch 3 Sec 3.4  -- multipole expansion
  Ch 9 Sec 9.2  -- EM waves: wave equation -> Helmholtz
  Ch 4 Sec 4.1  -- polarisation: Laplace with dielectric boundary

Usage:
    from dgs.pde_separation import (
        legendre_P, legendre_Pm, spherical_harmonic_Y,
        laplace_spherical_solution, multipole_expansion,
        bessel_J, cylindrical_laplace_mode,
        four_cylinder_sphere_modes, dispersive_ft_pde
    )
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from math import factorial


# ---------------------------------------------------------------------------
# Legendre polynomials  (Griffiths Ch 3 Eq 3.65)
# ---------------------------------------------------------------------------

def legendre_P(l: int, x: np.ndarray) -> np.ndarray:
    """Legendre polynomial P_l(x) via Bonnet's recursion.

    P_0(x) = 1
    P_1(x) = x
    (l+1) P_{l+1}(x) = (2l+1) x P_l(x) - l P_{l-1}(x)

    These are the angular solutions to Laplace's equation with azimuthal
    symmetry (Griffiths 3.3).  They are orthogonal on [-1,1]:
        int_{-1}^{1} P_l(x) P_{l'}(x) dx = 2/(2l+1) * delta_{l,l'}

    Parameters
    ----------
    l : degree (non-negative integer)
    x : evaluation points (array)

    Returns
    -------
    P_l(x) evaluated at x.
    """
    x = np.asarray(x, dtype=float)
    if l == 0:
        return np.ones_like(x)
    if l == 1:
        return x.copy()
    P_prev2 = np.ones_like(x)
    P_prev1 = x.copy()
    for k in range(1, l):
        P_curr = ((2*k + 1) * x * P_prev1 - k * P_prev2) / (k + 1)
        P_prev2 = P_prev1
        P_prev1 = P_curr
    return P_prev1


def legendre_Pm(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """Associated Legendre function P_l^m(x), m >= 0.

    Computed by the standard recurrence from P_m^m and P_{m+1}^m.
    The Condon-Shortley phase convention is NOT included here (matches
    Griffiths Table 3.5 which also omits it).

    P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^{m/2}
    P_{m+1}^m(x) = x (2m+1) P_m^m(x)
    P_l^m(x) = (x(2l-1) P_{l-1}^m(x) - (l+m-1) P_{l-2}^m(x)) / (l-m)

    Parameters
    ----------
    l, m : degree and order (0 <= m <= l)
    x    : evaluation points (cos(theta), values in [-1,1])
    """
    if m < 0 or m > l:
        raise ValueError(f"Must have 0 <= m <= l; got l={l}, m={m}")
    x = np.asarray(x, dtype=float)

    # Compute P_m^m
    Pmm = np.ones_like(x)
    if m > 0:
        factor = 1.0
        for i in range(1, m + 1):
            factor *= -(2*i - 1) * np.sqrt(1 - x**2)
        Pmm = np.ones_like(x)
        for i in range(1, m + 1):
            Pmm *= -(2*i - 1) * np.sqrt(np.maximum(1 - x**2, 0))
    # Actually simpler: P_m^m = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
    Pmm_scalar = (-1)**m
    for i in range(1, m + 1):
        Pmm_scalar *= (2*i - 1)
    Pmm = Pmm_scalar * (np.maximum(1 - x**2, 0))**(m/2)

    if l == m:
        return Pmm
    Pm1m = x * (2*m + 1) * Pmm
    if l == m + 1:
        return Pm1m
    P_prev2 = Pmm
    P_prev1 = Pm1m
    for k in range(m + 1, l):
        P_curr = (x * (2*k + 1) * P_prev1 - (k + m) * P_prev2) / (k - m + 1)
        P_prev2 = P_prev1
        P_prev1 = P_curr
    return P_prev1


def legendre_norm(l: int, m: int) -> float:
    """Normalisation constant for real spherical harmonics.

    N_l^m = sqrt( (2l+1)/(4*pi) * (l-|m|)! / (l+|m|)! )
    """
    m = abs(m)
    num = (2*l + 1) * factorial(l - m)
    den = 4 * np.pi * factorial(l + m)
    return np.sqrt(num / den)


# ---------------------------------------------------------------------------
# Spherical harmonics Y_l^m(theta, phi)  (real form, Griffiths Table 3.5)
# ---------------------------------------------------------------------------

def spherical_harmonic_Y(l: int, m: int,
                          theta: np.ndarray,
                          phi: np.ndarray) -> np.ndarray:
    """Real spherical harmonic Y_l^m(theta, phi).

    Uses the real (tesseral) form:
        Y_l^0   = N_l^0 * P_l^0(cos theta)
        Y_l^m   = sqrt(2) * N_l^m * P_l^m(cos theta) * cos(m*phi)   (m > 0)
        Y_l^{-m}= sqrt(2) * N_l^m * P_l^m(cos theta) * sin(m*phi)   (m > 0)

    These satisfy:
        int Y_l^m Y_{l'}^{m'} sin(theta) dtheta dphi = delta_{l,l'} delta_{m,m'}

    Parameters
    ----------
    l     : degree (l >= 0)
    m     : order (-l <= m <= l)
    theta : polar angle from z-axis (radians, array)
    phi   : azimuthal angle (radians, array)

    Returns
    -------
    Y_l^m(theta, phi) as a real array.
    """
    theta = np.asarray(theta, dtype=float)
    phi   = np.asarray(phi,   dtype=float)
    cos_theta = np.cos(theta)
    m_abs = abs(m)

    if m_abs > l:
        raise ValueError(f"|m| must be <= l; got l={l}, m={m}")

    N   = legendre_norm(l, m_abs)
    Plm = legendre_Pm(l, m_abs, cos_theta)

    if m == 0:
        return N * Plm
    elif m > 0:
        return np.sqrt(2) * N * Plm * np.cos(m * phi)
    else:
        return np.sqrt(2) * N * Plm * np.sin(m_abs * phi)


# ---------------------------------------------------------------------------
# Laplace equation in spherical coordinates  (Griffiths Ch 3.3)
# ---------------------------------------------------------------------------

def laplace_spherical_solution(
    A_coefs: List[float],
    B_coefs: List[float],
    r: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """General azimuthally-symmetric solution to Laplace's equation.

    Phi(r, theta) = sum_{l=0}^{L} (A_l r^l + B_l r^{-(l+1)}) P_l(cos theta)

    This is Griffiths Eq 3.65.  The A_l terms are regular at r=0 (use for
    interior problems), the B_l terms are regular at r->infinity (use for
    exterior problems or multipoles at origin).

    Parameters
    ----------
    A_coefs : list of A_l coefficients (length L+1)
    B_coefs : list of B_l coefficients (length L+1)
    r       : radial coordinates (array)
    theta   : polar angles (array, same shape as r)

    Returns
    -------
    Phi(r, theta) array.
    """
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    L = len(A_coefs) - 1
    result = np.zeros_like(r)
    x = np.cos(theta)
    for l in range(L + 1):
        Pl = legendre_P(l, x)
        result += (A_coefs[l] * r**l + B_coefs[l] * r**(-(l+1))) * Pl
    return result


def conducting_sphere_in_uniform_E(
    E0: float,
    R: float,
    r: np.ndarray,
    theta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Electric potential and field for a grounded conducting sphere in uniform E0.

    Solution (Griffiths Ex 3.8):
        Phi = -E0 r cos(theta) + E0 R^3 / r^2 * cos(theta)   (for r > R)

    The induced surface charge creates a dipole field that exactly cancels E0
    inside the sphere.  The field lines bend around the sphere.

    Parameters
    ----------
    E0    : applied field strength (V/m)
    R     : sphere radius (m)
    r     : observation radii (array, r > R)
    theta : observation angles (array)

    Returns
    -------
    (Phi, E_r, E_theta): potential and field components
    """
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    Phi = -E0 * (r - R**3 / r**2) * cos_t

    E_r     = E0 * (1 + 2*R**3/r**3) * cos_t
    E_theta = -E0 * (1 - R**3/r**3) * sin_t

    return Phi, E_r, E_theta


# ---------------------------------------------------------------------------
# Multipole expansion  (Griffiths Ch 3.4)
# ---------------------------------------------------------------------------

def multipole_expansion(
    charges: List[float],
    positions: np.ndarray,
    r_obs: float,
    theta_obs: np.ndarray,
    l_max: int = 5,
) -> np.ndarray:
    """Multipole expansion of a discrete charge distribution.

    V(r, theta) = 1/(4*pi*eps0) * sum_{l=0}^{l_max} 1/r^{l+1}
                  * int r'^l P_l(cos alpha) rho dV'

    For point charges:
        V = 1/(4*pi*eps0) * sum_l (1/r^{l+1}) * sum_i q_i r_i^l P_l(cos alpha_i)

    where alpha_i is the angle between r_obs direction and r_i.

    Parameters
    ----------
    charges   : list of point charges (C)
    positions : array of charge positions, shape (N, 3), [x, y, z]
    r_obs     : observation distance (m, must be > max(|r_i|))
    theta_obs : observation angles from z-axis (array, radians)
    l_max     : maximum multipole order

    Returns
    -------
    V(r_obs, theta_obs) in units of 1/(4*pi*eps0) = 9e9 (V.m/C)
    """
    K = 8.9875e9   # 1/(4*pi*eps0) in SI

    positions = np.asarray(positions, dtype=float)
    theta_obs = np.asarray(theta_obs, dtype=float)
    r_i = np.linalg.norm(positions, axis=1)         # |r_i|
    theta_i = np.arccos(np.clip(positions[:, 2] / (r_i + 1e-30), -1, 1))  # angle from z

    V = np.zeros_like(theta_obs)
    cos_obs = np.cos(theta_obs)

    for l in range(l_max + 1):
        moment = np.sum([
            charges[i] * r_i[i]**l * legendre_P(l, np.cos(theta_i[i]))
            for i in range(len(charges))
        ])
        V += moment / r_obs**(l + 1) * legendre_P(l, cos_obs)

    return K * V


# ---------------------------------------------------------------------------
# Cylindrical separation: Bessel functions via series
# ---------------------------------------------------------------------------

def bessel_J(n: int, x: np.ndarray, n_terms: int = 30) -> np.ndarray:
    """Bessel function of the first kind J_n(x) via power series.

    J_n(x) = sum_{k=0}^{inf} (-1)^k / (k! Gamma(n+k+1)) * (x/2)^{2k+n}

    This is the radial solution for cylindrical Laplace and Helmholtz equations:
        r^2 R'' + r R' + (k^2 r^2 - n^2) R = 0

    J_n is regular at r=0.  The Neumann function N_n is singular at r=0
    and is used only for problems with a cylindrical void (r > 0 throughout).
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    for k in range(n_terms):
        num = (-1)**k * (x/2)**(2*k + n)
        den = factorial(k) * factorial(n + k)   # Gamma(n+k+1) = (n+k)! for integer n
        result += num / den
    return result


def cylindrical_laplace_mode(
    n: int,
    k_z: float,
    r: np.ndarray,
    phi: np.ndarray,
    z: np.ndarray,
    A_n: float = 1.0,
) -> np.ndarray:
    """One mode of the cylindrical Laplace/Helmholtz solution.

    Phi_{n,k}(r, phi, z) = J_n(k_z * r) * cos(n * phi) * exp(-k_z * z)

    This satisfies: (1/r d/dr r d/dr + 1/r^2 d^2/dphi^2 + d^2/dz^2) Phi = 0
    when k_z is real and J_n is the radial factor with transverse wavenumber k_z.

    Parameters
    ----------
    n    : azimuthal mode number (integer)
    k_z  : axial wavenumber (1/m)
    r, phi, z : cylindrical coordinates (arrays, same shape)
    A_n  : amplitude coefficient

    Returns
    -------
    Phi_n evaluated at (r, phi, z).
    """
    r   = np.asarray(r,   dtype=float)
    phi = np.asarray(phi, dtype=float)
    z   = np.asarray(z,   dtype=float)
    return A_n * bessel_J(n, k_z * r) * np.cos(n * phi) * np.exp(-k_z * z)


# ---------------------------------------------------------------------------
# 4-cylinder-on-sphere: symmetry analysis and mode decomposition
# ---------------------------------------------------------------------------

def four_cylinder_sphere_modes(
    R_sphere: float = 1.0,
    R_cyl: float = 0.2,
    L_cyl: float = 0.5,
    n_modes: int = 6,
    n_phi: int = 360,
) -> dict:
    """Decompose the 4-cylinder-on-sphere boundary into spherical harmonic modes.

    Geometry: a sphere of radius R_sphere with 4 cylindrical protrusions
    attached at the equator at phi = 0, pi/2, pi, 3pi/2.  This is like a
    4-legged robot (or a 4-antenna array).

    The boundary condition Phi = V0 on the cylinders and Phi = 0 on the sphere
    breaks the azimuthal symmetry from continuous O(2) to discrete C4 symmetry.

    Only azimuthal modes m = 0, 4, 8, 12, ... (multiples of 4) are excited
    because the boundary has 4-fold symmetry.  This is a fundamental result
    of group theory: the Fourier decomposition of a C4-symmetric function
    contains only harmonics that are also C4-symmetric.

    Returns the phi-dependent boundary condition and its Fourier decomposition.
    """
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    # Boundary condition: V = 1 where cylinders touch sphere, 0 elsewhere
    # Cylinders at phi = 0, pi/2, pi, 3pi/2 with angular half-width = R_cyl/R_sphere
    delta_phi = R_cyl / R_sphere   # half-angular width of each cylinder footprint
    bc = np.zeros(n_phi)
    for phi_cyl in [0, np.pi/2, np.pi, 3*np.pi/2]:
        # Mark cylinder footprint
        diff = np.abs(np.angle(np.exp(1j * (phi - phi_cyl))))
        bc[diff < delta_phi] = 1.0

    # Fourier decomposition
    bc_fft = np.fft.rfft(bc) / n_phi
    freqs  = np.fft.rfftfreq(n_phi, d=1.0/n_phi)   # integer azimuthal modes

    # Extract modes m = 0, 4, 8, ... (C4-symmetric)
    c4_modes = {}
    for m in range(0, n_modes * 4, 4):
        idx = np.where(np.abs(freqs - m) < 0.5)[0]
        if len(idx) > 0:
            c4_modes[m] = abs(bc_fft[idx[0]])

    return {
        'phi': phi,
        'bc': bc,
        'fft_coeffs': bc_fft,
        'freqs': freqs,
        'c4_modes': c4_modes,
        'symmetry': 'C4 -- only m=0,4,8,12... modes are non-zero',
        'geometry': f'Sphere R={R_sphere}, 4 cylinders at equator phi=0,pi/2,pi,3pi/2',
    }


# ---------------------------------------------------------------------------
# Dispersive Fourier Transform: PDE view (Jalali lab)
# ---------------------------------------------------------------------------

def dispersive_ft_pde(
    E0: np.ndarray,
    t: np.ndarray,
    beta2: float,
    z: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispersive Fourier Transform: solve the GVD PDE for a pulse.

    The GVD equation (Jalali lab, dispersive Fourier transform):
        dE/dz = i*beta2/2 * d^2E/dt^2

    is solved exactly in the frequency domain:
        E_hat(omega, z) = E_hat(omega, 0) * exp(i*beta2/2 * omega^2 * z)

    In the far-field limit (|beta2*z| >> tau^2 where tau is pulse width),
    the output intensity maps the INPUT SPECTRUM:
        |E(t, z)|^2 -> |E_hat(omega)|^2  with  t = beta2*z*omega

    This is the DFT principle used in Jalali lab's time-stretch ADC and
    STEAM (Serial Time-Encoded Amplified Microscopy): a chirped pulse
    encodes the spectrum as a time waveform, which can be digitised.

    Parameters
    ----------
    E0    : input field E(t, z=0), complex array
    t     : time array (s)
    beta2 : GVD coefficient (s^2/m)
    z     : propagation distance (m)

    Returns
    -------
    (E_out, omega): output field and frequency array
    """
    N  = len(t)
    dt = t[1] - t[0]
    omega = np.fft.fftfreq(N, d=dt) * 2 * np.pi

    E0_hat  = np.fft.fft(E0)
    H_omega = np.exp(1j * beta2/2 * omega**2 * z)   # GVD transfer function
    E_out   = np.fft.ifft(E0_hat * H_omega)

    return E_out, omega


def dft_far_field_condition(
    tau_s: float,
    beta2: float,
    z: float,
) -> dict:
    """Check whether the DFT far-field (time-to-frequency mapping) condition holds.

    The far-field condition for the dispersive Fourier transform:
        |beta2 * z| >> tau^2

    where tau is the pulse duration.  When satisfied, |E(t,z)|^2 is a
    scaled copy of the spectrum |E_hat(omega)|^2.

    Parameters
    ----------
    tau_s  : pulse duration (s, 1/e half-width)
    beta2  : GVD coefficient (s^2/m, typical fiber: -21.7e-27 s^2/m)
    z      : propagation distance (m)

    Returns
    -------
    dict with dispersion_length, far_field_ratio, is_far_field
    """
    L_D = tau_s**2 / abs(beta2)        # dispersion length: pulse broadens by sqrt(2)
    far_field_ratio = abs(beta2 * z) / tau_s**2
    return {
        'dispersion_length_m': L_D,
        'far_field_ratio': far_field_ratio,
        'is_far_field': far_field_ratio > 10,    # >10 is conservative criterion
        'z_min_for_dft_m': 10 * tau_s**2 / abs(beta2),
        'time_to_freq_slope': beta2 * z,          # t = beta2*z*omega
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt

    print("Separation of Variables -- Demo")
    print("=" * 55)

    # 1. Legendre polynomials
    x = np.linspace(-1, 1, 300)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Separation of Variables in Electrodynamics", fontsize=12)

    ax = axes[0]
    for l in range(6):
        ax.plot(x, legendre_P(l, x), lw=1.5, label=f'P_{l}')
    ax.set_xlabel("x = cos(theta)")
    ax.set_ylabel("P_l(x)")
    ax.set_title("Legendre polynomials P_l(cos theta)\n(angular solutions, Griffiths Eq 3.65)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.axhline(0, color='k', lw=0.5)

    # 2. Conducting sphere in uniform E0
    r_vals = np.linspace(1.01, 4, 200)
    theta_vals = np.linspace(0, np.pi, 200)
    RR, TT = np.meshgrid(r_vals, theta_vals)
    Phi, _, _ = conducting_sphere_in_uniform_E(E0=1.0, R=1.0, r=RR, theta=TT)
    XX = RR * np.sin(TT)
    ZZ = RR * np.cos(TT)
    ax = axes[1]
    c = ax.contourf(XX, ZZ, Phi, levels=30, cmap='RdBu_r')
    theta_circ = np.linspace(0, np.pi, 200)
    ax.fill_between(np.sin(theta_circ), -np.cos(theta_circ), color='gray', alpha=0.7)
    plt.colorbar(c, ax=ax, label='Phi (V)')
    ax.set_xlabel("x/R"); ax.set_ylabel("z/R")
    ax.set_title("Conducting sphere in E0\n(Griffiths Ex 3.8: Phi = -E0(r-R^3/r^2)cos theta)")
    ax.set_aspect('equal')

    # 3. 4-cylinder-on-sphere symmetry
    ax = axes[2]
    result = four_cylinder_sphere_modes()
    ax.plot(result['phi'] * 180/np.pi, result['bc'], 'steelblue', lw=2,
            label='Boundary V(phi)')
    ax.set_xlabel("phi (degrees)")
    ax.set_ylabel("V (normalised)")
    ax.set_title("4-cylinder boundary condition\n(C4 symmetry: only m=0,4,8,... excited)")
    ax.set_xticks([0,90,180,270,360])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    modes = result['c4_modes']
    m_vals = list(modes.keys())
    a_vals = list(modes.values())
    ax2.bar(np.array(m_vals)*360/(4*max(m_vals+[1]))+20, a_vals, width=6,
            color='tomato', alpha=0.6, label='C4 mode amplitude')
    ax2.set_ylabel("Mode amplitude", color='tomato')

    plt.tight_layout()
    plt.savefig("pde_separation_demo.png", dpi=120, bbox_inches='tight')
    plt.show()

    # 4. Legendre orthogonality verification
    print("\nLegendre orthogonality check:")
    print(f"  int_{{-1}}^{{1}} P_l P_{{l'}} dx = 2/(2l+1) * delta_{{l,l'}}")
    x_fine = np.linspace(-1, 1, 10000)
    dx = x_fine[1] - x_fine[0]
    for l in range(4):
        for lp in range(4):
            inner = np.trapz(legendre_P(l, x_fine) * legendre_P(lp, x_fine), x_fine)
            expected = 2/(2*l+1) if l == lp else 0
            print(f"  <P_{l}, P_{lp}> = {inner:.4f}  (expect {expected:.4f})")

    # 5. DFT far-field condition for Jalali lab setup
    print("\nJalali lab DFT far-field condition:")
    # Typical STEAM: tau = 1 ns pulse, beta2 = -1200 ps^2/km = -1.2e-24 s^2/m, z = 10 km
    configs = [
        ('Fiber STEAM (Jalali 2009)', 1e-9, -1.2e-24, 10e3),
        ('GHz ADC (tau=100ps)',       1e-10, -1.2e-24, 1e3),
        ('Optical rogue wave (1ps)',  1e-12, -21.7e-27, 1.0),
    ]
    for label, tau, b2, z in configs:
        r = dft_far_field_condition(tau, b2, z)
        ff = "YES" if r['is_far_field'] else "NO"
        print(f"  {label}:")
        print(f"    L_D = {r['dispersion_length_m']:.2e} m  "
              f"ratio = {r['far_field_ratio']:.1f}  far-field: {ff}")

    # 6. 4-cylinder mode decomposition
    print("\n4-cylinder sphere: Fourier mode amplitudes")
    print(f"  Geometry: {result['geometry']}")
    print(f"  Symmetry: {result['symmetry']}")
    for m, amp in sorted(result['c4_modes'].items()):
        print(f"  m = {m:2d}: amplitude = {amp:.5f}")


if __name__ == "__main__":
    demo()
