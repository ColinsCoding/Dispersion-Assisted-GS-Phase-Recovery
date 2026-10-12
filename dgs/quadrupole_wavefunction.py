"""The l = 2 hydrogen wavefunctions as surfaces, and why they are "quadrupoles".

An electron in a d state (l = 2) has an angular wavefunction Y_2^m(theta, phi).
Its probability cloud is not spherical, so far away it produces not only the
1/r potential of a point charge but also a 1/r^3 QUADRUPOLE term -- the same
term, with the same angular shape P_2(cos theta), that four point charges
produce in electrostatics (dgs.electrostatics_multipoles). This module makes
that link computable:

  spherical_harmonic      Y_l^m, complex, Condon-Shortley phase
  real_d_orbital          the five real combinations chemists draw (d_z2, d_xz, ...)
  radial_wavefunction     hydrogen R_nl(r)
  hydrogen_psi            psi_nlm = R_nl * Y_l^m
  orbital_surface         the surface r = |Y(theta, phi)| as a mesh (X, Y, Z, sign)
  quadrupole_moment_zz    Q_zz = <3 z^2 - r^2> in closed form
  quadrupole_tensor       the full tensor <3 x_i x_j - r^2 delta_ij> by quadrature
  orbital_potential       exact electrostatic potential of the cloud |psi|^2
  far_field_potential     monopole + quadrupole approximation of the same

Units: lengths in units of the Bohr radius `a` (default 1), charge in units
of the particle's charge (multiply by -e for an electron), potential in units
of charge / (4 pi eps0 a). Sign and tensor conventions match
dgs.electrostatics_multipoles: Theta_ij = integral rho (3 x_i x_j - r^2 delta_ij).

Reading (cited in the companion notebook): Griffiths & Schroeter,
Introduction to Quantum Mechanics, 3rd ed., Sec. 4.1.2 (spherical harmonics)
and Sec. 4.2 (hydrogen); Griffiths, Introduction to Electrodynamics, 4th ed.,
Sec. 3.4 (multipole expansion).

What the pictures are NOT: the surface r = |Y| is a plot of an angular
function, not the "shape of the electron", and not a surface the electron
sits on. NumPy + SciPy; py-3.13.
"""
import math

import numpy as np
from scipy import special

D_ORBITALS = ("z2", "xz", "yz", "xy", "x2-y2")


# ----------------------------------------------------------------------
# 1. Angular part
# ----------------------------------------------------------------------

def _check_lm(l, m):
    if int(l) != l or int(m) != m or l < 0 or abs(m) > l:
        raise ValueError(f"need integers with l >= 0 and |m| <= l, got l={l}, m={m}")


def spherical_harmonic(l, m, theta, phi):
    """Complex Y_l^m(theta, phi); theta is the polar angle from +z.

        Y_l^m = sqrt((2l+1)/(4 pi) * (l-m)!/(l+m)!) * P_l^m(cos theta) * exp(i m phi)

    for m >= 0, with the Condon-Shortley phase (-1)^m inside P_l^m, and
    Y_l^{-m} = (-1)^m conj(Y_l^m). Orthonormal over the sphere.
    """
    _check_lm(l, m)
    theta, phi = np.asarray(theta, dtype=float), np.asarray(phi, dtype=float)
    ma = abs(int(m))
    norm = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - ma) / math.factorial(l + ma))
    y = norm * special.lpmv(ma, int(l), np.cos(theta)) * np.exp(1j * ma * phi)
    return y if m >= 0 else (-1) ** ma * np.conj(y)


def real_d_orbital(name, theta, phi):
    """The five real l = 2 functions, named by their Cartesian form.

        z2     (3z^2 - r^2)/r^2 :  Y_2^0
        xz     x z / r^2        :  (Y_2^-1 - Y_2^1) / sqrt(2)
        yz     y z / r^2        :  i (Y_2^-1 + Y_2^1) / sqrt(2)
        x2-y2  (x^2 - y^2)/r^2  :  (Y_2^2 + Y_2^-2) / sqrt(2)
        xy     x y / r^2        :  -i (Y_2^2 - Y_2^-2) / sqrt(2)

    Each is real and they are orthonormal over the sphere.
    """
    if name not in D_ORBITALS:
        raise ValueError(f"name must be one of {D_ORBITALS}, got {name!r}")
    y = lambda m: spherical_harmonic(2, m, theta, phi)
    root2 = math.sqrt(2.0)
    value = {
        "z2": lambda: y(0),
        "xz": lambda: (y(-1) - y(1)) / root2,
        "yz": lambda: 1j * (y(-1) + y(1)) / root2,
        "x2-y2": lambda: (y(2) + y(-2)) / root2,
        "xy": lambda: -1j * (y(2) - y(-2)) / root2,
    }[name]()
    return np.real(value)


def orbital_surface(angular, n_theta=91, n_phi=181):
    """Mesh of the surface r(theta, phi) = |angular(theta, phi)|.

    `angular` is any function of (theta, phi) arrays, real or complex.
    Returns X, Y, Z (each n_theta x n_phi) and `sign`: +1 / -1 where a real
    function is positive / negative (all +1 for a complex one), for colouring
    the lobes. The mesh has n_theta * n_phi vertices and
    (n_theta - 1) * (n_phi - 1) quadrilateral faces.
    """
    if n_theta < 3 or n_phi < 4:
        raise ValueError("need n_theta >= 3 and n_phi >= 4")
    theta, phi = np.meshgrid(np.linspace(0.0, np.pi, n_theta),
                             np.linspace(0.0, 2 * np.pi, n_phi), indexing="ij")
    value = np.asarray(angular(theta, phi))
    radius = np.abs(value)
    sign = np.sign(value.real) if np.isrealobj(value) else np.ones_like(radius)
    return (radius * np.sin(theta) * np.cos(phi), radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta), sign)


# ----------------------------------------------------------------------
# 2. Radial part and the full wavefunction
# ----------------------------------------------------------------------

def _check_nl(n, l):
    if int(n) != n or int(l) != l or n < 1 or not 0 <= l < n:
        raise ValueError(f"need integers with n >= 1 and 0 <= l < n, got n={n}, l={l}")


def radial_wavefunction(n, l, r, a=1.0):
    """Hydrogen R_nl(r), normalized so that integral R^2 r^2 dr = 1.

        R_nl = sqrt((2/(n a))^3 (n-l-1)! / (2n (n+l)!)) * exp(-rho/2) * rho^l * L_{n-l-1}^{2l+1}(rho),
        rho = 2 r / (n a),  L = associated Laguerre polynomial.
    """
    _check_nl(n, l)
    if a <= 0:
        raise ValueError("a must be positive")
    r = np.asarray(r, dtype=float)
    if np.any(r < 0):
        raise ValueError("r must be non-negative")
    rho = 2 * r / (n * a)
    norm = math.sqrt((2 / (n * a)) ** 3 * math.factorial(n - l - 1) / (2 * n * math.factorial(n + l)))
    return norm * np.exp(-rho / 2) * rho ** l * special.eval_genlaguerre(n - l - 1, 2 * l + 1, rho)


def hydrogen_psi(n, l, m, r, theta, phi, a=1.0):
    """psi_nlm(r, theta, phi) = R_nl(r) * Y_l^m(theta, phi)."""
    _check_nl(n, l)
    return radial_wavefunction(n, l, r, a) * spherical_harmonic(l, m, theta, phi)


def mean_r2(n, l, a=1.0):
    """<r^2> = (n^2 / 2) (5 n^2 + 1 - 3 l (l+1)) a^2 for hydrogen."""
    _check_nl(n, l)
    return n * n / 2 * (5 * n * n + 1 - 3 * l * (l + 1)) * a * a


def mean_cos2(l, m):
    """<cos^2 theta> = (2 l^2 + 2 l - 1 - 2 m^2) / ((2l - 1)(2l + 3)) in the state Y_l^m."""
    _check_lm(l, m)
    return (2 * l * l + 2 * l - 1 - 2 * m * m) / ((2 * l - 1) * (2 * l + 3))


# ----------------------------------------------------------------------
# 3. Quadrupole moment of the probability cloud
# ----------------------------------------------------------------------

def quadrupole_moment_zz(n, l, m, a=1.0):
    """Q_zz = <3 z^2 - r^2> = <r^2> (3 <cos^2 theta> - 1), per unit charge.

    Zero for every s state. Summed over all m of a filled subshell it is
    zero: a closed shell is spherical (Unsold's theorem).
    """
    return mean_r2(n, l, a) * (3 * mean_cos2(l, m) - 1)


def quadrupole_tensor(density, r_max, n_r=400, n_theta=64, n_phi=64):
    """Theta_ij = integral rho (3 x_i x_j - r^2 delta_ij) d^3r by quadrature.

    density(r, theta, phi) -> charge density (broadcastable arrays), assumed
    negligible beyond r_max. Gauss-Legendre in r and cos(theta), uniform in
    phi. Returns (Theta 3x3, total charge).
    """
    if r_max <= 0 or min(n_r, n_theta) < 4 or n_phi < 8:
        raise ValueError("need r_max > 0, n_r, n_theta >= 4 and n_phi >= 8")
    xr, wr = np.polynomial.legendre.leggauss(n_r)
    r, wr = 0.5 * r_max * (xr + 1), 0.5 * r_max * wr
    mu, wmu = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    R, MU, PHI = np.meshgrid(r, mu, phi, indexing="ij")
    theta = np.arccos(MU)
    weight = (wr[:, None, None] * wmu[None, :, None] * (2 * np.pi / n_phi)) * R ** 2
    rho = np.asarray(density(R, theta, PHI), dtype=float) * weight
    sin_t = np.sqrt(1 - MU ** 2)
    xyz = (R * sin_t * np.cos(PHI), R * sin_t * np.sin(PHI), R * MU)
    tensor = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            tensor[i, j] = np.sum(rho * (3 * xyz[i] * xyz[j] - (R ** 2 if i == j else 0.0)))
    return tensor, float(np.sum(rho))


# ----------------------------------------------------------------------
# 4. Electrostatic potential of the cloud |psi_nlm|^2
# ----------------------------------------------------------------------

def _legendre_coefficients(l, m, l_max):
    """c_L in |Y_l^m|^2 = sum_L c_L P_L(cos theta) (no phi dependence), L = 0 .. l_max."""
    mu, w = np.polynomial.legendre.leggauss(4 * l + 2 * l_max + 8)
    f = np.abs(spherical_harmonic(l, m, np.arccos(mu), 0.0)) ** 2
    return np.array([(2 * L + 1) / 2 * np.sum(w * f * special.eval_legendre(L, mu))
                     for L in range(l_max + 1)])


def orbital_potential(n, l, m, r, theta, a=1.0, n_grid=20001):
    """Exact potential of the charge cloud rho = |psi_nlm|^2 (unit total charge).

        V(r, theta) = sum_L  c_L  4 pi/(2L+1)  P_L(cos theta)
                      * [ r^-(L+1) int_0^r R^2 s^(L+2) ds  +  r^L int_r^inf R^2 s^(1-L) ds ]

    Only even L up to 2l contribute. In units of charge / (4 pi eps0 a).
    r must lie inside the integration grid (r <= 40 n^2 a).
    """
    _check_nl(n, l)
    _check_lm(l, m)
    r, theta = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(theta, dtype=float))
    s = np.linspace(0.0, 40.0 * n * n * a, n_grid)
    if np.any(r <= 0) or np.any(r > s[-1]):
        raise ValueError(f"r must be in (0, {s[-1]:g}]")
    r2 = radial_wavefunction(n, l, s, a) ** 2
    coeffs = _legendre_coefficients(l, m, 2 * l)
    total = np.zeros(r.shape)
    ds = s[1] - s[0]
    for L, c in enumerate(coeffs):
        if abs(c) < 1e-14:
            continue
        inner_integrand = r2 * s ** (L + 2)
        outer_integrand = np.zeros_like(s)
        outer_integrand[1:] = r2[1:] * s[1:] ** (1 - L)
        inner = np.concatenate([[0.0], np.cumsum(0.5 * (inner_integrand[1:] + inner_integrand[:-1]) * ds)])
        outer_cum = np.concatenate([[0.0], np.cumsum(0.5 * (outer_integrand[1:] + outer_integrand[:-1]) * ds)])
        outer = outer_cum[-1] - outer_cum
        radial = np.interp(r, s, inner) / r ** (L + 1) + r ** L * np.interp(r, s, outer)
        total += c * 4 * np.pi / (2 * L + 1) * special.eval_legendre(L, np.cos(theta)) * radial
    return total


def far_field_potential(n, l, m, r, theta, a=1.0):
    """Monopole + quadrupole terms of the same potential:

        V ~ 1/r + Q_zz P_2(cos theta) / (2 r^3)

    (Griffiths, Electrodynamics, Sec. 3.4.) There is no dipole term: |psi|^2
    is unchanged by r -> -r. Valid for r much larger than the orbital.
    """
    r = np.asarray(r, dtype=float)
    if np.any(r <= 0):
        raise ValueError("r must be positive")
    p2 = special.eval_legendre(2, np.cos(np.asarray(theta, dtype=float)))
    return 1 / r + quadrupole_moment_zz(n, l, m, a) * p2 / (2 * r ** 3)


if __name__ == "__main__":
    print("=== quadrupole_wavefunction: l = 2 hydrogen states ===\n")
    print("Q_zz = <3z^2 - r^2> for the 3d states, in units of a^2 per unit charge:")
    for m in (-2, -1, 0, 1, 2):
        print(f"  m = {m:+d}:  <r^2> = {mean_r2(3, 2):.0f},  <cos^2> = {mean_cos2(2, m):.4f},  "
              f"Q_zz = {quadrupole_moment_zz(3, 2, m):+.1f}")
    print(f"  sum over m = {sum(quadrupole_moment_zz(3, 2, m) for m in range(-2, 3)):+.1e}  (closed subshell is spherical)")

    density = lambda r, th, ph: np.abs(hydrogen_psi(3, 2, 0, r, th, ph)) ** 2
    tensor, charge = quadrupole_tensor(density, r_max=90.0)
    print(f"\nnumerical tensor for 3d, m = 0 (total charge {charge:.6f}):")
    print(np.round(tensor, 3))

    r = np.array([30.0, 100.0, 300.0])
    exact = orbital_potential(3, 2, 0, r, 0.0)
    approx = far_field_potential(3, 2, 0, r, 0.0)
    print("\npotential on the z axis, exact vs monopole + quadrupole:")
    for ri, e, ap in zip(r, exact, approx):
        print(f"  r = {ri:5.0f} a:  exact {e:.8f}   approx {ap:.8f}   (quadrupole part {ap - 1/ri:+.2e})")
    X, Y, Z, sign = orbital_surface(lambda th, ph: real_d_orbital("z2", th, ph))
    print(f"\nd_z2 surface mesh: {X.size} vertices, {(X.shape[0]-1)*(X.shape[1]-1)} quad faces, "
          f"max radius {np.sqrt(X**2 + Y**2 + Z**2).max():.4f}")
    print("\n=== DONE ===")
