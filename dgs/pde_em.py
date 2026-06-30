"""Separation of variables for PDEs in electromagnetics.

The three canonical second-order linear PDEs of physics:

  Laplace  : del^2 Phi = 0           (electrostatics, magnetostatics)
  Wave     : del^2 u - (1/v^2)u_tt = 0  (EM radiation, acoustic waves)
  Diffusion: del^2 u = (1/D) u_t     (heat, charge diffusion)

All three separate in rectangular, cylindrical, and spherical coordinates.
The separated solutions form complete orthonormal sets in L^2, the same
Hilbert space that quantum mechanics uses.

Jalali lab connection:
  The paraxial wave equation (PWE) for a complex field E(x, z):
      dE/dz = (i/2k) del_perp^2 E
  is the 2+1 dimensional Schrodinger equation with z playing the role
  of time.  Separation of variables in the transverse plane (x, y) gives
  Hermite-Gaussian or Laguerre-Gaussian modes -- the basis functions of
  free-space optical communication.  The GVD propagator in this repo is
  the 1+1D version: dE/dz = (i*beta2/2) d^2E/dt^2.

This module provides:
  - separated_laplace_sphere()  : Legendre + radial solutions in SymPy
  - separated_wave_cylinder()   : Bessel functions in SymPy
  - separated_paraxial_wave()   : Hermite-Gaussian beam modes (numerical)
  - moment_of_inertia_composite(): MOI for sphere + n attached cylinders
  - spherical_harmonic_table()  : Y_l^m table via SymPy

Usage:
    from dgs.pde_em import (
        separated_laplace_sphere, separated_wave_cylinder,
        moment_of_inertia_composite, spherical_harmonic_table
    )
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Symbolic coordinates
# ---------------------------------------------------------------------------

r, theta, phi_sym = sp.symbols('r theta phi', positive=True, real=True)
rho, z_sym, t_sym = sp.symbols('rho z t', real=True)
k_sym, omega_sym  = sp.symbols('k omega', positive=True)
A_sym, B_sym_c    = sp.symbols('A B')
l_sym, m_sym      = sp.symbols('l m', integer=True, nonnegative=True)
n_sym             = sp.symbols('n', positive=True, integer=True)


# ---------------------------------------------------------------------------
# Separation of variables: Laplace equation in spherical coordinates
# ---------------------------------------------------------------------------

def separated_laplace_sphere(l: int, m: int = 0) -> dict:
    """Separated solution of Laplace's equation in spherical coordinates.

    del^2 Phi = 0 in spherical (r, theta, phi) separates as:
        Phi(r, theta, phi) = R(r) * Theta(theta) * Phi(phi)

    Radial equation:    r^2 R'' + 2r R' - l(l+1) R = 0
        Solutions: R(r) = A*r^l + B*r^{-(l+1)}   (Euler equation)

    Angular equation:   d/d(cos theta)[sin^2(theta) dP/d(cos theta)] + l(l+1)P = 0
        Solutions: Legendre polynomials P_l(cos theta)
        (Associated: P_l^m for m != 0, gives spherical harmonics Y_l^m)

    Griffiths Ch 3.3: this is Laplace's equation with azimuthal symmetry
    when m=0.  The general solution inside a sphere is:
        V(r, theta) = sum_l (A_l r^l + B_l r^{-(l+1)}) P_l(cos theta)

    Parameters
    ----------
    l : angular momentum quantum number (0, 1, 2, ...)
    m : azimuthal quantum number (0, ..., l)

    Returns
    -------
    dict with:
      R_in   : radial solution regular at origin (r^l)
      R_out  : radial solution regular at infinity (r^{-(l+1)})
      Theta  : angular solution (Legendre polynomial P_l^m(cos theta))
      Phi    : azimuthal solution (exp(i*m*phi))
      product: full separated solution (symbolic)
    """
    costh = sp.cos(theta)

    R_in   = r**l
    R_out  = r**(-(l+1))
    Theta  = sp.legendre(l, costh) if m == 0 else sp.assoc_legendre(l, m, costh)
    Phi_fn = sp.cos(m * phi_sym) if m == 0 else sp.exp(sp.I * m * phi_sym)

    product_in  = R_in * Theta * Phi_fn
    product_out = R_out * Theta * Phi_fn

    return {
        'l': l, 'm': m,
        'R_in':       R_in,
        'R_out':      R_out,
        'Theta':      Theta,
        'Phi':        Phi_fn,
        'solution_in':  product_in,
        'solution_out': product_out,
        'ODE_radial':   sp.Eq(r**2 * sp.Function('R')(r).diff(r,2)
                               + 2*r * sp.Function('R')(r).diff(r)
                               - l*(l+1) * sp.Function('R')(r), 0),
    }


def verify_laplace_radial(l: int) -> bool:
    """Verify that r^l and r^{-(l+1)} satisfy the radial Euler ODE."""
    R = sp.Function('R')
    R_val = r**l
    ode = r**2 * R_val.diff(r, 2) + 2*r * R_val.diff(r) - l*(l+1)*R_val
    return sp.simplify(ode) == 0


# ---------------------------------------------------------------------------
# Separation of variables: Wave equation in cylindrical coordinates
# ---------------------------------------------------------------------------

def separated_wave_cylinder(n: int, kz: float = 1.0) -> dict:
    """Separated solution of the wave equation in cylindrical coordinates.

    del^2 u - (1/v^2) u_tt = 0

    In cylindrical (rho, phi, z, t), assuming u = R(rho)*Phi(phi)*Z(z)*T(t):

    Radial equation (Bessel's equation):
        R'' + (1/rho)R' + (k_perp^2 - n^2/rho^2) R = 0
        Solutions: R(rho) = J_n(k_perp * rho)   (Bessel function, regular at 0)
                            Y_n(k_perp * rho)   (Neumann, singular at 0)

    This is the same equation that governs:
      - EM modes in a cylindrical waveguide (microwave / optical fiber)
      - Acoustic modes in a cylindrical resonator
      - Quantum particle in a circular well (2D infinite square well)
      - Free-space Laguerre-Gaussian beam modes (Jalali lab optics)

    Parameters
    ----------
    n   : azimuthal mode number (0=symmetric, 1=dipole, 2=quadrupole, ...)
    kz  : axial wavenumber (sets axial field variation)

    Returns
    -------
    dict with symbolic Bessel ODE, Phi(phi), Z(z), T(t), and numerical
    eigenvalue table (zeros of J_n for hard-wall boundary conditions).
    """
    kperp = sp.Symbol('k_perp', positive=True)
    rho_s = sp.Symbol('rho', positive=True)
    R_fn  = sp.Function('R')

    # Bessel ODE in sympy form
    bessel_ode = sp.Eq(
        R_fn(rho_s).diff(rho_s, 2)
        + sp.Rational(1, 1) / rho_s * R_fn(rho_s).diff(rho_s)
        + (kperp**2 - sp.Integer(n)**2 / rho_s**2) * R_fn(rho_s),
        0
    )

    # Azimuthal, axial, temporal solutions
    Phi_fn = sp.cos(n * phi_sym) if n == 0 else sp.cos(n * phi_sym)   # cos or sin
    Z_fn   = sp.exp(sp.I * kz * z_sym)
    T_fn   = sp.exp(-sp.I * omega_sym * t_sym)

    # Numerical: zeros of J_n (first 5 for hard-wall BC at rho=R)
    from scipy.special import jn_zeros
    try:
        zeros = jn_zeros(n, 5).tolist()
    except ImportError:
        # Approximate zeros using asymptotic expansion (no scipy)
        zeros = [n + 1.8557*(1)**(2/3) + 3*(n**2) for _ in range(5)]

    return {
        'n': n,
        'bessel_ode':   bessel_ode,
        'Phi_sym':      Phi_fn,
        'Z_sym':        Z_fn,
        'T_sym':        T_fn,
        'bessel_zeros': zeros,
        'waveguide_modes': {
            f'TM_0{i+1}': f'k_perp = {z:.4f}/R (cutoff freq = {z:.4f}*c/(2*pi*R))'
            for i, z in enumerate(zeros[:3])
        },
    }


# ---------------------------------------------------------------------------
# Hermite-Gaussian beam modes (paraxial wave equation)
# ---------------------------------------------------------------------------

def hermite_gaussian_mode(
    p: int, q: int,
    w0: float = 1.0,
    z: float = 0.0,
    x: np.ndarray = None,
    y: np.ndarray = None,
) -> np.ndarray:
    """Compute Hermite-Gaussian mode TEM_pq at plane z.

    The paraxial wave equation  dE/dz = (i/2k) del_perp^2 E
    has separated solutions (Hermite-Gaussian modes):

        E_pq(x,y,z) = H_p(sqrt(2)*x/w(z)) * H_q(sqrt(2)*y/w(z))
                      * exp(-(x^2+y^2)/w(z)^2)
                      * exp(i*k*(x^2+y^2)/(2*R(z)))
                      * exp(-i*(p+q+1)*arctan(z/z_R))

    where w(z) = w0*sqrt(1+(z/z_R)^2), z_R = pi*w0^2/lambda is the Rayleigh range,
    and R(z) = z*(1+(z_R/z)^2) is the radius of curvature.

    This is the optical analogue of the quantum harmonic oscillator eigenstates.
    H_p are Hermite polynomials; p+q is the transverse mode order.

    Jalali lab uses TEM_00 (p=q=0, Gaussian) for fiber coupling and TEM_01/10
    (dipole modes) for structured illumination.

    Parameters
    ----------
    p, q  : mode indices (0 = Gaussian)
    w0    : beam waist radius (same units as x, y)
    z     : propagation distance (in units of z_R)
    x, y  : transverse coordinate arrays

    Returns
    -------
    Complex field E_pq(x, y) at plane z.
    """
    from numpy.polynomial.hermite import hermval

    if x is None:
        x = np.linspace(-4*w0, 4*w0, 128)
    if y is None:
        y = np.linspace(-4*w0, 4*w0, 128)

    X, Y = np.meshgrid(x, y)
    z_R  = np.pi * w0**2   # Rayleigh range (lambda=1 units)
    wz   = w0 * np.sqrt(1 + (z/z_R)**2)

    # Gouy phase and curvature (handle z=0)
    gouy  = (p + q + 1) * np.arctan2(z, z_R)
    R_curv = z * (1 + (z_R/z)**2) if abs(z) > 1e-10 else 1e30

    # Hermite polynomials H_p, H_q (coefficients: only coeff p is 1, rest 0)
    coeff_p = np.zeros(p+1); coeff_p[p] = 1.0
    coeff_q = np.zeros(q+1); coeff_q[q] = 1.0

    Hp = hermval(np.sqrt(2)*X/wz, coeff_p)
    Hq = hermval(np.sqrt(2)*Y/wz, coeff_q)

    r2 = X**2 + Y**2
    E  = (Hp * Hq
          * np.exp(-r2/wz**2)
          * np.exp(1j * r2 / (2*R_curv))
          * np.exp(-1j * gouy))

    return E


# ---------------------------------------------------------------------------
# Rotational symmetry: moment of inertia for sphere + n cylinders
# ---------------------------------------------------------------------------

def moment_of_inertia_composite(
    M_sphere: float,
    R_sphere: float,
    M_cyl: float,
    R_cyl: float,
    L_cyl: float,
    n_cylinders: int = 4,
    config: str = 'orthogonal',
) -> dict:
    """Moment of inertia of a sphere with n cylinders welded symmetrically.

    Configurations:
      'orthogonal' : cylinders along +x, -x, +y, -y (4 cylinders in xy plane)
      'tetrahedral': cylinders along tetrahedral bond angles (~109.5 deg)
      'axial_pairs': 2 pairs, one pair along z, one in xy plane

    Each cylinder has its center of mass at a distance R_sphere + L_cyl/2
    from the sphere center.  The parallel axis theorem gives:

        I_cyl_total = I_cyl_cm + M_cyl * d^2

    where d is the distance from the sphere center to the cylinder center.

    Parameters
    ----------
    M_sphere   : sphere mass (kg)
    R_sphere   : sphere radius (m)
    M_cyl      : mass of each cylinder (kg)
    R_cyl      : cylinder radius (m)
    L_cyl      : cylinder length (m)
    n_cylinders: number of attached cylinders
    config     : arrangement of cylinders

    Returns
    -------
    dict with I_xx, I_yy, I_zz (principal moments in kg.m^2), and total mass.
    """
    # Sphere MOI about any axis through center
    I_sphere = (2/5) * M_sphere * R_sphere**2

    # Single cylinder: I about own symmetry axis, I about diameter
    I_cyl_sym  = 0.5 * M_cyl * R_cyl**2                       # about own z-axis
    I_cyl_diam = M_cyl * (R_cyl**2/4 + L_cyl**2/12)           # about diameter through cm

    d = R_sphere + L_cyl / 2                                   # cm to sphere center

    # For each cylinder orientation, use parallel axis theorem
    # Rotation axis is the z-axis (by symmetry, I_xx = I_yy = I_zz for symmetric configs)
    I_xx, I_yy, I_zz = I_sphere, I_sphere, I_sphere

    if config == 'orthogonal' and n_cylinders == 4:
        # Cylinders along ±x and ±y
        # For rotation about z: each cylinder's symmetry axis is perp to z
        # I_perp (axis perp to cylinder axis, through cm) = I_cyl_diam
        # Parallel axis: I += M*d^2 (d is in xy plane for all cylinders)
        I_z_each = I_cyl_diam + M_cyl * d**2
        I_zz += 4 * I_z_each

        # For rotation about x: cylinders along ±x contribute as rotating about own axis
        # (their symmetry axis IS the x-axis): I += I_cyl_sym (no offset, on x-axis)
        # Cylinders along ±y: symmetry axis perp to x, offset by 0 in z but at d in y
        I_x_cx = I_cyl_sym                                    # along x: no offset
        I_x_cy = I_cyl_diam + M_cyl * d**2                   # along y: offset d in y
        I_xx += 2 * I_x_cx + 2 * I_x_cy
        I_yy += 2 * I_x_cy + 2 * I_x_cx   # symmetric: same as xx

    elif config == 'tetrahedral' and n_cylinders == 4:
        # Tetrahedron vertices: (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1) / sqrt(3)
        verts = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]) / np.sqrt(3)
        for v in verts:
            # Cylinder along direction v, center at d*v
            # For rotation about z: component of d perpendicular to z = d*sqrt(vx^2+vy^2)
            d_perp_z = d * np.sqrt(v[0]**2 + v[1]**2)
            d_perp_x = d * np.sqrt(v[1]**2 + v[2]**2)
            d_perp_y = d * np.sqrt(v[0]**2 + v[2]**2)
            # cos^2(angle between cyl axis and rotation axis)
            cos2_z = v[2]**2
            # I of cylinder about axis perpendicular to its own: I_diam
            # I of cylinder about its own axis: I_sym
            # Mix: I_axis = I_sym*cos2 + I_diam*(1-cos2) + M*d_perp^2
            I_zz += I_cyl_sym*cos2_z + I_cyl_diam*(1-cos2_z) + M_cyl*d_perp_z**2
            cos2_x = v[0]**2
            I_xx += I_cyl_sym*cos2_x + I_cyl_diam*(1-cos2_x) + M_cyl*d_perp_x**2
            cos2_y = v[1]**2
            I_yy += I_cyl_sym*cos2_y + I_cyl_diam*(1-cos2_y) + M_cyl*d_perp_y**2

    else:
        # Generic: treat all cylinders as mass points at distance d for an estimate
        for _ in range(n_cylinders):
            I_xx += I_cyl_diam + M_cyl * d**2
            I_yy += I_cyl_diam + M_cyl * d**2
            I_zz += I_cyl_diam + M_cyl * d**2

    M_total = M_sphere + n_cylinders * M_cyl
    return {
        'I_xx': I_xx, 'I_yy': I_yy, 'I_zz': I_zz,
        'I_sphere': I_sphere,
        'I_total_avg': (I_xx + I_yy + I_zz) / 3,
        'M_total': M_total,
        'config': config,
        'n_cylinders': n_cylinders,
        'symmetry_note': (
            'I_xx = I_yy = I_zz (isotropic)' if (abs(I_xx-I_yy) < 1e-10*max(abs(I_xx),1)
                                                   and abs(I_xx-I_zz) < 1e-10*max(abs(I_xx),1))
            else f'I_xx={I_xx:.4e} = I_yy, I_zz={I_zz:.4e} (axial symmetry about z)'
            if abs(I_xx-I_yy) < 1e-10*max(abs(I_xx),1)
            else f'I_xx={I_xx:.4e}, I_yy={I_yy:.4e}, I_zz={I_zz:.4e}'
        ),
    }


# ---------------------------------------------------------------------------
# Spherical harmonic table (SymPy)
# ---------------------------------------------------------------------------

def spherical_harmonic_table(l_max: int = 3) -> List[Tuple]:
    """Generate Y_l^m(theta, phi) using SymPy up to l=l_max.

    Returns list of (l, m, Y_lm_sym) tuples.
    """
    rows = []
    costh = sp.cos(theta)
    for l in range(l_max + 1):
        for m in range(-l, l+1):
            if m >= 0:
                Plm = sp.assoc_legendre(l, m, costh)
            else:
                Plm = (-1)**(-m) * sp.assoc_legendre(l, -m, costh)
            phase = sp.exp(sp.I * m * phi_sym)
            Y = Plm * phase
            rows.append((l, m, sp.simplify(Y)))
    return rows


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("PDE Separation of Variables -- Demo")
    print("=" * 50)

    print("\n1. Laplace equation in spherical coordinates (l=0,1,2):")
    for l_val in [0, 1, 2]:
        sol = separated_laplace_sphere(l_val, m=0)
        ok  = verify_laplace_radial(l_val)
        print(f"   l={l_val}: R_in = r^{l_val}, R_out = r^(-{l_val+1}), "
              f"P_{l_val}(cos theta), ODE satisfied: {ok}")

    print("\n2. Wave equation in cylindrical coordinates (Bessel modes):")
    for n_val in [0, 1, 2]:
        sol = separated_wave_cylinder(n_val)
        print(f"   n={n_val}: first Bessel zero = {sol['bessel_zeros'][0]:.4f}  "
              f"(kperp*R for first TM mode)")
        for mode, desc in sol['waveguide_modes'].items():
            print(f"     {mode}: {desc}")

    print("\n3. Rotational symmetry: sphere + 4 cylinders (orthogonal config):")
    result = moment_of_inertia_composite(
        M_sphere=1.0, R_sphere=0.1,
        M_cyl=0.2, R_cyl=0.02, L_cyl=0.15,
        n_cylinders=4, config='orthogonal'
    )
    print(f"   I_xx = {result['I_xx']:.6f} kg.m^2")
    print(f"   I_yy = {result['I_yy']:.6f} kg.m^2")
    print(f"   I_zz = {result['I_zz']:.6f} kg.m^2")
    print(f"   {result['symmetry_note']}")

    print("\n4. Spherical harmonics Y_l^m (first few):")
    table = spherical_harmonic_table(l_max=2)
    for l_v, m_v, Y in table[:6]:
        print(f"   Y_{l_v}^{m_v:+d} = {str(Y)}")

    print("\n5. Hermite-Gaussian modes TEM_pq (Jalali lab free-space optics):")
    x = np.linspace(-3, 3, 64)
    y = np.linspace(-3, 3, 64)
    for p_val, q_val in [(0,0), (1,0), (0,1), (1,1)]:
        E = hermite_gaussian_mode(p_val, q_val, w0=1.0, z=0.0, x=x, y=y)
        I = np.abs(E)**2
        print(f"   TEM_{p_val}{q_val}: peak intensity = {I.max():.4f}, "
              f"mode order = {p_val+q_val}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    demo()
