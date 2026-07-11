"""The hydrogen atom's quantum numbers: space quantization, shells, and the radial equation.

Solving the Schrodinger equation for a central Coulomb force gives three quantum numbers, each a
different "sharp" (definite) quantity a state can have at once:

    n = 1,2,3,...        principal      -> ENERGY is sharp:  E_n = -13.6 Z^2 / n^2  eV
    l = 0,1,...,n-1       orbital        -> |L| is sharp:      |L| = sqrt(l(l+1)) hbar
    m = -l,...,+l         magnetic       -> L_z is sharp:      L_z = m hbar

E depending only on n (not l) is the special "accidental" degeneracy of the pure 1/r potential; the
number of (l,m) states at energy E_n is the DEGENERACY n^2 (or 2n^2 with spin -- exactly the
2,8,18,32 shell capacities of the periodic table). SPACE QUANTIZATION is the fact that L_z can only
take the 2l+1 values m*hbar, so the angular momentum vector points only along a discrete set of cones
tilted at cos(theta)=m/sqrt(l(l+1)) -- and because |L|=sqrt(l(l+1))hbar always exceeds its largest
projection l*hbar, L can never lie exactly along z (the same uncertainty seen in dgs.angular_momentum).

The RADIAL wave equation carries an effective potential V_eff(r) = -Z e^2/(4 pi eps0 r) +
l(l+1) hbar^2/(2 m_e r^2): the centrifugal term l(l+1)/r^2 is a barrier that keeps l>0 electrons away
from the nucleus (only l=0 s-states reach r=0). Everything scales cleanly with nuclear charge Z
(hydrogen-like ions He+, Li2+, ...), and by CPT symmetry ANTIHYDROGEN has an identical spectrum -- a
prediction tested at CERN's ALPHA experiment.

Complements dgs.feynman_atomic_molecular (Bohr energies, spectral lines) with the full quantum-number
/ degeneracy / angular-momentum structure. Constants in SI + eV; NumPy/math; py-3.13.
"""

import math
import numpy as np

RYDBERG_EV = 13.605693122     # ionization energy of hydrogen, eV
A0 = 5.29177210903e-11        # Bohr radius, m
_E = 1.602176634e-19          # C
_EPS0 = 8.8541878128e-12      # F/m
_HBAR = 1.054571817e-34       # J s
_ME = 9.1093837015e-31        # kg
_KE = 1.0 / (4 * math.pi * _EPS0)


def energy_level(n, Z=1):
    """Energy of level n for a hydrogen-like ion of nuclear charge Z: E_n = -13.6 Z^2/n^2 eV.
    Depends only on n -- E is sharp. (Antihydrogen: identical spectrum, by CPT.)"""
    if n < 1 or int(n) != n:
        raise ValueError("n must be a positive integer")
    if Z < 1:
        raise ValueError("Z must be >= 1")
    return -RYDBERG_EV * Z ** 2 / n ** 2


def orbital_radius(n, Z=1):
    """Bohr orbit radius r_n = n^2 a0 / Z (the mean radius grows as n^2, shrinks with Z)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return n ** 2 * A0 / Z


def allowed_l(n):
    """The orbital quantum numbers for shell n: l = 0, 1, ..., n-1 (n values)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return list(range(n))


def allowed_m(l):
    """The magnetic quantum numbers for orbital l: m = -l, ..., +l (2l+1 values)."""
    if l < 0:
        raise ValueError("l must be >= 0")
    return list(range(-l, l + 1))


def valid_state(n, l, m):
    """True if (n, l, m) is an allowed hydrogen state: 1<=n, 0<=l<n, |m|<=l."""
    return n >= 1 and 0 <= l < n and -l <= m <= l


def shell_states(n):
    """Every (l, m) pair in shell n. Their count is the degeneracy n^2."""
    return [(l, m) for l in allowed_l(n) for m in allowed_m(l)]


def degeneracy(n, include_spin=False):
    """Number of states at energy E_n: n^2 (orbital), or 2n^2 with the electron's two spins --
    the 2, 8, 18, 32 capacities of the periodic-table shells."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return (2 if include_spin else 1) * n ** 2


def angular_momentum_magnitude(l):
    """|L| = sqrt(l(l+1)) hbar (in units of hbar). Sharp, and always > l for l>0."""
    if l < 0:
        raise ValueError("l must be >= 0")
    return math.sqrt(l * (l + 1))


def angular_momentum_z(m):
    """L_z = m hbar (in units of hbar) -- the quantized projection (space quantization)."""
    return float(m)


def space_quantization_angles(l):
    """The 2l+1 allowed tilt angles of L from the z-axis: cos(theta)=m/sqrt(l(l+1)). None reach
    theta=0, since |L|>l*hbar -- L never lies exactly along z."""
    if l == 0:
        return []                                     # |L|=0, direction undefined
    Lmag = angular_momentum_magnitude(l)
    return [math.acos(m / Lmag) for m in allowed_m(l)]


def effective_potential(r, l, Z=1):
    """Radial effective potential (Joules): Coulomb attraction plus the centrifugal barrier
    V_eff(r) = -Z e^2/(4 pi eps0 r) + l(l+1) hbar^2/(2 m_e r^2)."""
    r = np.asarray(r, float)
    coulomb = -_KE * Z * _E ** 2 / r
    centrifugal = l * (l + 1) * _HBAR ** 2 / (2 * _ME * r ** 2)
    return coulomb + centrifugal


def centrifugal_minimum_radius(l, Z=1):
    """Radius of the V_eff minimum for l>0: r_min = l(l+1) a0 / Z. None for l=0 (no barrier)."""
    if l == 0:
        return None
    return l * (l + 1) * A0 / Z


def transition_energy(n_i, n_f, Z=1):
    """Photon energy (eV) emitted in n_i -> n_f (n_i > n_f): 13.6 Z^2 (1/n_f^2 - 1/n_i^2)."""
    if n_i <= n_f:
        raise ValueError("emission requires n_i > n_f")
    return RYDBERG_EV * Z ** 2 * (1 / n_f ** 2 - 1 / n_i ** 2)


def transition_wavelength_nm(n_i, n_f, Z=1):
    """Wavelength (nm) of the n_i -> n_f photon (E = hc/lambda)."""
    E_J = transition_energy(n_i, n_f, Z) * _E
    return (6.62607015e-34 * 299792458.0 / E_J) * 1e9


if __name__ == "__main__":
    print("=== energy levels and shells (E sharp; degeneracy n^2) ===")
    print(" n   E_n(eV)   r_n(a0)   l-values      degeneracy  (2n^2 w/spin)")
    for n in (1, 2, 3, 4):
        print(f" {n}   {energy_level(n):7.3f}   {orbital_radius(n)/A0:5.0f}     "
              f"{allowed_l(n)!s:12s}  {degeneracy(n):3d}         {degeneracy(n, True)}")
    print(f"  He+ (Z=2) ground state: {energy_level(1, 2):.1f} eV  (= 4 x hydrogen)")
    print(f"  antihydrogen n=1:       {energy_level(1):.3f} eV  (identical to hydrogen, by CPT)")

    print("\n=== space quantization of L (l=2, the d-orbital) ===")
    l = 2
    print(f"  |L| = sqrt(l(l+1)) hbar = {angular_momentum_magnitude(l):.4f} hbar")
    print(f"  L_z = m hbar for m in {allowed_m(l)}  (2l+1 = {2*l+1} orientations)")
    print(f"  tilt angles (deg): {[round(math.degrees(a),1) for a in space_quantization_angles(l)]}")
    print(f"  smallest tilt {math.degrees(min(space_quantization_angles(l))):.1f} deg > 0: "
          f"L never lies on the axis")

    print("\n=== radial: the centrifugal barrier keeps l>0 electrons out ===")
    for l in (0, 1, 2):
        rmin = centrifugal_minimum_radius(l)
        print(f"  l={l}: V_eff minimum at r = "
              f"{'none (s-state reaches r=0)' if rmin is None else f'{rmin/A0:.0f} a0'}")

    print("\n=== spectral lines (cross-check Rydberg) ===")
    print(f"  Lyman-alpha (2->1): {transition_energy(2,1):.2f} eV, {transition_wavelength_nm(2,1):.1f} nm")
    print(f"  Balmer-alpha (3->2): {transition_energy(3,2):.2f} eV, {transition_wavelength_nm(3,2):.1f} nm")
