"""Stability of matter: why the electron does not spiral into the nucleus.

Classically, an atom should collapse -- the Coulomb energy -k e^2/r goes to MINUS
INFINITY as the electron falls toward r=0, and a circling charge radiates away its
energy. Matter should not be stable. It is, and the reason is the UNCERTAINTY
PRINCIPLE: confine the electron to a region of size r and its momentum spreads by at
least Delta_p ~ hbar/r, so it MUST carry kinetic energy
        KE ~ (Delta_p)^2 / 2m = hbar^2 / (2 m r^2).
Squeeze the electron smaller and this kinetic cost blows up faster (1/r^2) than the
Coulomb energy drops (1/r). The total energy
        E(r) = hbar^2/(2 m r^2) - Z k e^2 / r
therefore has a genuine MINIMUM at a finite radius -- the atom cannot collapse.

Minimizing E(r) (dE/dr = 0) gives, with no fitted constants, exactly the Bohr atom:
        r_min = hbar^2 / (Z m k e^2)          = a_0 / Z   (Bohr radius 0.529 Angstrom),
        E_min = -Z^2 m (k e^2)^2 / (2 hbar^2) = -13.6 Z^2 eV   (Rydberg).
The "sloppy" uncertainty estimate lands on the real ground state because the Coulomb
1/r makes the balance exact. At the minimum the virial theorem holds: KE = -E_total,
PE = 2 E_total.

This is stability of matter at the level of ONE atom -- the same kinetic-energy floor
(Pauli + uncertainty) is what holds up bulk matter and white dwarfs. Ties to the QM
thread (dgs.quantum_oscillator zero-point energy, dgs.angular_momentum). Verified
against the CODATA Bohr radius and Rydberg energy, and by numerically minimizing E(r).
NumPy + SciPy; py-3.13.
"""

import numpy as np
from scipy.optimize import minimize_scalar

HBAR = 1.054571817e-34        # J s
M_E = 9.1093837015e-31        # kg
E_CHARGE = 1.602176634e-19    # C
EPS0 = 8.8541878128e-12       # F/m
K_COULOMB = 1.0 / (4 * np.pi * EPS0)
EV_J = E_CHARGE               # 1 eV in joules


def kinetic_energy(r, m=M_E):
    """Uncertainty-principle kinetic energy of an electron confined to size r:
    KE = hbar^2 / (2 m r^2). Diverges as r -> 0 -- the collapse-preventing term."""
    r = np.asarray(r, float)
    if np.any(r <= 0):
        raise ValueError("r must be positive")
    return HBAR ** 2 / (2 * m * r ** 2)


def coulomb_energy(r, Z=1):
    """Coulomb potential energy -Z k e^2 / r (attractive, negative)."""
    r = np.asarray(r, float)
    if np.any(r <= 0):
        raise ValueError("r must be positive")
    if Z <= 0:
        raise ValueError("Z must be positive")
    return -Z * K_COULOMB * E_CHARGE ** 2 / r


def total_energy(r, Z=1, m=M_E):
    """E(r) = KE(r) + Coulomb(r): the energy whose minimum is the atom's ground
    state. Has a finite minimum (stable) despite the -inf Coulomb well."""
    return kinetic_energy(r, m) + coulomb_energy(r, Z)


def bohr_radius(Z=1, m=M_E):
    """The energy-minimizing radius r_min = hbar^2/(Z m k e^2) = a_0/Z. For Z=1,
    electron mass, this is the Bohr radius ~5.29e-11 m."""
    if Z <= 0 or m <= 0:
        raise ValueError("Z and m must be positive")
    return HBAR ** 2 / (Z * m * K_COULOMB * E_CHARGE ** 2)


def ground_state_energy(Z=1, m=M_E):
    """The minimum energy E_min = -Z^2 m (k e^2)^2 / (2 hbar^2) in joules. For Z=1
    this is -13.6 eV (the Rydberg); scales as Z^2."""
    if Z <= 0 or m <= 0:
        raise ValueError("Z and m must be positive")
    return -Z ** 2 * m * (K_COULOMB * E_CHARGE ** 2) ** 2 / (2 * HBAR ** 2)


def ground_state_energy_eV(Z=1, m=M_E):
    """The ground-state energy in electron-volts (-13.6 Z^2 eV)."""
    return ground_state_energy(Z, m) / EV_J


def minimize_numerically(Z=1, m=M_E):
    """Find the energy minimum of E(r) numerically (no analytic formula used) and
    return (r_min, E_min_J). Confirms bohr_radius / ground_state_energy from the
    raw curve -- the stability is in the shape of E(r), not a formula."""
    a0 = bohr_radius(Z, m)
    # minimize over the DIMENSIONLESS u = r/a0 so the optimizer's tolerance is
    # well-scaled (a raw minimum near 1e-11 m is below the default x-tolerance)
    res = minimize_scalar(lambda u: total_energy(u * a0, Z, m),
                          bounds=(1e-3, 1e3), method="bounded")
    return res.x * a0, res.fun


def virial_ratios(Z=1, m=M_E):
    """At the ground state, check the virial theorem: KE = -E_total and
    PE = 2 E_total. Returns (KE/|E|, PE/|E|) which must be (+1, -2)."""
    r = bohr_radius(Z, m)
    ke = float(kinetic_energy(r, m))
    pe = float(coulomb_energy(r, Z))
    E = ke + pe
    return ke / abs(E), pe / abs(E)


if __name__ == "__main__":
    print("hydrogen (Z=1): the uncertainty principle sets the size and energy")
    print(f"  r_min = {bohr_radius():.4e} m  (Bohr radius 5.2918e-11)")
    print(f"  E_min = {ground_state_energy_eV():.4f} eV  (Rydberg -13.6057)")

    r_num, E_num = minimize_numerically()
    print(f"  numerical minimum: r = {r_num:.4e} m, E = {E_num/EV_J:.4f} eV "
          f"(matches analytic)")

    ke_ratio, pe_ratio = virial_ratios()
    print(f"  virial: KE/|E| = {ke_ratio:+.3f} (=+1), PE/|E| = {pe_ratio:+.3f} (=-2)")

    print("\nwhy it is STABLE -- compare the two terms near r=0:")
    for r in (5e-11, 1e-11, 1e-12):
        print(f"  r={r:.0e} m: KE={kinetic_energy(r)/EV_J:+9.1f} eV, "
              f"Coulomb={coulomb_energy(r)/EV_J:+9.1f} eV, "
              f"total={total_energy(r)/EV_J:+9.1f} eV")
    print("  -> as r shrinks, KE (1/r^2) overwhelms Coulomb (1/r): the well has a floor")

    print("\nheavier nuclei (Z): tighter and deeper as Z^2")
    for Z in (1, 2, 3):
        print(f"  Z={Z}: r_min={bohr_radius(Z):.3e} m, E={ground_state_energy_eV(Z):.1f} eV")
