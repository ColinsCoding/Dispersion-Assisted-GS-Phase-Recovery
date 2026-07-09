"""Degrees of freedom: count the ways a molecule stores energy, get its heat capacity.

EQUIPARTITION says every independent QUADRATIC way a molecule can hold energy -- each
velocity component (1/2 m v^2), each rotation (1/2 I omega^2), each vibration (kinetic AND
potential) -- carries, on average, exactly 1/2 kT of thermal energy. So the total average
energy is just (f/2) kT, where f is the number of active degrees of freedom, and the molar
heat capacities fall straight out:
        C_v = (f/2) R,   C_p = C_v + R,   gamma = C_p/C_v = (f + 2)/f.
Counting f is the whole game:
        monatomic  (He, Ar):   3 translation                      -> f=3, gamma = 5/3
        diatomic   (N2, O2):   3 translation + 2 rotation          -> f=5, gamma = 7/5 = 1.4
        (hot diatomic, + vibration)                                -> f=7, gamma = 9/7
        nonlinear polyatomic:  3 translation + 3 rotation          -> f=6, gamma = 4/3
A linear molecule of N atoms has 3N-5 vibrational modes (each 2 quadratic DOF), a nonlinear
one 3N-6.

The measured gammas confirm it: helium 1.66, air (mostly N2/O2) 1.40, steam ~1.33 -- and
air's 1.4 is the number in every sound-speed and adiabatic-compression formula. The DOF also
FREEZE OUT at low temperature (a quantum effect): rotations and especially vibrations only
"turn on" once kT exceeds their energy spacing, which is why diatomic C_v steps up from 3/2 R
to 5/2 R to 7/2 R as you heat it. Ties to dgs.maxwell_boltzmann (the 3 translational DOF give
<KE>=3/2 kT) and the thermal thread. NumPy-free; py-3.13.
"""

R_GAS = 8.314462618        # J/(mol K)
K_BOLTZ = 1.380649e-23     # J/K


def equipartition_energy_per_molecule(dof, T):
    """Average thermal energy of a molecule with `dof` quadratic degrees of freedom:
    (f/2) kT -- 1/2 kT per degree of freedom."""
    if dof < 1 or T <= 0:
        raise ValueError("dof must be >= 1 and T > 0")
    return 0.5 * dof * K_BOLTZ * T


def equipartition_energy_per_mole(dof, T):
    """Average energy per mole: (f/2) RT."""
    if dof < 1 or T <= 0:
        raise ValueError("dof must be >= 1 and T > 0")
    return 0.5 * dof * R_GAS * T


def cv_molar(dof):
    """Molar heat capacity at constant volume C_v = (f/2) R."""
    if dof < 1:
        raise ValueError("dof must be >= 1")
    return 0.5 * dof * R_GAS


def cp_molar(dof):
    """Molar heat capacity at constant pressure C_p = C_v + R (Mayer's relation)."""
    return cv_molar(dof) + R_GAS


def heat_capacity_ratio(dof):
    """gamma = C_p/C_v = (f + 2)/f -- the adiabatic index in the sound speed and
    adiabatic-process laws. Nearer 1 for more degrees of freedom."""
    if dof < 1:
        raise ValueError("dof must be >= 1")
    return (dof + 2) / dof


def molecular_dof(n_atoms, linear=None, vibrational=False):
    """Count a molecule's active degrees of freedom: 3 translational, plus 2 (linear)
    or 3 (nonlinear) rotational, plus -- if `vibrational` -- 2 per vibrational mode
    (3N-5 modes for a linear molecule, 3N-6 for a nonlinear one). Monatomic = 3."""
    if n_atoms < 1:
        raise ValueError("n_atoms must be >= 1")
    if n_atoms == 1:
        return 3                                    # translation only
    if linear is None:
        linear = (n_atoms == 2)                     # a diatomic is linear
    dof = 3 + (2 if linear else 3)                  # translation + rotation
    if vibrational:
        modes = 3 * n_atoms - (5 if linear else 6)
        dof += 2 * modes                            # each mode: kinetic + potential
    return dof


if __name__ == "__main__":
    print("f   C_v/R   C_p/R   gamma   example")
    cases = [(molecular_dof(1), "He, Ar (monatomic)"),
             (molecular_dof(2), "N2, O2, air (diatomic)"),
             (molecular_dof(2, vibrational=True), "hot diatomic (+ vibration)"),
             (molecular_dof(3, linear=False), "H2O (nonlinear, no vib)"),
             (molecular_dof(3, linear=True), "CO2 (linear, no vib)")]
    for f, name in cases:
        print(f"{f:<3d} {cv_molar(f)/R_GAS:5.2f}   {cp_molar(f)/R_GAS:5.2f}   "
              f"{heat_capacity_ratio(f):.3f}   {name}")

    print(f"\nair (diatomic, f=5): gamma = {heat_capacity_ratio(5):.3f}  "
          f"(the 1.4 in the speed of sound and adiabatic laws)")
    print(f"monatomic average energy at 300 K = "
          f"{equipartition_energy_per_molecule(3, 300):.3e} J = 3/2 kT "
          f"(matches Maxwell-Boltzmann <KE>)")
    print(f"Mayer's relation C_p - C_v = R for any f: "
          f"{cp_molar(5) - cv_molar(5):.3f} = {R_GAS:.3f}")
