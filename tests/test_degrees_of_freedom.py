"""Test dgs.degrees_of_freedom: equipartition (1/2 kT per DOF), C_v=(f/2)R, C_p=C_v+R,
gamma=(f+2)/f, the DOF counts for mono/di/poly-atomic molecules, the known gammas
(5/3, 7/5, 4/3), Mayer's relation, and the tie to Maxwell-Boltzmann <KE>=3/2 kT."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import degrees_of_freedom as dof

R, k = dof.R_GAS, dof.K_BOLTZ

# 1. equipartition: 1/2 kT per degree of freedom
assert math.isclose(dof.equipartition_energy_per_molecule(1, 300), 0.5 * k * 300)
assert math.isclose(dof.equipartition_energy_per_molecule(6, 300), 6 * 0.5 * k * 300)
assert math.isclose(dof.equipartition_energy_per_mole(3, 300), 1.5 * R * 300)

# 2. heat capacities and gamma
assert math.isclose(dof.cv_molar(3), 1.5 * R)
assert math.isclose(dof.cp_molar(3), 2.5 * R)
assert math.isclose(dof.heat_capacity_ratio(5), 7 / 5)
assert math.isclose(dof.heat_capacity_ratio(3), 5 / 3)

# 3. degree-of-freedom counts by molecular structure
assert dof.molecular_dof(1) == 3                        # monatomic: translation only
assert dof.molecular_dof(2) == 5                        # diatomic: + 2 rotation
assert dof.molecular_dof(2, vibrational=True) == 7      # + 1 vib mode (2 DOF)
assert dof.molecular_dof(3, linear=False) == 6          # nonlinear (H2O): + 3 rotation
assert dof.molecular_dof(3, linear=True) == 5           # linear (CO2): + 2 rotation
assert dof.molecular_dof(3, linear=False, vibrational=True) == 6 + 2 * 3   # +3 vib modes

# 4. the measured gammas fall out of the counts
assert math.isclose(dof.heat_capacity_ratio(dof.molecular_dof(1)), 5/3)      # He ~1.67
assert math.isclose(dof.heat_capacity_ratio(dof.molecular_dof(2)), 1.4)      # air ~1.40
assert math.isclose(dof.heat_capacity_ratio(dof.molecular_dof(3, linear=False)), 4/3)  # ~1.33
# more degrees of freedom -> gamma closer to 1
assert (dof.heat_capacity_ratio(3) > dof.heat_capacity_ratio(5)
        > dof.heat_capacity_ratio(7))

# 5. Mayer's relation C_p - C_v = R for ANY f
for f in (3, 5, 6, 7, 12):
    assert math.isclose(dof.cp_molar(f) - dof.cv_molar(f), R)

# 6. tie to Maxwell-Boltzmann: 3 translational DOF give <KE> = 3/2 kT
from dgs import maxwell_boltzmann as mb
assert math.isclose(dof.equipartition_energy_per_molecule(3, 300),
                    mb.average_kinetic_energy(300))

# 7. kwarg bounds
for bad in (lambda: dof.equipartition_energy_per_molecule(0, 300),
            lambda: dof.equipartition_energy_per_molecule(3, 0),
            lambda: dof.cv_molar(0),
            lambda: dof.molecular_dof(0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_degrees_of_freedom: all checks passed")
