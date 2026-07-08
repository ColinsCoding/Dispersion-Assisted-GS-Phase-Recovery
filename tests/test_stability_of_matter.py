"""Test dgs.stability_of_matter: the uncertainty-principle atom. The Bohr radius
and -13.6 eV ground state come out of minimizing hbar^2/2mr^2 - ke^2/r (no fitted
constants), the numerical minimum matches the analytic one, the virial theorem
holds (KE=-E, PE=2E), and the 1/r^2 kinetic floor is what makes the well bounded
below (stability)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import stability_of_matter as sm

eV = sm.EV_J

# 1. Bohr radius from minimizing E(r): 5.2918e-11 m, and a_0/Z scaling
assert np.isclose(sm.bohr_radius(1), 5.29177e-11, rtol=1e-4)
assert np.isclose(sm.bohr_radius(2), sm.bohr_radius(1) / 2)          # a_0 / Z
assert np.isclose(sm.bohr_radius(1, m=2 * sm.M_E), sm.bohr_radius(1) / 2)  # ~ 1/m

# 2. ground-state energy: -13.6 eV (Rydberg), scaling as Z^2
assert np.isclose(sm.ground_state_energy_eV(1), -13.6057, rtol=1e-4)
assert np.isclose(sm.ground_state_energy_eV(2), -13.6057 * 4, rtol=1e-4)   # Z^2
assert np.isclose(sm.ground_state_energy(1), sm.ground_state_energy_eV(1) * eV)

# 3. kinetic term: hbar^2/(2 m r^2), diverging as r -> 0 (the collapse blocker)
r = 5e-11
assert np.isclose(sm.kinetic_energy(r), sm.HBAR**2 / (2 * sm.M_E * r**2))
assert sm.kinetic_energy(r / 10) == sm.kinetic_energy(r) * 100        # 1/r^2
# Coulomb term: -Z k e^2 / r
assert np.isclose(sm.coulomb_energy(r, 1), -sm.K_COULOMB * sm.E_CHARGE**2 / r)
assert sm.coulomb_energy(r, 1) < 0

# 4. E(r_min) equals the analytic ground-state energy
assert np.isclose(sm.total_energy(sm.bohr_radius(1), 1), sm.ground_state_energy(1))
# the KE at the minimum equals -E_total = +13.6 eV
assert np.isclose(sm.kinetic_energy(sm.bohr_radius(1)) / eV, 13.6057, rtol=1e-4)

# 5. numerical minimization recovers the same radius and energy (no formula used)
for Z in (1, 2, 3):
    r_num, E_num = sm.minimize_numerically(Z)
    assert np.isclose(r_num, sm.bohr_radius(Z), rtol=1e-3)
    assert np.isclose(E_num, sm.ground_state_energy(Z), rtol=1e-3)

# 6. virial theorem at the ground state: KE = -E, PE = 2E
ke_ratio, pe_ratio = sm.virial_ratios(1)
assert np.isclose(ke_ratio, 1.0, atol=1e-6)
assert np.isclose(pe_ratio, -2.0, atol=1e-6)

# 7. STABILITY: a_0 is a genuine minimum -- squeezing OR expanding costs energy,
#    and as r -> 0 the total energy goes to +inf (the well has a floor)
a0 = sm.bohr_radius(1)
assert sm.total_energy(a0, 1) < sm.total_energy(a0 * 0.1, 1)          # can't collapse
assert sm.total_energy(a0, 1) < sm.total_energy(a0 * 10, 1)           # bound state
assert sm.total_energy(1e-13, 1) > 0                                  # deep squeeze -> +inf-ish
# classically (Coulomb alone) there is NO minimum -- monotonically to -inf
assert sm.coulomb_energy(1e-13, 1) < sm.coulomb_energy(1e-11, 1)      # keeps dropping

# 8. kwarg bounds
for bad in (lambda: sm.kinetic_energy(0),
            lambda: sm.coulomb_energy(-1, 1),
            lambda: sm.bohr_radius(0),
            lambda: sm.ground_state_energy(1, m=0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_stability_of_matter: all checks passed")
