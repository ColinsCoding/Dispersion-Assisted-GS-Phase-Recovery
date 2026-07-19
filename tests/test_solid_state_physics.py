"""Test solid state physics: packing fractions, Bragg diffraction, Fermi
energy/density of states/Fermi-Dirac consistency, and semiconductor carrier
concentration."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import solid_state_physics as ssp

# 1. packing fractions match the known textbook constants exactly
assert abs(ssp.packing_fraction("SC") - np.pi / 6) < 1e-12
assert abs(ssp.packing_fraction("BCC") - np.sqrt(3) * np.pi / 8) < 1e-12
assert abs(ssp.packing_fraction("FCC") - np.sqrt(2) * np.pi / 6) < 1e-12
assert ssp.packing_fraction("FCC") > ssp.packing_fraction("BCC") > ssp.packing_fraction("SC")
assert ssp.atoms_per_unit_cell("SC") == 1
assert ssp.atoms_per_unit_cell("BCC") == 2
assert ssp.atoms_per_unit_cell("FCC") == 4

# 1b. unit_cell_positions: sharing-weight sums recover atoms_per_unit_cell exactly,
#     and the drawn atom counts match the expected 8/9/14
for structure, n_drawn_expected in (("SC", 8), ("BCC", 9), ("FCC", 14)):
    positions, weights = ssp.unit_cell_positions(structure)
    assert len(positions) == n_drawn_expected == len(weights)
    assert abs(weights.sum() - ssp.atoms_per_unit_cell(structure)) < 1e-12
    assert positions.shape[1] == 3
    assert np.all((positions >= 0) & (positions <= 1))
try:
    ssp.unit_cell_positions("HCP")
    assert False, "should reject unknown structure"
except ValueError:
    pass

try:
    ssp.packing_fraction("HCP")
    assert False, "should reject unknown structure"
except ValueError:
    pass

# 2. Bragg diffraction: first-order angle at d=lambda/2 is exactly 90 degrees
#    (the largest possible diffraction angle -- grazing exit)
theta = ssp.bragg_angle(d_spacing=0.5e-10, wavelength=1e-10, order=1)
assert abs(theta - np.pi / 2) < 1e-9
# smaller wavelength relative to d -> smaller angle
theta_small = ssp.bragg_angle(d_spacing=2e-10, wavelength=1e-10, order=1)
assert theta_small < theta
try:
    ssp.bragg_angle(d_spacing=1e-10, wavelength=3e-10, order=1)
    assert False, "should reject impossible diffraction (ratio > 1)"
except ValueError:
    pass

# 3. copper's Fermi energy matches the textbook value (~7.0 eV) closely
n_cu = 8.47e28
E_F_cu_eV = ssp.fermi_energy(n_cu) / ssp.E_CHARGE
assert abs(E_F_cu_eV - 7.0) < 0.2, E_F_cu_eV

# 4. Fermi temperature is just E_F/k_B -- internally consistent
assert abs(ssp.fermi_temperature(n_cu) - ssp.fermi_energy(n_cu) / ssp.K_B) < 1e-6

# 5. density_of_states_3d integrates (numerically) back to n over [0, E_F] --
#    the normalization is supposed to guarantee this by construction
E_F = ssp.fermi_energy(n_cu)
E_grid = np.linspace(0, E_F, 200_000)
g = ssp.density_of_states_3d(E_grid, n_cu)
n_recovered = np.trapezoid(g, E_grid)
assert abs(n_recovered - n_cu) / n_cu < 1e-3, (n_recovered, n_cu)

# 6. Fermi-Dirac at T=0 is a hard step: 1 below E_F, 0 above
f0 = ssp.fermi_dirac(np.array([E_F * 0.5, E_F * 1.5]), E_F, T=0)
assert f0[0] == 1.0 and f0[1] == 0.0
# at E = E_F exactly, occupation is always 1/2 regardless of T
assert ssp.fermi_dirac(E_F, E_F, T=300) == 0.5
assert ssp.fermi_dirac(E_F, E_F, T=0) == 0.5
# far above E_F at finite T, occupation should be small but positive
f_above = ssp.fermi_dirac(E_F + 20 * ssp.K_B * 300, E_F, T=300)
assert 0 < f_above < 1e-6

# 7. semiconductor carrier concentration rises with T (opposite of a metal)
n_200 = ssp.intrinsic_carrier_concentration(200.0, 1.12)
n_300 = ssp.intrinsic_carrier_concentration(300.0, 1.12)
n_400 = ssp.intrinsic_carrier_concentration(400.0, 1.12)
assert n_200 < n_300 < n_400
# a zero-gap material has weaker T-dependence than a large-gap one (activation
# energy term vanishes, leaving only the T^1.5 prefactor)
n0_200 = ssp.intrinsic_carrier_concentration(200.0, 0.0)
n0_400 = ssp.intrinsic_carrier_concentration(400.0, 0.0)
ratio_gapped = n_400 / n_200
ratio_gapless = n0_400 / n0_200
assert ratio_gapped > ratio_gapless

# 8. classify_material thresholds
assert ssp.classify_material(0.0) == "conductor"
assert ssp.classify_material(-0.5) == "conductor"
assert ssp.classify_material(1.12) == "semiconductor"
assert ssp.classify_material(5.0) == "insulator"

# 9. input validation
for bad_call in [
    lambda: ssp.fermi_energy(-1.0),
    lambda: ssp.bragg_angle(-1.0, 1e-10),
    lambda: ssp.bragg_angle(1e-10, 1e-10, order=0),
    lambda: ssp.fermi_dirac(1.0, 1.0, T=-1),
    lambda: ssp.intrinsic_carrier_concentration(-1.0, 1.0),
    lambda: ssp.intrinsic_carrier_concentration(300.0, -1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

# 10. Drude conductivity of copper matches the real value closely
n_cu = 8.47e28
sigma_cu = ssp.drude_conductivity(n_cu, tau=2.5e-14)
assert abs(sigma_cu - 5.96e7) / 5.96e7 < 0.01

# 11. Hall coefficient sign reveals carrier type: electrons (q<0) give R_H<0,
#     holes (q>0) give R_H>0, same |value| for the same density
R_H_electrons = ssp.hall_coefficient(n_cu, q=-ssp.E_CHARGE)
R_H_holes = ssp.hall_coefficient(n_cu, q=+ssp.E_CHARGE)
assert R_H_electrons < 0 < R_H_holes
assert abs(abs(R_H_electrons) - abs(R_H_holes)) < 1e-30

# 12. Hall voltage matches hall_coefficient*I*B/thickness exactly (definitional)
I, B, thickness = 1.0, 0.5, 1e-3
V_H = ssp.hall_voltage(I, B, n_cu, thickness)
assert abs(V_H - ssp.hall_coefficient(n_cu)*I*B/thickness) < 1e-30

# 13. Lorenz number matches the experimental value (~2.44e-8 W*Ohm/K^2) closely
L = ssp.lorenz_number()
assert abs(L - 2.44e-8) / 2.44e-8 < 0.01

# 14. Wiedemann-Franz predicts copper's real thermal conductivity (~401 W/(m K))
#     within a reasonable margin of the free-electron approximation
kappa_cu = ssp.thermal_conductivity_from_electrical(sigma_cu, 300.0)
assert abs(kappa_cu - 401.0) / 401.0 < 0.15

# 15. flux quantum and Josephson frequency match known exact values to high precision
assert abs(ssp.flux_quantum() - 2.067833848e-15) / 2.067833848e-15 < 1e-6
assert abs(ssp.josephson_frequency(1.0) - 4.835978484e14) / 4.835978484e14 < 1e-6
# Josephson frequency scales linearly with voltage
assert abs(ssp.josephson_frequency(2.0) / ssp.josephson_frequency(1.0) - 2.0) < 1e-9

# 16. more validation
for bad_call in [
    lambda: ssp.drude_conductivity(-1.0, 1e-14),
    lambda: ssp.drude_conductivity(n_cu, -1e-14),
    lambda: ssp.hall_coefficient(-1.0),
    lambda: ssp.hall_coefficient(n_cu, q=0.0),
    lambda: ssp.hall_voltage(I, B, n_cu, thickness=-1.0),
    lambda: ssp.thermal_conductivity_from_electrical(-1.0, 300.0),
    lambda: ssp.thermal_conductivity_from_electrical(sigma_cu, -1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.solid_state_physics tests passed")
