"""Test photon energy (E=hc/lambda, SI-exact constants), the Rayleigh
lithography resolution limit, Deal-Grove oxidation kinetics -- checked
against real known values: ArF (193nm) ~6.4 eV, KrF (248nm) ~5.0 eV,
EUV (13.5nm) ~91.8 eV; 193i lithography ~40nm-class resolution, EUV
~13nm-class."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import silicon_photonics_manufacturing_chemistry as spm

# 1. photon_energy_ev matches real known lithography photon energies
assert abs(spm.photon_energy_ev(193e-9) - 6.42) < 0.02
assert abs(spm.photon_energy_ev(248e-9) - 5.00) < 0.02
assert abs(spm.photon_energy_ev(13.5e-9) - 91.84) < 0.5

# 2. E=hc/lambda is exactly inversely proportional to wavelength
E1 = spm.photon_energy_ev(200e-9)
E2 = spm.photon_energy_ev(400e-9)
assert abs(E1 / E2 - 2.0) < 1e-9

# 3. rayleigh_resolution_m: shorter wavelength -> smaller (better) resolution;
#    matches real order-of-magnitude values for 193i and EUV
CD_193i = spm.rayleigh_resolution_m(193e-9, NA=1.35, k1=0.30)
CD_euv = spm.rayleigh_resolution_m(13.5e-9, NA=0.33, k1=0.30)
assert CD_euv < CD_193i
assert 30e-9 < CD_193i < 60e-9    # real: ~40nm-class
assert 8e-9 < CD_euv < 20e-9      # real: ~13nm-class

# 4. rayleigh_resolution_m scales linearly with k1 and inversely with NA
CD_k1_a = spm.rayleigh_resolution_m(193e-9, 1.35, k1=0.2)
CD_k1_b = spm.rayleigh_resolution_m(193e-9, 1.35, k1=0.4)
assert abs(CD_k1_b / CD_k1_a - 2.0) < 1e-9

# 5. depth_of_focus_m: larger NA -> shallower depth of focus (real tradeoff:
#    better resolution costs you focus margin)
DOF_low_NA = spm.depth_of_focus_m(193e-9, NA=0.5)
DOF_high_NA = spm.depth_of_focus_m(193e-9, NA=1.35)
assert DOF_high_NA < DOF_low_NA

# 6. deal_grove_oxide_thickness_m: monotonically increasing with time,
#    and thickness(0) = 0
x0 = spm.deal_grove_oxide_thickness_m(0.0, A_um=0.235, B_um2_per_hr=0.0117)
x1 = spm.deal_grove_oxide_thickness_m(1.0, A_um=0.235, B_um2_per_hr=0.0117)
x2 = spm.deal_grove_oxide_thickness_m(2.0, A_um=0.235, B_um2_per_hr=0.0117)
assert abs(x0) < 1e-12
assert x1 > 0
assert x2 > x1

# 7. deal_grove satisfies its own defining equation x^2 + A*x = B*(t+tau)
A_um, B_um2_per_hr, tau_hr, t_hr = 0.235, 0.0117, 0.0, 3.0
x_m = spm.deal_grove_oxide_thickness_m(t_hr, A_um, B_um2_per_hr, tau_hr)
x_um = x_m * 1e6
lhs = x_um**2 + A_um * x_um
rhs = B_um2_per_hr * (t_hr + tau_hr)
assert abs(lhs - rhs) < 1e-9

# 8. SILICON_PHOTONICS_PROCESS_CHEMISTRY has the expected real process steps
assert "thermal_oxidation" in spm.SILICON_PHOTONICS_PROCESS_CHEMISTRY
assert "waveguide_dry_etch" in spm.SILICON_PHOTONICS_PROCESS_CHEMISTRY
for step, d in spm.SILICON_PHOTONICS_PROCESS_CHEMISTRY.items():
    assert "reaction" in d and "role" in d and "kinetics" in d

# 9. input validation
for bad_call in [
    lambda: spm.photon_energy_ev(-1.0),
    lambda: spm.rayleigh_resolution_m(-1.0, 1.0),
    lambda: spm.rayleigh_resolution_m(193e-9, -1.0),
    lambda: spm.rayleigh_resolution_m(193e-9, 1.0, k1=-1.0),
    lambda: spm.depth_of_focus_m(-1.0, 1.0),
    lambda: spm.depth_of_focus_m(193e-9, -1.0),
    lambda: spm.depth_of_focus_m(193e-9, 1.0, k2=-1.0),
    lambda: spm.deal_grove_oxide_thickness_m(-1.0, 0.235, 0.0117),
    lambda: spm.deal_grove_oxide_thickness_m(1.0, -1.0, 0.0117),
    lambda: spm.deal_grove_oxide_thickness_m(1.0, 0.235, -1.0),
    lambda: spm.deal_grove_oxide_thickness_m(1.0, 0.235, 0.0117, tau_hr=-1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.silicon_photonics_manufacturing_chemistry tests passed")
