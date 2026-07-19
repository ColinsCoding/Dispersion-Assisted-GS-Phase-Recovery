"""Test Hund's rules ground-state determination against real known ion
ground terms (Fe3+ -> 6S5/2, Fe2+ -> 5D4, both textbook/experimentally
established), the Neel relaxation / AC susceptibility / SAR formulas
used in real magnetic hyperthermia cancer treatment."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import hunds_rules_magnetic_hyperthermia as hmh

# 1. Fe3+ (d5, half-filled): S=5/2, L=0, J=5/2, term 6S5/2 -- real known ground state
S, L, J, term = hmh.hunds_rules_ground_state(l=2, n_electrons=5)
assert abs(S - 2.5) < 1e-9
assert abs(L - 0) < 1e-9
assert abs(J - 2.5) < 1e-9
assert term == "6S5/2"

# 2. Fe2+ (d6, one past half-filled): S=2, L=2, J=4, term 5D4 -- real known ground state
S2, L2, J2, term2 = hmh.hunds_rules_ground_state(l=2, n_electrons=6)
assert abs(S2 - 2.0) < 1e-9
assert abs(L2 - 2.0) < 1e-9
assert abs(J2 - 4.0) < 1e-9
assert term2 == "5D4"

# 3. a single d-electron (Ti3+-like, d1): S=1/2, L=2, J=|L-S|=3/2 (less than half full)
S3, L3, J3, term3 = hmh.hunds_rules_ground_state(l=2, n_electrons=1)
assert abs(S3 - 0.5) < 1e-9
assert abs(L3 - 2.0) < 1e-9
assert abs(J3 - 1.5) < 1e-9

# 4. a fully-filled d-shell (d10, Zn2+-like): all paired, S=0, L=0, J=0
S4, L4, J4, term4 = hmh.hunds_rules_ground_state(l=2, n_electrons=10)
assert abs(S4) < 1e-9
assert abs(L4) < 1e-9
assert abs(J4) < 1e-9

# 5. lande_g_factor reproduces the same known hydrogen term-symbol values
#    as dgs.torch.stern_gerlach_zeeman_hydrogen's version (cross-check
#    between the two independent implementations)
assert abs(hmh.lande_g_factor(0, 0.5, 0.5) - 2.0) < 1e-9
assert abs(hmh.lande_g_factor(1, 0.5, 1.5) - 4.0 / 3.0) < 1e-9

# 6. effective_magnetic_moment_bohr_magnetons for Fe3+ matches the real,
#    famous free-ion value ~5.92 mu_B
mu_eff_fe3 = hmh.effective_magnetic_moment_bohr_magnetons(L, S, J)
assert abs(mu_eff_fe3 - 5.92) < 0.02

# 7. neel_relaxation_time_s: larger volume (bigger particle) -> longer
#    relaxation time (real physics: bigger anisotropy energy barrier)
tau_small = hmh.neel_relaxation_time_s(2e4, (10e-9)**3, 310.0)
tau_large = hmh.neel_relaxation_time_s(2e4, (20e-9)**3, 310.0)
assert tau_large > tau_small

# 8. ac_susceptibility_imaginary_part peaks at omega*tau = 1 (Debye
#    resonance condition) -- check it's higher there than well below or
#    well above resonance
chi0 = 2.0
tau = 1e-6
chi_at_resonance = hmh.ac_susceptibility_imaginary_part(chi0, 1.0 / tau, tau)
chi_below = hmh.ac_susceptibility_imaginary_part(chi0, 0.01 / tau, tau)
chi_above = hmh.ac_susceptibility_imaginary_part(chi0, 100.0 / tau, tau)
assert chi_at_resonance > chi_below
assert chi_at_resonance > chi_above

# 9. specific_absorption_rate_w_per_kg scales as H0^2 and linearly with f
sar_H1 = hmh.specific_absorption_rate_w_per_kg(0.5, 5e3, 1e5, 5000.0)
sar_H2 = hmh.specific_absorption_rate_w_per_kg(0.5, 10e3, 1e5, 5000.0)
assert abs(sar_H2 / sar_H1 - 4.0) < 1e-9   # (2x field)^2 = 4x SAR
sar_f1 = hmh.specific_absorption_rate_w_per_kg(0.5, 5e3, 1e5, 5000.0)
sar_f2 = hmh.specific_absorption_rate_w_per_kg(0.5, 5e3, 2e5, 5000.0)
assert abs(sar_f2 / sar_f1 - 2.0) < 1e-9

# 10. input validation
for bad_call in [
    lambda: hmh.hunds_rules_ground_state(-1, 5),
    lambda: hmh.hunds_rules_ground_state(2, 0),
    lambda: hmh.hunds_rules_ground_state(2, 11),
    lambda: hmh.lande_g_factor(0, 0.5, -1.0),
    lambda: hmh.lande_g_factor(-1, 0.5, 0.5),
    lambda: hmh.neel_relaxation_time_s(-1.0, 1e-24, 310.0),
    lambda: hmh.neel_relaxation_time_s(2e4, -1.0, 310.0),
    lambda: hmh.neel_relaxation_time_s(2e4, 1e-24, -1.0),
    lambda: hmh.ac_susceptibility_imaginary_part(-1.0, 1e6, 1e-6),
    lambda: hmh.ac_susceptibility_imaginary_part(2.0, -1.0, 1e-6),
    lambda: hmh.specific_absorption_rate_w_per_kg(-1.0, 5e3, 1e5, 5000.0),
    lambda: hmh.specific_absorption_rate_w_per_kg(0.5, -1.0, 1e5, 5000.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.hunds_rules_magnetic_hyperthermia tests passed")
