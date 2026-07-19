"""Test Stern-Gerlach silver-beam deflection physics and the anomalous
Zeeman effect's Lande g-factors against real known textbook values
(2S_1/2 -> g=2, 2P_1/2 -> g=2/3, 2P_3/2 -> g=4/3). Requires py-3.12 (torch)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.torch import stern_gerlach_zeeman_hydrogen as sgz

# 1. Lande g-factors match real known hydrogen textbook values exactly
assert abs(sgz.lande_g_factor(L=0, S=0.5, J=0.5) - 2.0) < 1e-9
assert abs(sgz.lande_g_factor(L=1, S=0.5, J=0.5) - 2.0 / 3.0) < 1e-9
assert abs(sgz.lande_g_factor(L=1, S=0.5, J=1.5) - 4.0 / 3.0) < 1e-9

# 2. zeeman_sublevel_energies_j: J=1/2 gives exactly 2 states, symmetric about 0
m_J, dE = sgz.zeeman_sublevel_energies_j(g_J=2.0, J=0.5, B_tesla=1.0)
assert len(m_J) == 2
assert abs(float(dE[0]) + float(dE[1])) < 1e-30   # symmetric split about zero

# 3. zeeman_sublevel_energies_j scales linearly with B
_, dE_1T = sgz.zeeman_sublevel_energies_j(2.0, 0.5, 1.0)
_, dE_2T = sgz.zeeman_sublevel_energies_j(2.0, 0.5, 2.0)
assert abs(float(dE_2T[-1]) / float(dE_1T[-1]) - 2.0) < 1e-9

# 4. is_even_split: J=1/2 (pure spin) is even; J=1 (integer, orbital-like) is odd
assert sgz.is_even_split(0.5) is True
assert sgz.is_even_split(1.0) is False
assert sgz.is_even_split(1.5) is True

# 5. most_probable_beam_speed_m_s: hotter oven -> faster beam
v_cold = sgz.most_probable_beam_speed_m_s(300.0, 1.79e-25)
v_hot = sgz.most_probable_beam_speed_m_s(1200.0, 1.79e-25)
assert v_hot > v_cold
assert abs(v_hot / v_cold - 2.0) < 1e-6   # sqrt(T) scaling -> sqrt(4)=2

# 6. stern_gerlach_deflection_m: stronger gradient -> more deflection;
#    deflection lands in a realistic sub-mm-to-few-mm range
z_weak = sgz.stern_gerlach_deflection_m(500.0, 0.035, 0.035, 1.79e-25, 430.0)
z_strong = sgz.stern_gerlach_deflection_m(1000.0, 0.035, 0.035, 1.79e-25, 430.0)
assert z_strong > z_weak
assert 1e-5 < z_strong < 1e-2   # meters -- realistic 0.01mm to 10mm range

# 7. zeeman_transition_lines: 2P_3/2 -> 2S_1/2 produces MORE than the "normal"
#    Zeeman triplet (3 lines) because g_upper != g_lower -- the anomalous pattern
lines, g_upper, g_lower = sgz.zeeman_transition_lines(
    L_upper=1, S_upper=0.5, J_upper=1.5,
    L_lower=0, S_lower=0.5, J_lower=0.5,
    B_tesla=0.5, E0_joules=1.634e-18)
assert abs(g_upper - 4.0 / 3.0) < 1e-9
assert abs(g_lower - 2.0) < 1e-9
assert len(lines) > 3

# 8. at B=0, all lines collapse to the same (unshifted) energy E0
lines_zero_field, _, _ = sgz.zeeman_transition_lines(
    L_upper=1, S_upper=0.5, J_upper=1.5,
    L_lower=0, S_lower=0.5, J_lower=0.5,
    B_tesla=0.0, E0_joules=1.634e-18)
energies = [l["energy_j"] for l in lines_zero_field]
assert max(energies) - min(energies) < 1e-30

# 9. plot_zeeman_spectrum produces a real file
import os
save_path = sgz.plot_zeeman_spectrum(lines, save_path="_test_zeeman_spectrum.png")
assert os.path.exists(save_path)
os.remove(save_path)

# 10. input validation
for bad_call in [
    lambda: sgz.lande_g_factor(L=0, S=0.5, J=-1.0),
    lambda: sgz.lande_g_factor(L=-1, S=0.5, J=0.5),
    lambda: sgz.zeeman_sublevel_energies_j(2.0, -1.0, 1.0),
    lambda: sgz.zeeman_sublevel_energies_j(2.0, 0.5, -1.0),
    lambda: sgz.is_even_split(-1.0),
    lambda: sgz.stern_gerlach_deflection_m(-1.0, 0.035, 0.035, 1.79e-25, 430.0),
    lambda: sgz.stern_gerlach_deflection_m(500.0, -1.0, 0.035, 1.79e-25, 430.0),
    lambda: sgz.stern_gerlach_deflection_m(500.0, 0.035, 0.035, -1.0, 430.0),
    lambda: sgz.stern_gerlach_deflection_m(500.0, 0.035, 0.035, 1.79e-25, -1.0),
    lambda: sgz.most_probable_beam_speed_m_s(-1.0, 1.79e-25),
    lambda: sgz.most_probable_beam_speed_m_s(300.0, -1.0),
    lambda: sgz.zeeman_transition_lines(1, 0.5, 1.5, 0, 0.5, 0.5, -1.0, 1.634e-18),
    lambda: sgz.plot_zeeman_spectrum([]),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.torch.stern_gerlach_zeeman_hydrogen tests passed")
