"""Test the Michelson-Morley interferometer geometry: zero-velocity limit,
the classic boat-crossing-a-river inequality, and the historical prediction."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import michelson_morley as mm

# 1. v=0 (no aether wind): both arms take the same time, exactly 2L/c
L = 11.0
t_par_0 = mm.longitudinal_time(L, 0.0)
t_perp_0 = mm.transverse_time(L, 0.0)
assert abs(t_par_0 - 2*L/mm.C) < 1e-15
assert abs(t_perp_0 - 2*L/mm.C) < 1e-15
assert abs(mm.time_difference(L, 0.0)) < 1e-15

# 2. for v > 0, the longitudinal (parallel) trip always takes LONGER than the
#    transverse trip -- the classic boat-crossing-a-river inequality
#    (t_downstream+upstream > t_straight-across at the same speed and distance)
for v in (1e4, 3e4, 1e5, 1e7):
    t_par = mm.longitudinal_time(L, v)
    t_perp = mm.transverse_time(L, v)
    assert t_par > t_perp, (v, t_par, t_perp)

# 3. historical check: Michelson & Morley's actual 1887 numbers predict a fringe
#    shift close to the well-documented ~0.4 fringes, dwarfing their <0.01 observation
v_earth = 3.0e4
wavelength = 590e-9
N = mm.predicted_fringe_shift(L, v_earth, wavelength)
assert 0.3 < N < 0.5, N
assert N > 0.01 * 20   # predicted shift is at least 20x the observed null-result ceiling

# 4. fringe shift scales as v^2 (quadratic in velocity, the reason the null
#    result was so decisive -- doubling v should very nearly quadruple N)
N_2x = mm.predicted_fringe_shift(L, 2*v_earth, wavelength)
assert abs(N_2x / N - 4.0) < 1e-9

# 5. fringe shift scales linearly with L and inversely with wavelength
N_2L = mm.predicted_fringe_shift(2*L, v_earth, wavelength)
assert abs(N_2L / N - 2.0) < 1e-9
N_half_lambda = mm.predicted_fringe_shift(L, v_earth, wavelength/2)
assert abs(N_half_lambda / N - 2.0) < 1e-9

# 6. time_difference is consistent with longitudinal_time - transverse_time exactly
for v in (1e4, 5e6):
    assert mm.time_difference(L, v) == mm.longitudinal_time(L, v) - mm.transverse_time(L, v)

# 7. input validation
for bad_call in [
    lambda: mm.longitudinal_time(-1.0, 1e4),
    lambda: mm.transverse_time(L, mm.C),        # v must be < c
    lambda: mm.transverse_time(L, -mm.C),
    lambda: mm.predicted_fringe_shift(L, v_earth, -1e-9),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.michelson_morley tests passed")
