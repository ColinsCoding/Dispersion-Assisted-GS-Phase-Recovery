"""Test food science kinetics: D-value/z-value round-trip consistency, the
Bigelow model, F-value integration, Q10, and shelf-life-to-threshold."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import food_science_kinetics as fsk
from dgs import reaction_rates as rr

# 1. D_value and microbial_survivors are inverses of each other
N0, D_true, t_probe = 1e6, 2.5, 4.0
N_at_t = fsk.microbial_survivors(N0, t_probe, D_true)
D_recovered = fsk.D_value(N0, N_at_t, t_probe)
assert abs(D_recovered - D_true) < 1e-9, (D_recovered, D_true)

# 2. one D-value drop is exactly a 90% reduction (N/N0 = 0.1 at t=D)
N_at_D = fsk.microbial_survivors(N0, D_true, D_true)
assert abs(N_at_D / N0 - 0.1) < 1e-12

# 3. z_value and D_at_temperature are inverses: build D2 from a known z, recover it
T1, T2, z_true = 121.1, 130.0, 10.0
D1 = 0.21
D2 = fsk.D_at_temperature(D1, T2, T1, z_true)
z_recovered = fsk.z_value(D1, D2, T1, T2)
assert abs(z_recovered - z_true) < 1e-9

# 4. lethality_rate(T_ref) == 1 always, regardless of z
for z in (5.0, 10.0, 18.0):
    assert abs(fsk.lethality_rate(121.1, 121.1, z) - 1.0) < 1e-12

# 5. isothermal process held exactly at T_ref for duration t has F-value == t exactly
t = np.linspace(0, 5.0, 1000)
T_profile = np.full_like(t, 121.1)
F = fsk.F_value(t, T_profile, T_ref=121.1, z=10.0)
assert abs(F - 5.0) < 1e-6

# 6. a hotter-than-reference process accumulates lethality FASTER than real time
T_hot = np.full_like(t, 131.1)  # one z above reference -> 10x lethality rate
F_hot = fsk.F_value(t, T_hot, T_ref=121.1, z=10.0)
assert abs(F_hot - 10 * 5.0) < 1e-3, F_hot   # constant 10x rate -> 10x the F-value

# 7. Q10: rate constant doubling over exactly 10 degrees -> Q10 == 2
assert abs(fsk.q10_coefficient(1.0, 2.0, 0.0, 10.0) - 2.0) < 1e-12
# tripling over 20 degrees -> Q10 = 3^(10/20) = sqrt(3)
assert abs(fsk.q10_coefficient(1.0, 3.0, 0.0, 20.0) - np.sqrt(3)) < 1e-9

# 8. shelf_life_first_order agrees with dgs.reaction_rates' own first-order curve:
#    plugging the computed shelf-life time back into integrated_concentration
#    must reproduce the requested remaining fraction exactly
k, frac = 0.05, 0.5
t_shelf = fsk.shelf_life_first_order(k, frac)
A_remaining = rr.integrated_concentration(A0=1.0, k=k, t=t_shelf, order=1)
assert abs(float(A_remaining) - frac) < 1e-9
# and it must equal reaction_rates' own half_life exactly when frac=0.5
assert abs(t_shelf - rr.half_life(A0=1.0, k=k, order=1)) < 1e-9

# 9. input validation
for bad_call in [
    lambda: fsk.D_value(-1, 1, 1),
    lambda: fsk.D_value(10, 20, 1),          # N must be < N0
    lambda: fsk.microbial_survivors(1e6, 1.0, -1.0),
    lambda: fsk.z_value(1.0, 1.0, 100.0, 100.0),   # T1 == T2
    lambda: fsk.D_at_temperature(-1.0, 130.0, 121.1, 10.0),
    lambda: fsk.lethality_rate(130.0, 121.1, -10.0),
    lambda: fsk.q10_coefficient(-1.0, 2.0, 0.0, 10.0),
    lambda: fsk.shelf_life_first_order(0.05, 1.5),   # fraction must be in (0,1)
    lambda: fsk.shelf_life_first_order(-0.05, 0.5),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.food_science_kinetics tests passed")
