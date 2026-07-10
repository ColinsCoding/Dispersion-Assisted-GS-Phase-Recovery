"""Test dgs.square_barrier: the joining-condition solver reproduces the closed-form T for E<V0,
E=V0 and E>V0, obeys R+T=1, ties to dgs.tunneling, and hits T=1 at over-barrier resonances."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from dgs import square_barrier as sb

m = hbar = 1.0
V0, L = 10.0, 1.0

# 1. matching solver == closed form, tunneling regime (E < V0)
for E in (0.5, 1.0, 3.0, 5.0, 9.0, 9.9):
    assert math.isclose(sb.transmission_coefficient(E, V0, L),
                        sb.transmission_closed_form(E, V0, L), rel_tol=1e-9)

# 2. matching solver == closed form, over-barrier regime (E > V0)
for E in (10.5, 12.0, 15.0, 25.0):
    assert math.isclose(sb.transmission_coefficient(E, V0, L),
                        sb.transmission_closed_form(E, V0, L), rel_tol=1e-9)

# 3. E = V0 limit is continuous (match, closed form, and the two side limits agree)
Tv = sb.transmission_coefficient(V0, V0, L)
assert math.isclose(Tv, sb.transmission_closed_form(V0, V0, L), rel_tol=1e-9)
assert math.isclose(sb.transmission_coefficient(V0*0.9999, V0, L), Tv, rel_tol=1e-3)
assert math.isclose(sb.transmission_coefficient(V0*1.0001, V0, L), Tv, rel_tol=1e-3)

# 4. probability conservation R + T = 1 in every regime
for E in (1.0, 5.0, 9.0, 10.0, 12.0, 30.0):
    T = sb.transmission_coefficient(E, V0, L)
    R = sb.reflection_coefficient(E, V0, L)
    assert math.isclose(T + R, 1.0, abs_tol=1e-9)

# 5. cross-check dgs.tunneling.rectangular_barrier_T for E < V0
from dgs import tunneling as tn
for E in (1.0, 4.0, 8.0):
    assert math.isclose(sb.transmission_coefficient(E, V0, L),
                        tn.rectangular_barrier_T(E, V0, L), rel_tol=1e-9)

# 6. tunneling: T tiny and falls exponentially with width; R -> 1
T_thin = sb.transmission_coefficient(2.0, V0, 0.5)
T_thick = sb.transmission_coefficient(2.0, V0, 2.0)
assert T_thick < T_thin < 1
assert sb.reflection_coefficient(2.0, V0, 3.0) > 0.999      # thick barrier reflects almost all
# exponential law: log T ~ -2 alpha L, so slope vs L ~ -2 alpha
alpha = math.sqrt(2*m*(V0-2.0))/hbar
Ls = np.array([1.0, 2.0, 3.0, 4.0])
logT = np.array([math.log(sb.transmission_coefficient(2.0, V0, Lx)) for Lx in Ls])
slope = np.polyfit(Ls, logT, 1)[0]
assert math.isclose(slope, -2*alpha, rel_tol=0.05)

# 7. over-barrier resonances: T = 1 exactly at k2 L = n pi
for Eres in sb.resonance_energies(V0, L, n_max=4):
    assert math.isclose(sb.transmission_coefficient(Eres, V0, L), 1.0, abs_tol=1e-9)
# and strictly below 1 between two resonances
E_between = (sb.resonance_energies(V0, L, 2)[0] + sb.resonance_energies(V0, L, 2)[1]) / 2
assert sb.transmission_coefficient(E_between, V0, L) < 0.9999

# 8. free particle (V0 = 0): full transmission, no reflection
assert math.isclose(sb.transmission_coefficient(5.0, 0.0, L), 1.0, abs_tol=1e-9)
assert sb.reflection_coefficient(5.0, 0.0, L) < 1e-9

# 9. kwarg bounds
for bad in (lambda: sb.wavevector(0),
            lambda: sb.transmission_coefficient(-1, V0, L),
            lambda: sb.transmission_coefficient(5, V0, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_square_barrier: all checks passed")
