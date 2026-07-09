"""Test dgs.harmonic_approximation: the local derivatives, stable-minimum test,
omega=sqrt(V''/m) on an exact spring / pendulum / Lennard-Jones well, the parabola,
and the anharmonic error growing with amplitude (zero for an exact spring)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import harmonic_approximation as ha

k = 3.0
spring = lambda x: 0.5 * k * x ** 2

# 1. derivatives of the spring: V'(x0)=k x0, V''=k, V'(0)=0
V0, Vp, Vpp = ha.derivatives(spring, 0.0)
assert np.isclose(V0, 0.0) and np.isclose(Vp, 0.0, atol=1e-6) and np.isclose(Vpp, k, rtol=1e-4)
_, Vp2, _ = ha.derivatives(spring, 2.0)
assert np.isclose(Vp2, k * 2.0, rtol=1e-4)                   # slope = k x0

# 2. stable minimum: a well is stable, a hill is not
assert ha.is_stable_minimum(spring, 0.0)
assert not ha.is_stable_minimum(spring, 2.0)                 # not stationary
assert not ha.is_stable_minimum(lambda x: -0.5 * k * x ** 2, 0.0)   # V''<0, a hill

# 3. find_minimum locates a shifted parabola's bottom
shifted = lambda x: 0.5 * k * (x - 2.0) ** 2 + 3.0
assert np.isclose(ha.find_minimum(shifted, (-10, 10)), 2.0, atol=1e-4)

# 4. harmonic frequency = sqrt(V''/m); exact spring gives sqrt(k/m)
assert np.isclose(ha.harmonic_frequency(spring, 0.0, mass=1.0), np.sqrt(k), rtol=1e-4)
assert np.isclose(ha.harmonic_frequency(spring, 0.0, mass=4.0), np.sqrt(k / 4), rtol=1e-4)
# pendulum V=g(1-cos), omega -> sqrt(g/l) (here l=m=1 -> sqrt(g))
g = 9.81
pend = lambda th: g * (1 - np.cos(th))
assert np.isclose(ha.harmonic_frequency(pend, 0.0), np.sqrt(g), rtol=1e-3)
assert np.isclose(ha.small_oscillation_period(pend, 0.0), 2 * np.pi / np.sqrt(g), rtol=1e-3)
# a maximum has no real oscillation -> error
try:
    ha.harmonic_frequency(lambda x: -0.5 * k * x ** 2, 0.0); assert False
except ValueError:
    pass

# 5. the parabola IS the potential for an exact spring (error identically 0)
x = np.linspace(-4, 4, 100)
assert np.allclose(ha.harmonic_potential(spring, 0.0, x), spring(x), atol=1e-6)
assert ha.approximation_error(spring, 0.0, 5.0) < 1e-8

# 6. anharmonic error grows with amplitude for a real (non-parabolic) potential
e_small = ha.approximation_error(pend, 0.0, 0.2)
e_large = ha.approximation_error(pend, 0.0, 1.5)
assert e_small < 1e-2 < e_large                             # good when small, breaks when large
assert e_large > e_small

# 7. Lennard-Jones well matches its analytic harmonic frequency
eps, sigma, mu = 1.0, 1.0, 200.0
lj = lambda r: 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
r0 = ha.find_minimum(lj, (0.9, 2.0))
assert np.isclose(r0, 2 ** (1 / 6), rtol=1e-4)
assert ha.is_stable_minimum(lj, r0)
assert np.isclose(ha.harmonic_frequency(lj, r0, mu),
                  np.sqrt(36 * 2 ** (2 / 3) / mu), rtol=1e-3)

# 8. kwarg bounds
for bad in (lambda: ha.harmonic_frequency(spring, 0.0, mass=0),
            lambda: ha.harmonic_frequency(lambda x: -x**2, 0.0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_harmonic_approximation: all checks passed")
