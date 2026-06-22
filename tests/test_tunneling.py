"""Test quantum tunneling: decay constant, exact barrier T, and the WKB exponent."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import tunneling as tn

E, V0, m, hbar = 1.0, 2.0, 1.0, 1.0

# 1. decay constant kappa = sqrt(2 m (V0-E))/hbar
k = tn.barrier_decay_constant(E, V0)
assert np.isclose(k, np.sqrt(2 * (V0 - E)))                # sqrt(2) here

# 2. exact transmission decreases monotonically (exponentially) with barrier width
widths = np.array([0.5, 1.0, 2.0, 3.0])
Te = np.array([tn.rectangular_barrier_T(E, V0, a) for a in widths])
assert np.all(np.diff(Te) < 0)                            # thicker -> less tunneling
# and decreases with barrier HEIGHT (bigger V0 -> smaller T at fixed width)
assert tn.rectangular_barrier_T(E, 3.0, 1.0) < tn.rectangular_barrier_T(E, 2.0, 1.0)

# 3. WKB on a rectangular grid equals exp(-2 kappa a)
a = 2.0
x = np.linspace(0, a, 600); V = np.full_like(x, V0)
assert np.isclose(tn.wkb_transmission(E, V, x), np.exp(-2 * k * a), rtol=1e-3)

# 4. WKB captures the EXACT exponent: for a thick barrier the ratio -> the O(1)
#    prefactor 16 E (V0-E) / V0^2 = 4 here
for a in (3.0, 4.0, 5.0):
    xa = np.linspace(0, a, 800); Va = np.full_like(xa, V0)
    ratio = tn.rectangular_barrier_T(E, V0, a) / tn.wkb_transmission(E, Va, xa)
    assert abs(ratio - 16 * E * (V0 - E) / V0**2) < 0.05   # -> 4.0, tighter as a grows

# 5. the log-slope of exact T vs width approaches -2 kappa (the WKB exponent)
a1, a2 = 4.0, 5.0
slope = (np.log(tn.rectangular_barrier_T(E, V0, a2)) -
         np.log(tn.rectangular_barrier_T(E, V0, a1))) / (a2 - a1)
assert abs(slope - (-2 * k)) < 1e-2                        # d(lnT)/da -> -2 kappa

# 6. the wavefunction decays as exp(-kappa x) inside the barrier
xb = np.linspace(0, 3, 50)
psi = tn.barrier_wavefunction(xb, 0.0, E, V0)
assert np.allclose(psi, np.exp(-k * xb))
assert psi[-1] < psi[0]                                    # decays into the wall

# 7. a smooth (triangular) barrier: WKB integrates the varying kappa, T < a thin slab
xt = np.linspace(-2, 2, 400)
Vtri = np.maximum(0.0, V0 * (1 - np.abs(xt) / 2)) + E*0    # triangular peak V0 at center
T_tri = tn.wkb_transmission(E, np.maximum(Vtri, 0) + 0.0, xt)
assert 0 < T_tri < 1

print(f"TEST PASS  (kappa=sqrt(2(V0-E))={k:.3f}; exact T exponential in width/height; "
      f"WKB=exp(-2 kappa a); ratio->prefactor 4; log-slope -> -2 kappa; "
      f"psi~exp(-kappa x))")
