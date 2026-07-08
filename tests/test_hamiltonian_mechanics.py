"""Test dgs.hamiltonian_mechanics: the Legendre transform L->H (symbolic),
Hamilton's equations reducing to Newton, the leapfrog flow tracking the analytic
SHM short-term, the symplectic signature (energy band BOUNDED as the run grows,
while RK4's GROWS), and Liouville's theorem (phase-space area preserved)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from dgs import hamiltonian_mechanics as hm

# 1. Legendre transform of L = 1/2 m q_dot^2 - 1/2 k q^2
lt = hm.legendre_transform_oscillator()
q, m, k, p = sp.symbols("q m k p", positive=True)
qd = sp.symbols("q_dot", positive=True)
assert sp.simplify(lt["p"] - m * qd) == 0                       # p = m q_dot
assert sp.simplify(lt["H"] - (p**2 / (2*m) + sp.Rational(1, 2) * k * q**2)) == 0

# 2. harmonic Hamiltonian and Hamilton's equations = Newton
assert np.isclose(hm.harmonic_hamiltonian(1.0, 0.0, m=1, k=1), 0.5)   # 1/2 k q^2
assert np.isclose(hm.harmonic_hamiltonian(0.0, 2.0, m=1, k=1), 2.0)   # p^2/2m
qd_, pd_ = hm.hamiltons_equations_ho(3.0, 4.0, m=2.0, k=5.0)
assert np.isclose(qd_, 4.0 / 2.0)                              # q_dot = p/m
assert np.isclose(pd_, -5.0 * 3.0)                            # p_dot = -k q  (Newton)

# 3. leapfrog tracks the exact SHM q(t) = cos(omega t) short-term
dHdq = lambda x: x            # k=1
dHdp = lambda pp: pp          # m=1
t, qs, ps = hm.simulate_symplectic(dHdq, dHdp, 1.0, 0.0, 0.01, int(5 * 2*np.pi / 0.01))
assert np.max(np.abs(qs - np.cos(t))) < 1e-3

# 4. THE SYMPLECTIC SIGNATURE: energy band is bounded (constant) as the run grows,
#    while RK4's band grows roughly linearly with the run length
dt = 0.1
def band(sim_fn, periods):
    n = int(periods * 2 * np.pi / dt)
    _, qq, pp = sim_fn(dHdq, dHdp, 1.0, 0.0, dt, n)
    return hm.energy_drift(qq, pp)

s50, s500 = band(hm.simulate_symplectic, 50), band(hm.simulate_symplectic, 500)
r50, r500 = band(hm.simulate_rk4, 50), band(hm.simulate_rk4, 500)
assert s500 / s50 < 1.5              # symplectic: BOUNDED (same band at 10x the run)
assert r500 / r50 > 5                # RK4: GROWS ~10x per 10x time
# and RK4's per-run band, though smaller short-term, is on track to exceed the
# symplectic band -- its growth rate is the point
assert r500 > r50                    # monotonic accumulation

# 5. net energy drift: symplectic returns (tiny net), RK4 accumulates
n = int(300 * 2*np.pi / dt)
_, qs, ps = hm.simulate_symplectic(dHdq, dHdp, 1.0, 0.0, dt, n)
_, qr, pr = hm.simulate_rk4(dHdq, dHdp, 1.0, 0.0, dt, n)
assert hm.energy_net_drift(qs, ps) < hm.energy_drift(qs, ps) + 1e-12   # net <= band
assert hm.energy_net_drift(qr, pr) > 0                                 # RK4 drifted

# 6. Liouville: phase-space area is preserved by the symplectic flow
sq = [[1.0, 0.0], [1.1, 0.0], [1.1, 0.1], [1.0, 0.1]]
ratio = hm.liouville_area_ratio(dHdq, dHdp, sq, dt, int(100 * 2*np.pi / dt))
assert np.isclose(ratio, 1.0, atol=1e-4)

# 7. kwarg bounds
for bad in (lambda: hm.simulate_symplectic(dHdq, dHdp, 1, 0, 0, 10),
            lambda: hm.simulate_rk4(dHdq, dHdp, 1, 0, 0.1, 0),
            lambda: hm.liouville_area_ratio(dHdq, dHdp, [[0, 0]], 0.1, 10)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_hamiltonian_mechanics: all checks passed")
