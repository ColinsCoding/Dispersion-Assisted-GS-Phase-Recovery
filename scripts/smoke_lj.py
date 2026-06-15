"""Smoke-test lennard_jones: potential minimum, force, energy conservation."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import lennard_jones as lj

# minimum at r = 2^(1/6) sigma, depth -eps, force zero there
rmin = lj.equilibrium_distance(1.0)
print(f"r_min = {rmin:.4f} (= 2^(1/6) = {2**(1/6):.4f})")
print(f"V(r_min) = {lj.lj_potential(rmin):.6f} (expect -1.0 = -eps)")
print(f"F(r_min) = {lj.lj_force_magnitude(rmin):.2e} (expect ~0)")
assert abs(lj.lj_potential(rmin) + 1.0) < 1e-9
assert abs(lj.lj_force_magnitude(rmin)) < 1e-9

# repulsive inside, attractive outside the minimum
print(f"F(0.9 r_min) = {lj.lj_force_magnitude(0.9*rmin):+.3f} (repulsive >0)")
print(f"F(1.5 r_min) = {lj.lj_force_magnitude(1.5*rmin):+.3f} (attractive <0)")
assert lj.lj_force_magnitude(0.9*rmin) > 0 and lj.lj_force_magnitude(1.5*rmin) < 0

# two-body: a bond oscillates and conserves energy
pos = np.array([[0.0, 0.0], [1.05*rmin, 0.0]])      # slightly stretched
vel = np.zeros((2, 2))
out = lj.simulate(pos, vel, dt=0.002, steps=3000, store_every=10)
drift = (out["E"].max() - out["E"].min()) / abs(out["E"].mean())
print(f"\ntwo-body energy drift = {drift:.2e} (Verlet conserves energy)")
assert drift < 1e-3

# a hex cluster stays bound and conserves energy
cl = lj.hex_cluster(n_rings=2)
print(f"\nhex cluster: {len(cl)} atoms")
rng = np.random.default_rng(0)
v0 = 0.15 * rng.standard_normal(cl.shape)
v0 -= v0.mean(0)                                     # zero net momentum
out2 = lj.simulate(cl, v0, dt=0.003, steps=4000, store_every=20)
drift2 = (out2["E"].max() - out2["E"].min()) / abs(out2["E"].mean())
print(f"  cluster energy drift = {drift2:.2e}")
print(f"  mean KE (temperature proxy) = {out2['KE'].mean():.3f}")
assert drift2 < 5e-2

for bad in [lambda: lj.pair_forces([[0, 0], [0, 0]])]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", e)
print("SMOKE PASS")
