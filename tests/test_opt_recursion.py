"""Smoke-test opt_recursion: closed form, stability, and the bifurcation."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from dgs import opt_recursion as opt

a, eta, x0 = sp.symbols("a eta x_0", positive=True)

# 1. closed form of GD on (a/2)x^2: x_n = (1-eta a)^n x0
xn, factor = opt.gd_closed_form(a, x0, eta)
print("closed form x_n =", xn, " (expect x0 (1-eta a)^n)")
print("contraction factor =", factor, "  converges iff |1-eta a|<1 -> 0<eta<2/a")
assert sp.simplify(xn - x0 * (1 - eta * a)**sp.Symbol("n", integer=True, nonnegative=True)) == 0

# 2. fixed-point stability for f = (a/2) x^2 (single min at 0)
fps = opt.fixed_point_stability(a/2 * sp.Symbol("x")**2, sp.Symbol("x"), eta)
for xs, mult, cond in fps:
    print(f"  fixed point x*={xs}, multiplier g'={mult}, stable when {cond}")

# 3. numeric convergence: GD on f=x^2/2 with eta=0.5 from x0=1 -> 0
g = opt.gd_step(lambda x: x, 0.5)            # f'=x
traj = opt.iterate_map(g, 1.0, 30)
print("\nGD eta=0.5 from x0=1:", np.round(traj[:6], 4), "-> converges to", round(traj[-1], 6))

# eta=2.1/a (a=1) overshoots and diverges
g_div = opt.gd_step(lambda x: x, 2.1)
traj_div = opt.iterate_map(g_div, 0.1, 12)
print("GD eta=2.1 (>2): diverges:", np.round(traj_div[-3:], 2))

# 4. bifurcation: GD on the double well f=x^4/4 - x^2/2, f'=x^3-x.
# The minimum x*=1 has GD multiplier 1-2*eta, crossing -1 at eta=1 -> cascade.
etas = np.linspace(0.7, 1.45, 400)
E, X = opt.bifurcation(lambda x: x**3 - x, etas, x0=0.95)
print(f"\ndouble-well bifurcation: {len(E)} attractor points over eta in [0.7, 1.45]")
low = X[np.abs(E - 0.85) < 0.01]
high = X[np.abs(E - 1.40) < 0.01]
print(f"  spread at eta=0.85: {np.ptp(low):.3f} (fixed point)")
print(f"  spread at eta=1.40: {np.ptp(high):.3f} (chaotic band)")
assert np.ptp(high) > 10 * max(np.ptp(low), 1e-6), "expected chaos at large eta"

for bad in [lambda: opt.gd_step(lambda x: x, -1)]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", e)
print("SMOKE PASS")
