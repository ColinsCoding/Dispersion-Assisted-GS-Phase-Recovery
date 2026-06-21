"""Test error propagation: linear formula vs closed forms vs Monte Carlo."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import error_propagation as ep

# 1. SUM: f = x + y -> sigmas add in quadrature
f_sum = lambda p: p[0] + p[1]
_, s = ep.propagate(f_sum, [3.0, 4.0], [0.3, 0.4])
assert abs(s - ep.add_in_quadrature(0.3, 0.4)) < 1e-6          # sqrt(.09+.16)=0.5
assert abs(s - 0.5) < 1e-6

# 2. PRODUCT: f = x*y -> RELATIVE sigmas add in quadrature
f_prod = lambda p: p[0] * p[1]
val, s = ep.propagate(f_prod, [2.0, 5.0], [0.1, 0.2])
closed = ep.product_rule(val, [(2.0, 0.1), (5.0, 0.2)])
assert abs(s - closed) < 1e-4, (s, closed)
# and Monte Carlo agrees (product is mildly nonlinear -> loose tol)
_, s_mc = ep.propagate_mc(f_prod, [2.0, 5.0], [0.1, 0.2], n=300_000, seed=1)
assert abs(s - s_mc) / s < 0.05

# 3. POWER: f = x^3 -> relative error x3
f_pow = lambda p: p[0] ** 3
val, s = ep.propagate(f_pow, [2.0], [0.05])
assert abs(s - ep.power_rule(val, 2.0, 0.05, 3)) < 1e-3        # 3 * 0.05/2 * 8 = 0.6

# 4. CORRELATION matters: positively-correlated inputs in a DIFFERENCE reduce sigma
f_diff = lambda p: p[0] - p[1]
indep = ep.propagate(f_diff, [10.0, 9.0], [0.5, 0.5])[1]
cov = [[0.25, 0.20], [0.20, 0.25]]                            # rho = 0.8
corr = ep.propagate(f_diff, [10.0, 9.0], cov=cov)[1]
assert corr < indep, (corr, indep)        # shared error cancels in a difference
# cross-check the correlated case against Monte Carlo
_, corr_mc = ep.propagate_mc(f_diff, [10.0, 9.0], cov=cov, n=300_000, seed=2)
assert abs(corr - corr_mc) / corr < 0.05

# 5. the flux-rule emf = B h v (Griffiths 7.13): three independent relative errors
B, h, v = 0.5, 2.0, 3.0
emf = lambda p: p[0] * p[1] * p[2]
val, s = ep.propagate(emf, [B, h, v], [0.01, 0.05, 0.1])
closed = ep.product_rule(val, [(B, 0.01), (h, 0.05), (v, 0.1)])
_, s_mc = ep.propagate_mc(emf, [B, h, v], [0.01, 0.05, 0.1], n=300_000, seed=3)
assert abs(s - closed) < 1e-4 and abs(s - s_mc) / s < 0.05
assert abs(val - 3.0) < 1e-9

# 6. the OOP Measurement class agrees with propagate() for the emf product
M = ep.Measurement
emf_m = M(B, 0.01) * M(h, 0.05) * M(v, 0.1)
assert abs(emf_m.value - 3.0) < 1e-9
assert abs(emf_m.sigma - s) < 1e-3, (emf_m.sigma, s)         # same sigma as propagate()
# sum in quadrature via the class
assert abs((M(3.0, 0.3) + M(4.0, 0.4)).sigma - 0.5) < 1e-9
# scalar operands are treated as exact (sigma 0)
assert abs((M(5.0, 0.2) * 2).value - 10.0) < 1e-9
assert abs((M(5.0, 0.2) * 2).sigma - 0.4) < 1e-9
# power: f = x^3 relative error x3
assert abs((M(2.0, 0.05) ** 3).sigma - ep.power_rule(8.0, 2.0, 0.05, 3)) < 1e-9

print(f"TEST PASS  (sum quad=0.5; product rel-quad matches; power x3; "
      f"correlation shrinks a difference {corr:.3f}<{indep:.3f}; "
      f"emf=3.00+/-{s:.2f} V linear==closed==MC)")
