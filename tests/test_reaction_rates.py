"""Test reaction_rates: rate law evaluation, integrated rate laws match
their differential definitions, half-life formulas, order-detection via
linearization, and Arrhenius temperature dependence."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import reaction_rates as rr

# 1. general rate law: rate = k[A]^1 [B]^2
assert abs(rr.rate_law({"A": 0.5, "B": 0.2}, k=3.0, orders={"A": 1, "B": 2}) - 3.0 * 0.5 * 0.2 ** 2) < 1e-12

# 2. integrated concentration satisfies d[A]/dt = -k[A]^order (checked numerically)
A0, k = 1.0, 0.4
t = np.linspace(0, 5, 2000)
for order in (0, 1, 2):
    A = rr.integrated_concentration(A0, k, t, order)
    dA_dt = np.gradient(A, t)
    rhs = -k * A ** order
    # stay well clear of the order-0 clip-at-zero point (a real kink in A(t))
    mask = A > 0.05 * A0
    assert np.max(np.abs(dA_dt[mask] - rhs[mask])) < 5e-3

# 3. half-life: order 1 is independent of A0 (the textbook 1st-order signature)
t_half_a = rr.half_life(A0=1.0, k=k, order=1)
t_half_b = rr.half_life(A0=5.0, k=k, order=1)
assert abs(t_half_a - t_half_b) < 1e-12
# orders 0 and 2 DO depend on A0
assert rr.half_life(1.0, k, 0) != rr.half_life(5.0, k, 0)
assert rr.half_life(1.0, k, 2) != rr.half_life(5.0, k, 2)

# 4. order-detection correctly identifies the true order from clean synthetic data
for true_order in (0, 1, 2):
    A_data = rr.integrated_concentration(A0=1.0, k=0.3, t=t[:30], order=true_order)
    fit = rr.fit_reaction_order(t[:30], A_data, candidate_orders=(0, 1, 2))
    assert fit["order"] == true_order

# 5. Arrhenius rate constant increases with temperature
k_low = rr.arrhenius_rate_constant(1e13, Ea_eV=1.0, T=300)
k_high = rr.arrhenius_rate_constant(1e13, Ea_eV=1.0, T=600)
assert k_high > k_low

print("test_reaction_rates: all checks passed")
