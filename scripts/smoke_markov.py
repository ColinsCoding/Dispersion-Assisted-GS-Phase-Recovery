"""Smoke-test Markov chains: stochasticity, evolution, stationary distribution, hitting time."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import markov as mk

P = np.array([[0.9, 0.1], [0.5, 0.5]])

# 1. stochasticity check
assert mk.is_stochastic(P)
assert not mk.is_stochastic([[0.9, 0.2], [0.5, 0.5]])      # row doesn't sum to 1
assert not mk.is_stochastic([[1.2, -0.2], [0.5, 0.5]])     # negative entry

# 2. a single step preserves total probability
d = mk.step([1.0, 0.0], P)
assert abs(d.sum() - 1.0) < 1e-12 and np.all(d >= 0)

# 3. stationary distribution: pi P = pi, matches analytic [5/6, 1/6]
pi = mk.stationary_distribution(P)
assert np.allclose(pi @ P, pi, atol=1e-9)                  # fixed point
assert np.allclose(pi, [5/6, 1/6], atol=1e-9)

# 4. the chain FORGETS its start: any initial distribution -> pi after many steps
for start in ([1, 0], [0, 1], [0.3, 0.7]):
    assert np.allclose(mk.evolve(start, P, 200), pi, atol=1e-8), start

# 5. a 3-state ring with uniform stationary distribution
R = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]) * 0.5 + \
    np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
assert mk.is_stochastic(R)
assert np.allclose(mk.stationary_distribution(R), [1/3, 1/3, 1/3], atol=1e-9)

# 6. hitting time: expected steps to reach the other state in the 2-state chain
H = mk.hitting_time_matrix(P)
assert np.all(np.diag(H) == 0)
# from sunny(0) to rainy(1): 1/P[0,1] = 10 steps expected (geometric)
assert abs(H[0, 1] - 10.0) < 1e-9
assert abs(H[1, 0] - 2.0) < 1e-9                           # rainy->sunny: 1/0.5 = 2

# 7. validation
for bad in (lambda: mk.step([0.5, 0.5], [[0.9, 0.2], [0.5, 0.5]]),
            lambda: mk.step([0.6, 0.6], P)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("should reject bad input")

print(f"SMOKE PASS  (stationary pi={np.round(pi,4).tolist()}; chain forgets its start; "
      f"hitting time sunny->rainy = {H[0,1]:.0f} steps)")
