"""Test dgs.torch.gnss_batched: the batched torch Gauss-Newton solve agrees
with dgs.gnss_positioning's existing NumPy per-problem solver on identical
data, and Monte-Carlo position-error RMS matches the closed-form DOP
prediction PDOP*sigma_rho (theory vs. simulation, not just self-consistency)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.gnss_positioning import pseudorange, solve_position, _sats_from_directions, C_LIGHT
from dgs.torch.gnss_batched import (
    batched_solve_position, monte_carlo_positioning_uncertainty, benchmark_batched_vs_loop,
    benchmark_repeated, scaling_sweep,
)

receiver_true = np.array([6.371e6, 0.0, 0.0])
bias_true = 100e-6 * C_LIGHT
dirs = np.array([[1, 0, 0], [0.5, 0.8, 0.3], [0.5, -0.8, 0.3],
                 [0.4, 0.3, -0.85], [0.6, -0.2, 0.75], [0.2, 0.6, -0.77]])
sats = _sats_from_directions(receiver_true, dirs, 26.56e6)

# 1. batched solve on a SINGLE noiseless problem must recover the true position,
# matching the existing NumPy solver's own noiseless recovery
rho_clean = np.array([pseudorange(receiver_true, s, bias_true) for s in sats])
single = batched_solve_position(sats, rho_clean[None, :])
assert np.max(np.abs(single["position"][0] - receiver_true)) < 1e-3   # mm-level
assert abs(single["clock_bias"][0] - bias_true) < 1e-3

# 2. batched solve on MANY noisy problems must agree with looping the existing
# NumPy solver over the same data (same math, different execution shape)
rng = np.random.default_rng(7)
n = 40
noise = rng.normal(0.0, 5.0, size=(n, len(sats)))
rho_batch = rho_clean[None, :] + noise
batched = batched_solve_position(sats, rho_batch)
looped = np.array([solve_position(sats, rho_batch[i])["position"] for i in range(n)])
assert np.max(np.abs(batched["position"] - looped)) < 1e-6

# 3. Monte Carlo empirical RMS position error must match the closed-form
# DOP prediction (PDOP * sigma_rho) to within Monte Carlo sampling noise
mc = monte_carlo_positioning_uncertainty(sats, receiver_true, bias_true, sigma_rho=5.0,
                                          n_realizations=4000, seed=0)
ratio = mc["empirical_rms_m"] / mc["predicted_rms_m"]
assert 0.9 < ratio < 1.1, f"empirical/predicted RMS ratio {ratio} not close to 1.0"

# 4. more measurement noise -> more position uncertainty, monotonically
mc_lo = monte_carlo_positioning_uncertainty(sats, receiver_true, bias_true, sigma_rho=1.0,
                                             n_realizations=1000, seed=1)
mc_hi = monte_carlo_positioning_uncertainty(sats, receiver_true, bias_true, sigma_rho=10.0,
                                             n_realizations=1000, seed=1)
assert mc_hi["empirical_rms_m"] > mc_lo["empirical_rms_m"]

# 5. the benchmark's batched result must numerically match its own loop baseline
# (the timing numbers vary by machine; the AGREEMENT must not)
bench = benchmark_batched_vs_loop(sats, receiver_true, bias_true, sigma_rho=5.0, n_realizations=30)
assert bench["max_position_disagreement_m"] < 1e-6
assert bench["device_used"] in ("cpu", "cuda")

# 6. kwarg bounds: fewer than 4 satellites must raise, same as the NumPy solver
try:
    batched_solve_position(sats[:3], rho_clean[None, :3])
    assert False
except ValueError:
    pass

# 7. benchmark_repeated: mean/std over repeats must still agree with the NumPy solver,
# and n_repeats < 1 must be rejected
rep = benchmark_repeated(sats, receiver_true, bias_true, sigma_rho=5.0, n_realizations=20, n_repeats=3)
assert rep["n_repeats"] == 3
assert rep["max_disagreement_m"] < 1e-6
assert rep["loop_mean_s"] > 0 and rep["batched_mean_s"] > 0
try:
    benchmark_repeated(sats, receiver_true, bias_true, sigma_rho=5.0, n_realizations=20, n_repeats=0)
    assert False
except ValueError:
    pass

# 8. scaling_sweep: one row per batch size, each internally consistent with benchmark_repeated,
# and disagreement with the NumPy solver stays negligible across every size swept
sweep = scaling_sweep(sats, receiver_true, bias_true, sigma_rho=5.0,
                       batch_sizes=(10, 30, 60), n_repeats=2)
assert len(sweep) == 3
assert [r["n_realizations"] for r in sweep] == [10, 30, 60]
assert all(r["max_disagreement_m"] < 1e-6 for r in sweep)
assert "speedup_grew" not in sweep[0]          # no prior row to compare the first one against
assert "speedup_grew" in sweep[1] and "speedup_grew" in sweep[2]

print("test_gnss_batched: all checks passed")
