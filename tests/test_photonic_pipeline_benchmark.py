"""Test the GS pipeline timing benchmark: correct arithmetic (not the wall-clock
values themselves, which are machine-dependent), input validation, and that the
scaling sweep reports sane predicted-vs-measured ratios."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import photonic_pipeline_benchmark as bench

# small/fast settings -- this test exercises the real GS pipeline, keep it cheap
N_PTS, N_ITER, N_TRIALS = 64, 5, 2

# 1. basic benchmark run: right shape of result, positive timing, correct backend label
result = bench.benchmark_gs_pipeline(n_pts=N_PTS, n_iter=N_ITER, n_trials=N_TRIALS)
assert result["n_pts"] == N_PTS and result["n_iter"] == N_ITER and result["n_trials"] == N_TRIALS
assert result["mean_s"] > 0.0
assert result["std_s"] >= 0.0
assert len(result["times"]) == N_TRIALS
assert result["backend"] == "numpy (CPU)"   # torch not installed under py-3.13 here

# 2. n_trials validation
try:
    bench.benchmark_gs_pipeline(n_pts=N_PTS, n_iter=N_ITER, n_trials=0)
    assert False, "should reject n_trials < 1"
except ValueError:
    pass

# 3. frame_rate_feasibility arithmetic is exact, independent of the real timing noise
mean_s, n_pts, rate = 0.002, 512, 1e9
feas = bench.frame_rate_feasibility(mean_s, n_pts, rate)
assert abs(feas["required_fps"] - rate / n_pts) < 1e-6
assert abs(feas["achieved_fps"] - 1.0 / mean_s) < 1e-6
assert abs(feas["headroom_factor"] - feas["achieved_fps"] / feas["required_fps"]) < 1e-9
assert feas["real_time_feasible"] == (feas["achieved_fps"] >= feas["required_fps"])

# 4. feasibility flips correctly at the boundary: slow enough should be infeasible,
#    absurdly fast should be feasible
slow = bench.frame_rate_feasibility(mean_s=1.0, n_pts=512, raw_sample_rate_hz=1e9)
assert slow["real_time_feasible"] is False
fast = bench.frame_rate_feasibility(mean_s=1e-12, n_pts=512, raw_sample_rate_hz=1e9)
assert fast["real_time_feasible"] is True

# 5. input validation on frame_rate_feasibility
for bad in [(-1.0, 512, 1e9), (0.002, -512, 1e9), (0.002, 512, -1e9), (0.0, 512, 1e9)]:
    try:
        bench.frame_rate_feasibility(*bad)
        assert False, f"should reject {bad}"
    except ValueError:
        pass

# 6. scaling sweep: right number of rows, each positive mean_s, and every row after
#    the first carries a (measured, predicted) ratio pair
n_list = (32, 64, 128)
sweep = bench.scaling_sweep(n_pts_list=n_list, n_iter=N_ITER, n_trials=N_TRIALS)
assert [row["n_pts"] for row in sweep] == list(n_list)
assert all(row["mean_s"] > 0 for row in sweep)
assert "measured_vs_predicted_scaling" not in sweep[0]
for row in sweep[1:]:
    meas, pred = row["measured_vs_predicted_scaling"]
    assert pred > 1.0   # n*log2(n) always grows with n over this range
    assert meas > 0

print("all dgs.photonic_pipeline_benchmark tests passed")
