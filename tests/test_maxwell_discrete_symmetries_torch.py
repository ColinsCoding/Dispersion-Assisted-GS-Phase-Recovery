"""Test dgs/maxwell_discrete_symmetries_torch.py's batched/GPU numeric
verification of the parity and time-reversal derivations in
dgs/maxwell_discrete_symmetries.py. Requires py -3.12 (torch is py-3.12
only in this repo, not 3.13)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.maxwell_discrete_symmetries_torch import (
    coulomb_parity_check_torch, biot_savart_parity_check_torch,
    time_reversal_velocity_check_torch, time_reversal_acceleration_check_torch,
)

# 1. E is polar (Coulomb's law), confirmed across a large random batch
r1 = coulomb_parity_check_torch(n_batch=10_000, seed=0)
assert r1["parity_confirmed"] is True
assert r1["max_err"] < 1e-9, f"expected near-exact match, got max_err={r1['max_err']}"

# 2. B is axial (Biot-Savart's law), confirmed across a large random batch
r2 = biot_savart_parity_check_torch(n_batch=10_000, seed=0)
assert r2["parity_confirmed"] is True
assert r2["max_err"] < 1e-9, f"expected near-exact match, got max_err={r2['max_err']}"

# 3. Velocity is T-odd, confirmed across a batch of random trajectories via
#    torch.func vmap+grad (the correct batched-autograd approach -- a naive
#    sum-then-backward trick would silently collapse independent per-sample
#    gradients into their sum, since t0 is a shared scalar, not batched)
r3 = time_reversal_velocity_check_torch(n_batch=200, seed=0)
assert r3["velocity_T_odd_confirmed"] is True
assert r3["max_err"] < 1e-9, f"expected near-exact match, got max_err={r3['max_err']}"

# 4. Acceleration is T-even, same batched-autograd approach
r4 = time_reversal_acceleration_check_torch(n_batch=200, seed=0)
assert r4["acceleration_T_even_confirmed"] is True
assert r4["max_err"] < 1e-9, f"expected near-exact match, got max_err={r4['max_err']}"

# 5. Bounds: n_batch < 1 must raise for all four functions
for fn in [coulomb_parity_check_torch, biot_savart_parity_check_torch,
           time_reversal_velocity_check_torch, time_reversal_acceleration_check_torch]:
    try:
        fn(n_batch=0)
        raise AssertionError(f"{fn.__name__}: expected ValueError for n_batch=0")
    except ValueError:
        pass

# 6. Reproducibility: same seed must give identical results (deterministic batch)
r1_again = coulomb_parity_check_torch(n_batch=10_000, seed=0)
assert r1["max_err"] == r1_again["max_err"]

print("all dgs.maxwell_discrete_symmetries_torch tests passed")
