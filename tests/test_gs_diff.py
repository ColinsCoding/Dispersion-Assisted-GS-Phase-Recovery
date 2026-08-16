"""Test dgs/gs_diff.py's gs_unrolled against dgs/gs_core.py's known-good
numpy retrieve_phase on identical synthetic measurements. Locks in a real
bug fix found this session: gs_unrolled's unit_amplitude=True branch used
to skip the I1/I2 amplitude constraint entirely (just forcing |E|=1,
ignoring the measured data), giving near-random phase recovery
(corr~0.03) instead of matching numpy's reference result (corr~0.59).
Requires py -3.12 (torch is py-3.12 only in this repo, not 3.13)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from dgs.gs_core import make_measurements, retrieve_phase
from dgs.gs_diff import gs_unrolled


def _phase_corr(phi_a, phi_b):
    return float(np.corrcoef(phi_a, phi_b)[0, 1])


# 1. Regression: torch gs_unrolled must match numpy retrieve_phase closely
#    (both start from the same sqrt(I1)-undispersed initial guess and run
#    the same alternating-projection algorithm -- should agree tightly,
#    not just both be "reasonable")
for seed in [0, 1, 2]:
    m = make_measurements('QPSK', n_symbols=64, sps=8, snr_db=30,
                          D1=-5000, D2=-5750, rng_seed=seed)
    phi_np, _ = retrieve_phase(m['I1'], m['I2'], m['D1'], m['D2'], n_iter=50, unit_amplitude=True)
    corr_np = _phase_corr(phi_np, m['phi_true'])

    I1 = torch.tensor(m['I1'], dtype=torch.float32)
    I2 = torch.tensor(m['I2'], dtype=torch.float32)
    E, _ = gs_unrolled(I1, I2, -5000, -5750, n_iter=50, unit_amplitude=True)
    phi_torch = torch.angle(E).detach().numpy()
    corr_torch = _phase_corr(phi_torch, m['phi_true'])

    assert corr_np > 0.5, f"seed {seed}: numpy reference itself looks wrong, corr={corr_np:.4f}"
    assert abs(corr_torch - corr_np) < 0.05, (
        f"seed {seed}: torch gs_unrolled (corr={corr_torch:.4f}) diverges from "
        f"numpy retrieve_phase (corr={corr_np:.4f}) by more than 0.05 -- "
        "the two should track closely since they run the same algorithm")

# 2. The bug being guarded against: unit_amplitude=True must actually use
#    I1/I2, not silently ignore them -- direct check that recovered phase
#    correlates well above chance level (a broken constraint gives ~0.0-0.1).
#    seed=5 is a harder case (numpy itself only reaches ~0.37 here, matching
#    this repo's known per-seed convergence variance, [[feedback_gs_convergence]])
#    so the threshold is set above "broken" (~0.03-0.1), not above "typical".
m2 = make_measurements('QPSK', n_symbols=64, sps=8, snr_db=30, D1=-5000, D2=-5750, rng_seed=5)
I1_2 = torch.tensor(m2['I1'], dtype=torch.float32)
I2_2 = torch.tensor(m2['I2'], dtype=torch.float32)
E2, errors2 = gs_unrolled(I1_2, I2_2, -5000, -5750, n_iter=50, unit_amplitude=True)
phi2 = torch.angle(E2).detach().numpy()
corr2 = _phase_corr(phi2, m2['phi_true'])
assert corr2 > 0.25, f"expected clearly-above-chance phase recovery, got corr={corr2:.4f}"

# 3. Error trace should decrease substantially over iterations (convergence,
#    not stagnation at the initial-guess error)
assert errors2[-1] < errors2[0] * 0.5, "expected the error trace to meaningfully decrease"

# 4. unit_amplitude=False path still runs and returns finite values (not
#    regression-testing its accuracy in detail, just that it wasn't broken
#    by the fix to the True path)
E3, errors3 = gs_unrolled(I1_2, I2_2, -5000, -5750, n_iter=20, unit_amplitude=False)
assert torch.all(torch.isfinite(torch.view_as_real(E3)))
assert all(np.isfinite(e) for e in errors3)

print("all dgs.gs_diff tests passed")
