"""Test dgs/gs_core.py's retrieve_phase_with_history: must match
retrieve_phase's own phi/errors exactly (same algorithm, just also
capturing per-iteration E snapshots for dgs/trajectory_viewer.py)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.gs_core import make_measurements, retrieve_phase, retrieve_phase_with_history

m = make_measurements('QPSK', n_symbols=32, sps=8, D1=-5000, D2=-5750, snr_db=30, rng_seed=1)

phi_ref, errors_ref = retrieve_phase(m['I1'], m['I2'], m['D1'], m['D2'], n_iter=30, unit_amplitude=True)
phi_hist, errors_hist, E_history = retrieve_phase_with_history(
    m['I1'], m['I2'], m['D1'], m['D2'], n_iter=30, unit_amplitude=True)

# 1. Identical algorithm -> identical phi and errors (both use the same
#    rng-free, deterministic sqrt(I1)-undispersed initial guess)
assert np.allclose(phi_ref, phi_hist), "retrieve_phase_with_history's phi diverged from retrieve_phase"
assert errors_ref == errors_hist, "retrieve_phase_with_history's errors diverged from retrieve_phase"

# 2. E_history shape: n_iter+1 rows (initial guess + one per iteration), N columns
N = len(m['I1'])
assert E_history.shape == (31, N)

# 3. The final row's phase matches the returned phi exactly
assert np.allclose(np.angle(E_history[-1]), phi_hist)

# 4. The first row is the pre-iteration initial guess: sqrt(I1) undispersed
#    through D1 -- should NOT already match phi_true well (otherwise the
#    "history" wouldn't show any actual convergence)
from dgs.gs_core import undisperse
E0_expected = undisperse(np.sqrt(np.maximum(m['I1'], 0)).astype(complex), m['D1'])
assert np.allclose(E_history[0], E0_expected)

# 5. Convergence: correlation with true phase should IMPROVE from first to
#    last frame (the whole point of having history to visualize)
corr_first = float(np.corrcoef(np.angle(E_history[0]), m['phi_true'])[0, 1])
corr_last = float(np.corrcoef(np.angle(E_history[-1]), m['phi_true'])[0, 1])
assert corr_last > corr_first, (
    f"expected recovered phase to improve over iterations, got corr_first={corr_first:.4f} "
    f"-> corr_last={corr_last:.4f}")

# 6. Input validation matches retrieve_phase's (same checks, same errors)
for bad_call in [
    lambda: retrieve_phase_with_history(m['I1'], m['I2'], 0, m['D2']),
    lambda: retrieve_phase_with_history(m['I1'], m['I2'], m['D1'], m['D1']),
    lambda: retrieve_phase_with_history(m['I1'], m['I2'], m['D1'], m['D2'], n_iter=0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.gs_core retrieve_phase_with_history tests passed")
