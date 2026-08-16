"""Test dgs/trajectory_viewer.py's make_gs_convergence_viewer -- structural
checks on the returned Plotly figure (frame count, per-frame data shapes,
amplitude-based marker sizing, axis ranges) since rendering itself can't be
verified headlessly. Requires py -3.12 (plotly is not installed under this
repo's py -3.13 environment)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.gs_core import make_measurements, retrieve_phase_with_history
from dgs.trajectory_viewer import make_gs_convergence_viewer

m = make_measurements('QPSK', n_symbols=32, sps=8, D1=-5000, D2=-5750, snr_db=30, rng_seed=1)
phi, errors, E_history = retrieve_phase_with_history(
    m['I1'], m['I2'], m['D1'], m['D2'], n_iter=30, unit_amplitude=True)
N = len(m['I1'])

# 1. Frame count matches n_iter+1 (initial guess + one per iteration)
fig = make_gs_convergence_viewer(E_history, t=m['t'], phi_true=m['phi_true'])
assert len(fig.frames) == 31

# 2. Each frame's Scatter3d data has N points, correct x/y/z mapping
for k in [0, 15, 30]:
    trace = fig.frames[k].data[0]
    assert len(trace.x) == N
    assert np.allclose(trace.x, m['t'])
    assert np.allclose(trace.y, np.real(E_history[k]))
    assert np.allclose(trace.z, np.imag(E_history[k]))

# 3. Marker size tracks |E(t)| (bigger amplitude -> bigger marker), and
#    varies frame to frame (unlike the original molecular-viewer's fixed
#    per-atom sizing) since amplitude actually changes during convergence
sizes_first = np.array(fig.frames[0].data[0].marker.size)
sizes_last = np.array(fig.frames[-1].data[0].marker.size)
assert not np.allclose(sizes_first, sizes_last), "marker sizes should change across iterations"
amp_last = np.abs(E_history[-1])
order = np.argsort(amp_last)
assert np.all(np.diff(sizes_last[order]) >= -1e-9), "marker size should be non-decreasing in |E|"

# 4. Ground-truth overlay present when phi_true is given (as a static extra
#    trace on top of the per-frame E(t) estimate trace)
assert len(fig.frames[0].data) == 2   # [E(t) estimate, true overlay]
true_trace = fig.frames[0].data[1]
E_true = np.exp(1j * m['phi_true'])
assert np.allclose(true_trace.y, np.real(E_true))
assert np.allclose(true_trace.z, np.imag(E_true))

# 5. No ground-truth overlay when phi_true is omitted
fig_no_truth = make_gs_convergence_viewer(E_history, t=m['t'])
assert len(fig_no_truth.frames[0].data) == 1

# 6. Axis ranges actually bound the data (not left at Plotly's defaults)
y_range = fig.layout.scene.yaxis.range
z_range = fig.layout.scene.zaxis.range
assert y_range[0] <= np.real(E_history).min() and y_range[1] >= np.real(E_history).max()
assert z_range[0] <= np.imag(E_history).min() and z_range[1] >= np.imag(E_history).max()

# 7. Input validation
for bad_call in [
    lambda: make_gs_convergence_viewer(np.zeros((5,)), t=None),                 # not 2D
    lambda: make_gs_convergence_viewer(np.zeros((0, 5)), t=None),               # no frames
    lambda: make_gs_convergence_viewer(E_history, t=np.arange(N - 1)),          # wrong t length
    lambda: make_gs_convergence_viewer(E_history, t=m['t'], phi_true=np.zeros(N - 1)),  # wrong phi_true length
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.trajectory_viewer tests passed")
