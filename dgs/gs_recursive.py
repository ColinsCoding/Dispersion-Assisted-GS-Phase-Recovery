"""Recursive form of the time-domain Gerchberg-Saxton algorithm (TDGSA).

gs_core.retrieve_phase() is a FOLD: start with E0, apply gs_iteration n times. A fold
is exactly a tail recursion, so the same algorithm reads as

    GS(E, n) = E                     if n == 0      (base case)
             = GS(step(E), n - 1)    otherwise      (recursive / tail case)

This is mathematically identical to the loop -- it returns the same phase. The only
difference is COST, and that is the lesson: Python has no tail-call optimization, so
each recursive call keeps a live stack frame. Time is the same O(n_iter * N log N)
(the FFTs dominate), but SPACE goes from O(1) for the loop to O(n_iter) call-stack
frames for the recursion, plus a hard sys.getrecursionlimit() ceiling (~1000). So:
use this to SEE the fold/recursion equivalence on the project's core algorithm; use
gs_core.retrieve_phase in production. See tests/test_gs_recursive.py for the proof
that the two agree to ~1e-9.
"""

import numpy as np

from dgs.gs_core import (
    gs_iteration, disperse, undisperse,
    _check_dispersion, _check_n_iter, _check_intensities,
)


def _amp_error(E, I2, D2):
    """RMS error between the model's D2 intensity and the measured I2 (same metric
    gs_core.retrieve_phase records each iteration)."""
    return float(np.sqrt(np.mean((np.abs(disperse(E, D2)) ** 2 - I2) ** 2)))


def retrieve_phase_recursive(I1, I2, D1, D2, n_iter=50, unit_amplitude=True,
                             return_errors=True):
    """Recover optical phase by the TDGSA, written as a tail recursion instead of a loop.

    Same arguments, same return as gs_core.retrieve_phase:
        phi    : recovered phase phi(t) (up to a global offset)
        errors : list of per-iteration RMS amplitude errors

    Uses O(n_iter) recursion depth -- fine for the usual n_iter ~= 50, but for very
    large n_iter prefer the iterative gs_core.retrieve_phase (O(1) stack).
    """
    D1 = _check_dispersion(D1, 'D1')
    D2 = _check_dispersion(D2, 'D2')
    n_iter = _check_n_iter(n_iter)
    I1 = _check_intensities(I1, 'I1')
    I2 = _check_intensities(I2, 'I2')
    if D1 == D2:
        raise ValueError("D1 == D2: identical dispersions provide zero measurement diversity.")
    N = min(len(I1), len(I2))
    I1, I2 = I1[:N], I2[:N]

    # same initial guess as the iterative version: undisperse sqrt(I1) with zero phase
    E0 = undisperse(np.sqrt(np.maximum(I1, 0)).astype(complex), D1)
    errors = []

    def go(E, n):
        if n == 0:                      # ---- base case: nothing left to do
            return E
        E = gs_iteration(E, I1, I2, D1, D2, unit_amplitude=unit_amplitude)
        errors.append(_amp_error(E, I2, D2))
        return go(E, n - 1)             # ---- recursive (tail) case

    E_final = go(E0, n_iter)
    phi = np.angle(E_final)
    return (phi, errors) if return_errors else phi


if __name__ == "__main__":
    from dgs.gs_core import make_qpsk_measurements, retrieve_phase
    d = make_qpsk_measurements(n_symbols=64, snr_db=30.0)
    phi_i, err_i = retrieve_phase(d["I1"], d["I2"], d["D1"], d["D2"], n_iter=40)
    phi_r, err_r = retrieve_phase_recursive(d["I1"], d["I2"], d["D1"], d["D2"], n_iter=40)
    print("recursive == iterative phase :", np.allclose(phi_i, phi_r, atol=1e-9))
    print("recursive == iterative errors:", np.allclose(err_i, err_r, atol=1e-9))
    print(f"final amplitude error (both): {err_r[-1]:.3e}")
    import sys
    print(f"recursion depth used: {len(err_r)} frames  (limit {sys.getrecursionlimit()})")
