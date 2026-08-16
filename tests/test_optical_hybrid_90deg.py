"""Test dgs/optical_hybrid_90deg.py: a faithful port of the original
MATLAB/Python optical_hybrid_90deg (YimingMLX90Deg, MIT license), the
energy-conserving normalized version, the physically-corrected version, and
I/Q balanced-detection recovery -- including a regression test locking in
the original matrix's degenerate (always-zero) Q output as a documented,
intentionally-preserved bug rather than something to silently fix."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import optical_hybrid_90deg as hyb

# 1. Regression: matches the exact numeric output of the original algorithm
#    (independently reimplemented here, not imported, to catch drift)
def _original(signal_input, local_oscillator, insertion_loss_signal=0.0,
              insertion_loss_local_oscillator=0.0, phase_imbalance_slo=0.0,
              phase_imbalance_iq=0.0, insertion_loss_imbalance_i=0.0,
              insertion_loss_imbalance_q=0.0):
    IL_signal = 10 ** (-insertion_loss_signal / 10)
    IL_lo = 10 ** (-insertion_loss_local_oscillator / 10)
    Imb_I = 10 ** (-insertion_loss_imbalance_i / 10)
    Imb_Q = 10 ** (-insertion_loss_imbalance_q / 10)
    s = signal_input * IL_signal
    lo = local_oscillator * IL_lo
    T = np.array([[1, 1], [1, -1], [1j, 1j], [1j, -1j]])
    phase_matrix = np.diag([1, np.exp(1j * phase_imbalance_slo), np.exp(1j * phase_imbalance_iq),
                             np.exp(1j * (phase_imbalance_slo + phase_imbalance_iq))])
    imbalance_matrix = np.diag([1, Imb_I, Imb_Q, Imb_I * Imb_Q])
    T_adjusted = imbalance_matrix @ phase_matrix @ T
    E_out = T_adjusted @ np.array([s, lo])
    return E_out[0], E_out[1], E_out[2], E_out[3]

kwargs = dict(insertion_loss_signal=1.0, insertion_loss_local_oscillator=0.5,
              phase_imbalance_slo=np.pi / 18, phase_imbalance_iq=np.pi / 36,
              insertion_loss_imbalance_i=0.2, insertion_loss_imbalance_q=0.1)
expected = _original(1 + 1j, 1 - 1j, **kwargs)
actual = hyb.optical_hybrid_90deg(1 + 1j, 1 - 1j, **kwargs)
for e, a in zip(expected, actual):
    assert abs(e - a) < 1e-12

# 2. Ideal case (no loss/imbalance): row order in the transmission matrix T
#    is [sum, difference, j*sum, j*difference] -> (output0, output90,
#    output180, output270) = (E_s+E_lo, E_s-E_lo, j*(E_s+E_lo), j*(E_s-E_lo))
#    -- i.e. output90/output270 carry the SAME sum/difference pair as
#    output0/output180, rotated by 90 degrees, not a separate I/Q pairing.
o0, o90, o180, o270 = hyb.optical_hybrid_90deg(1 + 0j, 1 + 0j)
assert abs(o0 - 2.0) < 1e-12          # E_s + E_lo = 1+1 = 2
assert abs(o90 - 0.0) < 1e-12         # E_s - E_lo = 1-1 = 0
assert abs(o180 - 2j) < 1e-12         # j*(E_s + E_lo) = 2j
assert abs(o270 - 0.0) < 1e-12        # j*(E_s - E_lo) = 0

# 3. Normalized version is exactly half the unnormalized one (the
#    documented missing-1/2-factor discrepancy from the original code)
outs_orig = hyb.optical_hybrid_90deg(1 + 1j, 1 - 1j, **kwargs)
outs_norm = hyb.optical_hybrid_90deg_normalized(1 + 1j, 1 - 1j, **kwargs)
for o, n in zip(outs_orig, outs_norm):
    assert abs(n - o / 2.0) < 1e-12

# 4. Energy conservation: for the LOSSLESS, BALANCED normalized hybrid
#    (all loss/imbalance kwargs at default), sum(|output|^2) over all 4
#    ports equals |E_s|^2 + |E_lo|^2 -- the actual physical correctness
#    check that motivated splitting out the normalized version
E_s, E_lo = 1 + 1j, 0.5 - 0.3j
outs_ideal_norm = hyb.optical_hybrid_90deg_normalized(E_s, E_lo)
total_out_power = sum(abs(o) ** 2 for o in outs_ideal_norm)
total_in_power = abs(E_s) ** 2 + abs(E_lo) ** 2
assert abs(total_out_power - total_in_power) < 1e-10

# 5. The UNNORMALIZED (original) version does NOT conserve energy this way
#    -- documenting the discrepancy numerically, not just in prose
outs_ideal_orig = hyb.optical_hybrid_90deg(E_s, E_lo)
total_out_power_orig = sum(abs(o) ** 2 for o in outs_ideal_orig)
assert abs(total_out_power_orig - 4 * total_in_power) < 1e-10   # 2x fields -> 4x power

# 6. iq_from_hybrid_outputs on the CORRECTED matrix recovers
#    Re/Im{E_s*conj(E_lo)} exactly, for the ideal lossless hybrid
outs_ideal_corrected = hyb.optical_hybrid_90deg_corrected(E_s, E_lo)
I, Q = hyb.iq_from_hybrid_outputs(*outs_ideal_corrected)
expected_I = np.real(E_s * np.conj(E_lo))
expected_Q = np.imag(E_s * np.conj(E_lo))
assert abs(I - expected_I) < 1e-10
assert abs(Q - expected_Q) < 1e-10

# 6b. The ORIGINAL (bug-preserving) matrix's Q is degenerate (~0) regardless
#     of the actual phase relationship between E_s and E_lo -- issue 2 from
#     the module docstring, locked in as a regression test so it's not
#     silently "fixed" by accident in optical_hybrid_90deg() itself, which
#     must keep faithfully reproducing the original source.
for E_s_test, E_lo_test in [(1 + 1j, 0.5 - 0.3j), (2 - 0.1j, -0.7 + 0.9j), (1j, 1.0)]:
    outs_buggy = hyb.optical_hybrid_90deg_normalized(E_s_test, E_lo_test)
    _, Q_buggy = hyb.iq_from_hybrid_outputs(*outs_buggy)
    assert abs(Q_buggy) < 1e-10, f"expected degenerate Q~0 from the original matrix, got {Q_buggy}"

# 7. Input validation (both the original port and the corrected version)
for bad_call in [
    lambda: hyb.optical_hybrid_90deg(1 + 1j, 1 + 1j, insertion_loss_signal=-1.0),
    lambda: hyb.optical_hybrid_90deg(1 + 1j, 1 + 1j, insertion_loss_imbalance_i=-0.1),
    lambda: hyb.optical_hybrid_90deg_corrected(1 + 1j, 1 + 1j, insertion_loss_signal=-1.0),
    lambda: hyb.optical_hybrid_90deg_corrected(1 + 1j, 1 + 1j, insertion_loss_imbalance_q=-0.1),
    lambda: hyb.iq_from_hybrid_outputs(1, 1, 1, 1, responsivity=0.0),
    lambda: hyb.iq_from_hybrid_outputs(1, 1, 1, 1, responsivity=-1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.optical_hybrid_90deg tests passed")
