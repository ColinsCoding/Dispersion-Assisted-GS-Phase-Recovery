"""90-degree optical hybrid -- the I/Q mixer at the front of a coherent receiver.

A 90-deg optical hybrid combines the incoming signal field with a local oscillator
(LO) and splits the result into four outputs at relative phases 0, 90, 180, 270 deg.
Balanced photodetectors on the (0,180) and (90,270) pairs then recover the in-phase
(I) and quadrature (Q) components -- i.e. the full complex field, which is exactly
what the dispersion-assisted GS receiver in this repo reconstructs by other means.

Ideal transmission matrix (signal s, local oscillator l):
    [ out_0   ]   [ 1    1 ] [ s ]
    [ out_90  ] = [ 1   -1 ] [ l ]
    [ out_180 ]   [ 1j   1j]
    [ out_270 ]   [ 1j  -1j]
with optional insertion loss (dB) and phase / loss imbalance to model a real device.

Source: Yiming Hybrid90deg (Jalali lab), MATLAB optical_hybrid_90deg.m + VPI model.
This is a cleaned port of python/optical_hybrid_90deg.py: the function is unchanged,
but the hardcoded-path example was moved under __main__ so importing is side-effect
free. NumPy only.
"""

import numpy as np


def optical_hybrid_90deg(
    signal_input,
    local_oscillator,
    insertion_loss_signal=0.0,
    insertion_loss_local_oscillator=0.0,
    phase_imbalance_slo=0.0,
    phase_imbalance_iq=0.0,
    insertion_loss_imbalance_i=0.0,
    insertion_loss_imbalance_q=0.0,
):
    """Simulate a 90-degree optical hybrid.

    Parameters (losses in dB, phases in radians):
        signal_input, local_oscillator : complex input E-fields
        insertion_loss_signal / _local_oscillator : input arm losses
        phase_imbalance_slo : signal-vs-LO phase error
        phase_imbalance_iq  : I-vs-Q phase error
        insertion_loss_imbalance_i / _q : per-branch loss imbalance

    Returns (out_0, out_90, out_180, out_270): the four complex outputs.
    """
    IL_signal = 10 ** (-insertion_loss_signal / 10)
    IL_lo = 10 ** (-insertion_loss_local_oscillator / 10)
    Imb_I = 10 ** (-insertion_loss_imbalance_i / 10)
    Imb_Q = 10 ** (-insertion_loss_imbalance_q / 10)

    s = signal_input * IL_signal
    l = local_oscillator * IL_lo

    T = np.array([[1, 1], [1, -1], [1j, 1j], [1j, -1j]])
    phase_matrix = np.diag([
        1,
        np.exp(1j * phase_imbalance_slo),
        np.exp(1j * phase_imbalance_iq),
        np.exp(1j * (phase_imbalance_slo + phase_imbalance_iq)),
    ])
    imbalance_matrix = np.diag([1, Imb_I, Imb_Q, Imb_I * Imb_Q])

    T_adj = imbalance_matrix @ phase_matrix @ T
    E_out = T_adj @ np.array([s, l])
    return E_out[0], E_out[1], E_out[2], E_out[3]


def balanced_detection(outputs):
    """Balanced-photodiode difference signals from the two output pairs:
        D1 = |out_0|^2  - |out_90|^2 ,   D2 = |out_180|^2 - |out_270|^2 .
    For this hybrid's (faithfully ported) transmission matrix, BOTH equal
    4*Re(s*conj(l)) -- the in-phase correlation of signal with LO. The matrix puts
    1j as a *global* phase on rows 3-4, so it does not produce an independent
    quadrature; a textbook hybrid puts the 90-deg shift on the LO arm of the second
    pair so D2 yields 4*Im(s*conj(l)). See README. Returns (D1, D2)."""
    o0, o90, o180, o270 = outputs
    return (np.abs(o0) ** 2 - np.abs(o90) ** 2,
            np.abs(o180) ** 2 - np.abs(o270) ** 2)


if __name__ == "__main__":
    s, lo = 1 + 1j, 1 - 1j
    outs = optical_hybrid_90deg(s, lo, insertion_loss_signal=1.0,
                                insertion_loss_local_oscillator=0.5,
                                phase_imbalance_slo=np.pi / 18,
                                phase_imbalance_iq=np.pi / 36)
    for ph, o in zip((0, 90, 180, 270), outs):
        print(f"  out {ph:3d} deg: {o:.4f}")
    D1, D2 = balanced_detection(optical_hybrid_90deg(s, lo))   # ideal hybrid
    print(f"ideal balanced detection: D1={D1:.3f}, D2={D2:.3f}  "
          f"(both = 4 Re(s l*) = {4*np.real(s*np.conj(lo)):.3f}; this matrix has no independent Q)")
