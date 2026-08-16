"""90-degree optical hybrid: the CLASSICAL coherent-receiver front-end that
this repo's LO-free TD-GS approach (dgs/gs_core.py, dgs/sbir_portfolio.py P7
"Photonic AI Receiver") is explicitly an alternative to.

PROVENANCE: ported from D:\\Spring2026\\MATLAB\\YimingMLX90Deg\\matlab\\
optical_hybrid_90deg.m (MIT License, copyright SpectralBinaryEight 2025),
work from the user's UCLA ECE 279AS course (Prof. Bahram Jalali) -- same
lineage as [[project_jalali_ucla]]. Faithfully reproduces the original
MATLAB/Python behavior (including the normalization issue documented below)
rather than silently "fixing" it, since the point is an honest port.

WHAT A 90-DEGREE HYBRID DOES: mixes a weak signal field E_s with a strong
local-oscillator field E_LO to produce FOUR outputs at 0/90/180/270 degrees
relative phase. Balanced photodetection on the 0/180 pair recovers the I
(in-phase) quadrature; the 90/270 pair recovers Q. This IS how conventional
coherent optical receivers (homodyne/heterodyne) get I/Q without needing GS
phase retrieval -- but it REQUIRES a phase-locked LO laser, the exact cost
dgs/sbir_portfolio.py P2/P7 cite time-domain GS as removing.

TWO SEPARATE ISSUES FOUND IN THE ORIGINAL CODE (the "questionable" part):

1. Missing normalization: the standard ideal 90-degree hybrid transfer
   matrix (two nested 3-dB/50-50 couplers + a 90-degree phase shift applied
   to the LO arm; see the notebook's derivation) is
       T_ideal = (1/2) * [[1, 1], [1, -1], [1, 1j], [1, -1j]]
   energy-conserving: sum(|output|^2) over all 4 ports = |E_s|^2 + |E_LO|^2
   for a lossless device. The original MATLAB/Python code omits the 1/2,
   so its outputs are exactly 2x the physically normalized fields.

2. A more substantive physics bug: the original code's transmission matrix
   is [[1,1],[1,-1],[1j,1j],[1j,-1j]] -- for the 3rd/4th rows (the 90/270
   degree ports), BOTH input coefficients get multiplied by j, i.e. the
   WHOLE combined field is globally phase-rotated. A real 90-degree hybrid
   instead applies the 90-degree shift to only the LOCAL OSCILLATOR arm
   before recombining: [[1,1],[1,-1],[1,1j],[1,-1j]] (only the second
   column -- the E_LO coefficient -- picks up j). Multiplying an ENTIRE
   row by j is a global phase rotation that does not change |output|^2, so
   the original code's output90/output270 carry EXACTLY the same magnitude
   information as output0/output180 (see test #2 in
   tests/test_optical_hybrid_90deg.py) -- balanced photodetection on those
   two pairs can only ever recover a Q quadrature of zero, regardless of
   the actual phase relationship between signal and LO. This is a real
   functional bug, not just a normalization convention, confirmed
   numerically below (optical_hybrid_90deg() vs
   optical_hybrid_90deg_corrected()).

optical_hybrid_90deg() below faithfully reproduces the ORIGINAL (matches
the MATLAB/Python source numerically, bug included -- the point of a port
is to reproduce what was actually run). optical_hybrid_90deg_corrected()
implements the physically standard matrix instead, and is what
iq_from_hybrid_outputs() actually needs to recover a meaningful Q.
"""
import numpy as np


def _check_prob_param(value, name):
    if value < 0:
        raise ValueError(f"{name} must be non-negative (insertion loss / imbalance in dB)")


def optical_hybrid_90deg(signal_input, local_oscillator,
                         insertion_loss_signal=0.0,
                         insertion_loss_local_oscillator=0.0,
                         phase_imbalance_slo=0.0,
                         phase_imbalance_iq=0.0,
                         insertion_loss_imbalance_i=0.0,
                         insertion_loss_imbalance_q=0.0):
    """Faithful port of the original MATLAB optical_hybrid_90deg.m --
    reproduces its outputs exactly, INCLUDING the missing 1/2 normalization
    (see module docstring). Use optical_hybrid_90deg_normalized() for the
    energy-conserving version.

    Parameters
    ----------
    signal_input, local_oscillator : complex -- input optical E-fields
    insertion_loss_signal, insertion_loss_local_oscillator : float, dB >= 0
    phase_imbalance_slo, phase_imbalance_iq : float, radians
    insertion_loss_imbalance_i, insertion_loss_imbalance_q : float, dB >= 0

    Returns
    -------
    (output0, output90, output180, output270) : complex tuple
    """
    for name, val in [
        ("insertion_loss_signal", insertion_loss_signal),
        ("insertion_loss_local_oscillator", insertion_loss_local_oscillator),
        ("insertion_loss_imbalance_i", insertion_loss_imbalance_i),
        ("insertion_loss_imbalance_q", insertion_loss_imbalance_q),
    ]:
        _check_prob_param(val, name)

    IL_signal = 10 ** (-insertion_loss_signal / 10)
    IL_lo = 10 ** (-insertion_loss_local_oscillator / 10)
    Imb_I = 10 ** (-insertion_loss_imbalance_i / 10)
    Imb_Q = 10 ** (-insertion_loss_imbalance_q / 10)

    E_s = signal_input * IL_signal
    E_lo = local_oscillator * IL_lo

    T = np.array([[1, 1], [1, -1], [1j, 1j], [1j, -1j]], dtype=complex)   # unnormalized, matches original
    phase_matrix = np.diag([
        1,
        np.exp(1j * phase_imbalance_slo),
        np.exp(1j * phase_imbalance_iq),
        np.exp(1j * (phase_imbalance_slo + phase_imbalance_iq)),
    ])
    imbalance_matrix = np.diag([1, Imb_I, Imb_Q, Imb_I * Imb_Q])
    T_adjusted = imbalance_matrix @ phase_matrix @ T

    E_out = T_adjusted @ np.array([E_s, E_lo], dtype=complex)
    return E_out[0], E_out[1], E_out[2], E_out[3]


def optical_hybrid_90deg_normalized(signal_input, local_oscillator, **kwargs):
    """Energy-conserving version of the ORIGINAL (bug-preserving) matrix:
    same missing-normalization fix as issue 1 in the module docstring, but
    still carries issue 2 (90/270 ports are globally phase-rotated
    duplicates of 0/180, not genuine quadrature ports). For a lossless,
    balanced device (all loss/imbalance kwargs at default 0),
    sum(|output|^2) over the four ports equals |E_s|^2 + |E_LO|^2."""
    outputs = optical_hybrid_90deg(signal_input, local_oscillator, **kwargs)
    return tuple(o / 2.0 for o in outputs)


def optical_hybrid_90deg_corrected(signal_input, local_oscillator,
                                   insertion_loss_signal=0.0,
                                   insertion_loss_local_oscillator=0.0,
                                   phase_imbalance_slo=0.0,
                                   phase_imbalance_iq=0.0,
                                   insertion_loss_imbalance_i=0.0,
                                   insertion_loss_imbalance_q=0.0):
    """Physically standard 90-degree hybrid: the 90-degree phase shift is
    applied to only the LOCAL OSCILLATOR arm (fixing issue 2), with the
    energy-conserving 1/2 normalization (fixing issue 1). Same
    loss/imbalance parameter handling as optical_hybrid_90deg() for a
    direct comparison. This is the version iq_from_hybrid_outputs()
    actually needs to recover a meaningful, non-degenerate Q."""
    for name, val in [
        ("insertion_loss_signal", insertion_loss_signal),
        ("insertion_loss_local_oscillator", insertion_loss_local_oscillator),
        ("insertion_loss_imbalance_i", insertion_loss_imbalance_i),
        ("insertion_loss_imbalance_q", insertion_loss_imbalance_q),
    ]:
        _check_prob_param(val, name)

    IL_signal = 10 ** (-insertion_loss_signal / 10)
    IL_lo = 10 ** (-insertion_loss_local_oscillator / 10)
    Imb_I = 10 ** (-insertion_loss_imbalance_i / 10)
    Imb_Q = 10 ** (-insertion_loss_imbalance_q / 10)

    E_s = signal_input * IL_signal
    E_lo = local_oscillator * IL_lo

    # Row order follows actual phase angle (0, 90, 180, 270 deg relative to
    # the sum port), so ports 180 deg apart -- (0,180) and (90,270) -- are
    # the real/imaginary sum/difference pairs iq_from_hybrid_outputs expects.
    T = 0.5 * np.array([[1, 1], [1, 1j], [1, -1], [1, -1j]], dtype=complex)   # j on LO column only
    phase_matrix = np.diag([
        1,
        np.exp(1j * phase_imbalance_slo),
        np.exp(1j * phase_imbalance_iq),
        np.exp(1j * (phase_imbalance_slo + phase_imbalance_iq)),
    ])
    imbalance_matrix = np.diag([1, Imb_I, Imb_Q, Imb_I * Imb_Q])
    T_adjusted = imbalance_matrix @ phase_matrix @ T

    E_out = T_adjusted @ np.array([E_s, E_lo], dtype=complex)
    return E_out[0], E_out[1], E_out[2], E_out[3]


def iq_from_hybrid_outputs(output0, output90, output180, output270, responsivity=1.0):
    """Balanced photodetection: recover I/Q from the four hybrid outputs,
    the actual signal-processing step a coherent receiver's ADC front end
    performs. I = R*(|E0|^2 - |E180|^2), Q = R*(|E90|^2 - |E270|^2) --
    balanced detection cancels the common DC term that appears in both
    ports of each pair, leaving a term proportional to Re/Im{E_s *
    conj(E_LO)}. This is what makes homodyne detection work, and why it
    needs the LO to be phase-locked to the signal. NOTE: this only
    produces a meaningful (non-zero, phase-dependent) Q when fed outputs
    from optical_hybrid_90deg_corrected() -- outputs from the original
    optical_hybrid_90deg() give Q=0 always, per issue 2 in the module
    docstring."""
    if responsivity <= 0:
        raise ValueError("responsivity must be positive")
    I = responsivity * (np.abs(output0) ** 2 - np.abs(output180) ** 2)
    Q = responsivity * (np.abs(output90) ** 2 - np.abs(output270) ** 2)
    return I, Q


if __name__ == "__main__":
    print("=== 90-degree optical hybrid: classical LO-based coherent receiver front end ===\n")

    E_s = 1 + 1j
    E_lo = 1 - 1j
    kwargs = dict(
        insertion_loss_signal=1.0,
        insertion_loss_local_oscillator=0.5,
        phase_imbalance_slo=np.pi / 18,
        phase_imbalance_iq=np.pi / 36,
        insertion_loss_imbalance_i=0.2,
        insertion_loss_imbalance_q=0.1,
    )

    outs_original = optical_hybrid_90deg(E_s, E_lo, **kwargs)
    outs_normalized = optical_hybrid_90deg_normalized(E_s, E_lo, **kwargs)
    outs_corrected = optical_hybrid_90deg_corrected(E_s, E_lo, **kwargs)
    print("Original (unnormalized, matches MATLAB/Python source exactly):")
    for label, val in zip([0, 90, 180, 270], outs_original):
        print(f"  output{label}: {val:.4f}")
    print("\nNormalized (energy-conserving, still has issue 2, 1/2 the original):")
    for label, val in zip([0, 90, 180, 270], outs_normalized):
        print(f"  output{label}: {val:.4f}")
    print("\nCorrected (physically standard: j applied to LO arm only):")
    for label, val in zip([0, 90, 180, 270], outs_corrected):
        print(f"  output{label}: {val:.4f}")

    I_orig, Q_orig = iq_from_hybrid_outputs(*outs_normalized)
    I_corr, Q_corr = iq_from_hybrid_outputs(*outs_corrected)
    print(f"\nI/Q from ORIGINAL matrix's outputs:  I={I_orig:.4f}, Q={Q_orig:.4f}  <- Q degenerate")
    print(f"I/Q from CORRECTED matrix's outputs: I={I_corr:.4f}, Q={Q_corr:.4f}  <- Q meaningful")
    print("\nThis is the LO-based alternative dgs/sbir_portfolio.py P2/P7 contrast")
    print("their carrier-less time-domain GS approach against.")
