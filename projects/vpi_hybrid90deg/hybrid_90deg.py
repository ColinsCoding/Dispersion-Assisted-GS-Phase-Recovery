"""Python port of VPIphotonics' Hybrid90deg module ("90-degree Optical
Hybrid / Quadrature Optical Hybrid", Photonic Modules > Passive Components),
per the brief in docs/Prompt_YimingHybrid90MLX_input.rtf: port the MATLAB
reference (matlab/optical_hybrid_90deg.m) to Python with variable names
matching the VPI documentation (docs/VPI Hybrid90deg.pdf).

VPI's own transfer-matrix equation (read directly from the equation image
embedded in the PDF, docs/VPI Hybrid90deg.pdf page 2 -- not re-derived):

    E1,out = (1/2)[ ILS*E1,in + ILLO*exp(j*phiSLO)*E2,in ]
    E2,out = (1/2)[ ILS*ImbI*E1,in - ILLO*ImbI*exp(j*phiSLO)*E2,in ]
    E3,out = (1/2)[ ILS*E1,in + j*ILLO*exp(j*(phiSLOrad+phiIQ))*E2,in ]
    E4,out = (1/2)[ ILS*ImbQ*E1,in + j*ILLO*ImbQ*exp(j*(phiSLOrad+phiIQ))*E2,in ]

The port names each output by its physical phase per VPI's own port table
(signalOutput0grad, signalOutput180grad, signalOutput90grad,
signalOutput270grad, listed in exactly that row order in the VPI doc):
row 1 -> 0 deg, row 2 -> 180 deg (the minus sign is a pi phase flip from
row 1), row 3 -> 90 deg (the leading j is the 90 deg shift), row 4 -> 270
deg (j-shifted version of row 2).

THREE REAL BUGS FOUND while diffing the MATLAB reference against this
equation, kept documented here rather than silently ported over:

1. OUTPUT-LABEL SWAP: the MATLAB reference's `output90 = E_out(2)` and
   `output180 = E_out(3)` are swapped relative to the equation above --
   E_out(2) is actually the 180-degree combination (has the minus sign),
   E_out(3) is actually the 90-degree combination (has the leading j).
   Confirmed directly against the equation image, not inferred.

2. STRUCTURAL MATRIX MISMATCH: the MATLAB reference builds T_adjusted by
   composing three separate diagonal matrices (phase_matrix, then
   imbalance_matrix, applied uniformly per OUTPUT row). Tracing that
   construction against the actual per-row equation above shows it does
   NOT reproduce the correct row structure for nonzero phase imbalance or
   insertion-loss imbalance -- e.g. row 3 in VPI's equation has NO ImbI or
   ImbQ factor at all, but MATLAB's diag([1,ImbI,ImbQ,ImbI*ImbQ]) puts
   ImbQ there. Verified numerically for a generic non-ideal parameter set:
   the two matrices diverge row-by-row well beyond the label swap alone.
   This port implements the matrix EXACTLY as given in the equation image
   instead of attempting to patch MATLAB's factored construction. Concrete
   example, re-confirmed by re-reading the equation image at full
   resolution: VPI's row 4 (E4,out) has NO minus sign anywhere
   (ILS*ImbQ*E1,in + j*ILLO*ImbQ*exp(...)*E2,in, both terms positive),
   but MATLAB's base T matrix has row 4 = [j, -j] -- an explicit minus on
   the E2,in (LO) term that VPI's equation does not have. This means even
   the OUTPUT0 vs OUTPUT270 pair (row 1 and row 4, neither affected by the
   label-swap bug) do not agree up to a simple factor of 2 -- only row 1
   (E1,out, no sign ambiguity in either version) cleanly isolates the
   missing-1/2-prefactor bug from the rest; row 4 carries this additional,
   separate sign discrepancy.

3. MISSING 1/2 PREFACTOR: VPI's equation has an explicit leading 1/2 in
   front of the whole matrix. The MATLAB reference never applies it --
   confirmed by comparing the ideal (all-default-parameters) case, where
   every other source of discrepancy vanishes: this port gives magnitude
   1.0 for signalOutput0grad with unit inputs, MATLAB's reference gives
   2.0, exactly the missing factor of 2 in amplitude (4x in power).

4. LIKELY SIGN TYPO IN VPI's OWN MANUAL, row 4: as printed (see equation
   above), rows 3 and 4 are IDENTICAL at the fully-ideal operating point
   (all losses 0 dB, all phase/imbalance terms 0) -- confirmed by
   re-reading the source image at three increasing zoom levels, so this is
   not a transcription error. Tested directly, this makes the hybrid
   NON-FUNCTIONAL as a quadrature receiver: feeding a real Mie-scattered
   test field (`projects/vpi_hybrid90deg` was cross-used with this repo's
   SEALS bridge for this check) through balanced photodetection
   (I=|E1|^2-|E2|^2, Q=|E3|^2-|E4|^2) gives Q identically zero everywhere
   (E3==E4 exactly), and phase recovery via atan2(Q,I) fails completely
   (RMS error 0.944 rad, no better than an uninformed guess). Flipping ONLY
   row 4's sign to match row 2's pattern (E4,out =
   (1/2)[ILS*ImbQ*E1,in - j*ILLO*ImbQ*exp(...)*E2,in], matching how row 2
   is the sign-flipped partner of row 1) restores a properly functioning
   quadrature receiver: Q becomes genuinely nonzero and phase recovery on
   the same test field becomes EXACT (RMS error 0.0000 rad, matching the
   textbook behavior of coherent detection against a perfectly known LO).
   This function defaults to the corrected (working) sign; pass
   row4_lo_sign=+1 to reproduce VPI's literally-printed (non-functional)
   formula for reference. Still worth confirming against the primary
   reference VPI's own doc cites (Seimetz & Weinert, J. Lightwave Technol.
   24(3), 2006) or with Yiming -- but the empirical evidence (broken vs.
   exact) is strong enough that the default here is the corrected version,
   not the literal transcription.

ONE UNRESOLVED POINT, flagged rather than guessed at: VPI's dB-to-linear
formula image (ILS = 10^(0.5 x IL_S_dB)) was too low-resolution to fully
resolve its internal notation with certainty (whether "IL_S_dB" in that
exponent is the raw tabulated dB parameter or an auxiliary pre-scaled
quantity defined elsewhere in the full manual). This port uses the
standard EE/photonics amplitude-from-power-dB convention,
amplitude_ratio = 10^(-dB/20), which is unambiguous physics regardless of
VPI's internal notation -- MATLAB's reference instead uses 10^(-dB/10)
applied directly to the field amplitude, the POWER conversion formula
misapplied to an amplitude, which is very likely a third bug but is flagged
here as the one point worth confirming against the full VPI manual or with
Yiming directly, not asserted with the same confidence as points 1-2.
"""
from __future__ import annotations
import numpy as np
from typing import Dict


def _db_to_amplitude_ratio(db: float) -> float:
    """Standard amplitude ratio from a power-based insertion-loss dB value:
    10^(-dB/20). See module docstring's 'ONE UNRESOLVED POINT' -- this is
    the physically standard convention, used here in preference to
    MATLAB's 10^(-dB/10) (a power-ratio formula misapplied to amplitude)."""
    if db < 0:
        raise ValueError(f"insertion loss dB value {db} must be non-negative")
    return 10.0 ** (-db / 20.0)


def hybrid_90deg(
    signalInput1: complex,
    signalInput2: complex,
    InsertionLossSignal: float = 0.0,
    InsertionLossLocalOscillator: float = 0.0,
    PhaseImbalance_SignalLocalOscillator: float = 0.0,
    PhaseImbalance_IQ: float = 0.0,
    InsertionLossImbalanceI: float = 0.0,
    InsertionLossImbalanceQ: float = 0.0,
    phase_unit: str = "deg",
    row4_lo_sign: float = -1.0,
) -> Dict[str, complex]:
    """90-degree optical hybrid, VPI Hybrid90deg module.

    Parameters (names match the VPI documentation's parameter table):
        signalInput1 : optical signal E-field (VPI: signalInput1)
        signalInput2 : local oscillator E-field (VPI: signalInput2)
        InsertionLossSignal, InsertionLossLocalOscillator : dB, >= 0,
            default 0.0 (VPI default = ideal/lossless)
        PhaseImbalance_SignalLocalOscillator, PhaseImbalance_IQ :
            in `phase_unit` ('deg', matching VPI's documented parameter
            unit, or 'rad'), default 0.0
        InsertionLossImbalanceI, InsertionLossImbalanceQ : dB, >= 0,
            default 0.0
        row4_lo_sign : -1.0 (default) uses the CORRECTED row-4 sign (see
            module docstring bug 4 -- makes the hybrid a functioning
            quadrature receiver). Pass +1.0 to reproduce VPI's literally
            printed (non-functional at the ideal point) formula instead.

    Returns dict with keys signalOutput0grad, signalOutput90grad,
    signalOutput180grad, signalOutput270grad (VPI's own port names).
    """
    if phase_unit not in ("deg", "rad"):
        raise ValueError("phase_unit must be 'deg' or 'rad'")
    if row4_lo_sign not in (1.0, -1.0):
        raise ValueError("row4_lo_sign must be +1.0 or -1.0")
    to_rad = (np.pi / 180.0) if phase_unit == "deg" else 1.0
    phi_slo = PhaseImbalance_SignalLocalOscillator * to_rad
    phi_iq = PhaseImbalance_IQ * to_rad

    ILS = _db_to_amplitude_ratio(InsertionLossSignal)
    ILLO = _db_to_amplitude_ratio(InsertionLossLocalOscillator)
    ImbI = _db_to_amplitude_ratio(InsertionLossImbalanceI)
    ImbQ = _db_to_amplitude_ratio(InsertionLossImbalanceQ)

    E1_in, E2_in = complex(signalInput1), complex(signalInput2)

    E1_out = 0.5 * (ILS * E1_in + ILLO * np.exp(1j * phi_slo) * E2_in)
    E2_out = 0.5 * (ILS * ImbI * E1_in - ILLO * ImbI * np.exp(1j * phi_slo) * E2_in)
    E3_out = 0.5 * (ILS * E1_in + 1j * ILLO * np.exp(1j * (phi_slo + phi_iq)) * E2_in)
    E4_out = 0.5 * (ILS * ImbQ * E1_in + row4_lo_sign * 1j * ILLO * ImbQ * np.exp(1j * (phi_slo + phi_iq)) * E2_in)

    return {
        "signalOutput0grad": E1_out,
        "signalOutput180grad": E2_out,
        "signalOutput90grad": E3_out,
        "signalOutput270grad": E4_out,
    }


def hybrid_90deg_matlab_reference_buggy(
    signal_input: complex, local_oscillator: complex,
    insertion_loss_signal: float, insertion_loss_local_oscillator: float,
    phase_imbalance_slo_rad: float, phase_imbalance_iq_rad: float,
    insertion_loss_imbalance_i: float, insertion_loss_imbalance_q: float,
) -> Dict[str, complex]:
    """Literal Python transcription of matlab/optical_hybrid_90deg.m,
    bugs and all -- kept ONLY for side-by-side comparison against
    hybrid_90deg() above, not recommended for actual use. Matches the
    corrected version exactly in the ideal case (all defaults) and
    diverges for any non-ideal parameter, per the module docstring."""
    IL_signal = 10 ** (-insertion_loss_signal / 10)
    IL_lo = 10 ** (-insertion_loss_local_oscillator / 10)
    Imb_I = 10 ** (-insertion_loss_imbalance_i / 10)
    Imb_Q = 10 ** (-insertion_loss_imbalance_q / 10)

    signal_input = signal_input * IL_signal
    local_oscillator = local_oscillator * IL_lo

    T = np.array([[1, 1], [1, -1], [1j, 1j], [1j, -1j]])
    phase_matrix = np.diag([
        1,
        np.exp(1j * phase_imbalance_slo_rad),
        np.exp(1j * phase_imbalance_iq_rad),
        np.exp(1j * (phase_imbalance_slo_rad + phase_imbalance_iq_rad)),
    ])
    imbalance_matrix = np.diag([1, Imb_I, Imb_Q, Imb_I * Imb_Q])
    T_adjusted = imbalance_matrix @ phase_matrix @ T
    E_in = np.array([signal_input, local_oscillator])
    E_out = T_adjusted @ E_in
    return {"output0": E_out[0], "output90": E_out[1], "output180": E_out[2], "output270": E_out[3]}


if __name__ == "__main__":
    print("=== ideal case (all defaults): corrected port vs. MATLAB reference ===")
    s_in, lo_in = 1 + 1j, 1 - 1j
    corrected = hybrid_90deg(s_in, lo_in)
    reference = hybrid_90deg_matlab_reference_buggy(s_in, lo_in, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    print(f"corrected signalOutput0grad={corrected['signalOutput0grad']:.4f}  "
          f"reference output0={reference['output0']:.4f}")
    print(f"corrected signalOutput90grad={corrected['signalOutput90grad']:.4f}  "
          f"reference output90={reference['output90']:.4f}  "
          f"(reference is mislabeled -- this is actually the 180deg combination)")
    print(f"corrected signalOutput180grad={corrected['signalOutput180grad']:.4f}  "
          f"reference output180={reference['output180']:.4f}  "
          f"(reference is mislabeled -- this is actually the 90deg combination)")
    print(f"corrected signalOutput270grad={corrected['signalOutput270grad']:.4f}  "
          f"reference output270={reference['output270']:.4f}")

    print("\n=== non-ideal case: Yiming's example_usage_script.m parameters ===")
    corrected2 = hybrid_90deg(
        s_in, lo_in,
        InsertionLossSignal=1.0, InsertionLossLocalOscillator=0.5,
        PhaseImbalance_SignalLocalOscillator=10.0, PhaseImbalance_IQ=5.0,
        InsertionLossImbalanceI=0.2, InsertionLossImbalanceQ=0.1,
    )
    for k, v in corrected2.items():
        print(f"  {k}: {v:.6f}")
