import numpy as np
import pytest
from hybrid_90deg import hybrid_90deg, hybrid_90deg_matlab_reference_buggy, _db_to_amplitude_ratio


def test_db_to_amplitude_ratio_zero_db_is_unity():
    assert _db_to_amplitude_ratio(0.0) == pytest.approx(1.0)


def test_db_to_amplitude_ratio_standard_convention():
    # 20 dB power loss -> amplitude ratio 0.1 (10^(-20/20))
    assert _db_to_amplitude_ratio(20.0) == pytest.approx(0.1)


def test_db_to_amplitude_ratio_rejects_negative():
    with pytest.raises(ValueError):
        _db_to_amplitude_ratio(-1.0)


def test_ideal_case_output0_is_average_of_inputs():
    # phi=0, no imbalance, no loss: E1,out = (1/2)(E1in + E2in)
    result = hybrid_90deg(1 + 1j, 1 - 1j)
    assert result["signalOutput0grad"] == pytest.approx(1.0 + 0j)


def test_ideal_case_four_outputs_have_correct_phase_relationships():
    # with equal-magnitude, in-phase-ish inputs, verify the documented
    # 0/180/90/270 structure directly from the formulas. Uses the
    # CORRECTED row-4 sign (row4_lo_sign=-1, the default) -- with VPI's
    # literally-printed sign, output270 would equal output90 exactly,
    # which is the bug documented in the module docstring (point 4).
    E1, E2 = 2.0 + 0j, 1.0 + 0j
    result = hybrid_90deg(E1, E2)
    expected_0 = 0.5 * (E1 + E2)
    expected_180 = 0.5 * (E1 - E2)
    expected_90 = 0.5 * (E1 + 1j * E2)
    expected_270 = 0.5 * (E1 - 1j * E2)
    assert result["signalOutput0grad"] == pytest.approx(expected_0)
    assert result["signalOutput180grad"] == pytest.approx(expected_180)
    assert result["signalOutput90grad"] == pytest.approx(expected_90)
    assert result["signalOutput270grad"] == pytest.approx(expected_270)


def test_row4_sign_flag_reproduces_literally_printed_vpi_formula():
    # row4_lo_sign=+1.0 reproduces VPI's literally-printed formula, where
    # output90 and output270 are identical at the ideal operating point --
    # the bug documented in the module docstring (point 4), kept testable
    # rather than silently removed.
    E1, E2 = 2.0 + 0j, 1.0 + 0j
    result_printed = hybrid_90deg(E1, E2, row4_lo_sign=1.0)
    assert result_printed["signalOutput90grad"] == pytest.approx(result_printed["signalOutput270grad"])

    result_corrected = hybrid_90deg(E1, E2)  # default row4_lo_sign=-1.0
    assert result_corrected["signalOutput90grad"] != pytest.approx(result_corrected["signalOutput270grad"])


def test_rejects_bad_row4_lo_sign():
    with pytest.raises(ValueError):
        hybrid_90deg(1.0, 1.0, row4_lo_sign=0.5)


def test_180deg_output_is_negative_relative_to_0deg_for_real_inputs():
    # confirms row 2 really is the phase-flipped (180deg) combination
    E1, E2 = 3.0, 1.0
    result = hybrid_90deg(E1, E2)
    assert result["signalOutput0grad"].real > 0
    assert result["signalOutput180grad"].real == pytest.approx((E1 - E2) / 2)


def test_90deg_output_has_imaginary_contribution_from_signal2():
    # confirms row 3 really carries the j (90deg) factor on E2in
    result = hybrid_90deg(1.0, 1.0)
    assert result["signalOutput90grad"].imag == pytest.approx(0.5)


def test_insertion_loss_reduces_output_magnitude():
    result_lossless = hybrid_90deg(1.0, 1.0)
    result_lossy = hybrid_90deg(1.0, 1.0, InsertionLossSignal=6.0)
    assert abs(result_lossy["signalOutput0grad"]) < abs(result_lossless["signalOutput0grad"])


def test_phase_unit_deg_vs_rad_equivalence():
    result_deg = hybrid_90deg(1.0, 1.0, PhaseImbalance_SignalLocalOscillator=90.0, phase_unit="deg")
    result_rad = hybrid_90deg(1.0, 1.0, PhaseImbalance_SignalLocalOscillator=np.pi / 2, phase_unit="rad")
    assert result_deg["signalOutput0grad"] == pytest.approx(result_rad["signalOutput0grad"])


def test_rejects_bad_phase_unit():
    with pytest.raises(ValueError):
        hybrid_90deg(1.0, 1.0, phase_unit="grad")


def test_rejects_negative_insertion_loss():
    with pytest.raises(ValueError):
        hybrid_90deg(1.0, 1.0, InsertionLossSignal=-1.0)


def test_corrected_and_reference_agree_on_row1_up_to_factor_of_two():
    # row 1 (E1,out / output0) has no sign ambiguity in either version, so
    # it cleanly isolates the missing-1/2-prefactor bug from everything
    # else. Row 4 (output270) does NOT reduce to a clean 2x relationship
    # even in the ideal case -- MATLAB's base T matrix has an extra minus
    # sign on that row's LO term that VPI's own equation does not have
    # (see module docstring bug list, point 2) -- so it is deliberately
    # NOT asserted here.
    s_in, lo_in = 1 + 1j, 1 - 1j
    corrected = hybrid_90deg(s_in, lo_in)
    reference = hybrid_90deg_matlab_reference_buggy(s_in, lo_in, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert reference["output0"] == pytest.approx(2 * corrected["signalOutput0grad"])


def test_reference_row4_has_extra_sign_discrepancy_vs_corrected():
    # documents the row-4 sign mismatch found while testing, rather than
    # letting it silently pass as "close enough"
    s_in, lo_in = 1 + 1j, 1 - 1j
    corrected = hybrid_90deg(s_in, lo_in)
    reference = hybrid_90deg_matlab_reference_buggy(s_in, lo_in, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert reference["output270"] != pytest.approx(2 * corrected["signalOutput270grad"], rel=1e-6)


def test_corrected_and_reference_disagree_for_nonideal_imbalance():
    # confirms the structural mismatch (bug 2) is real and not just the
    # label swap / missing 1/2 -- even after correcting for both, the
    # imbalanced case should NOT match
    s_in, lo_in = 1 + 1j, 1 - 1j
    corrected = hybrid_90deg(s_in, lo_in, InsertionLossImbalanceQ=3.0)
    reference = hybrid_90deg_matlab_reference_buggy(
        s_in, lo_in, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0)
    # reference's "output180" (its label for row 3, the true 90deg row) should NOT
    # equal 2x the corrected 90deg output, because MATLAB's ImbQ placement is wrong
    assert reference["output180"] != pytest.approx(2 * corrected["signalOutput90grad"], rel=1e-6)
