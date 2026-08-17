import numpy as np
import pytest
from dgs.acid_base_chemistry import (
    pH_from_H_concentration, H_concentration_from_pH,
    pOH_from_OH_concentration, OH_concentration_from_pOH,
    pH_pOH_relationship, water_autoionization_check,
    strong_acid_pH, pKa_from_Ka, weak_acid_pH, henderson_hasselbalch,
    titration_curve, KW_25C,
)


def test_pH_from_H_concentration_neutral_water():
    assert pH_from_H_concentration(1e-7) == pytest.approx(7.0)


def test_pH_H_concentration_round_trip():
    for H_conc in [1e-1, 1e-4, 1e-9, 1e-13]:
        pH = pH_from_H_concentration(H_conc)
        back = H_concentration_from_pH(pH)
        assert back == pytest.approx(H_conc, rel=1e-9)


def test_pH_from_H_concentration_rejects_nonpositive():
    with pytest.raises(ValueError):
        pH_from_H_concentration(0.0)
    with pytest.raises(ValueError):
        pH_from_H_concentration(-1.0)


def test_pOH_OH_concentration_round_trip():
    for OH_conc in [1e-2, 1e-6, 1e-11]:
        pOH = pOH_from_OH_concentration(OH_conc)
        back = OH_concentration_from_pOH(pOH)
        assert back == pytest.approx(OH_conc, rel=1e-9)


def test_pH_pOH_relationship_sums_to_14_at_25C():
    result = pH_pOH_relationship(pH=4.0)
    assert result["pOH"] == pytest.approx(10.0)
    assert result["pH_plus_pOH"] == pytest.approx(14.0)


def test_pH_pOH_relationship_from_pOH():
    result = pH_pOH_relationship(pOH=3.0)
    assert result["pH"] == pytest.approx(11.0)


def test_pH_pOH_relationship_rejects_both_given():
    with pytest.raises(ValueError):
        pH_pOH_relationship(pH=7.0, pOH=7.0)


def test_pH_pOH_relationship_rejects_neither_given():
    with pytest.raises(ValueError):
        pH_pOH_relationship()


def test_water_autoionization_check_consistent_pair():
    result = water_autoionization_check(1e-7, 1e-7)
    assert result["consistent"] is True


def test_water_autoionization_check_inconsistent_pair():
    result = water_autoionization_check(1e-3, 1e-3)
    assert result["consistent"] is False


def test_water_autoionization_check_rejects_nonpositive():
    with pytest.raises(ValueError):
        water_autoionization_check(-1e-7, 1e-7)


def test_strong_acid_pH_matches_direct_formula():
    assert strong_acid_pH(0.1) == pytest.approx(1.0)
    assert strong_acid_pH(1.0) == pytest.approx(0.0)


def test_strong_acid_pH_rejects_nonpositive():
    with pytest.raises(ValueError):
        strong_acid_pH(0.0)


def test_pKa_from_Ka_of_acetic_acid():
    # acetic acid Ka=1.8e-5 -> pKa ~= 4.74 (textbook value)
    assert pKa_from_Ka(1.8e-5) == pytest.approx(4.745, abs=0.01)


def test_pKa_from_Ka_rejects_nonpositive():
    with pytest.raises(ValueError):
        pKa_from_Ka(0.0)


def test_weak_acid_pH_less_dissociated_than_strong_acid():
    weak = weak_acid_pH(Ka=1.8e-5, concentration=0.1)
    strong = strong_acid_pH(0.1)
    assert weak["pH"] > strong  # weak acid is less acidic (higher pH) at same nominal concentration
    assert 0.0 < weak["fraction_dissociated"] < 1.0


def test_weak_acid_pH_matches_known_acetic_acid_value():
    # textbook: 0.1 M acetic acid has pH ~= 2.87
    result = weak_acid_pH(Ka=1.8e-5, concentration=0.1)
    assert result["pH"] == pytest.approx(2.87, abs=0.02)


def test_weak_acid_pH_satisfies_equilibrium_expression():
    Ka, C = 1.8e-5, 0.1
    result = weak_acid_pH(Ka, C)
    x = result["H_conc"]
    Ka_check = x ** 2 / (C - x)
    assert Ka_check == pytest.approx(Ka, rel=1e-6)


def test_weak_acid_pH_rejects_nonpositive():
    with pytest.raises(ValueError):
        weak_acid_pH(Ka=-1e-5, concentration=0.1)
    with pytest.raises(ValueError):
        weak_acid_pH(Ka=1.8e-5, concentration=0.0)


def test_henderson_hasselbalch_equal_concentrations_equals_pKa():
    pKa = 4.74
    assert henderson_hasselbalch(pKa, base_conc=0.2, acid_conc=0.2) == pytest.approx(pKa)


def test_henderson_hasselbalch_more_base_raises_pH():
    pKa = 4.74
    low = henderson_hasselbalch(pKa, base_conc=0.05, acid_conc=0.2)
    high = henderson_hasselbalch(pKa, base_conc=0.2, acid_conc=0.05)
    assert high > pKa > low


def test_henderson_hasselbalch_rejects_nonpositive():
    with pytest.raises(ValueError):
        henderson_hasselbalch(4.74, base_conc=0.0, acid_conc=0.1)


def test_titration_curve_equivalence_volume_matches_moles_balance():
    curve = titration_curve(C_acid=0.1, V_acid=25.0, C_base=0.1)
    assert curve["equivalence_volume"] == pytest.approx(25.0)


def test_titration_curve_starts_acidic_ends_basic():
    curve = titration_curve(C_acid=0.1, V_acid=25.0, C_base=0.1)
    assert curve["pH"][0] < 7.0
    assert curve["pH"][-1] > 7.0


def test_titration_curve_has_sharp_jump_near_equivalence():
    curve = titration_curve(C_acid=0.1, V_acid=25.0, C_base=0.1, n_points=500)
    eq_idx = int(np.argmin(np.abs(curve["V_base"] - curve["equivalence_volume"])))
    # pH should change a lot in a narrow window right around equivalence
    window = 5
    jump = curve["pH"][eq_idx + window] - curve["pH"][eq_idx - window]
    assert jump > 3.0  # characteristic steep titration jump


def test_titration_curve_rejects_bad_input():
    with pytest.raises(ValueError):
        titration_curve(C_acid=0.0, V_acid=25.0, C_base=0.1)
    with pytest.raises(ValueError):
        titration_curve(C_acid=0.1, V_acid=25.0, C_base=0.1, n_points=2)
