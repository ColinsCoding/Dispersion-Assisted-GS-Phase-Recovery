import numpy as np
import pytest
from dgs.laser_safety import (
    beam_area_cm2, irradiance_W_cm2, beam_diameter_at_distance_cm,
    nominal_hazard_distance_cm, required_optical_density,
)


def test_beam_area_matches_circle_formula():
    assert beam_area_cm2(2.0) == pytest.approx(np.pi)  # d=2cm -> r=1cm -> area=pi*1^2


def test_beam_area_rejects_nonpositive_diameter():
    with pytest.raises(ValueError):
        beam_area_cm2(0.0)


def test_irradiance_scales_inversely_with_area():
    small = irradiance_W_cm2(1.0, diameter_cm=0.1)
    large = irradiance_W_cm2(1.0, diameter_cm=1.0)
    assert small > large


def test_irradiance_rejects_nonpositive_power():
    with pytest.raises(ValueError):
        irradiance_W_cm2(-1.0, 0.1)


def test_beam_diameter_grows_with_distance():
    d_near = beam_diameter_at_distance_cm(d0_cm=0.2, divergence_rad=0.001, distance_cm=100)
    d_far = beam_diameter_at_distance_cm(d0_cm=0.2, divergence_rad=0.001, distance_cm=10000)
    assert d_far > d_near > 0.2


def test_beam_diameter_rejects_negative_divergence():
    with pytest.raises(ValueError):
        beam_diameter_at_distance_cm(0.2, -0.001, 100)


def test_nhz_zero_when_source_already_below_mpe():
    # tiny power, huge beam -> source irradiance already under MPE
    nhz = nominal_hazard_distance_cm(power_W=1e-9, mpe_W_cm2=1.0, d0_cm=1.0, divergence_rad=0.001)
    assert nhz == 0.0


def test_nhz_matches_hand_solved_case():
    # power=1W, mpe=1e-3 W/cm^2, d0=0.1cm, divergence=0.001 rad
    # NHZ = (sqrt(4*1/(pi*1e-3)) - 0.1) / 0.001
    power_W, mpe, d0, div = 1.0, 1e-3, 0.1, 0.001
    expected = (np.sqrt(4 * power_W / (np.pi * mpe)) - d0) / div
    assert nominal_hazard_distance_cm(power_W, mpe, d0, div) == pytest.approx(expected)


def test_nhz_rejects_zero_divergence():
    with pytest.raises(ValueError):
        nominal_hazard_distance_cm(1.0, 1e-3, 0.1, 0.0)


def test_nhz_rejects_nonpositive_mpe():
    with pytest.raises(ValueError):
        nominal_hazard_distance_cm(1.0, 0.0, 0.1, 0.001)


def test_required_optical_density_zero_when_already_safe():
    assert required_optical_density(1e-4, mpe_W_cm2=1e-3) == 0.0


def test_required_optical_density_matches_log_formula():
    od = required_optical_density(incident_irradiance_W_cm2=1.0, mpe_W_cm2=1e-3)
    assert od == pytest.approx(3.0)  # log10(1.0/1e-3) = 3


def test_required_optical_density_rejects_nonpositive_incident():
    with pytest.raises(ValueError):
        required_optical_density(0.0, 1e-3)
