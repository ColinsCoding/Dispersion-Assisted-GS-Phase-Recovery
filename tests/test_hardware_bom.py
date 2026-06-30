import pytest
import sympy as sp
from dgs.hardware_bom import (
    fiber_gdd, fiber_length_for_gdd, photodetector_snr,
    sampling_requirement, bom_total_cost,
    BOM_MINIMUM_VIABLE, BOM_FULL_LAB, hardware_sympy_5,
)


def test_fiber_gdd_300m_is_small():
    # SMF-28 at 1550 nm: beta_2 ~ -21.7 ps^2/km, so 300 m = 0.3 km -> ~6.5 ps^2
    # 300 m is NOT enough for the 5000 ps^2 threshold -- need a CFBG or 230 km
    r = fiber_gdd(300.0)
    assert r["meets_5000_ps2_threshold"] is False
    assert abs(r["gdd_ps2"]) < 50.0   # should be ~6.5 ps^2


def test_fiber_gdd_235km_exceeds_5000():
    r = fiber_gdd(235000.0)  # 235 km
    assert r["meets_5000_ps2_threshold"] is True


def test_fiber_gdd_negative_length_raises():
    with pytest.raises(ValueError):
        fiber_gdd(-1.0)


def test_fiber_gdd_sign_smf28():
    # SMF-28 at 1550 nm is anomalous dispersion: negative GDD
    r = fiber_gdd(1000.0)
    assert r["gdd_ps2"] < 0


def test_fiber_length_for_5000_ps2():
    r = fiber_length_for_gdd(5000.0)
    # SMF-28 beta_2 ~ 21.7 ps^2/km -> need ~230 km for 5000 ps^2
    assert 100e3 < r["length_m"] < 400e3


def test_fiber_length_roundtrip():
    target = 8000.0
    r = fiber_length_for_gdd(target)
    check = fiber_gdd(r["length_m"])
    assert abs(check["gdd_ps2"]) == pytest.approx(target, rel=0.01)


def test_fiber_length_invalid_raises():
    with pytest.raises(ValueError):
        fiber_length_for_gdd(-100)


def test_snr_positive_power():
    r = photodetector_snr(1.0)
    assert r["SNR_dB"] > 0


def test_snr_higher_power_gives_higher_snr():
    r1 = photodetector_snr(0.1)
    r2 = photodetector_snr(10.0)
    assert r2["SNR_dB"] > r1["SNR_dB"]


def test_snr_zero_power_raises():
    with pytest.raises(ValueError):
        photodetector_snr(0.0)


def test_snr_noise_regime():
    # at 1 mW with 1 GHz BW, InGaAs should be shot-noise limited
    r = photodetector_snr(1.0, bandwidth_GHz=1.0)
    assert r["noise_limited_by"] in ("shot", "thermal")


def test_sampling_requirement_nyquist():
    r = sampling_requirement(1.0)
    assert r["nyquist_rate_GSa_s"] == pytest.approx(2.0)


def test_sampling_requirement_practical():
    r = sampling_requirement(2.0, oversampling_factor=3.0)
    assert r["practical_rate_GSa_s"] == pytest.approx(6.0)


def test_sampling_requirement_invalid():
    with pytest.raises(ValueError):
        sampling_requirement(0.0)


def test_bom_minimum_has_fiber():
    names = [item["item"].lower() for item in BOM_MINIMUM_VIABLE]
    assert any("fiber" in n for n in names)


def test_bom_minimum_has_detector():
    names = [item["item"].lower() for item in BOM_MINIMUM_VIABLE]
    assert any("detector" in n or "photodetector" in n for n in names)


def test_bom_total_cost_minimum():
    total = bom_total_cost(BOM_MINIMUM_VIABLE)
    assert 500 < total < 5000


def test_bom_total_cost_full_lab_expensive():
    total = bom_total_cost(BOM_FULL_LAB)
    assert total > 50000


def test_hardware_sympy_5_count():
    eqs = hardware_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)
