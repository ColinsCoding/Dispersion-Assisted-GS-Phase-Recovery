import numpy as np
import pytest
import sympy as sp
from dgs.biophotonics import (
    beer_lambert, penetration_depth, depth_for_fraction,
    seawater_transmission, find_optimal_wavelength_for_depth,
    pdt_fluence_at_depth, spike_biosensor_response,
    optogenetics_activation, biophotonics_sympy_5,
)


def test_beer_lambert_zero_depth():
    r = beer_lambert(100, 0.5, 0)
    assert r["I"] == pytest.approx(100.0)
    assert r["fraction_transmitted"] == pytest.approx(1.0)


def test_beer_lambert_one_e_depth():
    mu = 2.0
    r = beer_lambert(1.0, mu, 1/mu)
    assert r["fraction_transmitted"] == pytest.approx(1/np.e, rel=1e-4)


def test_beer_lambert_invalid():
    with pytest.raises(ValueError):
        beer_lambert(-1, 0.5, 1)
    with pytest.raises(ValueError):
        beer_lambert(1, -0.1, 1)
    with pytest.raises(ValueError):
        beer_lambert(1, 0.5, -1)


def test_penetration_depth():
    assert penetration_depth(2.0) == pytest.approx(0.5)


def test_penetration_depth_zero_raises():
    with pytest.raises(ValueError):
        penetration_depth(0)


def test_depth_for_fraction_half():
    mu = 1.0
    z = depth_for_fraction(1.0, mu, 0.5)
    assert np.exp(-mu * z) == pytest.approx(0.5, rel=1e-4)


def test_depth_for_fraction_invalid():
    with pytest.raises(ValueError):
        depth_for_fraction(1, 1, 0.0)
    with pytest.raises(ValueError):
        depth_for_fraction(1, 1, 1.0)


def test_seawater_532nm_low_attenuation():
    r = seawater_transmission(532, 10)
    # 532 nm should transmit well -- expect > 90% at 10 m
    assert r["fraction_transmitted"] > 0.85


def test_seawater_1064nm_high_attenuation():
    r = seawater_transmission(1064, 10)
    # 1064 nm absorbs heavily
    assert r["fraction_transmitted"] < 0.1


def test_seawater_out_of_range():
    with pytest.raises(ValueError):
        seawater_transmission(200, 10)


def test_seawater_532_better_than_1064_at_depth():
    r532 = seawater_transmission(532, 100)
    r1064 = seawater_transmission(1064, 100)
    assert r532["fraction_transmitted"] > r1064["fraction_transmitted"]


def test_find_optimal_wavelength():
    r = find_optimal_wavelength_for_depth(100)
    assert r["best_wavelength_nm"] in (480, 510, 532)


def test_pdt_positive_fluence():
    r = pdt_fluence_at_depth(1000, 630, 1.0)
    assert r["fluence_W_m2"] > 0


def test_pdt_fluence_decreases_with_depth():
    f1 = pdt_fluence_at_depth(1000, 630, 0.5)["fluence_W_m2"]
    f2 = pdt_fluence_at_depth(1000, 630, 2.0)["fluence_W_m2"]
    assert f1 > f2


def test_spike_biosensor_zero_conc():
    r = spike_biosensor_response(0.0)
    assert r["delta_n"] == pytest.approx(0.0)
    assert r["detectable"] is False


def test_spike_biosensor_high_conc_detectable():
    r = spike_biosensor_response(1000.0)
    assert r["detectable"] is True


def test_spike_biosensor_invalid():
    with pytest.raises(ValueError):
        spike_biosensor_response(-1)
    with pytest.raises(ValueError):
        spike_biosensor_response(1, binding_efficiency=1.5)


def test_optogenetics_activation_deep():
    r = optogenetics_activation(470, 0.1, 10, depth_mm=5.0)
    # low power + deep tissue -> low activation
    assert r["P_activation"] < 0.5


def test_optogenetics_activation_surface():
    r = optogenetics_activation(470, 10.0, 10, depth_mm=0.01)
    assert r["P_activation"] > 0.5


def test_optogenetics_invalid():
    with pytest.raises(ValueError):
        optogenetics_activation(470, -1, 10)


def test_biophotonics_sympy_5_count():
    eqs = biophotonics_sympy_5()
    assert len(eqs) == 5
    for eq in eqs.values():
        assert isinstance(eq, sp.Eq)
