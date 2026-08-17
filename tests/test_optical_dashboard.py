import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dgs.optical_dashboard import phase_retrieval_dashboard
from dgs.gs_core import make_measurements


def _toy_measurements():
    return make_measurements('QPSK', n_symbols=16, sps=8, D1=-5000, D2=-5750, snr_db=30, rng_seed=1)


def test_phase_retrieval_dashboard_returns_expected_keys():
    m = _toy_measurements()
    result = phase_retrieval_dashboard(m['I1'], m['I2'], m['D1'], m['D2'], n_iter=10, phi_true=m['phi_true'])
    assert set(result.keys()) == {"figure", "phi", "errors", "E_history", "laser_safety"}
    assert result["laser_safety"] is None  # no laser params passed
    plt.close(result["figure"])


def test_phase_retrieval_dashboard_phi_length_matches_measurement():
    m = _toy_measurements()
    result = phase_retrieval_dashboard(m['I1'], m['I2'], m['D1'], m['D2'], n_iter=10)
    assert len(result["phi"]) == len(m['I1'])
    plt.close(result["figure"])


def test_phase_retrieval_dashboard_laser_safety_in_range():
    m = _toy_measurements()
    result = phase_retrieval_dashboard(
        m['I1'], m['I2'], m['D1'], m['D2'], n_iter=5,
        laser_power_W=0.001, wavelength_nm=850.0, beam_diameter_um=50.0, exposure_s=1.0)
    assert result["laser_safety"] is not None
    assert "exceeds_mpe" in result["laser_safety"]
    assert "disclaimer" in result["laser_safety"]
    plt.close(result["figure"])


def test_phase_retrieval_dashboard_laser_safety_out_of_range_returns_none():
    # this repo's own default telecom wavelength (1550nm) is outside the
    # 400-1050nm range dgs/laser_safety_mpe.py's illustrative model covers
    m = _toy_measurements()
    result = phase_retrieval_dashboard(
        m['I1'], m['I2'], m['D1'], m['D2'], n_iter=5,
        laser_power_W=0.001, wavelength_nm=1550.0, beam_diameter_um=50.0, exposure_s=1.0)
    assert result["laser_safety"] is None
    plt.close(result["figure"])


def test_phase_retrieval_dashboard_no_laser_params_gives_none():
    m = _toy_measurements()
    result = phase_retrieval_dashboard(m['I1'], m['I2'], m['D1'], m['D2'], n_iter=5)
    assert result["laser_safety"] is None
    plt.close(result["figure"])


def test_phase_retrieval_dashboard_figure_has_four_axes():
    m = _toy_measurements()
    result = phase_retrieval_dashboard(m['I1'], m['I2'], m['D1'], m['D2'], n_iter=5)
    assert len(result["figure"].axes) == 4
    plt.close(result["figure"])
