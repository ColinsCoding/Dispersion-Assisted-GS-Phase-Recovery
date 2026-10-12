"""Physics tests: Rayleigh, Rayleigh-Debye and Mie must agree where their ranges
of validity overlap, disagree where they should, and Mie must satisfy exact
identities. Also the SEALS map, the source lineshape and Brownian motion."""
from pathlib import Path

import numpy as np
import pytest

from scarlett import physics as ph

LAM = 1590e-9
THETA = np.linspace(0.05, np.pi - 0.05, 41)


# ------------------------------------------------------------ basics
def test_size_parameter_definition():
    assert ph.size_parameter(1e-6, 500e-9, 1.33) == pytest.approx(np.pi * 1e-6 * 1.33 / 500e-9)
    assert ph.size_parameter(9940e-9, LAM, 1.0) == pytest.approx(19.6399, rel=1e-5)


@pytest.mark.parametrize("bad", [(-1e-6, LAM, 1.0), (1e-6, 0.0, 1.0), (1e-6, LAM, -1.33)])
def test_size_parameter_rejects_non_positive(bad):
    with pytest.raises(ValueError, match="must be positive"):
        ph.size_parameter(*bad)


def test_photon_energy():
    assert ph.photon_energy(LAM) / ph.E_CHARGE == pytest.approx(1239.84198 / 1590, rel=1e-6)  # eV


# ------------------------------------------------------------ Rayleigh limit of Mie
@pytest.mark.parametrize("nmed", [1.0, 1.33])
def test_mie_reduces_to_rayleigh_for_small_particles(nmed):
    npar, dia = 1.39 * nmed, 20e-9
    mie = ph.mie_scattering(npar, nmed, dia, LAM, THETA, 0.1)
    rayleigh = ph.rayleigh_intensity(dia, LAM, nmed, npar, THETA, 0.1)
    assert mie.x < 0.06
    np.testing.assert_allclose(mie.i_unpolarized, rayleigh, rtol=5e-3)
    assert mie.sigma_sca == pytest.approx(ph.rayleigh_cross_section(dia, LAM, nmed, npar), rel=1e-3)


def test_rayleigh_polarization_pattern():
    """Dipole: perpendicular component isotropic, parallel component ~ cos^2(theta)."""
    mie = ph.mie_scattering(1.39, 1.0, 20e-9, LAM, THETA, 0.1)
    np.testing.assert_allclose(mie.i_par / mie.i_perp, np.cos(THETA) ** 2, atol=5e-3)
    assert np.ptp(mie.i_perp) / mie.i_perp.mean() < 5e-3


def test_rayleigh_scaling_laws():
    """sigma ~ dia^6 and ~ lambda^-4, for the formula and for Mie at small x."""
    base = ph.rayleigh_cross_section(10e-9, LAM, 1.0, 1.39)
    assert ph.rayleigh_cross_section(20e-9, LAM, 1.0, 1.39) / base == pytest.approx(2 ** 6)
    assert ph.rayleigh_cross_section(10e-9, LAM / 2, 1.0, 1.39) / base == pytest.approx(2 ** 4)
    mie_ratio = ph.mie_cross_section(1.39, 1.0, 20e-9, LAM) / ph.mie_cross_section(1.39, 1.0, 10e-9, LAM)
    assert mie_ratio == pytest.approx(2 ** 6, rel=1e-3)


def test_rayleigh_fails_for_large_particles():
    """At x = 19.6 Rayleigh is far outside its range and overestimates grossly."""
    dia = 9940e-9
    ratio = ph.rayleigh_cross_section(dia, LAM, 1.0, 1.39) / ph.mie_cross_section(1.39, 1.0, dia, LAM)
    assert ratio > 100
    assert ph.regime(dia, LAM, 1.39) == "mie"
    assert ph.regime(20e-9, LAM, 1.39) == "rayleigh"
    assert ph.regime(1e-6, 633e-9, 1.34, 1.33) == "rayleigh-debye"


# ------------------------------------------------------------ Rayleigh-Debye
def test_rayleigh_debye_reduces_to_rayleigh():
    rdg = ph.rayleigh_debye_intensity(20e-9, LAM, 1.33, 1.39, THETA, 0.1)
    np.testing.assert_allclose(rdg, ph.rayleigh_intensity(20e-9, LAM, 1.33, 1.39, THETA, 0.1), rtol=5e-3)


def test_rayleigh_debye_matches_mie_for_soft_particles():
    """m = 1.34/1.33: forward lobe within 5 %, total cross section within 3 %."""
    forward = THETA[:8]
    mie = ph.mie_scattering(1.34, 1.33, 1e-6, 633e-9, forward, 0.1).i_unpolarized
    rdg = ph.rayleigh_debye_intensity(1e-6, 633e-9, 1.33, 1.34, forward, 0.1)
    np.testing.assert_allclose(rdg, mie, rtol=0.05)
    total = ph.rayleigh_debye_cross_section(1e-6, 633e-9, 1.33, 1.34)
    assert total == pytest.approx(ph.mie_cross_section(1.34, 1.33, 1e-6, 633e-9), rel=0.03)


def test_rayleigh_debye_fails_for_hard_large_particles():
    """m = 1.39, x = 19.6 violates 2x|m-1| << 1: the models must disagree."""
    mie = ph.mie_cross_section(1.39, 1.0, 9940e-9, LAM)
    rdg = ph.rayleigh_debye_cross_section(9940e-9, LAM, 1.0, 1.39)
    assert abs(rdg / mie - 1) > 0.5


def test_form_amplitude_limit_and_no_nan_at_forward_angle():
    assert ph.sphere_form_amplitude(0.0) == 1.0
    assert ph.sphere_form_amplitude(1e-5) == pytest.approx(1.0, abs=1e-9)
    assert ph.sphere_form_amplitude(np.pi) == pytest.approx(3 / np.pi ** 2)
    assert np.isfinite(ph.rayleigh_debye_intensity(1e-6, 633e-9, 1.33, 1.34, 0.0, 0.1))[0]


# ------------------------------------------------------------ exact Mie identities
@pytest.mark.parametrize("x", [0.5, 3.0, 20.0, 60.0])
def test_optical_theorem_and_no_absorption_for_real_index(x):
    q = ph.mie_efficiencies(1.39, x)
    s1, s2 = ph.mie_amplitudes(1.39, x, [0.0])
    assert 4 / x ** 2 * s1[0].real == pytest.approx(q["qext"], rel=1e-9)
    assert q["qsca"] == pytest.approx(q["qext"], rel=1e-9)
    assert abs(q["qabs"]) < 1e-9
    assert s1[0] == pytest.approx(s2[0])


@pytest.mark.parametrize("dia", [0.2e-6, 2e-6, 9.94e-6])
def test_differential_cross_section_integrates_to_total(dia):
    theta = np.linspace(0, np.pi, 20001)
    dsigma = ph.mie_differential_cross_section(1.39, 1.0, dia, LAM, theta)
    total = 2 * np.pi * np.trapezoid(dsigma * np.sin(theta), theta)
    assert total == pytest.approx(ph.mie_cross_section(1.39, 1.0, dia, LAM), rel=1e-5)


def test_absorbing_particle_has_positive_absorption():
    q = ph.mie_efficiencies(1.39 + 0.05j, 5.0)
    assert q["qabs"] > 0.01 and q["qext"] > q["qsca"]


def test_extinction_paradox_large_sphere():
    """Q_ext -> 2 for a very large sphere."""
    assert ph.mie_efficiencies(1.39, 2000.0)["qext"] == pytest.approx(2.0, abs=0.02)


def test_irradiance_scales_with_distance_and_incident_power():
    near = ph.mie_scattering(1.39, 1.0, 2e-6, LAM, THETA, 0.1)
    far = ph.mie_scattering(1.39, 1.0, 2e-6, LAM, THETA, 0.2, incident_irradiance=3.0)
    np.testing.assert_allclose(far.i_unpolarized, near.i_unpolarized * 3 / 4, rtol=1e-12)


def test_pattern_is_symmetric_in_scattering_angle():
    plus = ph.mie_differential_cross_section(1.39, 1.0, 5e-6, LAM, THETA)
    minus = ph.mie_differential_cross_section(1.39, 1.0, 5e-6, LAM, -THETA)
    np.testing.assert_allclose(plus, minus, rtol=1e-12)


# ------------------------------------------------------------ documented legacy behaviour
def test_legacy_total_is_twice_unpolarized():
    mie = ph.mie_scattering(1.39, 1.0, 9940e-9, LAM, THETA, 0.1)
    np.testing.assert_allclose(mie.i_legacy_total, 2 * mie.i_unpolarized, rtol=1e-12)


def test_legacy_rayleigh_debye_is_off_by_nmed_to_the_fourth():
    nmed = 1.33
    legacy = ph.rayleigh_debye_intensity(20e-9, LAM, nmed, 1.39 * nmed, THETA, 0.1, legacy=True)
    fixed = ph.rayleigh_debye_intensity(20e-9, LAM, nmed, 1.39 * nmed, THETA, 0.1)
    np.testing.assert_allclose(fixed / legacy, nmed ** 4, rtol=5e-3)


def test_legacy_rayleigh_debye_uses_unsquared_form_factor():
    """For a large particle the legacy curve is |amplitude|, the corrected one amplitude^2."""
    theta = np.array([0.3])
    u = 2 * ph.wavenumber(LAM) * 2.5e-6 * np.sin(theta / 2)
    amplitude = ph.sphere_form_amplitude(u)
    legacy = ph.rayleigh_debye_intensity(5e-6, LAM, 1.0, 1.39, theta, 0.1, legacy=True)
    fixed = ph.rayleigh_debye_intensity(5e-6, LAM, 1.0, 1.39, theta, 0.1)
    np.testing.assert_allclose(fixed / legacy, np.abs(amplitude), rtol=1e-9)


# ------------------------------------------------------------ SEALS map and lineshape
def _seals(lam, **override):
    p = {**ph.SEALS_DEFAULTS, **override}
    return ph.seals_wavelength_to_angle(p["groove_spacing"], p["grating_distance"], p["grating_angle"],
                                        p["dcorr"], p["pupil_diameter"], p["numerical_aperture"], lam)


def test_seals_map_matches_paper_displacement_and_is_monotonic():
    lam = np.linspace(1580e-9, 1600e-9, 500)
    y, theta = _seals(lam)
    assert y[-1] == 0.0
    assert y[0] == pytest.approx(5.2275e-3, abs=1e-6)        # paper Fig. 2: about 5.3 mm
    assert np.all(np.diff(theta) < 0)
    assert np.degrees(theta[0]) == pytest.approx(36.50, abs=0.01)
    assert np.degrees(theta[-1]) == pytest.approx(-45.76, abs=0.01)


def test_seals_map_rejects_evanescent_order_and_bad_input():
    lam = np.linspace(1580e-9, 1600e-9, 50)
    with pytest.raises(ValueError, match="no solution"):
        _seals(lam, grating_angle=0.1)
    with pytest.raises(ValueError, match="increasing"):
        _seals(lam[::-1])
    with pytest.raises(ValueError, match="numerical_aperture"):
        _seals(lam, numerical_aperture=1.2)


def test_seals_resolution_formulas():
    assert np.degrees(ph.seals_min_resolvable_angle(100e-6, 5.73e-3)) == pytest.approx(1.0, abs=0.01)
    assert ph.seals_wavelength_resolution(1590e-9, 9.0909e-7, 1.5e-3) == pytest.approx(0.48e-9, rel=0.01)


def test_lineshape_peaks_at_centre_and_legacy_is_reversed():
    lam = np.linspace(1580e-9, 1600e-9, 501)
    shape = ph.laser_lineshape(lam, 1590e-9, 20e-9)
    assert lam[np.argmax(shape)] == pytest.approx(1590e-9, abs=0.05e-9)
    assert shape.max() == 1.0
    assert shape[0] == pytest.approx(0.5, abs=0.01)           # half a bandwidth from centre: half power
    off_centre = ph.laser_lineshape(lam, 1585e-9, 5e-9)
    legacy = ph.laser_lineshape(lam, 1585e-9, 5e-9, legacy=True)
    assert lam[np.argmax(off_centre)] == pytest.approx(1585e-9, abs=0.05e-9)
    assert lam[np.argmax(legacy)] == pytest.approx(1595e-9, abs=0.1e-9)   # mirrored


# ------------------------------------------------------------ motion
def test_stokes_einstein_round_trip_and_scaling():
    d = ph.stokes_einstein_diffusion(5e-6)
    assert d == pytest.approx(8.575e-14, rel=1e-3)
    assert ph.diameter_from_diffusion(d) == pytest.approx(5e-6)
    assert ph.stokes_einstein_diffusion(1e-6) == pytest.approx(5 * d)       # D ~ 1/dia


def test_dls_decay_rate_is_d_q_squared():
    q = ph.scattering_vector(LAM, 1.33, np.pi / 2)
    assert q == pytest.approx(4 * np.pi * 1.33 / LAM * np.sin(np.pi / 4))
    assert ph.dls_decay_rate(5e-6, LAM, 1.33, np.pi / 2) == pytest.approx(
        ph.stokes_einstein_diffusion(5e-6) * q ** 2)


# ------------------------------------------------------------ MATLAB parity (optional)
def test_python_matches_matlab_output_if_available():
    """Compares with results/matlab/parity_default.csv written by run_scarlett_plots.m."""
    folder = Path(__file__).resolve().parents[1] / "results" / "matlab"
    if not (folder / "parity_default.csv").exists():
        pytest.skip("run matlab/visualization/run_scarlett_plots.m first to create the CSV")
    table = np.genfromtxt(folder / "parity_default.csv", delimiter=",", names=True)
    scalars = np.genfromtxt(folder / "parity_scalars.csv", delimiter=",", names=True)
    p = ph.SEALS_DEFAULTS
    lam = np.linspace(p["lam_min"], p["lam_max"], p["n_wavelengths"])
    lam0 = 0.5 * (p["lam_min"] + p["lam_max"])
    y, theta = _seals(lam)
    theta = theta + p["measurement_angle"]
    mie = ph.mie_scattering(p["npar"], p["nmed"], p["dia"], lam0, theta, p["r"])
    np.testing.assert_allclose(theta, table["theta_rad"], rtol=1e-12)
    np.testing.assert_allclose(y, table["beam_displacement_m"], rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(mie.i_par, table["I_par"], rtol=1e-8)
    np.testing.assert_allclose(mie.i_perp, table["I_perp"], rtol=1e-8)
    np.testing.assert_allclose(np.exp(1j * mie.phase_par), np.exp(1j * table["phase_par"]), atol=1e-8)
    np.testing.assert_allclose(
        ph.rayleigh_debye_intensity(p["dia"], lam0, p["nmed"], p["npar"], theta, p["r"]),
        table["I_rayleigh_debye"], rtol=1e-8)
    np.testing.assert_allclose(ph.laser_lineshape(lam, lam0, p["source_bandwidth"]),
                               table["lineshape"], rtol=1e-10)
    assert mie.sigma_sca == pytest.approx(float(scalars["sigma_sca_m2"]), rel=1e-10)
    assert mie.x == pytest.approx(float(scalars["x"]), rel=1e-12)
