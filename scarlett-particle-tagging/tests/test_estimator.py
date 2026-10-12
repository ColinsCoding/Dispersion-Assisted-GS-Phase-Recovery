"""Estimator tests on synthetic data: diameter, concentration, intervals, motion."""
import numpy as np
import pytest

from conftest import scan_volts
from scarlett import estimator as es
from scarlett import synthetic as sy


@pytest.mark.parametrize("dia", [5.0e-6, 5.013e-6, 4.871e-6])
def test_diameter_and_concentration_recovered_within_interval(dia, bench, sensor, lut):
    truth = sy.SampleTruth(dia=dia)
    scan, _, volts, sigma = scan_volts(truth, bench, sensor, seed=3,
                                       ripple=sy.channel_ripple(bench.n_channels))
    est = es.estimate_particle(volts, sigma, lut, bench, sensor, channel_ok=~scan["clipped"])
    assert est.refined and not est.ambiguous
    assert est.dia_ci[0] <= dia <= est.dia_ci[1]
    assert est.dia == pytest.approx(dia, abs=15e-9)                    # 0.3 %
    assert est.dia_ci[1] - est.dia_ci[0] < 40e-9
    assert est.concentration == pytest.approx(truth.concentration, rel=0.03)
    assert est.concentration_ci[0] < est.concentration < est.concentration_ci[1]
    assert est.fit_probability > 1e-4


def test_grid_only_estimate_is_limited_by_the_grid_step(bench, sensor, lut):
    truth = sy.SampleTruth(dia=5.013e-6)
    _, _, volts, sigma = scan_volts(truth, bench, sensor, seed=3)
    est = es.estimate_particle(volts, sigma, lut, bench, sensor=None)
    assert not est.refined
    assert est.dia in lut.dia and abs(est.dia - truth.dia) <= 20e-9
    assert est.dia_ci[1] - est.dia_ci[0] >= 20e-9 - 1e-15            # never narrower than one step


def test_interval_width_matches_cramer_rao(bench, sensor, lut):
    """A 95 % interval should be about 2 * 1.96 sigma of the Cramer-Rao bound."""
    truth = sy.SampleTruth()
    _, _, volts, sigma = scan_volts(truth, bench, sensor, seed=5)
    est = es.estimate_particle(volts, sigma, lut, bench, sensor)
    crb = es.diameter_cramer_rao(truth.dia, truth.npar, 20, sigma, bench, sensor)
    assert 1e-9 < crb < 10e-9
    assert (est.dia_ci[1] - est.dia_ci[0]) == pytest.approx(2 * 1.96 * crb, rel=0.35)


def test_scatter_of_estimates_matches_cramer_rao(bench, sensor, lut):
    truth = sy.SampleTruth()
    errors = []
    for seed in range(12):
        _, _, volts, sigma = scan_volts(truth, bench, sensor, seed=100 + seed,
                                        ripple=sy.channel_ripple(bench.n_channels, seed=seed))
        errors.append(es.estimate_particle(volts, sigma, lut, bench, sensor).dia - truth.dia)
    crb = es.diameter_cramer_rao(truth.dia, truth.npar, 20, sigma, bench, sensor)
    assert np.std(errors) == pytest.approx(crb, rel=0.6)             # 12 trials: loose
    assert abs(np.mean(errors)) < 3 * crb


def test_more_noise_gives_a_wider_interval(bench, sensor, lut):
    truth = sy.SampleTruth()
    _, _, volts, sigma = scan_volts(truth, bench, sensor, seed=7)
    narrow = es.estimate_particle(volts, sigma, lut, bench, sensor)
    wide = es.estimate_particle(volts, 3 * sigma, lut, bench, sensor)
    assert (wide.dia_ci[1] - wide.dia_ci[0]) > 2 * (narrow.dia_ci[1] - narrow.dia_ci[0])


def test_wrong_model_is_flagged_by_fit_probability(bench, sensor, lut):
    """A particle with a different refractive index does not fit the index-1.57 table."""
    truth = sy.SampleTruth(npar=1.45)
    _, _, volts, sigma = scan_volts(truth, bench, sensor, seed=9)
    est = es.estimate_particle(volts, sigma, lut, bench, sensor)
    assert est.fit_probability < 1e-6


def test_joint_fit_recovers_refractive_index(bench, sensor):
    lut2 = es.build_lookup_table(np.arange(4.8e-6, 5.2e-6 + 1e-12, 40e-9), [1.55, 1.57, 1.59], bench, sensor)
    truth = sy.SampleTruth()
    _, _, volts, sigma = scan_volts(truth, bench, sensor, seed=11)
    est = es.estimate_particle(volts, sigma, lut2, bench, sensor)
    assert est.npar == 1.57 and est.npar_ci[0] < 1.57 < est.npar_ci[1]
    assert est.dia == pytest.approx(truth.dia, abs=15e-9)
    assert est.dof == int(bench.usable_channels().sum()) - 3


def test_clipped_channels_are_dropped(bench, sensor, lut):
    truth = sy.SampleTruth()
    _, _, volts, sigma = scan_volts(truth, bench, sensor, seed=13)
    corrupted = volts.copy()
    corrupted[10] = 10 * volts[10]                                    # a saturated channel
    ok = np.ones(volts.size, dtype=bool)
    ok[10] = False
    assert es.estimate_particle(corrupted, sigma, lut, bench, sensor).fit_probability < 1e-6
    good = es.estimate_particle(corrupted, sigma, lut, bench, sensor, channel_ok=ok)
    assert good.dia == pytest.approx(truth.dia, abs=15e-9) and good.fit_probability > 1e-4


def test_estimator_rejects_bad_input(bench, sensor, lut):
    volts = np.ones(bench.n_channels)
    with pytest.raises(ValueError, match="channels"):
        es.estimate_particle(volts[:-1], 1.0, lut, bench)
    with pytest.raises(ValueError, match="sigma must be positive"):
        es.estimate_particle(volts, 0.0, lut, bench)
    with pytest.raises(ValueError, match="confidence"):
        es.estimate_particle(volts, 1.0, lut, bench, confidence=1.5)
    with pytest.raises(ValueError, match="fewer than 3"):
        es.estimate_particle(volts, 1.0, lut, bench, channel_ok=np.zeros(volts.size, dtype=bool))
    with pytest.raises(ValueError, match="positive"):
        es.build_lookup_table([-1e-6], [1.57], bench, sensor)


def test_combined_sigma():
    np.testing.assert_allclose(es.combined_sigma([3.0], [0.04], 0.01), [0.05])
    with pytest.raises(ValueError):
        es.combined_sigma([1.0], [0.1], -0.1)


# ------------------------------------------------------------ motion
def test_autocorrelation_of_constant_signal_is_one():
    np.testing.assert_allclose(es.intensity_autocorrelation(np.full(100, 7.0), 10), 1.0)


def test_decay_fit_is_exact_on_an_ideal_curve():
    dt, gamma, beta = 1e-3, 12.0, 0.8
    g2 = 1 + beta * np.exp(-2 * gamma * dt * np.arange(200))
    fitted_gamma, fitted_beta = es.fit_decay_rate(g2, dt)
    assert fitted_gamma == pytest.approx(gamma, rel=1e-9)
    assert fitted_beta == pytest.approx(beta, rel=1e-9)


def test_motion_estimate_recovers_diffusion_within_interval(bench):
    truth = sy.SampleTruth()
    counts, gamma = sy.generate_dls_trace(truth, bench, 120_000, 5e-3, rng=np.random.default_rng(2))
    motion = es.estimate_motion(counts, 5e-3, bench.lam_center, bench.nmed, np.pi / 2, max_lag=120)
    assert motion["decay_rate"] == pytest.approx(gamma, rel=0.2)
    assert motion["hydrodynamic_dia"] == pytest.approx(truth.dia, rel=0.2)
    lo, hi = motion["hydrodynamic_dia_ci"]
    assert lo < motion["hydrodynamic_dia"] < hi and (hi - lo) / truth.dia < 0.6


def test_motion_estimator_rejects_bad_input():
    with pytest.raises(ValueError, match="max_lag"):
        es.intensity_autocorrelation(np.ones(10), 10)
    with pytest.raises(ValueError, match="nothing to fit"):
        es.fit_decay_rate(np.ones(20), 1e-3)
    with pytest.raises(ValueError, match="too short"):
        es.estimate_motion(np.ones(100), 1e-3, 1590e-9, 1.33, 1.0, max_lag=50)
    with pytest.raises(ValueError, match="n_segments"):
        es.estimate_motion(np.ones(1000), 1e-3, 1590e-9, 1.33, 1.0, max_lag=5, n_segments=2)
