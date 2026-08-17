import numpy as np
import pytest
from dgs.fisher_information_curvature import (
    log_likelihood_curvature, fisher_information_numeric, gaussian_fisher_information,
    cramer_rao_bound, dirac_delta_fisher_information_limit, bayesian_concentration_demo,
)


def test_log_likelihood_curvature_of_gaussian_is_constant_negative():
    theta = np.linspace(-5, 5, 2001)
    sigma = 2.0
    log_L = -0.5 * (theta / sigma) ** 2
    curvature = log_likelihood_curvature(log_L, theta)
    interior = curvature[50:-50]
    assert np.allclose(interior, -1.0 / sigma ** 2, atol=1e-3)


def test_log_likelihood_curvature_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        log_likelihood_curvature(np.zeros(10), np.zeros(20))


def test_log_likelihood_curvature_rejects_too_few_points():
    with pytest.raises(ValueError):
        log_likelihood_curvature(np.zeros(3), np.zeros(3))


def test_fisher_information_numeric_matches_analytic_gaussian():
    theta = np.linspace(-6, 6, 3001)
    sigma = 1.5
    log_L = -0.5 * (theta / sigma) ** 2
    I_numeric = fisher_information_numeric(log_L, theta, theta_hat=0.0)
    assert I_numeric == pytest.approx(gaussian_fisher_information(sigma), rel=1e-2)


def test_gaussian_fisher_information_narrower_means_more_information():
    assert gaussian_fisher_information(0.5) > gaussian_fisher_information(2.0)


def test_gaussian_fisher_information_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        gaussian_fisher_information(0.0)


def test_cramer_rao_bound_is_inverse_of_fisher_information():
    assert cramer_rao_bound(4.0) == pytest.approx(0.25)


def test_cramer_rao_bound_rejects_nonpositive():
    with pytest.raises(ValueError):
        cramer_rao_bound(0.0)


def test_dirac_delta_fisher_information_limit_diverges_as_sigma_shrinks():
    result = dirac_delta_fisher_information_limit(np.array([1.0, 0.1, 0.01]))
    fi = result["fisher_information"]
    assert fi[0] < fi[1] < fi[2]
    np.testing.assert_allclose(fi, 1.0 / result["sigma_values"] ** 2)


def test_dirac_delta_fisher_information_limit_peak_height_grows_too():
    result = dirac_delta_fisher_information_limit(np.array([1.0, 0.1, 0.01]))
    peaks = result["delta_peak_height"]
    assert peaks[0] < peaks[1] < peaks[2]


def test_dirac_delta_fisher_information_limit_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        dirac_delta_fisher_information_limit(np.array([1.0, -0.5]))


def test_bayesian_concentration_precision_grows_with_n():
    demo = bayesian_concentration_demo(n_values=[1, 10, 100], sigma=2.0, sigma0=10.0)
    prec = demo["posterior_precision"]
    assert prec[0] < prec[1] < prec[2]


def test_bayesian_concentration_precision_matches_closed_form():
    demo = bayesian_concentration_demo(n_values=[1, 5, 50], sigma=2.0, sigma0=10.0)
    expected = 1.0 / 10.0 ** 2 + demo["n_values"] / 2.0 ** 2
    np.testing.assert_allclose(demo["posterior_precision"], expected)


def test_bayesian_concentration_posterior_std_shrinks_with_n():
    demo = bayesian_concentration_demo(n_values=[1, 10, 100])
    std = demo["posterior_std"]
    assert std[0] > std[1] > std[2]


def test_bayesian_concentration_rejects_bad_n_values():
    with pytest.raises(ValueError):
        bayesian_concentration_demo(n_values=[0, 5])


def test_bayesian_concentration_converges_toward_true_mean():
    demo = bayesian_concentration_demo(n_values=[1000], mu_true=3.0, rng_seed=1)
    assert abs(demo["posterior_mean"][0] - 3.0) < 0.2
