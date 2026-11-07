"""Tests for dgs.steam3d_qm_classifier."""
import numpy as np
import pytest

from dgs import steam3d_qm_classifier as c


# ---- 3D STEAM

def test_noise_free_scan_recovers_heights():
    d = c.steam3d_scan(n_lines=40, snr_db=300.0)
    n = d["R"].shape[1]
    core = slice(n // 10, -n // 10)                    # Fourier-filter ripple decays slowly from the band edges
    assert np.max(np.abs(d["R"][:, core] - d["H"][:, core])) < 1e-10      # < 1 angstrom in the central 80%
    est, se = c.step_estimate(d)
    assert abs(est - 3.1e-10) < 1e-12


def test_noisy_step_is_unbiased_within_errors():
    ests = [c.step_estimate(c.steam3d_scan(n_lines=200, seed=s)) for s in range(4)]
    for est, se in ests:
        assert abs(est - 3.1e-10) < 3 * se
    assert np.mean([e for e, _ in ests]) > 0                                   # step detected


def test_height_precision_formula():
    assert np.isclose(c.height_precision(1550e-9, 1e-3), 1550e-9 * 1e-3 / (4 * np.pi))
    assert np.isclose(c.height_precision(1550e-9, 1e-3, 100), c.height_precision(1550e-9, 1e-3) / 10)


# ---- angstrom / floating-point limit

def test_second_difference_has_optimal_step():
    hs = np.logspace(-1, -7, 25)
    err = c.second_difference_error(hs)
    h_best = hs[np.argmin(err)]
    assert 1e-5 < h_best < 1e-3
    assert np.isclose(c.optimal_step(), (48 * np.finfo(float).eps) ** 0.25)
    assert err[0] > err.min() * 100 and err[-1] > err.min() * 100            # both sides get worse


# ---- Schrodinger spectra

def test_box_and_harmonic_match_closed_forms():
    for p, mr in c.MASS.items():
        assert np.allclose(c.spectrum(p, "box"), c.box_exact([1, 2, 3, 4], 4.0, mr), rtol=1e-4)
        assert np.allclose(c.spectrum(p, "harmonic"), c.harmonic_exact([0, 1, 2, 3], 10.0, mr), rtol=1e-4)


def test_linear_well_matches_airy_zeros():
    for p, mr in c.MASS.items():
        assert np.allclose(c.spectrum(p, "linear"), c.linear_exact(4, 5.0, mr), rtol=1e-4)


def test_feature_ratios_identify_potential():
    f = {pot: c.features(c.spectrum("proton", pot)) for pot in c.POTENTIALS}
    assert np.allclose(f["box"][1:], [4, 9, 16], rtol=1e-3)
    assert np.allclose(f["harmonic"][1:], [3, 5, 7], rtol=1e-3)
    assert np.allclose(f["linear"][1:], [2.2954, 3.1883, 4.0125], rtol=1e-3)


def test_mass_scaling_exponents():
    for pot, expo in [("box", -1.0), ("harmonic", -0.5), ("linear", -1 / 3)]:
        r = c.spectrum("deuteron", pot)[0] / c.spectrum("proton", pot)[0]
        assert np.isclose(r, (c.MASS["deuteron"] / c.MASS["proton"]) ** expo, rtol=1e-3)


# ---- classifier

def test_classifier_accuracy_degrades_with_jitter():
    accs = []
    for jit in (0.05, 0.3):
        X, y = c.make_dataset(40, jitter=jit)
        accs.append(c.train_classifier(X, y)[3])
    assert accs[0] > 0.97
    assert accs[1] < accs[0]


def test_physics_rule_on_clean_data():
    X, y = c.make_dataset(5, jitter=0.0, meas_noise=0.0)
    assert np.mean([c.physics_rule(r) == lab for r, lab in zip(X, y)]) == 1.0


@pytest.mark.parametrize("call", [
    lambda: c.spectrum("tauon", "box"),
    lambda: c.spectrum("electron", "finite"),
    lambda: c.lowest_levels(np.zeros(3), 0.1, 1.0),
    lambda: c.lowest_levels(np.zeros(50), 0.0, 1.0),
    lambda: c.make_dataset(2, jitter=1.5),
    lambda: c.wavenumber_grid(lam_min=1600e-9, lam_max=1500e-9),
    lambda: c.steam3d_scan(n_lines=1),
    lambda: c.height_precision(1550e-9, 0.0),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
