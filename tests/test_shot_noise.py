"""Shot-noise study: the two regimes described in dgs/shot_noise.py."""
import numpy as np
import pytest

from dgs import shot_noise as S


@pytest.fixture(scope="module")
def ctx():
    return S.setup()


def test_exact_probabilities_recover_phase(ctx):
    assert S.trial(None, 0, ctx) < 1e-3


def test_few_shots_mostly_fail(ctx):
    errs = [S.trial(300, k, ctx) for k in range(4)]
    assert np.mean(np.array(errs) < S.SUCCESS_RAD) <= 0.25


def test_many_shots_error_follows_shot_noise_law(ctx):
    lo = np.median([S.trial(300_000, k, ctx) for k in range(3)])
    hi = np.median([S.trial(3_000_000, k, ctx) for k in range(3)])
    assert lo < S.SUCCESS_RAD and hi < S.SUCCESS_RAD
    slope, _ = S.fit_slope([3e5, 3e6], [lo, hi])
    assert -0.75 < slope < -0.3          # theory: -0.5


def test_fit_and_inverse_round_trip():
    shots = np.array([1e4, 1e5, 1e6])
    slope, pref = S.fit_slope(shots, 2.0 * shots**-0.5)
    assert slope == pytest.approx(-0.5) and pref == pytest.approx(2.0)
    assert S.shots_needed(2.0 * 1e5**-0.5, slope, pref) == pytest.approx(1e5)
