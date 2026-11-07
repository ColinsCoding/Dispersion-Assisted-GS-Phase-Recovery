import numpy as np
import pytest
from dgs import real_data, gs_core

FILES = real_data.list_files()


def test_six_sample_files_present():
    assert len(FILES) == 6


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_load_fields(path):
    d = real_data.load(path)
    assert d["I1"].shape == d["I2"].shape == (4096,)
    assert np.all(d["I1"] >= 0) and np.all(d["I2"] >= 0)
    assert d["fs_GSas"] == 56.0 and (abs(d["D1"]), abs(d["D2"])) == (600.0, 1200.0)
    assert np.isclose(d["t"][1] - d["t"][0], 1 / 56.0)     # rebuilt; the file's t is all zeros
    assert "Synthetic" in d["description"]                  # not lab measurements


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_gs_converges_on_the_honest_arm(path):
    _, E, hist = real_data.retrieve(path, n_iter=500)
    assert hist[0] > 0.3                   # the starting guess is far off
    assert hist[-1] < 0.08                 # 2-7 % after 500 iterations
    assert hist[-1] < hist[9] < hist[0]    # and it keeps improving
    assert np.all(np.isfinite(E))


def test_builtin_error_metric_cannot_see_failure():
    """gs_core.retrieve_phase reports ~0 error after ONE iteration, while the
    other arm is still ~50 % off. Kept as a test so nobody trusts that metric."""
    d = real_data.load(FILES[0])
    _, err = gs_core.retrieve_phase(d["I1"], d["I2"], d["D1"], d["D2"], n_iter=1, unit_amplitude=False)
    assert err[0] / np.sqrt(np.mean(d["I2"] ** 2)) < 1e-10
    _, _, hist = real_data.retrieve(FILES[0], n_iter=1)
    assert hist[0] > 0.3


def test_unit_amplitude_mode_fails_on_these_signals():
    d = real_data.load(FILES[0])
    _, err = gs_core.retrieve_phase(d["I1"], d["I2"], d["D1"], d["D2"], n_iter=100, unit_amplitude=True)
    assert err[-1] / np.sqrt(np.mean(d["I2"] ** 2)) > 0.1
