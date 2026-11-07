"""The recursive TDGSA must return exactly what the iterative one does -- it is the
same fold, written as a tail recursion."""
import numpy as np

from dgs.gs_core import make_qpsk_measurements, retrieve_phase
from dgs.gs_recursive import retrieve_phase_recursive


def test_recursive_matches_iterative():
    d = make_qpsk_measurements(n_symbols=64, snr_db=30.0)
    phi_i, err_i = retrieve_phase(d["I1"], d["I2"], d["D1"], d["D2"], n_iter=40)
    phi_r, err_r = retrieve_phase_recursive(d["I1"], d["I2"], d["D1"], d["D2"], n_iter=40)
    assert np.allclose(phi_i, phi_r, atol=1e-9)          # identical phase
    assert np.allclose(err_i, err_r, atol=1e-9)          # identical convergence trace


def test_shapes_and_error_count():
    d = make_qpsk_measurements(n_symbols=32, snr_db=30.0)
    phi, err = retrieve_phase_recursive(d["I1"], d["I2"], d["D1"], d["D2"], n_iter=15)
    assert phi.shape == d["phi_true"].shape
    assert len(err) == 15                                # one recorded error per recursion level


def test_return_errors_false():
    d = make_qpsk_measurements(n_symbols=32, snr_db=30.0)
    phi = retrieve_phase_recursive(d["I1"], d["I2"], d["D1"], d["D2"],
                                   n_iter=15, return_errors=False)
    assert phi.shape == d["phi_true"].shape
