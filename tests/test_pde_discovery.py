"""Tests for dgs/pde_discovery.py."""
import json

import numpy as np
import pytest

from dgs import pde_discovery as P


def test_exact_data_satisfy_their_pdes():
    for d in (P.heat(), P.burgers()):
        D = P.derivatives(d)
        rhs = sum(c * P.standard_library(D)[k] for k, c in d['truth'].items())
        assert np.abs(D['u_t'] - rhs).max() < 1e-3 * np.abs(D['u_t']).max()
    H = P.hermite_functions(4, np.linspace(-8, 8, 2001)); dx = 16 / 2000
    assert np.allclose(H @ H.T * dx, np.eye(5), atol=1e-6)            # orthonormal eigenstates


@pytest.mark.parametrize('name,data,lib', [
    ('heat', P.heat(), 'standard'), ('burgers', P.burgers(), 'standard'),
    ('schrodinger', P.schrodinger(), 'standard'), ('black_scholes', P.black_scholes(), 'finance')])
def test_sparse_regression_recovers_clean_pdes(name, data, lib):
    found = P.discover(data, lib)
    err = P.coefficient_error(found, data['truth'])
    assert err['exact_terms'] and err['max_rel_err'] < 5e-3, (name, found)


def test_discovered_schrodinger_is_unitary():
    found = P.discover(P.schrodinger())
    assert all(abs(c.real) < 1e-3 * abs(c) for c in found.values())   # purely imaginary: anti-Hermitian generator


def test_finite_differences_fail_with_noise():
    d = P.burgers()
    err = P.coefficient_error(P.discover(P.add_noise(d, 0.01)), d['truth'])
    assert not err['exact_terms'] and err['max_rel_err'] > 0.5


def test_parsimony_path_prefers_fewer_terms():
    d = P.derivatives(P.burgers())
    lib = P.standard_library(d)
    chosen, path = P.stridge_path(lib, d['u_t'])
    assert set(chosen) == {'u*u_x', 'u_xx'} and min(p[1] for p in path if p[1]) <= 2


def test_two_networks_recover_burgers_from_noisy_data():
    pytest.importorskip('torch')
    d = P.burgers(nx=128, nt=51)
    m = P.HiddenPhysics(width=32, depth=3, dev=P.device(prefer_gpu=False)).fit(
        P.add_noise(d, 0.01), steps=1500, n_points=2500, lr=3e-3)
    found = m.discover(d)
    assert set(found) == {'u*u_x', 'u_xx'}
    assert abs(found['u*u_x'] + 1) < 0.1 and abs(found['u_xx'] - 0.1) < 0.02
    assert m.black_box_residual(d) < 0.15


def test_kaggle_package(tmp_path):
    pytest.importorskip('pandas')
    out = P.kaggle_package(tmp_path / 'kg', {'heat': P.heat(nx=32, nt=11), 'schrodinger': P.schrodinger(nx=32, nt=11)})
    meta = json.loads((out / 'dataset-metadata.json').read_text())
    assert meta['id'].endswith('/pde-discovery-benchmark') and meta['licenses'][0]['name'] == 'CC0-1.0'
    import pandas as pd
    assert list(pd.read_csv(out / 'schrodinger.csv').columns) == ['x', 't', 'u_real', 'u_imag']
    assert len(pd.read_csv(out / 'heat.csv')) == 32 * 11


def test_validation():
    with pytest.raises(ValueError):
        P.heat(nu=0)
    with pytest.raises(ValueError):
        P.burgers(a0=0.5)
    with pytest.raises(ValueError):
        P.black_scholes(sigma=0)
