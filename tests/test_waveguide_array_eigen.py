"""Tests for dgs/waveguide_array_eigen.py."""
import numpy as np
import pytest
import sympy as sp

from dgs.waveguide_array_eigen import (
    coupling_matrix, propagate_eig, propagate_expm, two_guide_symbolic,
    uniform_array_eigen_check, uniform_array_eigen_numeric, bessel_identity_check,
    discrete_diffraction, rms_width_infinite, pst_couplings, pst_eigenvalues_symbolic,
    transfer_fidelity, coincidence_probability_symbolic, coincidence_distinguishable_symbolic,
    mc_coincidences,
)


def test_two_guides():
    d = two_guide_symbolic()
    k, z = sp.symbols("kappa z", positive=True)
    assert sp.simplify(d["P2"] - sp.sin(k * z) ** 2) == 0
    vals = sorted(str(e[0]) for e in d["eigenvects"])
    assert vals == ["beta + kappa", "beta - kappa"]


def test_uniform_array_eigenvectors():
    assert uniform_array_eigen_check() == 0
    for N in (2, 3, 5, 10):
        lam, V = uniform_array_eigen_numeric(N, kappa=0.7)
        l2, V2 = np.linalg.eigh(coupling_matrix(N, 0.7))
        assert np.allclose(lam, l2)
        assert np.allclose(np.abs(V.T @ V2), np.eye(N), atol=1e-10)


def test_two_propagators_agree_and_conserve_power():
    H = coupling_matrix(7, couplings=[1, 0.5, 2, 1, 0.3, 1.1])
    a0 = np.zeros(7); a0[2] = 1
    for z in (0.3, 2.0, 9.0):
        a = propagate_eig(H, a0, z)
        assert np.allclose(a, propagate_expm(H, a0, z))
        assert np.sum(np.abs(a) ** 2) == pytest.approx(1.0)


def test_discrete_diffraction_bessel():
    assert bessel_identity_check() == 0
    N, kz = 201, 5.0
    H = coupling_matrix(N); a0 = np.zeros(N); a0[100] = 1
    P = np.abs(propagate_eig(H, a0, kz)) ** 2
    n = np.arange(N) - 100
    assert np.allclose(P, discrete_diffraction(n, kz), atol=1e-12)
    assert np.sqrt(np.sum(n ** 2 * P)) == pytest.approx(rms_width_infinite(kz))


def test_perfect_state_transfer():
    assert pst_eigenvalues_symbolic(6) == [-5, -3, -1, 1, 3, 5]
    for N in (4, 7, 10):
        assert transfer_fidelity(N, np.pi / 2, pst_couplings(N)) == pytest.approx(1.0, abs=1e-12)
    # a uniform array of 6 never fully transfers
    zs = np.linspace(0, 30, 6001)
    assert max(transfer_fidelity(6, z) for z in zs) < 0.95


def test_hong_ou_mandel():
    P, k, z = coincidence_probability_symbolic()
    assert sp.simplify(P - sp.cos(2 * k * z) ** 2) == 0
    Pd, _, _ = coincidence_distinguishable_symbolic()
    assert float(Pd.subs(k * z, sp.pi / 4).subs({k: 1, z: sp.pi / 4})) == pytest.approx(0.5)
    mc, p = mc_coincidences(np.pi / 4)
    assert mc == 0.0 and p == pytest.approx(0.0, abs=1e-15)


def test_validation():
    with pytest.raises(ValueError):
        coupling_matrix(1)
    with pytest.raises(ValueError):
        coupling_matrix(4, couplings=[1, 2])
