"""Tests for dgs/hilbert_space.py."""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.hilbert_space import (
    inner_product, norm_L2, normalize, gram_schmidt_L2,
    triangle_inequality_check, qubit_state, expectation_value,
    commutator_matrix, heisenberg_uncertainty, time_bandwidth_product,
    projection_operator, spectral_decomposition, density_matrix,
    von_neumann_entropy, harmonic_oscillator_states, hilbert_space_sympy_5,
)


def test_inner_product_discrete_real():
    u = np.array([1., 0., 0.])
    v = np.array([0., 1., 0.])
    assert abs(inner_product(u, v)) < 1e-14


def test_inner_product_discrete_self():
    u = np.array([1., 2., 3.])
    ip = inner_product(u, u)
    assert abs(ip - 14.0) < 1e-10


def test_inner_product_L2():
    x = np.linspace(0, 2*np.pi, 1000)
    # <sin|cos> = 0
    ip = inner_product(np.sin(x), np.cos(x), x)
    assert abs(ip) < 0.01


def test_inner_product_L2_norm():
    x = np.linspace(0, 2*np.pi, 1000)
    # ||sin||² = pi
    ip = inner_product(np.sin(x), np.sin(x), x)
    assert abs(ip - np.pi) < 0.01


def test_norm_L2_unit():
    x = np.linspace(0, 1, 1000)
    f = np.ones_like(x)
    assert abs(norm_L2(f, x) - 1.0) < 1e-4


def test_normalize_unit_norm():
    x = np.linspace(0, np.pi, 500)
    f = np.sin(x) * 5.7   # arbitrary scale
    fn = normalize(f, x)
    assert abs(norm_L2(fn, x) - 1.0) < 1e-6


def test_gram_schmidt_orthonormal():
    x = np.linspace(0, 1, 500)
    fs = [np.ones_like(x), x, x**2, x**3]
    basis = gram_schmidt_L2(fs, x)
    for i, phi_i in enumerate(basis):
        for j, phi_j in enumerate(basis):
            ip = float(np.real(inner_product(phi_i, phi_j, x)))
            expected = 1.0 if i == j else 0.0
            assert abs(ip - expected) < 1e-4


def test_triangle_inequality():
    x = np.linspace(0, 2*np.pi, 500)
    res = triangle_inequality_check(np.sin(x), np.cos(x), x)
    assert res["triangle_holds"] is True
    assert res["cauchy_schwarz"] is True


def test_qubit_north_pole():
    psi = qubit_state(0, 0)   # |0>
    assert abs(psi[0] - 1.0) < 1e-10
    assert abs(psi[1]) < 1e-10


def test_qubit_south_pole():
    psi = qubit_state(np.pi, 0)   # |1>
    assert abs(psi[0]) < 1e-10
    assert abs(psi[1] - 1.0) < 1e-10


def test_qubit_normalized():
    for theta in [0, np.pi/4, np.pi/2, np.pi]:
        psi = qubit_state(theta, 0.5)
        assert abs(np.dot(np.conj(psi), psi) - 1.0) < 1e-10


def test_expectation_value_pauli_z():
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    psi_0 = np.array([1., 0.])   # |0> -> <Z> = +1
    psi_1 = np.array([0., 1.])   # |1> -> <Z> = -1
    assert abs(expectation_value(psi_0, sigma_z) - 1.0) < 1e-10
    assert abs(expectation_value(psi_1, sigma_z) + 1.0) < 1e-10


def test_commutator_pauli_xy():
    # [sigma_x, sigma_y] = 2i*sigma_z
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    comm = commutator_matrix(sx, sy)
    np.testing.assert_allclose(comm, 2j * sz, atol=1e-10)


def test_heisenberg_uncertainty_gaussian():
    x = np.linspace(-10, 10, 2048)
    psi = np.exp(-x**2 / 4)   # sigma_x=1 Gaussian
    res = heisenberg_uncertainty(psi, x)
    assert res["uncertainty_satisfied"] is True
    assert res["ratio"] >= 0.99   # must satisfy hbar/2


def test_time_bandwidth_gaussian():
    t = np.linspace(-100e-15, 100e-15, 4096)
    T0 = 10e-15
    E_t = np.exp(-t**2 / (2*T0**2))
    res = time_bandwidth_product(t, E_t)
    assert res["transform_limited"] is True
    assert abs(res["TBP"] - 0.5) < 0.1


def test_projection_idempotent():
    phi = np.array([1., 1., 0.]) / np.sqrt(2)
    P = projection_operator(phi)
    np.testing.assert_allclose(P @ P, P, atol=1e-10)


def test_projection_hermitian():
    phi = np.array([1., 0., 1j]) / np.sqrt(2)
    P = projection_operator(phi)
    np.testing.assert_allclose(P, P.conj().T, atol=1e-10)


def test_spectral_decomposition_pauli_x():
    sx = np.array([[0,1],[1,0]], dtype=complex)
    res = spectral_decomposition(sx)
    assert set(np.round(np.real(res["eigenvalues"]), 6)) == {-1.0, 1.0}
    assert res["recon_error"] < 1e-12
    assert res["hermitian"] is True


def test_spectral_decomposition_recon():
    A = np.array([[3., 1.], [1., 2.]], dtype=complex)
    res = spectral_decomposition(A)
    assert res["recon_error"] < 1e-10


def test_density_matrix_pure():
    psi = np.array([1., 0.])
    rho = density_matrix(psi)
    assert abs(np.trace(rho) - 1.0) < 1e-10
    np.testing.assert_allclose(rho @ rho, rho, atol=1e-10)   # pure: rho^2 = rho


def test_von_neumann_entropy_pure():
    psi = np.array([1., 0.])
    rho = density_matrix(psi)
    S = von_neumann_entropy(rho)
    assert abs(S) < 1e-10   # pure state: S=0


def test_von_neumann_entropy_mixed():
    rho = np.eye(2) / 2   # maximally mixed qubit
    S = von_neumann_entropy(rho)
    assert abs(S - np.log(2)) < 1e-6   # S = ln(2) for qubit


def test_qho_energies():
    res = harmonic_oscillator_states(5)
    assert res["energies"] == [0.5, 1.5, 2.5, 3.5, 4.5]


def test_qho_orthonormal():
    res = harmonic_oscillator_states(4)
    assert res["orthonormal"] is True


def test_hilbert_space_sympy_5():
    eqs = hilbert_space_sympy_5()
    assert len(eqs) == 5
    for k, v in eqs.items():
        assert isinstance(v, sp.Basic)
