"""Tests for dgs.schrodinger_separation_industry."""
import numpy as np
import pytest
import sympy as sp

from dgs import schrodinger_separation_industry as s


@pytest.fixture(scope="module")
def sym():
    return s.sympy_separation()


def test_separated_solutions_have_zero_residual(sym):
    assert sym["box"] == 0 and sym["plane"] == 0 and sym["box3"] == 0
    assert sym["superposition"] == 0
    assert all(r == 0 for r in sym["qho"])
    assert sym["box_bc"] == (0, 0)


def test_wrong_energy_has_nonzero_residual(sym):
    assert sym["wrong_energy"] != 0


def test_time_factor_and_bohr_frequency(sym):
    S = sym["symbols"]
    E, t = sp.symbols("E"), S["t"]
    assert sp.simplify(sym["T_sol"] / sp.exp(-sp.I * E * t / S["hbar"])).free_symbols <= {sp.Symbol("C1")}
    assert sp.simplify(sym["bohr_omega"] - 3 * sp.pi ** 2 * S["hbar"] / (2 * S["m"] * S["L"] ** 2)) == 0


def test_fd_residual_converges_as_dx_squared():
    good = s.box_residual_convergence()
    slope = np.polyfit(np.log(good[:, 0]), np.log(good[:, 1]), 1)[0]
    assert 1.9 < slope < 2.1
    bad = s.box_residual_convergence(energy_scale=1.01)
    assert bad[-1, 1] > 1000 * good[-1, 1]
    assert np.isclose(bad[-1, 1], 0.01 * np.pi ** 2 / 2, rtol=0.01)


def test_fd_eigenpairs_and_qho():
    x = np.linspace(-10, 10, 1500)
    H = s.fd_hamiltonian(0.5 * x ** 2, x[1] - x[0])
    E, v = np.linalg.eigh(H)
    assert np.allclose(E[:4], [0.5, 1.5, 2.5, 3.5], atol=1e-3)
    assert np.all(s.eigen_residuals(H, E[:4], v[:, :4]) < 1e-10)


def test_quantum_well_gaas():
    E = s.box_energy_eV([1, 2], 10.0, 0.067)
    assert np.isclose(E[0], 0.0561, rtol=1e-2) and np.isclose(E[1] / E[0], 4)


def test_gate_oxide_leakage():
    T = s.gate_oxide_transmission(np.array([1.0, 1.1]))
    assert 3.0 < T[0] / T[1] < 4.0
    assert np.isclose(s.equivalent_oxide_thickness(3.2, 25.0), 0.4992)


def test_fowler_nordheim_constant():
    # textbook B = 6.83e9 sqrt(m*/m) phi^1.5 V/m
    assert np.isclose(s.fowler_nordheim_B(3.1, 0.5), 6.83e9 * np.sqrt(0.5) * 3.1 ** 1.5, rtol=1e-3)


def test_strong_field_numbers():
    assert np.isclose(s.ponderomotive_energy_eV(1e14, 800e-9), 9.33e-14 * 1e14 * 0.8 ** 2, rtol=0.01)
    assert np.isclose(s.keldysh_gamma(15.76, 1e14, 800e-9), 1.148, rtol=1e-2)
    # atomic-unit form 2 (2 Ip)^1.5 / (3 F) for hydrogen at 1 a.u. of field
    Ip_au, F_au = 0.5, 5.14220674763e11
    assert np.isclose(s.triangular_barrier_exponent(13.605693, F_au), 2 * (2 * Ip_au) ** 1.5 / 3, rtol=1e-4)


def test_transmon():
    EC = 0.25
    E, _, H = s.transmon_levels(50 * EC, EC)
    f01, alpha = E[1] - E[0], (E[2] - E[1]) - (E[1] - E[0])
    f01_a, alpha_a = s.transmon_asymptotic(50 * EC, EC)
    assert abs(f01 - f01_a) / f01_a < 0.01
    assert abs(alpha - alpha_a) / abs(alpha_a) < 0.2
    assert np.allclose(H, H.T)
    for ej in (20, 30, 50):
        num = abs(s.transmon_levels(ej, 1, 0.5)[0][0] - s.transmon_levels(ej, 1, 0.0)[0][0])
        assert abs(num / s.transmon_charge_dispersion(ej, 1) - 1) < 0.1


def test_trapped_ion():
    assert np.isclose(s.ion_ground_state_size(170.936, 1e6), 5.44e-9, rtol=1e-2)
    assert np.isclose(s.lamb_dicke(170.936, 1e6, 355e-9), 0.136, rtol=1e-2)


def test_morse_h2():
    G = s.morse_levels_cm([0, 1], 4401, 4.75)
    assert 4100 < G[1] - G[0] < 4200            # measured H2 fundamental ~4161 cm^-1
    fd, res = s.morse_fd_levels_cm(3, 4401, 4.75, 0.504)
    assert np.allclose(fd, s.morse_levels_cm([0, 1, 2], 4401, 4.75), rtol=1e-4)
    assert np.all(res < 1e-9)


def test_industry_table_functions_exist():
    for row in s.INDUSTRY:
        f = row["function"]
        if not f.startswith(("dgs.", "(")):
            assert callable(getattr(s, f))


@pytest.mark.parametrize("call", [
    lambda: s.fd_hamiltonian(np.zeros(2), 0.1),
    lambda: s.fd_hamiltonian(np.zeros(5), 0.0),
    lambda: s.box_energy_eV(1, -1.0),
    lambda: s.gate_oxide_transmission(1.0, barrier_eV=3.1, E_eV=4.0),
    lambda: s.triangular_barrier_exponent(3.1, 0.0),
    lambda: s.transmon_levels(-1.0, 1.0),
    lambda: s.transmon_levels(1.0, 1.0, n_charge=0),
    lambda: s.lamb_dicke(171, 1e6, 0.0),
    lambda: s.box_residual_convergence(level=0),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
