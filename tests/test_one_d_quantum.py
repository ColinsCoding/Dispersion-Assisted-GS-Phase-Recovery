"""Tests for dgs/one_d_quantum.py: analytic models checked against the finite-difference solver."""
import math

import numpy as np
import pytest

from dgs import one_d_quantum as Q


def box_grid(L, N=1500):
    return np.linspace(0, L, N + 2)[1:-1]   # interior points; psi = 0 at the walls


# --- constants and rigid box --------------------------------------------------------------
def test_constants():
    assert abs(Q.HC - 1239.84198) < 1e-4
    assert abs(Q.C_NM - 0.0380998) < 1e-6


def test_box_ground_state_1nm_electron():
    # h^2 / (8 m L^2) = (hc)^2 / (8 mc^2 L^2)
    assert abs(Q.box_energy(1, 1.0) - Q.HC ** 2 / (8 * Q.M_E_C2)) < 1e-12
    assert abs(Q.box_energy(1, 1.0) - 0.37603) < 1e-4
    assert abs(Q.box_energy(3, 1.0) / Q.box_energy(1, 1.0) - 9) < 1e-12


def test_box_matches_finite_difference():
    L = 1.0
    x = box_grid(L)
    E, psi = Q.solve_schrodinger(x, 0.0, n_states=4)
    for n in range(1, 5):
        assert abs(E[n - 1] / Q.box_energy(n, L) - 1) < 1e-4
        assert np.allclose(psi[n - 1], Q.box_psi(x, n, L), atol=2e-3 * math.sqrt(2 / L))


def test_box_probability():
    L = 2.0
    assert abs(Q.box_probability(3, L, 0, L) - 1) < 1e-12
    assert abs(Q.box_probability(1, L, 0, L / 4) - (0.25 - 1 / (2 * math.pi))) < 1e-12
    # correspondence principle: quantum -> classical 1/4 as n grows
    errs = [abs(Q.box_probability(n, L, 0, L / 4) - 0.25) for n in (1, 11, 101, 1001)]
    assert errs == sorted(errs, reverse=True) and errs[-1] < 2e-4
    x = np.linspace(0, 0.7, 7001)
    assert abs(np.trapezoid(Q.box_psi(x, 2, L) ** 2, x) - Q.box_probability(2, L, 0, 0.7)) < 1e-6


# --- finite well --------------------------------------------------------------------------
def test_finite_well_matches_finite_difference():
    U0, L = 10.0, 0.5
    lv = Q.finite_well_levels(U0, L)
    dx = 1e-3
    x = (np.arange(4000) + 0.5) * dx - 2.0       # well edges fall between grid points
    U = np.where(np.abs(x) <= L / 2, 0.0, U0)
    E, _ = Q.solve_schrodinger(x, U, n_states=len(lv))
    assert np.all(lv < U0)
    assert np.allclose(E, lv, rtol=2e-4)
    # wave leaks into the walls, so each level sits below the rigid-box value
    assert all(lv[i] < Q.box_energy(i + 1, L) for i in range(len(lv)))


def test_finite_well_limits():
    assert len(Q.finite_well_levels(1e-4, 0.1)) == 1               # always one bound state
    deep = Q.finite_well_levels(1e5, 1.0)
    assert abs(deep[0] / Q.box_energy(1, 1.0) - 1) < 0.01         # deep well -> rigid box


def test_penetration_depth():
    eta = Q.penetration_depth(1.0, 2.0)
    assert abs(eta - math.sqrt(Q.C_NM)) < 1e-12
    with pytest.raises(ValueError):
        Q.penetration_depth(3.0, 2.0)


# --- harmonic oscillator ------------------------------------------------------------------
def test_oscillator_levels_and_states():
    hw = 0.5
    b = Q.ho_length(hw)
    x = np.linspace(-10 * b, 10 * b, 3001)
    U = 0.5 * hw * (x / b) ** 2            # (1/2) m w^2 x^2 = (hbar w / 2)(x/b)^2
    E, psi = Q.solve_schrodinger(x, U, n_states=5)
    dx = x[1] - x[0]
    for n in range(1, 6):
        assert abs(E[n - 1] - Q.ho_energy(n, hw)) < 1e-3 * hw
        exact = Q.ho_psi(x, n, hw)
        assert abs(np.sum(exact ** 2) * dx - 1) < 1e-6
        assert abs(abs(np.sum(exact * psi[n - 1]) * dx) - 1) < 1e-4
    assert np.allclose(np.diff(E), hw, rtol=1e-3)                  # equally spaced
    assert abs(np.sum(Q.ho_psi(x, 1, hw) * Q.ho_psi(x, 3, hw)) * dx) < 1e-10


def test_node_count_is_n_minus_1():
    x = box_grid(1.0, 800)
    _, psi = Q.solve_schrodinger(x, 0.0, n_states=6)
    assert [Q.count_nodes(p) for p in psi] == list(range(6))


def test_local_wavelength():
    lam = Q.local_wavelength(1.0, np.array([0.0, 2.0]))
    assert abs(lam[0] - Q.HC / math.sqrt(2 * Q.M_E_C2 * 1.0)) < 1e-9   # h / p
    assert np.isnan(lam[1])


# --- tunneling ----------------------------------------------------------------------------
def test_tunneling_thick_barrier_limit():
    E, U0, w = 1.0, 5.0, 1.0
    T = Q.tunneling_probability(E, U0, w, exact=True)
    P = Q.tunneling_probability(E, U0, w)
    assert abs(T / (16 * E * (U0 - E) / U0 ** 2 * P) - 1) < 1e-6


def test_tunneling_resonance_and_continuity():
    U0, w = 1.0, 1.0
    E_res = U0 + Q.C_NM * (math.pi / w) ** 2          # sin(k'w) = 0 -> full transmission
    assert abs(Q.tunneling_probability(E_res, U0, w, exact=True) - 1) < 1e-12
    at = Q.tunneling_probability(U0, U0, w, exact=True)
    for d in (1e-7, -1e-7):
        assert abs(Q.tunneling_probability(U0 + d, U0, w, exact=True) - at) < 1e-5


# --- spectra and qubits -------------------------------------------------------------------
def test_transition_wavelengths_hydrogen():
    H = [-13.6 / n ** 2 for n in range(1, 6)]
    rows = Q.transition_wavelengths(H)
    lyman_a = [r for r in rows if (r[0], r[1]) == (2, 1)][0]
    assert abs(lyman_a[3] - Q.HC / 10.2) < 1e-9
    assert Q.transition_wavelengths(H, max_excitation=10.0) == []       # 10 eV can't reach n=2
    assert len(Q.transition_wavelengths(H, max_excitation=12.8)) == 6   # n <= 4


def test_qubit_anharmonicity():
    assert abs(Q.qubit_from_levels([Q.ho_energy(n, 0.1) for n in (1, 2, 3)])["alpha_eV"]) < 1e-15
    box = [Q.box_energy(n, 1.0) for n in (1, 2, 3)]
    q = Q.qubit_from_levels(box)
    assert abs(q["alpha_eV"] - 2 * box[0]) < 1e-12
    assert abs(q["f01_Hz"] - 3 * box[0] / Q.H_EV_S) < 1


def test_rabi_pi_pulse_is_not_gate():
    Om = 2 * math.pi * 1e6
    assert abs(Q.rabi_population(Om, math.pi / Om) - 1) < 1e-12
    assert Q.rabi_population(Om, 0.0) == 0.0
    t = np.linspace(0, 10 / Om, 2000)
    assert Q.rabi_population(Om, t, detuning=Om).max() <= 0.5 + 1e-12   # off-resonant: never fully flips


# --- input validation ---------------------------------------------------------------------
@pytest.mark.parametrize("call", [
    lambda: Q.box_energy(0, 1.0),
    lambda: Q.box_energy(1.5, 1.0),
    lambda: Q.box_energy(1, -1.0),
    lambda: Q.box_energy(1, 1.0, m=0),
    lambda: Q.ho_energy(1, 0.0),
    lambda: Q.finite_well_levels(-1.0, 1.0),
    lambda: Q.tunneling_probability(2.0, 1.0, 1.0),          # estimate needs E < U0
    lambda: Q.solve_schrodinger([0, 1, 3], 0.0),             # non-uniform grid
    lambda: Q.solve_schrodinger(np.linspace(0, 1, 10), 0.0, n_states=0),
    lambda: Q.qubit_from_levels([0.0, 1.0]),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
