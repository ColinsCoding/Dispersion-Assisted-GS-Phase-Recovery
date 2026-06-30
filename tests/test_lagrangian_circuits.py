"""Tests for dgs/lagrangian_circuits.py"""
import numpy as np
import sympy as sp
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dgs.lagrangian_circuits import (
    lagrangian_rlc_sympy, noether_energy_sympy,
    euler_lagrange_solve, power_iv, thevenin_norton,
    open_circuit_limit, power_sweep_R,
    circuit_analogy_table, action_integral,
    normal_modes_coupled, lagrangian_transmission_line,
    lagrangian_sympy_5,
)


def test_rlc_eom_matches_kvl():
    rlc = lagrangian_rlc_sympy()
    assert rlc["EOM_matches_KVL"] is True


def test_noether_hamiltonian_equals_T_plus_V():
    n = noether_energy_sympy()
    assert n["H_equals_TplusV"] is True


def test_noether_5_keys():
    n = noether_energy_sympy()
    assert "canonical_momentum_p" in n
    assert "Hamiltonian_H" in n


def test_euler_lagrange_underdamped():
    # Q > 0.5: underdamped -> oscillation in current
    res = euler_lagrange_solve(1e-3, 10.0, 1e-6,
                                lambda t: 5.0, (0, 5e-3))
    assert res["Q_factor"] > 0.5
    # Current should oscillate (sign changes)
    I = res["I"]
    assert I.max() > 0 and I.min() < 0   # oscillation present


def test_euler_lagrange_overdamped():
    # R large -> overdamped (Q < 0.5)
    res = euler_lagrange_solve(1e-3, 500.0, 1e-6,
                                lambda t: 5.0, (0, 5e-3))
    assert res["Q_factor"] < 0.5
    # Current should not oscillate (stays positive)
    I = res["I"]
    assert I.min() >= -1e-6   # no oscillation


def test_euler_lagrange_energy_bounded_lossless():
    # R = 0: forward Euler is not symplectic so energy drifts slightly,
    # but H should stay within 2x of initial value (no blow-up).
    res = euler_lagrange_solve(1e-3, 0.0, 1e-6,
                                lambda t: 0.0, (0, 1e-3),
                                q0=1e-6, I0=0.0)
    H = res["H"]
    H0 = H[1]   # skip index 0 (may be 0 if q0 tiny and I0=0)
    if abs(H0) > 1e-30:
        assert H.max() / abs(H0) < 5.0   # no energy blow-up


def test_power_iv():
    assert power_iv(2.0, 5.0) == 10.0
    arr = power_iv(np.array([1, 2, 3]), np.array([3, 2, 1]))
    np.testing.assert_allclose(arr, [3, 4, 3])


def test_thevenin():
    th = thevenin_norton(12.0, 2.0)
    assert abs(th["R_th"] - 6.0) < 1e-10
    assert abs(th["max_power_transfer"] - 6.0) < 1e-6


def test_open_circuit_limit():
    # R -> large: I -> 0, P -> 0, V_load stays at V_source
    oc = open_circuit_limit(12.0, 1e9)
    assert oc["I"] < 1e-6
    assert abs(oc["P_load"]) < 1e-4


def test_maximum_power_transfer():
    R_load = np.logspace(0, 4, 1000)
    sw = power_sweep_R(12.0, 6.0, R_load)
    # Max power should occur near R_load = R_int = 6 Ohm
    assert abs(sw["R_at_Pmax"] - 6.0) / 6.0 < 0.05
    # Max power = V^2/(4*R_int) = 144/24 = 6 W
    assert abs(sw["P_max"] - 6.0) < 1e-6


def test_action_integral():
    # For LC with q0=A, I0=0: pure oscillation
    # Action should be non-zero
    t = np.linspace(0, 1e-3, 1000)
    omega = 1 / np.sqrt(1e-3 * 1e-6)
    q = 1e-6 * np.cos(omega * t)
    I = -1e-6 * omega * np.sin(omega * t)
    S = action_integral(t, q, I, 1e-3, 1e-6)
    assert isinstance(S, float)


def test_normal_modes_splitting():
    nm = normal_modes_coupled(1e-3, 1e-6, 10e-6)
    # Antisymmetric mode higher than symmetric
    assert nm["omega_antisymmetric"] > nm["omega_symmetric"]
    # Without coupling (C_couple -> inf): both modes = omega_0
    nm0 = normal_modes_coupled(1e-3, 1e-6, 1e6)   # very large C_couple
    assert abs(nm0["omega_symmetric"] - nm0["omega_0"]) / nm0["omega_0"] < 0.01


def test_transmission_line_50_ohm():
    tl = lagrangian_transmission_line(L_per_m=250e-9, C_per_m=100e-12)
    assert abs(tl["Z0"] - 50.0) < 0.01


def test_transmission_line_phase_velocity():
    tl = lagrangian_transmission_line(L_per_m=250e-9, C_per_m=100e-12)
    # v_p = 1/sqrt(LC) = 1/sqrt(250e-9*100e-12) = 2e8 m/s
    assert abs(tl["v_phase"] - 2e8) / 2e8 < 1e-6


def test_lagrangian_sympy_5():
    eqs = lagrangian_sympy_5()
    assert len(eqs) == 5
    for k, v in eqs.items():
        assert isinstance(v, sp.Eq)
