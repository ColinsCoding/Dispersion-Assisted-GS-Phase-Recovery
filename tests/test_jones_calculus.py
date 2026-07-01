import numpy as np
import pytest
import sympy as sp
from dgs.jones_calculus import (
    jones_horizontal, jones_vertical, jones_linear, jones_right_circular,
    jones_left_circular, jones_elliptical,
    jones_matrix_linear_polarizer, jones_matrix_phase_retarder,
    jones_matrix_hwp, jones_matrix_qwp, jones_matrix_rotation,
    jones_rotate_matrix, jones_cascade, jones_propagate,
    stokes_from_jones, poincare_angles, degree_of_polarization,
    fiber_pmd_jones, jones_sympy_5,
)


# ── Jones vectors normalized ──────────────────────────────────────────

def test_jones_h_normalized():
    E = jones_horizontal()
    assert np.linalg.norm(E) == pytest.approx(1.0)

def test_jones_rcp_normalized():
    E = jones_right_circular()
    assert np.linalg.norm(E) == pytest.approx(1.0)

def test_jones_linear_45():
    E = jones_linear(45)
    assert np.abs(E[0] - E[1]) < 1e-10   # equal x and y components

def test_jones_linear_0_is_horizontal():
    E = jones_linear(0)
    assert np.allclose(E, jones_horizontal())

def test_jones_linear_90_is_vertical():
    E = jones_linear(90)
    assert np.allclose(np.abs(E), np.abs(jones_vertical()), atol=1e-10)


# ── Jones matrices: unitarity ─────────────────────────────────────────

def test_hwp_unitary():
    M = jones_matrix_hwp()
    prod = M @ M.conj().T
    assert np.allclose(prod, np.eye(2), atol=1e-10)

def test_qwp_unitary():
    M = jones_matrix_qwp()
    prod = M @ M.conj().T
    assert np.allclose(prod, np.eye(2), atol=1e-10)

def test_hwp_twice_is_identity():
    M = jones_matrix_hwp()
    M2 = M @ M
    assert np.allclose(M2, np.eye(2), atol=1e-10)


# ── QWP: linear to circular ───────────────────────────────────────────

def test_qwp_45_linear_to_circular():
    # +45 LP through QWP (fast axis at 0) -> RCP
    E_in = jones_linear(45)
    M_qwp = jones_matrix_qwp(fast_axis_deg=0)
    E_out = M_qwp @ E_in
    S = stokes_from_jones(E_out)
    # S3 should be large (circularly polarized)
    assert abs(S["S3"]) / S["S0"] > 0.9


# ── polarizer ─────────────────────────────────────────────────────────

def test_polarizer_passes_aligned():
    E_h = jones_horizontal()
    M_pol = jones_matrix_linear_polarizer(0)
    E_out = M_pol @ E_h
    assert np.abs(E_out[0]) == pytest.approx(1.0, abs=1e-10)
    assert np.abs(E_out[1]) == pytest.approx(0.0, abs=1e-10)

def test_polarizer_blocks_orthogonal():
    E_v = jones_vertical()
    M_pol = jones_matrix_linear_polarizer(0)  # horizontal polarizer
    E_out = M_pol @ E_v
    assert np.linalg.norm(E_out) < 1e-10

def test_polarizers_crossed_zero_intensity():
    E_h = jones_horizontal()
    M_h = jones_matrix_linear_polarizer(0)    # horizontal
    M_v = jones_matrix_linear_polarizer(90)   # vertical
    result = jones_propagate(E_h, [M_h, M_v])
    assert result["intensity"] < 1e-10


# ── chain rule / cascade ──────────────────────────────────────────────

def test_cascade_empty_is_identity():
    M = jones_cascade([])
    assert np.allclose(M, np.eye(2))

def test_cascade_single():
    M_hwp = jones_matrix_hwp()
    assert np.allclose(jones_cascade([M_hwp]), M_hwp)

def test_cascade_noncommutative():
    M1 = jones_matrix_qwp(fast_axis_deg=0)
    M2 = jones_matrix_hwp(fast_axis_deg=22.5)
    M_12 = jones_cascade([M1, M2])  # M2 * M1
    M_21 = jones_cascade([M2, M1])  # M1 * M2
    assert not np.allclose(M_12, M_21)   # order matters

def test_cascade_propagate_intensity_conserved():
    E_in = jones_linear(30)
    M_hwp = jones_matrix_hwp(fast_axis_deg=45)
    result = jones_propagate(E_in, [M_hwp])
    assert result["intensity"] == pytest.approx(1.0, abs=1e-6)


# ── rotation matrix ───────────────────────────────────────────────────

def test_rotation_0_is_identity():
    R = jones_matrix_rotation(0)
    assert np.allclose(R, np.eye(2))

def test_rotation_90_swaps():
    R = jones_matrix_rotation(90)
    E_h = jones_horizontal()
    E_out = R @ E_h
    # H rotated 90 deg -> V (modulo sign)
    assert np.abs(E_out[1]) > 0.9

def test_rotate_matrix_round_trip():
    M = jones_matrix_qwp()
    M_rot = jones_rotate_matrix(M, 45)
    M_back = jones_rotate_matrix(M_rot, -45)
    assert np.allclose(M_back, M, atol=1e-10)


# ── Stokes parameters ─────────────────────────────────────────────────

def test_stokes_h_is_s1_positive():
    S = stokes_from_jones(jones_horizontal())
    assert S["S1"] == pytest.approx(S["S0"])
    assert S["S2"] == pytest.approx(0.0, abs=1e-10)
    assert S["S3"] == pytest.approx(0.0, abs=1e-10)

def test_stokes_v_is_s1_negative():
    S = stokes_from_jones(jones_vertical())
    assert S["S1"] == pytest.approx(-S["S0"])

def test_stokes_rcp_s3_nonzero():
    # optics convention [1,-i]/sqrt(2) gives S3 = -1 (IEEE convention)
    S = stokes_from_jones(jones_right_circular())
    assert abs(S["S3"]) == pytest.approx(S["S0"], abs=1e-6)
    # RCP and LCP should have opposite S3 signs
    S_lcp = stokes_from_jones(jones_left_circular())
    assert np.sign(S["S3"]) != np.sign(S_lcp["S3"])

def test_stokes_dop_pure_state():
    S = stokes_from_jones(jones_linear(37))
    assert S["DOP"] == pytest.approx(1.0, abs=1e-6)

def test_stokes_s0_is_intensity():
    E = jones_linear(45)
    S = stokes_from_jones(E)
    intensity = np.abs(E[0])**2 + np.abs(E[1])**2
    assert S["S0"] == pytest.approx(intensity, rel=1e-6)


# ── Poincare sphere ───────────────────────────────────────────────────

def test_poincare_rcp_lcp_at_poles():
    # optics convention [1,-i]: RCP is at one pole, LCP at the other
    p_rcp = poincare_angles(jones_right_circular())
    p_lcp = poincare_angles(jones_left_circular())
    assert abs(p_rcp["chi_deg"]) == pytest.approx(45.0, abs=1.0)
    assert abs(p_lcp["chi_deg"]) == pytest.approx(45.0, abs=1.0)
    # they must be at opposite poles
    assert np.sign(p_rcp["chi_deg"]) != np.sign(p_lcp["chi_deg"])

def test_poincare_linear_equator():
    p = poincare_angles(jones_linear(30))
    assert abs(p["chi_deg"]) < 1.0   # on equator


# ── degree of polarization ────────────────────────────────────────────

def test_dop_pure_ensemble():
    E = jones_linear(45)
    dop = degree_of_polarization([E, E, E])
    assert dop["DOP"] == pytest.approx(1.0, abs=0.01)

def test_dop_mixed_ensemble():
    states = [jones_horizontal(), jones_vertical(),
              jones_right_circular(), jones_left_circular()]
    dop = degree_of_polarization(states)
    assert dop["DOP"] < 0.1   # nearly unpolarized


# ── fiber PMD ─────────────────────────────────────────────────────────

def test_pmd_preserves_intensity():
    M = fiber_pmd_jones(10.0, theta_deg=30, omega_rad_per_ps=0.1)
    E = jones_horizontal()
    E_out = M @ E
    assert np.sum(np.abs(E_out)**2) == pytest.approx(1.0, abs=1e-6)

def test_pmd_zero_dgd_is_phase_only():
    M = fiber_pmd_jones(0.0, theta_deg=45, omega_rad_per_ps=1.0)
    # zero DGD: both components get same phase, just identity up to global phase
    assert np.allclose(np.abs(M), np.eye(2), atol=1e-10)

def test_pmd_invalid_dgd():
    with pytest.raises(ValueError):
        fiber_pmd_jones(-1.0, 0, 0.1)


# ── SymPy 5 ──────────────────────────────────────────────────────────

def test_jones_sympy_5_count():
    eqs = jones_sympy_5()
    assert len(eqs) == 5

def test_jones_sympy_5_types():
    for k, eq in jones_sympy_5().items():
        assert isinstance(eq, sp.Basic), f"{k} not SymPy"
