"""Tests for dgs.quantum_gyroscope."""
import numpy as np
import pytest
import sympy as sp

from dgs import quantum_gyroscope as q


def test_sagnac_light():
    dt_exact, dt_first = q.sagnac_time_symbolic()
    A, W, c, lam = sp.symbols("A Omega c lambda", positive=True)
    assert sp.simplify(dt_first - 4 * A * W / c**2) == 0
    assert sp.simplify(q.sagnac_phase_light_symbolic() - 8 * sp.pi * A * W / (lam * c)) == 0


def test_stokes_and_matter_sagnac():
    curl, circ = q.rotating_frame_circulation_symbolic()
    W, R = sp.symbols("Omega R", positive=True)
    assert curl == sp.Matrix([0, 0, 2 * W])
    assert sp.simplify(circ - 2 * W * sp.pi * R**2) == 0
    m, hb, A = sp.symbols("m hbar A", positive=True)
    assert sp.simplify(q.sagnac_phase_matter_symbolic() - 2 * m * A * W / hb) == 0
    assert q.matter_light_ratio_symbolic()[1] == 0


def test_matter_gain_numbers_consistent():
    A, lam = 1e-4, 780e-9
    ratio = q.sagnac_phase_matter(A, q.OMEGA_EARTH, q.M_RB87) / q.sagnac_phase_light(A, q.OMEGA_EARTH, lam)
    assert ratio == pytest.approx(q.matter_light_gain(q.M_RB87, lam))
    assert 1e10 < ratio < 1e11


def test_atom_gyro_phase_is_coriolis():
    k, v, W, T = sp.symbols("k_eff v Omega T", positive=True)
    assert sp.simplify(q.atom_gyro_phase_symbolic() - 2 * k * v * W * T**2) == 0
    # phase = matter Sagnac with the enclosed area A = v_rec v T^2
    phi = q.atom_gyro_phase(q.OMEGA_EARTH, 10, 0.01)
    assert phi == pytest.approx(q.sagnac_phase_matter(q.atom_gyro_area(10, 0.01), q.OMEGA_EARTH, q.M_RB87))


def test_transfer_matrix_equals_qubit_circuit():
    M, P = q.mach_zehnder_symbolic()
    assert sp.simplify((M.H * M) - sp.eye(2)) == sp.zeros(2)
    phi = sp.symbols("phi", real=True)
    assert sp.simplify(P[0] + P[1] - 1) == 0
    assert q.mz_equals_qubit_circuit() == sp.zeros(2)


def test_heisenberg_vs_shot_noise():
    for n in (2, 4, 6):
        assert q.quantum_fisher_information(q.product_phase_state, n) == pytest.approx(n, rel=1e-5)
        assert q.quantum_fisher_information(q.ghz_phase_state, n) == pytest.approx(n**2, rel=1e-5)
    phis = np.linspace(0, 1, 7)
    assert np.allclose(q.ghz_parity_fringe(5, phis), np.cos(5 * phis))


def test_spiral_area_green_theorem_matches_sympy():
    xy = q.spiral_path(0.02, 0.0005, 20, pts_per_turn=720)
    r0, p, N = sp.symbols("r_0 p N", positive=True)
    exact = float(q.spiral_area_symbolic().subs({r0: 0.02, p: 0.0005, N: 20}))
    assert q.shoelace_area(xy) == pytest.approx(exact, rel=1e-4)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], float)
    assert q.shoelace_area(square) == 1.0 and q.shoelace_area(square[::-1]) == -1.0


def test_dxf_and_gcode_round_trip():
    xy = q.spiral_path(0.01, 0.001, 3, pts_per_turn=90)
    segs = q.read_dxf_lines(q.to_dxf(xy))
    assert len(segs) == len(xy) - 1
    assert np.allclose(segs[:, :2], xy[:-1] * 1e3, atol=1e-3)
    g, L = q.to_gcode(xy)
    moves = q.gcode_moves(g)
    assert np.allclose(moves, xy[1:] * 1e3, atol=1e-3)
    assert L == pytest.approx(q.path_length(xy) * 1e3)
    assert g.splitlines()[1].startswith("G21") and g.rstrip().endswith("M2")
    assert "<svg" in q.to_svg(xy)


def test_scale_factor_and_validation():
    xy = q.spiral_path(0.02, 0.0005, 20)
    assert q.coil_scale_factor(xy) == pytest.approx(8 * np.pi * q.shoelace_area(xy) / (1550e-9 * q.C))
    with pytest.raises(ValueError):
        q.spiral_path(0, 0.001, 3)
    with pytest.raises(ValueError):
        q.gyro_sensitivity(0, 1e6)
