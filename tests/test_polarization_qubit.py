"""Tests for dgs/polarization_qubit.py (and the jones_calculus conventions it pins)."""
import numpy as np
import pytest
import sympy as sp

from dgs import jones_calculus as jc
from dgs.polarization_qubit import (
    th, KETS, X, Y, Z, S_GATE, HADAMARD, qwp, hwp, braket, equal_up_to_phase, is_unitary,
    bloch_vector, malus_born, solid_angle_octant, pancharatnam_phase, numeric,
)


def test_wave_plates_are_quantum_gates():
    assert qwp(0) == S_GATE
    assert hwp(0) == Z
    assert sp.simplify(hwp(sp.pi / 8) - HADAMARD) == sp.zeros(2, 2)
    assert is_unitary(qwp(th)) and is_unitary(hwp(th))
    assert equal_up_to_phase(qwp(sp.pi / 4) ** 2, hwp(sp.pi / 4))


def test_qwp_makes_circular_hwp_flips():
    assert equal_up_to_phase(qwp(sp.pi / 4) * KETS["H"], KETS["-i"])
    assert equal_up_to_phase(hwp(sp.pi / 4) * KETS["H"], KETS["V"])


def test_matches_jones_calculus_code():
    for deg in (0, 30, 45, 90):
        assert np.allclose(np.array(qwp(sp.pi * deg / 180).evalf(), complex), jc.jones_matrix_qwp(deg))
    # pin the actual (code) convention, which differs from the module's docstring
    assert np.allclose(jc.jones_matrix_qwp(0), np.diag([1, 1j]))
    assert jc.stokes_from_jones(jc.jones_right_circular())["S3"] == pytest.approx(-1.0)


def test_stokes_is_bloch():
    for k in KETS:
        S = jc.stokes_from_jones(numeric(k))
        s1, s2, s3 = (float(v) for v in bloch_vector(KETS[k])["stokes"])
        assert (s1, s2, s3) == pytest.approx((S["S1"], S["S2"], S["S3"]), abs=1e-12)


def test_born_rule_is_malus_and_orthogonality():
    assert sp.simplify(malus_born(th) - sp.cos(th) ** 2) == 0
    assert braket(KETS["H"], KETS["V"]) == 0 and braket(KETS["+i"], KETS["-i"]) == 0
    assert sp.simplify(sp.Abs(braket(KETS["D"], KETS["+i"])) ** 2) == sp.Rational(1, 2)


def test_geometric_phase_half_solid_angle():
    omega = solid_angle_octant()
    assert omega == sp.pi / 2
    g1 = pancharatnam_phase([numeric("H"), numeric("D"), numeric("+i")])
    g2 = pancharatnam_phase([numeric("H"), numeric("+i"), numeric("D")])
    assert abs(g1) == pytest.approx(float(omega) / 2) and g2 == pytest.approx(-g1)
