"""Tests for dgs/nv_diamond.py: spin-1 algebra, Zeeman levels, ODMR lines, Bloch rotations."""
import math

import numpy as np
import pytest

from dgs import nv_diamond as N
from dgs import one_d_quantum as Q


def test_spin1_algebra():
    comm = N.SX @ N.SY - N.SY @ N.SX
    assert np.allclose(comm, 1j * N.SZ)                                     # [Sx, Sy] = i Sz
    S2 = N.SX @ N.SX + N.SY @ N.SY + N.SZ @ N.SZ
    assert np.allclose(S2, 2 * np.eye(3))                                   # s(s+1) = 2


def test_gamma_value():
    assert abs(N.GAMMA_HZ_T - 28.03e9) < 0.01e9
    assert abs(N.GAMMA_HZ_T * 1e-4 - 2.803e6) < 1e3                         # MHz per gauss


@pytest.mark.parametrize("B", [0.0, 5e-3, 0.05, 0.2])
def test_axial_levels_exact(B):
    w, _ = N.levels([0, 0, B])
    assert np.allclose(np.sort(w), np.sort(N.axial_levels(B)), atol=1e-3)


def test_odmr_axial_and_inversion():
    B = 5e-3
    fm, fp = N.odmr_frequencies([0, 0, B])
    assert fm == pytest.approx(N.D_HZ - N.GAMMA_HZ_T * B, abs=1)
    assert fp == pytest.approx(N.D_HZ + N.GAMMA_HZ_T * B, abs=1)
    assert N.field_from_splitting(fm, fp) == pytest.approx(B)
    assert np.allclose(N.odmr_frequencies([0, 0, 0]), N.D_HZ)


def test_strain_splits_zero_field_lines():
    fm, fp = N.odmr_frequencies([0, 0, 0], E=5e6)
    assert fm == pytest.approx(N.D_HZ - 5e6, abs=1) and fp == pytest.approx(N.D_HZ + 5e6, abs=1)


def test_transverse_field_shifts_both_lines_up():
    Bt = 5e-3
    fm, fp = N.odmr_frequencies([Bt, 0, 0])
    # no first-order Zeeman shift for a transverse field; second order pushes the pair upward
    assert fm > N.D_HZ - 1e6 and fp > N.D_HZ
    assert (fm + fp) / 2 > N.D_HZ


def test_four_orientations():
    # field along [111]: one NV sees full axial field, three see cos = 1/3
    B = 3e-3
    lines = N.odmr_lines_all_orientations(B * N.NV_AXES[0])
    assert lines[0] == pytest.approx([N.D_HZ - N.GAMMA_HZ_T * B, N.D_HZ + N.GAMMA_HZ_T * B], abs=1)
    for k in (1, 2, 3):
        assert np.allclose(lines[k], lines[1], atol=1)
    assert lines[1][1] - lines[1][0] < lines[0][1] - lines[0][0]
    generic = N.odmr_lines_all_orientations(np.array([1.0, 2.0, 3.5]) * 1e-3)
    assert len(np.unique(np.round(generic.ravel() / 1e5))) == 8               # 8 distinct lines


def test_odmr_spectrum_depth():
    s = N.odmr_spectrum([N.D_HZ, N.D_HZ + 1e9], [N.D_HZ], 1e6, 0.1)
    assert s[0] == pytest.approx(0.9) and s[1] == pytest.approx(1, abs=1e-6)


def test_cw_sensitivity():
    eta = N.cw_sensitivity(1e6, 0.1, 1e10)
    assert eta == pytest.approx(4 / (3 * math.sqrt(3)) * 1e6 / (N.GAMMA_HZ_T * 0.1 * 1e5))
    assert N.cw_sensitivity(0.5e6, 0.1, 1e10) == pytest.approx(eta / 2)


def test_bloch_rotations():
    Om = 2 * math.pi * 10e6
    assert np.allclose(N.bloch_vector(Om, 0, math.pi / Om), [0, 0, -1], atol=1e-12)        # pi pulse
    assert np.allclose(N.bloch_vector(Om, 0, math.pi / (2 * Om)), [0, -1, 0], atol=1e-12)  # pi/2 -> equator
    t = np.linspace(0, 5 / 10e6, 300)
    for det in (0.0, 0.7 * Om, 2 * Om):
        r = N.bloch_vector(Om, det, t)
        assert np.allclose(np.linalg.norm(r, axis=1), 1)                                   # stays on the sphere
        assert np.allclose((1 - r[:, 2]) / 2, Q.rabi_population(Om, t, detuning=det))     # same physics as Ch 40


def test_ramsey_and_thermometry():
    assert N.ramsey_signal(0.0, 1e6, 1e-6) == pytest.approx(1)
    assert N.ramsey_signal(0.5e-6, 1e6, 1e3) == pytest.approx(0, abs=1e-9)
    assert N.zero_field_splitting(310) - N.zero_field_splitting(300) == pytest.approx(-742e3)


@pytest.mark.parametrize("call", [
    lambda: N.hamiltonian([0, 0]),
    lambda: N.hamiltonian([0, 0, 0], D=-1),
    lambda: N.field_from_splitting(3e9, 2e9),
    lambda: N.odmr_spectrum([2.87e9], [2.87e9], 0, 0.1),
    lambda: N.cw_sensitivity(1e6, 1.5, 1e10),
    lambda: N.bloch_vector(0, 0, 1e-6),
    lambda: N.ramsey_signal(1e-6, 1e6, 0),
    lambda: N.zero_field_splitting(0),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
