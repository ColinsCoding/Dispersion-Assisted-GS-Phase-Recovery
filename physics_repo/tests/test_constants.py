"""Tests for physkit.constants: internal consistency of the SI constant set."""
import math
from physkit import constants as c


def test_hbar_matches_h():
    assert math.isclose(c.HBAR, c.H / (2 * math.pi), rel_tol=1e-12)


def test_coulomb_constant():
    assert math.isclose(c.COULOMB_K, 1 / (4 * math.pi * c.EPS0), rel_tol=1e-12)


def test_fine_structure_from_constants():
    # alpha = e^2 / (4 pi eps0 hbar c) must reproduce the tabulated value
    alpha = c.E**2 / (4 * math.pi * c.EPS0 * c.HBAR * c.C)
    assert math.isclose(alpha, c.ALPHA, rel_tol=1e-6)


def test_rydberg_from_constants():
    # Ry = m_e e^4 / (8 eps0^2 h^2) in joules -> eV
    Ry_J = c.M_E * c.E**4 / (8 * c.EPS0**2 * c.H**2)
    assert math.isclose(Ry_J / c.E, c.RYDBERG_EV, rel_tol=1e-4)


def test_bohr_radius_from_constants():
    a0 = 4 * math.pi * c.EPS0 * c.HBAR**2 / (c.M_E * c.E**2)
    assert math.isclose(a0, c.A0, rel_tol=1e-4)


def test_table_has_rows():
    df = c.table()
    assert len(df) >= 10 and "value_SI" in df.columns
