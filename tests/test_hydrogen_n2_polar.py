"""Tests for dgs.hydrogen_n2_polar."""
import numpy as np
import pytest
import sympy as sp

from dgs import hydrogen_n2_polar as h


def test_four_states_are_orthonormal_unit_vectors():
    assert h.gram_matrix() == sp.eye(4)


def test_spherical_unit_vectors_orthonormal():
    _, G = h.spherical_unit_vectors()
    assert G == sp.eye(3)


def test_polar_area_needs_jacobian():
    for nl, wrong in [((1, 0), 2), ((2, 0), sp.Rational(1, 4)), ((2, 1), sp.Rational(1, 12))]:
        assert h.radial_area(*nl) == 1
        assert h.radial_area(*nl, with_jacobian=False) == wrong


def test_peaks_and_nodes():
    assert h.most_probable_radius(2, 1) == 4
    assert sp.simplify(h.most_probable_radius(2, 0) - (3 + sp.sqrt(5))) == 0
    assert h.radial_nodes(2, 0) == [2] and h.radial_nodes(2, 1) == []


def test_numeric_matches_sympy():
    for (n, l, m) in h.N2_STATES + [(1, 0, 0)]:
        f = sp.lambdify((h.r_s, h.th_s, h.ph_s), h.psi_sym(n, l, m), "numpy")
        for pt in [(0.5, 0.3, 0.2), (3.0, 1.2, -2.0), (7.0, 2.9, 1.0)]:
            assert np.isclose(f(*pt), h.psi(n, l, m, *pt), atol=1e-14)


def test_density_slice_normalized_on_plane_axis_symmetry():
    ax, rho = h.density_slice_xz({(2, 1, 0): 1.0}, extent=20, n=201)
    assert np.allclose(rho, rho[::-1, :])                       # |2p0|^2 symmetric in z
    assert rho[100, 100] == 0                                    # node at the origin


def test_degenerate_superposition_is_stationary_but_1s_2p_moves():
    a0 = h.density_slice_xz({(2, 0, 0): 2 ** -0.5, (2, 1, 0): 2 ** -0.5}, n=101, t=0)[1]
    a1 = h.density_slice_xz({(2, 0, 0): 2 ** -0.5, (2, 1, 0): 2 ** -0.5}, n=101, t=37.0)[1]
    assert np.allclose(a0, a1)
    b0 = h.density_slice_xz({(1, 0, 0): 2 ** -0.5, (2, 1, 0): 2 ** -0.5}, n=101, t=0)[1]
    b1 = h.density_slice_xz({(1, 0, 0): 2 ** -0.5, (2, 1, 0): 2 ** -0.5}, n=101, t=np.pi / 0.375)[1]
    assert not np.allclose(b0, b1)


def test_dipole_oscillates_at_bohr_frequency():
    w = h.energy_ha(2) - h.energy_ha(1)
    assert np.isclose(h.dipole_z(0.0), 2 * 0.5 * 128 * np.sqrt(2) / 243)
    assert np.isclose(h.dipole_z(np.pi / w), -h.dipole_z(0.0))
    assert np.isclose(h.bohr_frequency_hz(2, 1), 2.466e15, rtol=1e-3)


def test_spectrum_table_e_equals_hf():
    for row in h.spectrum_table():
        assert np.isclose(row["energy_eV"], h.H_EV_S * row["freq_Hz"])
    names = {r["name"]: r for r in h.spectrum_table()}
    assert np.isclose(names["H 21 cm hyperfine"]["wavelength_m"], 0.2110, rtol=1e-3)


def test_taylor_exp_variants():
    x = np.array([1.0, 10.0, 30.0])
    naive = h.exp_neg_naive(x) / np.exp(-x) - 1
    assert abs(naive[-1]) > 1e10                                # catastrophic cancellation
    assert np.all(np.abs(h.exp_range_reduced(-x) / np.exp(-x) - 1) < 1e-14)
    assert h.terms_needed(1.0) < h.terms_needed(10.0)


@pytest.mark.parametrize("call", [
    lambda: h.psi(3, 0, 0, 1.0, 0.0, 0.0),
    lambda: h.radial(3, 2, 1.0),
    lambda: h.ylm(2, 0, 0.1, 0.1),
    lambda: h.taylor_exp(1.0, 0),
])
def test_validation(call):
    with pytest.raises(ValueError):
        call()
