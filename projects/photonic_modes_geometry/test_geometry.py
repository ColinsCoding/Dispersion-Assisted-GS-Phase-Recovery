import numpy as np
import pytest
from geometry import make_grid, make_rectangle, make_circle, make_two_core_structure, make_slot


def test_make_grid_centered_at_origin():
    x, y, X, Y = make_grid(8, 8, 1.0, 1.0)
    assert np.isclose(x.mean(), -0.5)  # even-length grid: center sits between two samples


def test_make_grid_rejects_small_grid():
    with pytest.raises(ValueError):
        make_grid(1, 8, 1.0, 1.0)


def test_make_grid_rejects_bad_spacing():
    with pytest.raises(ValueError):
        make_grid(8, 8, -1.0, 1.0)


def test_make_rectangle_center_is_core_index():
    n, _ = make_rectangle(64, 64, 0.1, 0.1, width=2.0, height=1.0, n_core=3.4, n_clad=1.44)
    assert n[32, 32] == pytest.approx(3.4)


def test_make_rectangle_corner_is_cladding_index():
    n, _ = make_rectangle(64, 64, 0.1, 0.1, width=1.0, height=1.0, n_core=3.4, n_clad=1.44)
    assert n[0, 0] == pytest.approx(1.44)


def test_make_rectangle_rejects_nonpositive_dims():
    with pytest.raises(ValueError):
        make_rectangle(64, 64, 0.1, 0.1, width=0.0, height=1.0)


def test_make_circle_center_is_core():
    n, _ = make_circle(64, 64, 0.1, 0.1, radius=1.0)
    assert n[32, 32] == pytest.approx(3.4)


def test_make_circle_area_matches_analytic_formula_approximately():
    dx = dy = 0.05
    radius = 1.0
    n, _ = make_circle(200, 200, dx, dy, radius=radius)
    core_area_numeric = np.sum(n == n.max()) * dx * dy
    core_area_analytic = np.pi * radius ** 2
    assert core_area_numeric == pytest.approx(core_area_analytic, rel=0.02)


def test_make_circle_rejects_nonpositive_radius():
    with pytest.raises(ValueError):
        make_circle(64, 64, 0.1, 0.1, radius=-1.0)


def test_make_two_core_structure_has_two_disjoint_cores():
    n, (x, y) = make_two_core_structure(128, 64, 0.1, 0.1, core_width=1.0, core_height=1.0, gap=0.5)
    mid_x_idx = len(x) // 2
    # the gap (center) should be cladding, not core
    assert n[mid_x_idx, len(y) // 2] == pytest.approx(1.44)


def test_make_two_core_structure_rejects_bad_gap():
    with pytest.raises(ValueError):
        make_two_core_structure(64, 64, 0.1, 0.1, core_width=1.0, core_height=1.0, gap=0.0)


def test_make_two_core_structure_rejects_bad_shape():
    with pytest.raises(ValueError):
        make_two_core_structure(64, 64, 0.1, 0.1, 1.0, 1.0, 0.5, shape="triangle")


def test_make_slot_gap_is_slot_material_not_cladding():
    n, (x, y) = make_slot(128, 64, 0.1, 0.1, core_width=1.0, core_height=1.0, gap=0.5,
                           n_core=3.4, n_clad=1.44, n_slot=1.0)
    mid_x_idx = len(x) // 2
    assert n[mid_x_idx, len(y) // 2] == pytest.approx(1.0)  # slot material, not cladding (1.44)


def test_make_slot_rejects_nonpositive_n_slot():
    with pytest.raises(ValueError):
        make_slot(64, 64, 0.1, 0.1, 1.0, 1.0, 0.5, n_slot=0.0)


def test_make_slot_and_two_core_structure_agree_outside_the_gap():
    # both should place identical cores at the same locations -- they only
    # differ in what fills the gap between them
    n_slot, _ = make_slot(128, 64, 0.1, 0.1, core_width=1.0, core_height=1.0, gap=0.5)
    n_two_core, _ = make_two_core_structure(128, 64, 0.1, 0.1, core_width=1.0, core_height=1.0, gap=0.5)
    core_only_mask = (n_two_core == n_two_core.max())
    np.testing.assert_allclose(n_slot[core_only_mask], n_two_core[core_only_mask])
