"""Tests for dgs/wifi_floorplan.py."""
import numpy as np
import pytest
import sympy as sp

from dgs.wifi_floorplan import (
    friis_symbolic, fspl_db, segments_cross, wall_loss_db, default_floorplan, coverage_map,
    coverage_fraction, distance_to_segment, best_router_position, wien_peak_um, band_radiance,
)


def test_friis_and_fspl():
    _, fspl, per_doubling, val = friis_symbolic()
    assert val == pytest.approx(6.0206, abs=1e-4)
    assert fspl_db(2.0, 2.4e9) - fspl_db(1.0, 2.4e9) == pytest.approx(6.0206, abs=1e-4)
    assert fspl_db(1.0, 2.4e9) == pytest.approx(40.05, abs=0.02)          # textbook 1 m @ 2.4 GHz
    assert fspl_db(1.0, 5e9) - fspl_db(1.0, 2.4e9) == pytest.approx(20 * np.log10(5 / 2.4))


def test_geometry():
    assert segments_cross((0, 0), (2, 2), (0, 2), (2, 0))
    assert not segments_cross((0, 0), (1, 0), (0, 1), (1, 1))
    walls = [((1, -1), (1, 1), "drywall"), ((2, -1), (2, 1), "concrete")]
    assert wall_loss_db((0, 0), (3, 0), walls, band=0) == pytest.approx(3 + 12)
    assert wall_loss_db((0, 0), (0.5, 0), walls) == 0
    assert distance_to_segment((0, 1), (-1, 0), (1, 0)) == pytest.approx(1.0)


def test_coverage_physics():
    W, H, walls = default_floorplan()
    _, _, P24 = coverage_map((10, 3), walls, W, H, 2.4e9, res=0.4)
    _, _, P5 = coverage_map((10, 3), walls, W, H, 5e9, res=0.4)
    assert np.all(P24 > P5)                                                  # 5 GHz loses more everywhere
    assert coverage_fraction(P24) > coverage_fraction(P5)
    _, _, Pc = coverage_map((2, 2), walls, W, H, 5e9, res=0.4)
    assert coverage_fraction(P5) > coverage_fraction(Pc)                    # centre beats corner
    with pytest.raises(ValueError):
        coverage_map((1, 1), walls, W, H, res=0)


def test_best_position_respects_walls():
    W, H, walls = default_floorplan()
    pos, frac = best_router_position(walls, W, H, 5e9, step=2.0, res=0.5)
    assert min(distance_to_segment(pos, a, b) for a, b, _ in walls) >= 0.3
    assert 0 < frac <= 1


def test_thermal():
    assert wien_peak_um(306) == pytest.approx(9.47, abs=0.01)
    assert wien_peak_um(289.8) == pytest.approx(10.0, abs=0.01)
    assert band_radiance(306, emissivity=0.98) > band_radiance(293, emissivity=0.95)
    with pytest.raises(ValueError):
        wien_peak_um(0)
