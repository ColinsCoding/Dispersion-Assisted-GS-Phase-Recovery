"""Tests for dgs.bench_calculator -- physical <-> normalized dispersion
conversions for planning a real TD-GSA bench measurement. Formulas are
extracted from (not re-derived from) notebooks/phase_retrieval.ipynb's
SymPy-verified dimensional-analysis section; the regression values below
were cross-checked against that notebook's own printed output."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from dgs import bench_calculator as bc


def test_alpha_matches_notebook_value():
    assert bc.alpha_ghz() == pytest.approx(2.515892116749826e-05, rel=1e-9)


def test_alpha_rejects_nonpositive_wavelength():
    with pytest.raises(ValueError, match="positive"):
        bc.alpha_ghz(lambda0_nm=0.0)
    with pytest.raises(ValueError, match="positive"):
        bc.alpha_ghz(lambda0_nm=-1550.0)


def test_normalize_denormalize_round_trip():
    """Physical -> normalized -> physical must return the original value,
    for a range of D_phys and sample rates."""
    for D_phys in [-695.0, 100.0, 6243.5, -0.6]:
        for fs in [1.0, 10.0, 56.0, 200.0]:
            D_norm = bc.normalize_dispersion(D_phys, fs)
            back = bc.denormalize_dispersion(D_norm, fs)
            assert back == pytest.approx(D_phys, rel=1e-9)


def test_normalize_dispersion_rejects_nonpositive_fs():
    with pytest.raises(ValueError, match="positive"):
        bc.normalize_dispersion(-695.0, fs_GHz=0.0)
    with pytest.raises(ValueError, match="positive"):
        bc.denormalize_dispersion(-5000.0, fs_GHz=-10.0)


def test_min_physical_dispersion_matches_notebook_table():
    """Regression: reproduces notebooks/phase_retrieval.ipynb's printed
    |D_phys|_min table exactly (B=100 GHz -> ~62.4 ps/nm, etc.)."""
    expected = {1: 624349.6357960458, 10: 6243.496357960458, 100: 62.43496357960458,
                500: 2.497398543184183, 1000: 0.6243496357960457}
    for B, D_min in expected.items():
        assert bc.min_physical_dispersion(B) == pytest.approx(D_min, rel=1e-9)


def test_min_physical_dispersion_decreases_with_bandwidth():
    """Wider-bandwidth signals accumulate quadratic phase faster, so need
    LESS physical dispersion to hit the same convergence threshold."""
    values = [bc.min_physical_dispersion(B) for B in [1, 10, 100, 1000]]
    assert values == sorted(values, reverse=True)


def test_min_physical_dispersion_rejects_nonpositive_bandwidth():
    with pytest.raises(ValueError, match="positive"):
        bc.min_physical_dispersion(0.0)


def test_fiber_length_km_basic():
    assert bc.fiber_length_km(17.0, D_per_km=17.0) == pytest.approx(1.0)
    assert bc.fiber_length_km(62.43496357960458, D_per_km=bc.SMF28_D_PER_KM) == pytest.approx(3.6726449164473283, rel=1e-9)


def test_fiber_length_km_rejects_zero_d_per_km():
    with pytest.raises(ValueError, match="nonzero"):
        bc.fiber_length_km(100.0, D_per_km=0.0)


def test_bench_plan_target_is_above_minimum():
    p = bc.bench_plan(bandwidth_GHz=100.0, margin=1.5)
    assert p["D_phys_target_ps_nm"] == pytest.approx(p["D_phys_min_ps_nm"] * 1.5)
    assert p["length_target_km"] > p["length_min_km"]
    assert p["length_min_km"] == pytest.approx(3.6726449164473283, rel=1e-9)


def test_bench_plan_rejects_margin_below_one():
    with pytest.raises(ValueError, match="margin"):
        bc.bench_plan(bandwidth_GHz=100.0, margin=0.5)


def test_bench_plan_default_margin_is_at_least_one():
    p = bc.bench_plan(bandwidth_GHz=100.0)
    assert p["D_phys_target_ps_nm"] >= p["D_phys_min_ps_nm"]
