"""Tests for scarlett.seals_hardware (paper specs, SymPy design equations)."""
import numpy as np
import pytest
import sympy as sp

from scarlett import seals_hardware as sh


def test_mapping_matches_paper_ranges():
    lam, y, th = sh.mapping()
    assert y[0] * 1e3 == pytest.approx(5.3, abs=0.15)           # Fig. 2: ~5.3 mm span
    i1, i2 = np.argmin(abs(lam - 1582e-9)), np.argmin(abs(lam - 1598e-9))
    span = np.degrees(th[i1] - th[i2])
    assert span == pytest.approx(68, abs=3)                      # Fig. 3: ~+41 .. -27 deg
    assert np.all(np.diff(th) < 0)                               # one-to-one mapping


def test_grating_dispersion_symbolic():
    beta, disp, (lam, d, a, m) = sh.grating_symbolic()
    expect = m / (d * sp.cos(beta))
    for vals in ({lam: 1.59e-6, d: 9.09e-7, a: 0.9, m: 1}, {lam: 1.2e-6, d: 1.1e-6, a: 0.5, m: 1}):
        assert float(disp.subs(vals)) == pytest.approx(float(expect.subs(vals)), rel=1e-12)


def test_angle_slope_and_resolution_symbolic():
    _, slope, (y, yc, dc, P, NA) = sh.angle_symbolic()
    assert sp.simplify(slope - 2 * NA / (P * sp.sqrt(1 - NA ** 2))) == 0
    f, (a, L) = sh.min_resolvable_angle_symbolic()
    assert float(sp.deg(f.subs({a: 100e-6, L: 2e-3}))) == pytest.approx(2.864, abs=1e-3)


def test_resolution_reconciliation_only_one_reading_gives_042():
    rows = sh.resolution_reconciliation()
    hits = [r for r in rows if abs(r[1] * 1e9 - 0.48) < 0.01]
    assert len(hits) == 1 and "AND" in hits[0][0]
    assert hits[0][2] == pytest.approx(42, abs=1)


def test_larger_bead_more_lobes():
    small, large = sh.BEADS["7.32 um"], sh.BEADS["9.94 um"]
    assert sh.lobe_count(large, 0.2, 90) > sh.lobe_count(small, 0.2, 90)
    assert sh.lobe_period_deg(large) < sh.lobe_period_deg(small)


def test_equalization_reduces_dynamic_range():
    _, T, before, after = sh.equalization_filter(sh.BEADS["9.94 um"])
    assert before > 15 and after == pytest.approx(0, abs=1e-9) and T.max() == pytest.approx(1)
    _, _, b2, a2 = sh.equalization_filter(sh.BEADS["9.94 um"], max_atten_db=10)
    assert a2 == pytest.approx(b2 - 10, abs=0.2)
    with pytest.raises(ValueError):
        sh.equalization_filter(sh.BEADS["9.94 um"], max_atten_db=-1)


def test_time_stretch_budget():
    rec = sh.time_stretch_window(1000)                           # 1000 ps/nm x 20 nm
    assert rec == pytest.approx(20.0)
    assert sh.digitizer_rate_needed(42, rec) == pytest.approx(4.2e9)
    with pytest.raises(ValueError):
        sh.time_stretch_window(0)
