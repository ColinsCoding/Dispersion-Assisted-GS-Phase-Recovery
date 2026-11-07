import math

import pytest

from dgs import jaws_of_life_game as g
from dgs.jaws_of_life import Linkage


def test_piston_time_is_volume_over_flow():
    lk = Linkage()
    assert g.piston_time(lk, 0.1, 1.0) == pytest.approx(0.1 * lk.piston_area / (1e-3 / 60))
    assert g.piston_time(lk, 0.1, 0.5) == pytest.approx(2 * g.piston_time(lk, 0.1, 1.0))
    with pytest.raises(ValueError):
        g.piston_time(lk, 0.1, 0)


def test_cut_capacity_respects_throat_and_fit():
    lk = Linkage()
    cap, d, th = g.cut_capacity(lk, 0.008)
    assert d >= g.THROAT_MIN - 1e-12
    assert 2 * d * math.sin(th) == pytest.approx(0.008)
    with pytest.raises(ValueError):
        g.cut_capacity(lk, 1.0)


def test_metal_table_needed_force_scales_with_uts():
    mt = g.metal_table()
    rows = {r["metal"]: r for r in mt["rows"]}
    assert rows["boron steel (press-hardened)"]["needed_kN"] / rows["mild steel"]["needed_kN"] == pytest.approx(1500 / 400)
    assert len({r["capacity_kN"] for r in mt["rows"]}) == 1
    assert rows["mild steel"]["cuts"]


def test_spread_stalls_when_resistance_too_high():
    lk = Linkage()
    ok = g.spread_check(lk, 0.05, 1e3)
    bad = g.spread_check(lk, 0.05, 1e7)
    assert ok["ok"] and not bad["ok"] and "relief" in bad["why"]


def test_levels():
    r = g.run_level("1990s sedan", "door removal")
    assert r["success"] and r["total_s"] == pytest.approx(r["timeline"][-1]["t_end"])
    suv = g.run_level("2020s SUV", "B-pillar rip")
    assert not suv["success"] and suv["failed_step"] == "cut B-pillar top"
    assert g.run_level("2020s SUV", "B-pillar rip, relief cuts")["success"]
    slow = g.run_level("1990s sedan", "door removal", g.Kit(flow_lpm=0.5))
    assert slow["total_s"] > r["total_s"]


def test_bad_inputs():
    with pytest.raises(ValueError):
        g.Section(0.1, 0.05, 0.002, "unobtainium")
    with pytest.raises(ValueError):
        g.run_level("1990s sedan", ["weld door"])
