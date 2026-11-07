"""Tests for dgs.jaws_of_life: kinematics vs closed forms, virtual work vs force balance, engine."""
import math

import numpy as np
import pytest

from dgs import jaws_of_life as j

LK = j.Linkage()
TH = np.linspace(0, LK.theta_max, 25)


def test_law_of_cosines_closes_the_triangle():
    for th in TH:
        E = LK.ear_point(th)
        C = np.array([-LK.crosshead_distance(th), 0.0])
        assert np.linalg.norm(E - C) == pytest.approx(LK.link_length, rel=1e-12)
        assert np.linalg.norm(E) == pytest.approx(LK.ear_radius, rel=1e-12)


def test_inverse_kinematics_round_trip():
    x = LK.piston_extension(TH)
    assert np.allclose(LK.theta_from_extension(x), TH, atol=1e-10)


def test_extension_and_gap_monotonic_over_working_range():
    th = np.linspace(0, LK.theta_max, 400)
    assert np.all(np.diff(LK.piston_extension(th)) > 0)
    assert np.all(np.diff(LK.tip_gap(th)) > 0)
    assert LK.piston_extension(0.0) == 0.0


def test_tip_gap_closed_form():
    assert float(LK.tip_gap(LK.theta_max)) == pytest.approx(2 * LK.tip_length * math.sin(LK.theta_max))


def test_derivative_matches_finite_difference():
    h = 1e-6
    th = TH[1:-1]
    fd = (LK.piston_extension(th + h) - LK.piston_extension(th - h)) / (2 * h)
    assert np.allclose(LK.dxp_dtheta(th), fd, rtol=1e-6)


def test_virtual_work_equals_force_balance():
    for th in TH:
        fb = LK.force_balance(th)
        assert fb["separating_force"] == pytest.approx(float(LK.tip_force(th)), rel=1e-10)
        assert fb["torque"] == pytest.approx(float(LK.arm_torque(th)), rel=1e-10)
        assert fb["link_length_check"] == pytest.approx(LK.link_length, rel=1e-12)


def test_mechanical_advantage_closed_form():
    r, l, L, d = LK.ear_radius, LK.link_length, LK.tip_length, LK.ear_angle
    for th in TH:
        psi = th + d
        root = math.sqrt(l ** 2 - (r * math.sin(psi)) ** 2)
        c = r * math.cos(psi) + root
        ma = r * c * math.sin(psi) / (2 * L * math.cos(th) * root)
        assert float(LK.mechanical_advantage(th)) == pytest.approx(ma, rel=1e-12)


def test_energy_conservation_integrated():
    # work done by the piston equals work done at the tips over the full stroke
    th = np.linspace(0, LK.theta_max, 20001)
    Fp = LK.piston_force()
    W_tips = np.trapezoid(LK.tip_force(th) * 2 * LK.tip_length * np.cos(th), th)
    assert W_tips == pytest.approx(Fp * LK.stroke, rel=1e-6)


def test_limit_zero_ear_angle_has_no_force_when_closed():
    lk = j.Linkage(ear_angle=0.0, theta_max=math.radians(60))
    assert float(lk.tip_force(0.0)) == pytest.approx(0.0, abs=1e-9)   # toggle: link through pivot


def test_limit_long_link_approaches_lever_law():
    # l -> infinity: crosshead pushes the ear straight along x, MA -> r sin(psi) / (2 L cos th)
    lk = j.Linkage(link_length=1e4)
    th = 0.3
    psi = th + lk.ear_angle
    assert float(lk.mechanical_advantage(th)) == pytest.approx(
        lk.ear_radius * math.sin(psi) / (2 * lk.tip_length * math.cos(th)), rel=1e-4)


def test_hydraulics_force_is_pressure_times_area():
    assert LK.piston_force(700e5) == pytest.approx(700e5 * math.pi * 0.06 ** 2 / 4)
    assert LK.piston_force(0.0) == 0.0


def test_cutter_closed_forms():
    assert j.contact_distance(0.04, math.radians(30)) == pytest.approx(0.04)
    assert j.shear_area_tube(0.04, 0.002) == pytest.approx(math.pi * (0.04 ** 2 - 0.036 ** 2) / 4)
    assert j.shear_area_box(0.05, 0.08, 0.002) == pytest.approx(0.05 * 0.08 - 0.046 * 0.076)
    assert j.cut_force(1e-4, 500e6) == pytest.approx(30e3)
    # force is inversely proportional to distance from the pivot
    lk = j.Linkage(tip_length=0.18, theta_max=math.radians(30), ear_angle=math.radians(50))
    assert j.blade_force(lk, 0.2, 0.05) == pytest.approx(2 * j.blade_force(lk, 0.2, 0.10))
    assert j.slides_out(math.radians(20), 0.2) and not j.slides_out(math.radians(5), 0.2)


def test_transform_compose_inverse_and_mirror():
    A = j.Transform2D(0.7, 1.0, -2.0)
    B = j.Transform2D(-0.2, 0.3, 0.5, mirror_y=True)
    p = np.array([[0.1, 0.2], [1.0, -3.0]])
    assert np.allclose((A @ B).apply(p), A.apply(B.apply(p)))
    assert np.allclose(B.inverse().apply(B.apply(p)), p)
    assert np.allclose(j.Transform2D(math.pi / 2).apply([[1, 0]]), [[0, 1]])
    assert np.allclose(j.Transform2D(mirror_y=True).apply([[1, 2]]), [[1, -2]])


def test_polygon_primitives():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    assert j.polygon_area(sq) == pytest.approx(1.0)
    assert j.polygon_area(sq[::-1]) == pytest.approx(-1.0)
    assert j.point_in_polygon((0.5, 0.5), sq) and not j.point_in_polygon((1.5, 0.5), sq)
    depth, n = j.circle_polygon_contact((0.5, -0.1), 0.3, sq)
    assert depth == pytest.approx(0.2) and np.allclose(n, [0, 1])
    depth, _ = j.circle_polygon_contact((0.5, -1.0), 0.3, sq)
    assert depth == pytest.approx(-0.7)
    half = j.clip_polygon_halfplane(sq, (0.5, 0), (1, 0))
    assert j.polygon_area(half) == pytest.approx(0.5)
    circ = j.circle_polygon((0, 0), 1.0, n=2000)
    assert j.polygon_area(circ) == pytest.approx(math.pi, rel=1e-5)


def test_spreader_game_loop_pops_door():
    frames = j.simulate_spreader()
    events = [f.event for f in frames if f.event]
    assert events == ["contact", "latch popped", "full open"]
    assert all(f.pressure <= LK.pressure_max + 1e-6 for f in frames)
    assert all(f.force_required <= f.force_available + 1e-6 for f in frames)
    t = [f.t for f in frames]
    assert np.all(np.diff(t) > 0)
    # piston speed = Q / A
    v = 1.0 / 1000 / 60 / LK.piston_area
    assert frames[1].extension == pytest.approx(v * 0.25)


def test_spreader_stalls_on_strong_panel():
    frames = j.simulate_spreader(panel=j.Panel(yield_force=200e3))
    assert frames[-1].event == "stall: relief valve"
    assert frames[-1].extra["stalled"]


def test_cutter_near_pivot_cuts_far_stalls():
    near = j.simulate_cutter(distance=0.05)
    assert near[-1].event == "cut through"
    assert near[-1].extra["area_frac"] == pytest.approx(0.0, abs=1e-9)
    far = j.simulate_cutter(distance=0.07)
    assert far[-1].event == "stall: relief valve"
    mild = j.simulate_cutter(distance=0.07, uts=400e6)     # mild steel cuts even far out
    assert mild[-1].event == "cut through"


@pytest.mark.parametrize("kw", [dict(tip_length=-1), dict(bore=0), dict(link_length=0.05),
                                dict(ear_angle=-0.1), dict(theta_max=1.5),
                                dict(pressure_max=float("nan"))])
def test_linkage_validation(kw):
    with pytest.raises(ValueError):
        j.Linkage(**kw)


def test_input_validation():
    with pytest.raises(ValueError):
        LK.tip_gap(2.0)
    with pytest.raises(ValueError):
        LK.theta_from_extension(LK.stroke + 0.01)
    with pytest.raises(ValueError):
        j.shear_area_tube(0.01, 0.01)
    with pytest.raises(ValueError):
        j.cut_force(1e-4, 500e6, shear_ratio=1.5)
    with pytest.raises(ValueError):
        j.simulate_spreader(gap=1.0)
    with pytest.raises(ValueError):
        j.simulate_cutter(diameter=0.2)
    with pytest.raises(ValueError):
        j.Panel(stiffness=0)


# --- cutting force vs metal, timed extrication ------------------------------------------------

def test_metal_table_closed_forms_and_ordering():
    rows = j.metal_cut_table()
    assert len(rows) == len(j.METALS) * len(j.SECTIONS)
    cu = j.default_cutter()
    for r in rows:
        area, hw = j.SECTIONS[r["section"]]
        assert r["required_kN"] == pytest.approx(0.6 * r["uts_MPa"] * 1e6 * area / 1e3)
        assert r["cuts"] == (r["available_kN"] >= r["required_kN"])
        assert r["distance_mm"] / 1e3 * math.sin(cu.theta_max) > hw    # section fits
    by = {(r["metal"], r["section"]): r for r in rows}
    assert by[("mild steel", "box pillar 50x80x1.5")]["cuts"]
    assert not by[("boron / UHSS (press-hardened)", "box pillar 50x80x1.5")]["cuts"]
    # required force rises with strength
    req = [by[(m, "tube 40x2")]["required_kN"] for m in j.METALS]
    assert req == sorted(req)


def test_min_cutter_force_agrees_with_simulate_cutter():
    cu = j.default_cutter()
    for uts in (400e6, 1500e6):
        for d in (0.05, 0.07):
            F_req = j.cut_force(j.shear_area_tube(0.04, 0.002), uts)
            sim_cuts = j.simulate_cutter(distance=d, uts=uts, n_frames=400)[-1].event == "cut through"
            assert sim_cuts == (j.min_cutter_force(cu, d, 0.02) >= F_req * (1 - 1e-3))


def test_extrication_clock_follows_pump_flow():
    run = j.extrication_run()
    assert run["success"] and run["score"] == pytest.approx(run["target"] - run["total"])
    names = [s[0] for s in run["steps"]]
    assert names[0].startswith("spread") and names[2].startswith("cut") and names[-1] == "open door fully"
    cu = j.default_cutter()
    assert run["steps"][2][1] == pytest.approx(cu.stroke * cu.piston_area / (1.0 / 1000 / 60))
    fast = j.extrication_run(pump_flow_lpm=2.0)
    hydraulic = lambda r: r["total"] - 2 * 30.0
    assert hydraulic(fast) == pytest.approx(hydraulic(run) / 2, rel=0.02)   # time ~ 1/Q


def test_extrication_fails_on_boron_pillar_and_validates():
    run = j.extrication_run(pillar_metal="boron / UHSS (press-hardened)")
    assert not run["success"] and run["steps"][-1][2].startswith("FAILED")
    with pytest.raises(ValueError):
        j.extrication_run(pillar_metal="unobtainium")
    with pytest.raises(ValueError):
        j.extrication_run(pillar_section="tree trunk")
    with pytest.raises(ValueError):
        j.min_cutter_force(j.default_cutter(), 0.03, 0.02)
