"""Tests for dgs/chicken_chase_game.py and the headless pygame viewer."""
import numpy as np
import pytest

from dgs.chicken_chase_game import (
    World, Params, PursuitLab, bouguer_capture_time, render_bits, WORLD_W, WORLD_H,
)


def test_autopilot_catches_everything():
    w = World(seed=3)
    for _ in range(4000):
        w.step(autopilot=True)
    assert w.caught == w.params.n_chickens
    assert w.catch_steps == sorted(w.catch_steps)


def test_chickens_stay_in_bounds():
    w = World(seed=1)
    for _ in range(2000):
        w.step(move=(1, 1))
        live = w.chickens[w.alive]
        assert np.all((live[:, 0] >= 2) & (live[:, 0] <= WORLD_W - 3))
        assert np.all((live[:, 1] >= 2) & (live[:, 1] <= WORLD_H - 3))


def test_flee_prediction_formula_and_faster_chicken():
    w = World(seed=0)
    i, d = w.nearest()
    p = w.params
    assert w.flee_prediction() == pytest.approx(
        (d - p.catch_radius - p.v_player) / (p.v_player - p.v_flee))
    fast = World(Params(v_flee=1.0, v_player=0.9), seed=0)
    assert fast.flee_prediction() == np.inf


def test_straight_flight_capture_time():
    """One chicken fleeing in open space is caught when the formula says."""
    p = Params(n_chickens=1, flee_radius=1e3, v_wander=0.0)
    w = World(p, seed=0)
    # gap 6, closing at 0.15/step; the player's move-before-check makes the
    # effective catch distance r_c + v_p -> (6 - 2 - 0.9)/0.15 = 20.7 steps
    w.player[:] = (10, 48)
    w.chickens[0, :2] = (16, 48)                       # flees straight toward +x
    pred = w.flee_prediction()
    while w.caught == 0:
        w.step(autopilot=True)
    assert w.catch_steps[0] == int(np.floor(pred)) + 1


@pytest.mark.parametrize("k", [0.0, 0.3, 0.6, 0.85])
def test_pursuit_lab_matches_bouguer(k):
    lab = PursuitLab(k=k)
    while not lab.done:
        lab.advance(1)
    assert lab.t == pytest.approx(bouguer_capture_time(lab.d, lab.v_p, lab.v_c), rel=2e-3)
    with pytest.raises(ValueError):
        PursuitLab(k=1.0)


def test_bits_layout():
    img, packed = render_bits(World(seed=0))
    assert img.shape == (WORLD_H, WORLD_W) and packed.size == 1536
    assert np.array_equal(np.unpackbits(packed).reshape(WORLD_H, WORLD_W), img)


def test_params_validation():
    with pytest.raises(ValueError):
        Params(v_player=0)
    with pytest.raises(ValueError):
        Params(n_chickens=0)


def test_headless_viewer(tmp_path):
    pytest.importorskip("pygame")
    from dgs.viewer_chicken_pygame import run
    shot = tmp_path / "game.png"
    r = run(headless=True, frames=60, shot=str(shot))
    assert shot.exists() and r["steps"] == 60
    r = run(headless=True, frames=200, start="lab")
    assert r["lab_done"] and r["lab_t"] == pytest.approx(r["lab_predicted"], rel=2e-3)
    with pytest.raises(ValueError):
        run(headless=True, frames=1, start="nope")
