"""Tests for dgs/viewer4d.py (4D data views) and dgs/nl_commands.py (intent + slot parsing)."""
import math

import numpy as np
import pytest

from dgs import nl_commands as NL
from dgs import one_d_quantum as Q
from dgs import viewer4d as V


@pytest.fixture(scope="module")
def vs():
    return V.box_superposition(L=1.0, n=32, frames=41)


# --- physics of the example dataset -------------------------------------------------------
def test_normalized_every_frame(vs):
    dV = (vs.x[1] - vs.x[0]) ** 3
    assert np.allclose(vs.data.sum(axis=(1, 2, 3)) * dV, 1.0, atol=1e-12)


def test_beat_period(vs):
    E1 = Q.box_energy(1, 1.0)
    period = 2 * math.pi * V.HBAR_EV_FS / (6 * E1 - 3 * E1)
    assert vs.t[-1] == pytest.approx(period)
    assert period == pytest.approx(3.67, abs=0.01)                    # fs
    assert np.allclose(vs.data[0], vs.data[-1], atol=1e-10)           # back where it started


def test_center_of_mass_sloshes_along_x(vs):
    com = vs.center_of_mass()
    X12 = -16 / (9 * math.pi ** 2)                                     # <1|x|2> for L = 1
    expected = 0.5 + X12 * np.cos(2 * math.pi * vs.t / vs.t[-1])       # c1 = c2 = 1/sqrt2
    assert np.allclose(com[:, 0], expected, atol=2e-3)
    assert np.allclose(com[:, 1:], 0.5, atol=1e-12)


# --- views --------------------------------------------------------------------------------
def test_views(vs):
    img, hv = vs.slice("z", 0.5, 0.0)
    assert img.shape == (32, 32) and hv == ("x", "y")
    proj, hv = vs.projection("y", 1.0, "max")
    assert hv == ("x", "z") and proj.max() == pytest.approx(vs.data[vs.index("t", 1.0)].max())
    s, _ = vs.projection("x", 0.0, "sum")
    assert s.sum() == pytest.approx(vs.data[0].sum())
    assert vs.probe((0.5, 0.5, 0.5)).shape == (41,)
    pts, vals = vs.isosurface_points(0.0, 0.6)
    assert pts.shape[1] == 3 and (vals >= 0.6 * vs.data[0].max()).all()
    tp, xyz, v = vs.peak()
    assert v == pytest.approx(vs.data.max())


@pytest.mark.parametrize("call", [
    lambda vs: vs.slice("t", 0.5, 0.0),
    lambda vs: vs.slice("z", 5.0, 0.0),                # outside the box
    lambda vs: vs.projection("z", 0.0, "median"),
    lambda vs: vs.probe((0.5, 0.5)),
    lambda vs: vs.isosurface_points(0.0, 1.5),
    lambda vs: V.VolumeSeries(np.zeros((2, 3, 3)), [0, 1], [0, 1, 2], [0, 1, 2], [0, 1, 2]),
    lambda vs: V.VolumeSeries(np.zeros((2, 2, 2, 2)), [1, 0], [0, 1], [0, 1], [0, 1]),
])
def test_view_validation(vs, call):
    with pytest.raises(ValueError):
        call(vs)


# --- NLP: intent classification -----------------------------------------------------------
HELD_OUT = [
    ("cut the box at z = 0.4 at t = 2 fs", "slice"),
    ("show me the xz plane at y 0.5", "slice"),
    ("max intensity along the y axis", "projection"),
    ("sum everything along z", "projection"),
    ("what is the value at (0.3, 0.5, 0.5) as time goes on", "probe"),
    ("time trace at the middle point", "probe"),
    ("isosurface at 40 percent", "isosurface"),
    ("show the 3d region above 80%", "isosurface"),
    ("make an animation of the x slice", "animate"),
    ("play it as a movie", "animate"),
    ("where and when is the maximum density", "peak"),
    ("find the highest point", "peak"),
    ("plot how the centroid moves in time", "center"),
    ("center of mass trajectory", "center"),
]


def test_held_out_intent_accuracy():
    correct = sum(NL.parse(text).intent == want for text, want in HELD_OUT)
    assert correct >= 12, f"only {correct}/{len(HELD_OUT)} held-out phrases classified correctly"


def test_animate_with_position_regression():
    # was classified as "slice": position words outweighed "animate" before more data was added
    for text in ("animate the y slice at y = 0.4", "animate the z slice at z = 0.5"):
        cmd = NL.parse(text)
        assert cmd.intent == "animate", (text, cmd.intent)
        assert "position" in cmd.slots


def test_training_set_is_fit():
    for intent, phrases in NL.TRAINING.items():
        for p in phrases:
            assert NL.MODEL.predict(p)[0] == intent, p


def test_naive_bayes_math():
    m = NL.NaiveBayesIntent(alpha=1.0).fit({"a": ["red red blue"], "b": ["green"]})
    # P(red | a) = (2 + 1) / (3 + 1 * V); V counts unigrams + bigrams
    V = len(m.vocab)
    s = m.scores("red")
    assert s["a"] == pytest.approx(math.log(0.5) + math.log(3 / (5 + V)))
    assert sum(m.predict("red")[2].values()) == pytest.approx(1)


def test_gibberish_not_understood():
    cmd = NL.parse("purple monkey dishwasher")
    assert not cmd.understood
    assert NL.execute(V.box_superposition(n=8, frames=3), cmd)["kind"] == "unknown"


# --- NLP: slot filling --------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("show the z slice at z = 0.3 at t = 2 fs", {"axis": "z", "position": 0.3, "t": 2.0}),
    ("show the xy plane", {"axis": "z"}),
    ("show the yz plane at x = 0.25", {"axis": "x", "position": 0.25}),
    ("sum along x at time 1.5", {"axis": "x", "kind": "sum", "t": 1.5}),
    ("probe the point (0.2, 0.5, 0.6)", {"point": (0.2, 0.5, 0.6)}),
    ("time series at x = 0.3 y = 0.5 z = 0.5", {"point": (0.3, 0.5, 0.5)}),
    ("isosurface at 70% at 1 fs", {"level": 0.7, "t": 1.0, "view": "isosurface"}),
    ("3d blob above half max", {"level": 0.5}),
])
def test_slots(text, expected):
    slots = NL.extract_slots(text)
    for k, v in expected.items():
        assert slots.get(k) == pytest.approx(v) if isinstance(v, (float, tuple)) else slots.get(k) == v, (k, slots)


def test_execute_end_to_end(vs):
    out = NL.execute(vs, NL.parse("show the x slice at 0.5 at t = 1 fs"))
    assert out["kind"] == "image" and out["axes"] == ("y", "z")
    out = NL.execute(vs, NL.parse("probe the point (0.25, 0.5, 0.5)"))
    assert out["kind"] == "series" and len(out["y"]) == len(vs.t)
    out = NL.execute(vs, NL.parse("track the center of mass"))
    assert out["y"].shape == (len(vs.t), 3)
    with pytest.raises(ValueError):
        NL.parse("   ")
