"""Tests for the puck C code (gcc) and CUDA renderer (nvcc); skip when a toolchain is missing."""
import pytest

from dgs import puck_game as P
from dgs.eventsim import ToolUnavailable


def test_c_unit_tests_pass():
    try:
        passed, total, out = P.run_c_tests()
    except ToolUnavailable:
        pytest.skip("gcc not installed")
    assert passed == total and total >= 10, out


def test_game_scores_straight_shot_and_misses_slow_angled_one():
    try:
        scored, board, info = P.play(0, 3.0, 40)
    except ToolUnavailable:
        pytest.skip("gcc not installed")
    assert scored and "O" in board and float(info["KE"]) > 0
    scored, _, info = P.play(35, 2.5, 10)
    assert not scored and info["result"] == "stopped"


@pytest.mark.parametrize("tilt", [0, 30, 60, 90])
def test_renderer_projected_area_matches_linear_algebra(tilt):
    try:
        img, info = P.render(tilt_deg=tilt, spin_deg=20, h_over_R=0.3, size=400)
    except ToolUnavailable:
        pytest.skip("nvcc or CUDA device not available")
    frac = info["puck_pixels"] / img.shape[0] / img.shape[1]
    assert frac == pytest.approx(P.projected_area_fraction(tilt, 0.3), rel=0.02)


def test_spin_does_not_change_area_but_moves_the_stripe():
    try:
        a, ia = P.render(tilt_deg=0, spin_deg=0, size=200)
        b, ib = P.render(tilt_deg=0, spin_deg=90, size=200)
    except ToolUnavailable:
        pytest.skip("nvcc or CUDA device not available")
    assert ia["puck_pixels"] == pytest.approx(ib["puck_pixels"], rel=0.01)   # rotation keeps area
    red = lambda im: (im[..., 0] > 150) & (im[..., 1] < 80)
    assert red(a).sum() > 50 and (red(a) != red(b)).sum() > 50
