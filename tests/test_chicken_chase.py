"""Tests for learn/chicken_chase/chicken_chase.c via dgs/chicken_chase.py.
Skipped when gcc is not available."""
import numpy as np
import pytest
import sympy as sp

from dgs.chicken_chase import (
    find_tool, build, run_pursuit, run_selftest, run_sim, read_pbm, pixel_address,
    framebuffer_bytes, disassemble_px_set, capture_time_symbolic, capture_time, W, H,
)

needs_gcc = pytest.mark.skipif(find_tool("gcc") is None, reason="gcc not installed")


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    return build(tmp_path_factory.mktemp("cc"))


def test_pixel_address_binary_algebra():
    assert pixel_address(0, 0) == (0, 0x80)
    assert pixel_address(9, 0) == (1, 0x40)
    assert pixel_address(W - 1, H - 1) == (framebuffer_bytes() - 1, 0x01)
    for x, y in ((5, 3), (127, 40), (64, 95)):
        i = y * W + x
        b, m = pixel_address(x, y)
        assert b == i // 8 and m == 2 ** (7 - i % 8)


def test_memory_sizes():
    assert framebuffer_bytes() == 1536
    assert framebuffer_bytes(bits_per_pixel=8) == 8 * 1536


def test_capture_time_formula():
    T, (d, vp, vc) = capture_time_symbolic()
    assert sp.simplify(T.subs(vc, 0) - d / vp) == 0
    assert sp.limit(T.subs({vp: 1, d: 1}), vc, 1, "-") == sp.oo
    assert capture_time(0.5) == pytest.approx(10 / 0.75)
    assert capture_time(1.0) is None and capture_time(10.0) is None


@needs_gcc
def test_selftest_bits(exe):
    assert run_selftest(exe) == ["80", "40", "01", "1", "0"]


@needs_gcc
@pytest.mark.parametrize("k", [0.0, 0.001, 0.1, 0.5, 0.8, 0.9])
def test_c_pursuit_matches_sympy(exe, k):
    assert run_pursuit(exe, k) == pytest.approx(capture_time(k), rel=2e-3)


@needs_gcc
@pytest.mark.parametrize("k", [1.0, 10.0, 1000.0])
def test_c_pursuit_never_catches_faster_target(exe, k):
    assert run_pursuit(exe, k) is None


@needs_gcc
def test_sim_frames_and_catches(exe, tmp_path):
    frames, catches = run_sim(exe, seed=7, steps=1500, every=100, out_dir=tmp_path)
    assert frames
    img = read_pbm(frames[0])
    assert img.shape == (H, W) and set(np.unique(img)) <= {0, 1}
    assert img[0].all() and img[-1].all()                    # border drawn
    counts = [c for _, c in catches]
    assert counts == list(range(1, len(counts) + 1))         # one catch at a time
    steps = [s for s, _ in catches]
    assert steps == sorted(steps)


@needs_gcc
def test_disassembly_shows_shift_and_mask():
    asm = " ".join(disassemble_px_set())
    assert "or" in asm and ("shr" in asm or "sar" in asm) and "and" in asm
