"""dgs.pst_cuda vs the real phycv.PST on synthetic crack surfaces.

GPU cases run in float32 (complex64), phycv runs in float64, so the analog
feature is compared to 1e-4 and the binary edge map may differ only at
pixels whose phycv feature value sits within 1e-4 of a quantile threshold
(float32 rounding can push a tie either way); every other pixel must match.
float64 runs must agree to ~1e-10.
"""
import numpy as np
import pytest

phycv = pytest.importorskip("phycv")
from dgs import pst_cuda as pc

XP, BACKEND = pc.get_backend("cupy")
gpu = pytest.mark.skipif(BACKEND != "cupy", reason="no CuPy / CUDA device")
PARAMS = dict(S=0.4, W=20.0, sigma_LPF=0.1, thresh_min=0.0, thresh_max=0.95)
CASES = [(256, 256, 0), (512, 512, 1), (200, 320, 2), (257, 255, 3)]   # incl. odd + non-square


def _surface(h, w, seed):
    img, mask = pc.synthetic_crack_surface(max(h, w), n_cracks=2, seed=seed)
    return img[:h, :w], mask[:h, :w]


def _compare(backend, dtype, fused, h, w, seed):
    img, _ = _surface(h, w, seed)
    ref_f, ref_o = pc.phycv_reference(img, morph_flag=True, **PARAMS)
    p = pc.PSTCuda(backend=backend, dtype=dtype, fused=fused)
    p.run(img, PARAMS["S"], PARAMS["W"], PARAMS["sigma_LPF"], PARAMS["thresh_min"],
          PARAMS["thresh_max"], True)
    f, o = p.to_numpy(p.pst_feature), p.to_numpy(p.pst_output)
    assert f.shape == ref_f.shape == o.shape
    sub = ref_f[::4, ::4]
    q = np.array([np.quantile(sub, PARAMS["thresh_max"]), np.quantile(sub, PARAMS["thresh_min"])])
    bad = o != ref_o
    on_threshold = np.abs(ref_f[bad][:, None] - q[None, :]).min(axis=1) < 1e-4
    assert on_threshold.all(), "edge-map mismatch away from a quantile threshold"
    return np.abs(f - ref_f).max(), bad.mean()


@gpu
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("h,w,seed", CASES)
def test_gpu_float32_matches_phycv(fused, h, w, seed):
    df, frac = _compare("cupy", "float32", fused, h, w, seed)
    assert df < 1e-4, df
    assert frac < 1e-3, frac


@gpu
@pytest.mark.parametrize("fused", [False, True])
def test_gpu_float64_matches_phycv_exactly(fused):
    df, frac = _compare("cupy", "float64", fused, 256, 256, 0)
    assert df < 1e-10 and frac == 0


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("h,w,seed", CASES[:3])
def test_numpy_fallback_float64_matches_phycv(fused, h, w, seed):
    df, frac = _compare("numpy", "float64", fused, h, w, seed)
    assert df < 1e-10 and frac == 0


def test_fusion_identity():
    """FFT(Re IFFT(Y)) == (Y + conj(Y[-k])) / 2 for any complex Y, odd or even sizes."""
    rng = np.random.default_rng(0)
    for shape in [(8, 8), (7, 10), (33, 16)]:
        Y = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        lhs = np.fft.fft2(np.real(np.fft.ifft2(Y)))
        rhs = 0.5 * (Y + np.conj(pc._neg_index(Y, np)))
        assert np.allclose(lhs, rhs, atol=1e-12)


def test_kernels_match_phycv_kernel():
    p = phycv.PST(h=64, w=96)
    p.init_kernel(0.4, 20.0)
    _, phase = pc.pst_kernels(64, 96, 0.4, 20.0, 0.1, np, "float64")
    assert np.allclose(phase, np.fft.fftshift(np.exp(-1j * p.pst_kernel)), atol=1e-12)


def test_edges_follow_cracks():
    """The binary edge map lands on/near the crack: most edge pixels are within 6 px of it."""
    img, mask = pc.synthetic_crack_surface(256, n_cracks=2, seed=4)
    out = pc.pst_cuda(img, **PARAMS)
    yy, xx = np.nonzero(mask)
    near = np.zeros_like(mask)
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            near[np.clip(yy + dy, 0, 255), np.clip(xx + dx, 0, 255)] = True
    edges = out > 0.5
    assert edges.sum() > 0
    assert edges[near].mean() > 3 * edges[~near].mean()


def test_kernel_cache_and_resize():
    p = pc.PSTCuda(backend="numpy")
    p.load_img(np.ones((32, 32))); p.init_kernel(0.4, 20.0)
    k = p.pst_phase
    p.init_kernel(0.4, 20.0)
    assert p.pst_phase is k                       # reused
    p.load_img(np.ones((16, 24))); p.init_kernel(0.4, 20.0)
    assert p.pst_phase.shape == (16, 24)          # rebuilt for the new size


def test_validation():
    with pytest.raises(ValueError):
        pc.get_backend("opencl")
    with pytest.raises(ValueError):
        pc.PSTCuda(dtype="float16")
    with pytest.raises(ValueError):
        pc.pst_kernels(64, 64, -1, 20, 0.1)
    with pytest.raises(ValueError):
        pc.pst_cuda(np.ones((32, 32)), thresh_max=1.5)
    with pytest.raises(ValueError):
        pc.synthetic_crack_surface(8)


def test_fit_ns_per_pixel_recovers_line():
    px = np.array([1e4, 1e5, 1e6, 4e6])
    ov, ns, r2 = pc.fit_ns_per_pixel(px, 2e-4 + 3e-9 * px)
    assert abs(ov - 2e-4) < 1e-9 and abs(ns - 3.0) < 1e-6 and r2 > 0.999999
