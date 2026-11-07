"""Phase Stretch Transform (PST) on the GPU with CuPy, matched to PhyCV's PST.

In plain words: PST finds edges by giving every spatial frequency of an image a
phase delay that grows with frequency, then reading the phase of the result.
Sharp features (cracks, boundaries) are made of high spatial frequencies, so
they collect the most phase and light up. It is the image version of this
repo's dispersion operator H(nu) = exp(i pi D nu^2): FFT -> multiply by a
phase kernel -> inverse FFT -> look at the phase. A GPU does FFTs and
pixel-by-pixel multiplies very fast, so PST maps onto CUDA almost for free.

What is already out there (checked against phycv 1.2.6, installed on py-3.12):
  * phycv.PST      -- NumPy, float64, CPU.
  * phycv.PST_GPU  -- PyTorch (torch.fft), float32 on CUDA. So phycv already
                      ships a GPU PST; this module is NOT the first one.
  * hardware/phycv_pst_cuda.cu -- this repo's raw CUDA/cuFFT port (one image,
                      checked against a text-file reference).
This module adds what those do not: a CuPy version that (1) caches both
frequency-domain kernels so a video stream pays for them once, (2) can fuse
the denoise step so a frame costs 2 FFTs instead of 4 (see below), and
(3) falls back to NumPy with the identical code path when CuPy or a GPU is
missing. It is benchmarked against phycv.PST and phycv.PST_GPU, not assumed.

The algorithm (phycv/pst.py, line for line):
  1. u, v = linspace(-0.5, 0.5, h), linspace(-0.5, 0.5, w);  rho = hypot(U, V)
  2. L(rho)   = exp(-0.5 * (rho / sqrt(sigma_LPF^2 / ln 2))^2)       (low pass)
  3. phi(rho) = W rho atan(W rho) - 0.5 ln(1 + (W rho)^2);  phi = S phi / max phi
  4. denoised = Re IFFT2( FFT2(img) * fftshift(L) )
  5. img_pst  = IFFT2( FFT2(denoised) * fftshift(exp(-i phi)) )
  6. feature  = normalize(angle(img_pst)) to [0, 1]
  7. optional morph: keep pixels above the thresh_max quantile or below the
     thresh_min quantile (quantiles from feature[::4, ::4]), zero where
     img < max(img)/20.

The fusion trick (mathematically exact, not an approximation): step 4 takes
the real part and step 5 immediately FFTs it again. For any array y,
FFT(Re y)[k] = (Y[k] + conj(Y[-k])) / 2 with Y = FFT(y). Here Y = F * L, so
FFT2(denoised) = (F L + conj((F L)[-k])) / 2 with no inverse/forward FFT pair:
4 FFTs per frame become 2. The [-k] index is a flip plus a one-sample roll.

Precision: the GPU path runs in complex64 by default (the RTX 4060 has ~1/64
rate FP64), phycv.PST runs in float64, so agreement is ~1e-5, not 1e-15.
dtype="float64" is accepted for an exact cross-check.

Also here: synthetic_crack_surface(), a seeded generator of concrete-like
textures with dark random-walk cracks, used as the test and benchmark images.

Civilian computer vision / structural inspection. Education.
"""

import time

import numpy as np

_BACKEND_CACHE = {}


def get_backend(prefer="cupy"):
    """Return (xp, name): CuPy if a working CUDA device exists, else NumPy.

    prefer="numpy" forces the CPU path. The CuPy probe runs one tiny FFT so a
    broken driver/toolkit falls back instead of failing later.
    """
    if prefer not in ("cupy", "numpy"):
        raise ValueError("prefer must be 'cupy' or 'numpy'")
    if prefer == "numpy":
        return np, "numpy"
    if "cupy" not in _BACKEND_CACHE:
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
            cp.fft.fft2(cp.ones((4, 4), cp.complex64)).get()
            _BACKEND_CACHE["cupy"] = cp
        except Exception:                      # import error, no driver, no device
            _BACKEND_CACHE["cupy"] = None
    cp = _BACKEND_CACHE["cupy"]
    return (cp, "cupy") if cp is not None else (np, "numpy")


def _dtypes(dtype):
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    return (np.float32, np.complex64) if dtype == "float32" else (np.float64, np.complex128)


def pst_kernels(h, w, S, W, sigma_LPF, xp=np, dtype="float32"):
    """Both frequency-domain kernels, already fftshifted, on backend xp.

    Returns (lpf, phase): lpf is the real low-pass fftshift(L(rho)), phase is
    the complex fftshift(exp(-i phi(rho))). Same grid as phycv (linspace, not
    fftfreq), so results match phycv pixel for pixel.
    """
    if h < 2 or w < 2:
        raise ValueError("image must be at least 2x2")
    if S <= 0 or W <= 0 or sigma_LPF <= 0:
        raise ValueError("S, W and sigma_LPF must be > 0")
    rdt, cdt = _dtypes(dtype)
    u = xp.linspace(-0.5, 0.5, h, dtype=rdt)
    v = xp.linspace(-0.5, 0.5, w, dtype=rdt)
    U, V = xp.meshgrid(u, v, indexing="ij")
    rho = xp.hypot(U, V)
    Wr = W * rho
    phi = Wr * xp.arctan(Wr) - 0.5 * xp.log(1 + Wr**2)
    phi = S * phi / xp.max(phi)
    lpf = xp.exp(-0.5 * (rho / np.sqrt(sigma_LPF**2 / np.log(2))) ** 2).astype(rdt)
    phase = xp.exp(-1j * phi).astype(cdt)
    return xp.fft.fftshift(lpf), xp.fft.fftshift(phase)


def _neg_index(Y, xp):
    """Y[(-k) mod N] along both axes: flip, then roll by one sample."""
    return xp.roll(Y[::-1, ::-1], 1, axis=(0, 1))


class PSTCuda:
    """PhyCV-compatible PST on CuPy (or NumPy fallback).

    Same method names and parameters as phycv.PST: load_img(img_array=...),
    init_kernel(S, W), apply_kernel(sigma_LPF, thresh_min, thresh_max,
    morph_flag), run(...). Results (pst_feature, pst_output) live on the
    backend; call .to_numpy(x) to bring them to the host.

    backend: "cupy" (falls back to NumPy if unavailable) or "numpy".
    dtype:   "float32" (default, GPU-native) or "float64".
    fused:   True = 2-FFT exact fusion of the denoise step; False = phycv's
             literal 4-FFT sequence.
    """

    def __init__(self, h=None, w=None, backend="cupy", dtype="float32", fused=True):
        self.xp, self.backend = get_backend(backend)
        self.rdt, self.cdt = _dtypes(dtype)
        self.dtype = dtype
        self.fused = bool(fused)
        self.h, self.w = h, w
        self._S = self._W = None
        self._lpf_cache = {}

    def to_numpy(self, x):
        return x.get() if self.backend == "cupy" else np.asarray(x)

    def load_img(self, img_array):
        """Load a 2-D (grayscale) or HxWx3 (RGB, averaged with BT.601) array."""
        img = img_array
        if img.ndim == 3:
            if img.shape[2] != 3:
                raise ValueError("colour images must be HxWx3")
            img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        if img.ndim != 2:
            raise ValueError("image must be 2-D (grayscale) or HxWx3")
        self.img = self.xp.asarray(img, dtype=self.rdt)
        if (self.h, self.w) != self.img.shape:
            self.h, self.w = self.img.shape
            self._S = None                     # size changed: rebuild kernels
            self._lpf_cache = {}

    def init_kernel(self, S, W):
        """Build fftshift(exp(-i phi)) once; reused while (h, w, S, W) stay the same."""
        if (S, W) == (self._S, self._W):
            return
        xp = self.xp
        _, self.pst_phase = pst_kernels(self.h, self.w, S, W, 1.0, xp, self.dtype)
        self._S, self._W = S, W

    def _lpf(self, sigma_LPF):
        if sigma_LPF not in self._lpf_cache:
            self._lpf_cache[sigma_LPF], _ = pst_kernels(self.h, self.w, 1.0, 1.0, sigma_LPF,
                                                        self.xp, self.dtype)
        return self._lpf_cache[sigma_LPF]

    def apply_kernel(self, sigma_LPF, thresh_min, thresh_max, morph_flag):
        if not (0 <= thresh_min <= 1 and 0 <= thresh_max <= 1):
            raise ValueError("thresholds are quantiles in [0, 1]")
        xp = self.xp
        L = self._lpf(sigma_LPF)
        F = xp.fft.fft2(self.img.astype(self.cdt))
        Y = F * L
        if self.fused:
            G = 0.5 * (Y + xp.conj(_neg_index(Y, xp)))          # = FFT2(Re IFFT2(Y)), exactly
        else:
            G = xp.fft.fft2(xp.real(xp.fft.ifft2(Y)).astype(self.cdt))
        self.img_pst = xp.fft.ifft2(G * self.pst_phase)
        a = xp.angle(self.img_pst)
        self.pst_feature = (a - a.min()) / (a.max() - a.min())
        if not morph_flag:
            self.pst_output = self.pst_feature
        else:
            sub = self.pst_feature[::4, ::4]
            q_max = xp.quantile(sub, thresh_max)
            q_min = xp.quantile(sub, thresh_min)
            f = self.pst_feature
            out = ((f > q_max) | (f < q_min)) & (self.img >= xp.max(self.img) / 20)
            self.pst_output = out.astype(xp.float32)
        return self.pst_output

    def run(self, img_array, S, W, sigma_LPF, thresh_min, thresh_max, morph_flag):
        self.load_img(img_array)
        self.init_kernel(S, W)
        return self.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)


def pst_cuda(img, S=0.4, W=20.0, sigma_LPF=0.1, thresh_min=0.0, thresh_max=0.95,
             morph_flag=True, backend="cupy", dtype="float32", fused=True):
    """One-call PST; returns a NumPy array (feature if morph_flag is False, else binary edges)."""
    p = PSTCuda(backend=backend, dtype=dtype, fused=fused)
    return p.to_numpy(p.run(img, S, W, sigma_LPF, thresh_min, thresh_max, morph_flag))


def phycv_reference(img, S=0.4, W=20.0, sigma_LPF=0.1, thresh_min=0.0, thresh_max=0.95,
                    morph_flag=True):
    """The real phycv.PST (CPU, float64) on the same image; returns (feature, output)."""
    from phycv import PST
    p = PST()
    p.load_img(img_array=np.asarray(img, float))
    p.init_kernel(S, W)
    p.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
    return p.pst_feature, p.pst_output


# ----------------------------------------------------------------------------
# Synthetic crack surfaces
# ----------------------------------------------------------------------------

def synthetic_crack_surface(n=256, n_cracks=2, width=2.0, depth=0.6, seed=0):
    """Concrete-like texture with dark random-walk cracks.

    Background: mean 150 grey levels plus band-limited "aggregate" noise
    (white noise low-passed in Fourier space) plus fine grain. Each crack is a
    random walk with slowly drifting heading, drawn as a dark Gaussian line of
    the given width; depth = fractional darkening at the centreline.
    Returns (image float64 in [0, 255], mask bool of pixels within `width` of
    a centreline).
    """
    if n < 16 or n_cracks < 0 or width <= 0 or not 0 < depth <= 1:
        raise ValueError("need n >= 16, n_cracks >= 0, width > 0, 0 < depth <= 1")
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    FR = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    blob = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * np.exp(-(FR / 0.04) ** 2)))
    blob /= blob.std()
    img = 150 + 18 * blob + 6 * rng.standard_normal((n, n))
    dist = np.full((n, n), np.inf)
    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(n_cracks):
        y, x = rng.uniform(0.1 * n, 0.9 * n, 2)
        heading = rng.uniform(0, 2 * np.pi)
        pts = []
        for _ in range(int(1.2 * n)):
            heading += rng.normal(0, 0.12)
            y, x = y + np.sin(heading), x + np.cos(heading)
            if not (0 <= y < n and 0 <= x < n):
                break
            pts.append((y, x))
        pts = np.array(pts[::2])                                 # every 2 px is plenty
        for py, px in pts:
            r0, r1 = int(max(py - 4 * width, 0)), int(min(py + 4 * width + 1, n))
            c0, c1 = int(max(px - 4 * width, 0)), int(min(px + 4 * width + 1, n))
            d = np.hypot(yy[r0:r1, c0:c1] - py, xx[r0:r1, c0:c1] - px)
            dist[r0:r1, c0:c1] = np.minimum(dist[r0:r1, c0:c1], d)
    img *= 1 - depth * np.exp(-0.5 * (dist / width) ** 2)
    return np.clip(img, 0, 255), dist <= width


# ----------------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------------

def _best_of(fn, repeats, sync=None):
    """Best wall time of `repeats` calls (perf_counter), syncing the device if given."""
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        best = min(best, time.perf_counter() - t0)
    return best


def _best_of_events(fn, repeats, cp):
    """Best GPU time of `repeats` calls measured with CUDA events."""
    best = np.inf
    for _ in range(repeats):
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        best = min(best, cp.cuda.get_elapsed_time(start, stop) * 1e-3)
    return best


def benchmark(sizes=(256, 512, 1024, 2048, 4096), repeats=5, S=0.4, W=20.0, sigma_LPF=0.1,
              thresh_min=0.0, thresh_max=0.95, morph_flag=True, include_torch=True, seed=0):
    """Time one PST frame (kernels already built, image already loaded) per method and size.

    Steady-state video-frame cost: init_kernel is outside the timed region for
    every method, matching how a camera stream would use it. GPU methods are
    timed with CUDA events (device time) and also end to end with host upload
    and download (perf_counter + synchronize). Returns a dict of method ->
    list of seconds, plus 'sizes' and 'pixels'. Machine- and load-dependent.
    """
    from phycv import PST
    args = (sigma_LPF, thresh_min, thresh_max, morph_flag)
    out = {"sizes": list(sizes), "pixels": [s * s for s in sizes]}
    cp, name = get_backend("cupy")
    methods = {}

    for s in sizes:
        img, _ = synthetic_crack_surface(s, n_cracks=3, seed=seed)
        r = repeats if s <= 1024 else max(2, repeats // 2)

        p = PST(); p.load_img(img_array=img); p.init_kernel(S, W)
        methods.setdefault("phycv PST (NumPy, CPU f64)", []).append(
            _best_of(lambda: p.apply_kernel(*args), r))
        del p

        q = PSTCuda(backend="numpy", dtype="float32", fused=True)
        q.load_img(img); q.init_kernel(S, W); q.apply_kernel(*args)
        methods.setdefault("pst_cuda NumPy fallback (CPU f32, fused)", []).append(
            _best_of(lambda: q.apply_kernel(*args), r))
        del q

        if name == "cupy":
            for fused in (False, True):
                g = PSTCuda(backend="cupy", dtype="float32", fused=fused)
                g.load_img(img); g.init_kernel(S, W); g.apply_kernel(*args)   # warm-up: FFT plans
                tag = "fused 2-FFT" if fused else "4-FFT"
                methods.setdefault(f"pst_cuda CuPy ({tag}, device)", []).append(
                    _best_of_events(lambda: g.apply_kernel(*args), repeats, cp))

                def e2e(g=g):
                    g.load_img(img)
                    g.apply_kernel(*args)
                    return g.to_numpy(g.pst_output)
                methods.setdefault(f"pst_cuda CuPy ({tag}, with host copies)", []).append(
                    _best_of(e2e, repeats, cp.cuda.runtime.deviceSynchronize))
                del g
            cp.get_default_memory_pool().free_all_blocks()

        if include_torch:
            try:
                import torch
                from phycv import PST_GPU
                if torch.cuda.is_available():
                    dev = torch.device("cuda")
                    t = PST_GPU(device=dev)
                    t.load_img(img_array=torch.from_numpy(img.astype(np.float32)).to(dev))
                    t.init_kernel(S, W); t.apply_kernel(*args)
                    torch.cuda.synchronize()
                    methods.setdefault("phycv PST_GPU (PyTorch, device)", []).append(
                        _best_of(lambda: t.apply_kernel(*args), repeats, torch.cuda.synchronize))
                    del t
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    out.update(methods)
    return out


def fit_ns_per_pixel(pixels, times):
    """time = overhead + slope * pixels (least squares, dgs.rigid_body_harsh.regress_line).

    Returns (overhead_s, ns_per_pixel, r2). FFTs are N log N, so a straight
    line is an approximation; r2 says how good it is over the range.
    """
    from dgs.rigid_body_harsh import regress_line
    ov, slope, r2 = regress_line(pixels, times)
    return ov, slope * 1e9, r2


if __name__ == "__main__":
    img, mask = synthetic_crack_surface(512, seed=1)
    ref_f, ref_o = phycv_reference(img)
    xp, name = get_backend()
    print("backend:", name)
    for fused in (False, True):
        g = PSTCuda(dtype="float32", fused=fused)
        g.run(img, 0.4, 20.0, 0.1, 0.0, 0.95, True)
        f, o = g.to_numpy(g.pst_feature), g.to_numpy(g.pst_output)
        print(f"fused={fused}: max|feature diff| = {np.abs(f - ref_f).max():.2e}, "
              f"edge-map pixels differing = {int((o != ref_o).sum())} / {o.size}")
