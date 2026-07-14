"""Generate notebooks/realtime_vr_photonics.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# Real-time photonics for VR optics with CUDA on Windows

A holographic near-eye display forms an image by shaping the phase of a wavefront with a spatial light
modulator (SLM), then letting it propagate to the eye. Two problems must be solved every frame, inside
a $\sim11\ \mathrm{ms}$ budget at $90\ \mathrm{Hz}$:

1. **Propagate** the SLM field to the eye plane -- free-space diffraction, computed with the angular
   spectrum method.
2. **Design** the SLM phase so the propagated intensity matches the target image -- a phase-retrieval
   problem solved by **Gerchberg-Saxton**, exactly the algorithm behind the dispersion phase-recovery
   notebook.

Both are dominated by two-dimensional FFTs, so the frame budget is a GPU question: this notebook
builds the wave-optics forward model and the GS hologram synthesis in NumPy (the reference), then
sizes the real-time compute and, when a CUDA build of PyTorch is present, runs it on the GPU. The
NumPy path is authoritative; CUDA is the accelerator.

Self-contained: NumPy, SymPy, Pandas, Matplotlib; optional PyTorch."""),
setup_cell(),

md(r"""## Free-space propagation (angular spectrum)

A field $u(x,y)$ propagates a distance $z$ by multiplying its 2-D spectrum by the free-space transfer
function $H(f_x,f_y)=\exp\!\big(i k z\sqrt{1-(\lambda f_x)^2-(\lambda f_y)^2}\big)$, then transforming
back. It is all-pass for propagating waves ($|H|=1$) and evanescent beyond the cutoff -- the 2-D twin
of the dispersion all-pass. This is the display-to-eye model."""),
co("""def angular_spectrum(u, dx, z, lam):
    ny, nx = u.shape
    fx = np.fft.fftfreq(nx, d=dx); fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    arg = 1 - (lam*FX)**2 - (lam*FY)**2
    kz = (2*np.pi/lam) * np.sqrt(arg.astype(complex))
    H = np.exp(1j*z*kz)
    H[arg < 0] = 0                                # drop evanescent (band-limited ASM)
    return np.fft.ifft2(np.fft.fft2(u) * H)

lam = 0.5e-6; dx = 8e-6                            # 8 um SLM pixels, green light
u0 = np.zeros((128, 128), complex); u0[60:68, 60:68] = 1.0     # a small aperture
uz = angular_spectrum(u0, dx, z=2e-3, lam=lam)
print("energy in  =", round(np.sum(np.abs(u0)**2), 3),
      " energy out =", round(np.sum(np.abs(uz)**2), 3), " (propagation conserves energy)")
assert np.isclose(np.sum(np.abs(uz)**2), np.sum(np.abs(u0)**2), rtol=1e-6)"""),

md(r"""## Computer-generated holography = Gerchberg-Saxton

A phase-only SLM emits $u=e^{i\varphi}$ under uniform illumination. Its far-field (a lens Fourier
transform) is $U=\mathcal F\{u\}$, and we want $|U|$ to equal a target image amplitude. GS alternates:
enforce the target amplitude in the image plane (keep phase), transform back, and enforce unit
amplitude in the SLM plane (keep phase). After a few dozen iterations the phase $\varphi$ is a hologram
whose reconstruction reproduces the target."""),
co("""def make_target(n=128):
    y, x = np.mgrid[0:n, 0:n] - n/2
    r = np.hypot(x, y)
    ring = ((r > 22) & (r < 30)).astype(float)     # a ring...
    dot = (np.hypot(x, y-0) < 6).astype(float)*0    # (kept simple)
    t = ring + (np.abs(x) < 3).astype(float)*((np.abs(y) < 30))   # ring + a vertical bar
    return t/ t.max()

target = make_target()
tgt_amp = np.sqrt(target)                          # target amplitude in the image plane

def gs_hologram(tgt_amp, n_iter=60, seed=0):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2*np.pi, tgt_amp.shape)   # random initial SLM phase
    for _ in range(n_iter):
        U = np.fft.fftshift(np.fft.fft2(np.exp(1j*phi)))   # far-field (image plane)
        U = tgt_amp * np.exp(1j*np.angle(U))                # enforce target amplitude
        u = np.fft.ifft2(np.fft.ifftshift(U))               # back to SLM plane
        phi = np.angle(u)                                   # phase-only SLM constraint
    return phi

phi = gs_hologram(tgt_amp)
recon = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j*phi))))**2
recon /= recon.max()
# quality: correlation of reconstruction with the target over the image
mask = target > 0.5
corr = np.corrcoef(recon.ravel(), target.ravel())[0, 1]
eff = recon[mask].sum() / recon.sum()               # fraction of light in the target
print(f"reconstruction-target correlation = {corr:.3f}")
print(f"diffraction efficiency (light in target) = {eff:.1%}")
assert corr > 0.5"""),

md(r"""## Real-time budget: why this needs a GPU

Each frame runs GS ($\sim n_{\rm iter}$ iterations, two 2-D FFTs each) on an $N\times N$ SLM. A 2-D FFT
costs $\approx 5N^2\log_2(N^2)$ FLOPs, so the per-frame work times the $90\ \mathrm{Hz}$ rate gives the
sustained throughput. For a $1024\times1024$ panel it lands in the hundreds of GFLOP/s -- beyond a
comfortable CPU budget, squarely in GPU territory. That is the reason for CUDA."""),
co("""def frame_flops(N, n_iter):
    fft2 = 5 * N*N * np.log2(N*N)
    return n_iter * 2 * fft2                          # two FFTs per GS iteration
rows = []
for N in (256, 512, 1024, 2048):
    F = frame_flops(N, 20)
    rows.append({"SLM N x N": f"{N}x{N}", "FLOP/frame": f"{F:.2e}",
                 "GFLOP/s @ 90 Hz": round(F*90/1e9, 1),
                 "CPU (~200 GF/s)?": "ok" if F*90 < 2e11 else "too slow",
                 "GPU (~15 TF/s)?": "ok" if F*90 < 1.5e13 else "too slow"})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
print("frame budget at 90 Hz =", round(1000/90, 2), "ms")"""),

md(r"""## Optional CUDA (PyTorch) on Windows

When a CUDA build of PyTorch is installed and a GPU is visible, the same FFTs run on the device. The
cell is guarded: if PyTorch or CUDA is unavailable (as on a CPU-only or memory-constrained Windows
box), it reports so and the NumPy reconstruction above remains the result. On a working RTX-class GPU
the hundreds of GFLOP/s are delivered with large headroom inside the 11 ms frame."""),
co("""if torch is not None and torch.cuda.is_available():
    dev = "cuda"
    a = torch.rand(1024, 1024, dtype=torch.complex64, device=dev)
    import time
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(40):                              # 20 iterations x 2 FFTs
        a = torch.fft.fft2(a)
    torch.cuda.synchronize()
    dt = (time.time()-t0)/1
    print(f"GPU: 40 x 1024^2 FFTs in {dt*1e3:.1f} ms  -> {'meets' if dt < 1/90 else 'misses'} the 11 ms budget")
elif torch is not None:
    print("PyTorch present but no CUDA device -- CPU FFT path; the NumPy reconstruction stands.")
else:
    print("PyTorch absent -- running CPU-only. NumPy computed the hologram; a CUDA GPU would accelerate it.")
# CPU reference timing of one frame's FFTs (always available)
import time
a = np.random.rand(1024, 1024) + 0j
t0 = time.time()
for _ in range(40): a = np.fft.fft2(a)
print(f"CPU: 40 x 1024^2 FFTs in {(time.time()-t0)*1e3:.0f} ms")"""),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].imshow(target, cmap="gray"); ax[0].set_title("target image"); ax[0].axis("off")
ax[1].imshow(phi, cmap="twilight"); ax[1].set_title("SLM phase hologram (rad)"); ax[1].axis("off")
ax[2].imshow(recon**0.5, cmap="gray"); ax[2].set_title("reconstruction from the hologram"); ax[2].axis("off")
plt.tight_layout(); plt.show()""" ),

md(r"""## Summary

- A holographic VR display **propagates** an SLM field (angular spectrum, all-pass, energy-conserving)
  and **designs** the SLM phase by **Gerchberg-Saxton** so the reconstruction matches a target -- the
  same phase-retrieval loop as the dispersion notebook, now in two dimensions.
- The frame is FFT-bound: at $1024^2$ and $90\ \mathrm{Hz}$ the work is hundreds of GFLOP/s, which is
  why real-time holographic VR is a **GPU/CUDA** task, not a CPU one.
- The computation is written NumPy-first (authoritative, portable) with an optional CUDA path; on this
  Windows box PyTorch/CUDA may be unavailable, and the NumPy result still stands.

Subject-verb-object: the SLM shapes the phase; propagation forms the image; Gerchberg-Saxton designs
the hologram; the GPU computes the FFTs; the eye sees the reconstruction."""),
]

write("realtime_vr", "photonics", cells)
