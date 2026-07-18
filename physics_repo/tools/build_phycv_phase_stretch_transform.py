"""Generate notebooks/phycv_phase_stretch_transform.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# PhyCV: the Phase Stretch Transform -- physics-inspired edge detection

The Jalali Lab (UCLA) built **PhyCV**, a computer-vision library whose algorithms are literal optical
physics: the flagship **Phase Stretch Transform (PST)** detects edges by passing an image through a
*dispersive phase operator* and reading out the **phase** -- exactly the operator family this repository
already uses. Where the time-stretch instrument applies $H_D(f)=e^{\,i\pi Df^2}$ and recovers phase, PST
applies a **warped, nonlinear** all-pass phase $H(\boldsymbol\rho)=e^{\,i\phi(\rho)}$ in the 2-D spatial-
frequency domain and takes the phase of the result as the edge map. Same idea -- an all-pass phase filter
plus a phase readout -- moved from a fiber to an image.

(PhyCV is **open-source** and the algorithm is published, so there is nothing to license or copy; we
implement PST from the equations, self-contained.)

The PST phase kernel, isotropic in spatial frequency $\rho=\sqrt{u^2+v^2}$, is
$$\phi(\rho)=S\,\Big[\,W\rho\,\arctan(W\rho)-\tfrac12\ln\!\big(1+(W\rho)^2\big)\Big]\Big/\max(\cdot),$$
a strength-$S$, warp-$W$ dispersive phase whose *derivative* $d\phi/d\rho=S\arctan(W\rho)$ grows with
frequency -- it "stretches" the high-frequency (edge) content in phase. With a localization low-pass to
suppress noise, the output phase forms bright ridges along edges. We implement it, verify the operator is
shift-correct via its impulse response, and show the phase response is several times stronger on true
edges than elsewhere. Self-contained: NumPy, SciPy, Matplotlib."""),
setup_cell(),
co("""from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion"""),

md(r"""## The PST operator

Build the isotropic frequency grid, a Gaussian **localization** low-pass $L(\rho)=e^{-\rho^2/2\sigma^2}$
(denoising), and the **phase kernel** $\phi(\rho)$ above. The transform is
$$\text{PST}[I]=\arg\Big(\mathcal F^{-1}\big\{\,L(\rho)\,e^{\,i\phi(\rho)}\,\mathcal F\{I\}\big\}\Big).$$
Because $\phi$ depends only on $\rho$, this is an all-pass, isotropic, dispersive filter -- the 2-D analogue
of the 1-D dispersion operator."""),
co("""def pst(image, strength=4.0, warp=15.0, sigma_lpf=0.2):
    m, n = image.shape
    u = np.fft.fftfreq(m)[:, None]; v = np.fft.fftfreq(n)[None, :]
    rho = np.sqrt(u**2 + v**2)                                   # isotropic spatial frequency
    lpf = np.exp(-0.5*(rho/sigma_lpf)**2)                        # localization (denoise)
    wr = warp*rho
    phase_kernel = strength*(wr*np.arctan(wr) - 0.5*np.log(1 + wr**2))
    phase_kernel = phase_kernel/phase_kernel.max()*strength     # normalize, scale by strength
    filtered = np.fft.ifft2(np.fft.fft2(image)*lpf*np.exp(1j*phase_kernel))
    return np.angle(filtered)                                   # the phase IS the edge feature

# the phase kernel's derivative is S*arctan(W rho): high frequencies are stretched most
rho_1d = np.linspace(0, 0.7, 200)
print("phase-kernel slope d(phi)/d(rho) grows with frequency (stretches edges):",
      f"{4.0*np.arctan(15*0.05):.3f} at rho=0.05  ->  {4.0*np.arctan(15*0.5):.3f} at rho=0.5")"""),

md(r"""## The operator is shift-correct (impulse response)

A shift-invariant filter must map an impulse to a response centered on that impulse. We confirm the PST
operator's magnitude response peaks exactly at the input impulse -- so any structure it highlights is
located where the image feature actually is."""),
co("""m = n = 128
impulse = np.zeros((m, n)); impulse[64, 64] = 1.0
u = np.fft.fftfreq(m)[:, None]; v = np.fft.fftfreq(n)[None, :]; rho = np.sqrt(u**2+v**2)
# the same normalized phase kernel pst() uses (strength scaled, /max); apply it to an impulse
wr = 15.0*rho
pk = wr*np.arctan(wr) - 0.5*np.log(1 + wr**2); pk = pk/pk.max()          # normalized, unit strength
ker = np.exp(-0.5*(rho/0.2)**2)*np.exp(1j*pk)
resp = np.abs(np.fft.ifft2(np.fft.fft2(impulse)*ker))
peak = tuple(int(k) for k in np.unravel_index(np.argmax(resp), resp.shape))
print("impulse-response peak at", peak, " (input impulse at (64, 64)) -> shift-correct")
assert peak == (64, 64)"""),

md(r"""## Edge detection on a test image

Build a synthetic image (a disk and a bar) on a small DC pedestal, denoise lightly, and apply PST. Against
a ground-truth edge mask, the **mean phase response is several times larger on edges than off** them, and a
simple threshold recovers the boundaries. We report the on/off contrast and precision/recall."""),
co("""M = N = 200
yy, xx = np.mgrid[0:M, 0:N]
shape = np.zeros((M, N), bool)
shape[(xx - 70)**2 + (yy - 70)**2 < 35**2] = True                # disk
shape[120:160, 40:170] = True                                    # bar
rng = np.random.default_rng(0)
img = 0.2 + 0.8*shape                                            # DC pedestal
img = gaussian_filter(img, 1.0) + 0.01*rng.standard_normal((M, N))

edge_gt = binary_dilation(shape, iterations=2) & ~binary_erosion(shape, iterations=2)  # true boundary
phase = pst(img, strength=4.0, warp=15.0, sigma_lpf=0.2)
resp = np.abs(phase)

contrast = resp[edge_gt].mean()/resp[~edge_gt].mean()
detected = resp > resp.mean() + resp.std()
tp = np.logical_and(detected, edge_gt).sum()
precision = tp/detected.sum(); recall = tp/edge_gt.sum()
print(f"on-edge / off-edge phase response = {contrast:.2f}x")
print(f"thresholded edge map: precision {precision:.2f}, recall {recall:.2f}")
assert contrast > 3.0 and recall > 0.6"""),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 4, figsize=(15, 4))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("input image"); ax[0].axis("off")
ax[1].imshow(resp, cmap="magma"); ax[1].set_title("PST phase response |arg|"); ax[1].axis("off")
ax[2].imshow(detected, cmap="gray"); ax[2].set_title("thresholded edges"); ax[2].axis("off")
gy, gx = np.gradient(gaussian_filter(img, 1.0)); gmag = np.sqrt(gx**2 + gy**2)
ax[3].imshow(gmag, cmap="gray"); ax[3].set_title("gradient magnitude (baseline)"); ax[3].axis("off")
plt.tight_layout(); plt.show()"""),

md(r"""## Connection to this project, and PST's siblings

PST is the **same all-pass dispersive-phase operator** as the time-stretch dispersion $H_D(f)=e^{i\pi
Df^2}$ this repository does phase retrieval for -- moved to 2-D spatial frequency and given a **warped
$\arctan$ phase** instead of a quadratic one. Both are unitary phase-only filters; both put the information
in the **phase**, which is then read out (edge map here, recovered field there). The Jalali-lab
"physics-inspired vision" family extends the idea:

- **PST** (this notebook): dispersive phase -> edges.
- **PAGE** (Phase-stretch Adaptive Gradient-field Extractor): oriented, directional phase kernels -> edges
  with orientation.
- **VEViD** (Vision Enhancement via Virtual diffraction and coherent Detection): a phase operator for
  low-light enhancement and colour.

Because the operator is a phase-only FFT filter, it is fast, differentiable, and portable -- and it slots
directly into the feature-extraction stage of the pipeline in `photonics_ml_pipeline/`.

## Summary

- The **Phase Stretch Transform** applies a warped, nonlinear all-pass phase $e^{i\phi(\rho)}$ in the
  spatial-frequency domain and reads the **phase** of the result as an edge feature -- optical physics used
  as a vision algorithm (PhyCV, open-source).
- The operator is shift-correct (impulse response centered) and, on a test image, gives an on-edge phase
  response $>3\times$ the off-edge level, recovering boundaries by a simple threshold.
- It is the **2-D warped-phase cousin** of this project's dispersion operator: all-pass phase filter,
  phase readout -- the through-line from the fiber to the image.

Subject-verb-object: the kernel stretches the phase; the high frequencies accumulate the most; the edges
light up; a threshold reads them off."""),
]

write("phycv", "phase_stretch_transform", cells)
