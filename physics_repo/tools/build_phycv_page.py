"""Generate notebooks/phycv_page_directional_edges.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# PhyCV PAGE: directional edge detection with a bank of oriented phase kernels

**PAGE** (Phase-stretch Adaptive Gradient-field Extractor) is the oriented sibling of the Phase Stretch
Transform. Where PST uses one isotropic phase kernel, PAGE uses a **bank of directional phase kernels**
$\phi_k(\rho,\theta)=\phi_r(\rho)\,D_k(\theta)$: the same radial phase-stretch profile $\phi_r$ as PST,
multiplied by an **angular window** $D_k(\theta)$ centered on frequency orientation $\theta_k$. Each kernel
responds to edges of one orientation, so the transform returns not just *where* the edges are but *which
way they run* -- an orientation-encoded edge field, usually shown in HSV (hue = orientation, value =
strength).

It is the same all-pass dispersive-phase operator this project uses for time-stretch, now made
**anisotropic** in 2-D spatial frequency. Linear-algebra view: PAGE is a bank of linear FFT filters
followed by a per-pixel argmax over orientation. PhyCV is open-source; we build PAGE from the equations.

Kernel: $\phi_r(\rho)=S\big[W\rho\arctan(W\rho)-\tfrac12\ln(1+(W\rho)^2)\big]/\max(\cdot)$, and
$D_k(\theta)=e^{-\Delta\theta_k^2/2\sigma_\theta^2}+e^{-\Delta(\theta_k+\pi)^2/2\sigma_\theta^2}$ (symmetric
in $\theta$, so opposite frequencies pair up). Self-contained: NumPy, SciPy, Matplotlib."""),
setup_cell(),
co("""from scipy.ndimage import gaussian_filter
from matplotlib.colors import hsv_to_rgb"""),

md(r"""## The oriented phase-kernel bank

Build the frequency grid with radius $\rho$ and angle $\theta$, the PST radial phase $\phi_r(\rho)$, and a
directional window for each orientation. `page` returns the stack of per-orientation phase responses."""),
co("""def page(image, orientations, strength=3.0, warp=15.0, sigma_lpf=0.3, sigma_theta=0.4):
    m, n = image.shape
    u = np.fft.fftfreq(m)[:, None]; v = np.fft.fftfreq(n)[None, :]
    rho = np.sqrt(u**2 + v**2)
    theta = np.arctan2(v*np.ones_like(u), u*np.ones_like(v))         # frequency-plane angle
    lpf = np.exp(-0.5*(rho/sigma_lpf)**2)
    wr = warp*rho
    radial = strength*(wr*np.arctan(wr) - 0.5*np.log(1 + wr**2))
    radial = radial/radial.max()*strength                           # PST-style radial phase
    ang_diff = lambda a, b: np.arctan2(np.sin(a - b), np.cos(a - b))
    fft = np.fft.fft2(image)
    responses = []
    for tk in orientations:
        window = (np.exp(-0.5*(ang_diff(theta, tk)/sigma_theta)**2)
                  + np.exp(-0.5*(ang_diff(theta, tk + np.pi)/sigma_theta)**2))   # symmetric in theta
        out = np.fft.ifft2(fft*lpf*np.exp(1j*radial*window))
        responses.append(np.abs(np.angle(out)))
    return np.asarray(responses)

orientations = np.linspace(0, np.pi, 8, endpoint=False)
print("orientation bank (deg):", np.round(np.degrees(orientations), 1))"""),

md(r"""## Verification: single-orientation edges land in distinct orientation bins

Three test images -- a horizontal, a vertical, and a diagonal edge -- should excite **different** kernels.
The dominant orientation (the bin with the most total phase response) tracks the edge direction, confirming
PAGE is orientation-selective."""),
co("""M = N = 160
grid_r, grid_c = np.mgrid[0:M, 0:N]
def make_edge(kind):
    img = np.zeros((M, N))
    if kind == "horizontal": img[M//2:, :] = 1.0
    elif kind == "vertical": img[:, N//2:] = 1.0
    else: img[grid_r > grid_c] = 1.0                                # diagonal
    return 0.2 + 0.8*gaussian_filter(img, 1.0)

dominant = {}
for kind in ("horizontal", "vertical", "diagonal"):
    resp = page(make_edge(kind), orientations)
    energy = resp.reshape(len(orientations), -1).sum(axis=1)
    dominant[kind] = float(np.degrees(orientations[int(np.argmax(energy))]))
    print(f"{kind:11s} edge -> dominant orientation {dominant[kind]:.1f} deg")
assert dominant["horizontal"] != dominant["vertical"] != dominant["diagonal"]
assert dominant["horizontal"] == 0.0 and dominant["vertical"] == 90.0"""),

md(r"""## Orientation-encoded edge map (HSV)

On a multi-edge image, assign each pixel the orientation of its strongest kernel (hue) weighted by the edge
strength (value). The result colors every edge by the direction it runs."""),
co("""shape = np.zeros((M, N))
shape[40:120, 40:44] = 1.0                                          # vertical bar
shape[40:44, 40:120] = 1.0                                          # horizontal bar
for d in range(80): shape[40 + d, 118 - d] = 1.0                    # diagonal stroke
img = 0.2 + 0.8*gaussian_filter(shape, 1.0)

resp = page(img, orientations)
strength = resp.max(axis=0)
orient_idx = resp.argmax(axis=0)
hue = orient_idx/len(orientations)
val = np.clip(strength/np.percentile(strength, 99.5), 0, 1)
hsv = np.stack([hue, np.ones_like(hue), val], axis=-1)
rgb = hsv_to_rgb(hsv)
print("orientation map built: hue = edge orientation, brightness = edge strength")
print("distinct orientations present:", len(np.unique(orient_idx[val > 0.3])))"""),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 4, figsize=(15, 4))
for a, kind in zip(ax[:3], ("horizontal", "vertical", "diagonal")):
    r = page(make_edge(kind), orientations).max(axis=0)
    a.imshow(r, cmap="magma"); a.set_title(f"{kind} edge\n-> {dominant[kind]:.0f} deg"); a.axis("off")
ax[3].imshow(rgb); ax[3].set_title("PAGE orientation map (HSV)"); ax[3].axis("off")
plt.tight_layout(); plt.show()"""),

md(r"""## Summary

- **PAGE** replaces PST's single isotropic phase kernel with a **bank of oriented** kernels
  $\phi_r(\rho)D_k(\theta)$ -- the PST radial phase times an angular window -- so the phase response is
  **direction-selective**.
- A per-pixel argmax over orientation yields an **orientation-encoded edge map** (hue = direction); on
  single-orientation test edges the dominant bin correctly tracks the edge (horizontal 0 deg, vertical
  90 deg, diagonal in between).
- Structurally it is the same all-pass dispersive-phase operator as this project's $H_D(f)$, made
  anisotropic: a linear FFT filter bank plus an argmax. Companion notebooks: `phycv_phase_stretch_transform`
  (PST) and `phycv_vevid` (low-light enhancement).

Subject-verb-object: the angular window steers the phase kernel; each orientation lights its own edges; the
argmax reads the direction; the HSV map paints it."""),
]

write("phycv_page", "directional_edges", cells)
