"""Generate notebooks/phycv_vevid_low_light.ipynb."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nbkit import md, co, setup_cell, write

cells = [
md(r"""# PhyCV VEViD: low-light enhancement by virtual diffraction and coherent detection

**VEViD** (Vision Enhancement via Virtual diffraction and coherent Detection) is the third PhyCV algorithm:
instead of finding edges, it **brightens dark images**. Its physics idea is to treat pixel brightness as a
**phase**, add a small regularization bias (the "virtual diffraction" offset), and read out an enhanced
image from the **phase of a coherent field** (coherent detection). The result lifts shadows while
compressing highlights -- tone mapping derived from an optical phase operator rather than an ad-hoc curve.

We implement the fast closed form **VEViD-lite**, a pointwise phase map on the brightness (value) channel,
$$V_{\text{out}}=\frac{\arctan\!\big(g\,(V+b)\big)}{\arctan\!\big(g\,(1+b)\big)},$$
with gain $g$ and bias $b$. The $\arctan$ is the coherent phase readout; the bias $b$ is the diffraction
regularization that keeps dark pixels ($V\to0$) from mapping to zero. It is monotonic (structure
preserved), lifts shadows, and compresses highlights. PhyCV is open-source; this is built from the idea.
Self-contained: NumPy, SciPy, Matplotlib. (Colour images: apply to the HSV value channel only.)"""),
setup_cell(),
co("""from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr"""),

md(r"""## VEViD-lite: the phase tone map

A single vectorized function on the normalized brightness channel. The mapping is monotonic in $V$ (so it
never reorders brightness / destroys structure), lifts the shadows via the bias, and rolls off the
highlights via the $\arctan$ saturation."""),
co("""def vevid_lite(value, gain=8.0, bias=0.1):
    \"\"\"Phase-derived tone map on a normalized [0,1] brightness channel.\"\"\"
    value = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return np.arctan(gain*(value + bias))/np.arctan(gain*(1.0 + bias))

# shadow lift and highlight compression: small inputs gain more than large ones
for v in (0.02, 0.1, 0.5, 0.9):
    print(f"V={v:.2f} -> {float(vevid_lite(np.array(v))):.3f}  (gain factor {float(vevid_lite(np.array(v)))/v:.1f}x)")"""),

md(r"""## Verification on a dark image

Build a very dark synthetic scene with faint hidden detail. VEViD-lite should (1) raise the mean
brightness, (2) *increase* the contrast (std) inside the dark regions -- revealing detail -- and (3) stay
strictly monotonic (Spearman correlation 1.0), so no structure is inverted or lost."""),
co("""M = N = 160
yy, xx = np.mgrid[0:M, 0:N]
scene = 0.05 + 0.10*np.exp(-((xx - 80)**2 + (yy - 80)**2)/(2*40**2))   # dark background + faint blob
scene[40:60, 40:120] += 0.06                                          # faint detail buried in shadow
V = np.clip(scene, 0, 1)
E = vevid_lite(V, gain=8.0, bias=0.1)

dark = V < 0.15
mean_gain = E.mean()/V.mean()
contrast_gain = E[dark].std()/V[dark].std()
rho = float(spearmanr(V.ravel(), E.ravel()).correlation)
print(f"mean brightness: {V.mean():.3f} -> {E.mean():.3f}  ({mean_gain:.1f}x)")
print(f"dark-region contrast (std): {V[dark].std():.4f} -> {E[dark].std():.4f}  ({contrast_gain:.1f}x)")
print(f"monotonic (Spearman) = {rho:.4f}  -> structure preserved")
assert mean_gain > 2.0 and contrast_gain > 1.2 and rho > 0.999"""),

md(r"""## Bias and gain control the enhancement

The bias $b$ sets how aggressively shadows are lifted (larger $b$ = brighter shadows); the gain $g$ sets the
overall contrast/roll-off. Sweeping them shows the tunable trade-off."""),
co("""import pandas as pd
rows = []
for g in (4.0, 8.0, 16.0):
    for b in (0.05, 0.2):
        out = vevid_lite(V, gain=g, bias=b)
        rows.append({"gain": g, "bias": b, "mean out": round(float(out.mean()), 3),
                     "dark std out": round(float(out[dark].std()), 4)})
print(pd.DataFrame(rows).to_string(index=False))"""),

md(r"""## Plots"""),
co(r"""fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
ax[0].imshow(V, cmap="gray", vmin=0, vmax=1); ax[0].set_title("dark input"); ax[0].axis("off")
ax[1].imshow(E, cmap="gray", vmin=0, vmax=1); ax[1].set_title("VEViD-lite enhanced"); ax[1].axis("off")
ax[2].hist(V.ravel(), bins=50, alpha=0.6, label="input", color="#4C78A8")
ax[2].hist(E.ravel(), bins=50, alpha=0.6, label="enhanced", color="#E45756")
ax[2].set_xlabel("brightness"); ax[2].set_ylabel("count"); ax[2].set_title("histogram: shadows lifted")
ax[2].legend(fontsize=9)
plt.tight_layout(); plt.show()"""),

md(r"""## Summary

- **VEViD** enhances low-light images by treating brightness as **phase**, adding a diffraction
  regularization bias, and reading out an enhanced image via a **coherent (arctan) phase detection** -- a
  tone map with an optical-physics origin.
- **VEViD-lite** is the pointwise closed form $V_{\text{out}}=\arctan(g(V+b))/\arctan(g(1+b))$: monotonic
  (Spearman 1.0), it lifts mean brightness (>2x here) and *increases* dark-region contrast (detail
  revealed), tunable through gain $g$ and bias $b$.
- With PST (edges) and PAGE (oriented edges), VEViD completes the PhyCV trio -- all phase operators from the
  same all-pass family this project uses for dispersion. Apply to the HSV value channel for colour images.

Subject-verb-object: the bias lifts the shadow; the arctan detects the phase; the highlights compress; the
dark detail appears."""),
]

write("phycv_vevid", "low_light", cells)
