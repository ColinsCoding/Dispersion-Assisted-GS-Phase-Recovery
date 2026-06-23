"""Build notebooks/single_slit_diffraction.ipynb -- diffraction as Fourier optics."""
import pathlib
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
nb = nbf.v4.new_notebook()

nb.cells = [
md("""# Single-slit diffraction and resolving power -- Fourier optics
### the far-field pattern is the Fourier transform of the aperture

Light through a slit of width $a$ spreads into the Fraunhofer pattern
$$I(\\theta)=I_0\\left(\\frac{\\sin\\beta}{\\beta}\\right)^2,\\qquad \\beta=\\frac{\\pi a\\sin\\theta}{\\lambda},$$
a bright central lobe with dark minima where $a\\sin\\theta=m\\lambda$. The deep fact: this
pattern **is the Fourier transform of the aperture** -- a rectangular slit transforms to
a sinc, squared into an intensity -- the same FFT that runs the dispersion operator and
the GS receiver here. A **narrower slit gives a wider pattern** (the spatial uncertainty
reciprocity), and two sources are just resolvable at the **Rayleigh limit**
$\\theta_{\\min}\\approx\\lambda/D$ -- the resolving power. Uses `dgs/diffraction.py`.
Civilian education."""),

co("""import numpy as np, matplotlib.pyplot as plt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
from dgs import diffraction as df
a, lam = 50e-6, 600e-9          # 50 um slit, 600 nm light
print("ready")"""),

md("""## 1. The single-slit intensity pattern

A tall central maximum flanked by dark minima at $\\sin\\theta=m\\lambda/a$, with side lobes
that fall off fast (the first is only ~4.7% of the center). This is "intensity vs angle"
on a screen."""),
co("""th = np.linspace(-np.radians(3), np.radians(3), 4000)
I = df.single_slit_intensity(th, a, lam)
plt.figure(figsize=(7.5,4))
plt.plot(np.degrees(th), I, lw=2)
for m in df.single_slit_minima(a, lam, 3):
    plt.axvline(np.degrees(m), ls=":", color="C3"); plt.axvline(-np.degrees(m), ls=":", color="C3")
plt.xlabel("angle theta (deg)"); plt.ylabel("intensity / I0")
plt.title("single-slit diffraction: I = (sin beta/beta)^2 (red = minima)"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
print("first side lobe peak =", round(I[(th>df.single_slit_minima(a,lam)[0])&(th<df.single_slit_minima(a,lam)[1])].max(),3), "of center")"""),

md("""## 2. Narrower slit -> wider pattern (reciprocity)

Squeeze the slit and the diffraction pattern *spreads* -- the central lobe half-width is
$\\arcsin(\\lambda/a)$, bigger for smaller $a$. Position and angle (space and spatial
frequency) trade off, exactly like time and bandwidth in `dgs.uncertainty`."""),
co("""plt.figure(figsize=(7,4))
for aw, c in [(100e-6,"C0"), (50e-6,"C1"), (25e-6,"C3")]:
    plt.plot(np.degrees(th), df.single_slit_intensity(th, aw, lam), c,
             label=f"a={aw*1e6:.0f} um (half-width {np.degrees(df.central_lobe_halfwidth(aw,lam)):.2f} deg)")
plt.xlabel("angle (deg)"); plt.ylabel("intensity / I0"); plt.legend()
plt.title("narrower slit -> wider diffraction (space-frequency reciprocity)"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()"""),

md("""## 3. Diffraction IS the Fourier transform of the aperture

No special formula needed: sample any aperture, take its **FFT**, square it, and that is
the far-field pattern. A rectangular slit gives exactly the $\\mathrm{sinc}^2$ above -- they
agree to 6 digits. This is the same FFT as the dispersion operator and GS receiver:
diffraction is Fourier optics."""),
co("""x = np.linspace(-5e-3, 5e-3, 200000); dx = x[1]-x[0]
ap = (np.abs(x) < a/2).astype(float)                      # a rectangular slit
theta, I_fft = df.aperture_diffraction(ap, dx, lam)
I_formula = df.single_slit_intensity(theta, a, lam)
core = np.abs(theta) < np.radians(3)
plt.figure(figsize=(7,4))
plt.plot(np.degrees(theta[core]), I_fft[core], lw=3, alpha=0.4, label="|FFT(slit)|^2")
plt.plot(np.degrees(theta[core]), I_formula[core], "k--", label="sinc^2 formula")
plt.xlabel("angle (deg)"); plt.ylabel("intensity / I0"); plt.legend()
plt.title("the diffraction pattern = Fourier transform of the aperture"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
print("max diff (FFT vs formula) in core:", np.max(np.abs(I_fft-I_formula)[core]))"""),

md("""## 4. Resolving power -- the Rayleigh criterion

Two point sources blur into one when their diffraction patterns overlap too much. The
**Rayleigh criterion** says they are *just* resolved when the peak of one falls on the
first minimum of the other -- a separation $\\theta_{\\min}=\\lambda/D$ (slit) or
$1.22\\lambda/D$ (circular aperture). Bigger aperture or shorter wavelength -> finer
detail. That is why telescopes are huge and microscopes use short wavelengths."""),
co("""sep = df.rayleigh_resolution(lam, a)                  # just-resolvable separation
th2 = np.linspace(-np.radians(2.5), np.radians(2.5), 4000)
A = df.single_slit_intensity(th2 - sep/2, a, lam)
B = df.single_slit_intensity(th2 + sep/2, a, lam)
plt.figure(figsize=(7,4))
plt.plot(np.degrees(th2), A, label="source 1"); plt.plot(np.degrees(th2), B, label="source 2")
plt.plot(np.degrees(th2), A+B, "k", lw=2, label="combined (just resolved)")
plt.xlabel("angle (deg)"); plt.ylabel("intensity"); plt.legend()
plt.title(f"Rayleigh limit: separation = lambda/a = {np.degrees(sep):.2f} deg"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
print(f"Rayleigh resolution at D=100mm (circular): {np.degrees(df.rayleigh_resolution(lam,0.1,True))*3600:.2f} arcsec")"""),

md("""## Takeaway

1. A single slit gives $I=I_0(\\sin\\beta/\\beta)^2$ -- a central lobe, dark minima at
   $a\\sin\\theta=m\\lambda$, fast-decaying side lobes.
2. **Narrower slit -> wider pattern** (space-frequency reciprocity, the diffraction
   uncertainty principle).
3. **Diffraction = the Fourier transform of the aperture** -- the same FFT as the
   dispersion operator / GS receiver. Fourier optics.
4. **Resolving power** $\\theta_{\\min}\\sim\\lambda/D$ sets the finest detail any telescope
   or microscope can see.

A slit, a Fourier transform, a resolution limit -- diffraction is the optics half of the
same Fourier story this whole repo tells. Civilian education."""),
]

nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = pathlib.Path("notebooks/single_slit_diffraction.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(nb.cells), "cells")
