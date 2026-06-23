"""Build notebooks/z_transform_discrete_calculus.ipynb -- continuous -> discrete calculus."""
import pathlib
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
nb = nbf.v4.new_notebook()

nb.cells = [
md("""# The Z-transform -- calculus for discrete (sampled) signals
### from continuous calculus to digital filters, with the same Fundamental Theorem

Continuous calculus has the Laplace transform ($d/dt \\leftrightarrow s$); sampled data
has the **Z-transform** $X(z)=\\sum_n x[n]z^{-n}$, the discrete analog. The dictionary:

| continuous | Laplace | discrete | Z |
|---|---|---|---|
| derivative $d/dt$ | $s$ | backward difference | $1-z^{-1}$ |
| integral $\\int dt$ | $1/s$ | accumulator (running sum) | $1/(1-z^{-1})$ |

And the **Fundamental Theorem survives discretization**: difference and accumulator
multiply to $(1-z^{-1})\\cdot\\frac{1}{1-z^{-1}}=1$, so accumulating a difference returns
the signal. A digital filter is a difference equation, its transfer function
$H(z)=B(z)/A(z)$ a ratio of polynomials; it is **stable iff every pole is inside the unit
circle** (the discrete version of 'poles in the left half-plane'). This is the math under
every DSP receiver. Uses `dgs/z_transform.py`. Civilian education."""),

co("""import numpy as np, matplotlib.pyplot as plt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
from dgs import z_transform as zt
print("ready")"""),

md("""## 1. The discrete derivative and integral as filters

The backward **difference** $1-z^{-1}$ is the discrete derivative: it removes DC and
amplifies high frequencies (a high-pass, $|H|=2|\\sin(\\omega/2)|$). The **accumulator**
$1/(1-z^{-1})$ is the discrete integral: it boosts low frequencies and rolls off (a
leaky-looking low-pass with a pole at DC)."""),
co("""omega = np.linspace(0.001, np.pi, 500)
Hd = np.abs(zt.frequency_response(*zt.DIFFERENCE, omega))
Ha = np.abs(zt.frequency_response(*zt.ACCUMULATOR, omega))
plt.figure(figsize=(7,4))
plt.plot(omega, Hd, lw=2, label="difference 1 - z^-1 (discrete d/dt, high-pass)")
plt.plot(omega, Ha, lw=2, label="accumulator 1/(1 - z^-1) (discrete integral, low-pass)")
plt.xlabel("normalized frequency omega (0..pi)"); plt.ylabel("|H(e^jw)|"); plt.yscale("log")
plt.legend(); plt.title("discrete derivative vs integral: frequency response"); plt.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.show()
print("difference removes DC: H(z=1) =", zt.filter_response(*zt.DIFFERENCE, 1.0))"""),

md("""## 2. The discrete Fundamental Theorem of Calculus

Differentiate then integrate and you get back exactly what you started with -- the same
$d/dt\\int = \\mathrm{id}$ as continuous calculus. Take $x=n^2$: the difference gives the odd
numbers $1,3,5,\\dots$, and accumulating those returns $n^2$. In the $z$-domain it is just
$(1-z^{-1})\\cdot\\frac{1}{1-z^{-1}}=1$."""),
co("""n = np.arange(12); x = n.astype(float)**2
d = zt.apply_filter(*zt.DIFFERENCE, x)            # discrete derivative
recon = zt.apply_filter(*zt.ACCUMULATOR, d)       # integrate it back
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].stem(n, x, "C0", markerfmt="C0o", basefmt=" ", label="x = n^2")
ax[0].stem(n, d, "C3", markerfmt="C3s", basefmt=" ", label="difference (odd numbers)")
ax[0].set(xlabel="n", title="x and its discrete derivative"); ax[0].legend()
ax[1].stem(n, x, "C0", markerfmt="C0o", basefmt=" ", label="x = n^2")
ax[1].stem(n, recon, "C2", markerfmt="C2x", basefmt=" ", label="accumulate(difference) = x")
ax[1].set(xlabel="n", title="discrete FTC: integrate the derivative -> x"); ax[1].legend()
plt.tight_layout(); plt.show()
print("accumulate(difference(x)) == x :", np.allclose(recon, x))"""),

md("""## 3. Poles, zeros, and stability -- the unit circle

A filter's poles (roots of $A(z)$) decide stability: **inside the unit circle = stable**,
on it = marginal (the accumulator's pole sits at $z=1$), outside = unstable. This is the
discrete twin of the $s$-plane left-half-plane test from `dgs.pid` / `dgs.spice`."""),
co("""def plot_pz(ax, b, a, title):
    p, zz = zt.poles_zeros(b, a)
    th = np.linspace(0, 2*np.pi, 200); ax.plot(np.cos(th), np.sin(th), "k", lw=1)
    ax.scatter(p.real, p.imag, marker="x", s=90, color="C3", label="poles")
    if len(zz): ax.scatter(zz.real, zz.imag, marker="o", s=70, facecolors="none", edgecolors="C0", label="zeros")
    ax.set(title=title+("  (STABLE)" if zt.is_stable(a) else "  (UNSTABLE)")); ax.axis("equal"); ax.grid(alpha=0.3); ax.legend()
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
plot_pz(ax[0], [1, 0], [1, -0.6, 0.25], "2-pole resonator")        # poles inside
plot_pz(ax[1], [1, 0], [1, -2.2, 1.2], "poles pushed out")         # roots 1.2, 1.0 -> unstable
plt.tight_layout(); plt.show()
print("stable [1,-0.6,0.25]:", zt.is_stable([1,-0.6,0.25]), " | [1,-2.2,1.2]:", zt.is_stable([1,-2.2,1.2]))"""),

md("""## What ties together

1. The **Z-transform is discrete calculus:** $1-z^{-1}$ is the derivative,
   $1/(1-z^{-1})$ the integral -- the same operators as continuous calculus, on samples.
2. The **Fundamental Theorem survives:** difference $\\times$ accumulator $=1$, so
   integrating a difference returns the signal.
3. **Stability lives on the unit circle:** poles inside $=$ stable, the discrete analog of
   the left-half $s$-plane.

This is the bridge from the finite differences in `dgs.numerical_methods` and the Fourier
work to real **digital filters** -- the DSP every sampled receiver runs. (Next step: design
filter coefficients by gradient **optimization in torch** to hit a target response.)
Civilian education."""),
]

nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = pathlib.Path("notebooks/z_transform_discrete_calculus.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(nb.cells), "cells")
