"""Build notebooks/selection_rules.ipynb -- atomic selection rules by sympy/torch/pandas."""
import pathlib
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
nb = nbf.v4.new_notebook()

nb.cells = [
md("""# Selection rules -- why an atomic spectrum has bright lines and missing ones
### $\\Delta l=\\pm1,\\ \\Delta m=0,\\pm1$, derived with SymPy and Torch, then read off a Grotrian diagram

A hydrogen atom emits a photon when it drops between levels, conserving energy as
$\\Delta E=h f=h c/\\lambda$. But the photon carries one unit of angular momentum and odd parity, so an
electric-dipole transition only happens when
$$\\Delta l=\\pm1,\\qquad \\Delta m=0,\\pm1,$$
with $\\Delta n$ free. Everything else is **forbidden**, which is exactly why a spectrum is a specific
pattern of lines. Driving `dgs/selection_rules.py` and `dgs/hydrogen_atom.py`, we:

1. **derive $\\Delta l=\\pm1$ in SymPy** from the dipole integral $\\int P_{l'}(x)\\,x\\,P_l(x)\\,dx$;
2. **recover both rules numerically in Torch** by integrating the dipole matrix element over the sphere;
3. draw the **Grotrian diagram** with only the allowed transitions;
4. table the **metastable timing** (why 2s lives ~$10^8\\times$ longer than 2p);
5. split a line into its **Zeeman $\\sigma/\\pi$** components (the $\\Delta m$ rule made visible).

Runs on the Python 3.12 + Torch kernel."""),

co("""import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""       # CPU-only; avoid loading CUDA DLLs
import sympy as sp
import numpy as np, torch, pandas as pd
import matplotlib.pyplot as plt
from scipy.special import lpmv, factorial
import math, sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
from dgs import selection_rules as sr
from dgs import hydrogen_atom as H
sp.init_printing()
print("torch", torch.__version__, "| sympy", sp.__version__)"""),

md("""## 1. Where $\\Delta l=\\pm1$ comes from (SymPy)

The intensity of a $z$-polarized transition is set by the dipole matrix element, whose angular part
(for $\\Delta m=0$) is $\\int_{-1}^{1}P_{l'}(x)\\,x\\,P_l(x)\\,dx$ with $x=\\cos\\theta$. Because
$x\\,P_l=\\frac{(l+1)P_{l+1}+l\\,P_{l-1}}{2l+1}$ and Legendre polynomials are orthogonal, this integral
is **nonzero only when $l'=l\\pm1$**. SymPy evaluates the whole table and the $\\pm1$ band appears."""),

co("""x = sp.symbols('x')
Lmax = 4
M = sp.zeros(Lmax, Lmax)
for lp in range(Lmax):
    for l in range(Lmax):
        M[lp, l] = sp.integrate(sp.legendre(lp, x) * x * sp.legendre(l, x), (x, -1, 1))
sp.pprint(M)
# every nonzero entry has |l'-l| == 1
for lp in range(Lmax):
    for l in range(Lmax):
        if M[lp, l] != 0:
            assert abs(lp - l) == 1
print("\\nnonzero  <=>  |l' - l| = 1   ->  Delta_l = +-1 (verified by SymPy)")

Mn = np.array(M.tolist(), float)
plt.figure(figsize=(4.6, 4.2))
plt.imshow(Mn != 0, cmap="Greens", origin="lower")
plt.xticks(range(Lmax), ["s","p","d","f"]); plt.yticks(range(Lmax), ["s","p","d","f"])
plt.xlabel("l (initial)"); plt.ylabel("l' (final)")
plt.title(r"$\\int P_{l'} x P_l\\,dx \\neq 0$ only on the $\\Delta l=\\pm1$ band")
for lp in range(Lmax):
    for l in range(Lmax):
        if Mn[lp,l] != 0: plt.text(l, lp, "OK", ha="center", va="center", fontsize=8)
plt.tight_layout(); plt.show()"""),

md("""## 2. Both rules from the full dipole integral (Torch)

Now the complete angular matrix element over the sphere,
$\\langle l'm'|\\hat r|lm\\rangle=\\int \\bar Y_{l'}^{m'}\\,\\hat r\\,Y_l^m\\,d\\Omega$, with the dipole
direction $\\hat r=(\\sin\\theta\\cos\\phi,\\sin\\theta\\sin\\phi,\\cos\\theta)$. We build the spherical
harmonics on a $(\\theta,\\phi)$ grid, integrate the three components as Torch tensors, and take the
total coupling $|d|^2=|\\langle x\\rangle|^2+|\\langle y\\rangle|^2+|\\langle z\\rangle|^2$. It is nonzero
**exactly** for $\\Delta l=\\pm1$ and $\\Delta m=0,\\pm1$ -- matching `selection_rules.is_dipole_allowed`."""),

co("""def Ylm(l, m, theta, phi):
    mm = abs(m)
    N = math.sqrt((2*l+1)/(4*math.pi) * factorial(l-mm)/factorial(l+mm))
    Y = N * lpmv(mm, l, np.cos(theta)) * np.exp(1j*mm*phi)
    return ((-1)**mm) * np.conj(Y) if m < 0 else Y

nth, nph = 240, 240
th = np.linspace(1e-4, np.pi-1e-4, nth)
ph = np.linspace(0, 2*np.pi, nph, endpoint=False)
TH, PH = np.meshgrid(th, ph, indexing="ij")
w = torch.tensor(np.sin(TH) * (th[1]-th[0]) * (ph[1]-ph[0]))          # dOmega
rx = torch.tensor(np.sin(TH)*np.cos(PH)); ry = torch.tensor(np.sin(TH)*np.sin(PH)); rz = torch.tensor(np.cos(TH))

states = [(l, m) for l in range(3) for m in range(-l, l+1)]           # s, p, d
def coupling(si, sf):
    Yi = torch.tensor(Ylm(si[0], si[1], TH, PH))
    Yf = torch.tensor(Ylm(sf[0], sf[1], TH, PH))
    d2 = 0.0
    for comp in (rx, ry, rz):
        integ = torch.sum(torch.conj(Yf) * comp * Yi * w)
        d2 += float(torch.abs(integ)**2)
    return d2

n = len(states)
D = np.array([[coupling(states[i], states[j]) for j in range(n)] for i in range(n)])
# numeric coupling nonzero  <=>  the analytic selection rule
for i in range(n):
    for j in range(n):
        li, mi = states[i]; lf, mf = states[j]
        assert (D[i,j] > 1e-8) == sr.is_dipole_allowed(li, mi, lf, mf)
print("Torch dipole integral nonzero  <=>  Delta_l=+-1 and Delta_m=0,+-1  (matches is_dipole_allowed)")

labels = [f"{'spd'[l]}{m:+d}" for (l,m) in states]
plt.figure(figsize=(6, 5))
plt.imshow(D > 1e-8, cmap="Blues", origin="lower")
plt.xticks(range(n), labels, rotation=90, fontsize=7); plt.yticks(range(n), labels, fontsize=7)
plt.title("allowed dipole couplings (Torch): the $\\\\Delta l=\\\\pm1,\\\\Delta m=0,\\\\pm1$ pattern")
plt.tight_layout(); plt.show()"""),

md("""## 3. The Grotrian diagram: only allowed lines drawn

Arrange the hydrogen levels by energy (rows $n$) and orbital type (columns $s,p,d,f$). Every allowed
transition slants by exactly one column ($\\Delta l=\\pm1$); a vertical drop ($\\Delta l=0$, like
$2s\\to1s$) is missing. The visible spectrum is these diagonals."""),

co("""fig, ax = plt.subplots(figsize=(8, 5.5))
pos = {}
for n in range(1, 5):
    for l in range(n):
        E = H.energy_level(n); pos[(n,l)] = (l, E)
        ax.hlines(E, l-0.35, l+0.35, color="k", lw=2)
        ax.text(l-0.45, E, f"{n}{'spdf'[l]}", ha="right", va="center", fontsize=8)
series_color = {1: "#4C78A8", 2: "#E45756", 3: "#54A24B"}
for (ni, li) in list(pos):
    for (nf, lf) in list(pos):
        if H.energy_level(ni) > H.energy_level(nf) and sr.is_dipole_allowed(li, 0, lf, 0):
            x0, y0 = pos[(ni,li)]; x1, y1 = pos[(nf,lf)]
            ax.plot([x0, x1], [y0, y1], color=series_color.get(nf, "gray"), lw=0.8, alpha=0.7)
ax.set_xticks(range(4)); ax.set_xticklabels(["s (l=0)","p (l=1)","d (l=2)","f (l=3)"])
ax.set_ylabel("energy (eV)"); ax.set_ylim(-14.5, 0.5)
ax.set_title("Grotrian diagram: allowed transitions slant by one column (blue=Lyman, red=Balmer)")
plt.tight_layout(); plt.show()"""),

md("""## 4. Timing: the metastable 2s state (Pandas)

Selection rules set lifetimes. A state with an allowed decay empties in nanoseconds; one with **no**
allowed decay is metastable. Hydrogen's **2s** can only reach 1s, but that is $\\Delta l=0$
(forbidden), so it is stranded -- ~0.12 s versus 2p's ~1.6 ns."""),

co("""rows = []
for (n, l, m) in [(1,0,0),(2,0,0),(2,1,0),(3,0,0),(3,1,0),(3,2,0)]:
    decays = sr.allowed_decays(n, l, m)
    rows.append({"state": f"{n}{'spdf'[l]}", "n_decays": len(decays),
                 "example decay": (f"{decays[0][0]}{'spdf'[decays[0][1]]}" if decays else "-- none --"),
                 "class": sr.lifetime_class(n, l, m)})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
assert sr.is_metastable(2,0,0) and not sr.is_metastable(2,1,0) and not sr.is_metastable(3,0,0)
print("\\nonly 2s is metastable: its sole lower neighbour 1s needs Delta_l=0 (forbidden).")"""),

md("""## 5. The $\\Delta m$ rule made visible: the Zeeman triplet

Put the atom in a magnetic field $B$ and each level splits by $\\Delta E=\\mu_B B\\,m$. A transition then
fans into components at $\\Delta m=0$ (the **$\\pi$** line, unshifted, linearly polarized) and
$\\Delta m=\\pm1$ (the two **$\\sigma$** lines, shifted by $\\pm\\mu_B B$, circularly polarized). The
$\\Delta m=\\pm2$ line is absent -- forbidden. This "normal triplet" is the selection rule you can see."""),

co("""mu_B = 5.7883818e-5           # eV/T
B = 5.0                       # tesla
shift = mu_B * B              # eV per unit m
plt.figure(figsize=(8, 3.2))
for dm, col, tag in [(-1, "#4C78A8", r"$\\sigma^-$ ($\\Delta m$=-1)"),
                     (0, "#333333", r"$\\pi$ ($\\Delta m$=0)"),
                     (+1, "#E45756", r"$\\sigma^+$ ($\\Delta m$=+1)")]:
    assert sr.polarization(0, dm) is not None      # all three are allowed
    plt.vlines(dm*shift*1e6, 0, 1, color=col, lw=2, label=tag)
plt.vlines(2*shift*1e6, 0, 0.5, color="gray", ls=":", lw=1)
plt.text(2*shift*1e6, 0.55, r"$\\Delta m$=+2" + "\\nforbidden", color="gray", fontsize=8, ha="center")
plt.xlabel(r"shift from line centre ($\\mu$eV) at B=5 T"); plt.yticks([]); plt.legend(fontsize=8)
plt.title("normal Zeeman triplet: the Delta_m = 0, +-1 selection rule, in polarization")
plt.tight_layout(); plt.show()
print(f"Zeeman shift mu_B*B = {shift*1e6:.2f} ueV at 5 T; Delta_m=+-2 line is forbidden (absent).")"""),

md("""## What we did

* **Derived $\\Delta l=\\pm1$** from the Legendre dipole integral in SymPy (nonzero only on the $\\pm1$
  band).
* **Recovered $\\Delta l=\\pm1$ and $\\Delta m=0,\\pm1$ numerically in Torch** by integrating the full
  angular dipole matrix element -- matching `is_dipole_allowed` exactly.
* Drew the **Grotrian diagram** where allowed lines slant one column and forbidden vertical drops are
  absent.
* Tabled the **metastable 2s** (no allowed decay) against fast $2p,3s$ -- selection rules as timing.
* Split a line into its **Zeeman $\\sigma/\\pi$** triplet, the $\\Delta m$ rule you can photograph.

`dgs/hydrogen_atom.py` says which states exist; `dgs/selection_rules.py` says which lines between them
appear -- together they turn the atom into its spectrum."""),
]

nb.metadata["kernelspec"] = {"display_name": "Python 3.12 (torch)", "language": "python", "name": "py312"}
out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "selection_rules.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
