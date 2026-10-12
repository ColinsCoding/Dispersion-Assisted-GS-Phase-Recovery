"""Build two notebooks:
  notebooks/zero_flip_phase_retrieval.ipynb  -- why one intensity can't give the phase
  notebooks/opamp_inverting_p5_1.ipynb       -- Nilsson & Riedel Probs. 5.1-5.2 in SymPy

Build:   py -3.13 scripts/build_zero_flip_opamp_nbs.py
Execute: py -3.13 -m jupyter nbconvert --to notebook --execute --inplace <notebook>
"""
import pathlib
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks"
SETUP = """import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Math
sp.init_printing()"""

# ════════════════════════ zero-flip notebook ════════════════════════
z = []
z.append(md(r"""# Zero Flipping: Why One Intensity Can't Give You the Phase

This notebook explains, with complex analysis, why this repo's phase retrieval needs **two**
dispersed measurements $I_1, I_2$, and why they must differ enough.

| § | idea | result |
|---|---|---|
| 1 | a spectrum is a polynomial; its **zeros** fix it | roots in the complex plane |
| 2 | flipping a zero through the unit circle keeps $\lvert X\rvert$ | SymPy identity; 32 signals, one magnitude |
| 3 | time-stretch at large $D$ only sees $\lvert X\rvert$ | flipped signals become indistinguishable |
| 4 | your `retrieve_phase` | fails when $I_1 \approx I_2$, works with diversity |

**References (IEEE)**
1. M. H. Hayes, "The reconstruction of a multidimensional sequence from the phase or magnitude of its Fourier transform," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 30, no. 2, pp. 140–154, 1982.
2. K. Goda and B. Jalali, "Dispersive Fourier transformation for fast continuous single-shot measurements," *Nat. Photon.*, vol. 7, no. 2, pp. 102–112, 2013.

Module: [`dgs/zero_flip_phase_retrieval.py`](../dgs/zero_flip_phase_retrieval.py)"""))
z.append(co(SETUP + """
from dgs.zero_flip_phase_retrieval import (
    zeros_of, flip_zero, all_flips, fourier_magnitude, distinguishability, symbolic_flip_identity,
    max_dispersion_without_wrap,
)
from dgs.gs_core import make_measurements, retrieve_phase
rng = np.random.default_rng(1)
x = rng.normal(size=6) + 1j*rng.normal(size=6)        # a short complex signal, 6 samples"""))

z.append(md(r"""# 1. A spectrum is a polynomial

For a length-$L$ signal, $X(\omega) = \sum_n x[n]\,w^n$ with $w = e^{-j\omega}$. That's a degree-$(L-1)$
polynomial in $w$, evaluated on the **unit circle**. By the fundamental theorem of algebra it's
fixed, up to a constant, by its $L-1$ zeros:"""))
z.append(co("""zk = zeros_of(x)
th = np.linspace(0, 2*np.pi, 400)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(np.cos(th), np.sin(th), "k", lw=1, label="unit circle (where |X| is measured)")
ax.plot(zk.real, zk.imag, "o", ms=10, color="C0", label="zeros of X(w)")
ax.plot((1/np.conj(zk)).real, (1/np.conj(zk)).imag, "x", ms=10, color="C3", label="their reflections 1/conj(w_k)")
for a in zk:
    b = 1/np.conj(a); ax.plot([a.real, b.real], [a.imag, b.imag], "0.7", lw=0.8)
ax.set(aspect="equal", xlabel="Re w", ylabel="Im w", title="Zeros and their unit-circle reflections"); ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.show()
print("|zeros| =", np.round(np.abs(zk), 3))"""))

z.append(md(r"""# 2. Flipping a zero keeps the magnitude

On the unit circle, $|w - a| = |1 - \bar a\,w|$. So swapping the factor $(w - a)$ for $(1 - \bar a w)$,
which moves the zero from $a$ to $1/\bar a$, can't change $|X|$. SymPy proves the identity in general:"""))
z.append(co("""expr, (wr, wi, ar, ai) = symbolic_flip_identity()
display(Math(r"|w-a|^2 - |1-\\bar a w|^2 = " + sp.latex(expr) + r"\\;\\xrightarrow{\\;|w|=1\\;}\\; 0"))
F = all_flips(x)
M0 = fourier_magnitude(x)
print(f"{len(F)} signals from flipping subsets of {len(zk)} zeros")
print(f"largest |X| difference among them: {max(np.abs(fourier_magnitude(f) - M0).max() for f in F)/M0.max():.1e} (relative)")
print(f"smallest difference between two of the SIGNALS: {min(np.linalg.norm(F[i]-F[j]) for i in range(len(F)) for j in range(i+1, len(F))):.3f}")"""))
z.append(co("""fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
w = np.linspace(-np.pi, np.pi, 256, endpoint=False)
for f, col in zip(F[:4], ("C0", "C1", "C2", "C3")):
    Xf = np.fft.fftshift(np.fft.fft(f, 256))
    axes[0].plot(w, np.abs(Xf), col, lw=2 if col == "C0" else 1)
    axes[1].plot(w, np.unwrap(np.angle(Xf)), col)
    axes[2].stem(np.arange(6) + 0.1*"C0C1C2C3".index(col)/2, np.abs(f), linefmt=col, markerfmt=col+"o", basefmt=" ")
axes[0].set(title="|X(ω)|: all four identical", xlabel="ω"); axes[1].set(title="phase of X(ω): all different", xlabel="ω")
axes[2].set(title="|x[n]|: different signals", xlabel="n")
for a in axes: a.grid(alpha=0.3)
plt.tight_layout(); plt.show()"""))
z.append(md(r"""With 5 zeros there are $2^5 = 32$ signals sharing **one** Fourier magnitude. Flipping *all* the zeros
gives the time-reversed conjugate, which is the "trivial" ambiguity every phase-retrieval method lives
with. The other 30 are genuinely different signals. A spectrum analyzer can't tell any of them apart.

# 3. Time-stretch only sees $|X|$ in the far field

`disperse(E, D)` applies $H(\nu) = e^{j\pi D\nu^2}$. For large $D$ the output intensity is the
**spectrum mapped onto time**: that's the dispersive Fourier transform behind photonic time-stretch [2].
So a far-field measurement inherits the zero-flip ambiguity. At moderate $D$, the quadratic phase mixes
phase into the intensity, and the flipped signals produce *different* traces:"""))
z.append(co("""N = 65536
print(f"window N = {N}: |D| must stay below ~{max_dispersion_without_wrap(N):.0f} to avoid FFT wrap-around")
Ds = [0, 300, 1000, 3000, 1e4, 3e4, 1e5]
dist = [distinguishability(F, D, N=N) for D in Ds]
for D, d in zip(Ds, dist):
    print(f"D = {D:>8.0f}:  closest pair of flipped signals differs by {100*d:5.2f} %")
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.semilogx(np.maximum(Ds, 100), np.array(dist)*100, "o-")
ax.set(xlabel="dispersion |D| (D=0 plotted at 100)", ylabel="min pairwise difference (%)",
       title="Large D → far field → only |X| survives → flips indistinguishable"); ax.grid(alpha=0.3, which="both"); plt.show()"""))
z.append(md(r"""A numerical lesson came out of this section: with too small a window, the stretched pulse **wraps around**
the periodic FFT grid and the curve turns back up, which is an artifact, not physics. The guard
$|D|\,\nu_\max < N/2$ is now part of the module.

# 4. Your `retrieve_phase`: diversity is what breaks the ambiguity

Two measurements at *different* $D$ constrain the zeros in two different ways, and a zero flip that
fools one doesn't fool the other. If $D_1 \approx D_2$, the second measurement adds almost nothing:"""))
z.append(co("""def phase_error(phi, true):
    d = np.angle(np.exp(1j*(phi - true))); d = np.angle(np.exp(1j*(d - np.angle(np.mean(np.exp(1j*d))))))
    return float(np.sqrt(np.mean(d**2)))
rows = []
for D2 in (-5050, -5250, -5750, -7000):
    m = make_measurements("QPSK", D1=-5000, D2=D2, snr_db=40, rng_seed=0)
    phi, _ = retrieve_phase(m["I1"], m["I2"], -5000, D2, n_iter=100, unit_amplitude=m["unit_amplitude"])
    rows.append({"D2 (D1 = -5000)": D2, "corr(I1, I2)": round(float(np.corrcoef(m["I1"], m["I2"])[0, 1]), 4),
                 "RMS phase error (rad)": round(phase_error(phi, m["phi_true"]), 3)})
pd.DataFrame(rows)"""))
z.append(md(r"""When the two traces are nearly the same (correlation 0.99), Gerchberg–Saxton can't decide between
ambiguous solutions and the phase error stays large. Once they differ (correlation ≲ 0.4), it locks on.
That's the complex-analysis reason behind the repo's "$|D|$ large, $D_1 \ne D_2$" rules.

## Exercises
1. Put one zero *on* the unit circle. What happens to its flip, and to $|X|$ at that frequency?
2. Count the distinct flips for a *real* signal (zeros come in conjugate pairs). Why are there fewer?
3. Find the smallest $|D_1 - D_2|$ at which `retrieve_phase` succeeds, and relate it to the correlation."""))

nbz = nbf.v4.new_notebook(); nbz["cells"] = z
nbz["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nbz, OUT / "zero_flip_phase_retrieval.ipynb")

# ════════════════════════ op-amp notebook ════════════════════════
o = []
o.append(md(r"""# Op-Amp Problems 5.1–5.2: The Inverting Amplifier in SymPy

**Source:** J. W. Nilsson and S. A. Riedel, *Electric Circuits*. Hoboken, NJ, USA: Pearson, Probs. 5.1–5.2,
Fig. P5.1. *(Add your edition and year for a full IEEE citation.)*

The circuit: $v_s = 2$ V through $R_1 = 4$ kΩ into the inverting input; $R_f = 12$ kΩ feedback;
non-inverting input grounded; ±15 V supplies; 6 kΩ load. Module:
[`dgs/opamp_inverting.py`](../dgs/opamp_inverting.py)"""))
o.append(co(SETUP + """
from dgs.opamp_inverting import (
    ideal_solution, finite_gain_solution, output_voltage, is_saturated, linear_input_range,
    output_current, PROBLEM_5_2_SOURCES,
)"""))
o.append(md(r"""## 5.1 (a)–(c): terminals and the two ideal rules

| terminal | symbol |
|---|---|
| inverting input | $v_n$ (−) |
| non-inverting input | $v_p$ (+), grounded here |
| output | $v_o$ |
| positive supply | $+V_{CC} = +15$ V |
| negative supply | $-V_{CC} = -15$ V |

- **(b)** $i_n = 0$, because an ideal op amp has **infinite input resistance**.
- **(c)** $v_p - v_n = 0$ (the **virtual short**). This holds only *with negative feedback* and *while
  the output is not saturated*.

## The gap the book skips: where the virtual short comes from

Model the op amp as $v_o = A(v_p - v_n)$ with finite gain $A$ and solve KCL at $v_n$. Then let $A\to\infty$:"""))
o.append(co("""vo_A, vo_inf, vn_A, (vs, R1, Rf, A) = finite_gain_solution()
display(Math(r"v_o(A) = " + sp.latex(vo_A)))
display(Math(r"v_n(A) = " + sp.latex(vn_A) + r"\\;\\xrightarrow{A\\to\\infty}\\; " + sp.latex(sp.limit(vn_A, A, sp.oo))))
display(Math(r"\\lim_{A\\to\\infty} v_o = " + sp.latex(vo_inf)))
for Av in (1e2, 1e4, 1e6):
    print(f"A = {Av:8.0e}: v_o = {float(vo_A.subs({vs: 2, R1: 4e3, Rf: 12e3, A: Av})):.6f} V,  v_n = {float(vn_A.subs({vs: 2, R1: 4e3, Rf: 12e3, A: Av}))*1e3:.4f} mV")"""))
o.append(md(r"""As the gain grows, $v_n$ is driven to 0 V (the virtual short) and $v_o \to -(R_f/R_1)v_s$. A real
op amp with $A \approx 10^5$ is already within 0.004% of the ideal answer.

## 5.1 (d)

KCL at $v_n$ with $i_n = 0$ and $v_n = v_p = 0$: $\;\dfrac{2 - 0}{4\text{ k}} + \dfrac{v_o - 0}{12\text{ k}} = 0$."""))
o.append(co("""vo_ideal, _ = ideal_solution()
display(Math(r"v_o = " + sp.latex(vo_ideal)))
print(f"v_o = {float(output_voltage(2.0)):.2f} V   (gain −12k/4k = −3)")
print(f"op-amp output current = load + feedback = {float(output_current(2.0))*1e3:.2f} mA")"""))
o.append(md(r"""## 5.2: other sources, and saturation

The output can't go beyond the supplies, so $v_o = \text{clip}(-3v_s, -15, +15)$:"""))
o.append(co("""vo = output_voltage(PROBLEM_5_2_SOURCES)
display(pd.DataFrame({"v_s (V)": PROBLEM_5_2_SOURCES, "−3·v_s (V)": [-3*v for v in PROBLEM_5_2_SOURCES],
                      "v_o (V)": vo, "saturated?": is_saturated(PROBLEM_5_2_SOURCES)}))
lo, hi = linear_input_range()
print(f"(b) linear (no saturation) for {lo:.1f} V ≤ v_s ≤ {hi:.1f} V")
vv = np.linspace(-7, 7, 400)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(vv, output_voltage(vv), "k"); ax.plot(PROBLEM_5_2_SOURCES, vo, "o", color="C3")
ax.axvspan(lo, hi, color="C2", alpha=0.12, label="linear region")
ax.set(xlabel="v_s (V)", ylabel="v_o (V)", title="Transfer curve: slope −3, clipped at ±15 V"); ax.legend(); ax.grid(alpha=0.3); plt.show()"""))
o.append(md(r"""The two saturated points ($v_s = -6$ V and $5.4$ V) are where the linear formula would ask for
+18 V and −16.2 V, more than the supplies can provide. Inside $|v_s| \le 5$ V, every answer is just $-3v_s$."""))
nbo = nbf.v4.new_notebook(); nbo["cells"] = o
nbo["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nbo, OUT / "opamp_inverting_p5_1.ipynb")
print("wrote both notebooks")
