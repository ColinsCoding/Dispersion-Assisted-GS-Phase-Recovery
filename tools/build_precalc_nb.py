"""Build notebooks/precalc_to_calc2_physics.ipynb (companion to dgs/precalc_steps.py).

Run:  py -3.12 tools/build_precalc_nb.py
Then: py -3.12 -m jupyter nbconvert --to notebook --execute --inplace notebooks/precalc_to_calc2_physics.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "precalc_to_calc2_physics.ipynb"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


def example(title, fn, where, extra=""):
    md(f"## {title}\n\n**Where it shows up:** {where}")
    code(f"S = P.{fn}()\nS.show()\ndisplay(S.table())" + (f"\n{extra}" if extra else ""))


md(r"""
# The precalculus steps under Calc 2, Griffiths, and modern physics

In most "hard" integrals in Griffiths or a modern-physics course, the calculus is one line.
The rest is precalculus: completing the square, a trig identity, exponent and log rules,
partial fractions, a geometric series. That's also where most exam points get lost.

Every derivation below runs through `dgs/precalc_steps.py`:
* each **"=" step** is checked by SymPy to equal the previous one, so a wrong step raises
  `StepError` and is never shown as if it were right;
* each **operation** (integrate, expand in a series, substitute, solve) is checked by redoing
  it forward, or by undoing it backward (sum the series back, differentiate the answer);
* at the end, **Torch autograd** checks the antiderivatives numerically, as a second,
  independent method.

| Precalc skill | Calc 2 technique it unlocks | Where it shows up |
|---|---|---|
| Completing the square | Gaussian integrals | free wave packet (Griffiths QM 2.4), statistics |
| sin² = (1 − cos 2A)/2, product-to-sum | integrals of trig powers, orthogonality | ⟨x⟩ in a box, Fourier series |
| Integration by parts (from the product rule) | ∫ polynomial × trig/exp | expectation values, Laplace transforms |
| Geometric series r/(1 − r) | series inside integrals | Planck spectrum → Stefan–Boltzmann |
| Exponent and log rules | solving exponential equations | half-life, RC circuits |
| Binomial series (1 + q)ᵖ | Taylor approximations | relativistic K → ½mv² |
| 1 + tan² = sec², right triangles | trig substitution | field of a line charge (Griffiths E&M Ex. 2.2) |
| Partial fractions | rational integrals, inverse Laplace | radioactive decay chains |
""")

code(r"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import sympy as sp
sp.init_printing(use_unicode=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from dgs import precalc_steps as P
from dgs.precalc_steps import Steps, StepError
pd.set_option("display.max_colwidth", 70); pd.set_option("display.width", 200)
""")

md(r"""
## 0. The precalc toolkit, each identity checked

Each row is an identity you're expected to know cold. The `check` column is SymPy confirming
left − right = 0.
""")

code(r"""
A, B, q, r, k = sp.symbols("A B q r k", real=True)
pp, mm = sp.symbols("p m", positive=True)
ident = [
    ("exponent: e^A e^B = e^(A+B)", sp.exp(A) * sp.exp(B), sp.exp(A + B)),
    ("log: ln(p m) = ln p + ln m", sp.log(pp * mm), sp.log(pp) + sp.log(mm)),
    ("log: ln(p^k) = k ln p", sp.log(pp**k), k * sp.log(pp)),
    ("Pythagorean: sin^2 + cos^2 = 1", sp.sin(A)**2 + sp.cos(A)**2, 1),
    ("1 + tan^2 = sec^2", 1 + sp.tan(A)**2, 1 / sp.cos(A)**2),
    ("double angle: sin 2A = 2 sin A cos A", sp.sin(2 * A), 2 * sp.sin(A) * sp.cos(A)),
    ("power reduction: sin^2 A = (1 - cos 2A)/2", sp.sin(A)**2, (1 - sp.cos(2 * A)) / 2),
    ("power reduction: cos^2 A = (1 + cos 2A)/2", sp.cos(A)**2, (1 + sp.cos(2 * A)) / 2),
    ("product-to-sum: sin A sin B", sp.sin(A) * sp.sin(B), (sp.cos(A - B) - sp.cos(A + B)) / 2),
    ("sum of angles: cos(A + B)", sp.cos(A + B), sp.cos(A) * sp.cos(B) - sp.sin(A) * sp.sin(B)),
    ("complete the square", q**2 + B * q, (q + B / 2)**2 - B**2 / 4),
    ("finite geometric sum (r != 1)", sum(r**j for j in range(6)), (1 - r**6) / (1 - r)),
]
rows = [(name, sp.sstr(lhs), sp.sstr(rhs), P._equal(lhs, rhs)) for name, lhs, rhs in ident]
display(pd.DataFrame(rows, columns=["identity", "left", "right", "check"]))
assert all(r[-1] for r in rows)
""")

example("1. Completing the square → Gaussian integral", "example_complete_square",
        "the free-particle wave packet (Griffiths QM §2.4), the normal distribution, Fourier transforms of Gaussians.",
        extra=r"""xs = np.linspace(-3, 5, 400)
plt.figure(figsize=(7, 3)); plt.plot(xs, np.exp(-xs**2 + 2 * xs), label="exp(-x^2 + 2x)")
plt.plot(xs, np.e * np.exp(-(xs - 1)**2), "--", label="e * exp(-(x - 1)^2)  (same curve)")
plt.legend(); plt.title("completing the square shifts the Gaussian to x = b/2a"); plt.tight_layout(); plt.show()""")

example("2. Power reduction + integration by parts → ⟨x⟩ in a box", "example_box_mean_x",
        "Knight Ch. 40 / PHYS 106 expectation values. The same moves give ⟨x²⟩ and Δx.")

example("3. Integration by parts, checked by differentiating", "example_by_parts_rule",
        "it comes from the product rule: (uv)′ = u′v + uv′, then integrate both sides. Checking by differentiating catches sign errors.")

example("4. Geometric series inside an integral → Stefan–Boltzmann", "example_stefan_boltzmann",
        "blackbody radiation (modern physics): total power ∝ T⁴ with the constant π⁴/15.",
        extra=r"""zs = np.linspace(0.01, 12, 400)
plt.figure(figsize=(7, 3)); plt.plot(zs, zs**3 / np.expm1(zs)); plt.fill_between(zs, zs**3 / np.expm1(zs), alpha=0.3)
plt.title(f"area under z^3/(e^z - 1) = pi^4/15 = {math.pi**4 / 15:.4f};  numeric {np.trapezoid(zs**3 / np.expm1(zs), zs):.4f}")
plt.xlabel("z = hf/kT"); plt.tight_layout(); plt.show()""")

example("5. Exponent and log rules → half-life", "example_half_life",
        "radioactive decay and RC/RL circuits (EEE 117: the time to fall to half is τ ln 2). Equation steps are checked by comparing solution sets.")

example("6. Binomial series → relativistic K becomes ½mv²", "example_relativistic_ke",
        "PHYS 106 relativity: why Newton works at low speed, and how large the first correction is.",
        extra=r"""bs = np.linspace(0, 0.9, 200)
exact = 1 / np.sqrt(1 - bs**2) - 1
plt.figure(figsize=(7, 3)); plt.plot(bs, exact, label="exact (gamma - 1)")
plt.plot(bs, bs**2 / 2, "--", label="(1/2) beta^2  (Newton)"); plt.plot(bs, bs**2 / 2 + 3 * bs**4 / 8, ":", label="+ 3/8 beta^4")
plt.xlabel("beta = v/c"); plt.ylabel("K / mc^2"); plt.legend(); plt.title("the series is good at low beta, fails near c")
plt.tight_layout(); plt.show()
for b0 in (0.1, 0.5, 0.9):
    ex0 = 1 / math.sqrt(1 - b0**2) - 1
    print(f"beta = {b0}: Newton's (1/2) m v^2 is {(ex0 - b0**2 / 2) / ex0:.2%} low")""")

example("7. Trig substitution → field of a finite line charge", "example_trig_sub",
        "Griffiths E&M Ex. 2.2 (the field of a finite line charge). Note the step that needs cos θ > 0: the substitution only works on −π/2 < θ < π/2.")

example("8. Partial fractions → a decay chain A → B → C", "example_partial_fractions",
        "modern physics decay chains and EEE 117 inverse Laplace transforms: two exponentials whose difference rises, then falls.",
        extra=r"""ts = np.linspace(0, 6, 300); k1v, k2v = 1.0, 2.5
plt.figure(figsize=(7, 3)); plt.plot(ts, (np.exp(-k1v * ts) - np.exp(-k2v * ts)) / (k2v - k1v))
plt.xlabel("t"); plt.ylabel("amount of B (per unit of A)"); plt.title("Bateman curve: B builds up, then decays"); plt.tight_layout(); plt.show()""")

md(r"""
## 9. Mistake gallery: what the checker catches

These are real, common errors. Each one is fed to the engine, and each one is rejected.
""")

code(r"""
from dgs.precalc_steps import a, b, x, t, lam
mistakes = [
    ("freshman's dream: (x + b)^2 = x^2 + b^2", lambda: Steps("m").start((x + b)**2).eq("?", x**2 + b**2)),
    ("power reduction with the wrong sign", lambda: Steps("m").start(sp.sin(x)**2).eq("?", (1 + sp.cos(2 * x)) / 2)),
    ("completing the square: b^2/2a instead of b^2/4a",
     lambda: Steps("m").start(-a * x**2 + b * x).eq("?", -a * (x - b / (2 * a))**2 + b**2 / (2 * a))),
    ("ln(1/2) = +ln 2", lambda: Steps("m", unknown=t).start(sp.Eq(sp.exp(-lam * t), sp.Rational(1, 2))).eq("?", sp.Eq(-lam * t, sp.log(2)))),
    ("antiderivative of x cos(kx) with the wrong sign",
     lambda: Steps("m").start(x * sp.cos(2 * x)).op("?", x * sp.sin(2 * x) / 2 - sp.cos(2 * x) / 4, "", check=lambda e: sp.integrate(e, x))),
]
rows = []
for name, attempt in mistakes:
    try:
        attempt(); rows.append((name, "NOT caught"))
    except StepError:
        rows.append((name, "caught: StepError"))
display(pd.DataFrame(rows, columns=["mistake", "result"]))
assert all(r[1].startswith("caught") for r in rows)
""")

md(r"""
## 10. A second, independent check: Torch autograd

Autograd applies the chain rule automatically, the same machinery that trains neural
networks. Differentiating each antiderivative numerically must give back the integrand.
This uses a different method from SymPy, so the two can't share a bug.
""")

code(r"""
try:
    import torch
    zz = torch.linspace(-3, 3, 201, dtype=torch.float64, requires_grad=True)
    checks = {
        "by parts: x sin(kx)/k + cos(kx)/k^2 -> x cos(kx)": (lambda v: v * torch.sin(2.3 * v) / 2.3 + torch.cos(2.3 * v) / 2.3**2,
                                                           lambda v: v * torch.cos(2.3 * v)),
        "trig sub: z/(s^2 sqrt(z^2+s^2)) -> (z^2+s^2)^(-3/2)": (lambda v: v / (0.7**2 * torch.sqrt(v**2 + 0.7**2)),
                                                              lambda v: (v**2 + 0.7**2) ** -1.5),
        "Gaussian antiderivative check: d/dx erf": (lambda v: torch.erf(v), lambda v: 2 / math.sqrt(math.pi) * torch.exp(-v**2)),
        "chain rule: d/dt e^(-lambda t) = -lambda e^(-lambda t)": (lambda v: torch.exp(-0.8 * v), lambda v: -0.8 * torch.exp(-0.8 * v)),
    }
    rows = []
    for name, (F, f) in checks.items():
        (dF,) = torch.autograd.grad(F(zz).sum(), zz)
        rows.append((name, float((dF - f(zz.detach())).abs().max())))
    display(pd.DataFrame(rows, columns=["antiderivative", "max |autograd - integrand|"]))
    print("torch", torch.__version__, "| device used: cpu")
except ImportError:
    print("torch not installed: skipping the autograd cross-check (SymPy checks above still stand).")
""")

md(r"""
## Practice (my own problems: do the precalc step first)

1. Complete the square in −3x² + 12x, then find ∫e^(−3x²+12x) dx over all x.
2. Use power reduction to find ∫₀^π cos²(x) dx.
3. Solve 2^(t/5) = 50 for t with log rules.
4. Expand (1 − x)^(−1/2) to second order and use it to approximate 1/√0.98.
5. Split 1/(x² − 1) into partial fractions, and integrate it from 2 to 3.

Each answer below is built with the step engine, so the work is checked too.
""")

code(r"""
xx, tt = sp.symbols("x t", real=True)
S1 = Steps("1").start(-3 * xx**2 + 12 * xx).eq("factor", -3 * (xx**2 - 4 * xx)).eq("complete", -3 * (xx - 2)**2 + 12)
print("1.", S1.result, "-> integral =", sp.simplify(sp.integrate(sp.exp(S1.result), (xx, -sp.oo, sp.oo))))
S2 = Steps("2").start(sp.cos(xx)**2).eq("power reduction", (1 + sp.cos(2 * xx)) / 2)
print("2.", sp.integrate(S2.result, (xx, 0, sp.pi)))
S3 = Steps("3", unknown=tt).start(sp.Eq(2**(tt / 5), 50)).eq("ln both sides", sp.Eq(tt / 5 * sp.log(2), sp.log(50)))
t3 = 5 * sp.log(50) / sp.log(2)
assert P._equal(sp.solve(S3.result, tt)[0], t3)
print("3. t = 5 ln 50 / ln 2 =", sp.N(t3, 5))
ser = sp.series((1 - xx)**sp.Rational(-1, 2), xx, 0, 3).removeO()
print("4.", ser, "-> at x = 0.02:", sp.N(ser.subs(xx, 0.02), 8), " exact:", sp.N(1 / sp.sqrt(0.98), 8))
S5 = Steps("5").start(1 / (xx**2 - 1)).eq("partial fractions", (1 / (xx - 1) - 1 / (xx + 1)) / 2)
print("5.", S5.result, "-> integral from 2 to 3 =", sp.simplify(sp.integrate(S5.result, (xx, 2, 3))))
""")

md(r"""
## Checklist

- [ ] Before integrating, ask which precalc move turns this into a known form: square, identity, series, fractions, or substitution.
- [ ] Every "=" must really be equal. Equation steps must keep the same solutions.
- [ ] Substitutions have domains: trig sub needs −π/2 < θ < π/2, and the geometric series needs |r| < 1.
- [ ] Check an antiderivative by differentiating it (SymPy, Torch autograd, or by hand).
- [ ] If SymPy hangs, use structure (substitution, matrices, known series) instead of brute-force `simplify`.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("wrote", OUT, "with", len(cells), "cells")
