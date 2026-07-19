"""Build notebooks/calculus2_review.ipynb -- a focused Calc 2 review with SymPy."""
import pathlib
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
nb = nbf.v4.new_notebook()

nb.cells = [
md("""# Calculus 2 review -- integration, series, and applications
### the whole course on one page, worked with SymPy so you can change it and re-run

Calc 2 is three things: **harder integrals** (techniques), **infinite series**
(do they add up?), and **applications** (volume, arc length). This is a compact
review you can study from and tinker with -- every result is computed, not just
stated. Civilian education / college prep."""),

co("""import numpy as np, matplotlib.pyplot as plt
import sympy as sp
sp.init_printing()
x = sp.Symbol("x")
print("ready, sympy", sp.__version__)"""),

md("""## 1. Integration techniques

The four you must own:
- **u-substitution** (the chain rule backwards)
- **integration by parts** $\\int u\\,dv = uv - \\int v\\,du$ (the product rule backwards)
- **trig substitution** (for $\\sqrt{a^2-x^2}$ etc.)
- **partial fractions** (split a rational function into simple pieces)"""),
co("""print("by parts:  integral x e^x dx =", sp.integrate(x*sp.exp(x), x))
print("by parts:  integral ln(x) dx   =", sp.integrate(sp.log(x), x))
print("trig sub:  integral 1/sqrt(1-x^2) dx =", sp.integrate(1/sp.sqrt(1-x**2), x))
print("partial fractions: 1/((x-1)(x+2)) =", sp.apart(1/((x-1)*(x+2)), x))
print("   integral of that =", sp.integrate(1/((x-1)*(x+2)), x))"""),

md("""## 2. Improper integrals (does the area converge?)

Integrals to infinity converge only if the tail dies fast enough. The boundary is
$1/x$: $\\int_1^\\infty x^{-p}\\,dx$ **converges iff $p>1$**. Same threshold as the
p-series below -- the integral test makes them the same question."""),
co("""p = sp.Symbol("p", positive=True)
print("integral_1^inf x^-2 dx =", sp.integrate(x**-2, (x, 1, sp.oo)), " (converges, p=2>1)")
print("integral_1^inf 1/x  dx =", sp.integrate(1/x, (x, 1, sp.oo)), " (diverges, p=1)")
print("integral_0^1 1/sqrt(x) dx =", sp.integrate(1/sp.sqrt(x), (x, 0, 1)), " (converges at 0)")"""),

md("""## 3. Sequences and series -- the convergence tests

A series $\\sum a_n$ converges if its partial sums settle. Key facts:
- **geometric** $\\sum r^n$ converges to $1/(1-r)$ iff $|r|<1$.
- **p-series** $\\sum 1/n^p$ converges iff $p>1$ (so $\\sum 1/n$ **diverges**, $\\sum 1/n^2$ converges to $\\pi^2/6$).
- **ratio test**: if $\\lim|a_{n+1}/a_n|<1$, it converges.
- **alternating** series with terms decreasing to 0 converge."""),
co("""n = sp.Symbol("n", positive=True, integer=True)
print("geometric sum_{n=0}^inf (1/2)^n =", sp.summation(sp.Rational(1,2)**n, (n, 0, sp.oo)))
print("p=2  sum 1/n^2 =", sp.summation(1/n**2, (n, 1, sp.oo)), " (Basel problem)")
print("harmonic sum 1/n =", sp.summation(1/n, (n, 1, sp.oo)), " (diverges)")

# watch the partial sums: 1/n^2 settles, 1/n crawls up forever
N = np.arange(1, 2000)
plt.figure(figsize=(7,3.6))
plt.plot(N, np.cumsum(1.0/N**2), label="sum 1/n^2 -> pi^2/6 = %.4f" % (np.pi**2/6))
plt.plot(N, np.cumsum(1.0/N)/5, label="sum 1/n (scaled /5) keeps rising")
plt.xlabel("number of terms"); plt.ylabel("partial sum"); plt.legend()
plt.title("convergent vs divergent series"); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()"""),

md("""## 4. Power series and Taylor (the payoff)

A function equals its **Taylor/Maclaurin series** near a point, valid within a
**radius of convergence**. These are the approximations physics runs on."""),
co("""print("e^x   :", sp.series(sp.exp(x), x, 0, 6))
print("sin x :", sp.series(sp.sin(x), x, 0, 8))
print("ln(1+x):", sp.series(sp.log(1+x), x, 0, 6), "  (radius of convergence 1)")
print("1/(1-x):", sp.series(1/(1-x), x, 0, 6), "  (geometric, |x|<1)")
# radius of convergence by the ratio test for sum x^n / n: R = 1
print("\\nratio test on sum x^n/n -> radius R = 1")"""),

md("""## 5. Applications -- volume and arc length

- **Volume of revolution** (disks): rotate $y=f(x)$ about the x-axis,
  $V=\\pi\\int f(x)^2\\,dx$.
- **Arc length**: $L=\\int\\sqrt{1+f'(x)^2}\\,dx$."""),
co("""# volume: rotate y = sqrt(x) on [0,4] about x-axis
f = sp.sqrt(x)
V = sp.pi * sp.integrate(f**2, (x, 0, 4))
print("volume of revolution (y=sqrt(x), 0..4) =", V)
# arc length of y = x^(3/2) on [0,1]
g = x**sp.Rational(3,2)
L = sp.integrate(sp.sqrt(1 + sp.diff(g, x)**2), (x, 0, 1))
print("arc length (y=x^(3/2), 0..1) =", sp.simplify(L), "=", float(L))"""),

md("""## The Calc 2 checklist

1. **Integration:** u-sub, by parts, trig sub, partial fractions.
2. **Improper integrals & p-series:** converge iff the tail beats $1/x$ ($p>1$).
3. **Series tests:** geometric ($|r|<1$), ratio test, alternating, comparison.
4. **Taylor series:** the function as a polynomial, within its radius of convergence.
5. **Applications:** volume by disks/shells, arc length.

These are the exact tools that show up again in physics -- Taylor series for
approximations, integrals for work/energy/charge, series for perturbation theory.
Master the five and Calc 2 is done. Civilian education."""),
]

nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = pathlib.Path("notebooks/calculus2_review.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(nb.cells), "cells")
