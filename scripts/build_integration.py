"""Generate notebooks/symbolic_integration_plus_C.ipynb -- sympy + init_printing
+ loops + pandas: indefinite integrals (the +C), a verified table of integrals,
techniques, and definite integrals. Runs on py312 (pandas).
NOTE: no triple-double-quote docstrings inside cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "symbolic_integration_plus_C.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3.12 (torch)", "language": "python", "name": "py312"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Symbolic Integration with SymPy — the +C, a Verified Table, and `init_printing`

The indefinite integral is a *family* of functions: $\displaystyle\int f(x)\,dx = F(x) + C$, where any
constant $C$ shifts the antiderivative without changing its derivative. SymPy returns one
representative $F$; the **+C** is the freedom we add back. Here we drive SymPy with
`init_printing` (so every result renders as typeset math), build a **table of integrals** with a loop
into pandas, and *verify* each one by differentiating it back to the integrand (the Fundamental
Theorem in action). Runs on the **py312** kernel (pandas).""")

code(r"""import sympy as sp
import pandas as pd
from IPython.display import display, Math

sp.init_printing(use_latex="mathjax")     # typeset every SymPy output
x, C = sp.symbols("x C")
print("sympy", sp.__version__, "| pandas", pd.__version__, "| init_printing on")""")

# ── §1 the +C ──────────────────────────────────────────────────────
md(r"""## §1 The indefinite integral and its +C

`sympy.integrate(f, x)` returns one antiderivative; we append $+C$ to show the whole family. The
point of $C$: differentiating kills it, so *every* member has the same derivative $f$.""")

code(r"""f = sp.cos(x)
F = sp.integrate(f, x)
display(Math(r"\int " + sp.latex(f) + r"\,dx = " + sp.latex(F) + r" + C"))
display(Math(r"\frac{d}{dx}\big(" + sp.latex(F) + r" + C\big) = "
             + sp.latex(sp.diff(F + C, x)) + r" = " + sp.latex(f) + r"\ \checkmark"))
print("the +C is invisible to differentiation -- which is why definite integrals don't need it.")""")

# ── §2 the table (sympy + loop + pandas) ───────────────────────────
md(r"""## §2 A table of integrals — loop, integrate, verify, tabulate

Loop over a list of integrands; for each, SymPy computes $\int f\,dx$ and we confirm
$\frac{d}{dx}\!\int f\,dx = f$. The results go into a pandas table (with the $+C$ noted). Every row
is checked, not asserted.""")

code(r"""n = sp.Symbol("n")
integrands = [x**3, sp.sin(x), sp.exp(2*x), 1/x, 1/(1 + x**2),
              sp.sec(x)**2, x*sp.exp(x), sp.log(x), 1/sp.sqrt(1 - x**2), sp.sinh(x)]

rows = []
for f in integrands:
    F = sp.integrate(f, x)
    verified = sp.simplify(sp.diff(F, x) - f) == 0
    rows.append({"f(x)": f"{f}",
                 "integral  F(x) + C": f"{F}  + C",
                 "d/dx F == f": "yes" if verified else "NO"})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
print("\nall verified:", all(r["d/dx F == f"] == "yes" for r in rows))""")

code(r"""# the same table, but rendered as typeset math via init_printing (first few)
for f in integrands[:5]:
    F = sp.integrate(f, x)
    display(Math(r"\int " + sp.latex(f) + r"\,dx = " + sp.latex(F) + r" + C"))""")

# ── §3 techniques ──────────────────────────────────────────────────
md(r"""## §3 The techniques, each verified

Substitution, integration by parts, and partial fractions — the three workhorses. SymPy handles them
all; we loop and confirm each derivative returns the integrand.""")

code(r"""examples = {
    "u-substitution":      x * sp.cos(x**2),
    "integration by parts": x**2 * sp.exp(x),
    "partial fractions":   (3*x + 5) / ((x + 1)*(x + 2)),
    "trig power":          sp.sin(x)**3,
}
for name, f in examples.items():
    F = sp.integrate(f, x)
    ok = sp.simplify(sp.diff(F, x) - f) == 0
    display(Math(r"\text{(" + name + r")}\quad \int " + sp.latex(f) + r"\,dx = "
                 + sp.latex(sp.simplify(F)) + r" + C" + (r"\ \checkmark" if ok else r"\ ?")))""")

# ── §4 definite integrals ──────────────────────────────────────────
md(r"""## §4 Definite integrals — the +C cancels

$\int_a^b f\,dx = F(b)-F(a)$: the constant subtracts out, which is exactly *why* the indefinite +C is
harmless. A loop over a few classics, symbolic and numeric.""")

code(r"""a, b = sp.symbols("a b")
defints = [(sp.sin(x), 0, sp.pi),
           (sp.exp(-x**2), -sp.oo, sp.oo),       # the Gaussian -> sqrt(pi)
           (1/x, 1, sp.E),                        # -> 1
           (x**2, 0, 1)]
rows = []
for f, lo, hi in defints:
    val = sp.integrate(f, (x, lo, hi))
    rows.append({"integrand": f"{f}", "from": f"{lo}", "to": f"{hi}",
                 "value": f"{val}", "numeric": f"{float(val):.4f}"})
print(pd.DataFrame(rows).to_string(index=False))
display(Math(r"\int_{-\infty}^{\infty} e^{-x^2}\,dx = "
             + sp.latex(sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo)))))""")

# ── §5 wrap ────────────────────────────────────────────────────────
md(r"""## §5 The +C, everywhere

- **The constant of integration is a reference.** It is the same freedom as the electrostatic
  potential's arbitrary zero (the infinite line charge in PS#2 needed a finite reference *because* its
  +C couldn't be pinned at infinity), and the gauge freedom in the vector potential. Physics chooses
  $C$ by a boundary condition; mathematics leaves it free.
- **Differentiate to verify.** Integration is hard, differentiation is easy, so *every* symbolic
  integral should be checked by differentiating back — the loop in §2 is the habit to keep.
- **It is the whole session, quietly.** The Coulomb integral, the Fourier transform, Parseval's
  theorem, the Lennard-Jones energy, the radiated power — all are integrals; `sympy.integrate` with
  `init_printing` is the tool that did them throughout the `griffiths` package.

Pattern shown: SymPy + `init_printing` (typeset output) + Python loops + pandas tables + verify-by-
differentiation. A self-contained notebook (no module).""")

nb.cells = cells
nbf.validate(nb)
OUT.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")
