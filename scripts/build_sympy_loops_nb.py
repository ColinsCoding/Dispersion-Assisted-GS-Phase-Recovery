"""Build notebooks/sympy_loops.ipynb -- loops + SymPy + init_printing, outputs baked."""
import io
import contextlib
import pathlib
import nbformat as nbf

ns = {}
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(src):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(src, ns)
    cell = nbf.v4.new_code_cell(src)
    text = buf.getvalue()
    cell.outputs = [nbf.v4.new_output("stream", name="stdout", text=text)] if text else []
    cell.execution_count = None
    cells.append(cell)


md("""# Loops + SymPy (with `init_printing`)
A capital math operator is a **loop**: `Sigma` is a for-loop that adds, `Pi` is a
for-loop that multiplies, and the Gamma function is a product loop (factorial). Here we
build each one with an explicit loop *and* get SymPy's closed form, pretty-printed.
`sp.init_printing()` renders these as typeset math when you run the notebook live;
the baked outputs below use `sp.pretty(...)` (2-D ASCII) so they read without a kernel.""")

code("""import sympy as sp
sp.init_printing()                 # typeset math output when run live
k, n = sp.symbols("k n", integer=True, positive=True)
x = sp.symbols("x")
print("SymPy", sp.__version__, "ready; init_printing on")""")

md("""## Sigma -- a sum loop (accumulator starts at 0)
`sum_{k=1}^{n} k`. The loop accumulates; SymPy gives the closed form `n(n+1)/2`.""")

code("""total = 0
for i in range(1, 6):              # the loop IS the Sigma
    total += i
print("loop sum 1..5 =", total)
S = sp.summation(k, (k, 1, n))     # symbolic closed form
print("sum_{k=1}^{n} k =")
print(sp.pretty(S))
print("check n=5:", S.subs(n, 5))""")

md("""## Pi -- a product loop (accumulator starts at 1)
`prod_{k=1}^{n} k = n!`. Same loop, multiply instead of add.""")

code("""prod = 1
for i in range(1, 6):              # the loop IS the Pi
    prod *= i
print("loop product 1..5 =", prod, "= 5!")
P = sp.factorial(n)
print("prod_{k=1}^{n} k = n! =")
print(sp.pretty(P))
print("Gamma(6) = 5! =", sp.gamma(6))   # the Gamma function = a product loop, extended""")

md("""## A Taylor series, built term-by-term in a loop
`e^x = sum_k x^k / k!`. Loop over `k`, add each term, watch the partial sum grow --
then compare to SymPy's own series.""")

code("""series = 0
for kk in range(5):
    series += x**kk / sp.factorial(kk)
    print(f"after k={kk}:  {series}")          # clean one-line partial sum
print("\\nSymPy series of e^x:")
print(sp.pretty(sp.series(sp.exp(x), x, 0, 5)))""")

md("""## Takeaway
- `Sigma` = a loop that **adds** (accumulator starts at 0); `Pi` = a loop that
  **multiplies** (starts at 1); `Gamma(n) = (n-1)!` = a product loop.
- SymPy gives the **closed form** (`summation`, `factorial`, `series`); `init_printing`
  makes it readable.
- The loop and the symbol are the same computation -- one written for a computer, one
  for a page.""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = pathlib.Path("notebooks/sympy_loops.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(nb.cells), "cells (outputs baked)")
