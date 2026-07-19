"""Generate notebooks/exponential_series_sympy.ipynb -- e^x and a^x Taylor
series built explicitly with a loop (term-by-term partial sums), reusing
dgs.taylor's generic taylor_coefficients/taylor_series (already correct,
not duplicated), cross-checked against SymPy's own series(), and rendered
via display()/init_printing throughout (not sp.pprint). NOTE: no
triple-double-quote docstrings inside cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "exponential_series_sympy.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Exponential Series: $e^x$ and $a^x$, Built Term-by-Term in a Loop

$e^x = \sum_{k=0}^{\infty} \dfrac{x^k}{k!}$ isn't typed in as a known formula
here -- it's built by an explicit loop over Taylor coefficients
$a_k = f^{(k)}(0)/k!$ (reusing `dgs.taylor.taylor_coefficients`, already
correct and not duplicated), then $a^x$ is derived from the SAME $e^x$
series via $a^x = e^{x\ln a}$, not re-derived from scratch -- one series,
one substitution.""")

code(r"""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

from IPython.display import display
import sympy as sp
sp.init_printing()

from dgs.taylor import taylor_coefficients

x, a = sp.symbols('x a', positive=True)
print("SymPy", sp.__version__, "loaded, init_printing enabled")""")

# ── e^x built by an explicit loop ───────────────────────────────────────
md(r"""## $e^x$: the loop, made visible

`taylor_coefficients(exp(x), x, 0, N)` returns $a_k = f^{(k)}(0)/k!$ for
$k=0..N$ -- here the partial sum $S_N(x) = \sum_{k=0}^{N} a_k x^k$ is built
by an explicit `for` loop over those coefficients, not `sum()` hiding it.""")

code(r"""N = 8
coeffs_exp = taylor_coefficients(sp.exp(x), x, 0, N)

S_N = sp.Integer(0)
partial_sums = []
for k, a_k in enumerate(coeffs_exp):
    S_N = S_N + a_k * x**k     # explicit accumulation, term by term
    partial_sums.append(S_N)

print(f"coefficients a_k for k=0..{N}:")
display(coeffs_exp)

print(f"\npartial sum S_{N}(x) built by the loop above:")
display(S_N)

# cross-check against SymPy's own series() -- independent implementation
sympy_series = sp.series(sp.exp(x), x, 0, N+1).removeO()
difference = sp.simplify(sp.expand(S_N) - sp.expand(sympy_series))
print(f"\ndifference from sympy.series(exp(x)): {difference}")
assert difference == 0
print("Verified: the loop-built partial sum EXACTLY matches SymPy's independent series() call.")""")

code(r"""# numeric convergence check: S_N(1) -> e as N grows
import numpy as np
print(f"{'N':>4} {'S_N(1)':>18} {'error vs e':>18}")
e_true = float(sp.E)
for N_test in (1, 2, 4, 6, 8, 10, 15):
    coeffs = taylor_coefficients(sp.exp(x), x, 0, N_test)
    S = sum(a_k * 1**k for k, a_k in enumerate(coeffs))
    S_val = float(S)
    print(f"{N_test:>4} {S_val:>18.10f} {abs(S_val-e_true):>18.2e}")
print(f"\ntrue e = {e_true:.10f}")
print("Error shrinks by roughly 1/(N+1)! per added term -- factorial convergence,")
print("the reason only 8-10 terms already agree with e to many decimal places.")""")

# ── a^x derived from the SAME series, not re-derived ────────────────────
md(r"""## $a^x$: NOT a new series -- the same $e^x$ series, substituted

$a^x = e^{x\ln a}$ (definition of a general base exponential). So the
partial sum for $a^x$ is just $S_N(x)$ from above with $x \to x\ln a$
substituted in -- reusing the loop's output directly, no new derivation.""")

code(r"""S_N_a = S_N.subs(x, x*sp.log(a))
S_N_a_expanded = sp.expand(S_N_a)
print(f"a^x partial sum (degree {N}), from substituting x -> x*ln(a) into the e^x series:")
display(S_N_a)

# cross-check against SymPy's own series of a**x directly
sympy_series_a = sp.series(a**x, x, 0, N+1).removeO()
difference_a = sp.simplify(sp.expand(S_N_a) - sp.expand(sympy_series_a))
print(f"\ndifference from sympy.series(a**x): {sp.simplify(difference_a)}")
assert sp.simplify(difference_a) == 0
print("Verified: substituting into the e^x series EXACTLY reproduces SymPy's")
print("independent series expansion of a^x -- one loop, two functions.")""")

code(r"""# numeric check: does the a^x partial sum actually converge to a^x for a real base/x?
a_val, x_val = 2.0, 0.7
true_value = a_val ** x_val
print(f"{'N':>4} {'partial sum':>18} {'error':>18}")
for N_test in (1, 2, 4, 6, 8, 10, 15):
    coeffs = taylor_coefficients(sp.exp(x), x, 0, N_test)
    S = sum(a_k * (x_val*np.log(a_val))**k for k, a_k in enumerate(coeffs))
    print(f"{N_test:>4} {float(S):>18.10f} {abs(float(S)-true_value):>18.2e}")
print(f"\ntrue 2^0.7 = {true_value:.10f}")""")

md(r"""## Summary

Both series came from ONE loop over Taylor coefficients
(`dgs.taylor.taylor_coefficients`) -- $a^x$ was never independently
derived, only obtained by substituting $x\to x\ln a$ into the already-built
$e^x$ partial sum. Both were cross-checked against SymPy's own `series()`
(an independent code path) and both converge numerically to the true
values with the expected factorial-fast rate.""")

nb["cells"] = cells
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
