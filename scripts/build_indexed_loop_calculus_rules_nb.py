"""Generate notebooks/indexed_loop_calculus_rules.ipynb -- the generalized
product rule and chain rule, built with SMALL, CONTROLLABLE indexed loops
(the loop length is a single variable at the top of each section, easy to
change and re-run), rendered with real LaTeX via sp.init_printing(use_latex=
'mathjax') + display() -- NOT the ASCII-art sp.pprint() used when this was
first worked out interactively in the terminal. Each loop-built result is
verified against SymPy's own direct differentiation, not just displayed.
NOTE: no triple-double-quote docstrings inside cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "indexed_loop_calculus_rules.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Product Rule and Chain Rule, Generalized -- Built with Small, Controllable Loops

Both the product rule and the chain rule have generalizations to $n$
factors / $n$ composed functions. Rather than write the generalized
formula down and trust it, each one here is built by an explicit,
**small, controllable** Python loop (change one variable -- the number of
factors or links -- at the top of each section and rerun), then checked
against SymPy's own direct differentiation. Real LaTeX rendering
throughout via `init_printing(use_latex='mathjax')` + `display()` -- not
ASCII `pprint`.""")

code(r"""import sympy as sp
from IPython.display import display, Math

sp.init_printing(use_latex='mathjax')
x = sp.symbols('x')""")

md(r"""## 1. The generalized product rule

$$\frac{d}{dx}(u_0 u_1 \cdots u_{n-1}) = \sum_{i=0}^{n-1}\left[\frac{du_i}{dx}\prod_{j\neq i}u_j\right]$$

**Controllable parameter:** the list `u` below -- add or remove factors
and every cell after it adapts automatically (the loops index over
`len(u)`, nothing is hard-coded to 4).""")

code(r"""u = [x, sp.sin(x), sp.exp(x), x**2 + 1]   # <-- change this list, rerun everything below
n = len(u)

for i, ui in enumerate(u):
    display(Math(fr"u_{{{i}}} = " + sp.latex(ui)))""")

code(r"""P = sp.Integer(1)
for ui in u:
    P *= ui
display(Math(r"P = " + sp.latex(P)))

dP_direct = sp.expand(sp.diff(P, x))
display(Math(r"\frac{dP}{dx}\ \text{(direct SymPy diff)} = " + sp.latex(dP_direct)))""")

code(r"""dP_via_rule = sp.Integer(0)
for i in range(n):
    term = sp.diff(u[i], x)
    for j in range(n):
        if j != i:
            term *= u[j]
    display(Math(fr"\text{{term }} i={i}:\quad \frac{{du_{{{i}}}}}{{dx}}\prod_{{j\neq {i}}}u_j = "
                 + sp.latex(term)))
    dP_via_rule += term

dP_via_rule = sp.expand(dP_via_rule)
display(Math(r"\sum_i \text{term}_i = " + sp.latex(dP_via_rule)))

matches = sp.simplify(dP_direct - dP_via_rule) == 0
print(f"n = {n} factors -- loop-built sum matches sp.diff(P,x) exactly: {matches}")
assert matches""")

md(r"""## 2. The generalized chain rule

For a tower $x \to f_0(x) \to f_1(f_0(x)) \to \cdots$:

$$\frac{d}{dx}\Big[f_{n-1}(\cdots f_1(f_0(x)))\Big] = \prod_{k=0}^{n-1}\Big[\text{local derivative of } f_k,\ \text{evaluated at the output of link } k-1\Big]$$

**Controllable parameter:** the list `funcs` below -- add, remove, or
reorder links and every cell after it adapts (indexed over `len(funcs)`).""")

code(r"""funcs = [sp.sin, sp.exp, lambda t: t**2]   # <-- change this list, rerun everything below
n_links = len(funcs)

y = [x]
for k, f in enumerate(funcs):
    y_next = f(y[-1])
    display(Math(fr"y_{{{k+1}}} = f_{{{k}}}(y_{{{k}}}) = " + sp.latex(y_next)))
    y.append(y_next)

total = y[-1]
d_direct = sp.simplify(sp.diff(total, x))
display(Math(r"\frac{d(\text{total})}{dx}\ \text{(direct SymPy diff)} = " + sp.latex(d_direct)))""")

code(r"""t = sp.symbols('t')
d_chain = sp.Integer(1)
for k in range(n_links):
    local_deriv_expr = sp.diff(funcs[k](t), t)
    local_deriv_at_yk = local_deriv_expr.subs(t, y[k])
    display(Math(fr"\text{{link }} {k}:\quad \left.\frac{{df_{{{k}}}}}{{dt}}\right|_{{t=y_{{{k}}}}} = "
                 + sp.latex(local_deriv_at_yk)))
    d_chain *= local_deriv_at_yk

d_chain = sp.simplify(d_chain)
display(Math(r"\prod_k(\text{local derivative}_k) = " + sp.latex(d_chain)))

matches_chain = sp.simplify(d_direct - d_chain) == 0
print(f"n = {n_links} links -- loop-built product matches sp.diff(total,x) exactly: {matches_chain}")
assert matches_chain""")

md(r"""## 3. Why fractions cancel in the chain rule -- back to $\frac{dy}{dx}$ notation

The chain rule "looks like" fraction cancellation, $\dfrac{dy}{du}\cdot\dfrac{du}{dx}=\dfrac{dy}{dx}$,
because that IS exactly true for genuinely finite (nonzero) changes
$\Delta y,\Delta u,\Delta x$ -- ordinary algebra, before any limit is taken.""")

code(r"""dy, du, dx_ = sp.symbols(r'\Delta_y \Delta_u \Delta_x', nonzero=True)

lhs = (dy/du)*(du/dx_)
rhs = dy/dx_
display(Math(sp.latex(sp.Eq(lhs, rhs))))
print("Exact algebra (Delta_u cancels), true for ANY nonzero finite changes:",
      sp.simplify(lhs - rhs) == 0)
print()
print("Taking the limit Delta_x -> 0 (which forces Delta_u -> 0 too, since u is")
print("continuous) turns this exact finite-difference identity into the chain rule")
print("dy/dx = (dy/du)(du/dx) -- the 'fraction' was never just notation.")""")

nb["cells"] = cells
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
