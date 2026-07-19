"""Generate notebooks/six_product_rules.ipynb -- all SIX of Griffiths'
vector-calculus product rules (2 for gradients, 2 for divergences, 2 for
curls), verified in one small, controllable loop over generic scalar/
vector functions f,g,A,B (no specific example baked in -- change the
RULES dict and everything downstream adapts), summarized in a pandas
table, and cross-checked numerically via torch autograd for one concrete
example per rule family. NOTE: no triple-double-quote docstrings inside
cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "six_product_rules.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# All Six Product Rules -- One Small Loop, One Table

Two ways to make a scalar from $f,g$ ($fg$) and two ways to make a vector
from $\mathbf A,\mathbf B$ ($f\mathbf A$, $\mathbf A\times\mathbf B$) give
SIX product rules total: two for gradients, two for divergences, two for
curls. Each is verified below for fully GENERIC $f,g,\mathbf A,\mathbf B$
(SymPy `Function` objects, no specific example assumed), in one small
loop -- add/remove/reorder rules in the `RULES` dict and the table below
adapts automatically.""")

code(r"""import sympy as sp
import pandas as pd
import numpy as np

x, y, z = sp.symbols('x y z', real=True)
coords = [x, y, z]

f = sp.Function('f')(*coords)
g = sp.Function('g')(*coords)
A = [sp.Function(f'A{i}')(*coords) for i in range(3)]
B = [sp.Function(f'B{i}')(*coords) for i in range(3)]

def grad(s): return [sp.diff(s, c) for c in coords]
def div(V): return sum(sp.diff(V[i], coords[i]) for i in range(3))
def curl(V):
    return [sp.diff(V[2], coords[1]) - sp.diff(V[1], coords[2]),
            sp.diff(V[0], coords[2]) - sp.diff(V[2], coords[0]),
            sp.diff(V[1], coords[0]) - sp.diff(V[0], coords[1])]
def dot(U, V): return sum(U[i]*V[i] for i in range(3))
def cross(U, V): return [U[1]*V[2]-U[2]*V[1], U[2]*V[0]-U[0]*V[2], U[0]*V[1]-U[1]*V[0]]
def scale(s, V): return [s*V[i] for i in range(3)]
def vsub(U, V): return [U[i]-V[i] for i in range(3)]
def vadd(*vecs): return [sum(v[i] for v in vecs) for i in range(3)]
def dirderiv(U, V): return [sum(U[j]*sp.diff(V[i], coords[j]) for j in range(3)) for i in range(3)]
def is_zero_vec(V): return all(sp.simplify(c) == 0 for c in V)""")

md(r"""## The controllable loop

Each entry: a human-readable rule string, a function returning `(lhs, rhs)`
as SymPy expressions (or 3-lists for vector-valued rules), and its
"family" (gradient/divergence/curl) for the summary table.""")

code(r"""RULES = {
    'i':   ("grad(fg) = f*grad(g) + g*grad(f)",
            lambda: (grad(f*g), vadd(scale(f, grad(g)), scale(g, grad(f)))), 'gradient'),
    'ii':  ("grad(A.B) = AxcurlB + BxcurlA + (A.grad)B + (B.grad)A",
            lambda: (grad(dot(A, B)),
                     vadd(cross(A, curl(B)), cross(B, curl(A)), dirderiv(A, B), dirderiv(B, A))), 'gradient'),
    'iii': ("div(fA) = f*div(A) + A.grad(f)",
            lambda: (div(scale(f, A)), f*div(A) + dot(A, grad(f))), 'divergence'),
    'iv':  ("div(AxB) = B.curlA - A.curlB",
            lambda: (div(cross(A, B)), dot(B, curl(A)) - dot(A, curl(B))), 'divergence'),
    'v':   ("curl(fA) = f*curl(A) - Axgrad(f)",
            lambda: (curl(scale(f, A)), vsub(scale(f, curl(A)), cross(A, grad(f)))), 'curl'),
    'vi':  ("curl(AxB) = (B.grad)A - (A.grad)B + A*div(B) - B*div(A)",
            lambda: (curl(cross(A, B)),
                     vadd(dirderiv(B, A), scale(-1, dirderiv(A, B)), scale(div(B), A), scale(-div(A), B))), 'curl'),
}

rows = []
for name, (text, builder, family) in RULES.items():
    lhs, rhs = builder()
    if isinstance(lhs, list):
        holds = is_zero_vec(vsub(lhs, rhs))
    else:
        holds = sp.simplify(lhs - rhs) == 0
    rows.append({'rule': name, 'family': family, 'statement': text, 'holds': holds})

df = pd.DataFrame(rows)
df""")

md(r"""## Sanity check: all six actually hold

If any entry in the `holds` column were `False`, one of the six formulas
above would be wrong -- this isn't decoration, it's the actual proof for
generic $f,g,\mathbf A,\mathbf B$.""")

code(r"""assert df['holds'].all(), df[~df['holds']]
print(f"All {len(df)} rules verified for fully generic f, g, A, B.")
df.groupby('family')['rule'].apply(list)""")

md(r"""## Numerical cross-check via finite-difference Jacobians

One concrete example per family, using plain NumPy central-difference
Jacobians (the same trace-is-divergence / antisymmetric-part-is-curl
formalization used in `dgs.torch.harmonic_gradient_fields`, just via
finite differences here rather than autograd, since this notebook runs
under the py-3.13 kernel the rest of this repo's Griffiths notebooks use,
and torch is py-3.12-only in this environment) -- confirms the symbolic
identities above aren't an artifact of SymPy's own simplification rules.""")

code(r"""def numpy_grad(f_func, p, h=1e-6):
    p = np.asarray(p, dtype=float)
    g = np.zeros(3)
    for i in range(3):
        dp = np.zeros(3); dp[i] = h
        g[i] = (f_func(p+dp) - f_func(p-dp)) / (2*h)
    return g

def numpy_jacobian(v_func, p, h=1e-6):
    p = np.asarray(p, dtype=float)
    J = np.zeros((3, 3))
    for j in range(3):
        dp = np.zeros(3); dp[j] = h
        J[:, j] = (np.array(v_func(p+dp)) - np.array(v_func(p-dp))) / (2*h)
    return J

def numpy_div(v_func, p): return np.trace(numpy_jacobian(v_func, p))
def numpy_curl(v_func, p):
    J = numpy_jacobian(v_func, p)
    return np.array([J[2,1]-J[1,2], J[0,2]-J[2,0], J[1,0]-J[0,1]])

# concrete f, g, A, B
f_num = lambda p: p[0]**2*p[1] + np.sin(p[2])
g_num = lambda p: p[0]*p[1]*p[2]
A_num = lambda p: np.array([p[1], p[2], p[0]])
B_num = lambda p: np.array([p[0]*p[2], p[1]**2, p[2]-p[0]])

p0 = np.array([1.3, -0.6, 0.9])

# rule (i): grad(fg) numerically
fg_num = lambda p: f_num(p)*g_num(p)
lhs_i = numpy_grad(fg_num, p0)
rhs_i = f_num(p0)*numpy_grad(g_num, p0) + g_num(p0)*numpy_grad(f_num, p0)
print('rule (i)  max error:', np.max(np.abs(lhs_i - rhs_i)))

# rule (iv): div(AxB) numerically
cross_num = lambda p: np.cross(A_num(p), B_num(p))
lhs_iv = numpy_div(cross_num, p0)
rhs_iv = np.dot(B_num(p0), numpy_curl(A_num, p0)) - np.dot(A_num(p0), numpy_curl(B_num, p0))
print('rule (iv) max error:', np.abs(lhs_iv - rhs_iv))

# rule (v): curl(fA) numerically
fA_num = lambda p: f_num(p)*A_num(p)
lhs_v = numpy_curl(fA_num, p0)
rhs_v = f_num(p0)*numpy_curl(A_num, p0) - np.cross(A_num(p0), numpy_grad(f_num, p0))
print('rule (v)  max error:', np.max(np.abs(lhs_v - rhs_v)))

assert np.max(np.abs(lhs_i - rhs_i)) < 1e-4
assert np.abs(lhs_iv - rhs_iv) < 1e-4
assert np.max(np.abs(lhs_v - rhs_v)) < 1e-4
print('\nAll three numerically cross-checked rules agree with the symbolic proof above.')""")

nb["cells"] = cells
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
