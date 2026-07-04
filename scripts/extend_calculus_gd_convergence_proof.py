"""Extends notebooks/calculus_for_college.ipynb with a rigorous convergence
proof for gradient descent on a convex quadratic cost function -- a direct
sequel to the epsilon-delta proof already added (section 1b). Corrects a
common terminology slip: convergence of an ITERATIVE SEQUENCE (gradient
descent's x_n) is an epsilon-N proof, not epsilon-delta (delta belongs to
limits of a continuous function's INPUT, not a discrete iteration count).
Reuses dgs.opt_recursion.gd_closed_form (the exact rsolve'd closed form)
rather than re-deriving it. NOTE: no triple-double-quote docstrings inside
cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "calculus_for_college.ipynb"

nb = nbf.read(str(NB_PATH), as_version=4)
cells = nb.cells
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

gd_md = md(r"""## 1c. Gradient descent's convergence proof (epsilon-N, not epsilon-delta)

A common slip: "epsilon-delta proof for gradient descent" isn't quite the
right name. Epsilon-delta is for the limit of a *continuous function* as
its *input* shrinks to a point. Gradient descent produces a *discrete
sequence* $x_0,x_1,x_2,\dots$ indexed by iteration count $n$ -- its
convergence is properly an **epsilon-N proof**: for every $\epsilon>0$
there is an integer $N$ such that $n\ge N \implies |x_n-x^\*|<\epsilon$.

**Setup.** Minimize the convex quadratic cost $f(x)=\tfrac{a}{2}x^2$ (min
at $x^*=0$) with gradient descent $x_{n+1}=x_n-\eta f'(x_n)=(1-\eta a)x_n$.
`dgs.opt_recursion.gd_closed_form` already solves this recurrence exactly
via `sympy.rsolve`:
$$x_n = x_0\,r^n,\qquad r=1-\eta a.$$

**Proof (epsilon-N).** If $|r|<1$ (equivalently $0<\eta<2/a$, the standard
gradient-descent step-size condition for a quadratic), then
$$|x_n-x^*|=|x_n|=|x_0|\,|r|^n.$$
Given any $\epsilon>0$, solve $|x_0||r|^n<\epsilon$ for $n$:
$$n>\frac{\ln(\epsilon/|x_0|)}{\ln|r|}\quad\Longrightarrow\quad
N=\left\lceil\frac{\ln(\epsilon/|x_0|)}{\ln|r|}\right\rceil$$
(the inequality flips because $\ln|r|<0$). This $N$ is exactly what the
cell below computes and then verifies numerically. $\blacksquare$""")

gd_code = code(r"""import sys, pathlib as _pl
sys.path.insert(0, str(_pl.Path.cwd().parent))
from dgs import opt_recursion as optr

a, x0, eta = 2.0, 5.0, 0.3          # curvature, start point, learning rate
x_n_expr, r = optr.gd_closed_form(a, x0, eta)
print(f"closed form: x_n = {x_n_expr}")
print(f"contraction factor r = 1-eta*a = {r}  (|r|<1 required for convergence: {abs(float(r))<1})")

def N_for_epsilon(epsilon, x0, r):
    r = abs(float(r))
    if r == 0:
        return 0
    return int(np.ceil(np.log(epsilon/abs(x0)) / np.log(r)))

import numpy as np
r_val = float(r)
print()
for epsilon in (1.0, 0.1, 0.01, 1e-4):
    N = N_for_epsilon(epsilon, x0, r_val)
    x_N = x0 * r_val**N
    x_N_minus_1 = x0 * r_val**max(N-1, 0)
    print(f"epsilon={epsilon:<8} N={N:<3} |x_N|={abs(x_N):.3e} (<eps: {abs(x_N)<epsilon})   "
          f"|x_(N-1)|={abs(x_N_minus_1):.3e} (still >=eps just before N: "
          f"{abs(x_N_minus_1)>=epsilon or N==0})")

print("\n==> for every epsilon, N(epsilon) exists and works -- gradient descent")
print("    provably converges to x*=0 for this eta, exactly as the eigenvalue")
print("    condition |r|=|1-eta*a|<1 predicts.")

# and the boundary case: eta too large (|r|>1) genuinely diverges --
# confirming N-existence FAILS exactly when the stability condition fails
eta_bad = 1.5   # eta > 2/a = 1.0, so r = 1-1.5*2 = -2, |r|>1
_, r_bad = optr.gd_closed_form(a, x0, eta_bad)
xs_bad = optr.iterate_map(optr.gd_step(lambda x: a*x, eta_bad), x0, 8)
print(f"\ncounter-example: eta={eta_bad} > 2/a={2/a} -> r={float(r_bad)}, |r|>1")
print(f"iterates: {np.array2string(xs_bad, precision=2)}")
print("no N exists here -- the sequence diverges, matching |r|>1 exactly as predicted.")""")

insert_at = None
for i, c in enumerate(cells):
    if c.cell_type == "markdown" and c.source.strip().startswith("## 2. The derivative rules"):
        insert_at = i
        break
if insert_at is None:
    raise RuntimeError("could not find '## 2. The derivative rules' section to insert before")

cells[insert_at:insert_at] = [gd_md, gd_code]
nb["cells"] = cells
nbf.write(nb, str(NB_PATH))
print(f"inserted gradient-descent convergence proof at index {insert_at}, wrote {NB_PATH}")
