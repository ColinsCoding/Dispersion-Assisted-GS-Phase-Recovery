"""Extends notebooks/calculus_for_college.ipynb's section 1 (currently only
a NUMERICAL illustration of the difference quotient shrinking) with an
actual epsilon-delta PROOF that f'(x)=2x for f(x)=x^2: explicitly
constructing delta(epsilon) algebraically, verifying the definition holds
via SymPy, then confirming computationally at the delta boundary (just
inside passes, just outside can fail). NOTE: no triple-double-quote
docstrings inside cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "calculus_for_college.ipynb"

nb = nbf.read(str(NB_PATH), as_version=4)
cells = nb.cells
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

proof_md = md(r"""## 1b. A genuine epsilon-delta proof (not just a numerical picture)

The cells above show the difference quotient's error shrinking as $h\to 0$ --
that's evidence, not a proof. A real epsilon-delta proof requires
**constructing $\delta(\epsilon)$ explicitly** and showing it works for
*every* $\epsilon>0$, not just the four sample values tried above.

**Claim:** for $f(x)=x^2$, $f'(x_0)=2x_0$.

**Proof.** The difference quotient is
$$Q(h)=\frac{f(x_0+h)-f(x_0)}{h}=\frac{(x_0+h)^2-x_0^2}{h}=\frac{2x_0 h+h^2}{h}=2x_0+h
\quad (h\neq 0).$$
So $|Q(h)-2x_0| = |h|$ **exactly** -- no approximation. Given any
$\epsilon>0$, choose $\delta=\epsilon$. Then
$$0<|h|<\delta \implies |Q(h)-2x_0|=|h|<\delta=\epsilon,$$
which is precisely the definition of $\lim_{h\to0}Q(h)=2x_0$. $\blacksquare$

This $\delta=\epsilon$ (a 1-to-1 tolerance map) is a special feature of
$x^2$ having no higher-than-quadratic terms; the cell below verifies the
algebra symbolically, then checks the definition computationally right at
the boundary $|h|=\delta$.""")

proof_code = code(r"""h_sym, x0_sym, eps = sp.symbols('h x0 epsilon', real=True)
f_expr = x0_sym**2

Q = sp.simplify((( (x0_sym+h_sym)**2 - x0_sym**2) / h_sym))
print("Q(h) simplified:", Q)                  # should be 2*x0 + h
assert sp.simplify(Q - (2*x0_sym + h_sym)) == 0

error_expr = sp.simplify(Q - 2*x0_sym)
print("Q(h) - 2*x0 =", error_expr)             # should be exactly h
assert sp.simplify(error_expr - h_sym) == 0
print("\n==> |Q(h)-2*x0| = |h| exactly, so delta(epsilon) = epsilon works for ANY x0.")

# computational check: for several (x0, epsilon) pairs, delta=epsilon really
# is the boundary -- just inside satisfies the definition, just outside can fail
import numpy as np
rng = np.random.default_rng(0)
all_passed = True
for _ in range(8):
    x0 = float(rng.uniform(-5, 5))
    epsilon = float(rng.uniform(1e-4, 1.0))
    delta = epsilon                             # the constructed delta

    def Q_num(h):
        return ((x0 + h)**2 - x0**2) / h

    h_inside = 0.99 * delta                      # just inside the delta-neighborhood
    h_outside = 1.5 * delta                      # just outside it

    err_inside = abs(Q_num(h_inside) - 2*x0)
    err_outside = abs(Q_num(h_outside) - 2*x0)

    ok_inside = err_inside < epsilon             # definition MUST hold here
    fails_outside = err_outside >= epsilon       # and (for this delta=epsilon choice) breaks exactly here
    all_passed &= ok_inside and fails_outside

    print(f"x0={x0:+.3f} eps={epsilon:.4f}  "
          f"|h|=0.99*delta -> err={err_inside:.4f} (<eps: {ok_inside})   "
          f"|h|=1.5*delta -> err={err_outside:.4f} (>=eps: {fails_outside})")

assert all_passed
print("\nVerified: delta=epsilon is not just sufficient, it's the EXACT boundary --")
print("the proof's algebra (|Q(h)-2x0|=|h|) is confirmed computationally at the edge.")""")

# insert right after the existing epsilon-delta illustration (index 3, the
# secant-line code cell) and before "## 2. The derivative rules"
insert_at = None
for i, c in enumerate(cells):
    if c.cell_type == "markdown" and c.source.strip().startswith("## 2. The derivative rules"):
        insert_at = i
        break
if insert_at is None:
    raise RuntimeError("could not find '## 2. The derivative rules' section to insert before")

cells[insert_at:insert_at] = [proof_md, proof_code]
nb["cells"] = cells
nbf.write(nb, str(NB_PATH))
print(f"inserted epsilon-delta proof section at index {insert_at}, wrote {NB_PATH}")
