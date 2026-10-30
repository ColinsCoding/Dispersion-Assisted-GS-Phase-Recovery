"""Build notebooks/linear_algebra_proofs.ipynb: SymPy + Lean linear algebra, supply and demand, attention.

The Lean cells are cut from lean/LinearAlgebraProofs.lean and compiled once by %lean_check.

Run:  py -3.12 tools/build_linalg_nb.py
Then: py -3.12 -m jupyter nbconvert --to notebook --execute --inplace notebooks/linear_algebra_proofs.ipynb
"""
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "linear_algebra_proofs.ipynb"
LEAN_SRC = ROOT / "lean" / "LinearAlgebraProofs.lean"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


text = LEAN_SRC.read_text(encoding="utf-8")
body, footer = text.rsplit("end LinAlg", 1)
parts = re.split(r"(?m)^(?=/-! ## \d+\.)", body)
HEADER, LEAN, FOOTER = parts[0], {int(re.match(r"/-! ## (\d+)\.", p).group(1)): p for p in parts[1:]}, "end LinAlg" + footer


def lean(name, src):
    code(f"%%lean {name}\n{src.strip()}")


md(r"""
# Linear algebra proofs: SymPy, Lean, supply and demand, and attention in torch

Two kinds of proof, side by side:

* **SymPy** proves an identity for one size: make every matrix entry its own symbol, expand
  lhs − rhs, and get exactly 0. That is a real proof for 2×2 or 3×3, but only for that size.
* **Lean** proves it for **every** size and checks each step by machine.

Then the same algebra runs two applications: a supply-and-demand market (a 2×2 linear system,
also solved in C with gcc) and attention in torch (softmax of a matrix product).

| # | Claim | Lean | Check here |
|---|---|---|---|
| 1 | det(AB) = det A det B, (AB)ᵀ = BᵀAᵀ, tr(AB) = tr(BA) | `det_product`, `transpose_product`, `trace_cyclic` | symbolic 2×2, 3×3 |
| 2 | a null space is closed under midpoints | `kernel_midpoint` | `nullspace()` |
| 3 | Cauchy-Schwarz | `cauchy_schwarz_2d` | Lagrange identity in 2-D and 3-D |
| 4 | P² = P ⇒ (1 − P)² = 1 − P | `complement_projection` | P = vvᵀ/vᵀv |
| 5 | market price P* = (a − c)/(b + d) | `equilibrium_price`, `demand_shift_raises_price` | SymPy + gcc |
| 6 | softmax: positive, sums to 1, shift-invariant | `softmax_pos`, `softmax_sum_one`, `softmax_shift` | torch attention |
""")
code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import numpy as np, sympy as sp, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
sp.init_printing(use_unicode=True)
from dgs import linear_algebra_proofs as L
""")
code("%load_ext dgs.lean_magic\n%lean_reset")
lean("header", HEADER)

md(r"""
## 1. Matrix rules, proved symbolically

Every entry of A and B is a separate symbol, so a zero difference holds for *all* real 2×2
(or 3×3) matrices. The determinant rule is why "a product of invertible matrices is invertible":
det(AB) ≠ 0 when both determinants are nonzero.
""")
code(r"""
A = L.symbolic_matrix("a", 2)
display(A, sp.Eq(sp.Symbol("det A"), A.det()))
display(pd.DataFrame({"2x2": L.prove_identities(2), "3x3": L.prove_identities(3)}))
""")
lean("matrix", LEAN[1])

md(r"""
## 2. Midpoints: why a null space is a subspace

If Av = 0 and Aw = 0, then A((v + w)/2) = (Av + Aw)/2 = 0. Linearity is the whole proof.
Compare the phase-retrieval notebook: the set |z| = 1 fails this test (the midpoint of 1 and −1
is 0). Linear constraints are convex; magnitude constraints are not.
""")
code(r"""
M = sp.Matrix([[1, 1, 1], [1, -1, 0]])
basis, ok = L.kernel_midpoint(M)
display(M, basis)
print("midpoint of two null vectors is null:", ok, "  |  rank + nullity =", M.rank(), "+", len(basis), "=", M.shape[1])
""")
lean("kernel", LEAN[2])

md(r"""
## 3. Cauchy-Schwarz from a sum of squares

|u|²|v|² − (u·v)² equals Σ (uᵢvⱼ − uⱼvᵢ)², a sum of squares, so it can't be negative. Lean's
`nlinarith` finds the same certificate when given the hint (ad − bc)² ≥ 0.
""")
code(r"""
for n in (2, 3):
    gap, sos = L.cauchy_schwarz(n)
    display(sp.Eq(sp.factor(gap), sp.factor(sos)) if n == 2 else sos)
    print(f"n={n}: gap - sum of squares =", sp.expand(gap - sos))
""")
lean("cauchy", LEAN[3])

md(r"""
## 4. Projections

P = vvᵀ/(vᵀv) projects onto the line through v. Projecting twice changes nothing (P² = P), and
1 − P projects onto everything perpendicular. Lean proves the complement fact in any ring, so it
holds for matrices of every size, and for operators in quantum mechanics too.
""")
code(r"""
P, props = L.projection([1, 2, 2])
display(P)
display(pd.Series(props, name="holds"))
print("eigenvalues of P:", P.eigenvals(), " (a projection only has eigenvalues 0 and 1)")
""")
lean("projection", LEAN[4])

md(r"""
## 5. Supply and demand as a 2×2 system (photonics example)

Demand Q = a − bP and supply Q = c + dP. Written as a matrix:

[[1, b], [1, −d]] · [Q, P] = [a, c], with det = −(b + d).

The numbers are made up, for a 1550 nm DFB laser diode (Q in thousands per month, P in dollars).
The same 2×2 solve is in ANSI C (`dgs/c/market/market.c`, Cramer's rule, `gcc -std=c89 -Werror`).
""")
code(r"""
a, b, c, d = sp.symbols("a b c d", positive=True)
m = L.market(a, b, c, d)
display(m["A"], sp.Eq(sp.Symbol("P^*"), m["P"]), sp.Eq(sp.Symbol("Q^*"), m["Q"]))
pm = L.PHOTONICS_MARKET
s = L.market_shift(**pm, demand_shift=10, supply_shift=-5)
print("SymPy: before (P, Q) =", s["before"], " after =", tuple(float(v) for v in s["after"]))
try:
    print("gcc  :", L.market_c(**pm, demand_shift=10, supply_shift=-5))
except RuntimeError as e:
    print("gcc unavailable:", e)
Pg = np.linspace(0, 200, 200)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(pm["a"] - pm["b"] * Pg, Pg, label="demand"); ax.plot(pm["c"] + pm["d"] * Pg, Pg, label="supply")
ax.plot(pm["a"] + 10 - pm["b"] * Pg, Pg, "--", label="demand +10 (more fiber links)")
ax.plot(pm["c"] - 5 + pm["d"] * Pg, Pg, ":", label="supply -5 (wafer shortage)")
ax.plot(*[float(v) for v in s["before"][::-1]], "ko"); ax.plot(*[float(v) for v in s["after"][::-1]], "rs")
ax.set(xlabel="Q (thousands / month)", ylabel="P ($)", xlim=(0, 150), ylim=(0, 200)); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")
lean("market", LEAN[5])

md(r"""
## 6. Attention in torch is linear algebra plus softmax

out = softmax(QKᵀ/√d) V. Each output row is a weighted average of the rows of V, because the
softmax weights are positive and sum to 1 (Lean proves both). Adding a constant to a row of
scores changes nothing (`softmax_shift`), which is why torch subtracts the row max before exp:
same answer, no overflow. Syntax to remember:

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(d)     # (n_q, n_k)
weights = torch.softmax(scores, dim=-1)              # each row sums to 1
out = weights @ V                                    # (n_q, d_v)
```
""")
code(r"""
w, total, shift = L.softmax_symbolic(3)
display(w[0], sp.Eq(sp.Symbol("sum"), total))
print("shift by c changes each weight by:", shift)
try:
    display(pd.Series(L.check_attention(), name="result"))
except ImportError:
    print("torch not installed: skipped")
""")
lean("softmax", LEAN[6])

md("## 7. Compile every Lean cell")
lean("footer", FOOTER)
code(r"""
lean_result = %lean_check
from dgs.lean_magic import SESSION
display(pd.DataFrame({"Lean theorem": SESSION.last.theorems, "status": SESSION.last.status}))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("wrote", OUT, "with", len(cells), "cells")
