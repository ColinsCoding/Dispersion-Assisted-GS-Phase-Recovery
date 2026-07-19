"""Generate notebooks/torch_div_cross_product.ipynb -- rule (iv),
div(AxB) = B.(curlA) - A.(curlB), verified purely via torch autograd
(reusing dgs.torch.harmonic_gradient_fields' Jacobian-trace/antisymmetric-
part machinery), over a SMALL CONTROLLABLE loop of vector-field pairs --
add/remove entries in the TEST_FIELDS list and every cell downstream
adapts. Requires py-3.12 (torch). NOTE: no triple-double-quote docstrings
inside cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "torch_div_cross_product.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3.12", "language": "python", "name": "python3"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# div(A x B), Verified Purely via Torch Autograd

$$\nabla\cdot(\mathbf A\times\mathbf B) = \mathbf B\cdot(\nabla\times\mathbf A) - \mathbf A\cdot(\nabla\times\mathbf B)$$

Already proven symbolically (component-by-component and via SymPy) in
`griffiths_ch1_solutions.ipynb` and `six_product_rules.ipynb`. Here it's
checked a THIRD, independent way: `torch.autograd.functional.jacobian`
computes the full $3\times3$ Jacobian of a vector field, divergence is its
**trace**, curl is built from its **antisymmetric part** (reusing
`dgs.torch.harmonic_gradient_fields`, not reimplemented) -- over a small,
controllable list of test field pairs.

**Requires py-3.12** (torch is py-3.12-only in this environment).""")

code(r"""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import torch
from dgs.torch.harmonic_gradient_fields import jacobian_torch, divergence_from_jacobian, curl_from_jacobian""")

md(r"""## The controllable loop

Each entry: a name, and two vector-field functions `A_func`, `B_func`
(torch tensors in, torch tensors out). Add/remove pairs here and every
cell below adapts -- nothing downstream is hard-coded to a specific field.""")

code(r"""TEST_FIELDS = {
    "curl-rich fields": (
        lambda p: torch.stack([p[1]*p[2], -p[0]*p[2], p[0]**2 + p[1]]),
        lambda p: torch.stack([torch.sin(p[0]*p[1]), p[2]**2, p[0]*p[2]]),
    ),
    "polynomial fields": (
        lambda p: torch.stack([p[1]*p[2], p[0]*p[2], p[0]*p[1]]),
        lambda p: torch.stack([p[0]**2, p[1]**2, p[2]**2]),
    ),
    "trig + exponential": (
        lambda p: torch.stack([torch.cos(p[1]), torch.exp(p[2]), p[0]*p[1]]),
        lambda p: torch.stack([p[2]**2, torch.sin(p[0]), p[1]*p[2]]),
    ),
}
TEST_POINT = [1.3, -0.7, 0.5]""")

md(r"""## div(A x B): the LEFT side, via autograd on the composed field""")

code(r"""def div_of_cross(A_func, B_func, point):
    def cross_func(p):
        A, B = A_func(p), B_func(p)
        return torch.stack([
            A[1]*B[2] - A[2]*B[1],
            A[2]*B[0] - A[0]*B[2],
            A[0]*B[1] - A[1]*B[0],
        ])
    J = jacobian_torch(cross_func, point)
    return divergence_from_jacobian(J)""")

md(r"""## B.(curl A) - A.(curl B): the RIGHT side, via autograd on A and B separately""")

code(r"""def rhs_curl_combination(A_func, B_func, point):
    JA = jacobian_torch(A_func, point)
    JB = jacobian_torch(B_func, point)
    curlA = curl_from_jacobian(JA)
    curlB = curl_from_jacobian(JB)
    pt = torch.as_tensor(point, dtype=torch.float64)
    A_val, B_val = A_func(pt), B_func(pt)
    return torch.dot(B_val, curlA) - torch.dot(A_val, curlB)""")

md(r"""## Run the loop -- every test pair must agree""")

code(r"""print(f"{'field pair':>20} | {'div(AxB)':>14} {'B.curlA - A.curlB':>18} {'match':>8}")
all_match = True
for name, (A_func, B_func) in TEST_FIELDS.items():
    lhs = div_of_cross(A_func, B_func, TEST_POINT)
    rhs = rhs_curl_combination(A_func, B_func, TEST_POINT)
    match = abs(lhs.item() - rhs.item()) < 1e-8
    all_match &= match
    print(f"{name:>20} | {lhs.item():>14.6f} {rhs.item():>18.6f} {str(match):>8}")

assert all_match
print(f"\nAll {len(TEST_FIELDS)} field pairs confirm div(AxB) = B.(curlA) - A.(curlB),")
print("computed entirely via torch's Jacobian (trace = divergence, antisymmetric")
print("part = curl) -- a third independent verification of rule (iv), after the")
print("component-by-component proof and the SymPy symbolic check done elsewhere.")""")

md(r"""## Sanity check: a field pair where BOTH sides are trivially zero

Worth confirming the test fields above aren't accidentally always
nonzero -- and that a genuinely curl-free pair correctly gives 0=0, not a
false positive from some unrelated cancellation.""")

code(r"""A_curl_free = lambda p: torch.stack([p[1]*p[2], p[0]*p[2], p[0]*p[1]])   # = grad(xyz), curl-free
B_curl_free = lambda p: torch.stack([p[0]**2, p[1]**2, p[2]**2])               # = grad(x^3/3+y^3/3+z^3/3)/1, curl-free
lhs0 = div_of_cross(A_curl_free, B_curl_free, TEST_POINT)
rhs0 = rhs_curl_combination(A_curl_free, B_curl_free, TEST_POINT)
print(f"curl-free pair: div(AxB)={lhs0.item():.2e}, B.curlA-A.curlB={rhs0.item():.2e}")
assert abs(lhs0.item()) < 1e-8 and abs(rhs0.item()) < 1e-8
print("Both correctly ~0 -- curl-free fields give a trivial (but still consistent) check.")""")

nb["cells"] = cells
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
