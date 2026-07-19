"""Extends notebooks/griffiths_notation_glossary_sympy.ipynb with the
Helmholtz theorem: any vector field decomposes into a curl-free part
(gradient of a scalar potential) plus a divergence-free part (curl of a
vector potential). Direct sequel to the Dirac-delta section already in
this notebook -- the same nabla^2(1/r)=-4*pi*delta^3(r) Green's function
verified there is exactly what makes the potentials' defining Poisson
equations solvable. NOTE: no triple-double-quote docstrings inside cell
strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "griffiths_notation_glossary_sympy.ipynb"

nb = nbf.read(str(NB_PATH), as_version=4)
cells = nb.cells
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

helm_md = md(r"""## The Helmholtz theorem: every vector field is a gradient plus a curl

Any sufficiently smooth vector field $\vec F$ (vanishing at infinity) can be
written as
$$\vec F = -\nabla\phi + \nabla\times\vec A$$
a curl-free piece ($-\nabla\phi$) plus a divergence-free piece
($\nabla\times\vec A$) -- this is WHY electrostatics gets to write
$\vec E=-\nabla V$ and magnetostatics gets to write $\vec B=\nabla\times\vec A$:
Helmholtz's theorem guarantees the decomposition exists for any field, not
just these two.

$\phi$ and $\vec A$ are built directly from $\vec F$'s divergence and curl:
$$\phi(\vec r)=\frac{1}{4\pi}\int\frac{\nabla'\cdot\vec F(\vec r\,')}{\mathscr{r}}\,d^3r',
\qquad
\vec A(\vec r)=\frac{1}{4\pi}\int\frac{\nabla'\times\vec F(\vec r\,')}{\mathscr{r}}\,d^3r'$$
using the SAME $1/\mathscr{r}$ Green's function whose Laplacian identity
($\nabla^2(1/r)=-4\pi\delta^3(r)$) was already verified above -- that
identity is precisely what makes $\phi,\vec A$ solve
$\nabla^2\phi=-\nabla\cdot\vec F$ and $\nabla^2\vec A=-\nabla\times\vec F$
(apply $\nabla^2$ under the integral and the delta function collapses it
onto the source point).

Rather than re-deriving the full convolution integral, the proof that this
decomposition actually reconstructs $\vec F$ rests on ONE vector identity --
verified symbolically below for a fully generic vector field, not a
specific example.""")

helm_code1 = code(r"""xs, ys, zs = sp.symbols('x y z', real=True)
Ax = sp.Function('A_x')(xs, ys, zs)
Ay = sp.Function('A_y')(xs, ys, zs)
Az = sp.Function('A_z')(xs, ys, zs)
A_vec = [Ax, Ay, Az]
coords3 = [xs, ys, zs]

def curl3(F):
    Fx, Fy, Fz = F
    return [sp.diff(Fz, ys) - sp.diff(Fy, zs),
            sp.diff(Fx, zs) - sp.diff(Fz, xs),
            sp.diff(Fy, xs) - sp.diff(Fx, ys)]

def div3(F):
    return sum(sp.diff(F[i], coords3[i]) for i in range(3))

def grad3(f):
    return [sp.diff(f, c) for c in coords3]

curl_A = curl3(A_vec)
curl_curl_A = curl3(curl_A)
grad_div_A = grad3(div3(A_vec))
laplacian_A = [sum(sp.diff(A_vec[i], c, 2) for c in coords3) for i in range(3)]

print("Verifying curl(curl(A)) = grad(div(A)) - laplacian(A) for a FULLY GENERIC")
print("vector field A(x,y,z) -- the one identity Helmholtz's reconstruction needs:")
for i, name in enumerate(('x', 'y', 'z')):
    lhs = sp.simplify(curl_curl_A[i])
    rhs = sp.simplify(grad_div_A[i] - laplacian_A[i])
    residual = sp.simplify(lhs - rhs)
    print(f"  component {name}: curl(curl A) - [grad(div A) - laplacian(A)] = {residual}")
    assert residual == 0
print("\nVerified: all three components vanish EXACTLY, for an arbitrary A -- not a")
print("special-case field chosen to make it work.")""")

helm_md2 = md(r"""### Why that identity finishes the proof

With $\vec F=-\nabla\phi+\nabla\times\vec A$ and the Coulomb-gauge choice
$\nabla\cdot\vec A=0$:
$$\nabla\cdot\vec F = -\nabla^2\phi + \nabla\cdot(\nabla\times\vec A) = -\nabla^2\phi + 0$$
(divergence of a curl is always zero -- already implicit in the div/curl
section above) which matches $\nabla^2\phi=-\nabla\cdot\vec F$ by construction.
For the curl:
$$\nabla\times\vec F = -\nabla\times\nabla\phi + \nabla\times(\nabla\times\vec A)
= 0 + \big[\nabla(\nabla\cdot\vec A)-\nabla^2\vec A\big] = -\nabla^2\vec A$$
(curl of a gradient is always zero; the bracket is EXACTLY the identity just
verified, and $\nabla\cdot\vec A=0$ kills the first term) -- which matches
$\nabla^2\vec A=-\nabla\times\vec F$ by construction. Both defining equations
are satisfied automatically, so $\vec F$ is reconstructed exactly.""")

def find_index(snippet):
    for i, c in enumerate(cells):
        if c.cell_type == "markdown" and snippet in c.source:
            return i
    raise ValueError(f"not found: {snippet!r}")

idx_lagrangian = find_index("## Lagrangian mechanics")
for c in (helm_md2, helm_code1, helm_md):
    cells.insert(idx_lagrangian, c)

glossary_cell = cells[find_index("## Glossary")]
glossary_cell.source += (
    "\n| Helmholtz theorem | $\\vec F=-\\nabla\\phi+\\nabla\\times\\vec A$ for any smooth field |"
    "\n| $\\nabla\\times(\\nabla\\times\\vec A)$ | $=\\nabla(\\nabla\\cdot\\vec A)-\\nabla^2\\vec A$ (verified symbolically above) |"
)

nb.cells = cells
nbf.write(nb, str(NB_PATH))
print(f"wrote {NB_PATH} with {len(cells)} cells")
