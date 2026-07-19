"""Further extends notebooks/griffiths_notation_glossary_sympy.ipynb with the
Dirac delta function in electrodynamics: point-charge density rho = q*delta^3(r),
and the key identity div(r_hat/r^2) = 4*pi*delta^3(r) that makes Gauss's law
work AT a point charge -- verified two ways (divergence is exactly zero away
from the origin, symbolically; total flux through any enclosing sphere is
4*pi regardless of radius, symbolically), the same flux-argument Griffiths
uses rather than naive differentiation of a singular function. NOTE: no
triple-double-quote docstrings inside cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "griffiths_notation_glossary_sympy.ipynb"

nb = nbf.read(str(NB_PATH), as_version=4)
cells = nb.cells
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

delta_md = md(r"""## The Dirac delta function in electrodynamics

A point charge $q$ sitting at the source point $\vec r\,'$ (defined above)
has charge density $\rho(\vec r) = q\,\delta^3(\vec r - \vec r\,')$ -- all the
charge concentrated at one point, zero everywhere else, but integrating to
the total charge $q$.

The identity that makes Gauss's law ($\nabla\cdot\vec E=\rho/\epsilon_0$)
actually hold AT the point charge is
$$\nabla\cdot\left(\frac{\hat{\mathscr{r}}}{\mathscr{r}^2}\right) = 4\pi\,\delta^3(\mathscr{r})$$
You can't get this by naively differentiating $1/r^2$ (it blows up at the
origin) -- Griffiths' actual argument is a FLUX argument: show the
divergence is exactly zero everywhere except the origin, then show the
total flux through ANY sphere enclosing the origin is $4\pi$ regardless of
radius. Both halves are checked symbolically below, not asserted.""")

delta_code1 = code(r"""xs, ys, zs = sp.symbols('x y z', real=True)
r_cart = sp.sqrt(xs**2 + ys**2 + zs**2)
E_field = sp.Matrix([xs, ys, zs]) / r_cart**3   # this IS r_hat/r^2, in Cartesian components

divergence = sum(sp.diff(E_field[i], (xs, ys, zs)[i]) for i in range(3))
divergence_simplified = sp.simplify(divergence)
print("div(r_hat/r^2) computed directly in Cartesian coordinates, for r != 0:")
display(divergence_simplified)
assert divergence_simplified == 0
print("Verified: EXACTLY zero everywhere except the origin (where r=0 makes E_field singular")
print("and this Cartesian computation doesn't apply -- that singular point is where all the")
print("delta function's content has to live).")""")

delta_code2 = code(r"""theta, phi, R = sp.symbols('theta phi R', positive=True)

# on a sphere of radius R: r_hat/r^2 = (1/R^2) r_hat, and the outward normal
# IS r_hat, so E . n_hat = 1/R^2 exactly (constant over the whole sphere)
E_dot_n = 1 / R**2
area_element = R**2 * sp.sin(theta)   # spherical-coordinates area element

flux = sp.integrate(sp.integrate(E_dot_n * area_element, (phi, 0, 2*sp.pi)), (theta, 0, sp.pi))
print("Total flux of r_hat/r^2 through a sphere of radius R (any R > 0):")
display(sp.simplify(flux))
assert sp.simplify(flux - 4*sp.pi) == 0
print("Verified: EXACTLY 4*pi, independent of R -- shrink the sphere down to the origin")
print("and the flux doesn't change, so all 4*pi of it must be concentrated at a single")
print("point. Zero divergence everywhere except a point, with 4*pi of total flux")
print("concentrated there, IS the defining property of 4*pi*delta^3(r) -- both halves")
print("of the identity now demonstrated, not just quoted.")""")

delta_md2 = md(r"""### Consistency check: the companion identity for the potential

The point-charge potential $V=\dfrac{q}{4\pi\epsilon_0 r}$ satisfies Poisson's
equation $\nabla^2 V = -\rho/\epsilon_0$ only because of the companion
identity $\nabla^2(1/r) = -4\pi\delta^3(r)$. Same structure: verify
$\nabla^2(1/r)=0$ away from the origin symbolically (the flux argument above
already established the delta strength, so only the "zero away from the
source" half needs re-checking here).""")

delta_code3 = code(r"""V_shape = 1 / r_cart
laplacian_V = sum(sp.diff(V_shape, coord, 2) for coord in (xs, ys, zs))
laplacian_V_simplified = sp.simplify(laplacian_V)
print("Laplacian of 1/r in Cartesian coordinates, for r != 0:")
display(laplacian_V_simplified)
assert laplacian_V_simplified == 0
print("Verified: also exactly zero away from the origin -- consistent with")
print("nabla^2(1/r) = -4*pi*delta^3(r), the identity Poisson's equation needs.")""")

def find_index(snippet):
    for i, c in enumerate(cells):
        if c.cell_type == "markdown" and snippet in c.source:
            return i
    raise ValueError(f"not found: {snippet!r}")

idx_lagrangian = find_index("## Lagrangian mechanics")
for c in (delta_code3, delta_md2, delta_code2, delta_code1, delta_md):
    cells.insert(idx_lagrangian, c)

glossary_cell = cells[find_index("## Glossary")]
glossary_cell.source += (
    "\n| $\\delta^3(\\vec r)$ | 3D Dirac delta: zero except at origin, integrates to 1 |"
    "\n| $\\nabla\\cdot(\\hat{\\mathscr{r}}/\\mathscr{r}^2)=4\\pi\\delta^3(\\mathscr{r})$ | Gauss's law works AT a point charge |"
)

nb.cells = cells
nbf.write(nb, str(NB_PATH))
print(f"wrote {NB_PATH} with {len(cells)} cells")
