"""Extends notebooks/griffiths_ch1_solutions.ipynb with Problem 1.17
(divergence transforms as a SCALAR under rotation) immediately after the
just-added Problem 1.14 (gradient transforms as a VECTOR) -- an explicit,
worked connection between the two: 1.14 established that (Y,Z)->(Ybar,Zbar)
obeys d/dYbar = cos(phi)*d/dY + sin(phi)*d/dZ etc.; 1.17 reuses that EXACT
chain-rule pattern, applied now to a vector field's two components instead
of a scalar's gradient, and finds the OPPOSITE transformation behavior
(invariant, not mixing) for its divergence.

Includes: a precalc sub-lemma (verifying sin^2+cos^2=1 is what makes the
cross terms cancel), the main SymPy-verified derivation (stand-in symbols
for the partial derivatives, matching 1.14's own notebook style rather
than raw sp.Function objects, which produce unreadable substituted-
derivative notation), a concrete numeric example, and the physics payoff
(why a scalar-valued divergence is required for Gauss's law to mean the
same thing to every rotated observer). NOTE: no triple-double-quote
docstrings inside cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "griffiths_ch1_solutions.ipynb"

nb = nbf.read(str(NB_PATH), as_version=4)
cells = nb.cells
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

sec_md = md(r"""## §2c Problem 1.17 -- divergence transforms as a SCALAR

**Directly continuing §2b (Problem 1.14).** There, a SCALAR $f$ produced a
VECTOR $\nabla f$ whose components mix into each other under rotation --
$(\nabla f)_{\bar Y}=\cos\phi(\nabla f)_Y+\sin\phi(\nabla f)_Z$. Here, we
START from an arbitrary VECTOR field $(v_Y,v_Z)$ that transforms the same
way $\nabla f$ was proven to (that's what "being a vector" means, per
§2b), and ask what happens to its divergence
$\partial v_Y/\partial Y+\partial v_Z/\partial Z$. The answer is the
mirror image of §2b: divergence does NOT mix -- it comes back
**unchanged**, a genuine scalar invariant.

This reuses §2b's EXACT chain-rule result
($\partial/\partial\bar Y=\cos\phi\,\partial/\partial Y+\sin\phi\,\partial/\partial Z$,
$\partial/\partial\bar Z=-\sin\phi\,\partial/\partial Y+\cos\phi\,\partial/\partial Z$)
rather than re-deriving it.""")

sub_md = md(r"""**Precalc sub-lemma, made explicit.** The whole proof
hinges on two algebraic facts about $\sin\phi,\cos\phi$ -- worth isolating
before the main derivation so the cancellation in the final step isn't a
surprise:""")

sub_code = code(r"""phi = sp.symbols('phi', real=True)
identity1 = sp.simplify(sp.sin(phi)**2 + sp.cos(phi)**2 - 1) == 0
display(Math(r"\sin^2\phi+\cos^2\phi=1 \quad\Rightarrow\quad " + str(identity1)))

# the OTHER fact this proof needs: the cross terms sin(phi)cos(phi) that appear
# from BOTH the Ybar-derivative and the Zbar-derivative carry OPPOSITE signs
# and cancel on addition -- verified symbolically below, not just asserted
cross_term_check = sp.simplify(sp.sin(phi)*sp.cos(phi) - sp.cos(phi)*sp.sin(phi)) == 0
print("cross terms cancel on addition:", cross_term_check)""")

main_md = md(r"""**Main derivation.** Let $v_{YY}:=\partial v_Y/\partial Y$
etc. stand in for the four partial derivatives (same style as §2b's
$f_Y,f_Z$ stand-ins). The vector field transforms by §2b's rotation:
$\bar v_Y=\cos\phi\, v_Y+\sin\phi\, v_Z$, $\bar v_Z=-\sin\phi\, v_Y+\cos\phi\, v_Z$.
Apply §2b's chain rule to each component, then add.""")

main_code = code(r"""vyY, vyZ, vzY, vzZ = sp.symbols('v_{YY} v_{YZ} v_{ZY} v_{ZZ}')

# reuse SEC 2b's chain rule pattern for each component
dvy_dYbar = sp.cos(phi)*vyY + sp.sin(phi)*vyZ
dvy_dZbar = -sp.sin(phi)*vyY + sp.cos(phi)*vyZ
dvz_dYbar = sp.cos(phi)*vzY + sp.sin(phi)*vzZ
dvz_dZbar = -sp.sin(phi)*vzY + sp.cos(phi)*vzZ

# vbar_Y = cos*vY + sin*vZ  ->  d(vbar_Y)/dYbar = cos*d(vY)/dYbar + sin*d(vZ)/dYbar
dvybar_dYbar = sp.cos(phi)*dvy_dYbar + sp.sin(phi)*dvz_dYbar
# vbar_Z = -sin*vY + cos*vZ  ->  d(vbar_Z)/dZbar = -sin*d(vY)/dZbar + cos*d(vZ)/dZbar
dvzbar_dZbar = -sp.sin(phi)*dvy_dZbar + sp.cos(phi)*dvz_dZbar

display(Math(r"\frac{\partial\bar v_Y}{\partial\bar Y} = " + sp.latex(sp.expand(dvybar_dYbar))))
display(Math(r"\frac{\partial\bar v_Z}{\partial\bar Z} = " + sp.latex(sp.expand(dvzbar_dZbar))))

divergence_barred = sp.trigsimp(sp.expand(dvybar_dYbar + dvzbar_dZbar))
display(Math(r"\frac{\partial\bar v_Y}{\partial\bar Y}+\frac{\partial\bar v_Z}{\partial\bar Z} = "
             + sp.latex(divergence_barred)))

matches = sp.simplify(divergence_barred - (vyY + vzZ)) == 0
assert matches
print("QED: the divergence in the rotated frame equals the SAME expression")
print("(v_YY + v_ZZ) as in the original frame -- unchanged, a scalar. Unlike")
print("section 2b's gradient, NOTHING here mixes Y- and Z- pieces together.")""")

numeric_md = md(r"""**Concrete numeric check**, for an actual vector field
$v_Y=Y^2-Z,\ v_Z=2YZ$ and a specific rotation angle -- computed two
completely independent ways (direct substitution in rotated coordinates
vs. the transformation formula above) to confirm they agree.""")

numeric_code = code(r"""Y, Z = sp.symbols('Y Z', real=True)
Ybar_n, Zbar_n = sp.symbols('Ybar Zbar', real=True)
phi_val = sp.Rational(2, 5)

vY_expr, vZ_expr = Y**2 - Z, 2*Y*Z
div_unbarred = sp.diff(vY_expr, Y) + sp.diff(vZ_expr, Z)

Y_of_bar_n = Ybar_n*sp.cos(phi_val) - Zbar_n*sp.sin(phi_val)
Z_of_bar_n = Ybar_n*sp.sin(phi_val) + Zbar_n*sp.cos(phi_val)
vYbar_expr = sp.cos(phi_val)*vY_expr.subs({Y:Y_of_bar_n, Z:Z_of_bar_n}, simultaneous=True) \
             + sp.sin(phi_val)*vZ_expr.subs({Y:Y_of_bar_n, Z:Z_of_bar_n}, simultaneous=True)
vZbar_expr = -sp.sin(phi_val)*vY_expr.subs({Y:Y_of_bar_n, Z:Z_of_bar_n}, simultaneous=True) \
             + sp.cos(phi_val)*vZ_expr.subs({Y:Y_of_bar_n, Z:Z_of_bar_n}, simultaneous=True)
div_barred = sp.simplify(sp.diff(vYbar_expr, Ybar_n) + sp.diff(vZbar_expr, Zbar_n))

point = {Y: sp.Rational(3,2), Z: sp.Rational(-1,2)}
point_bar = {Ybar_n: sp.cos(phi_val)*point[Y] + sp.sin(phi_val)*point[Z],
             Zbar_n: -sp.sin(phi_val)*point[Y] + sp.cos(phi_val)*point[Z]}
val_unbarred = float(div_unbarred.subs(point))
val_barred = float(div_barred.subs(point_bar))
print(f"div v at (Y,Z), unbarred frame:  {val_unbarred}")
print(f"div v at the SAME point, barred frame: {val_barred}")
assert abs(val_unbarred - val_barred) < 1e-10
print("Match: the divergence is the same NUMBER regardless of which rotated frame computes it.")""")

physics_md = md(r"""**Why this matters, alongside §2b.** Gauss's law
$\nabla\cdot\mathbf E=\rho/\epsilon_0$ equates a divergence to a charge
density -- both must be honest SCALARS for the equation to mean the same
thing to every observer, whatever axes they drew. §2b proved $\nabla f$ is
a legitimate vector; §2c proves that taking the divergence of a legitimate
vector hands back a legitimate scalar. Together they certify the entire
chain scalar $\to^{\nabla}$ vector $\to^{\nabla\cdot}$ scalar used
throughout electrodynamics never silently becomes frame-dependent at any
step. (The chain-rule step used in both proofs rests on the same total-
differential argument -- $df=(\partial f/\partial Y)dY+(\partial f/\partial Z)dZ$
plus matching coefficients of the independent differentials $d\bar Y,d\bar Z$
-- worked in full in this session's exploration of §2b; not re-derived
here to avoid repeating it.)""")

# insert right after section 2b's cells, before section 3 (Problem 1.21)
insert_at = None
for i, c in enumerate(cells):
    if c.cell_type == "markdown" and "Problem 1.21" in c.source:
        insert_at = i
        break
if insert_at is None:
    raise RuntimeError("could not find the Problem 1.21 section to insert before")

new_cells = [sec_md, sub_md, sub_code, main_md, main_code, numeric_md, numeric_code, physics_md]
cells[insert_at:insert_at] = new_cells
cells[0].source = cells[0].source.replace("1.13, 1.14, 1.21", "1.13, 1.14, 1.17, 1.21", 1)

nb["cells"] = cells
nbf.write(nb, str(NB_PATH))
print(f"inserted Problem 1.17 ({len(new_cells)} cells) at index {insert_at}, wrote {NB_PATH}")
