"""Extend notebooks/griffiths_notation_glossary_sympy.ipynb (covers r, del,
div/curl, QM notation) with:
  - source point r', field point r, and the separation vector script-R = r-r'
    (Griffiths' notation for Coulomb's law / the potential integral), plus
    equipotential surfaces
  - Lagrangian mechanics: generalized coordinates, degrees of freedom, and
    the Euler-Lagrange equation derived symbolically for a pendulum (1 DOF)
Extends the existing notebook rather than duplicating it. NOTE: no
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

sep_md = md(r"""## Source point, field point, and the separation vector $\mathscr{r}$

Griffiths' most-used piece of notation, introduced in Ch. 1 and used in
every field/potential integral from Ch. 2 on: two DIFFERENT position
vectors, not one.

- $\vec r$ -- the **field point**: where you're computing the field/potential.
- $\vec r\,'$ -- the **source point**: where the charge/current actually sits.
- $\mathscr{r} \equiv \vec r - \vec r\,'$ -- the **separation vector**, pointing
  FROM the source TO the field point. Every Coulomb's-law-type formula is
  written in terms of $\mathscr{r}$ and $\hat{\mathscr{r}}=\mathscr{r}/|\mathscr{r}|$,
  not the raw $\vec r$, because the source isn't always at the origin.

$$\vec E(\vec r) = \frac{1}{4\pi\epsilon_0}\sum_i q_i \frac{\hat{\mathscr{r}}_i}{\mathscr{r}_i^2},
\qquad \mathscr{r}_i = \vec r - \vec r_i\,'$$""")

sep_code = code(r"""x, y, z, xp, yp, zp = sp.symbols("x y z x' y' z'", real=True)
eps0, q = sp.symbols('epsilon_0 q', positive=True)

r_field = sp.Matrix([x, y, z])          # field point (where E is measured)
r_source = sp.Matrix([xp, yp, zp])      # source point (where q sits)

script_r = r_field - r_source           # THE separation vector, Griffiths' script-r
script_r_mag = sp.sqrt(sum(c**2 for c in script_r))
script_r_hat = script_r / script_r_mag

print("field point r =", r_field.T)
print("source point r' =", r_source.T)
print("\nseparation vector script_r = r - r' =")
sp.pprint(script_r.T)
print("\n|script_r| =")
sp.pprint(script_r_mag)

E_point_charge = q/(4*sp.pi*eps0) * script_r_hat / script_r_mag**2
print("\nE(r) from a point charge q sitting at the SOURCE point r' (not the origin):")
sp.pprint(sp.simplify(E_point_charge).T)

# sanity check: if the source sits at the origin (r'=0), this collapses to
# the textbook single-charge formula in terms of r alone
E_at_origin_source = E_point_charge.subs({xp: 0, yp: 0, zp: 0})
E_textbook = q/(4*sp.pi*eps0) * r_field/sp.sqrt(sum(c**2 for c in r_field))**3
assert sp.simplify(E_at_origin_source - E_textbook) == sp.zeros(3, 1)
print("\nVerified: setting r'=0 recovers the textbook single-charge-at-origin formula exactly.")""")

equipot_md = md(r"""## Equipotential surfaces

$V(\vec r) = \text{const}$ defines a surface (an "equipotential"). For a
point charge at the source point $\vec r\,'$, the equipotentials are spheres
centered on $\vec r\,'$ -- not the origin, unless the charge happens to sit
there.""")

equipot_code = code(r"""V = q/(4*sp.pi*eps0*script_r_mag)
V_const = sp.symbols('V_0', positive=True)

# solve V(r) = V_0 for the surface equation
surface_eq = sp.Eq(V, V_const)
solved_r_mag = sp.solve(surface_eq, script_r_mag)[0]
print("Solving V(r) = V_0 for |script_r| gives:")
sp.pprint(solved_r_mag)

surface_eq_explicit = sp.Eq(script_r_mag**2, solved_r_mag**2)
surface_expanded = sp.expand((x-xp)**2 + (y-yp)**2 + (z-zp)**2 - solved_r_mag**2)
print("\n(x-x')^2 + (y-y')^2 + (z-z')^2 - R^2 = 0, expanded:")
sp.pprint(surface_expanded)
print("\n--> this is exactly the equation of a SPHERE of radius R = q/(4*pi*eps0*V_0)")
print("    centered at the SOURCE point (x',y',z') -- not the origin. Equipotentials")
print("    follow the source, because V only depends on script_r = r - r'.")""")

# ── Lagrangian mechanics: degrees of freedom ───────────────────────────
lagrangian_md = md(r"""## Lagrangian mechanics: degrees of freedom, made concrete

**Degrees of freedom (DOF)** = the number of independent generalized
coordinates $q_i$ needed to fully specify the system's configuration. A free
particle in 3D has 3 DOF ($x,y,z$); a rigid pendulum swinging in a plane has
just **1 DOF** (the angle $\theta$) even though the bob's Cartesian position
has 2 coordinates $(x,y)$ -- the constraint (fixed rod length $L$) removes one.

The Euler-Lagrange equation, one per DOF:
$$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot q_i}\right) - \frac{\partial L}{\partial q_i} = 0,
\qquad L = T - V$$
derived here symbolically for the pendulum, to show the 1-DOF reduction
happening explicitly rather than asserted.""")

lagrangian_code = code(r"""t = sp.symbols('t', real=True)
theta = sp.Function('theta')(t)   # the ONE generalized coordinate (1 DOF)
L_len, m, g = sp.symbols('L m g', positive=True)

# Cartesian position of the bob IN TERMS OF the single DOF theta --
# this is where "3 coordinates, 1 DOF" gets enforced: x,y both depend on
# theta alone, not on two independent coordinates.
x_bob = L_len * sp.sin(theta)
y_bob = -L_len * sp.cos(theta)

vx = sp.diff(x_bob, t)
vy = sp.diff(y_bob, t)
speed_sq = sp.simplify(vx**2 + vy**2)
print("bob speed^2 in terms of the single DOF theta(t):")
sp.pprint(speed_sq)

T = sp.Rational(1, 2) * m * speed_sq
V = m * g * y_bob
Lagrangian = sp.simplify(T - V)
print("\nL = T - V =")
sp.pprint(Lagrangian)

theta_dot = sp.diff(theta, t)
dL_dthetadot = sp.diff(Lagrangian, theta_dot)
d_dt_dL_dthetadot = sp.diff(dL_dthetadot, t)
dL_dtheta = sp.diff(Lagrangian, theta)

euler_lagrange = sp.simplify(d_dt_dL_dthetadot - dL_dtheta)
print("\nEuler-Lagrange: d/dt(dL/d theta_dot) - dL/d theta =")
sp.pprint(euler_lagrange)

theta_ddot = sp.diff(theta, t, 2)
eom = sp.solve(sp.Eq(euler_lagrange, 0), theta_ddot)[0]
print("\nSolved for the equation of motion, theta_ddot =")
sp.pprint(sp.simplify(eom))

expected = -g/L_len * sp.sin(theta)
assert sp.simplify(eom - expected) == 0
print("\nVerified: matches the textbook pendulum equation theta'' = -(g/L)*sin(theta)")
print("EXACTLY -- derived from L=T-V and the Euler-Lagrange equation, not assumed.")
print("\n--> 1 DOF (theta) produced 1 Euler-Lagrange equation -- that correspondence")
print("    (n DOF => n coupled 2nd-order ODEs) IS what 'degrees of freedom' means")
print("    operationally, not just a count of numbers.")""")

# insert both new sections right before the final "Glossary" cell
def find_index(snippet):
    for i, c in enumerate(cells):
        if c.cell_type == "markdown" and snippet in c.source:
            return i
    raise ValueError(f"could not find markdown containing: {snippet!r}")

idx_glossary = find_index("## Glossary")

cells.insert(idx_glossary, lagrangian_code)
cells.insert(idx_glossary, lagrangian_md)
cells.insert(idx_glossary, equipot_code)
cells.insert(idx_glossary, equipot_md)
cells.insert(idx_glossary, sep_code)
cells.insert(idx_glossary, sep_md)

# append new glossary rows
glossary_cell = cells[find_index("## Glossary")]
glossary_cell.source += (
    "\n| $\\vec r\\,'$ | source point: where the charge/current sits |"
    "\n| $\\mathscr{r}=\\vec r-\\vec r\\,'$ | separation vector: source to field point |"
    "\n| equipotential | surface where $V=$const; spheres around a point source |"
    "\n| DOF | number of independent generalized coordinates $q_i$ |"
    "\n| $L=T-V$ | Lagrangian; Euler-Lagrange gives 1 ODE per DOF |"
)

nb.cells = cells
nbf.write(nb, str(NB_PATH))
print(f"wrote {NB_PATH} with {len(cells)} cells")
