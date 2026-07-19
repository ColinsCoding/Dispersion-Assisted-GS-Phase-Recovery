"""Extends notebooks/griffiths_ch1_solutions.ipynb's existing Sec.3
(Problem 1.21) with the electrodynamics payoff of product rule (iv): with
A=E, B=H, div(A x B)=B.(curl A)-A.(curl B) becomes, after substituting
Faraday's and Ampere-Maxwell's laws, POYNTING'S THEOREM (energy
conservation in the EM field) -- the actual reason this "abstract" vector
identity is worth memorizing. Verified both by direct symbolic
substitution and by a concrete vacuum plane-wave example (matches
dgs.curl_div_modern_physics's plane-wave Faraday-law check in spirit, but
carries the derivation one step further, all the way to energy
conservation). NOTE: no triple-double-quote docstrings inside cell
strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "griffiths_ch1_solutions.ipynb"

nb = nbf.read(str(NB_PATH), as_version=4)
cells = nb.cells
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

sec_md = md(r"""### §3b Electrodynamics payoff: rule (iv) IS Poynting's theorem

Rule (iv), with $\mathbf A=\mathbf E$, $\mathbf B=\mathbf H$:
$$\nabla\cdot(\mathbf E\times\mathbf H) = \mathbf H\cdot(\nabla\times\mathbf E) - \mathbf E\cdot(\nabla\times\mathbf H)$$
is pure vector-identity bookkeeping -- but substitute Maxwell's curl
equations (Faraday: $\nabla\times\mathbf E=-\partial\mathbf B/\partial t$;
Ampere-Maxwell: $\nabla\times\mathbf H=\mathbf J_f+\partial\mathbf D/\partial t$)
and it becomes physics:
$$\nabla\cdot(\mathbf E\times\mathbf H) = -\mathbf H\cdot\frac{\partial\mathbf B}{\partial t}
-\mathbf E\cdot\frac{\partial\mathbf D}{\partial t} - \mathbf E\cdot\mathbf J_f$$
For a linear medium ($\mathbf D=\epsilon\mathbf E$, $\mathbf B=\mu\mathbf H$), each
time-derivative term is itself a perfect derivative of an energy density
($\mathbf H\cdot\partial\mathbf B/\partial t=\partial(\frac12\mu H^2)/\partial t$,
similarly for $E$), so rearranging gives **Poynting's theorem**:
$$\frac{\partial u}{\partial t} + \nabla\cdot\mathbf S = -\mathbf E\cdot\mathbf J_f,
\qquad u=\tfrac12\epsilon E^2+\tfrac12\mu H^2,\quad \mathbf S=\mathbf E\times\mathbf H$$
Energy density change, plus energy flowing out through the Poynting
vector $\mathbf S$, equals minus the work done on free charges -- energy
conservation, derived entirely from one "abstract" product rule plus two
of Maxwell's equations.""")

sub_code = code(r"""# concrete check: a vacuum plane wave (E along x, H along y, propagating along z)
z, t, k, w, E0, eps0, mu0 = sp.symbols('z t k omega E0 epsilon0 mu0', real=True, positive=True)
c_speed = 1/sp.sqrt(eps0*mu0)
w_expr = k*c_speed   # vacuum dispersion relation omega = c*k

Ex = E0*sp.cos(k*z - w_expr*t)
eta0 = sp.sqrt(mu0/eps0)              # vacuum wave impedance
Hy = (E0/eta0)*sp.cos(k*z - w_expr*t)  # matched so Faraday's law is satisfied exactly

display(Math(r"E_x = " + sp.latex(Ex) + r",\qquad H_y = " + sp.latex(Hy)))

# Poynting vector S = E x H = (0, 0, Ex*Hy) for E=(Ex,0,0), H=(0,Hy,0)
Sz = Ex*Hy
div_S = sp.diff(Sz, z)   # div(S) for S depending only on z, pointing along z

# energy density u = (1/2)eps0 E^2 + (1/2)mu0 H^2
u = sp.Rational(1,2)*eps0*Ex**2 + sp.Rational(1,2)*mu0*Hy**2
du_dt = sp.diff(u, t)

residual = sp.simplify(du_dt + div_S)
display(Math(r"\frac{\partial u}{\partial t} + \nabla\cdot\mathbf S = " + sp.latex(residual)
             + r"\quad\text{(vacuum, no free current: RHS should be 0)}"))
assert residual == 0
print("Confirmed: Poynting's theorem holds EXACTLY for this plane wave, with zero free")
print("current on the right-hand side -- all the energy that leaves one region via S")
print("is accounted for by the local energy density's own decrease, nothing lost.")""")

physics_md = md(r"""**Why this matters alongside the abstract proof above.**
The component-by-component proof of rule (iv) shown earlier in this
section is pure algebra -- it would be true for ANY two vector fields,
electromagnetic or not. What turns it into Poynting's theorem is
substituting two SPECIFIC physical laws (Faraday, Ampere-Maxwell) for the
curls. This is the general pattern behind most "useful" vector identities
in electrodynamics: the identity itself is coordinate-free algebra: the
physics arrives when Maxwell's equations tell you what the curls
*actually are*.""")

# insert right after section 3's existing content, before section 4 (Problem 1.22)
insert_at = None
for i, c in enumerate(cells):
    if c.cell_type == "markdown" and "Problem 1.22" in c.source:
        insert_at = i
        break
if insert_at is None:
    raise RuntimeError("could not find the Problem 1.22 section to insert before")

new_cells = [sec_md, sub_code, physics_md]
cells[insert_at:insert_at] = new_cells
nb["cells"] = cells
nbf.write(nb, str(NB_PATH))
print(f"inserted Poynting's-theorem extension ({len(new_cells)} cells) at index {insert_at}, wrote {NB_PATH}")
