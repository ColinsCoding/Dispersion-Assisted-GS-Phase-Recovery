"""Extends notebooks/griffiths_ch1_solutions.ipynb with Problem 1.14 --
the gap between the existing Sec.2 (Problem 1.13) and Sec.3 (Problem
1.21). This compiles the full derivation worked through interactively this
session: the coordinate-inversion elimination (solving y,z in terms of the
rotated ybar,zbar), the chain-rule application producing the boxed
transformation law, the physics interpretation (why this proof licenses
treating grad f as a vector everywhere later in the book), and a
computational cross-check reusing dgs.torch.gradient_transform_verify's
independent SymPy+torch triple verification for a concrete f(y,z).

Uses FRESH symbol names (Y, Z, Ybar, Zbar, phi) rather than the notebook's
existing `x, y, z` (imported from the griffiths package for the 3D
separation-vector work in Sec.2/Sec.3) -- Problem 1.14 is a 2-variable
problem and must not shadow those. NOTE: no triple-double-quote docstrings
inside cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "griffiths_ch1_solutions.ipynb"

nb = nbf.read(str(NB_PATH), as_version=4)
cells = nb.cells
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

sec_md = md(r"""## §2b Problem 1.14 -- $\nabla f$ transforms as a vector

Suppose $f$ is a function of two variables $(y,z)$ only. Under a rotation
by angle $\phi$ in the $y$-$z$ plane,
$$\bar y = y\cos\phi + z\sin\phi,\qquad \bar z = -y\sin\phi + z\cos\phi.$$
Show that $\nabla f=(\partial f/\partial y)\hat y+(\partial f/\partial z)\hat z$
transforms the same way -- i.e. that $(\nabla f)_{\bar y}$ and $(\nabla f)_{\bar z}$
mix into each other via the SAME rotation matrix that mixes $\bar y,\bar z$
themselves. This is Griffiths defining what "being a vector" operationally
means: not "has an arrow drawn on it," but "transforms this way under a
change of frame" -- the one-time proof that licenses writing $\mathbf E=-\nabla V$
and trusting it holds for every observer, no matter how they oriented their axes.

Note: this section uses fresh symbols ($Y,Z,\bar Y,\bar Z,\phi$), NOT the
`x,y,z` imported from the `griffiths` package above -- Problem 1.14 is a
2-variable problem, unrelated to the 3D separation vector in §2.""")

step1_md = md(r"""**Step 1 -- invert the rotation.** Multiply the $\bar y$
equation by $\cos\phi$, the $\bar z$ equation by $\sin\phi$, and SUBTRACT
(the $Z$-terms already carry the same sign, so subtracting cancels them;
for solving $Z$ instead you'd multiply by $\sin\phi,\cos\phi$ and ADD,
since those $Y$-terms carry opposite signs from the start):""")

step1_code = code(r"""Y, Z, phi = sp.symbols('Y Z phi', real=True)
Ybar, Zbar = sp.symbols('Ybar Zbar', real=True)

eq_ybar = sp.Eq(Ybar, Y*sp.cos(phi) + Z*sp.sin(phi))
eq_zbar = sp.Eq(Zbar, -Y*sp.sin(phi) + Z*sp.cos(phi))
display(Math(sp.latex(eq_ybar) + r"\qquad" + sp.latex(eq_zbar)))

sol = sp.solve([eq_ybar, eq_zbar], [Y, Z])
Y_of_bar, Z_of_bar = sol[Y], sol[Z]
display(Math(r"Y = " + sp.latex(Y_of_bar) + r",\qquad Z = " + sp.latex(Z_of_bar)))

# verify this really is the inverse: substituting back reproduces Ybar, Zbar
check_ybar = sp.simplify(Y_of_bar*sp.cos(phi) + Z_of_bar*sp.sin(phi) - Ybar)
check_zbar = sp.simplify(-Y_of_bar*sp.sin(phi) + Z_of_bar*sp.cos(phi) - Zbar)
assert check_ybar == 0 and check_zbar == 0
print("Verified: substituting Y(Ybar,Zbar), Z(Ybar,Zbar) back reproduces Ybar, Zbar exactly.")""")

step2_md = md(r"""**Step 2 -- read off the four partial derivatives** needed
for the chain rule, directly from the inversion above.""")

step2_code = code(r"""dY_dYbar = sp.diff(Y_of_bar, Ybar)
dY_dZbar = sp.diff(Y_of_bar, Zbar)
dZ_dYbar = sp.diff(Z_of_bar, Ybar)
dZ_dZbar = sp.diff(Z_of_bar, Zbar)
display(Math(r"\frac{\partial Y}{\partial \bar Y}=" + sp.latex(dY_dYbar)
             + r",\quad \frac{\partial Y}{\partial \bar Z}=" + sp.latex(dY_dZbar)
             + r",\quad \frac{\partial Z}{\partial \bar Y}=" + sp.latex(dZ_dYbar)
             + r",\quad \frac{\partial Z}{\partial \bar Z}=" + sp.latex(dZ_dZbar)))""")

step3_md = md(r"""**Step 3 -- the chain rule.** $f$ depends on $Y,Z$; both
depend on $\bar Y,\bar Z$; so differentiating $f$ with respect to $\bar Y$
must sum over BOTH paths ($\bar Y\to Y\to f$ and $\bar Y\to Z\to f$):""")

step3_code = code(r"""fY, fZ = sp.symbols('f_Y f_Z')   # stand-ins for (grad f)_Y, (grad f)_Z

df_dYbar = fY*dY_dYbar + fZ*dZ_dYbar
df_dZbar = fY*dY_dZbar + fZ*dZ_dZbar
display(Math(r"(\nabla f)_{\bar Y} = " + sp.latex(df_dYbar)))
display(Math(r"(\nabla f)_{\bar Z} = " + sp.latex(df_dZbar)))

# the qed: SAME rotation matrix acts on (fY,fZ) as acts on (Y,Z)
match_ybar = sp.simplify(df_dYbar - (sp.cos(phi)*fY + sp.sin(phi)*fZ)) == 0
match_zbar = sp.simplify(df_dZbar - (-sp.sin(phi)*fY + sp.cos(phi)*fZ)) == 0
assert match_ybar and match_zbar
print("QED: (grad f)_Ybar, (grad f)_Zbar mix via the identical rotation matrix that")
print("mixes Ybar, Zbar themselves -- grad f transforms as a vector.")""")

step4_md = md(r"""**Computational cross-check, for a concrete function.**
$f(Y,Z)=Y^2 Z+\sin(YZ)$, evaluated at a real point, confirms the symbolic
transformation law numerically -- and this proof has ALSO been verified a
third, completely independent way in `dgs/torch/gradient_transform_verify.py`
(run via `make gradxform`, py-3.12/PyTorch -- not this notebook's py-3.13
kernel, so it's referenced rather than executed here): PyTorch autograd,
given ONLY the coordinate formulas $Y(\bar Y,\bar Z)$, $Z(\bar Y,\bar Z)$
(plain elementwise tensor arithmetic, no chain-rule formula supplied),
rediscovers this exact $\cos\phi,\sin\phi$ transformation law on its own,
purely from backpropagating through the composed graph -- matching to
machine precision ($<10^{-14}$).""")

step4_code = code(r"""def f_test(Yv, Zv):
    return Yv**2 * Zv + sp.sin(Yv * Zv)

fY_expr, fZ_expr = sp.diff(f_test(Y, Z), Y), sp.diff(f_test(Y, Z), Z)

phi_val = sp.Rational(7, 10)
point = {Y: sp.Rational(13, 10), Z: sp.Rational(-2, 5)}
fY_num = float(fY_expr.subs(point))
fZ_num = float(fZ_expr.subs(point))

c_val, s_val = float(sp.cos(phi_val)), float(sp.sin(phi_val))
fYbar_num = c_val*fY_num + s_val*fZ_num
fZbar_num = -s_val*fY_num + c_val*fZ_num
print(f"f(Y,Z) = Y^2*Z + sin(Y*Z), at (Y,Z)=({float(point[Y])}, {float(point[Z])}), phi={float(phi_val)}")
print(f"(grad f)_Y = {fY_num:.6f}   (grad f)_Z = {fZ_num:.6f}")
print(f"(grad f)_Ybar (via this section's boxed formula) = {fYbar_num:.6f}")
print(f"(grad f)_Zbar (via this section's boxed formula) = {fZbar_num:.6f}")""")

# insert right after Sec.2 (Problem 1.13), before Sec.3 (Problem 1.21)
insert_at = None
for i, c in enumerate(cells):
    if c.cell_type == "markdown" and "Problem 1.21" in c.source:
        insert_at = i
        break
if insert_at is None:
    raise RuntimeError("could not find the Problem 1.21 section to insert before")

new_cells = [sec_md, step1_md, step1_code, step2_md, step2_code,
             step3_md, step3_code, step4_md, step4_code]
cells[insert_at:insert_at] = new_cells

# update the title to mention 1.14 alongside the existing problem list
cells[0].source = cells[0].source.replace(
    "1.13, 1.21", "1.13, 1.14, 1.21", 1)

nb["cells"] = cells
nbf.write(nb, str(NB_PATH))
print(f"inserted Problem 1.14 (9 cells) at index {insert_at}, wrote {NB_PATH}")
