"""Upgrades notebooks/griffiths_problem_1p16.ipynb's SymPy cells from plain
print() (ASCII) to real LaTeX rendering via sp.init_printing(use_latex=
'mathjax') + display(Math(...)) -- the notebook's content/structure is
already solid (product rule, chain rule, flux, Dirac delta resolution,
Gauss's law), this only changes HOW the SymPy results are shown. NOTE: no
triple-double-quote docstrings inside cell strings.
"""
import pathlib
import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "griffiths_problem_1p16.ipynb"

nb = nbf.read(str(NB_PATH), as_version=4)
cells = nb.cells

# cell 4: v in Cartesian -- add init_printing, switch to display(Math(...))
cells[4].source = r"""import sympy as sp
from IPython.display import display, Math
sp.init_printing(use_latex='mathjax')

x, y, z = sp.symbols("x y z", real=True)
r = sp.sqrt(x**2 + y**2 + z**2)
vx, vy, vz = x/r**3, y/r**3, z/r**3
display(Math(r"v_x = " + sp.latex(vx)))
display(Math(r"v_y = " + sp.latex(vy)))
display(Math(r"v_z = " + sp.latex(vz)))"""

# cell 6: the three partial derivatives
cells[6].source = r"""dvx = sp.simplify(sp.diff(vx, x))
dvy = sp.simplify(sp.diff(vy, y))
dvz = sp.simplify(sp.diff(vz, z))
display(Math(r"\frac{\partial v_x}{\partial x} = " + sp.latex(dvx)
             + r"\quad\left(=\frac{1}{r^3}-\frac{3x^2}{r^5}\right)"))
display(Math(r"\frac{\partial v_y}{\partial y} = " + sp.latex(dvy)))
display(Math(r"\frac{\partial v_z}{\partial z} = " + sp.latex(dvz)))"""

# cell 8: the divergence sum
cells[8].source = r"""div = sp.simplify(sp.diff(vx, x) + sp.diff(vy, y) + sp.diff(vz, z))
display(Math(r"\nabla\cdot\mathbf v = " + sp.latex(div)
             + r"\quad\text{(zero everywhere EXCEPT the origin, where the formula blows up)}"))"""

# cell 12: the delta-strength conclusion -- add a LaTeX-rendered boxed result
cells[12].source = r"""print("integral of div v over any volume containing 0  =  flux  =  4 pi  =", round(4*np.pi, 5))
display(Math(r"\boxed{\ \nabla\cdot\left(\frac{\hat r}{r^2}\right) = 4\pi\,\delta^3(\mathbf r)\ }"
             + r"\quad\text{: zero off-origin, a } 4\pi \text{ spike at } r=0"))"""

nb["cells"] = cells
nbf.write(nb, str(NB_PATH))
print(f"upgraded cells 4, 6, 8, 12 to LaTeX rendering, wrote {NB_PATH}")
