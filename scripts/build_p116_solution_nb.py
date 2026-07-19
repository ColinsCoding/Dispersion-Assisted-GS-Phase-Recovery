"""Build notebooks/problem_1_16_solution.ipynb -- written solution + baked verification."""
import pathlib
import numpy as np
import nbformat as nbf

SOLUTION = r"""# Griffiths Problem 1.16 -- Solution

**Problem.** Sketch the vector function $\mathbf v = \dfrac{\hat r}{r^2}$ and compute its
divergence.

**Sketch.** $\mathbf v$ points radially *outward* everywhere, magnitude $1/r^2$ (long
arrows near the origin, short far away).

**Divergence for $r\ne 0$.** For a radial field $\mathbf v = v_r\hat r$,
$$\nabla\cdot\mathbf v=\frac{1}{r^2}\frac{\partial}{\partial r}\!\left(r^2 v_r\right).$$
With $v_r=1/r^2$, $r^2 v_r=1$ (constant), so
$$\nabla\cdot\mathbf v=\frac{1}{r^2}\frac{\partial}{\partial r}(1)=0.$$
(Cartesian check: $\mathbf v=\mathbf r/r^3$, $\partial_x(x/r^3)=1/r^3-3x^2/r^5$, and the three
terms sum to $3/r^3-3r^2/r^5=0$.) So $\nabla\cdot\mathbf v=0$ **except possibly at $r=0$**,
where the formula is undefined.

**The surprise, resolved.** The field fans out yet the divergence is zero. Apply the
divergence theorem on a sphere of radius $R$:
$$\oint\mathbf v\cdot d\mathbf a=\int_0^\pi\!\!\int_0^{2\pi}\frac{\hat r}{R^2}\cdot(\hat r R^2\sin\theta\,d\theta\,d\phi)
=\int_0^\pi\!\!\int_0^{2\pi}\sin\theta\,d\theta\,d\phi=4\pi,$$
**independent of $R$**. But $\int_V(\nabla\cdot\mathbf v)\,d\tau=\oint\mathbf v\cdot d\mathbf a=4\pi$;
if the divergence were zero everywhere this would be $0$ -- contradiction. So the divergence
is nonzero only at $r=0$.

**Answer.**
$$\boxed{\;\nabla\cdot\!\left(\frac{\hat r}{r^2}\right)=4\pi\,\delta^3(\mathbf r)\;}$$
zero everywhere, infinite at the origin, integrating to $4\pi$ over any volume enclosing it.

**Why it matters.** A point charge has $\mathbf E=\frac{q}{4\pi\varepsilon_0}\frac{\hat r}{r^2}$ and
$\nabla\cdot\mathbf E=\rho/\varepsilon_0$ with $\rho=q\,\delta^3(\mathbf r)$: the divergence of
$\mathbf E$ sits exactly on the charge -- Gauss's law. (Equivalently $\nabla^2(1/r)=-4\pi\delta^3(\mathbf r)$.)"""

CODE = '''import numpy as np
from griffiths import inverse_square as isq

print("div(r-hat/r^2) for r != 0 :", isq.divergence_inverse_square(), "  (the surprise: zero)")
for R in (0.5, 1.0, 5.0):
    print(f"  flux through sphere R={R}: {isq.flux_through_sphere(R):.5f}  (= 4 pi for ALL R)")
print(f"4 pi = {4*np.pi:.5f}  ->  div(r-hat/r^2) = 4 pi delta^3(r)")'''


def run_code():
    import sys, pathlib as pl
    sys.path.insert(0, str(pl.Path(__file__).resolve().parents[1]))
    from griffiths import inverse_square as isq
    lines = [f"div(r-hat/r^2) for r != 0 : {isq.divergence_inverse_square()}   (the surprise: zero)"]
    for R in (0.5, 1.0, 5.0):
        lines.append(f"  flux through sphere R={R}: {isq.flux_through_sphere(R):.5f}  (= 4 pi for ALL R)")
    lines.append(f"4 pi = {4*np.pi:.5f}  ->  div(r-hat/r^2) = 4 pi delta^3(r)")
    return "\n".join(lines) + "\n"


nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell(SOLUTION)
code = nbf.v4.new_code_cell(CODE)
code.outputs = [nbf.v4.new_output("stream", name="stdout", text=run_code())]
code.execution_count = 1
nb.cells = [md, code]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = pathlib.Path("notebooks/problem_1_16_solution.ipynb")
nbf.write(nb, out)
print("wrote", out, "with baked-in verification output")
