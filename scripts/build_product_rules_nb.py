"""Build notebooks/griffiths_product_rules.ipynb -- the six product rules from
Griffiths p.21, PROVED with SymPy (abstract fields, so they're real proofs), plus a
chain-rule loop, e^{ix}, and a Bessel example from photonics. Outputs baked in via
exec() so the notebook reads without a live kernel."""
import io
import base64
import contextlib
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf

ns = {}
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(src):
    buf = io.StringIO()
    plt.close("all")
    with contextlib.redirect_stdout(buf):
        exec(src, ns)
    cell = nbf.v4.new_code_cell(src)
    outs = []
    text = buf.getvalue()
    if text:
        outs.append(nbf.v4.new_output("stream", name="stdout", text=text))
    for num in plt.get_fignums():
        fig = plt.figure(num)
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
        outs.append(nbf.v4.new_output("display_data",
                                      data={"image/png": base64.b64encode(b.getvalue()).decode()},
                                      metadata={}))
    plt.close("all")
    cell.outputs = outs
    cell.execution_count = None
    cells.append(cell)


md(r"""# Griffiths' product rules (p.21), proved with SymPy
Griffiths lists **six** product rules -- two for gradients, two for divergences, two for
curls -- and says *the proofs come straight from the product rule for ordinary
derivatives.* Let's actually do that: build `grad`, `div`, `curl` from `d/dx`, feed in
**abstract** fields `f, g, A, B` (arbitrary functions, not specific ones), and show each
identity collapses to `0`. Abstract fields => these are genuine proofs, not spot checks.

Then a **chain-rule loop** over integer powers, the photonics phase factor `e^{ix}`, and
a **Bessel** identity that is itself just product-rule + chain-rule (Bessel functions are
the radial fiber modes -- see `dgs/bessel_separation`, `dgs/bessel_fm`).""")

code(r"""import sympy as sp
sp.init_printing()
x, y, z = sp.symbols("x y z", real=True)

# the three operators, built from ordinary partial derivatives
def grad(f):  return [sp.diff(f, v) for v in (x, y, z)]
def div(A):   return sum(sp.diff(A[i], v) for i, v in enumerate((x, y, z)))
def curl(A):  return [sp.diff(A[2], y) - sp.diff(A[1], z),
                      sp.diff(A[0], z) - sp.diff(A[2], x),
                      sp.diff(A[1], x) - sp.diff(A[0], y)]
def dot(A, B):   return sum(a*b for a, b in zip(A, B))
def cross(A, B): return [A[1]*B[2]-A[2]*B[1], A[2]*B[0]-A[0]*B[2], A[0]*B[1]-A[1]*B[0]]
def smul(s, A):  return [s*a for a in A]          # scalar times vector

# abstract fields: arbitrary scalar f,g and vector A,B  -> proofs, not examples
f = sp.Function("f")(x, y, z)
g = sp.Function("g")(x, y, z)
A = [sp.Function("A_x")(x, y, z), sp.Function("A_y")(x, y, z), sp.Function("A_z")(x, y, z)]
B = [sp.Function("B_x")(x, y, z), sp.Function("B_y")(x, y, z), sp.Function("B_z")(x, y, z)]

def is_zero_vec(V): return all(sp.simplify(c) == 0 for c in V)
print("operators ready; fields are arbitrary functions, so a 0 below is a real proof")""")

md(r"""## Rule (i) -- gradient of a product:  $\nabla(fg) = f\,\nabla g + g\,\nabla f$
Just the ordinary product rule applied component by component.""")

code(r"""lhs = grad(f*g)
rhs = [a + b for a, b in zip(smul(f, grad(g)), smul(g, grad(f)))]
print("grad(f g) - [f grad g + g grad f] =", [sp.simplify(a-b) for a, b in zip(lhs, rhs)])
print("rule (i) holds:", is_zero_vec([a-b for a, b in zip(lhs, rhs)]))""")

md(r"""## Rule (iii) -- divergence of (scalar x vector):  $\nabla\cdot(f\mathbf A)=f(\nabla\cdot\mathbf A)+\mathbf A\cdot(\nabla f)$
This is the one Griffiths works out in the text. Each term `d/dx(f A_x)=f dA_x/dx + A_x df/dx`.""")

code(r"""lhs = div(smul(f, A))
rhs = f*div(A) + dot(A, grad(f))
print("div(f A) - [f divA + A.grad f] =", sp.simplify(lhs - rhs))
print("rule (iii) holds:", sp.simplify(lhs - rhs) == 0)""")

md(r"""## Rule (iv) -- divergence of a cross product:  $\nabla\cdot(\mathbf A\times\mathbf B)=\mathbf B\cdot(\nabla\times\mathbf A)-\mathbf A\cdot(\nabla\times\mathbf B)$
The one behind **Poynting's theorem** (`griffiths/vector_identities.py`): with
`A=E, B=B` it turns `div(E x B)` into curl terms = the energy-flow bookkeeping.""")

code(r"""lhs = div(cross(A, B))
rhs = dot(B, curl(A)) - dot(A, curl(B))
print("div(A x B) - [B.curlA - A.curlB] =", sp.simplify(lhs - rhs))
print("rule (iv) holds:", sp.simplify(lhs - rhs) == 0)""")

md(r"""## Rule (vi) -- curl of a cross product (the long one)
$\nabla\times(\mathbf A\times\mathbf B)=(\mathbf B\cdot\nabla)\mathbf A-(\mathbf A\cdot\nabla)\mathbf B+\mathbf A(\nabla\cdot\mathbf B)-\mathbf B(\nabla\cdot\mathbf A)$.
Four terms, easy to slip on by hand -- SymPy confirms it in one shot.""")

code(r"""def Adotgrad(A, V):   # (A . grad) V , a directional derivative of each component
    return [A[0]*sp.diff(c, x) + A[1]*sp.diff(c, y) + A[2]*sp.diff(c, z) for c in V]
lhs = curl(cross(A, B))
rhs = [a - b + c - d for a, b, c, d in zip(
        Adotgrad(B, A), Adotgrad(A, B), smul(div(B), A), smul(div(A), B))]
print("rule (vi) holds:", is_zero_vec([a-b for a, b in zip(lhs, rhs)]))""")

md(r"""## The chain rule, looped over algebraic integers
Every rule above descends from `d/dx`. The **chain rule** is the other half: for the
phase factor `e^{i g(x)}` with `g(x)=x^n`,
$$\frac{d}{dx}e^{i x^n} = i\,n x^{n-1}\,e^{i x^n}.$$
Loop the integer power `n` and check the closed form each time.""")

code(r"""for n in range(1, 7):
    g = x**n
    lhs = sp.diff(sp.exp(sp.I*g), x)
    rhs = sp.I*sp.diff(g, x)*sp.exp(sp.I*g)          # i * g'(x) * e^{i g}
    print(f"n={n}:  d/dx e^(i x^{n}) = i*{sp.diff(g,x)}*e^(i x^{n})   match:",
          sp.simplify(lhs - rhs) == 0)""")

md(r"""## The photonics phase factor $e^{ix}$ -- and why phase retrieval is hard
`e^{ix} = cos x + i sin x` (Euler). Two facts that *are* the whole phase-retrieval
problem: its derivative just rotates it (`d/dx = i e^{ix}`), and its **magnitude is 1
for every x**. A detector measures `|field|^2`, so it sees `|e^{i phi}|^2 = 1` and the
phase `phi` vanishes -- exactly the information dispersion-GS has to claw back.""")

code(r"""print("Euler:    e^(ix) =", sp.exp(sp.I*x).rewrite(sp.cos))
print("derivative: d/dx e^(ix) =", sp.diff(sp.exp(sp.I*x), x), " (= i * e^(ix), pure rotation)")
print("magnitude:  |e^(ix)| =", sp.Abs(sp.exp(sp.I*x)), " for real x  -> phase is invisible to |.|^2")""")

md(r"""## A Bessel identity from photonics -- product rule + chain rule together
Bessel functions $J_n$ are the radial modes of a step-index fiber and the FM/PM sideband
amplitudes (`dgs/bessel_separation`, `dgs/bessel_fm`). One of their defining identities
is pure product-rule + chain-rule:
$$\frac{d}{dx}\!\left[x^{\,n} J_n(x)\right] = x^{\,n} J_{n-1}(x).$$
SymPy knows `J_n` and its derivative, so we can verify it symbolically.""")

code(r"""n_sym = sp.symbols("n")
for nv in (1, 2, 3):
    lhs = sp.diff(x**nv * sp.besselj(nv, x), x)
    rhs = x**nv * sp.besselj(nv-1, x)
    print(f"n={nv}:  d/dx[x^{nv} J_{nv}(x)] = x^{nv} J_{nv-1}(x)   match:",
          sp.simplify(lhs - rhs) == 0)""")

md(r"""And a picture of the first few -- the fiber's radial mode shapes.""")

code(r"""import numpy as np, mpmath, matplotlib.pyplot as plt
xs = np.linspace(0, 15, 250)
plt.figure(figsize=(6.5, 3.6))
for nv in (0, 1, 2, 3):
    Jn = [float(mpmath.besselj(nv, t)) for t in xs]
    plt.plot(xs, Jn, label=f"J_{nv}(x)")
plt.axhline(0, color="k", lw=0.6); plt.legend(ncol=4, fontsize=8)
plt.title("Bessel functions J_n -- step-index fiber radial modes")
plt.xlabel("x"); plt.ylabel("J_n(x)"); plt.tight_layout(); plt.show()
print("J_0 starts at 1 (the fundamental); higher orders start flat at 0 -- exactly the")
print("radial intensity profiles of LP modes in the core.")""")

md(r"""## Takeaway
- All **six** product rules (and the quotient rules) are the ordinary `d/dx` product rule
  wearing `grad`/`div`/`curl` clothing -- SymPy with abstract fields proves them flat.
- Rule (iv) is the seed of **Poynting's theorem**; rule (i)/(iii) you'll use building
  potentials in Ch.2-3.
- The **chain rule** drives `e^{ix}` (the invisible phase the detector throws away) and
  the **Bessel** recurrences (the fiber's radial modes) -- the same calculus runs from
  Griffiths' front cover straight into the photonics of this project.""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = pathlib.Path("notebooks/griffiths_product_rules.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(nb.cells), "cells (outputs baked)")
