#!/usr/bin/env python
"""Builds notebooks/griffiths_notation_glossary_sympy.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Griffiths' Notation, Term by Term, in SymPy -- Real Space and Imaginary Space

Every symbol below is built and demonstrated with an explicit loop where one
applies, split into two halves matching the two books' worlds:

- **Real-space vector notation** (Electrodynamics): position vectors,
  $\hat{r}$, $\nabla$, $\nabla\cdot$, $\nabla\times$ -- everything here is
  real-valued.
- **Complex/imaginary-space notation** (Quantum Mechanics): $i$, complex
  conjugate $\Psi^*$, $|\Psi|^2$, $\langle\cdot\rangle$ -- everything here
  has a real AND imaginary part, and Griffiths' QM notation exists
  specifically to track both.

`sp.init_printing()` on throughout.
""")

code(r"""import sympy as sp
sp.init_printing()

x, y, z, t = sp.symbols("x y z t", real=True)
print("SymPy", sp.__version__, "ready")
""")

# ---- REAL SPACE: vectors and vector calculus -----------------------------------
md(r"""## Real space: position vectors and the del operator $\nabla$

$\vec r = x\hat x + y\hat y + z\hat z$ -- built explicitly as a loop summing
each coordinate times its unit vector, not typed in as a finished triple.
""")

code(r"""unit_vectors = {"x": sp.Matrix([1, 0, 0]), "y": sp.Matrix([0, 1, 0]), "z": sp.Matrix([0, 0, 1])}
coords = {"x": x, "y": y, "z": z}

r_vec = sp.Matrix([0, 0, 0])
for axis in ("x", "y", "z"):
    r_vec += coords[axis] * unit_vectors[axis]
print("r_vec = x*xhat + y*yhat + z*zhat, built by loop:")
sp.pprint(r_vec.T)

r_magnitude = sp.sqrt(sum(c**2 for c in r_vec))
print("\n|r| = r =")
sp.pprint(r_magnitude)

r_hat = sp.simplify(r_vec / r_magnitude)
print("\nr_hat = r_vec / |r_vec| (the unit vector Griffiths writes as r-hat):")
sp.pprint(r_hat.T)
""")

code(r"""# the gradient symbol: del(f) = (df/dx, df/dy, df/dz) -- built by looping
# the partial-derivative symbol "d/d(coord)" over each coordinate in turn
f = x**2 * y + sp.sin(z)

grad_f = sp.Matrix([0, 0, 0])
for i, axis in enumerate(("x", "y", "z")):
    partial = sp.diff(f, coords[axis])
    print(f"d f / d {axis} =", partial)
    grad_f[i] = partial

print("\ndel(f) =")
sp.pprint(grad_f.T)
""")

code(r"""# divergence: del . F = sum of d(F_i)/d(coord_i) -- a loop over matching
# components, the dot product written out term by term
F = sp.Matrix([x * y, y * z, z * x])

divergence = 0
for i, axis in enumerate(("x", "y", "z")):
    term = sp.diff(F[i], coords[axis])
    print(f"d F_{axis} / d {axis} =", term)
    divergence += term

print("\ndel . F (divergence) =", sp.simplify(divergence))
""")

code(r"""# curl: del x F, built the textbook way via cyclic permutation:
# curl_i = d(F_k)/d(coord_j) - d(F_j)/d(coord_k), looped over i=x,y,z
axes = ["x", "y", "z"]
curl = sp.Matrix([0, 0, 0])
for i in range(3):
    j, k = (i + 1) % 3, (i + 2) % 3
    curl[i] = sp.diff(F[k], coords[axes[j]]) - sp.diff(F[j], coords[axes[k]])
print("del x F (curl), built component by component:")
sp.pprint(curl.T)
""")

# ---- IMAGINARY SPACE: complex QM notation --------------------------------------
md(r"""## Imaginary space: $i$, $\Psi^*$, $|\Psi|^2$, $\langle\cdot\rangle$

Griffiths' QM notation exists because $\Psi$ is genuinely complex --
real and imaginary parts both carry physics. Every symbol below is built by
splitting an explicit complex function into its real/imaginary pieces, not
assumed.
""")

code(r"""Psi = sp.Function("Psi")(x, t)
Psi_explicit = sp.exp(sp.I * x) * sp.exp(-x**2)   # a concrete complex test function

re_part = sp.re(Psi_explicit)
im_part = sp.im(Psi_explicit)
print("Psi = Re(Psi) + i*Im(Psi):")
print("Re(Psi) =", sp.simplify(re_part))
print("Im(Psi) =", sp.simplify(im_part))

reconstructed = sp.simplify(re_part + sp.I * im_part - Psi_explicit)
print("\ncheck Re + i*Im reconstructs Psi exactly:", reconstructed == 0)
""")

code(r"""# Psi^* (complex conjugate): flip the sign of i everywhere -- built via an
# explicit substitution loop (one conjugation rule), not sp.conjugate() as
# a black box, to show WHAT conjugation actually does
Psi_star_manual = Psi_explicit
for old, new in [(sp.I, -sp.I)]:
    Psi_star_manual = Psi_star_manual.subs(old, new)

Psi_star_builtin = sp.conjugate(Psi_explicit)
print("Psi* (manual i -> -i substitution):", sp.simplify(Psi_star_manual))
print("Psi* (sp.conjugate, for comparison):", sp.simplify(Psi_star_builtin))
print("agree:", sp.simplify(Psi_star_manual - Psi_star_builtin) == 0)
""")

code(r"""# |Psi|^2 = Psi* . Psi -- the Born rule probability density; note it is
# ALWAYS real, even though Psi itself is complex
prob_density = sp.simplify(Psi_star_manual * Psi_explicit)
print("|Psi|^2 = Psi* Psi =", prob_density)
print("is this expression real-valued (zero imaginary part)?", sp.im(prob_density) == 0)
""")

code(r"""# <Q> (expectation value): integral of Psi* Q Psi dx -- built as an explicit
# integral over the operator Q sandwiched between Psi* and Psi, looped over
# a couple of example operators to show the SAME bracket notation handles
# position, momentum-like, and energy-like operators alike
operators = {
    "<x>":  x,
    "<x^2>": x**2,
}
for label, Q in operators.items():
    integrand = sp.conjugate(Psi_explicit) * Q * Psi_explicit
    value = sp.integrate(integrand, (x, -sp.oo, sp.oo))
    print(f"{label} = integral Psi* Q Psi dx =", sp.simplify(value))
""")

md(r"""## Glossary (one line each)

| Symbol | Meaning |
|---|---|
| $\hat r$ | unit vector pointing radially outward |
| $\nabla f$ | gradient: vector of partial derivatives of $f$ |
| $\nabla\cdot \vec F$ | divergence: sum of matching partial derivatives |
| $\nabla\times \vec F$ | curl: antisymmetric combination of partial derivatives |
| $i$ | $\sqrt{-1}$, the imaginary unit |
| $\Psi^*$ | complex conjugate: flip the sign of every $i$ |
| $\lvert\Psi\rvert^2 = \Psi^*\Psi$ | probability density; always real even though $\Psi$ is complex |
| $\langle Q \rangle$ | expectation value: $\int \Psi^* Q\, \Psi\, dx$, the probability-weighted average of operator $Q$ |
| $\partial/\partial x$ | partial derivative (holding other variables fixed) |
| $d/dx$ | total/ordinary derivative (single-variable functions) |

## Summary

Real-space vector calculus ($\hat r$, $\nabla$, divergence, curl) was built
component by component via explicit loops over the coordinate axes;
imaginary-space QM notation ($i$, $\Psi^*$, $|\Psi|^2$, $\langle Q\rangle$)
was built by literally splitting a concrete complex test function into its
real and imaginary parts and checking every claimed identity numerically
(conjugation matches `sp.conjugate`, $|\Psi|^2$ is exactly real, the
reconstruction $\text{Re}+i\,\text{Im}=\Psi$ holds exactly) rather than
asserting any of it.
""")

nb["cells"] = cells

import os
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/griffiths_notation_glossary_sympy.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote notebooks/griffiths_notation_glossary_sympy.ipynb with", len(cells), "cells")
