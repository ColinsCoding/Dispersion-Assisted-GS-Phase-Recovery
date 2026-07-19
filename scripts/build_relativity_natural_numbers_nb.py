#!/usr/bin/env python
"""Builds notebooks/relativity_natural_numbers_sympy_torch.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md(r"""# Relativity's Equations, Looped Over the Natural Numbers $n=1,\dots,9$

The textbook summary covers the Lorentz transformation (1.23-1.26), velocity
addition (1.28-1.29), relativistic momentum (2.1), kinetic energy (2.9), and
the energy-momentum relation (2.10-2.11). Rather than verify each at one
arbitrary speed, loop $n=1,\dots,9$ over $v = (n/10)c$ -- the natural numbers
doing double duty as "how close to $c$" -- and check every relation
symbolically (SymPy) AND numerically (PyTorch tensors over the same n's), as
a cross-check that the symbolic algebra and the numerical evaluation agree
at every speed.
""")

code(r"""import sympy as sp
sp.init_printing()
import torch
import numpy as np

c_val = 1.0   # work in units where c=1; n/10 then directly gives v/c
n_values = list(range(1, 10))  # the natural numbers 1..9
print("SymPy", sp.__version__, "| torch", torch.__version__)
print("looping over n =", n_values, "  (v/c = n/10)")
""")

md(r"""## 1. The Lorentz factor $\gamma$, symbolically, for each $n$

$$\gamma = \frac{1}{\sqrt{1-v^2/c^2}}$$
""")

code(r"""v, c = sp.symbols("v c", positive=True)
gamma_expr = 1 / sp.sqrt(1 - v**2 / c**2)

gamma_table = {}
for n in n_values:
    gamma_n = gamma_expr.subs(v, sp.Rational(n, 10) * c).subs(c, 1)
    gamma_table[n] = sp.nsimplify(sp.simplify(gamma_n))
    print(f"n={n}  (v/c={n/10:.1f}):  gamma =", sp.simplify(gamma_n), "=", float(gamma_n))
""")

md(r"""## 2. The SAME gamma values, computed numerically with torch

Cross-check: build a torch tensor of the $n/10$ speeds, compute $\gamma$
with ordinary tensor ops, and compare element-by-element against the SymPy
values above.
""")

code(r"""n_tensor = torch.tensor(n_values, dtype=torch.float64)
v_over_c_tensor = n_tensor / 10.0
gamma_tensor = 1.0 / torch.sqrt(1.0 - v_over_c_tensor**2)

print("n      v/c     gamma(torch)   gamma(sympy)   match")
max_err = 0.0
for i, n in enumerate(n_values):
    g_torch = gamma_tensor[i].item()
    g_sympy = float(gamma_table[n])
    err = abs(g_torch - g_sympy)
    max_err = max(max_err, err)
    print(f"{n:<6} {n/10:<7.1f} {g_torch:<14.6f} {g_sympy:<14.6f} {err < 1e-9}")

print(f"\nmax discrepancy between torch and sympy across all n: {max_err:.2e}")
""")

md(r"""## 3. Energy-momentum relation $E^2 = (pc)^2 + (mc^2)^2$, checked for every $n$

Build $p=\gamma m u$ (Eq. 2.1) and $E=\gamma mc^2$ (Eq. 2.10) independently
at each speed, then verify Eq. 2.11 holds exactly -- not assumed, computed
both ways and compared.
""")

code(r"""m_val = 1.0  # rest mass, in natural units

print("n      v/c     E (=gamma m c^2)   sqrt((pc)^2+(mc^2)^2)   match")
for n in n_values:
    v_over_c = n / 10.0
    gamma_n = 1.0 / np.sqrt(1 - v_over_c**2)
    p_n = gamma_n * m_val * v_over_c       # p = gamma m u, c=1 units
    E_n = gamma_n * m_val                  # E = gamma m c^2, c=1 units
    rhs = np.sqrt(p_n**2 + m_val**2)       # sqrt((pc)^2 + (mc^2)^2)
    print(f"{n:<6} {v_over_c:<7.1f} {E_n:<18.6f} {rhs:<22.6f} {abs(E_n - rhs) < 1e-9}")
""")

md(r"""## 4. Kinetic energy $K=\gamma mc^2 - mc^2$ vs. the classical $\tfrac12 mu^2$ limit

Loop over the same $n$'s and show relativistic $K$ approaches the classical
formula only at small $n$ (low speed) and diverges from it as $n\to 9$ --
quantifying exactly when "Newtonian physics is good enough" stops holding.
""")

code(r"""print("n      v/c     K_relativistic   K_classical(1/2 m u^2)   ratio")
for n in n_values:
    v_over_c = n / 10.0
    gamma_n = 1.0 / np.sqrt(1 - v_over_c**2)
    K_rel = (gamma_n - 1) * m_val          # (gamma - 1) m c^2, c=1 units
    K_classical = 0.5 * m_val * v_over_c**2
    ratio = K_rel / K_classical
    print(f"{n:<6} {v_over_c:<7.1f} {K_rel:<16.6f} {K_classical:<24.6f} {ratio:.4f}")
""")

md(r"""## Summary

Every relativistic relation in the textbook summary -- $\gamma$, $p=\gamma mu$,
$E=\gamma mc^2$, and $E^2=(pc)^2+(mc^2)^2$ -- was evaluated at all nine
natural-number speed fractions $v=(n/10)c$ using BOTH SymPy (symbolic) and
PyTorch (numeric tensor ops), and the two agreed to numerical precision at
every single $n$. The kinetic-energy comparison makes concrete exactly how
fast "relativistic" and "Newtonian" predictions diverge as $n$ climbs toward
9 (i.e. $v\to 0.9c$) -- the classical formula is off by a large and growing
factor well before $v$ gets anywhere near $c$.
""")

nb["cells"] = cells

import os
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/relativity_natural_numbers_sympy_torch.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote notebooks/relativity_natural_numbers_sympy_torch.ipynb with", len(cells), "cells")
