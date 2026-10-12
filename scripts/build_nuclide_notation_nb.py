"""Build notebooks/nuclide_notation_loop.ipynb -- "Protons, Neutrons, Electrons
from ^A X^q: Three Rules, SymPy, a Loop Over the Options, and MATLAB".

Companion to dgs/nuclide_notation.py and learn/matlab/nuclide_options.m.

Build:   py -3.13 scripts/build_nuclide_notation_nb.py
Execute: py -3.13 -m jupyter nbconvert --to notebook --execute --inplace
         notebooks/nuclide_notation_loop.ipynb
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
cells = []

cells.append(md(r"""# Protons, Neutrons, Electrons from $^{A}\mathrm{X}^{q}$

> *Which is the number of protons, neutrons, and electrons in $^{16}\mathrm{O}^{2-}$?*

Each number in the notation has a fixed place and a fixed meaning:

| where | symbol | meaning | gives |
|---|---|---|---|
| top-left | $A = 16$ | **mass number** = protons + neutrons | neutrons $= A - Z$ |
| the letters | O | the **element** fixes the atomic number | protons $= Z = 8$ |
| top-right | $q = 2-$ | **net charge** = protons − electrons | electrons $= Z - q$ |

The trap: a *negative* ion has gained electrons. With $q = -2$,
$Z - q = 8 - (-2) = 10$.

Module: [`dgs/nuclide_notation.py`](../dgs/nuclide_notation.py). MATLAB twin:
[`learn/matlab/nuclide_options.m`](../learn/matlab/nuclide_options.m)."""))

cells.append(co("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import sympy as sp
from IPython.display import display, Math
sp.init_printing()
from dgs.nuclide_notation import parse, particle_counts, symbolic_rules, check_options, KNIGHT_OPTIONS"""))

cells.append(md(r"""# 1. The rules as equations

Three definitions, three unknowns. SymPy solves the linear system:"""))

cells.append(co("""sol, (A, Z, q) = symbolic_rules()
for k, v in sol.items():
    display(Math(sp.latex(k) + " = " + sp.latex(v)))
A_, sym, q_ = parse("16O2-")
print(f"16O2-  →  A = {A_}, element {sym} (Z = 8), q = {q_:+d}")
display(Math(r"\\text{substituted: } " + r",\\;".join(f"{sp.latex(k)} = {v.subs({A: A_, Z: 8, q: q_})}" for k, v in sol.items())))"""))

cells.append(md(r"""# 2. Loop over the answer options and test each one

Instead of eyeballing the options, *test* every one against the three rules, and
explain each failure. This is the same idea as a unit test, applied to a quiz:"""))

cells.append(co("""labels = ["8p, 8n, 10e", "8p, 8n, 2e", "None of these", "16p, 16n, 18e", "16p, 16n, 2e", "16p, 0n, 18e"]
for (i, opt, ok, why), lab in zip(check_options("16O2-", KNIGHT_OPTIONS), labels):
    print(f"{'✓' if ok else '✗'} option {i + 1}: {lab:15s} {why}")"""))

cells.append(md(r"""Each wrong option is a specific misreading:

- **2 electrons** counts only the *extra* charge, not the electrons a neutral atom already has.
- **16 protons** reads the mass number as the atomic number.
- **16 neutrons** forgets to subtract the protons from $A$.

# 3. Practice: loop over many ions"""))

cells.append(co("""print(f"{'ion':>8s} {'p':>4s} {'n':>4s} {'e':>4s}   note")
for s in ("1H", "4He2+", "12C", "14C", "23Na+", "35Cl-", "37Cl-", "56Fe3+", "238U"):
    p, n, e = particle_counts(s)
    note = "isotope of the line above (same p, more n)" if s in ("14C", "37Cl-") else ""
    print(f"{s:>8s} {p:4d} {n:4d} {e:4d}   {note}")"""))

cells.append(md(r"""**Isotopes** (same element, different $A$) change only the neutron count. **Ions**
(different $q$) change only the electron count. The element symbol alone fixes the
protons.

# 4. The same loop in MATLAB

[`learn/matlab/nuclide_options.m`](../learn/matlab/nuclide_options.m) parses the
notation with `regexp`, then loops over a matrix of options and uses `isequal`. Run it
headless:

```
matlab -batch "run('learn/matlab/nuclide_options.m')"
```

It prints `8 protons, 8 neutrons, 10 electrons`, marks only option 1 `true`, and
asserts it. `tests/test_nuclide_notation.py` cross-checks it against Python
(`RUN_MATLAB=1`).

## Exercises
1. Write the notation for the ion with 26 protons, 30 neutrons, and 24 electrons.
2. Which has more electrons: $^{40}\mathrm{Ca}^{2+}$ or $^{40}\mathrm{Ar}$? (Same $A$, different everything else.)
3. Extend `check_options` to flag "used the charge with the wrong sign" and add a test for it.
4. An alpha particle is $^{4}\mathrm{He}^{2+}$. How many electrons does it have, and why did Rutherford's experiment (§37.6) need it charged?"""))

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "nuclide_notation_loop.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
