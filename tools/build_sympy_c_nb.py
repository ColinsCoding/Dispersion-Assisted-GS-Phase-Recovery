"""Build notebooks/netlist_power_sympy_to_c.ipynb (dgs/field_circuits.py + dgs/sympy_to_c.py).

Run:  py -3.12 tools/build_sympy_c_nb.py
Then: py -3.12 -m jupyter nbconvert --to notebook --execute --inplace notebooks/netlist_power_sympy_to_c.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "netlist_power_sympy_to_c.ipynb"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# Netlist polarity, power for every element, and SymPy → C

1. **Netlists:** node definitions, polarity references, and p = vi for every element (Tellegen)
2. **Operators:** `=`, `==`, `<`, `<=`, `>`, `>=` in math, SymPy, and C (`sp.ccode`)
3. **Even and odd functions**
4. **Air hockey puck:** spin vs slide, the radius and height, and a wall bounce written with comparisons
5. **Generated C,** compiled with `gcc -std=c99 -Wall -Werror -pedantic` and checked against SymPy
""")
code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import numpy as np, pandas as pd, sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
sp.init_printing(use_unicode=True)
pd.set_option("display.max_colwidth", 90)
from dgs.field_circuits import Circuit
from dgs import sympy_to_c as S
""")

md(r"""
## 1. Netlist definitions and polarity

| Term | Definition |
|---|---|
| **node** | a set of wires at one voltage (one name in the netlist); **node 0** is ground, V = 0 |
| **branch / element** | `kind name a b value`: its **+ reference is node a**, − is node b |
| **v** | V(a) − V(b), the voltage of + relative to − |
| **i** | reference current flowing from a **through the element** to b, so it *enters the + terminal* |
| **p = v i** | power **absorbed** (passive sign convention). p > 0: absorbs (resistors always). p < 0: delivers |

A *negative* answer for v or i doesn't mean it's wrong. It means the real direction is opposite
to the reference you picked. **Tellegen's theorem:** the absorbed powers of all elements add up
to exactly 0, because energy is conserved. It's a free check on every circuit you solve.
""")
code(r"""
ckt = (Circuit().V("V1", "a", "0", 10).R("R1", "a", "b", 1)
       .R("R2", "b", "0", 4).V("V2", "b", "c", 2).R("R3", "c", "0", 2))
print(ckt.paper())
display(pd.DataFrame(ckt.element_table()))
print("Tellegen: sum of absorbed power =", ckt.tellegen())
""")
code(r"""
# the same thing symbolically: every quantity as an expression
Vs, R1, R2 = sp.symbols("V_s R_1 R_2", positive=True)
sym = Circuit().V("Vs", "a", "0", Vs).R("R1", "a", "b", R1).R("R2", "b", "0", R2)
display(pd.DataFrame(sym.element_table()))
print("Tellegen:", sym.tellegen())
""")

md(r"""
## 2. `=` vs `==` and the comparison operators

In math, "=" states a fact. In C, `=` **stores** a value and `==` **asks** whether two values are
equal. SymPy keeps them apart: `Eq(a, b)` is the math equals (printed `a == b`), and
`assign_to=` makes the C assignment.
""")
code(r"""
display(pd.DataFrame(S.operator_table(), columns=["meaning", "SymPy", "sp.ccode", "note"]))
""")

md(r"""
## 3. Even and odd functions

Every function splits into even + odd: f(x) = [f(x) + f(−x)]/2 + [f(x) − f(−x)]/2. Even means
mirror-symmetric (cos, x², |x|); odd means point-symmetric (sin, x³). In physics, the
particle-in-a-box states alternate even and odd about the center, and integrals of odd
functions over symmetric limits vanish. That's why ⟨x⟩ = 0 for symmetric states.
""")
code(r"""
fs = [S.x**2, S.x**3, sp.cos(S.x), sp.sin(S.x), sp.exp(S.x), S.x**2 + S.x]
display(pd.DataFrame([(sp.sstr(f), *map(sp.sstr, S.even_odd_parts(f)), S.parity(f)) for f in fs],
                     columns=["f(x)", "even part", "odd part", "parity"]))
xs = np.linspace(-2, 2, 200); ev, od = S.even_odd_parts(sp.exp(S.x))
plt.figure(figsize=(6, 3.2))
for f, lab in [(sp.exp(S.x), "e^x"), (ev, "even part = cosh x"), (od, "odd part = sinh x")]:
    plt.plot(xs, sp.lambdify(S.x, f)(xs), label=lab)
plt.axvline(0, c="gray", lw=0.5); plt.axhline(0, c="gray", lw=0.5); plt.legend(); plt.tight_layout(); plt.show()
""")

md(r"""
## 4. Air hockey puck: radius, height, and spin

Model the puck as a solid cylinder (radius R, height h, mass M). SymPy integrates r² dm:
- Spinning flat on the table, about its axis: I = ½MR². **The height drops out.**
- Flipping, about a diameter: I = M(R²/4 + h²/12). Here the height matters.

The kinetic energy is sliding plus spinning, K = ½Mv² + ½Iω².
""")
code(r"""
I = S.puck_inertia(); E = S.puck_energy()
display(sp.Eq(sp.Symbol("I_axis"), I["I_axis"]), sp.Eq(sp.Symbol("I_diameter"), I["I_diameter"]))
display(sp.Eq(sp.Symbol("K"), E["K"]), sp.Eq(sp.Symbol("spin fraction"), E["spin_fraction"]))
num = {S.M: 0.017, S.R: 0.0325, S.v: 3.0, S.w: 40.0}      # roughly a 17 g, 6.5 cm puck
print(f"17 g puck, 3 m/s, 40 rad/s: K = {float(E['K'].subs(num))*1e3:.1f} mJ, "
      f"{float(E['spin_fraction'].subs(num))*100:.1f}% of it in the spin")
""")
code(r"""
bounce = S.wall_bounce(); display(bounce)
step = sp.lambdify((S.px, S.vx, S.R, S.xmin, S.xmax, S.e), bounce)
xp, vxp, dt, xs_, ts = 0.5, 2.0, 1e-3, [], []
for k in range(1500):
    vxp = step(xp, vxp, 0.0325, 0.0, 1.0, 0.85); xp += vxp * dt
    xs_.append(xp); ts.append(k * dt)
plt.figure(figsize=(6.5, 3)); plt.plot(ts, xs_); plt.axhline(0.0325, ls="--", c="gray"); plt.axhline(1 - 0.0325, ls="--", c="gray")
plt.xlabel("t (s)"); plt.ylabel("x (m)"); plt.title("puck between two walls, restitution 0.85"); plt.tight_layout(); plt.show()
""")

md(r"""
## 5. SymPy → C, compiled and checked

`sp.ccode` turns each expression into C. Assigned to a variable, the Piecewise becomes an
`if / else` built from `<=`, `>=`, `<`, `>`, `&&` and `||`. SymPy merged the two wall branches,
since both give −e vₓ, into one condition in a rearranged form. That's logically the same, but
it's a reminder to *test* generated code, not just read it. Without `assign_to` you'd get a
nested ternary `cond ? a : b`. gcc compiles it with every warning treated as an error,
and the numbers must match SymPy's.
""")
code(r"""
fns = [S.c_function("puck_KE", E["K"], [S.M, S.R, S.v, S.w]),
       S.c_function("wall_vx", bounce, [S.px, S.vx, S.R, S.xmin, S.xmax, S.e])]
for f in fns: print(f)
calls = ["puck_KE(0.017, 0.0325, 3.0, 40.0)", "wall_vx(0.03, -2.0, 0.0325, 0.0, 1.0, 0.85)",
         "wall_vx(0.5, -2.0, 0.0325, 0.0, 1.0, 0.85)"]
want = [float(E["K"].subs(num)), float(step(0.03, -2.0, 0.0325, 0.0, 1.0, 0.85)), float(step(0.5, -2.0, 0.0325, 0.0, 1.0, 0.85))]
try:
    got, src = S.compile_and_run(fns, calls)
    display(pd.DataFrame({"C call": calls, "gcc result": got, "SymPy": want,
                          "match": np.isclose(got, want, rtol=1e-12)}))
except S.ToolUnavailable as err:
    print("skipped:", err)
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("wrote", OUT, "with", len(cells), "cells")
