"""From SymPy to C: relational operators, even/odd functions, and an air hockey puck.

1. Operators: what =, ==, <, <=, >, >= mean in math, in SymPy, and in C. In C, ``=`` assigns and
   ``==`` compares; SymPy's ``Eq(a, b)`` is the math "=", printed by ``sp.ccode`` as ``a == b``,
   and ``sp.ccode(expr, assign_to="y")`` prints the assignment ``y = ...;``.
2. Even and odd parts: f(x) = [f(x) + f(-x)]/2 + [f(x) - f(-x)]/2, with symbolic checks.
3. Air hockey puck (solid cylinder, radius R, height h, mass M): moments of inertia, kinetic
   energy split between sliding and spinning, and a wall bounce written as a Piecewise, so the
   generated C uses comparison operators.
4. ``c_function`` wraps expressions into a C99 function, and ``compile_and_run`` checks it with gcc
   against SymPy's own numbers.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import sympy as sp


class ToolUnavailable(RuntimeError):
    """gcc is not installed."""


x = sp.Symbol("x", real=True)

# ---------------------------------------------------------------------------------------------
# 1. Operators
# ---------------------------------------------------------------------------------------------
a_, b_ = sp.symbols("a b", real=True)
OPERATORS = [
    ("equal (math =)", sp.Eq(a_, b_), "compare: is a equal to b?"),
    ("not equal", sp.Ne(a_, b_), ""),
    ("less than", sp.Lt(a_, b_), ""),
    ("less or equal", sp.Le(a_, b_), ""),
    ("greater than", sp.Gt(a_, b_), ""),
    ("greater or equal", sp.Ge(a_, b_), ""),
]


def operator_table():
    """Rows (meaning, SymPy object, C text) for the six relational operators, plus assignment."""
    rows = [(m, sp.srepr(e).split("(")[0], sp.ccode(e), note) for m, e, note in OPERATORS]
    rows.append(("assignment (store a value)", "assign_to=", sp.ccode(a_ + b_, assign_to="y"),
                 "C '=' is NOT equality; mixing them up (if (x = 0)) is a classic C bug"))
    return rows


# ---------------------------------------------------------------------------------------------
# 2. Even and odd
# ---------------------------------------------------------------------------------------------
def even_odd_parts(f, var=x):
    """(even, odd) with f = even + odd, even(-x) = even(x), odd(-x) = -odd(x)."""
    f = sp.sympify(f)
    fm = f.subs(var, -var)
    return sp.simplify((f + fm) / 2), sp.simplify((f - fm) / 2)


def parity(f, var=x):
    """'even', 'odd', 'neither', or 'zero' (both)."""
    ev, od = even_odd_parts(f, var)
    if ev == 0 and od == 0:
        return "zero"
    if od == 0:
        return "even"
    if ev == 0:
        return "odd"
    return "neither"


# ---------------------------------------------------------------------------------------------
# 3. Air hockey puck
# ---------------------------------------------------------------------------------------------
M, R, h, v, w, e = sp.symbols("M R h v omega e", positive=True)
px, vx, xmin, xmax = sp.symbols("x v_x x_min x_max", real=True)


def puck_inertia():
    """Solid cylinder: about its symmetry axis (the spin on the table) and about a diameter.

    Axis: I = M R^2 / 2, independent of the height h. Diameter: I = M (R^2/4 + h^2/12)
    (a wobble or flip). Both come from integrating r^2 dm over the cylinder.
    """
    rho = M / (sp.pi * R**2 * h)
    rr, th, z = sp.symbols("r theta z", positive=True)
    I_axis = sp.integrate(rho * rr**2 * rr, (rr, 0, R), (th, 0, 2 * sp.pi), (z, -h / 2, h / 2))
    # distance^2 to the y axis: x^2 + z^2 with x = r cos(theta)
    I_diam = sp.integrate(rho * ((rr * sp.cos(th))**2 + z**2) * rr, (rr, 0, R), (th, 0, 2 * sp.pi), (z, -h / 2, h / 2))
    return {"I_axis": sp.simplify(I_axis), "I_diameter": sp.simplify(I_diam)}


def puck_energy():
    """KE = M v^2/2 + I w^2/2: sliding plus spinning. Returns total and the spin fraction."""
    I_ax = puck_inertia()["I_axis"]
    K_t, K_r = M * v**2 / 2, I_ax * w**2 / 2
    return {"K_trans": K_t, "K_rot": sp.simplify(K_r), "K": sp.simplify(K_t + K_r),
            "spin_fraction": sp.simplify(K_r / (K_t + K_r))}


def wall_bounce():
    """New v_x after the puck meets the side walls (restitution e): a Piecewise with comparisons.

    The puck's edge touches the left wall when x - R <= x_min and the right wall when
    x + R >= x_max; it reverses only if it is moving into that wall.
    """
    return sp.Piecewise(
        (-e * vx, sp.And(px - R <= xmin, vx < 0)),
        (-e * vx, sp.And(px + R >= xmax, vx > 0)),
        (vx, True))


# ---------------------------------------------------------------------------------------------
# 4. C generation and checking
# ---------------------------------------------------------------------------------------------
def c_function(name, expr, args, ctype="double"):
    """A C99 function computing ``expr`` from ``args`` (SymPy symbols), via sp.ccode."""
    if not name.isidentifier():
        raise ValueError(f"{name!r} is not a valid C identifier")
    missing = sp.sympify(expr).free_symbols - set(args)
    if missing:
        raise ValueError(f"expression uses symbols not in args: {sorted(map(str, missing))}")
    sig = ", ".join(f"{ctype} {sp.ccode(a)}" for a in args)
    body = sp.ccode(expr, assign_to="result", standard="C99")
    body = "\n    ".join(body.splitlines())
    return f"{ctype} {name}({sig}) {{\n    {ctype} result;\n    {body}\n    return result;\n}}\n"


def find_gcc():
    return shutil.which("gcc") or next((p for p in [r"C:\msys64\mingw64\bin\gcc.exe"] if os.path.exists(p)), None)


def compile_and_run(functions, calls):
    """Compile C functions plus a main() that prints each call; return the printed doubles.

    ``functions``: list of C source strings. ``calls``: list of C call expressions, e.g.
    "puck_KE(0.17, 0.05, 2.0, 30.0)". Uses -std=c99 -Wall -Werror -pedantic.
    """
    gcc = find_gcc()
    if gcc is None:
        raise ToolUnavailable("gcc not found (MinGW-w64 at C:\\msys64\\mingw64\\bin, or build-essential)")
    prints = "\n".join(f'    printf("%.17g\\n", (double)({c}));' for c in calls)
    src = "#include <math.h>\n#include <stdio.h>\n\n" + "\n".join(functions) + \
        f"\nint main(void) {{\n{prints}\n    return 0;\n}}\n"
    with tempfile.TemporaryDirectory() as d:
        c_path, exe = Path(d) / "gen.c", Path(d) / "gen.exe"
        c_path.write_text(src)
        env = dict(os.environ, PATH=str(Path(gcc).parent) + os.pathsep + os.environ.get("PATH", ""))
        cp = subprocess.run([gcc, "-std=c99", "-Wall", "-Werror", "-pedantic", "-O2", str(c_path),
                             "-o", str(exe), "-lm"], capture_output=True, text=True, env=env)
        if cp.returncode:
            raise RuntimeError("gcc failed:\n" + cp.stderr[-2000:] + "\n--- source ---\n" + src)
        out = subprocess.run([str(exe)], capture_output=True, text=True, env=env, check=True).stdout
    return [float(line) for line in out.split()], src
