"""External variables and scope: how a name finds its value (K&R chapter 4).

A variable is not just a value -- it has a SCOPE (where the name is visible), a
LINKAGE (whether the same name in different files means the same object), and a
STORAGE DURATION (how long the object lives). C's storage classes set these:

  declaration                scope      linkage    storage duration
  -----------------------    --------   --------   ----------------
  int g;        (file)       file       external   static (whole program)
  static int s; (file)       file       internal   static
  int x;        (in a block)  block      none       automatic (dies at block end)
  static int c; (in a block)  block      none       static (PERSISTS across calls)
  extern int g; (in a block)  block      external   static (refers to the file's g)

An EXTERNAL variable (file scope, external linkage) is the shared, global state
every function can reach; an AUTOMATIC local is private and transient. Getting
these right is the difference between a rate constant every step reads and a
scratch value that must not leak.

This module makes scope concrete three ways:
  1. a lexical SCOPE CHAIN (define / lookup / assign) that resolves names by
     walking outward -- the exact rule a compiler uses, including SHADOWING;
  2. declaration_properties(), the K&R storage-class table as code;
  3. a KINETICS integrator where the rate constant k is an EXTERNAL variable held
     in an outer scope that every step reaches into, while each step's dA is a
     block-scoped local that does not leak -- verified against A(t)=A0 exp(-k t)
     (same physics as dgs.physical_chemistry.first_order_kinetics).

And because "external variables and scope" is a C idea, static vs automatic vs
external storage is also demonstrated in real compiled C (gcc-guarded, same
pattern as dgs.syntax_semantics / dgs.c_type_precision). NumPy + stdlib; py-3.13.
"""

import os
import subprocess

import numpy as np

GCC_DEFAULT = r"C:\msys64\mingw64\bin\gcc.exe"


# ----------------------------------------------------------------------
# 1. The lexical scope chain: name resolution by walking outward
# ----------------------------------------------------------------------

class Scope:
    """A lexical scope with an optional enclosing (parent) scope -- a block,
    function, file, or global frame. Names resolve by searching this scope then
    each enclosing one, so an inner definition SHADOWS an outer one."""

    def __init__(self, parent=None, name="block"):
        self.vars = {}
        self.parent = parent
        self.name = name

    def define(self, name, value):
        """Declare a name IN THIS scope (a new local; shadows any outer one)."""
        self.vars[name] = value

    def resolve(self, name):
        """Return the Scope where `name` is visible (nearest enclosing), or
        raise NameError. This is the lookup a compiler performs."""
        s = self
        while s is not None:
            if name in s.vars:
                return s
            s = s.parent
        raise NameError(f"name {name!r} is not in scope")

    def lookup(self, name):
        """The value `name` currently denotes, from the nearest scope binding it."""
        return self.resolve(name).vars[name]

    def assign(self, name, value):
        """Assign to the EXISTING binding in the nearest enclosing scope (as
        C assignment does) -- not a new local. Raises if the name is undeclared."""
        self.resolve(name).vars[name] = value

    def depth(self):
        """How many scopes enclose this one (global = 0)."""
        return 0 if self.parent is None else 1 + self.parent.depth()


# ----------------------------------------------------------------------
# 2. The C storage-class rules, as code
# ----------------------------------------------------------------------

def declaration_properties(storage_class=None, at_file_scope=False):
    """Return (scope, linkage, storage_duration) for a C declaration given its
    storage class and whether it sits at file scope -- the K&R rules verbatim.
    storage_class in {None,'extern','static','auto','register'}."""
    if storage_class not in (None, "extern", "static", "auto", "register"):
        raise ValueError(f"unknown storage class {storage_class!r}")
    if at_file_scope:
        if storage_class in ("auto", "register"):
            raise ValueError(f"{storage_class!r} is illegal at file scope")
        linkage = "internal" if storage_class == "static" else "external"
        return {"scope": "file", "linkage": linkage, "storage_duration": "static"}
    # block scope
    if storage_class == "extern":
        return {"scope": "block", "linkage": "external", "storage_duration": "static"}
    if storage_class == "static":
        return {"scope": "block", "linkage": "none", "storage_duration": "static"}
    return {"scope": "block", "linkage": "none", "storage_duration": "automatic"}


def make_static_counter():
    """Emulate a C function with a `static int count;` local: the count PERSISTS
    across calls (static storage duration) even though its name is block-scoped.
    A fresh counter starts over -- static storage is per-object, not shared."""
    state = {"count": 0}                 # the static object, private to this closure

    def tick():
        state["count"] += 1              # static: survives between calls
        scratch = state["count"] * 10    # automatic: recomputed, never persists
        return state["count"], scratch

    return tick


# ----------------------------------------------------------------------
# 3. Kinetics: the rate constant as an external variable, dA as a local
# ----------------------------------------------------------------------

def kinetics_with_scope(A0, k, t_end, dt):
    """Integrate first-order decay A -> B with the rate constant k held as an
    EXTERNAL variable in an outer (file) scope. Each Euler step opens a fresh
    block scope, reaches OUT to read k, and computes a block-local dA that dies
    with the step. Returns (trajectory Nx3 of [t, A, B], global_scope). After
    the run the external k is still visible but the per-step dA is not -- the
    whole point of scope."""
    if A0 < 0 or k < 0 or t_end <= 0 or dt <= 0:
        raise ValueError("need A0>=0, k>=0, t_end>0, dt>0")
    global_scope = Scope(name="file")
    global_scope.define("k", k)                 # external variable: shared config
    A, B, t = float(A0), 0.0, 0.0
    traj = [(0.0, A, B)]
    for _ in range(int(round(t_end / dt))):
        step = Scope(parent=global_scope, name="block")   # new block each step
        dA = -step.lookup("k") * A * dt          # reach out to the external k
        step.define("dA", dA)                    # block-scoped local, discarded
        A += dA
        B -= dA
        t += dt
        traj.append((t, A, B))
    return np.array(traj), global_scope


# ----------------------------------------------------------------------
# 4. The same idea in real C: external / static / automatic storage
# ----------------------------------------------------------------------

C_SOURCE_SCOPE = r"""
#include <stdio.h>

int g = 100;              /* external variable: file scope, external linkage */
static int s_file = 5;    /* internal linkage: private to this file */

int demo(void) {
    static int calls = 0; /* static storage duration: PERSISTS across calls */
    int local = 0;        /* automatic: reborn (reset) every call */
    calls++;
    local++;
    return calls * 1000 + local * 10 + s_file;   /* 1*1000+1*10+5 = 1015, then 2015 */
}

int main(void) {
    /* separate statements: C does NOT specify the evaluation order of function
       arguments, so demo(), demo() in one printf could run right-to-left */
    int r1 = demo();
    int r2 = demo();
    printf("%d %d\n", r1, r2);           /* static persists, automatic resets */
    printf("%d\n", g);                   /* the external variable */
    return 0;
}
"""


def gcc_available(gcc_path=GCC_DEFAULT):
    """Whether a C toolchain is present for the compiled demonstration."""
    return os.path.exists(gcc_path)


def compile_and_run_c(out_dir, gcc_path=GCC_DEFAULT):
    """Compile and run C_SOURCE_SCOPE, returning the two demo() results, the
    external g, and the derived facts (static persisted / automatic reset)."""
    src = os.path.join(out_dir, "scope_demo.c")
    exe = os.path.join(out_dir, "scope_demo.exe")
    with open(src, "w") as f:
        f.write(C_SOURCE_SCOPE)
    r = subprocess.run([gcc_path, "-O2", "-o", exe, src], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gcc failed: {r.stderr}")
    out = subprocess.run([exe], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"C program failed: {out.stderr}")
    line0, line1 = out.stdout.strip().splitlines()
    call1, call2 = map(int, line0.split())
    g = int(line1)
    return {
        "call1": call1, "call2": call2, "g": g,
        "static_persisted": (call2 // 1000) == (call1 // 1000) + 1,
        "automatic_reset": (call2 % 100) == (call1 % 100),   # local same each call
    }


if __name__ == "__main__":
    print("SCOPE CHAIN (shadowing + outward resolution):")
    g = Scope(name="global"); g.define("x", 1); g.define("k", 0.5)
    blk = Scope(parent=g, name="block"); blk.define("x", 99)   # shadows global x
    print(f"  inner x = {blk.lookup('x')} (shadows global 1); "
          f"inner sees global k = {blk.lookup('k')}; depth={blk.depth()}")

    print("\nSTORAGE CLASSES (K&R rules):")
    for sc, fs in [(None, True), ("static", True), (None, False),
                   ("static", False), ("extern", False)]:
        p = declaration_properties(sc, fs)
        print(f"  {str(sc):7s} @ {'file ' if fs else 'block'}: {p}")

    tick = make_static_counter()
    print(f"\nstatic local persists: {tick()[0]}, {tick()[0]}, {tick()[0]} "
          f"(fresh counter: {make_static_counter()()[0]})")

    print("\nKINETICS (k is external, dA is block-local):")
    traj, gs = kinetics_with_scope(A0=1.0, k=0.7, t_end=5.0, dt=0.001)
    A_end = traj[-1, 1]
    print(f"  A(5) numeric {A_end:.5f} vs analytic {np.exp(-0.7*5):.5f}")
    print(f"  external k still in scope: {gs.lookup('k')}; "
          f"block-local dA leaked? ", end="")
    try:
        gs.lookup("dA"); print("YES (bug)")
    except NameError:
        print("no (correct -- it was block-scoped)")

    print("\nC demonstration:")
    if gcc_available():
        c = compile_and_run_c(os.environ.get("TEMP", "."))
        print(f"  demo() -> {c['call1']}, {c['call2']}; g={c['g']}; "
              f"static persisted? {c['static_persisted']}; "
              f"automatic reset? {c['automatic_reset']}")
    else:
        print("  (gcc not found -- skipping)")
