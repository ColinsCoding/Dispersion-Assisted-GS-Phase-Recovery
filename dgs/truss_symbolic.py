"""Symbolic (SymPy) truss statics -- the gap dgs/truss.py leaves open.

dgs.truss.solve_truss does exact method-of-joints statics, but entirely
in NumPy: every coordinate and load must already be a number, so it can
only ever answer "what is the force in THIS member, for THIS specific
load." It cannot answer the question a real structural design tool
(the category of thing Autodesk's structural analysis products are
built for) actually needs: "what is the force in this member, as a
FORMULA in the load P, so I can see how it scales before committing to
a number." This module answers that -- the SAME method-of-joints
equilibrium system, but with SymPy symbols standing in for any
coordinate or load the caller wants to leave free, solved via
sp.linsolve to get exact closed-form expressions.

Verified two ways: (1) with everything numeric, it must agree with
dgs.truss.solve_truss's independent NumPy solve to machine precision --
same physics, different solver; (2) with a load left symbolic, it must
recover the SAME numeric solution as (1) after substituting the actual
load value back in -- the symbolic and numeric routes must be
consistent with each other, not just each internally plausible.
"""
import sympy as sp

from dgs.truss import solve_truss


def solve_truss_symbolic(nodes, members, supports, loads):
    """Method of joints, done symbolically. Same call signature as
    dgs.truss.solve_truss (nodes: {name:(x,y)}, members: [(a,b),...],
    supports: {name:(fix_x,fix_y)}, loads: {name:(Fx,Fy)}) -- but any
    coordinate or load component may be a plain number OR a SymPy
    Symbol/expression. Returns (member_forces, reactions) as dicts of
    SymPy expressions (a plain number in, a plain number out, if
    nothing symbolic was given -- verified in tests)."""
    node_list = list(nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)

    reaction_dofs = []
    for n, (fx, fy) in supports.items():
        if fx:
            reaction_dofs.append((n, 0))
        if fy:
            reaction_dofs.append((n, 1))

    n_unknowns = len(members) + len(reaction_dofs)
    if n_unknowns != 2 * N:
        raise ValueError(
            f"this symbolic solver requires a statically determinate truss "
            f"(members + reactions = 2*nodes); got {n_unknowns} unknowns for {2*N} equations"
        )

    force_syms = sp.symbols(f"f0:{len(members)}", real=True)
    reaction_syms = sp.symbols(f"r0:{len(reaction_dofs)}", real=True)
    unknowns = list(force_syms) + list(reaction_syms)

    # one equilibrium row per (node, axis): sum of member-force components
    # at that node + reaction components + applied load = 0
    equations = [sp.Integer(0) for _ in range(2 * N)]

    for m, (a, c) in enumerate(members):
        pa = sp.Matrix([sp.sympify(v) for v in nodes[a]])
        pc = sp.Matrix([sp.sympify(v) for v in nodes[c]])
        d = pc - pa
        length = sp.sqrt(d.dot(d))
        u = d / length   # unit vector a -> c (may be symbolic if coordinates are)
        ia, ic = idx[a], idx[c]
        equations[2 * ia] += u[0] * force_syms[m]
        equations[2 * ia + 1] += u[1] * force_syms[m]
        equations[2 * ic] -= u[0] * force_syms[m]
        equations[2 * ic + 1] -= u[1] * force_syms[m]

    for k, (n, axis) in enumerate(reaction_dofs):
        equations[2 * idx[n] + axis] += reaction_syms[k]

    for n, (fx, fy) in loads.items():
        equations[2 * idx[n]] += sp.sympify(fx)
        equations[2 * idx[n] + 1] += sp.sympify(fy)

    solution = sp.linsolve(equations, unknowns)
    if len(solution) == 0:
        raise ValueError("no solution found -- check for a mechanism (unstable truss) or bad geometry")
    values = next(iter(solution))
    values = [sp.simplify(v) for v in values]

    member_forces = {members[m]: values[m] for m in range(len(members))}
    reactions = {n: [sp.Integer(0), sp.Integer(0)] for n in supports}
    for k, (n, axis) in enumerate(reaction_dofs):
        reactions[n][axis] = values[len(members) + k]
    return member_forces, {n: tuple(v) for n, v in reactions.items()}


if __name__ == "__main__":
    print("=== SYMBOLIC TRUSS SOLVER: THE SAME 3-BAR TRUSS, TWO WAYS ===\n")

    nodes = {"A": (0.0, 0.0), "B": (2.0, 0.0), "C": (1.0, 1.5)}
    members = [("A", "C"), ("B", "C"), ("A", "B")]
    supports = {"A": (True, True), "B": (False, True)}

    print("--- fully numeric (must match dgs.truss.solve_truss) ---")
    loads_numeric = {"C": (0.0, -10000.0)}
    mf_numeric_ref, rx_numeric_ref = solve_truss(nodes, members, supports, loads_numeric)
    mf_sym, rx_sym = solve_truss_symbolic(nodes, members, supports, loads_numeric)
    for k in mf_numeric_ref:
        print(f"  {k[0]}-{k[1]}: numeric solve_truss = {mf_numeric_ref[k]:.4f}, "
              f"symbolic solve_truss_symbolic = {float(mf_sym[k]):.4f}")
        assert abs(mf_numeric_ref[k] - float(mf_sym[k])) < 1e-9

    print("\n--- genuinely symbolic: load P left as a free parameter ---")
    P = sp.Symbol("P", positive=True)
    loads_symbolic = {"C": (0, -P)}
    mf_param, rx_param = solve_truss_symbolic(nodes, members, supports, loads_symbolic)
    for k, expr in mf_param.items():
        print(f"  {k[0]}-{k[1]} = {expr}   (an exact formula in P, not a number)")

    print("\n--- substituting P=10000 back in must reproduce the numeric answer above ---")
    for k, expr in mf_param.items():
        recovered = float(expr.subs(P, 10000.0))
        print(f"  {k[0]}-{k[1]}: {recovered:.4f}  (numeric reference: {mf_numeric_ref[k]:.4f})")
        assert abs(recovered - mf_numeric_ref[k]) < 1e-9

    print("\n[OK] symbolic and numeric routes agree exactly, both directions")
    print("\n=== DONE ===")
