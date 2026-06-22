"""Truss analysis -- statics + linear algebra for civil engineering structures.

A pin-jointed truss carries load only as axial forces in its straight members
(tension or compression). The METHOD OF JOINTS writes static equilibrium at every
joint -- sum(Fx)=0 and sum(Fy)=0 -- which is two linear equations per node. Stack
them and you get a single linear system

    A f = -load ,

where the unknowns f are the member axial forces plus the support reactions, and A
is built from the members' direction cosines. Solve it with np.linalg.solve: the
truss is just a matrix equation. Sign convention: tension positive, compression
negative. This is how every roof, bridge, and crane truss is sized -- statics
(dgs.statics) turned into linear algebra. NumPy, 2-D. Education.
"""

import numpy as np


def is_determinate(n_nodes, n_members, n_reactions):
    """A 2-D truss is statically determinate when members + reactions = 2*nodes. Fewer
    is a mechanism (unstable); more is statically indeterminate (needs material props)."""
    return n_members + n_reactions == 2 * n_nodes


def solve_truss(nodes, members, supports, loads):
    """Solve a 2-D pin-jointed truss by the method of joints.

    nodes    : {name: (x, y)}
    members  : [(node_a, node_b), ...]   straight two-force members
    supports : {name: (fix_x, fix_y)}    booleans -- pin=(True,True), roller=(False,True)
    loads    : {name: (Fx, Fy)}          external joint loads

    Returns (member_forces, reactions): member_forces[(a,b)] is the axial force
    (tension +, compression -); reactions[name] is the (Rx, Ry) support reaction."""
    node_list = list(nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)

    reaction_dofs = []                              # (node, axis) per constrained DOF
    for n, (fx, fy) in supports.items():
        if fx:
            reaction_dofs.append((n, 0))
        if fy:
            reaction_dofs.append((n, 1))

    n_unknowns = len(members) + len(reaction_dofs)
    A = np.zeros((2 * N, n_unknowns))
    b = np.zeros(2 * N)

    for m, (a, c) in enumerate(members):           # each member's direction cosines
        pa, pc = np.array(nodes[a], float), np.array(nodes[c], float)
        u = (pc - pa) / np.linalg.norm(pc - pa)    # unit vector a -> c
        ia, ic = idx[a], idx[c]
        A[2 * ia, m] += u[0]; A[2 * ia + 1, m] += u[1]      # tension pulls a toward c
        A[2 * ic, m] -= u[0]; A[2 * ic + 1, m] -= u[1]      # and c toward a

    for k, (n, axis) in enumerate(reaction_dofs):
        A[2 * idx[n] + axis, len(members) + k] += 1.0

    for n, (fx, fy) in loads.items():              # sum(forces) + load = 0  ->  A f = -load
        b[2 * idx[n]] -= fx
        b[2 * idx[n] + 1] -= fy

    if n_unknowns == 2 * N:
        x = np.linalg.solve(A, b)
    else:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)

    member_forces = {members[m]: float(x[m]) for m in range(len(members))}
    reactions = {n: [0.0, 0.0] for n in supports}
    for k, (n, axis) in enumerate(reaction_dofs):
        reactions[n][axis] = float(x[len(members) + k])
    return member_forces, {n: tuple(v) for n, v in reactions.items()}


if __name__ == "__main__":
    # a simple triangular truss: pin at A, roller at B, apex C, 10 kN down at C
    nodes = {"A": (0, 0), "B": (4, 0), "C": (2, 2)}
    members = [("A", "C"), ("B", "C"), ("A", "B")]
    supports = {"A": (True, True), "B": (False, True)}
    loads = {"C": (0.0, -10.0)}
    print("determinate:", is_determinate(3, 3, 3))
    mf, rx = solve_truss(nodes, members, supports, loads)
    for m, f in mf.items():
        print(f"  member {m}: {f:+.3f} kN  ({'tension' if f > 0 else 'compression'})")
    print("  reactions:", {n: tuple(round(c, 3) for c in v) for n, v in rx.items()})
