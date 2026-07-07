"""Statics as linear algebra: a truss is the matrix equation A x = b.

The method of joints says every pin is in equilibrium: sum(F_x)=0 and sum(F_y)=0.
Write those two equations at every joint and the whole truss becomes one linear
system
        A x = b,
where x holds the unknown MEMBER FORCES (tension +, compression -) and SUPPORT
REACTIONS, b holds the applied loads, and each column of A is a member's direction
cosines dropped into the two rows of each joint it touches. Solving the truss is
solving a linear system -- and the questions statics asks map exactly onto linear
algebra:

  * STATICALLY DETERMINATE  <=>  A is square and full rank (a unique solution):
        members m + reactions r = 2 * joints j,   and rank(A) = 2j.
  * STATICALLY INDETERMINATE <=>  more unknowns than equations (m+r > 2j): the
    equilibrium equations alone cannot fix the forces (redundant members).
  * MECHANISM / UNSTABLE     <=>  too few unknowns, or a rank deficiency even when
    the counts match (a geometric instability the count test misses -- only the
    RANK catches it).

So "is this truss solvable, and how?" is "is this matrix invertible?" -- the same
linear algebra as eigenproblems (dgs.vibration_modes) and least squares, now holding
a bridge up. Verified against a hand-solved symmetric two-bar truss and by an
INDEPENDENT global-equilibrium check (the reactions must balance the loads in force
and moment). Companion to dgs.weld_statics (which handles a single loaded joint).
NumPy only; py-3.13.
"""

import numpy as np


def _reaction_dofs(supports):
    """Flatten supports {node: (fix_x, fix_y)} into an ordered list of the
    constrained DOFs [(node, 'x'), (node, 'y'), ...] -- the reaction unknowns."""
    dofs = []
    for node, (fx, fy) in supports.items():
        if fx:
            dofs.append((node, "x"))
        if fy:
            dofs.append((node, "y"))
    return dofs


def static_determinacy(n_joints, n_members, n_reactions):
    """Classify a planar truss from the COUNTS alone: with 2 equilibrium
    equations per joint, compare unknowns (m+r) to equations (2j).
    Returns (classification, degree) where degree = (m+r) - 2j:
      degree == 0 -> 'determinate' (necessary, not sufficient -- see the rank),
      degree  > 0 -> 'indeterminate' (redundant members),
      degree  < 0 -> 'mechanism' (too few constraints)."""
    if n_joints < 1:
        raise ValueError("need at least one joint")
    degree = n_members + n_reactions - 2 * n_joints
    cls = "determinate" if degree == 0 else ("indeterminate" if degree > 0 else "mechanism")
    return cls, degree


def assemble(nodes, members, supports, loads):
    """Build (A, b, columns) for the joint-equilibrium system A x = b.
    nodes    : {name: (x, y)}
    members  : list of (nodeP, nodeQ) -- an axial member (tension positive)
    supports : {node: (fix_x, fix_y)} booleans (pin=(1,1), roller=(1,0)/(0,1))
    loads    : {node: (Fx, Fy)} applied forces
    Columns of A are the member forces (in `members` order) then the reactions.
    Row 2k / 2k+1 are joint k's sum(F_x)=0 / sum(F_y)=0."""
    node_list = list(nodes)
    idx = {n: k for k, n in enumerate(node_list)}
    j = len(node_list)
    rdofs = _reaction_dofs(supports)
    m, r = len(members), len(rdofs)
    A = np.zeros((2 * j, m + r))
    b = np.zeros(2 * j)

    for col, (p, q) in enumerate(members):
        if p not in idx or q not in idx:
            raise ValueError(f"member ({p},{q}) references an unknown node")
        pp, qq = np.asarray(nodes[p], float), np.asarray(nodes[q], float)
        d = qq - pp
        L = float(np.hypot(*d))
        if L == 0:
            raise ValueError(f"member ({p},{q}) has zero length")
        u = d / L                                   # unit vector p -> q
        # a member in tension pulls joint p toward q (+u) and q toward p (-u)
        A[2 * idx[p], col] += u[0]; A[2 * idx[p] + 1, col] += u[1]
        A[2 * idx[q], col] -= u[0]; A[2 * idx[q] + 1, col] -= u[1]

    for k, (node, dof) in enumerate(rdofs):
        row = 2 * idx[node] + (0 if dof == "x" else 1)
        A[row, m + k] = 1.0                         # reaction is an external force

    for node, (fx, fy) in loads.items():
        if node not in idx:
            raise ValueError(f"load references unknown node {node!r}")
        b[2 * idx[node]] -= fx                       # move load to the RHS
        b[2 * idx[node] + 1] -= fy

    columns = list(members) + [("R", n, d) for (n, d) in rdofs]
    return A, b, columns


def solve_truss(nodes, members, supports, loads):
    """Solve the whole truss as one linear system. Returns a dict with:
      member_forces : {(p,q): axial force}, tension +, compression -,
      reactions     : {(node,'x'/'y'): reaction force},
      classification, degree, rank, n_equations, n_unknowns,
      residual      : ||A x - b|| (≈0 for a well-posed truss).
    A determinate, full-rank truss is solved exactly; otherwise the minimum-norm
    least-squares solution is returned and the classification says why."""
    A, b, cols = assemble(nodes, members, supports, loads)
    neq, nunk = A.shape
    j = neq // 2
    rank = int(np.linalg.matrix_rank(A))
    cls, degree = static_determinacy(j, len(members), nunk - len(members))
    if degree == 0 and rank < neq:
        cls = "mechanism"                            # rank catches geometric instability
    if neq == nunk and rank == neq:
        x = np.linalg.solve(A, b)
    else:
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    residual = float(np.linalg.norm(A @ x - b))
    member_forces = {m: float(x[i]) for i, m in enumerate(members)}
    reactions = {(n, d): float(x[len(members) + k])
                 for k, (_, n, d) in enumerate(cols[len(members):])}
    return {
        "member_forces": member_forces, "reactions": reactions,
        "classification": cls, "degree": degree, "rank": rank,
        "n_equations": neq, "n_unknowns": nunk, "residual": residual,
    }


def global_equilibrium_residual(nodes, reactions, loads):
    """Independent check: the reactions must balance the loads globally. Returns
    (sum_Fx, sum_Fy, sum_M_about_origin) over reactions + loads -- all ≈ 0 for a
    correct solution. This does NOT use the joint equations, so it genuinely
    validates the assembly, not just the solve."""
    Fx = Fy = M = 0.0
    for (node, dof), R in reactions.items():
        x, y = nodes[node]
        if dof == "x":
            Fx += R; M += -y * R
        else:
            Fy += R; M += x * R
    for node, (fx, fy) in loads.items():
        x, y = nodes[node]
        Fx += fx; Fy += fy; M += x * fy - y * fx
    return Fx, Fy, M


if __name__ == "__main__":
    # symmetric two-bar truss: A(-3,0) pin, B(3,0) pin, C(0,-4) with a 100 N pull.
    nodes = {"A": (-3, 0), "B": (3, 0), "C": (0, -4)}
    members = [("C", "A"), ("C", "B")]
    supports = {"A": (True, True), "B": (True, True)}
    loads = {"C": (0.0, -100.0)}
    sol = solve_truss(nodes, members, supports, loads)
    print("classification:", sol["classification"], "  degree:", sol["degree"],
          f"  ({sol['n_unknowns']} unknowns, {sol['n_equations']} eqns, rank {sol['rank']})")
    for mem, F in sol["member_forces"].items():
        print(f"  member {mem}: {F:+.2f} N  ({'tension' if F > 0 else 'compression'})")
    for rd, R in sol["reactions"].items():
        print(f"  reaction {rd}: {R:+.2f} N")
    print("  global equilibrium residual (Fx,Fy,M):",
          tuple(round(v, 9) for v in global_equilibrium_residual(nodes, sol["reactions"], loads)))
    print("  A x = b residual:", f"{sol['residual']:.1e}")

    # counts: add a redundant member -> indeterminate; drop a support -> mechanism
    print("\ndeterminacy by counts:")
    print("  triangle + redundant bar:", static_determinacy(3, 4, 3))
    print("  under-supported:", static_determinacy(3, 2, 2))
