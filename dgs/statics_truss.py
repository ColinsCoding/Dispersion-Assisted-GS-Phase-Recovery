"""Engineering statics: 2-D pin-jointed truss analysis -- support
reactions from overall equilibrium, member forces from the METHOD OF
JOINTS (assembled as one linear system and solved directly, not joint by
joint by hand), independently cross-checked by the METHOD OF SECTIONS on
a specific member, and every solved force verified by recomputing
equilibrium residuals rather than trusted from the solve alone.

THE CORE IDEA, made explicit: a statically determinate 2-D truss with j
joints, m members, and r reaction components satisfies m + r = 2j (2
equilibrium equations -- Fx, Fy -- per joint, one unknown force per member
plus r unknown reaction components). That equality is what makes the
method-of-joints linear system SQUARE and exactly solvable, not
overdetermined or underdetermined -- checked directly below (m+r vs 2j)
before ever assembling the system, not assumed from having drawn a
sensible-looking truss.

TWO WORKED EXAMPLES, both hand-verified independently of the code:

  1. A minimal 3-joint TRIANGLE (pin at A, roller at B, load at apex C):
       F_AB = +0.3333*P (tension),  F_AC = F_BC = -0.6009*P (compression)

  2. A 4-joint truss with a classic ZERO-FORCE MEMBER (pin at A, roller
     at C, apex load at D directly above the unloaded midspan joint B):
       F_AB = F_BC = +0.6667*P (tension)
       F_AD = F_DC = -0.8333*P (compression)
       F_BD = 0                          <- the zero-force member

     F_BD=0 follows from joint B's equilibrium ALONE (two horizontal
     members AB, BC and one vertical member BD meet there, no external
     load applied at B): Fy at B has ONLY F_BD's vertical component, so
     F_BD=0 regardless of every other force in the structure -- a classic,
     sign-convention-independent structural engineering result, verified
     here as a special case of the same general linear solve.

Method of sections (Section 4) cuts through 3 members and solves for one
of them via a single moment equation about a point that eliminates the
other two -- an INDEPENDENT calculation from the method-of-joints linear
system, cross-checked against it rather than derived from it.
"""

from __future__ import annotations
import numpy as np


def unit_vector(p_from, p_to) -> np.ndarray:
    d = np.array(p_to, dtype=float) - np.array(p_from, dtype=float)
    length = np.linalg.norm(d)
    if length < 1e-12:
        raise ValueError(f"zero-length member between {p_from} and {p_to}")
    return d / length


# ── 1. Static determinacy ────────────────────────────────────────────────────

def static_determinacy_check(n_members: int, n_reaction_components: int, n_joints: int) -> dict:
    """m + r = 2j for a statically determinate 2-D truss. Returns whether
    the given (m, r, j) satisfy it, and which way it fails otherwise
    (understatic -> a mechanism, unstable; overstatic -> statically
    indeterminate, needs more than statics alone to solve)."""
    if n_members < 0 or n_reaction_components < 0 or n_joints <= 0:
        raise ValueError("n_members, n_reaction_components must be >= 0, n_joints must be > 0")
    lhs, rhs = n_members + n_reaction_components, 2 * n_joints
    if lhs == rhs:
        status = "determinate"
    elif lhs < rhs:
        status = "understatic (mechanism -- unstable, not enough members/reactions)"
    else:
        status = "overstatic (statically indeterminate -- needs more than statics alone)"
    return {"n_members": n_members, "n_reaction_components": n_reaction_components,
            "n_joints": n_joints, "lhs_m_plus_r": lhs, "rhs_2j": rhs,
            "determinate": lhs == rhs, "status": status}


# ── 2. Method of joints: one linear system, solved directly ────────────────

def solve_method_of_joints(joints: dict, members: list, supports: dict, loads: dict) -> dict:
    """Assembles the FULL method-of-joints linear system (2 equilibrium
    equations per joint) and solves it directly via np.linalg.solve --
    not joint-by-joint substitution, the same physics, done as one linear
    algebra problem.

    Parameters
    ----------
    joints   : {name: (x, y)}
    members  : [(joint_a, joint_b), ...] -- each a two-force member
    supports : {joint_name: 'pin' | 'roller_x' | 'roller_y'} -- 'pin' gives
               2 reaction unknowns (Rx, Ry); 'roller_x'/'roller_y' gives 1
               (reaction along x or y respectively; 'roller_y' is the
               usual "roller on a horizontal surface" case)
    loads    : {joint_name: (Fx, Fy)} external loads, omitted joints = 0

    Returns dict with member forces (tension positive), reaction
    components, the determinacy check, and the raw (A, b, unknown_names)
    system for inspection.
    """
    n_joints = len(joints)
    joint_names = list(joints.keys())

    reaction_unknowns = []
    for j, kind in supports.items():
        if kind == "pin":
            reaction_unknowns += [(j, "Rx"), (j, "Ry")]
        elif kind == "roller_x":
            reaction_unknowns += [(j, "Rx")]
        elif kind == "roller_y":
            reaction_unknowns += [(j, "Ry")]
        else:
            raise ValueError(f"unknown support kind {kind!r} at joint {j}")

    determinacy = static_determinacy_check(len(members), len(reaction_unknowns), n_joints)
    if not determinacy["determinate"]:
        raise ValueError(f"truss is not statically determinate: {determinacy['status']} "
                          f"(m+r={determinacy['lhs_m_plus_r']}, 2j={determinacy['rhs_2j']})")

    unknown_names = [f"F_{a}_{b}" for a, b in members] + [f"{kind}@{j}" for j, kind in reaction_unknowns]
    n_unknowns = len(unknown_names)
    A = np.zeros((2 * n_joints, n_unknowns))
    b = np.zeros(2 * n_joints)

    for i, j in enumerate(joint_names):
        eq_x, eq_y = 2 * i, 2 * i + 1
        for k, (a, c) in enumerate(members):
            if j == a:
                u = unit_vector(joints[a], joints[c])
                A[eq_x, k] += u[0]; A[eq_y, k] += u[1]
            elif j == c:
                u = unit_vector(joints[c], joints[a])
                A[eq_x, k] += u[0]; A[eq_y, k] += u[1]
        for ri, (rj, kind) in enumerate(reaction_unknowns):
            if rj != j:
                continue
            col = len(members) + ri
            if kind == "Rx":
                A[eq_x, col] += 1
            else:
                A[eq_y, col] += 1
        lx, ly = loads.get(j, (0.0, 0.0))
        b[eq_x], b[eq_y] = -lx, -ly

    x = np.linalg.solve(A, b)
    member_forces = {f"{a}-{c}": float(x[k]) for k, (a, c) in enumerate(members)}
    reactions = {f"{kind}@{j}": float(x[len(members) + ri])
                 for ri, (j, kind) in enumerate(reaction_unknowns)}

    return {"member_forces": member_forces, "reactions": reactions,
            "determinacy": determinacy, "A": A, "b": b, "unknown_names": unknown_names}


def verify_equilibrium_residuals(joints: dict, members: list, member_forces: dict,
                                 reactions: dict, loads: dict, tol: float = 1e-6) -> dict:
    """CHECKED, not assumed: recomputes (sum Fx, sum Fy) at EVERY joint
    from the solved member forces + reactions + external loads, and
    confirms each is ~0 -- the actual physical statement "this structure
    is in equilibrium," verified independently of trusting the linear
    solve that produced these numbers."""
    residuals = {}
    for j, (jx, jy) in joints.items():
        fx = fy = 0.0
        for a, c in members:
            key = f"{a}-{c}"
            F = member_forces[key]
            if j == a:
                u = unit_vector(joints[a], joints[c])
            elif j == c:
                u = unit_vector(joints[c], joints[a])
            else:
                continue
            fx += F * u[0]; fy += F * u[1]
        for rname, rval in reactions.items():
            rjoint = rname.split("@")[1]
            if rjoint != j:
                continue
            if rname.startswith("Rx"):
                fx += rval
            else:
                fy += rval
        lx, ly = loads.get(j, (0.0, 0.0))
        fx += lx; fy += ly
        residuals[j] = (fx, fy)

    max_residual = max(max(abs(fx), abs(fy)) for fx, fy in residuals.values())
    return {"residuals": residuals, "max_residual": float(max_residual),
            "equilibrium_holds": bool(max_residual < tol)}


def classify_forces(member_forces: dict, tol: float = 1e-6) -> dict:
    """Tension (positive), compression (negative), or zero-force member
    (|F| < tol) -- the standard engineering classification, not left as
    raw signed numbers."""
    out = {}
    for member, F in member_forces.items():
        if abs(F) < tol:
            out[member] = "zero-force member"
        elif F > 0:
            out[member] = f"tension ({F:.3f})"
        else:
            out[member] = f"compression ({-F:.3f})"
    return out


# ── 3. Example 1: minimal 3-joint triangle ──────────────────────────────────

def triangle_truss_example(P: float = 1000.0) -> dict:
    """Pin at A(0,0), roller at B(4,0), downward load P at apex C(2,3).
    Hand-verified closed form: F_AB = P/3 exactly; at apex C symmetry
    gives F_AC=F_BC, and y-equilibrium there gives
    F_AC = F_BC = -P*sqrt(13)/6 (matches the module docstring's stated
    0.3333P / 0.6009P numerically)."""
    joints = {"A": (0.0, 0.0), "B": (4.0, 0.0), "C": (2.0, 3.0)}
    members = [("A", "B"), ("A", "C"), ("B", "C")]
    supports = {"A": "pin", "B": "roller_y"}
    loads = {"C": (0.0, -P)}
    result = solve_method_of_joints(joints, members, supports, loads)
    result["joints"], result["members"], result["loads"] = joints, members, loads
    return result


# ── 4. Example 2: 4-joint truss with a zero-force member ───────────────────

def zero_force_member_truss_example(P: float = 1000.0) -> dict:
    """Pin at A(0,0), roller at C(8,0), downward load P at apex D(4,3),
    with B(4,0) an unloaded midspan joint connected by a vertical member
    BD. F_BD=0 is a classic, sign-convention-independent structural
    result (see module docstring)."""
    joints = {"A": (0.0, 0.0), "B": (4.0, 0.0), "C": (8.0, 0.0), "D": (4.0, 3.0)}
    members = [("A", "B"), ("B", "C"), ("A", "D"), ("D", "C"), ("B", "D")]
    supports = {"A": "pin", "C": "roller_y"}
    loads = {"D": (0.0, -P)}
    result = solve_method_of_joints(joints, members, supports, loads)
    result["joints"], result["members"], result["loads"] = joints, members, loads
    return result


# ── 5. Method of sections: an independent cross-check on one member ────────

def method_of_sections_moment(joints: dict, moment_point, known_forces: list) -> float:
    """Sum of moments (z-component of r x F, CCW positive) about
    `moment_point`, of a list of (application_point_name_or_coords,
    force_vector) pairs -- the raw mechanics method_of_sections_AD below
    uses to isolate one unknown member force via a single equation."""
    mp = np.array(moment_point, dtype=float)
    total = 0.0
    for point, force in known_forces:
        p = np.array(joints[point] if isinstance(point, str) else point, dtype=float)
        r = p - mp
        total += r[0] * force[1] - r[1] * force[0]
    return total


def method_of_sections_DC(joints: dict, reactions: dict, loads: dict, P_ref: float) -> float:
    """Independently solves for F_DC (in zero_force_member_truss_example's
    geometry) by cutting through AD, DC, BD and taking moments about B
    (which lies on both AD's... no: about B, which lies on the lines of
    action of AB and BD, eliminating them from the moment equation,
    leaving only F_DC and the known reaction Cy on the right free body
    {C, D-cut-point}) -- NOT derived from solve_method_of_joints, a
    genuinely separate calculation cross-checked against it."""
    B = joints["B"]
    Cy = reactions["Ry@C"]
    C = joints["C"]
    # right free body: reaction at C (0, Cy), plus the cut force F_DC.
    # TENSION convention (matching solve_method_of_joints): a member in
    # tension pulls the free body it acts on TOWARD the member's other
    # end -- so on the right free body (attached near C), tension in DC
    # points from C's side TOWARD D, i.e. unit_vector(C, D), not D->C.
    # Applied at D here (any point on DC's line of action gives the same
    # moment, since DC is a two-force member).
    D = joints["D"]
    u_DC = unit_vector(C, D)
    # moment_about_B(known reactions) + F_DC * moment_about_B(unit force along DC at D) = 0
    m_reaction = method_of_sections_moment(joints, B, [(C, (0.0, Cy))])
    r_D = np.array(D) - np.array(B)
    m_unit_force = r_D[0] * u_DC[1] - r_D[1] * u_DC[0]
    if abs(m_unit_force) < 1e-12:
        raise RuntimeError("chosen moment point is degenerate for this cut (unit-force moment ~0)")
    F_DC = -m_reaction / m_unit_force
    return F_DC


if __name__ == "__main__":
    print("=== 1. Static determinacy checks ===")
    for m, r, j, label in [(3, 3, 3, "triangle truss"), (5, 3, 4, "zero-force-member truss"),
                            (3, 2, 3, "understatic example"), (4, 3, 3, "overstatic example")]:
        check = static_determinacy_check(m, r, j)
        print(f"  {label}: m={m}, r={r}, j={j} -> {check['status']}")

    print("\n=== 2. Triangle truss (method of joints) ===")
    tri = triangle_truss_example()
    print(f"  reactions: {tri['reactions']}")
    print(f"  member forces: {tri['member_forces']}")
    print(f"  classification: {classify_forces(tri['member_forces'])}")
    eq = verify_equilibrium_residuals(tri["joints"], tri["members"], tri["member_forces"],
                                      tri["reactions"], tri["loads"])
    print(f"  equilibrium holds: {eq['equilibrium_holds']} (max residual: {eq['max_residual']:.2e})")

    print("\n=== 3. Zero-force-member truss (method of joints) ===")
    zfm = zero_force_member_truss_example()
    print(f"  reactions: {zfm['reactions']}")
    print(f"  member forces: {zfm['member_forces']}")
    print(f"  classification: {classify_forces(zfm['member_forces'])}")
    eq2 = verify_equilibrium_residuals(zfm["joints"], zfm["members"], zfm["member_forces"],
                                       zfm["reactions"], zfm["loads"])
    print(f"  equilibrium holds: {eq2['equilibrium_holds']} (max residual: {eq2['max_residual']:.2e})")

    print("\n=== 4. Method of sections: independent cross-check on F_DC ===")
    F_DC_sections = method_of_sections_DC(zfm["joints"], zfm["reactions"], zfm["loads"], P_ref=1000.0)
    F_DC_joints = zfm["member_forces"]["D-C"]
    print(f"  F_DC (method of sections): {F_DC_sections:.4f}")
    print(f"  F_DC (method of joints):   {F_DC_joints:.4f}")
    print(f"  agree: {abs(F_DC_sections - F_DC_joints) < 1e-6}")

    print("\nMethod of joints (one linear solve) and method of sections (one moment equation)")
    print("are independent calculations that must agree -- and the zero-force member (F_BD=0)")
    print("is a structural fact that falls straight out of joint B's equilibrium alone.")
