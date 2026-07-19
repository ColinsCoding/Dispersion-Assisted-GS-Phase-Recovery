"""Differentiable 2D truss finite-element solver (direct stiffness method), in
torch -- the "Poly Bridge" physics. Given member cross-sectional areas, joint
loads, and supports, compute nodal DISPLACEMENTS and member axial forces under
load, then autograd gives sensitivities (d(deflection)/d(area)) for free.

This generalizes dgs.statics.truss_method_of_joints (pure statics -- forces
only, determinate trusses only) to the direct stiffness method: works for ANY
truss (determinate or indeterminate) and, because every operation is a torch
tensor op, torch.autograd.grad differentiates through the whole solve with no
hand-derived adjoint formula -- unlike dgs.statics.autograd_truss_sensitivity,
which hand-codes dU/dA_i = -F_i^2 L_i / (2 E A_i^2) as a closed form. Same
answer, but here the chain rule does the work.

Element stiffness (local axial stiffness EA, transformed to global 2D via
direction cosines cx, cy):
    k_global = (EA/L) * [[cx^2, cx*cy, -cx^2, -cx*cy],
                          [cx*cy, cy^2, -cx*cy, -cy^2],
                          [-cx^2, -cx*cy, cx^2, cx*cy],
                          [-cx*cy, -cy^2, cx*cy, cy^2]]
Assembled into a global K, partitioned by fixed/free DOFs, solved K_ff u_f = F_f.

bridge_load_sweep() steps a load from 0 up to a target over n_steps and
export_trajectory_json() writes the per-step deformed geometry + member
forces to disk for external visualization (see unreal/import_truss_sim.py,
which reads this JSON inside Unreal Engine 5.5).

torch is py-3.12 ONLY in this repo -- run this module with `py -3.12`.
"""

import json

import torch


def element_stiffness_global(x1, y1, x2, y2, EA):
    """4x4 global stiffness matrix for one 2-node truss element -- all torch
    tensors, differentiable w.r.t. EA and the joint coordinates."""
    dx, dy = x2 - x1, y2 - y1
    L = torch.sqrt(dx ** 2 + dy ** 2)
    cx, cy = dx / L, dy / L
    c2, s2, cs = cx * cx, cy * cy, cx * cy
    k = (EA / L) * torch.stack([
        torch.stack([c2, cs, -c2, -cs]),
        torch.stack([cs, s2, -cs, -s2]),
        torch.stack([-c2, -cs, c2, cs]),
        torch.stack([-cs, -s2, cs, s2]),
    ])
    return k, L


def assemble_global_stiffness(coords, members, EA):
    """Scatter every member's 4x4 element stiffness into the global
    (2*n_joints)x(2*n_joints) stiffness matrix (2 DOF per joint: ux, uy)."""
    n_joints = coords.shape[0]
    ndof = 2 * n_joints
    K = torch.zeros((ndof, ndof), dtype=coords.dtype, device=coords.device)
    lengths = []
    for m_idx, (i, j) in enumerate(members):
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        k_elem, L = element_stiffness_global(x1, y1, x2, y2, EA[m_idx])
        lengths.append(L)
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] = K[dofs[a], dofs[b]] + k_elem[a, b]
    return K, torch.stack(lengths)


def _fixed_dofs(supports):
    """Support-string convention matches dgs.statics.truss_method_of_joints
    exactly ('pin', 'roller'/'roller_y', 'roller_x') so the same support dict
    can cross-validate both solvers."""
    fixed = set()
    for j, sup in supports.items():
        if sup == "pin":
            fixed.add(2 * j)
            fixed.add(2 * j + 1)
        elif sup in ("roller", "roller_y"):
            fixed.add(2 * j + 1)
        elif sup == "roller_x":
            fixed.add(2 * j)
        else:
            raise ValueError(f"unrecognized support {sup!r} at joint {j}")
    return fixed


def solve_truss_fem(coords, members, EA, supports, loads):
    """Solve for nodal displacements and member axial forces under load via
    the direct stiffness method.

    coords: (n_joints, 2) torch tensor of joint positions.
    members: list of (i, j) joint-index pairs.
    EA: (n_members,) torch tensor of axial stiffness per member (E * area;
        set requires_grad=True on this to get sensitivities via autograd).
    supports: {joint_index: 'pin'|'roller'|'roller_x'} (see _fixed_dofs).
    loads: {joint_index: (Fx, Fy)} applied nodal loads.
    """
    n_joints = coords.shape[0]
    ndof = 2 * n_joints
    K, lengths = assemble_global_stiffness(coords, members, EA)

    fixed = _fixed_dofs(supports)
    free = [d for d in range(ndof) if d not in fixed]
    if not free:
        raise ValueError("truss has no free DOFs -- fully constrained")

    F = torch.zeros(ndof, dtype=coords.dtype, device=coords.device)
    for j, (Fx, Fy) in loads.items():
        F[2 * j] = F[2 * j] + Fx
        F[2 * j + 1] = F[2 * j + 1] + Fy

    free_idx = torch.tensor(free, dtype=torch.long, device=coords.device)
    K_ff = K[free_idx][:, free_idx]
    F_f = F[free_idx]

    u_f = torch.linalg.solve(K_ff, F_f)

    u = torch.zeros(ndof, dtype=coords.dtype, device=coords.device)
    u = u.index_copy(0, free_idx, u_f)
    displacements = u.reshape(n_joints, 2)
    deformed_coords = coords + displacements

    member_forces = []
    for m_idx, (i, j) in enumerate(members):
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        dx, dy = x2 - x1, y2 - y1
        L = lengths[m_idx]
        cx, cy = dx / L, dy / L
        u_i, u_j = displacements[i], displacements[j]
        elongation = (u_j[0] - u_i[0]) * cx + (u_j[1] - u_i[1]) * cy
        member_forces.append(EA[m_idx] / L * elongation)
    member_forces = torch.stack(member_forces)

    return {
        "displacements": displacements,
        "deformed_coords": deformed_coords,
        "member_forces": member_forces,
        "lengths": lengths,
    }


def bridge_load_sweep(coords, members, EA, supports, load_joint, load_target, n_steps=10):
    """Ramp a single downward point load at load_joint from 0 to load_target
    (Fy, negative = downward) over n_steps, solving the truss fresh at each
    step -- the "Poly Bridge" load-test animation: watch the deck sag as
    weight is added. Returns a list of n_steps+1 solve_truss_fem() results
    (step 0 = unloaded)."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    results = []
    for step in range(n_steps + 1):
        frac = step / n_steps
        loads = {load_joint: (0.0, load_target * frac)}
        results.append(solve_truss_fem(coords, members, EA, supports, loads))
    return results


def export_trajectory_json(path, coords, members, results_per_step, deflection_scale=1.0):
    """Write the per-load-step deformed geometry + member axial forces to
    JSON, for a viewer outside Python (see unreal/import_truss_sim.py) to
    animate. deflection_scale exaggerates displacement for visibility -- real
    structural deflections are tiny (mm on a meter-scale span) and are
    invisible at 1:1 scale in any 3D viewer, the same reason CAD/FEA tools
    default to an exaggerated "deformed shape" display."""
    coords0 = coords.detach().cpu().numpy().tolist()
    steps = []
    for res in results_per_step:
        deformed = coords.detach() + deflection_scale * res["displacements"].detach()
        steps.append({
            "joint_positions": deformed.cpu().numpy().tolist(),
            "member_forces": res["member_forces"].detach().cpu().numpy().tolist(),
        })
    data = {
        "undeformed_joint_positions": coords0,
        "members": [list(m) for m in members],
        "deflection_scale": deflection_scale,
        "steps": steps,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


if __name__ == "__main__":
    dtype = torch.float64
    # symmetric A-frame: pin at A, roller at B, apex C -- same geometry as
    # dgs.statics.three_bar_truss(), scaled to meters instead of the original's
    # (0,0)/(2,0)/(1,1.5) so member forces are directly comparable
    coords = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]], dtype=dtype)
    members = [(0, 2), (1, 2), (0, 1)]
    EA = torch.tensor([2.0e5, 2.0e5, 2.0e5], dtype=dtype, requires_grad=True)
    supports = {0: "pin", 1: "roller"}

    result = solve_truss_fem(coords, members, EA, supports, loads={2: (0.0, -10000.0)})
    print("member forces (N, tension +):", result["member_forces"].tolist())
    print("apex (C) displacement (m):", result["displacements"][2].tolist())

    # autograd sensitivity: no hand-derived adjoint, just backprop
    max_deflection = torch.norm(result["displacements"][2])
    max_deflection.backward()
    print("\nd(|apex deflection|)/d(EA) via autograd:", EA.grad.tolist())

    print("\n--- load sweep (bridge sag animation) ---")
    EA_ns = EA.detach()
    steps = bridge_load_sweep(coords, members, EA_ns, supports, load_joint=2,
                               load_target=-10000.0, n_steps=5)
    for i, res in enumerate(steps):
        print(f"  step {i}: apex y-displacement = {res['displacements'][2, 1].item():.5f} m")
