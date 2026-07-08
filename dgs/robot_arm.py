"""A planar robot arm: forward kinematics, the Jacobian, and inverse kinematics.

Wire N rigid links end to end with a motor at each joint (angles theta_1..theta_N,
lengths L_1..L_N). Three questions, and the Jacobian ties them together:

  FORWARD KINEMATICS -- where is the hand, given the joint angles? Each link points at
  the CUMULATIVE angle phi_i = theta_1 + ... + theta_i, so the end effector is
        x = sum L_i cos(phi_i),   y = sum L_i sin(phi_i).

  THE JACOBIAN (the "diff") -- how does the hand MOVE when the joints turn? Differentiate:
        J = d(x,y)/d(theta),   and   (x_dot, y_dot) = J theta_dot.
  J is the differential map from joint velocities to hand velocity -- the same object as
  the geometry matrix in dgs.gnss_positioning and the design matrix in dgs.statics_linalg.
  Where det(J) = 0 the arm is SINGULAR (fully stretched/folded): some hand direction
  becomes unreachable no matter how the joints spin.

  INVERSE KINEMATICS -- what joint angles put the hand on a target? Invert the diff:
  iterate theta <- theta + J^+ (target - forward_kinematics(theta)) with the Jacobian
  pseudoinverse J^+ (Newton on the kinematics) -- the same Gauss-Newton loop that solved
  for a receiver position from pseudoranges.

Verified: forward kinematics at known poses, the analytic Jacobian against a finite-
difference one, x_dot = J theta_dot, inverse kinematics landing on reachable targets (and
failing on unreachable ones), and the det(J)=0 singularity when the arm is straight.
Sibling of dgs.ptz_camera (a 2-DOF pointing arm). NumPy only; py-3.13.
"""

import numpy as np


def _check(angles, lengths):
    a = np.asarray(angles, float)
    L = np.asarray(lengths, float)
    if a.shape != L.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("angles and lengths must be 1-D arrays of equal length >= 1")
    if np.any(L <= 0):
        raise ValueError("link lengths must be positive")
    return a, L


def joint_positions(angles, lengths):
    """The (x,y) of every joint from the base (0,0) to the hand -- the arm's shape.
    Returns an (N+1, 2) array; the last row is the end effector."""
    a, L = _check(angles, lengths)
    phi = np.cumsum(a)                       # cumulative angle of each link
    xs = np.concatenate([[0.0], np.cumsum(L * np.cos(phi))])
    ys = np.concatenate([[0.0], np.cumsum(L * np.sin(phi))])
    return np.column_stack([xs, ys])


def forward_kinematics(angles, lengths):
    """End-effector position (x, y) for the given joint angles."""
    return joint_positions(angles, lengths)[-1]


def jacobian(angles, lengths):
    """Analytic 2xN Jacobian J = d(x,y)/d(theta). Column j is how the hand moves per
    unit turn of joint j: only links at or beyond j rotate about it, so
        dx/dtheta_j = -sum_{i>=j} L_i sin(phi_i),   dy/dtheta_j = sum_{i>=j} L_i cos(phi_i)."""
    a, L = _check(angles, lengths)
    phi = np.cumsum(a)
    n = len(a)
    J = np.zeros((2, n))
    for j in range(n):
        J[0, j] = -np.sum(L[j:] * np.sin(phi[j:]))
        J[1, j] = np.sum(L[j:] * np.cos(phi[j:]))
    return J


def jacobian_numeric(angles, lengths, h=1e-6):
    """Finite-difference Jacobian -- an independent check of the analytic one."""
    a, L = _check(angles, lengths)
    n = len(a)
    J = np.zeros((2, n))
    for j in range(n):
        da = a.copy(); da[j] += h
        db = a.copy(); db[j] -= h
        J[:, j] = (forward_kinematics(da, L) - forward_kinematics(db, L)) / (2 * h)
    return J


def end_effector_velocity(angles, lengths, angle_rates):
    """Hand velocity (x_dot, y_dot) = J theta_dot from the joint angular rates."""
    rates = np.asarray(angle_rates, float)
    return jacobian(angles, lengths) @ rates


def reach(lengths):
    """Maximum reach = sum of the link lengths (arm fully extended)."""
    L = np.asarray(lengths, float)
    if np.any(L <= 0):
        raise ValueError("link lengths must be positive")
    return float(np.sum(L))


def is_reachable(target, lengths):
    """Whether a target lies in the workspace: between the fully-folded inner radius
    and the fully-extended outer radius."""
    L = np.asarray(lengths, float)
    r = np.hypot(*np.asarray(target, float))
    inner = max(0.0, 2 * np.max(L) - np.sum(L))   # can't reach closer than this
    return inner - 1e-9 <= r <= np.sum(L) + 1e-9


def inverse_kinematics(target, lengths, angles0=None, max_iter=500, tol=1e-9):
    """Solve for joint angles that put the hand on `target`, by Newton iteration on
    the kinematics with the Jacobian pseudoinverse:
        theta <- theta + J^+ (target - forward_kinematics(theta)).
    Returns dict with angles, converged, error, iterations. Reachable, non-singular
    targets converge fast; unreachable ones do not (converged=False)."""
    L = np.asarray(lengths, float)
    target = np.asarray(target, float)
    theta = np.zeros(len(L)) if angles0 is None else np.array(angles0, float)
    if len(theta) != len(L):
        raise ValueError("angles0 must match the number of links")
    it = 0
    for it in range(1, max_iter + 1):
        err = target - forward_kinematics(theta, L)
        if np.linalg.norm(err) < tol:
            return {"angles": theta, "converged": True,
                    "error": float(np.linalg.norm(err)), "iterations": it}
        theta = theta + np.linalg.pinv(jacobian(theta, L)) @ err
    e = float(np.linalg.norm(target - forward_kinematics(theta, L)))
    return {"angles": theta, "converged": e < tol, "error": e, "iterations": it}


if __name__ == "__main__":
    L = [1.0, 1.0]
    print("2-link arm (L=1,1), forward kinematics:")
    for ang in ([0, 0], [np.pi/2, 0], [np.pi/2, -np.pi/2]):
        print(f"  theta={np.round(ang,3)} -> hand {np.round(forward_kinematics(ang, L), 3)}")

    print("\nJacobian at theta=(pi/4, pi/4):")
    th = [np.pi/4, np.pi/4]
    print("  analytic:\n", np.round(jacobian(th, L), 4))
    print("  numeric matches?", np.allclose(jacobian(th, L), jacobian_numeric(th, L)))

    print("\nsingularity (arm straight, theta2=0): det(J) =",
          round(float(np.linalg.det(jacobian([0.3, 0.0], L))), 6), "(-> 0)")

    print("\ninverse kinematics to (1.0, 1.0):")
    sol = inverse_kinematics([1.0, 1.0], L)
    print(f"  angles = {np.round(np.degrees(sol['angles']), 2)} deg, "
          f"converged={sol['converged']}, error={sol['error']:.1e}, iters={sol['iterations']}")
    print(f"  FK check -> {np.round(forward_kinematics(sol['angles'], L), 6)}")
    print(f"  unreachable (5,5)? reachable={is_reachable([5,5], L)}, "
          f"IK converged={inverse_kinematics([5,5], L)['converged']}")
