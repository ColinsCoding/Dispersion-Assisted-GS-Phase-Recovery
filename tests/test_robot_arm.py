"""Test dgs.robot_arm: forward kinematics at known poses, the analytic Jacobian vs
finite differences, x_dot = J theta_dot, the det(J)=0 straight-arm singularity, the
workspace/reach, and inverse kinematics landing on reachable targets while rejecting
unreachable ones."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import robot_arm as ra

L2 = [1.0, 1.0]
L3 = [1.0, 1.0, 1.0]

# 1. forward kinematics at known poses
assert np.allclose(ra.forward_kinematics([0, 0], L2), [2, 0])            # straight out
assert np.allclose(ra.forward_kinematics([np.pi/2, 0], L2), [0, 2])      # straight up
assert np.allclose(ra.forward_kinematics([np.pi/2, -np.pi/2], L2), [1, 1])
assert np.allclose(ra.forward_kinematics([0, 0, 0], L3), [3, 0])         # 3-link straight

# 2. joint_positions: base at origin, N+1 points, last is the hand
jp = ra.joint_positions([np.pi/2, 0], L2)
assert jp.shape == (3, 2)
assert np.allclose(jp[0], [0, 0]) and np.allclose(jp[-1], ra.forward_kinematics([np.pi/2, 0], L2))

# 3. analytic Jacobian matches the finite-difference one (2- and 3-link, several poses)
for L, th in [(L2, [np.pi/4, np.pi/4]), (L2, [0.3, -0.7]),
              (L3, [0.2, 0.5, -0.4]), (L3, [1.0, 1.0, 1.0])]:
    assert np.allclose(ra.jacobian(th, L), ra.jacobian_numeric(th, L), atol=1e-6)
    assert ra.jacobian(th, L).shape == (2, len(L))

# 4. end-effector velocity = J theta_dot
th = [np.pi/4, np.pi/4]
rates = [1.0, -0.5]
assert np.allclose(ra.end_effector_velocity(th, L2, rates), ra.jacobian(th, L2) @ rates)

# 5. reach and workspace
assert ra.reach(L2) == 2.0
assert ra.is_reachable([1, 1], L2) and not ra.is_reachable([5, 5], L2)
# an inner "hole": links 1 and 3 can't fold closer than |3-1| = 2
assert not ra.is_reachable([1, 0], [1.0, 3.0])
assert ra.is_reachable([2.5, 0], [1.0, 3.0])

# 6. singularity: det(J) = L1 L2 sin(theta2) -> 0 when the arm is straight
assert np.isclose(np.linalg.det(ra.jacobian([0.3, 0.0], L2)), 0.0, atol=1e-12)
assert abs(np.linalg.det(ra.jacobian([0.3, 0.5], L2))) > 0.1        # bent -> invertible
assert np.isclose(np.linalg.det(ra.jacobian([0.3, 0.5], L2)),
                  1.0 * 1.0 * np.sin(0.5))                          # = L1 L2 sin(theta2)

# 7. inverse kinematics: reachable targets recovered, unreachable rejected
for target in ([1.0, 1.0], [1.5, 0.3], [-0.5, 1.2], [0.2, -1.8]):
    sol = ra.inverse_kinematics(target, L2)
    assert sol["converged"]
    assert np.allclose(ra.forward_kinematics(sol["angles"], L2), target, atol=1e-6)
assert not ra.inverse_kinematics([5, 5], L2)["converged"]          # out of reach
assert not ra.inverse_kinematics([3, 3], L2)["converged"]

# 8. kwarg bounds
for bad in (lambda: ra.forward_kinematics([0, 0], [1.0]),          # mismatched
            lambda: ra.jacobian([0.1], [-1.0]),                    # bad length
            lambda: ra.forward_kinematics([], []),
            lambda: ra.inverse_kinematics([1, 1], L2, angles0=[0, 0, 0])):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_robot_arm: all checks passed")
