"""rotation_vs_boost.py -- an ordinary rotation and a Lorentz boost are both
"rotations," just in different geometries. A rotation is CIRCULAR
(parametrized by an angle theta, built from cos/sin, preserving the
EUCLIDEAN metric diag(1,1)); a boost is HYPERBOLIC (parametrized by a
rapidity phi, built from cosh/sinh, preserving the MINKOWSKI metric
diag(1,-1)). Same algebraic structure -- both are one-parameter matrix
groups where the parameters ADD under composition -- different invariant.

Reuses dgs/special_relativity.py's four_vector_boost and velocity_addition
for the physical cross-checks (called with c=1.0, natural units, the
standard convention for this kind of matrix work) rather than re-deriving
the boost physics.
"""
from __future__ import annotations
import numpy as np
from typing import Dict

from dgs.special_relativity import four_vector_boost, velocity_addition


def rotation_matrix(theta: float) -> np.ndarray:
    """R(theta) = [[cos, -sin], [sin, cos]] -- the ordinary 2D rotation
    matrix, SO(2)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def lorentz_boost_matrix(beta: float) -> np.ndarray:
    """Lambda(beta) = gamma*[[1, -beta], [-beta, 1]] -- the 2D Lorentz boost
    matrix acting on (ct, x), natural units (c=1). SO(1,1): the hyperbolic
    analog of a rotation, parametrized by beta=v/c (equivalently rapidity
    phi=arctanh(beta))."""
    if abs(beta) >= 1.0:
        raise ValueError(f"beta={beta}: must satisfy |beta| < 1 (c=1 natural units)")
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    return gamma * np.array([[1.0, -beta], [-beta, 1.0]])


def euclidean_metric() -> np.ndarray:
    """diag(1, 1) -- the metric an ordinary rotation preserves."""
    return np.eye(2)


def minkowski_metric_2d() -> np.ndarray:
    """diag(1, -1) -- the metric a Lorentz boost preserves (signature
    convention: timelike +, spacelike -, matching
    dgs.special_relativity.four_vector_boost's ct^2-x^2-y^2-z^2 invariant)."""
    return np.diag([1.0, -1.0])


def preserves_metric(M: np.ndarray, metric: np.ndarray, rtol: float = 1e-10) -> bool:
    """Check M^T @ metric @ M == metric -- the defining property of the
    matrix group that leaves that metric invariant (O(2) for the Euclidean
    metric, O(1,1) for the Minkowski one)."""
    M = np.asarray(M, dtype=float)
    metric = np.asarray(metric, dtype=float)
    return bool(np.allclose(M.T @ metric @ M, metric, rtol=rtol, atol=1e-10))


def rotation_invariant(theta: float, vector: np.ndarray) -> Dict:
    """A rotation preserves Euclidean length |v|^2 = x^2+y^2. Verified
    directly, not asserted: apply R(theta) to `vector` and compare."""
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (2,):
        raise ValueError("vector must have shape (2,)")
    R = rotation_matrix(theta)
    rotated = R @ vector
    return {
        "length_before": float(np.dot(vector, vector)),
        "length_after": float(np.dot(rotated, rotated)),
        "preserves_euclidean_metric": preserves_metric(R, euclidean_metric()),
    }


def boost_invariant(beta: float, ct: float, x: float) -> Dict:
    """A boost preserves the Minkowski interval ct^2-x^2, NOT Euclidean
    length ct^2+x^2. Both computed directly here via explicit 2x2 matrix
    multiplication, AND cross-checked against
    dgs.special_relativity.four_vector_boost's independent scalar-formula
    computation of the same invariant (c=1 natural units) -- two different
    code paths, not one path asserted twice."""
    Lambda = lorentz_boost_matrix(beta)
    v = np.array([ct, x])
    boosted = Lambda @ v
    minkowski_before = ct ** 2 - x ** 2
    minkowski_after = boosted[0] ** 2 - boosted[1] ** 2
    euclidean_before = ct ** 2 + x ** 2
    euclidean_after = boosted[0] ** 2 + boosted[1] ** 2

    cross_check = four_vector_boost([ct, x, 0.0, 0.0], v=beta, c=1.0)

    return {
        "ct_prime": float(boosted[0]), "x_prime": float(boosted[1]),
        "minkowski_before": float(minkowski_before), "minkowski_after": float(minkowski_after),
        "euclidean_before": float(euclidean_before), "euclidean_after": float(euclidean_after),
        "preserves_minkowski_metric": preserves_metric(Lambda, minkowski_metric_2d()),
        "cross_check_invariant_orig": float(cross_check["invariant_orig"]),
        "cross_check_invariant_prime": float(cross_check["invariant_prime"]),
    }


def rotation_composition_check(theta1: float, theta2: float) -> Dict:
    """R(theta1) @ R(theta2) == R(theta1+theta2): angles ADD -- the SO(2)
    group law."""
    composed = rotation_matrix(theta1) @ rotation_matrix(theta2)
    direct = rotation_matrix(theta1 + theta2)
    return {"matches": bool(np.allclose(composed, direct)), "composed": composed, "direct": direct}


def boost_composition_check(beta1: float, beta2: float) -> Dict:
    """Lambda(beta1) @ Lambda(beta2) == Lambda(beta_combined): RAPIDITIES
    add (phi = arctanh(beta)), which is exactly why VELOCITIES don't add
    linearly -- beta_combined is the relativistic velocity-addition formula,
    reused from dgs.special_relativity.velocity_addition (c=1) rather than
    re-derived."""
    composed = lorentz_boost_matrix(beta1) @ lorentz_boost_matrix(beta2)
    beta_combined = velocity_addition(beta1, beta2, c=1.0)["beta"]
    direct = lorentz_boost_matrix(beta_combined)
    phi1, phi2 = np.arctanh(beta1), np.arctanh(beta2)
    beta_from_rapidity_sum = np.tanh(phi1 + phi2)
    return {
        "matches": bool(np.allclose(composed, direct)),
        "beta_combined": float(beta_combined),
        "beta_from_rapidity_sum": float(beta_from_rapidity_sum),
        "rapidities_add": bool(np.isclose(beta_combined, beta_from_rapidity_sum)),
    }


if __name__ == "__main__":
    print("=== 1. Rotation preserves Euclidean length ===")
    r = rotation_invariant(theta=0.7, vector=np.array([3.0, 4.0]))
    print(f"  |v|^2 before={r['length_before']:.4f}  after={r['length_after']:.4f}  "
          f"preserves Euclidean metric: {r['preserves_euclidean_metric']}")

    print("\n=== 2. Boost preserves Minkowski interval, NOT Euclidean length ===")
    b = boost_invariant(beta=0.6, ct=5.0, x=3.0)
    print(f"  Minkowski ct^2-x^2: before={b['minkowski_before']:.4f}  after={b['minkowski_after']:.4f}")
    print(f"  Euclidean ct^2+x^2: before={b['euclidean_before']:.4f}  after={b['euclidean_after']:.4f}  "
          f"(NOT preserved -- this is the point)")
    print(f"  preserves Minkowski metric (matrix check): {b['preserves_minkowski_metric']}")
    print(f"  cross-check vs four_vector_boost: orig={b['cross_check_invariant_orig']:.4f}  "
          f"prime={b['cross_check_invariant_prime']:.4f}")

    print("\n=== 3. Composition: angles add (rotation) vs. rapidities add (boost) ===")
    rc = rotation_composition_check(0.3, 0.5)
    print(f"  R(0.3) @ R(0.5) == R(0.8): {rc['matches']}")
    bc = boost_composition_check(0.3, 0.5)
    print(f"  Lambda(0.3) @ Lambda(0.5) == Lambda(beta_combined={bc['beta_combined']:.4f}): {bc['matches']}")
    print(f"  beta_combined matches tanh(rapidity1+rapidity2): {bc['rapidities_add']}  "
          f"(NOT 0.3+0.5=0.8 -- velocities don't add linearly, rapidities do)")
