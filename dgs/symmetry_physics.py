"""Symmetry in physics: what stays the same tells you how the world must behave.

Two kinds of symmetry, both here:

DISCRETE symmetries -- parity P (mirror, r -> -r), time reversal T (t -> -t), charge
conjugation C (q -> -q) -- sort every vector into POLAR ("true") or AXIAL ("pseudo"):
  * a POLAR vector flips under a mirror: position r, velocity v, momentum p, force F,
    and the ELECTRIC field E.
  * an AXIAL vector does NOT flip under a mirror, because it is a cross product of two
    polar vectors ((-a) x (-b) = a x b): angular momentum L = r x p, torque, and the
    MAGNETIC field B.
That single fact -- E is polar, B is axial -- forces how the fields transform, and the
Lorentz force F = q(E + v x B) is CONSISTENT under P and T precisely because the two
sides transform the same way. (Under P: E->-E but B->+B and v->-v, so v x B -> -v x B,
and the whole force flips like the polar vector it is.)

CONTINUOUS symmetries give conservation laws (Noether's theorem): a system unchanged
under ROTATION conserves angular momentum, and one unchanged under TIME translation
conserves energy. A central force F(r) r-hat has both, so an orbit in it keeps L and E
fixed -- demonstrated here by integrating the orbit and watching both hold to
numerical precision. rotational symmetry -> L; time symmetry -> E.

This is the dgs.even_odd parity operator wearing physics clothes (polar = odd/-1
eigenvector, axial = even/+1), and the symmetry that made dgs.gauss_law solvable.
NumPy only; py-3.13.
"""

import numpy as np


# ----------------------------------------------------------------------
# Discrete symmetry: polar vs axial vectors
# ----------------------------------------------------------------------

# quantity -> (type, parity eigenvalue, time-reversal eigenvalue)
QUANTITIES = {
    "position":         ("polar", -1, +1),
    "velocity":         ("polar", -1, -1),
    "momentum":         ("polar", -1, -1),
    "force":            ("polar", -1, +1),
    "E_field":          ("polar", -1, +1),
    "angular_momentum": ("axial", +1, -1),
    "torque":           ("axial", +1, +1),
    "B_field":          ("axial", +1, -1),
    "magnetic_moment":  ("axial", +1, -1),
}


def parity(vec, axial=False):
    """Apply the parity operation r -> -r to a vector: a POLAR vector flips
    sign, an AXIAL (pseudo) vector does not."""
    vec = np.asarray(vec, float)
    return vec if axial else -vec


def time_reversal(vec, t_odd):
    """Apply t -> -t: quantities built from one time derivative (velocity,
    momentum, angular momentum, B) flip sign (t_odd=True); static ones
    (position, force, E) do not."""
    vec = np.asarray(vec, float)
    return -vec if t_odd else vec


def cross_product_is_axial(a, b):
    """Demonstrate WHY B and L are axial: the cross product of two polar vectors
    is unchanged by parity, because both inputs flip. Returns True if
    (Pa) x (Pb) == a x b (i.e. a x b transforms as axial)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.allclose(np.cross(parity(a), parity(b)), np.cross(a, b)))


def transform_fields(E, B, operation):
    """How the electromagnetic fields transform under a discrete symmetry.
      P (parity):        E -> -E (polar),  B -> +B (axial)
      T (time reversal): E -> +E,          B -> -B (currents reverse)
      C (charge conj.):  E -> -E,          B -> -B (source charge flips)
    Returns (E', B')."""
    E, B = np.asarray(E, float), np.asarray(B, float)
    op = operation.upper()
    if op == "P":
        return -E, B
    if op == "T":
        return E, -B
    if op == "C":
        return -E, -B
    raise ValueError("operation must be 'P', 'T', or 'C'")


def lorentz_force(q, E, v, B):
    """The Lorentz force F = q(E + v x B) -- a polar vector."""
    E, v, B = np.asarray(E, float), np.asarray(v, float), np.asarray(B, float)
    return q * (E + np.cross(v, B))


def lorentz_is_parity_consistent(q, E, v, B):
    """Check that the Lorentz force law respects parity: transforming the inputs
    (v -> -v, E -> -E, B -> +B) gives exactly -F, the way a polar force must
    flip. True confirms electromagnetism is parity-symmetric."""
    F = lorentz_force(q, E, v, B)
    Ep, Bp = transform_fields(E, B, "P")
    F_transformed = lorentz_force(q, Ep, parity(v), Bp)
    return bool(np.allclose(F_transformed, parity(F)))


def lorentz_is_time_reversal_consistent(q, E, v, B):
    """Under T (v -> -v, E -> +E, B -> -B) the Lorentz force is unchanged, as a
    force (F = dp/dt, T-even) must be. True confirms T-symmetry."""
    F = lorentz_force(q, E, v, B)
    Et, Bt = transform_fields(E, B, "T")
    F_transformed = lorentz_force(q, Et, time_reversal(v, t_odd=True), Bt)
    return bool(np.allclose(F_transformed, time_reversal(F, t_odd=False)))


# ----------------------------------------------------------------------
# Continuous symmetry (Noether): rotation -> L, time -> E
# ----------------------------------------------------------------------

def angular_momentum_2d(r, v, m=1.0):
    """L_z = m (x v_y - y v_x): the conserved quantity of ROTATIONAL symmetry."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    return float(m * (r[0] * v[1] - r[1] * v[0]))


def energy_2d(r, v, mu, m=1.0):
    """Total energy 1/2 m v^2 - mu/|r| in an inverse-square central field: the
    conserved quantity of TIME-translation symmetry."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    return float(0.5 * m * np.dot(v, v) - mu / np.linalg.norm(r))


def simulate_central_orbit(r0, v0, mu=1.0, t_end=20.0, dt=1e-3):
    """Integrate an orbit in a central inverse-square force F = -mu r/|r|^3 with
    RK4. Because the force has full rotational and time symmetry, angular
    momentum and energy are conserved -- returned as time series to check.
    Returns dict with trajectory, L(t), E(t), and their fractional drifts."""
    if mu <= 0 or t_end <= 0 or dt <= 0:
        raise ValueError("need mu, t_end, dt > 0")

    def accel(r):
        return -mu * r / np.linalg.norm(r) ** 3

    def deriv(state):
        r, v = state[:2], state[2:]
        return np.concatenate([v, accel(r)])

    n = int(t_end / dt)
    s = np.concatenate([np.asarray(r0, float), np.asarray(v0, float)])
    traj = np.empty((n + 1, 4)); traj[0] = s
    for i in range(n):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i + 1] = s
    L = np.array([angular_momentum_2d(p[:2], p[2:]) for p in traj])
    E = np.array([energy_2d(p[:2], p[2:], mu) for p in traj])
    return {
        "trajectory": traj, "L": L, "E": E,
        "L_drift": float(np.ptp(L) / abs(L[0])),
        "E_drift": float(np.ptp(E) / abs(E[0])),
    }


if __name__ == "__main__":
    print("DISCRETE symmetry -- polar vs axial:")
    print("  cross of two polar vectors is axial?",
          cross_product_is_axial([1, 2, 3], [4, 5, 6]))
    for name, (typ, P, T) in QUANTITIES.items():
        print(f"    {name:18s} {typ:5s}  P={P:+d}  T={T:+d}")

    print("\n  Lorentz force F = q(E + v x B):")
    q, E, v, B = 1.5, [1, 0, 0], [0, 2, 0], [0, 0, 3]
    print("    parity-consistent?", lorentz_is_parity_consistent(q, E, v, B))
    print("    time-reversal-consistent?", lorentz_is_time_reversal_consistent(q, E, v, B))

    print("\nCONTINUOUS symmetry (Noether) -- a central-force orbit:")
    orb = simulate_central_orbit([1.0, 0.0], [0.0, 1.2], mu=1.0, t_end=20.0)
    print(f"  rotational symmetry -> L conserved: drift {orb['L_drift']:.2e}")
    print(f"  time symmetry       -> E conserved: drift {orb['E_drift']:.2e}")
