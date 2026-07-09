"""Free body diagrams: draw the forces on ONE body, then set the sums to zero.

The whole method of statics is: isolate a body, mark every force on it (with WHERE it acts),
and demand that it neither accelerates nor spins -- three scalar equations in 2-D:
        sum F_x = 0,   sum F_y = 0,   sum M = 0   (moments about any point).
The applied loads are known; the SUPPORT REACTIONS are the unknowns, and those three equations
solve for them exactly when the body is statically determinate (three unknown reaction
components). A pin gives two (F_x, F_y), a roller one (normal), a smooth wall one -- so a beam on
a pin + roller, or a ladder on a floor + wall, is solvable.

A force here is ((F_x, F_y), (x, y)): a vector and the point it acts at. The moment it makes about
the origin is the z-component of r x F, x*F_y - y*F_x. Any force works, mechanical or
electromagnetic (a charge's Lorentz force is just another arrow on the diagram).

This is the single-rigid-body companion to dgs.statics_linalg (which assembles a whole truss) and
dgs.weld_statics (one loaded joint). Verified against hand-worked classics: a simply supported
beam (reactions from moment balance) and a ladder against a smooth wall (minimum friction
1/(2 tan theta)). NumPy only; py-3.13.
"""

import numpy as np


def net_force(forces):
    """Sum of the force vectors: (sum F_x, sum F_y). Zero at translational equilibrium."""
    if not forces:
        raise ValueError("need at least one force")
    Fx = sum(f[0][0] for f in forces)
    Fy = sum(f[0][1] for f in forces)
    return np.array([Fx, Fy])


def net_torque(forces, pivot=(0.0, 0.0)):
    """Net moment about `pivot`: sum of (x-px) F_y - (y-py) F_x (the z-component of
    r x F). Zero at rotational equilibrium -- and if sum F = 0 it is the SAME about
    every pivot."""
    px, py = pivot
    return float(sum((f[1][0] - px) * f[0][1] - (f[1][1] - py) * f[0][0] for f in forces))


def is_in_equilibrium(forces, tol=1e-6):
    """True if the body is in static equilibrium: net force and net torque both ~0."""
    return bool(np.all(np.abs(net_force(forces)) < tol) and abs(net_torque(forces)) < tol)


def solve_reactions(known_forces, reactions):
    """Solve for the unknown reaction magnitudes from sum F=0 and sum M=0.
    known_forces : list of ((F_x, F_y), (x, y)) -- the applied loads.
    reactions    : list of ((dir_x, dir_y), (x, y)) -- each an unknown-magnitude force
                   of KNOWN direction (unit vector) at a known point (a pin is two of
                   these, a roller one).
    Returns (magnitudes, reaction_forces) where reaction_forces are the full
    ((F_x,F_y),(x,y)) tuples. Requires 3 reactions (statically determinate)."""
    if len(reactions) != 3:
        raise ValueError("need exactly 3 unknown reaction components (determinate body)")
    A = np.zeros((3, 3))
    for j, ((dx, dy), (rx, ry)) in enumerate(reactions):
        A[0, j] = dx
        A[1, j] = dy
        A[2, j] = rx * dy - ry * dx          # moment of a unit force about the origin
    Fx, Fy = net_force(known_forces)
    M = net_torque(known_forces)
    b = np.array([-Fx, -Fy, -M])
    if abs(np.linalg.det(A)) < 1e-12:
        raise ValueError("reaction geometry is singular (mechanism / improper supports)")
    mags = np.linalg.solve(A, b)
    forces = [((m * dx, m * dy), pt) for m, ((dx, dy), pt) in zip(mags, reactions)]
    return mags, forces


def beam_reactions(length, point_loads):
    """Simply supported beam (pin at x=0, roller at x=length), horizontal. point_loads
    is a list of (magnitude_down, x). Returns (R_A, R_B) vertical reactions from
    sum M_A = 0 and sum F_y = 0."""
    if length <= 0:
        raise ValueError("length must be positive")
    loads = [((0.0, -P), (xp, 0.0)) for P, xp in point_loads]
    reactions = [((1.0, 0.0), (0.0, 0.0)),      # pin: horizontal
                 ((0.0, 1.0), (0.0, 0.0)),      # pin: vertical (R_A)
                 ((0.0, 1.0), (length, 0.0))]   # roller: vertical (R_B)
    mags, _ = solve_reactions(loads, reactions)
    return float(mags[1]), float(mags[2])       # R_A, R_B


if __name__ == "__main__":
    print("simply supported beam, L=10, 100 N at x=3:")
    Ra, Rb = beam_reactions(10.0, [(100.0, 3.0)])
    print(f"  R_A = {Ra:.1f} N, R_B = {Rb:.1f} N  (sum = {Ra+Rb:.0f} = load; R_B = 100*3/10 = 30)")
    print("  centered load: ", beam_reactions(10.0, [(100.0, 5.0)]), "(50/50 by symmetry)")

    print("\nladder against a smooth wall (W=200, theta=60 deg):")
    W, th, Lad = 200.0, np.radians(60), 4.0
    loads = [((0.0, -W), (0.5 * Lad * np.cos(th), 0.5 * Lad * np.sin(th)))]
    reactions = [((0.0, 1.0), (0.0, 0.0)),                       # floor normal N
                 ((1.0, 0.0), (0.0, 0.0)),                       # floor friction f
                 ((-1.0, 0.0), (Lad * np.cos(th), Lad * np.sin(th)))]  # wall normal
    mags, forces = solve_reactions(loads, reactions)
    N, f, Nw = mags
    print(f"  floor normal N = {N:.1f} (=W), friction f = {f:.1f}, wall N_w = {Nw:.1f}")
    print(f"  minimum coefficient of friction mu = f/N = {f/N:.4f}  "
          f"(= 1/(2 tan60) = {1/(2*np.tan(th)):.4f})")
    print(f"  the assembled reactions balance the load? "
          f"{is_in_equilibrium(loads + forces)}")
