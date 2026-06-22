"""Statics -- F = m a with a = 0: the engineering of things that don't move.

A body is in STATIC equilibrium when it has no acceleration, linear OR angular. That
is two conditions:

    sum(F) = 0     (forces balance -- no linear acceleration),
    sum(tau) = 0   (torques/moments balance -- no angular acceleration),

and everything in statics -- beams, trusses, brackets, bones, hinges -- is solving
those for the unknown reaction forces. The reactions are exactly the Lagrange
multipliers that enforce the "stays put" constraints (a tie to dgs.lagrangian): the
multiplier IS the constraint force. The lever (tau balance) gives mechanical
advantage; a simply-supported beam splits a load between its supports by the same
moment balance. NumPy, 2-D. Education.
"""

import numpy as np


def net_force(forces):
    """Vector sum of the applied forces. Zero in static equilibrium (a = 0)."""
    return np.sum(np.asarray(forces, float), axis=0)


def net_torque(forces, points, pivot=(0.0, 0.0)):
    """Net torque (moment) about `pivot`: sum of r x F. In 2-D this is the scalar
    z-component r_x F_y - r_y F_x. Zero in equilibrium (no angular acceleration)."""
    forces = np.asarray(forces, float)
    points = np.asarray(points, float)
    r = points - np.asarray(pivot, float)
    return float(np.sum(r[:, 0] * forces[:, 1] - r[:, 1] * forces[:, 0]))


def in_equilibrium(forces, points, pivot=(0.0, 0.0), tol=1e-9):
    """True if the body is in static equilibrium: net force = 0 AND net torque = 0.
    (If forces balance, the torque sum is independent of the chosen pivot.)"""
    F = net_force(forces)
    tau = net_torque(forces, points, pivot)
    return bool(np.all(np.abs(F) < tol) and abs(tau) < tol)


def lever(F1, d1, d2):
    """Lever / torque balance F1 d1 = F2 d2  ->  F2 = F1 d1 / d2. The mechanical
    advantage d1/d2: a small force on a long arm balances a large force on a short arm
    (seesaw, wrench, crowbar, bone-and-muscle)."""
    return F1 * d1 / d2


def beam_reactions(loads, support_a=0.0, support_b=1.0):
    """Simply-supported beam with two vertical supports at support_a and support_b.
    `loads` = list of (position, downward_force). Returns (R_a, R_b), the upward
    reactions, from sum(Fy)=0 and sum(moments about A)=0:
        R_b = [sum (x_i - a) W_i] / (b - a),   R_a = sum W_i - R_b.
    A central load splits 50/50; an off-center load splits by the lever rule."""
    W = sum(f for _, f in loads)
    M_a = sum((x - support_a) * f for x, f in loads)
    R_b = M_a / (support_b - support_a)
    return W - R_b, R_b


def hinged_bracket(L, weight, cable_angle_deg):
    """A horizontal beam (length L) hinged at the wall, held level by a cable from its
    far end up to the wall at `cable_angle_deg` above horizontal, with `weight` hung at
    the end. Returns (T, Hx, Hy): cable tension and the hinge reaction components.
    Moments about the hinge: T sin(theta) L = weight L  ->  T = weight / sin(theta)."""
    th = np.radians(cable_angle_deg)
    T = weight / np.sin(th)                      # moment balance about the hinge
    Hx = T * np.cos(th)                          # hinge pulls back against the cable's pull
    Hy = weight - T * np.sin(th)                 # = 0 here (vertical forces balance)
    return T, Hx, Hy


if __name__ == "__main__":
    # a 10 m beam, 100 N load at the center, supports at the ends -> 50 N each
    print("beam (center load):", beam_reactions([(5.0, 100.0)], 0.0, 10.0))
    print("beam (load at 8 m): ", beam_reactions([(8.0, 100.0)], 0.0, 10.0))
    # seesaw: a 200 N kid 1 m out balances an 80 N kid how far out?
    print("lever: 200 N at 1 m balances 80 N at", lever(200, 1, 80), "m")
    # a hinged bracket holding 500 N with a 30-degree cable
    T, Hx, Hy = hinged_bracket(2.0, 500.0, 30.0)
    print(f"bracket: cable tension T = {T:.0f} N, hinge (Hx,Hy) = ({Hx:.0f}, {Hy:.0f}) N")
    # check equilibrium of the beam: down load + two up reactions
    Ra, Rb = beam_reactions([(5.0, 100.0)], 0.0, 10.0)
    forces = [(0, Ra), (0, -100.0), (0, Rb)]; pts = [(0, 0), (5, 0), (10, 0)]
    print("beam in equilibrium:", in_equilibrium(forces, pts))
