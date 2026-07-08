"""Simple machines: trade force for distance, but never cheat work.

The six classical simple machines -- lever, inclined plane, pulley, wheel-and-axle,
screw, wedge -- all do the same trick: multiply your force by making you push over a
LONGER distance. Their MECHANICAL ADVANTAGE (how many times the output force beats the
input) is set purely by geometry:

    lever            MA = effort_arm / load_arm
    inclined plane   MA = length / height          (= 1/sin theta)
    pulley           MA = number of supporting ropes
    wheel and axle   MA = wheel_radius / axle_radius
    screw            MA = 2*pi*handle_radius / pitch
    wedge            MA = length / thickness

The deep fact underneath is CONSERVATION OF WORK. An ideal (frictionless) machine puts
out exactly the work you put in:
        F_effort * d_effort = F_load * d_load,
so the force ratio equals the inverse distance ratio -- MA equals the VELOCITY RATIO
(d_effort/d_load). Multiply the force N times and you must move N times as far. There is
no free lunch: the machine redistributes W = F*d, it never creates energy (the same
energy conservation as dgs.hamiltonian_mechanics and dgs.circuit_energy).

Real machines lose some work to friction, so EFFICIENCY = W_out/W_in < 1 and the actual
mechanical advantage is the ideal one scaled by the efficiency. Verified: every MA
formula, MA = velocity ratio for an ideal machine, and W_in = W_out (ideal) vs
W_out < W_in (real). NumPy-free; py-3.13.
"""

import math


# ----------------------------------------------------------------------
# Mechanical advantage of each machine (geometry only)
# ----------------------------------------------------------------------

def lever_ma(effort_arm, load_arm):
    """Lever MA = effort_arm / load_arm (distances from the fulcrum). A longer
    effort arm multiplies force."""
    if effort_arm <= 0 or load_arm <= 0:
        raise ValueError("arm lengths must be positive")
    return effort_arm / load_arm


def inclined_plane_ma(length, height):
    """Inclined plane MA = ramp length / rise = 1/sin(theta). A gentler ramp
    (longer for the same height) multiplies force more."""
    if length <= 0 or height <= 0 or height > length:
        raise ValueError("need 0 < height <= length")
    return length / height


def pulley_ma(n_supporting_ropes):
    """Pulley system MA = number of rope segments supporting the load (an integer)."""
    if n_supporting_ropes < 1 or int(n_supporting_ropes) != n_supporting_ropes:
        raise ValueError("number of supporting ropes must be a positive integer")
    return int(n_supporting_ropes)


def wheel_and_axle_ma(wheel_radius, axle_radius):
    """Wheel-and-axle MA = wheel_radius / axle_radius (turn the big wheel, the
    small axle exerts more force)."""
    if wheel_radius <= 0 or axle_radius <= 0:
        raise ValueError("radii must be positive")
    return wheel_radius / axle_radius


def screw_ma(handle_radius, pitch):
    """Screw MA = 2*pi*handle_radius / pitch: one full turn (distance 2*pi*r)
    advances the screw by one pitch. A fine thread (small pitch) is a huge MA."""
    if handle_radius <= 0 or pitch <= 0:
        raise ValueError("handle_radius and pitch must be positive")
    return 2 * math.pi * handle_radius / pitch


def wedge_ma(length, thickness):
    """Wedge MA = length / thickness (a long thin wedge splits with more force)."""
    if length <= 0 or thickness <= 0:
        raise ValueError("length and thickness must be positive")
    return length / thickness


# ----------------------------------------------------------------------
# Work conservation: MA = velocity ratio, and efficiency
# ----------------------------------------------------------------------

def velocity_ratio(effort_distance, load_distance):
    """VR = effort_distance / load_distance: how much farther the effort moves than
    the load. For an ideal machine VR equals the mechanical advantage."""
    if effort_distance <= 0 or load_distance <= 0:
        raise ValueError("distances must be positive")
    return effort_distance / load_distance


def output_force(effort_force, ma, efficiency=1.0):
    """Load force a machine can move: F_load = efficiency * MA * F_effort. Ideal
    (efficiency=1) gives the full MA; friction reduces it."""
    if effort_force < 0 or ma <= 0:
        raise ValueError("need effort_force >= 0 and ma > 0")
    if not 0 < efficiency <= 1:
        raise ValueError("efficiency must be in (0, 1]")
    return efficiency * ma * effort_force


def actual_mechanical_advantage(ideal_ma, efficiency):
    """The real MA = ideal geometric MA * efficiency (< ideal because of friction)."""
    if not 0 < efficiency <= 1:
        raise ValueError("efficiency must be in (0, 1]")
    return ideal_ma * efficiency


def efficiency(work_out, work_in):
    """eta = W_out / W_in. Ideal machines reach 1 (work conserved); real ones are
    below 1 -- the missing work went to friction/heat, never created or destroyed."""
    if work_in <= 0 or work_out < 0:
        raise ValueError("need work_in > 0 and work_out >= 0")
    return work_out / work_in


def work_in(effort_force, effort_distance):
    """Work you put in, W_in = F_effort * d_effort."""
    return effort_force * effort_distance


def work_out(load_force, load_distance):
    """Useful work out, W_out = F_load * d_load."""
    return load_force * load_distance


if __name__ == "__main__":
    print("mechanical advantage (geometry only):")
    print(f"  lever (2 m vs 0.5 m arms):   MA = {lever_ma(2, 0.5):.0f}")
    print(f"  ramp (5 m long, 1 m high):   MA = {inclined_plane_ma(5, 1):.0f}")
    print(f"  block-and-tackle (4 ropes):  MA = {pulley_ma(4)}")
    print(f"  wheel/axle (0.3 m / 0.05 m): MA = {wheel_and_axle_ma(0.3, 0.05):.0f}")
    print(f"  screw (0.2 m handle, 5 mm):  MA = {screw_ma(0.2, 0.005):.0f}")

    print("\nconservation of work (ideal lever, MA=4):")
    Fe, de = 100.0, 4.0                 # push 100 N over 4 m
    ma = 4
    Fl, dl = output_force(Fe, ma), de / ma
    print(f"  in:  {Fe} N over {de} m = {work_in(Fe, de):.0f} J")
    print(f"  out: {Fl:.0f} N over {dl:.1f} m = {work_out(Fl, dl):.0f} J  (equal -- no free lunch)")
    print(f"  MA = {ma}, velocity ratio = {velocity_ratio(de, dl):.0f}  (equal for an ideal machine)")

    print("\nreal machine at 80% efficiency:")
    eta = 0.8
    print(f"  actual MA = {actual_mechanical_advantage(4, eta):.1f} (< 4), "
          f"lifts {output_force(100, 4, eta):.0f} N (not 400)")
    print(f"  efficiency of 350 J out from 400 J in = {efficiency(350, 400):.0%}")
