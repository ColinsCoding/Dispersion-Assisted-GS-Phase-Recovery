"""Scale reading in an elevator -- Newton's second law you can stand on.

Stand on a bathroom scale in an elevator. The scale reads the NORMAL
FORCE N, not your weight mg. Newton's second law along the vertical:

    N - m g = m a      =>      N = m (g + a)

with a > 0 for upward acceleration. Four regimes:
  * at rest / constant velocity (a = 0):  N = mg      -- scale is honest
  * accelerating up (a > 0):              N > mg      -- you feel heavy
  * accelerating down (a < 0):            N < mg      -- you feel light
  * free fall (a = -g):                   N = 0       -- weightless

N is a CONTACT force: it cannot pull. If a < -g (elevator accelerates
down faster than gravity) your feet leave the floor and the scale reads
zero -- the reading clamps, it does not go negative.

THE PARITY PUNCHLINE (see dgs.even_odd): a smooth elevator trip uses an
acceleration profile that is ANTISYMMETRIC about the trip midpoint
(+a to speed up, -a to brake). It integrates to zero -- net velocity
change over the ride is 0 -- for the same reason any odd function's
symmetric integral vanishes.

NumPy only, py -3.13 safe. Education.
"""

import numpy as np

G_EARTH = 9.81  # m/s^2


def apparent_weight(mass_kg, a_up, g=G_EARTH):
    """Scale reading N = m(g + a), clamped at 0 (a scale cannot pull down).
    a_up > 0 means the elevator accelerates upward. Vectorized over a_up."""
    if mass_kg <= 0:
        raise ValueError("mass must be positive (kg)")
    if g <= 0:
        raise ValueError("g must be positive (m/s^2)")
    return np.maximum(mass_kg * (g + np.asarray(a_up, float)), 0.0)


def g_force(a_up, g=G_EARTH):
    """Scale reading as a multiple of true weight: N/(mg) = 1 + a/g."""
    if g <= 0:
        raise ValueError("g must be positive (m/s^2)")
    return np.maximum(1.0 + np.asarray(a_up, float) / g, 0.0)


def ride_profile(t, a_max=1.2, t_acc=2.0, t_cruise=4.0):
    """Acceleration a(t) for one upward trip, laid out as
        rest -- accel(+a_max) -- cruise -- decel(-a_max) -- rest
    over the window [0, 4*t_acc + t_cruise]. Starting and ending AT REST
    (a=0 at both ends) makes the two pulses fully interior reflections of
    each other, so a(t) is antisymmetric about the midpoint and integrates
    to exactly zero net dv -- the same reason any odd function's symmetric
    integral vanishes (dgs.even_odd)."""
    if a_max <= 0 or t_acc <= 0 or t_cruise < 0:
        raise ValueError("need a_max > 0, t_acc > 0, t_cruise >= 0")
    t = np.asarray(t, float)
    a = np.zeros_like(t)
    a[(t >= t_acc) & (t < 2 * t_acc)] = a_max                    # accel pulse
    a[(t >= 2 * t_acc + t_cruise) & (t < 3 * t_acc + t_cruise)] = -a_max  # decel
    return a


def trip_kinematics(t, a):
    """Integrate a(t) -> (v, y) by the trapezoid rule: the full ride record."""
    t, a = np.asarray(t, float), np.asarray(a, float)
    dt = np.diff(t)
    v = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * dt)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
    return v, y


if __name__ == "__main__":
    m = 70.0
    print(f"=== {m:.0f} kg rider, g = {G_EARTH} m/s^2 (true weight {m*G_EARTH:.1f} N) ===")
    for label, a in (("at rest", 0.0), ("accel up 2 m/s^2", 2.0),
                     ("accel down 2 m/s^2", -2.0), ("free fall", -G_EARTH),
                     ("cable snaps + rocket down", -15.0)):
        N = float(apparent_weight(m, a))
        print(f"  {label:26s} N = {N:6.1f} N   ({float(g_force(a)):.2f} g)")

    t = np.linspace(0, 12, 2401)   # rest-accel-cruise-decel-rest, ends at rest
    a = ride_profile(t)
    v, y = trip_kinematics(t, a)
    print(f"\nride: net dv = {v[-1]:.2e} m/s (antisymmetric a(t) integrates to 0),"
          f" floor gained: {y[-1]:.2f} m")
