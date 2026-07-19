"""The Michelson-Morley experiment: an interferometer geometry problem before
it's a relativity problem. Two perpendicular arms of equal length L; if
Earth moves through a stationary "aether" at velocity v, light in each arm
takes a DIFFERENT round-trip time -- the transverse arm's calculation is the
textbook "boat crossing a river with a current" trig problem (Pythagorean
theorem), not a relativistic one. The historically pivotal result was that
NO fringe shift was ever observed, which this repo's dgs/special_relativity.py
resolves (there is no aether to move through).

NumPy only. Education.
"""

import numpy as np

C = 299792458.0  # m/s


def longitudinal_time(L, v, c=C):
    """Round-trip time for the arm PARALLEL to the aether wind: downstream
    at (c-v), upstream at (c+v). t = L/(c-v) + L/(c+v) = 2Lc/(c^2-v^2)."""
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    if abs(v) >= c:
        raise ValueError(f"|v| must be < c, got v={v}, c={c}")
    return 2 * L * c / (c ** 2 - v ** 2)


def transverse_time(L, v, c=C):
    """Round-trip time for the arm PERPENDICULAR to the aether wind -- the
    classic 'boat crossing a river' problem: light must aim upstream at an
    angle to still hit the mirror straight across, so its path (in the
    aether frame) is the hypotenuse of a right triangle with legs L and
    v*t/2, traveled at speed c: (c*t/2)^2 = L^2 + (v*t/2)^2, solved for t."""
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    if abs(v) >= c:
        raise ValueError(f"|v| must be < c, got v={v}, c={c}")
    return 2 * L / np.sqrt(c ** 2 - v ** 2)


def time_difference(L, v, c=C):
    """Delta t = t_parallel - t_perpendicular -- the aether-wind prediction
    that would show up as a fringe shift when the apparatus is rotated 90
    degrees. Zero only if v=0 (no aether wind at all)."""
    return longitudinal_time(L, v, c) - transverse_time(L, v, c)


def predicted_fringe_shift(L, v, wavelength, c=C):
    """Expected fringe shift N when the interferometer is rotated 90 degrees
    (swapping which arm is parallel/perpendicular to the wind, doubling the
    effect): N = 2*L*v^2/(wavelength*c^2) for v << c (the actual experiment's
    regime -- Earth's orbital speed is ~1e-4 c)."""
    if wavelength <= 0:
        raise ValueError(f"wavelength must be positive, got {wavelength}")
    beta = v / c
    return 2 * L * beta ** 2 / wavelength


if __name__ == "__main__":
    # the actual 1887 Michelson-Morley apparatus (effective path length via
    # multiple reflections), Earth's orbital speed, sodium light
    L = 11.0            # m, effective arm length (folded via mirrors)
    v_earth = 3.0e4      # m/s, Earth's orbital speed around the Sun
    wavelength = 590e-9  # m, sodium light

    t_par = longitudinal_time(L, v_earth)
    t_perp = transverse_time(L, v_earth)
    dt = time_difference(L, v_earth)
    N_predicted = predicted_fringe_shift(L, v_earth, wavelength)

    print(f"L={L} m, v_earth={v_earth:.0e} m/s, lambda={wavelength*1e9:.0f} nm")
    print(f"t_parallel      = {t_par:.9e} s")
    print(f"t_perpendicular = {t_perp:.9e} s")
    print(f"delta_t         = {dt:.6e} s")
    print(f"predicted fringe shift (aether hypothesis): {N_predicted:.4f} fringes")
    print(f"actually observed (1887): < 0.01 fringes -- a null result by more than")
    print(f"a factor of {N_predicted/0.01:.0f}, the experiment that helped motivate")
    print(f"special relativity (see dgs/special_relativity.py): there IS no aether wind.")
