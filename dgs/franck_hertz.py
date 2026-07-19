"""The Franck-Hertz experiment: filament, accelerating grid, collector,
electrometer -- the tube that first proved atomic energy levels are
discrete, simulated electron-by-electron.

Electrons boil off a heated filament, get accelerated toward a grid by
voltage V_grid, and along the way can collide with mercury vapor atoms. An
ELASTIC collision costs them almost nothing (electron mass << mercury atom
mass). An INELASTIC collision can only happen once the electron's kinetic
energy reaches mercury's first excitation energy (4.9 eV) -- below that
threshold, mercury simply can't absorb the energy (no state to excite into),
so collisions stay elastic. Above it, an inelastic collision dumps exactly
4.9 eV into the atom and resets the electron's kinetic energy near zero.

Past the grid, a small retarding voltage V_retard pushes back any electron
that doesn't have enough leftover kinetic energy, so only electrons that
finish with KE > eV_retard reach the collector and register on the
electrometer. Sweep V_grid and the collector current rises, then drops
sharply every time V_grid crosses another multiple of 4.9V (electrons now
have enough room to suffer one more inelastic collision before the grid),
then rises again -- the famous periodic Franck-Hertz dips.
"""

import numpy as np


def simulate_electron(V_grid, V_excitation=4.9, n_segments=2000, p_collision=0.05, rng=None):
    """Monte Carlo one electron's flight from filament to grid across
    `n_segments` small steps. Energy gain per step is uniform
    (eV_grid / n_segments), simulating constant acceleration. At each step,
    with probability p_collision the electron collides with a mercury atom:
    if its accumulated kinetic energy (in eV) is >= V_excitation, the
    collision is INELASTIC and costs it exactly V_excitation eV; otherwise
    the collision is elastic and costs (approximately) nothing. Returns the
    electron's final kinetic energy in eV at the grid."""
    if rng is None:
        rng = np.random.default_rng()
    energy = 0.0
    step_energy = V_grid / n_segments
    for _ in range(n_segments):
        energy += step_energy
        if rng.random() < p_collision:
            if energy >= V_excitation:
                energy -= V_excitation
    return energy


def franck_hertz_iv_curve(V_grid_values, V_excitation=4.9, V_retard=1.5,
                           n_electrons=2000, n_segments=2000, p_collision=0.05, seed=0):
    """Sweep V_grid and compute the fraction of electrons that reach the
    collector (final KE > V_retard) at each voltage -- proportional to the
    electrometer current. Returns an array the same length as
    V_grid_values."""
    rng = np.random.default_rng(seed)
    currents = np.zeros(len(V_grid_values))
    for i, V_grid in enumerate(V_grid_values):
        survived = 0
        for _ in range(n_electrons):
            final_energy = simulate_electron(V_grid, V_excitation, n_segments, p_collision, rng)
            if final_energy > V_retard:
                survived += 1
        currents[i] = survived / n_electrons
    return currents


def find_dip_spacing(V_values, I_values, min_dip_separation=2.0):
    """Locate local minima (dips) in the I-V curve and return the spacing
    between consecutive dips -- should cluster near V_excitation (4.9V for
    mercury), the experimental signature of a single discrete excitation
    energy. Raw local-minimum detection on a noisy Monte Carlo curve can
    flag two adjacent samples near the same true dip; merge any detections
    closer than `min_dip_separation` (they're the same physical dip, not two)."""
    raw_dips = []
    for i in range(2, len(I_values) - 2):
        window = I_values[i - 2:i + 3]
        if I_values[i] == window.min() and I_values[i] < I_values[i - 2] and I_values[i] < I_values[i + 2]:
            raw_dips.append(V_values[i])

    dips = []
    for v in raw_dips:
        if not dips or v - dips[-1] >= min_dip_separation:
            dips.append(v)
    dips = np.array(dips)
    if len(dips) < 2:
        return dips, np.array([])
    spacings = np.diff(dips)
    return dips, spacings
