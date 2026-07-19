"""SRAM, DRAM, EEPROM: three ways to store one bit, three different circuit
physics, reusing the RC-decay and bistability math already in this repo.

  * SRAM -- two cross-coupled inverters (a bistable latch, no capacitor to
    leak): stores a bit as long as power is applied, fast, but needs 6
    transistors/bit -- the noise-margin analysis here is the same kind of
    bistable-equilibrium reasoning as dgs.lennard_jones' potential minimum.
  * DRAM -- one transistor + one capacitor: cheap and dense, but the
    capacitor leaks (dgs.spice's RC discharge, same equation as
    dgs.membrane_biophysics's membrane time constant) and MUST be refreshed
    before the stored voltage decays below the read threshold.
  * EEPROM -- charge trapped on a floating gate behind an insulating oxide:
    the same exponential leakage as DRAM, but with a leakage time constant
    measured in YEARS instead of milliseconds, because the oxide barrier is
    so much more resistive.
"""

import numpy as np


# -- SRAM: a bistable latch, not a leaky capacitor -------------------------------

def sram_latch_stability(V_dd, V_th, gain):
    """A cross-coupled-inverter SRAM cell's transfer characteristic V_out =
    V_dd / (1 + exp(-gain*(V_in - V_th))) composed with itself (inverter 2
    feeding inverter 1) has fixed points where V = f(f(V)). Returns the
    stable fixed points (the cell's two stored-bit voltages) found by
    scanning + bisection on g(V) = f(f(V)) - V."""
    def f(v):
        return V_dd / (1 + np.exp(-gain * (v - V_th)))

    def g(v):
        return f(f(v)) - v

    V_scan = np.linspace(0, V_dd, 2000)
    g_vals = g(V_scan)
    sign_changes = np.where(np.diff(np.sign(g_vals)) != 0)[0]

    roots = []
    for idx in sign_changes:
        lo, hi = V_scan[idx], V_scan[idx + 1]
        for _ in range(60):
            mid = (lo + hi) / 2
            if np.sign(g(lo)) == np.sign(g(mid)):
                lo = mid
            else:
                hi = mid
        roots.append((lo + hi) / 2)
    return roots


def sram_noise_margin(stable_states):
    """`stable_states` from sram_latch_stability has THREE fixed points for a
    bistable cell: a low-stable, an unstable middle (the switching
    threshold), and a high-stable. The noise margin is how far each stored
    level sits from that unstable threshold -- the actual definition of
    static noise margin -- reported as the worse (smaller) of the two."""
    if len(stable_states) != 3:
        return 0.0
    low, threshold, high = sorted(stable_states)
    return min(threshold - low, high - threshold)


# -- DRAM: a leaky capacitor that needs periodic refresh -------------------------

def dram_cell_decay(V0, t, R_leak, C_cell):
    """A DRAM bit cell discharges exponentially through its leakage
    resistance: V(t) = V0 * exp(-t / (R_leak * C_cell)) -- the identical RC
    decay as dgs.membrane_biophysics's membrane_charging, just discharging
    instead of charging."""
    tau = R_leak * C_cell
    return V0 * np.exp(-np.asarray(t) / tau)


def dram_refresh_interval(V0, V_read_threshold, R_leak, C_cell):
    """Time until the cell's voltage decays to the minimum readable
    threshold: t_refresh = tau * ln(V0 / V_threshold) -- solved directly from
    the exponential decay law, this is the maximum allowed gap between
    refresh cycles before a stored '1' could misread as '0'."""
    if not 0 < V_read_threshold < V0:
        raise ValueError("V_read_threshold must be in (0, V0)")
    tau = R_leak * C_cell
    return tau * np.log(V0 / V_read_threshold)


# -- EEPROM: the same decay law, with a MUCH larger time constant ---------------

def eeprom_retention_time(Q0, Q_read_threshold, R_oxide, C_gate):
    """Same exponential-decay structure as DRAM (charge leaking off a node
    through a resistance), but R_oxide (the tunneling/leakage resistance
    through the floating-gate oxide) is enormous -- which is the entire
    reason EEPROM holds a bit for ~10+ years while DRAM needs refreshing
    every few milliseconds. Same formula, ~10^15x larger R."""
    return dram_refresh_interval(Q0, Q_read_threshold, R_oxide, C_gate)


def compare_retention_times(V0=3.3, V_threshold=1.5, C_cell=30e-15):
    """Side-by-side retention time for representative DRAM vs EEPROM leakage
    resistances, holding V0/threshold/capacitance fixed -- isolates exactly
    how much of the retention-time difference comes from R_leak alone."""
    R_dram = 3e12    # leakage path giving a realistic tens-of-ms refresh interval
    R_eeprom = 3e24  # oxide tunneling resistance -> year-scale retention (same ratio idea, 10^12x larger)
    t_dram = dram_refresh_interval(V0, V_threshold, R_dram, C_cell)
    t_eeprom = eeprom_retention_time(V0, V_threshold, R_eeprom, C_cell)
    return {
        "dram_retention_s": t_dram, "eeprom_retention_s": t_eeprom,
        "dram_retention_ms": t_dram * 1e3,
        "eeprom_retention_years": t_eeprom / (3600 * 24 * 365),
        "ratio": t_eeprom / t_dram,
    }
