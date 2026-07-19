"""Photonic circuits as the SPICE engine's frequency-domain cousin.

dgs.spice models LUMPED electrical elements (R, L, C at a point) with MNA + ODEs.
Photonic "circuits" -- ring resonators, MZIs, directional couplers -- are
DISTRIBUTED (light picks up phase over a waveguide length) but the steady-state
frequency response collapses onto the *same* second-order resonant transfer
function as a series RLC tank:

    RLC bandpass:     H(w) = (R/L) jw / [ (jw)^2 + (R/L) jw + 1/(LC) ]
    Ring resonator:   H(w) = t - r / [ 1 - r t exp(j phi(w)) ]   (all-pass/notch)

Both are a single complex pole pair set by a loss rate and a resonance condition.
Q-factor and finesse measure the *same* thing (energy stored / energy lost per
cycle) in two different unit systems -- this module is the dictionary between them.

NumPy only. Education -- a window into photonic IC simulators (Lumerical INTERCONNECT,
ANSYS Photonics), not a replacement.
"""

import numpy as np

from dgs import spice


# ── ring resonator: an RLC tank with light instead of current ────────
def ring_round_trip_phase(omega, n_eff, L, c=2.998e8):
    """Phase accumulated in one trip around a ring of length L (m) at effective
    index n_eff: phi = omega * n_eff * L / c. The photonic analog of omega*t."""
    return omega * n_eff * L / c


def ring_transfer_function(omega, n_eff, L, r, alpha=0.0, c=2.998e8):
    """All-pass ring resonator response (single bus waveguide, self-coupling r).

    H = (r - a*exp(-j phi)) / (1 - r*a*exp(-j phi)),  a = exp(-alpha L / 2)

    r in (0,1) is the field self-coupling coefficient (r=1 -> no coupling, ring
    invisible); alpha (1/m) is the power loss rate. Same rational-in-exp(j phi)
    form as a Fabry-Perot etalon, and the same single-pole-pair physics as the
    RLC bandpass: a sharp resonance set by how much energy leaks out per cycle."""
    phi = ring_round_trip_phase(omega, n_eff, L, c)
    a = np.exp(-alpha * L / 2)
    return (r - a * np.exp(-1j * phi)) / (1 - r * a * np.exp(-1j * phi))


def ring_resonances(n_eff, L, c=2.998e8, n_max=5):
    """Resonant angular frequencies of the ring: phi = 2*pi*m -> omega_m = 2 pi m c/(n_eff L).
    The free spectral range (FSR) is the spacing omega_{m+1} - omega_m = 2 pi c/(n_eff L),
    the photonic analog of the RLC resonant frequency f0 = 1/(2 pi sqrt(LC))."""
    m = np.arange(1, n_max + 1)
    return 2 * np.pi * m * c / (n_eff * L)


def finesse(r, alpha=0.0, L=1.0):
    """Ring finesse F = pi*sqrt(r*a) / (1 - r*a), a = exp(-alpha L/2).
    F is FSR / (resonance linewidth) -- exactly what Q is to f0/bandwidth in an
    RLC circuit. High r (weak coupling, light makes many round trips) -> high F,
    same direction as low R (weak damping) -> high Q."""
    a = np.exp(-alpha * L / 2)
    ra = r * a
    return np.pi * np.sqrt(ra) / (1 - ra)


def finesse_to_equivalent_Q(F, n_eff, L, m=1, c=2.998e8):
    """Convert ring finesse into an equivalent electrical Q-factor: Q = m * F,
    where m is the resonance order (mode number). Both Q and m*F equal
    (energy stored)/(energy lost per radian) -- finesse counts round trips per
    photon lifetime, Q counts radians of oscillation per energy-decay time, and
    they coincide because one round trip IS one cycle at resonance."""
    return m * F


def equivalent_rlc(omega0, F):
    """Map a ring's (resonance, finesse) onto a series-RLC tank with the same
    Q and f0, fixing L=1 H arbitrarily (only ratios R/L and 1/(LC) are physical).
    Lets you reuse dgs.spice.rlc_step_response / rlc_damping to visualize a
    photonic resonance build-up/decay with the lumped-circuit machinery."""
    L = 1.0
    C = 1.0 / (omega0 ** 2 * L)
    Q = F  # m=1 fundamental
    R = omega0 * L / Q
    return R, L, C


# ── Mach-Zehnder interferometer: a balanced bridge for light ──────────
def mzi_transfer(phase_diff, split_ratio=0.5):
    """Two-port MZI bar/cross outputs given the phase imbalance between arms and
    a (possibly unbalanced) input splitter. Returns (P_bar, P_cross) normalized to
    unit input power. With split_ratio=0.5 this is the textbook 2x2 50/50 MZI:
        P_bar   = cos^2(phase_diff/2)
        P_cross = sin^2(phase_diff/2)
    Same role as a Wheatstone bridge or an analog summing junction: an MZI
    converts a PHASE difference into an intensity (amplitude) you can measure,
    just as a bridge converts a resistance imbalance into a voltage."""
    k = split_ratio
    p_bar = (1 - 2 * k * (1 - k) * (1 - np.cos(phase_diff)))
    p_cross = 1.0 - p_bar
    return p_bar, p_cross


def mzi_extinction_ratio(split_ratio=0.5):
    """Extinction ratio (max/min cross-port power) over phase sweep -- infinite
    for a perfectly balanced 50/50 splitter, finite once split_ratio != 0.5,
    exactly how an unbalanced Wheatstone bridge never nulls to zero."""
    k = split_ratio
    p_max = 1.0
    p_min = (1 - 2 * np.sqrt(k * (1 - k))) ** 2 if k not in (0.0, 1.0) else 1.0
    return float("inf") if p_min <= 1e-15 else p_max / p_min


if __name__ == "__main__":
    n_eff, L = 2.2, 200e-6  # silicon ring, 200 um circumference
    res = ring_resonances(n_eff, L, n_max=3)
    fsr = res[1] - res[0]
    print(f"ring resonances (rad/s): {np.round(res, -9)}")
    print(f"FSR = {fsr/2/np.pi/1e9:.2f} GHz")

    r, alpha = 0.96, 50.0  # self-coupling, loss (1/m)
    F = finesse(r, alpha, L)
    Q = finesse_to_equivalent_Q(F, n_eff, L)
    print(f"finesse F={F:.1f}  equivalent electrical Q={Q:.1f}")

    R, Lc, C = equivalent_rlc(res[0], F)
    regime, roots = spice.rlc_damping(R, Lc, C)
    print(f"equivalent RLC: R={R:.3e} L={Lc} C={C:.3e}  -> {regime}-damped (ring is always underdamped/ringing)")

    pb, pc = mzi_transfer(np.pi / 3)
    print(f"MZI 50/50 at dphi=pi/3: bar={pb:.3f} cross={pc:.3f} (sum={pb+pc:.3f})")
