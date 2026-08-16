"""A molecular laser you could actually build for about $1000: the DIY
TEA nitrogen (N2) laser, 337.1 nm, and why it MUST be pulsed in
nanoseconds rather than run continuously.

N2 is a genuinely MOLECULAR gain medium (unlike HeNe's atomic transition
already covered in dgs.laser_physics): the 337.1 nm line is the
C^3(Pi)_u -> B^3(Pi)_g vibronic transition of the nitrogen molecule.
Reuses dgs.laser_physics for the generic gain/threshold physics (Beer-
Lambert with gain, threshold condition, dB) and dgs.hardware_bom's
bom_total_cost/print_bom for costing -- this module adds the
MOLECULE-specific physics those two don't cover: a SELF-TERMINATING
transition, and why that forces a fast (nanosecond) electrical pump.

SELF-TERMINATING TRANSITION. The upper level (C, radiative lifetime
~40 ns) decays FASTER than the lower level (B, radiative lifetime
~10 microseconds) empties. Population dumped into the lower level by
stimulated emission has nowhere to go -- it accumulates until N1 catches
up to N2 and the inversion (and the gain) collapses, typically within
~10-20 ns of the pump starting. A laser that CANNOT sustain steady-state
inversion cannot run CW; it must be pumped by an electrical pulse fast
enough to build the inversion before self-termination kills it, which is
exactly why real N2 lasers use a fast spark-gap discharge (Transversely
Excited Atmospheric pressure, TEA) rather than a slow glow discharge.

Simplified two-level cascade kinetics (population fed from an
instantaneous pump into N2 at t=0, standard exponential cascade decay)
computes the self-termination time directly, rather than just quoting the
~10-20 ns literature figure.

HIGH SINGLE-PASS GAIN, NO CAVITY NEEDED. N2's gain coefficient is famously
large (~0.1-0.2 /cm = 10-20 /m); dgs.laser_physics.intensity_after_gain
over a realistic ~0.3-0.5 m discharge channel already gives large
amplification with NO mirrors at all -- real TEA N2 lasers commonly run
as amplified-spontaneous-emission (ASE / "superradiant") sources, using
Fresnel reflection off a bare window (R~0.04) instead of a resonant
cavity. dgs.laser_physics.threshold_gain shows what an actual
low-reflectivity cavity WOULD need, for comparison.

Numbers used below (upper/lower lifetimes, typical gain coefficient,
efficiency) are typical/textbook TEA N2 laser figures, not measurements
from a specific paper -- flagged as such rather than presented as a
primary-source citation. py-3.13.
"""

from __future__ import annotations
import numpy as np

from dgs.laser_physics import (
    H_PLANCK, C_LIGHT, EV_J, gain_coefficient, intensity_after_gain,
    gain_dB, threshold_gain, round_trip_gain,
)
from dgs.hardware_bom import bom_total_cost, print_bom

N2_WAVELENGTH_NM = 337.1
N2_UPPER_LIFETIME_NS = 40.0     # C^3(Pi)_u, typical literature value
N2_LOWER_LIFETIME_NS = 10000.0  # B^3(Pi)_g, typical literature value (~10 us)
N2_TYPICAL_GAIN_PER_M = 15.0    # typical small-signal gain coefficient, 0.1-0.2 /cm


def photon_energy(wavelength_nm: float = N2_WAVELENGTH_NM) -> dict:
    """E = h*c/lambda, in both joules and eV (reusing dgs.laser_physics'
    own H_PLANCK/C_LIGHT/EV_J constants rather than redefining them)."""
    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive")
    E_J = H_PLANCK * C_LIGHT / (wavelength_nm * 1e-9)
    return {"wavelength_nm": wavelength_nm, "E_J": E_J, "E_eV": E_J / EV_J}


def population_kinetics(t_ns, tau_upper_ns: float = N2_UPPER_LIFETIME_NS,
                         tau_lower_ns: float = N2_LOWER_LIFETIME_NS, N0: float = 1.0):
    """Simplified two-level cascade: an instantaneous pump loads N2(0)=N0,
    N1(0)=0; every upper-level decay feeds the lower level (a
    simplification of the real multi-level chemistry, but the right
    qualitative model for WHY a long-lived lower level bottlenecks the
    laser). Standard cascade-decay closed form:
        N2(t) = N0*exp(-t/tau2)
        N1(t) = N0*tau1/(tau1-tau2) * (exp(-t/tau1) - exp(-t/tau2))
    Requires tau_upper != tau_lower (the N2-laser case: tau_lower >> tau_upper)."""
    if tau_upper_ns <= 0 or tau_lower_ns <= 0:
        raise ValueError("lifetimes must be positive")
    if tau_upper_ns == tau_lower_ns:
        raise ValueError("this closed form requires tau_upper != tau_lower")
    t = np.asarray(t_ns, dtype=float)
    N2 = N0 * np.exp(-t / tau_upper_ns)
    N1 = N0 * tau_lower_ns / (tau_lower_ns - tau_upper_ns) * (np.exp(-t / tau_lower_ns) - np.exp(-t / tau_upper_ns))
    return N2, N1


def gain_window_duration(tau_upper_ns: float = N2_UPPER_LIFETIME_NS,
                          tau_lower_ns: float = N2_LOWER_LIFETIME_NS,
                          search_max_ns: float = 200.0, n_search: int = 200_000) -> float:
    """How long does N2(t) > N1(t) (positive inversion, i.e. usable gain)
    last, computed by scanning population_kinetics for the crossing point
    -- not just quoting the ~10-20 ns literature figure. Requires
    tau_lower > tau_upper (the self-terminating regime); raises otherwise,
    since with tau_lower < tau_upper the lower level never bottlenecks and
    this whole self-termination story doesn't apply."""
    if tau_lower_ns <= tau_upper_ns:
        raise ValueError("self-termination requires tau_lower_ns > tau_upper_ns")
    t = np.linspace(1e-6, search_max_ns, n_search)   # start just after t=0 (N2=N1=0 there)
    N2, N1 = population_kinetics(t, tau_upper_ns, tau_lower_ns)
    inverted = N2 > N1
    if not inverted[0]:
        raise AssertionError("expected N2 > N1 immediately after the pump; check lifetimes")
    if inverted.all():
        raise AssertionError(f"inversion never collapses within search_max_ns={search_max_ns}; "
                              f"increase search_max_ns")
    crossing_idx = int(np.argmax(~inverted))   # first index where inversion is lost
    return float(t[crossing_idx])


def discharge_energy(C_F: float, V: float) -> float:
    """Capacitor stored energy E = (1/2) C V^2 -- the electrical energy
    available to a TEA discharge from its pump capacitor(s)."""
    if C_F <= 0:
        raise ValueError("C_F must be positive")
    return 0.5 * C_F * V ** 2


def peak_electrical_power(energy_J: float, pulse_duration_s: float) -> float:
    """P = E/t. Even a modest stored energy (~0.1 J) becomes a MW-scale
    peak power once discharged in a few nanoseconds -- the physical reason
    a TEA laser needs a fast spark gap, not a slow switch."""
    if energy_J < 0 or pulse_duration_s <= 0:
        raise ValueError("energy_J must be >= 0 and pulse_duration_s must be > 0")
    return energy_J / pulse_duration_s


def optical_output(electrical_energy_J: float, efficiency: float, pulse_duration_s: float) -> dict:
    """Optical pulse energy = electrical_energy * efficiency (typical TEA
    N2 laser wall-plug/electrical efficiency: ~0.1%-1%), and the
    corresponding optical peak power over the same pulse duration."""
    if not (0 < efficiency < 1):
        raise ValueError("efficiency must be in (0, 1)")
    E_opt = electrical_energy_J * efficiency
    return {"optical_energy_J": E_opt, "optical_peak_power_W": peak_electrical_power(E_opt, pulse_duration_s)}


def amplification_over_gain_length(g_per_m: float = N2_TYPICAL_GAIN_PER_M, length_m: float = 0.4) -> dict:
    """Single-pass intensity gain I/I0 = exp(g*L) over the discharge
    channel (reuses dgs.laser_physics.intensity_after_gain directly), and
    the equivalent dB, showing N2's gain is large enough for meaningful
    amplification with NO cavity mirrors at all (ASE/superradiant
    operation, the way real TEA N2 lasers are commonly built)."""
    ratio = intensity_after_gain(1.0, g_per_m, length_m)
    return {"g_per_m": g_per_m, "length_m": length_m, "I_over_I0": ratio,
            "gain_dB": gain_dB(ratio, 1.0)}


def mirrorless_vs_cavity_threshold(g_per_m: float, length_m: float,
                                    R_fresnel: float = 0.04, alpha_loss_per_m: float = 1.0) -> dict:
    """Compares N2's actual single-pass gain against threshold_gain() for
    a WEAKLY reflective cavity (bare Fresnel-reflecting windows, R~0.04
    each) -- showing the single-pass gain alone already exceeds what a
    real resonant cavity would even need, which is why N2 lasers commonly
    skip mirrors entirely."""
    g_th = threshold_gain(alpha_loss_per_m, R_fresnel, R_fresnel, length_m)
    rt_gain = round_trip_gain(g_per_m, alpha_loss_per_m, R_fresnel, R_fresnel, length_m)
    return {"threshold_gain_per_m": g_th, "actual_gain_per_m": g_per_m,
            "exceeds_threshold": g_per_m > g_th, "round_trip_gain_factor": rt_gain}


# ── bill of materials: a DIY TEA N2 laser, ~$1000 budget ────────────────

BOM_DIY_N2_LASER = [
    {"item": "high-voltage DC power supply",
     "example_part": "15-25 kV, low-current bench supply or flyback-based module",
     "qty": 1, "approx_usd": 250,
     "spec": "15-25 kV DC, <5 mA, charges the pump capacitor between shots",
     "role": "charges the discharge capacitor to the switching voltage"},
    {"item": "high-voltage pulse capacitors",
     "example_part": "doorknob/ceramic HV capacitors, ~1 nF, 30 kV rated (pair, Blumlein-style)",
     "qty": 2, "approx_usd": 80,
     "spec": "~1 nF each, >=30 kV rating (2x safety margin over charge voltage)",
     "role": "stores the pump energy E=(1/2)C V^2; a Blumlein pair also shapes the fast pulse"},
    {"item": "spark-gap switch",
     "example_part": "adjustable brass-electrode spark gap, self-built or commercial",
     "qty": 1, "approx_usd": 60,
     "spec": "sub-5 ns closing time -- must be far faster than the ~10-20 ns self-termination window",
     "role": "the fast switch that dumps the capacitor energy into the discharge in nanoseconds"},
    {"item": "copper-clad laminate for the transmission-line/Blumlein pulser board",
     "example_part": "double-sided FR4 or PCB copper-clad sheet",
     "qty": 1, "approx_usd": 120,
     "spec": "large-area copper-clad board, forms the low-inductance strip-line pulser",
     "role": "delivers the discharge to the laser channel with minimal parasitic inductance"},
    {"item": "discharge channel electrodes + laser channel enclosure",
     "example_part": "aluminum bar electrodes, acrylic/PTFE channel body",
     "qty": 1, "approx_usd": 150,
     "spec": "~30-50 cm transversely-excited discharge channel, uniform preionization",
     "role": "the actual gain medium: nitrogen (or filtered air) between the electrodes"},
    {"item": "nitrogen gas cylinder + regulator (optional -- ambient air also works, less efficient)",
     "example_part": "small N2 cylinder, 2-stage regulator",
     "qty": 1, "approx_usd": 70,
     "spec": "research/industrial-grade N2, low flow, near-atmospheric pressure",
     "role": "pure N2 gain medium gives more consistent output than filtered room air"},
    {"item": "UV-transmitting output window",
     "example_part": "UV-fused silica window",
     "qty": 1, "approx_usd": 80,
     "spec": "transmits 337 nm; bare (uncoated) Fresnel reflection (R~0.04) is enough -- no mirror needed",
     "role": "output coupler for the ASE/superradiant beam"},
    {"item": "enclosure, HV interlock, cabling, misc.",
     "example_part": "project box, microswitch interlock, HV wire",
     "qty": 1, "approx_usd": 110,
     "spec": "basic HV safety interlock (REQUIRED -- these voltages are lethal), connectors, wiring",
     "role": "safety and mechanical integration"},
]


def budget_feasibility(bom_list=BOM_DIY_N2_LASER, budget_usd: float = 1000.0) -> dict:
    """Total BOM cost vs. the target budget."""
    total = bom_total_cost(bom_list)
    return {"total_usd": total, "budget_usd": budget_usd,
            "within_budget": total <= budget_usd, "margin_usd": budget_usd - total}


if __name__ == "__main__":
    print("=== N2 (337.1 nm) photon energy ===")
    E = photon_energy()
    print(f"  E = {E['E_eV']:.3f} eV = {E['E_J']:.3e} J")

    print("\n=== self-terminating transition: how long does the gain window last? ===")
    t_window = gain_window_duration()
    print(f"  upper lifetime {N2_UPPER_LIFETIME_NS} ns, lower lifetime {N2_LOWER_LIFETIME_NS} ns")
    print(f"  computed self-termination time: {t_window:.2f} ns")
    print(f"  -> the electrical pump must build the inversion FASTER than this -> needs a fast (ns) discharge")

    print("\n=== discharge energetics: mJ of energy, MW of peak power ===")
    C, V, t_pulse = 1e-9, 20000.0, 3e-9
    E_elec = discharge_energy(C, V)
    P_peak = peak_electrical_power(E_elec, t_pulse)
    print(f"  C={C*1e9:.0f} nF, V={V/1e3:.0f} kV -> E={E_elec*1e3:.1f} mJ stored")
    print(f"  discharged in {t_pulse*1e9:.0f} ns -> peak ELECTRICAL power = {P_peak/1e6:.1f} MW")
    opt = optical_output(E_elec, efficiency=0.005, pulse_duration_s=t_pulse)
    print(f"  at 0.5% efficiency: optical energy = {opt['optical_energy_J']*1e3:.3f} mJ, "
          f"optical peak power = {opt['optical_peak_power_W']/1e3:.1f} kW")

    print("\n=== single-pass gain: no cavity needed ===")
    amp = amplification_over_gain_length()
    print(f"  g={amp['g_per_m']} /m over L={amp['length_m']} m: I/I0={amp['I_over_I0']:.1f} ({amp['gain_dB']:.1f} dB)")
    cmp = mirrorless_vs_cavity_threshold(N2_TYPICAL_GAIN_PER_M, 0.4)
    print(f"  threshold gain for a bare-window (R=0.04) cavity: {cmp['threshold_gain_per_m']:.2f} /m")
    print(f"  actual gain {cmp['actual_gain_per_m']} /m exceeds threshold: {cmp['exceeds_threshold']} "
          f"-> ASE/superradiant operation works without a resonant cavity")

    print_bom(BOM_DIY_N2_LASER, "DIY TEA N2 LASER (~$1000 budget)")
    feas = budget_feasibility()
    print(f"total ${feas['total_usd']:,} vs ${feas['budget_usd']:,.0f} budget: "
          f"{'within budget' if feas['within_budget'] else 'OVER budget'} "
          f"(margin ${feas['margin_usd']:,.0f})")
