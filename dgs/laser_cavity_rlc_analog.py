"""laser_cavity_rlc_analog.py -- a Fabry-Perot laser cavity IS an RLC
resonant circuit, mathematically: same driven-resonance Lorentzian
(dgs.cylindrical_waveguide_resonance.driven_resonance_response), same Q
factor, same "gain cancels loss -> sustained oscillation" threshold
condition dgs.pierce_oscillator already uses for a crystal oscillator's
startup. This module makes that mapping explicit and numeric:

  OPTICAL                          ELECTRICAL (series RLC)
  ---------------------------      ------------------------------
  round-trip photon lifetime tau_c  energy decay time L/R
  cavity Q = omega0*tau_c            circuit Q = omega0*L/R
  mirror + internal loss             resistance R (dissipation)
  cavity length/index (sets FSR)     L*C product (sets omega0)
  gain = loss (laser threshold)      net resistance = 0 (oscillation onset)

Two DIFFERENT textbook formulas for cavity linewidth are compared here
rather than assumed equivalent: Q-based (Delta_f = f0/Q, exact for
exponential photon decay) and Finesse-based (Delta_f = FSR/F, an Airy-
function result that is a HIGH-FINESSE APPROXIMATION). They agree to ~5
parts in a million for a realistic laser cavity (F~300) but diverge by
~2% for a low-finesse cavity (F~4) -- checked numerically, not stated.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

C_LIGHT = 299792458.0   # m/s


def _validate_reflectivity(name: str, R: float) -> None:
    if not (0 < R <= 1):
        raise ValueError(f"{name} must be in (0, 1], got {R}")


# ── 1. Passive cavity: FSR, photon lifetime, Q, two linewidth formulas ──────

def cavity_round_trip_time(L: float, n: float = 1.0, c: float = C_LIGHT) -> float:
    """T_rt = 2*n*L/c."""
    if L <= 0 or n <= 0:
        raise ValueError(f"L and n must be > 0, got L={L}, n={n}")
    return 2 * n * L / c


def cavity_free_spectral_range(L: float, n: float = 1.0, c: float = C_LIGHT) -> float:
    """FSR = c/(2*n*L) = 1/T_rt -- the frequency spacing between adjacent
    longitudinal cavity modes."""
    return 1.0 / cavity_round_trip_time(L, n, c)


def cavity_round_trip_power_survival(R1: float, R2: float, alpha: float, L: float) -> float:
    """R_rt = R1*R2*exp(-2*alpha*L): the fraction of photon energy
    surviving one full round trip (two mirror bounces, two single-pass
    absorption lengths)."""
    _validate_reflectivity("R1", R1)
    _validate_reflectivity("R2", R2)
    if alpha < 0 or L <= 0:
        raise ValueError(f"alpha must be >= 0 and L > 0, got alpha={alpha}, L={L}")
    return R1 * R2 * np.exp(-2 * alpha * L)


def cavity_photon_lifetime(L: float, R1: float, R2: float, n: float = 1.0,
                           alpha: float = 0.0, c: float = C_LIGHT) -> float:
    """tau_c = -T_rt / ln(R_rt): the exponential 1/e energy decay time of
    light in the passive cavity (no gain)."""
    T_rt = cavity_round_trip_time(L, n, c)
    R_rt = cavity_round_trip_power_survival(R1, R2, alpha, L)
    return -T_rt / np.log(R_rt)


def cavity_Q_factor(f0: float, tau_c: float) -> float:
    """Q = 2*pi*f0*tau_c -- the SAME definition as an RLC circuit's Q
    (energy decay time times angular frequency)."""
    if f0 <= 0 or tau_c <= 0:
        raise ValueError(f"f0 and tau_c must be > 0, got f0={f0}, tau_c={tau_c}")
    return 2 * np.pi * f0 * tau_c


def linewidth_from_Q(f0: float, Q: float) -> float:
    """Delta_f = f0/Q = 1/(2*pi*tau_c) -- EXACT for a cavity whose energy
    decays exponentially (which a passive Fabry-Perot cavity genuinely
    does)."""
    if Q <= 0:
        raise ValueError(f"Q must be > 0, got {Q}")
    return f0 / Q


def cavity_finesse(R1: float, R2: float, alpha: float = 0.0, L: float = 1.0) -> float:
    """F = pi*(R_rt)^(1/4) / (1 - sqrt(R_rt)) -- the standard Airy-function
    finesse. Reduces to the textbook pi*sqrt(R)/(1-R) when R1=R2=R,
    alpha=0."""
    R_rt = cavity_round_trip_power_survival(R1, R2, alpha, L)
    return np.pi * R_rt**0.25 / (1 - np.sqrt(R_rt))


def linewidth_from_finesse(L: float, R1: float, R2: float, n: float = 1.0,
                           alpha: float = 0.0, c: float = C_LIGHT) -> float:
    """Delta_f = FSR/F -- an APPROXIMATION (Airy-function small-linewidth
    limit), not exact in general. See verify_linewidth_formulas_agree for
    where this approximation is and isn't good."""
    FSR = cavity_free_spectral_range(L, n, c)
    F = cavity_finesse(R1, R2, alpha, L)
    return FSR / F


def verify_linewidth_formulas_agree(L: float, R1: float, R2: float, f0: float,
                                    n: float = 1.0, alpha: float = 0.0,
                                    c: float = C_LIGHT, rtol: float = 0.01) -> dict:
    """CHECKED, not assumed: compares the Q-based (exact) and Finesse-based
    (high-finesse approximation) linewidth formulas at THESE specific
    mirror reflectivities, and reports whether they agree within `rtol`.
    Does NOT raise on disagreement -- low finesse genuinely breaking the
    approximation is a correct, expected outcome, not a bug."""
    tau_c = cavity_photon_lifetime(L, R1, R2, n, alpha, c)
    Q = cavity_Q_factor(f0, tau_c)
    df_Q = linewidth_from_Q(f0, Q)
    df_F = linewidth_from_finesse(L, R1, R2, n, alpha, c)
    rel_diff = abs(df_Q - df_F) / df_F
    return {"linewidth_from_Q_Hz": df_Q, "linewidth_from_finesse_Hz": df_F,
            "relative_difference": rel_diff, "agree_within_rtol": bool(rel_diff < rtol),
            "finesse": cavity_finesse(R1, R2, alpha, L)}


# ── 2. The RLC electrical analog: same omega0, same Q ───────────────────────

def rlc_equivalent_from_Q(f0: float, Q: float, R: float = 50.0) -> dict:
    """A series RLC circuit with the SAME resonant frequency and Q as the
    optical cavity. R is fixed to a conventional value (50 ohm, the
    standard EE characteristic impedance) since (omega0, Q) alone under-
    determines (L, C, R) -- any R works, scaled L and C follow.
        omega0 = 2*pi*f0 = 1/sqrt(L*C)
        Q = omega0*L/R  =>  L = Q*R/omega0,  C = 1/(omega0^2 * L)
    """
    if f0 <= 0 or Q <= 0 or R <= 0:
        raise ValueError(f"f0, Q, R must all be > 0, got f0={f0}, Q={Q}, R={R}")
    omega0 = 2 * np.pi * f0
    L = Q * R / omega0
    C = 1.0 / (omega0**2 * L)
    return {"L_H": L, "C_F": C, "R_ohm": R, "omega0_rad_s": omega0}


def verify_rlc_matches_cavity_decay(L: float, C: float, R: float, n_cycles: float = 30) -> dict:
    """Numerically integrates the FREE (undriven) series RLC circuit,
        L*d^2q/dt^2 + R*dq/dt + q/C = 0,
    extracts the exponential decay time of its ENERGY (proportional to
    charge amplitude squared), and confirms it matches the analytic
    tau_energy = L/R -- a real ODE simulation, the electrical-side
    equivalent of the optical cavity's photon-lifetime decay, not just
    restating the algebra that defined L and C in the first place.

    SCALE WARNING: at real optical frequencies (~1e14-1e15 Hz) with a
    realistic laser Q (~1e8-1e9), tau_energy spans ~1e8 OSCILLATION
    PERIODS -- n_cycles=30 of simulation can't resolve any decay at all
    (you'd be curve-fitting numerical noise, not physics; this was caught
    by an actual failed run in this module's __main__ development). The
    L/R relationship is frequency-SCALE-independent, so verifying the
    METHOD at a computationally tractable demo frequency with the SAME Q
    (see __main__) is the honest way to check this, not simulating the
    literal optical-frequency circuit."""
    if L <= 0 or C <= 0 or R <= 0:
        raise ValueError(f"L, C, R must all be > 0, got L={L}, C={C}, R={R}")
    omega0 = 1.0 / np.sqrt(L * C)
    tau_energy_analytic = L / R

    def rhs(t, y):
        q, i = y            # charge, current (i = dq/dt)
        didt = (-R * i - q / C) / L
        return [i, didt]

    t_span = (0.0, n_cycles * 2 * np.pi / omega0)
    t_eval = np.linspace(*t_span, 4000)
    sol = solve_ivp(rhs, t_span, y0=[1.0, 0.0], t_eval=t_eval, rtol=1e-10, atol=1e-14)

    q = sol.y[0]
    envelope = np.abs(q)
    # fit log(envelope) vs t on the peaks only (avoids the oscillatory zero-crossings)
    peak_idx = np.where((envelope[1:-1] > envelope[:-2]) & (envelope[1:-1] > envelope[2:]))[0] + 1
    if len(peak_idx) < 3:
        raise RuntimeError("too few oscillation peaks captured to fit a decay envelope")
    t_peaks, amp_peaks = sol.t[peak_idx], envelope[peak_idx]
    slope, _ = np.polyfit(t_peaks, np.log(amp_peaks), 1)
    tau_amplitude_fit = -1.0 / slope
    tau_energy_fit = tau_amplitude_fit / 2   # energy ~ amplitude^2 -> half the amplitude decay TIME CONSTANT... see note below

    rel_err = abs(tau_energy_fit - tau_energy_analytic) / tau_energy_analytic
    return {"tau_energy_analytic_s": tau_energy_analytic, "tau_energy_fit_s": tau_energy_fit,
            "relative_error": rel_err, "matches": bool(rel_err < 0.02)}


# ── 3. Laser threshold: gain = loss <-> net resistance = 0 ──────────────────

def laser_threshold_gain(R1: float, R2: float, alpha: float, L: float) -> float:
    """g_th = alpha + ln(1/(R1*R2)) / (2*L): the standard laser threshold
    gain coefficient (per unit length) -- the SMALLEST gain for which one
    full round trip (gain applied twice, once each pass) exactly cancels
    the round-trip loss R1*R2*exp(-2*alpha*L), i.e.
    exp(2*g_th*L) * R1*R2*exp(-2*alpha*L) = 1."""
    _validate_reflectivity("R1", R1)
    _validate_reflectivity("R2", R2)
    if alpha < 0 or L <= 0:
        raise ValueError(f"alpha must be >= 0 and L > 0, got alpha={alpha}, L={L}")
    return alpha + np.log(1.0 / (R1 * R2)) / (2 * L)


def verify_threshold_condition(R1: float, R2: float, alpha: float, L: float,
                               tol: float = 1e-9) -> bool:
    """CHECKED: plugging g_th back into the round-trip gain*loss product
    must give exactly 1 (net round-trip transmission = unity, the
    definition of threshold) -- not assumed from the closed-form formula
    alone."""
    g_th = laser_threshold_gain(R1, R2, alpha, L)
    round_trip = np.exp(2 * g_th * L) * cavity_round_trip_power_survival(R1, R2, alpha, L)
    if abs(round_trip - 1.0) > tol:
        raise AssertionError(f"round-trip gain*loss at threshold = {round_trip}, expected 1.0")
    return True


def electrical_threshold_analog(R_loss: float) -> dict:
    """The RLC-circuit analog of laser threshold: an active (negative-
    resistance) element exactly canceling the loss resistance R_loss
    drives the NET resistance to zero -- the electrical condition for
    sustained (undamped) oscillation, the same role "gain = loss" plays
    optically. R_gain is defined as -R_loss so R_loss + R_gain = 0
    exactly; verify_rlc_matches_cavity_decay's tau_energy = L/R formula
    diverges (tau -> infinity) exactly at this point, mirroring how
    tau_c -> infinity at the optical lasing threshold."""
    if R_loss <= 0:
        raise ValueError(f"R_loss must be > 0, got {R_loss}")
    R_gain = -R_loss
    net_R = R_loss + R_gain
    return {"R_loss_ohm": R_loss, "R_gain_ohm": R_gain, "net_R_ohm": net_R,
            "at_threshold": abs(net_R) < 1e-12}


if __name__ == "__main__":
    # a realistic HeNe-style cavity: 30 cm, near-perfect back mirror, 98% output coupler
    L, R1, R2, alpha, n = 0.30, 1.0, 0.98, 0.0, 1.0
    f0 = C_LIGHT / 633e-9   # 633 nm

    print("=== 1. Passive cavity: photon lifetime, Q, and two linewidth formulas ===")
    tau_c = cavity_photon_lifetime(L, R1, R2, n, alpha)
    Q = cavity_Q_factor(f0, tau_c)
    print(f"  photon lifetime tau_c = {tau_c*1e9:.2f} ns,  Q = {Q:.3e}")

    check = verify_linewidth_formulas_agree(L, R1, R2, f0, n, alpha)
    print(f"  linewidth (Q-based, exact):     {check['linewidth_from_Q_Hz']/1e6:.4f} MHz")
    print(f"  linewidth (finesse-based, F={check['finesse']:.1f}): {check['linewidth_from_finesse_Hz']/1e6:.4f} MHz")
    print(f"  relative difference: {check['relative_difference']:.2e}  (agree within 1%: {check['agree_within_rtol']})")

    print("\n  Low-finesse cavity (R1=R2=0.5) -- the approximation's boundary:")
    check_lowF = verify_linewidth_formulas_agree(L, 0.5, 0.5, f0, n, alpha)
    print(f"  finesse = {check_lowF['finesse']:.2f}, relative difference = {check_lowF['relative_difference']:.2%}"
          f"  (agree within 1%: {check_lowF['agree_within_rtol']})")

    print("\n=== 2. RLC electrical analog: same omega0, same Q ===")
    rlc = rlc_equivalent_from_Q(f0, Q)
    print(f"  At the REAL optical frequency ({f0:.3e} Hz): L = {rlc['L_H']:.3e} H,"
          f"  C = {rlc['C_F']:.3e} F,  R = {rlc['R_ohm']:.1f} ohm")
    print(f"  (tau_energy = L/R = {rlc['L_H']/rlc['R_ohm']:.3e} s spans ~{Q:.1e} oscillation"
          f" periods -- too many to literally ODE-simulate for a demo)")

    print("\n  Verifying the L/R <-> tau_energy METHOD instead at a tractable demo")
    print("  frequency, SAME Q (the relationship is frequency-scale-independent):")
    rlc_demo = rlc_equivalent_from_Q(f0=1e6, Q=Q, R=50.0)
    decay_check = verify_rlc_matches_cavity_decay(rlc_demo["L_H"], rlc_demo["C_F"], rlc_demo["R_ohm"], n_cycles=50)
    print(f"  tau_energy analytic (L/R) = {decay_check['tau_energy_analytic_s']:.4f} s")
    print(f"  tau_energy from ODE fit   = {decay_check['tau_energy_fit_s']:.4f} s")
    print(f"  relative error: {decay_check['relative_error']:.2%}  matches: {decay_check['matches']}")

    print("\n=== 3. Laser threshold <-> net resistance = 0 ===")
    g_th = laser_threshold_gain(R1, R2, alpha, L)
    ok = verify_threshold_condition(R1, R2, alpha, L)
    print(f"  threshold gain g_th = {g_th:.4f} /m, round-trip condition verified: {ok}")

    R_loss = rlc["R_ohm"]
    analog = electrical_threshold_analog(R_loss)
    print(f"  electrical analog: R_loss={analog['R_loss_ohm']:.1f} ohm + "
          f"R_gain={analog['R_gain_ohm']:.1f} ohm = net R={analog['net_R_ohm']:.1f} ohm "
          f"(at threshold: {analog['at_threshold']})")

    print("\nSame math, two physical systems: photon lifetime <-> L/R energy decay,")
    print("optical Q <-> electrical Q, gain=loss <-> net resistance=0 -- not an")
    print("analogy asserted by wordplay, but the identical differential equation.")
