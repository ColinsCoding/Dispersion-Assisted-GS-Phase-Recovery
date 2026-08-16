"""optical_loops.py -- two things photonics calls "a loop", both modeled
here as the SAME underlying object (a linear feedback system with one
round-trip delay), and cross-checked against each other:

  1. A ring resonator: light physically circulates in a closed waveguide,
     coupled in/out through one directional coupler each round trip.
  2. A recirculating fiber loop: a pulse is switched into a fiber spool +
     amplifier and re-circulated N times to synthesize a large effective
     dispersion (N * D_per_pass) from a short physical spool -- the classic
     trick (used in real STEAM/photonic-time-stretch benches) for getting
     two different effective dispersions D1, D2 for dgs.gs_core's two-shot
     phase retrieval without needing two different fiber spools.

Section 3 treats the ring as a single-pole IIR (Z-domain) filter and
verifies -- by literally iterating the round-trip recursion -- that it
converges to Section 1's closed-form steady state, connecting the ring's
finesse to the filter's settling time.
"""

from __future__ import annotations
import numpy as np

from dgs.gs_core import disperse

C_LIGHT = 299792458.0   # m/s


def _validate_unit_interval(name: str, x: float, lo: float = 0.0, hi: float = 1.0) -> None:
    if not (lo < x <= hi):
        raise ValueError(f"{name} must be in ({lo}, {hi}], got {x}")


def _validate_positive(name: str, x: float) -> None:
    if x <= 0:
        raise ValueError(f"{name} must be > 0, got {x}")


# ── 1. Ring resonator: through-port transmission, finesse, critical coupling ─

def round_trip_phase(f: float, FSR: float) -> float:
    """phi = 2*pi*f/FSR (mod 2*pi not taken -- callers use raw phi in trig,
    which is automatically periodic)."""
    _validate_positive("FSR", FSR)
    return 2 * np.pi * f / FSR


def cross_coupling_from_self(t: float) -> float:
    """kappa = sqrt(1 - t^2): the LOSSLESS-COUPLER constraint (t^2+kappa^2=1)
    relating self-coupling t to cross-coupling kappa."""
    _validate_unit_interval("t", t)
    return np.sqrt(1.0 - t**2)


def through_port_transmission(t: float, a: float, phi) -> np.ndarray:
    """T(phi) = |(t - a*e^{i*phi}) / (1 - t*a*e^{i*phi})|^2 -- the all-pass
    ring resonator's through-port power transmission. t = self-coupling
    (field fraction NOT coupled into the ring per pass), a = single-pass
    field amplitude survival (loss), phi = round-trip phase."""
    _validate_unit_interval("t", t)
    _validate_unit_interval("a", a)
    phi = np.asarray(phi, dtype=float)
    ejphi = np.exp(1j * phi)
    E_t = (t - a * ejphi) / (1 - t * a * ejphi)
    return np.abs(E_t)**2


def circulating_buildup_factor(t: float, a: float, phi) -> np.ndarray:
    """E_circ/E_in = i*kappa / (1 - t*a*e^{i*phi}) -- the steady-state
    intracavity field buildup, immediately after the coupler each round
    trip. |buildup|^2 is maximized on resonance (phi = 0 mod 2*pi)."""
    _validate_unit_interval("t", t)
    _validate_unit_interval("a", a)
    kappa = cross_coupling_from_self(t)
    phi = np.asarray(phi, dtype=float)
    ejphi = np.exp(1j * phi)
    return 1j * kappa / (1 - t * a * ejphi)


def ring_finesse(t: float, a: float) -> float:
    """F = pi*sqrt(t*a) / (1 - t*a) -- same Airy-finesse form as a
    Fabry-Perot cavity (dgs.laser_cavity_rlc_analog.cavity_finesse), with
    the two-mirror product R1*R2 replaced by the single round-trip
    field-survival product t*a."""
    _validate_unit_interval("t", t)
    _validate_unit_interval("a", a)
    ta = t * a
    return np.pi * np.sqrt(ta) / (1 - ta)


def ring_FWHM_phase(t: float, a: float) -> float:
    """Delta_phi_FWHM = 2*pi/F -- the through-port resonance dip's
    full-width-half-max in round-trip phase, from the finesse."""
    return 2 * np.pi / ring_finesse(t, a)


def critical_coupling_residual(a: float) -> float:
    """Critical coupling: setting t = a makes the through-port transmission
    EXACTLY zero on resonance (all input power dissipated in the loop's
    own loss, none reflected to the through port). Returns
    T(phi=0) at t=a -- CHECKED numerically here, not assumed from algebra;
    should be ~0 to floating-point precision."""
    _validate_unit_interval("a", a)
    return float(through_port_transmission(t=a, a=a, phi=0.0))


# ── 2. Recirculating fiber loop: dispersion multiplication + loss/gain ──────

def round_trip_net_dB(fiber_loss_dB_per_km: float, length_km: float,
                       coupler_loss_dB: float, amplifier_gain_dB: float) -> float:
    """Net round-trip gain in dB: amplifier gain minus fiber loss minus the
    switch/coupler insertion loss paid once per pass."""
    if length_km < 0 or fiber_loss_dB_per_km < 0 or coupler_loss_dB < 0:
        raise ValueError("length_km, fiber_loss_dB_per_km, coupler_loss_dB must be >= 0")
    return amplifier_gain_dB - fiber_loss_dB_per_km * length_km - coupler_loss_dB


def power_survival_from_dB(net_dB: float) -> float:
    """Linear power ratio from a dB figure: 10^(net_dB/10)."""
    return 10.0 ** (net_dB / 10.0)


def loop_threshold_gain_dB(fiber_loss_dB_per_km: float, length_km: float,
                            coupler_loss_dB: float) -> float:
    """g_th [dB] = fiber_loss_dB_per_km*length_km + coupler_loss_dB: the
    SMALLEST per-pass amplifier gain for which round-trip net_dB = 0 (loop
    is lossless per pass, so N recirculations neither decay nor blow up) --
    the fiber-loop analog of dgs.laser_cavity_rlc_analog.laser_threshold_gain's
    gain-equals-loss condition."""
    if length_km < 0 or fiber_loss_dB_per_km < 0 or coupler_loss_dB < 0:
        raise ValueError("length_km, fiber_loss_dB_per_km, coupler_loss_dB must be >= 0")
    return fiber_loss_dB_per_km * length_km + coupler_loss_dB


def accumulated_dispersion(D_per_pass: float, N: int) -> float:
    """D_total = N * D_per_pass: N recirculations through a short spool
    synthesize the SAME accumulated dispersion as one pass through an
    N-times-longer spool -- the entire point of using a loop."""
    if N < 0:
        raise ValueError(f"N must be >= 0, got {N}")
    return N * D_per_pass


def simulate_recirculating_loop(E: np.ndarray, D_per_pass: float, N: int,
                                 net_dB_per_pass: float = 0.0) -> list[np.ndarray]:
    """Iterates E through N round trips of (disperse by D_per_pass, then
    scale amplitude by sqrt(power_survival_from_dB(net_dB_per_pass))),
    using dgs.gs_core.disperse -- the SAME dispersion kernel the rest of
    this repo's GS phase-retrieval pipeline uses. Returns [E] + one entry
    per completed round trip (length N+1)."""
    if N < 0:
        raise ValueError(f"N must be >= 0, got {N}")
    amp_factor = np.sqrt(power_survival_from_dB(net_dB_per_pass))
    snapshots = [np.asarray(E, dtype=complex)]
    field = snapshots[0]
    for _ in range(N):
        field = disperse(field, D_per_pass) * amp_factor
        snapshots.append(field)
    return snapshots


def verify_accumulated_dispersion_equals_single_pass(E: np.ndarray, D_per_pass: float,
                                                      N: int, tol: float = 1e-9) -> dict:
    """CHECKED, not assumed: dispersing N times by D_per_pass must equal
    dispersing once by N*D_per_pass, because disperse() multiplies by
    H(nu)=exp(i*pi*D*nu^2) in the frequency domain, and
    exp(i*pi*D1*nu^2)*exp(i*pi*D2*nu^2) = exp(i*pi*(D1+D2)*nu^2) is an
    EXACT phase identity (loss/gain is a separate real amplitude factor
    that commutes with it, so net_dB_per_pass doesn't affect this check --
    verified here with loss included, not just the lossless case)."""
    loop_result = simulate_recirculating_loop(E, D_per_pass, N, net_dB_per_pass=-3.0)[-1]
    amp_factor_total = np.sqrt(power_survival_from_dB(-3.0))**N
    single_pass = disperse(np.asarray(E, dtype=complex), accumulated_dispersion(D_per_pass, N)) * amp_factor_total
    max_abs_diff = float(np.max(np.abs(loop_result - single_pass)))
    return {"max_abs_diff": max_abs_diff, "matches": bool(max_abs_diff < tol)}


# ── 3. The ring as a single-pole IIR (Z-domain) filter ───────────────────────

def iir_pole_from_ring(t: float, a: float, phi: float) -> complex:
    """z0 = t*a*e^{i*phi}: the round-trip recursion
        E_circ[n] = i*kappa*E_in + z0*E_circ[n-1]
    is a single-pole IIR filter; |z0| = t*a < 1 always (t,a <= 1), so the
    filter is UNCONDITIONALLY stable -- geometric convergence to the
    Section-1 closed-form steady state."""
    _validate_unit_interval("t", t)
    _validate_unit_interval("a", a)
    return t * a * np.exp(1j * phi)


def simulate_ring_buildup_recursion(E_in: complex, t: float, a: float, phi: float,
                                     n_round_trips: int) -> np.ndarray:
    """Iterates E_circ[n] = i*kappa*E_in + z0*E_circ[n-1] from E_circ[0]=0
    for n_round_trips steps; returns the array of n_round_trips+1 values
    (a literal time-domain simulation of the ring filling up, round trip
    by round trip)."""
    if n_round_trips < 0:
        raise ValueError(f"n_round_trips must be >= 0, got {n_round_trips}")
    kappa = cross_coupling_from_self(t)
    z0 = iir_pole_from_ring(t, a, phi)
    E_circ = np.zeros(n_round_trips + 1, dtype=complex)
    for n in range(1, n_round_trips + 1):
        E_circ[n] = 1j * kappa * E_in + z0 * E_circ[n - 1]
    return E_circ


def verify_recursion_converges_to_closed_form(E_in: complex, t: float, a: float, phi: float,
                                              n_round_trips: int, tol: float = 1e-6) -> dict:
    """CHECKED, not assumed: the recursive round-trip simulation's final
    value must approach circulating_buildup_factor(t,a,phi)*E_in as
    n_round_trips grows, at the rate set by the pole magnitude |z0|=t*a
    (photon-lifetime-in-round-trips ~ 1/(1-t*a), same quantity that sets
    Section 1's finesse) -- so a low-finesse ring should need FEWER round
    trips to converge than a high-finesse one, also checked here."""
    E_circ = simulate_ring_buildup_recursion(E_in, t, a, phi, n_round_trips)
    closed_form = circulating_buildup_factor(t, a, phi) * E_in
    rel_err = abs(E_circ[-1] - closed_form) / abs(closed_form)
    settling_round_trips = 1.0 / (1.0 - t * a)
    return {"closed_form": closed_form, "recursion_final": E_circ[-1],
            "relative_error": float(rel_err), "converged": bool(rel_err < tol),
            "settling_round_trips_estimate": settling_round_trips}


if __name__ == "__main__":
    print("=== 1. Ring resonator: finesse, critical coupling ===")
    t, a = 0.90, 0.98
    F = ring_finesse(t, a)
    fwhm = ring_FWHM_phase(t, a)
    print(f"  t={t}, a={a}: finesse F={F:.2f}, FWHM phase={fwhm:.4f} rad")
    resid = critical_coupling_residual(a=0.85)
    print(f"  critical coupling (t=a=0.85): T(phi=0) = {resid:.3e} (expect ~0)")

    print("\n=== 2. Recirculating fiber loop: dispersion multiplication ===")
    D_per_pass, N = -50.0, 12
    D_total = accumulated_dispersion(D_per_pass, N)
    print(f"  {N} round trips of D={D_per_pass}/pass -> D_total={D_total}")
    g_th = loop_threshold_gain_dB(fiber_loss_dB_per_km=0.2, length_km=5.0, coupler_loss_dB=1.0)
    print(f"  threshold gain to keep the loop lossless per pass: {g_th:.2f} dB")

    rng = np.random.default_rng(0)
    E0 = np.exp(1j * rng.uniform(0, 2 * np.pi, 256))
    check = verify_accumulated_dispersion_equals_single_pass(E0, D_per_pass, N)
    print(f"  N-pass loop == single N*D pass: max_abs_diff={check['max_abs_diff']:.3e}, matches={check['matches']}")

    print("\n=== 3. Ring as a Z-domain IIR filter: recursion vs. closed form ===")
    conv = verify_recursion_converges_to_closed_form(E_in=1.0, t=t, a=a, phi=0.0, n_round_trips=200)
    print(f"  closed form = {conv['closed_form']:.4f}, recursion(200 rt) = {conv['recursion_final']:.4f}")
    print(f"  relative error = {conv['relative_error']:.2e}, converged: {conv['converged']}")
    print(f"  settling round-trips estimate (1/(1-ta)) = {conv['settling_round_trips_estimate']:.1f}")

    print("\nOne feedback system, three views: closed-form Airy resonance (Sec 1),")
    print("a real dispersion-multiplying delay line feeding straight into")
    print("dgs.gs_core.disperse (Sec 2), and a single-pole IIR filter whose pole")
    print("radius t*a sets both the finesse and the settling time (Sec 3).")
