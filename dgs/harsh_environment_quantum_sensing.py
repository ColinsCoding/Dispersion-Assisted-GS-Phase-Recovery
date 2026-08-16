"""Harsh-environment quantum/photonic sensing: how three real environmental
stressors -- temperature, ionizing radiation, and mechanical vibration --
degrade a ring-resonator-based sensor, extending dgs.optical_loops's
EXISTING ring-resonator model directly (ring_finesse, ring_FWHM_phase,
loop_threshold_gain_dB) rather than building a parallel one.

THREE STRESSORS, each checked against the ring's own resonance width
(the actual quantity that matters for whether a sensor still works, not
just "is there some effect"):

1. THERMAL DRIFT: the thermo-optic effect (dn/dT) shifts a ring's
   resonance wavelength. Checked here against the resonance's own FWHM
   (derived from dgs.optical_loops.ring_FWHM_phase) to find the
   temperature swing at which the ring detunes by more than its own
   linewidth -- the real failure criterion, not an arbitrary threshold.

2. RADIATION-INDUCED ATTENUATION (RIA): ionizing radiation increases
   fiber/waveguide loss. A GENUINE SIZE-SCALE CONTRAST, both reusing
   dgs.optical_loops's existing functions unmodified: a compact
   MICRORING's round-trip path is only tens of microns, so RIA barely
   moves its finesse (checked via ring_finesse) even at a high dose;
   dgs.optical_loops's RECIRCULATING FIBER LOOP has a kilometer-scale
   path, where the SAME dose-dependent loss (fed into
   loop_threshold_gain_dB) demands a dramatically higher amplifier gain
   to stay lossless. Same physics, opposite outcome, purely from path
   length -- a real reason integrated photonics is often preferred for
   space/radiation environments over long fiber runs.

3. VIBRATION-INDUCED PHASE NOISE: mechanical vibration modulates the
   ring's effective optical path length, inducing phase jitter -- checked
   against ring_FWHM_phase the same way as the thermal case.

RIA dose-loss coefficients and vibration displacement amplitudes below are
ILLUSTRATIVE (chosen to land in literature-reported orders of magnitude
for radiation-hardened photonics and typical vibration-induced strain, not
citations of one specific measured device) -- flagged explicitly, not
asserted as measured values, matching this session's established honesty
norm for domain constants without a precise citation at hand.
"""

from __future__ import annotations
import numpy as np

from dgs.optical_loops import ring_finesse, ring_FWHM_phase, loop_threshold_gain_dB

C_LIGHT = 299792458.0


def _validate_positive(**kwargs) -> None:
    for name, value in kwargs.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


# ── 1. Thermal drift vs. the ring's own resonance width ─────────────────────

def ring_FSR_wavelength(radius_m: float, n_group: float, wavelength_m: float = 1550e-9) -> float:
    """FSR in wavelength units: FSR_lambda ~ lambda^2 / (n_group * L),
    L = 2*pi*radius (round-trip length)."""
    _validate_positive(radius_m=radius_m, n_group=n_group, wavelength_m=wavelength_m)
    L = 2 * np.pi * radius_m
    return wavelength_m**2 / (n_group * L)


def thermal_resonance_shift(delta_T_K: float, dn_dT: float = 1.8e-4, n_eff: float = 2.4,
                            wavelength_m: float = 1550e-9) -> float:
    """Resonance wavelength shift from a temperature change, via the
    thermo-optic effect: d(lambda)/lambda = d(n)/n_eff (to first order).
    Default dn_dT is silicon's well-documented thermo-optic coefficient
    (~1.8e-4 /K near room temperature)."""
    _validate_positive(n_eff=n_eff, wavelength_m=wavelength_m)
    delta_n = dn_dT * delta_T_K
    return wavelength_m * delta_n / n_eff


def verify_thermal_detuning_vs_linewidth(delta_T_K: float, t: float = 0.9, a: float = 0.98,
                                         radius_m: float = 10e-6, n_group: float = 4.2,
                                         dn_dT: float = 1.8e-4, n_eff: float = 2.4,
                                         wavelength_m: float = 1550e-9) -> dict:
    """CHECKED: compares the thermal resonance shift against the ring's
    own FWHM (in wavelength units, derived from
    dgs.optical_loops.ring_FWHM_phase and the FSR) -- the actual criterion
    for whether a temperature swing meaningfully detunes the sensor, not
    just a raw shift number left uninterpreted."""
    delta_lambda = thermal_resonance_shift(delta_T_K, dn_dT, n_eff, wavelength_m)
    FSR_lambda = ring_FSR_wavelength(radius_m, n_group, wavelength_m)
    fwhm_phase = ring_FWHM_phase(t, a)
    fwhm_lambda = fwhm_phase / (2 * np.pi) * FSR_lambda
    fraction_of_linewidth = abs(delta_lambda) / fwhm_lambda
    return {"delta_lambda_nm": delta_lambda * 1e9, "fwhm_lambda_nm": fwhm_lambda * 1e9,
            "fraction_of_linewidth": fraction_of_linewidth,
            "exceeds_linewidth": bool(fraction_of_linewidth > 1.0)}


# ── 2. Radiation-induced attenuation: microring vs. recirculating loop ─────

def microring_finesse_under_radiation(dose_krad: float, t: float = 0.9, a_baseline: float = 0.98,
                                      radius_m: float = 10e-6, ria_dB_per_km_per_krad: float = 5.0) -> dict:
    """CHECKED: a microring's finesse under radiation dose, via
    dgs.optical_loops.ring_finesse directly -- the extra round-trip loss
    from RIA is converted to an amplitude-survival factor and multiplied
    into `a`, showing how little a microring's TINY physical path length
    (tens of microns) is affected even at a substantial dose."""
    if dose_krad < 0:
        raise ValueError(f"dose_krad must be >= 0, got {dose_krad}")
    _validate_positive(a_baseline=a_baseline, radius_m=radius_m)
    L_round_trip_km = 2 * np.pi * radius_m / 1000.0
    extra_loss_dB = ria_dB_per_km_per_krad * dose_krad * L_round_trip_km
    extra_amplitude_factor = 10 ** (-extra_loss_dB / 20)
    a_degraded = a_baseline * extra_amplitude_factor
    F = ring_finesse(t, a_degraded)
    F_baseline = ring_finesse(t, a_baseline)
    return {"dose_krad": dose_krad, "extra_loss_dB": extra_loss_dB, "a_degraded": a_degraded,
            "finesse": F, "finesse_baseline": F_baseline,
            "finesse_relative_change": (F - F_baseline) / F_baseline}


def recirculating_loop_threshold_gain_under_radiation(dose_krad: float, length_km: float = 5.0,
                                                       coupler_loss_dB: float = 1.0,
                                                       baseline_fiber_loss_dB_per_km: float = 0.2,
                                                       ria_dB_per_km_per_krad: float = 5.0) -> dict:
    """CHECKED: a recirculating fiber loop's required threshold gain
    (dgs.optical_loops.loop_threshold_gain_dB, unmodified) under
    radiation dose -- the fiber's kilometer-scale path length makes RIA a
    dominant effect here, in sharp contrast to the microring case."""
    if dose_krad < 0:
        raise ValueError(f"dose_krad must be >= 0, got {dose_krad}")
    total_fiber_loss = baseline_fiber_loss_dB_per_km + ria_dB_per_km_per_krad * dose_krad
    g_th = loop_threshold_gain_dB(total_fiber_loss, length_km, coupler_loss_dB)
    g_th_baseline = loop_threshold_gain_dB(baseline_fiber_loss_dB_per_km, length_km, coupler_loss_dB)
    return {"dose_krad": dose_krad, "total_fiber_loss_dB_per_km": total_fiber_loss,
            "threshold_gain_dB": g_th, "threshold_gain_baseline_dB": g_th_baseline,
            "additional_gain_needed_dB": g_th - g_th_baseline}


# ── 3. Vibration-induced phase noise vs. the ring's own resonance width ────

def vibration_phase_jitter(displacement_amplitude_m: float, n_eff: float = 2.4,
                           wavelength_m: float = 1550e-9) -> float:
    """Phase jitter induced by a path-length modulation (from mechanical
    vibration/strain): dphi = 2*pi*n_eff*dL/lambda -- the same "extra
    optical path length" idea as thermal drift, now from mechanical
    displacement instead of temperature."""
    _validate_positive(n_eff=n_eff, wavelength_m=wavelength_m)
    if displacement_amplitude_m < 0:
        raise ValueError(f"displacement_amplitude_m must be >= 0, got {displacement_amplitude_m}")
    return 2 * np.pi * n_eff * displacement_amplitude_m / wavelength_m


def verify_vibration_jitter_vs_linewidth(displacement_amplitude_m: float, t: float = 0.9,
                                         a: float = 0.98, n_eff: float = 2.4,
                                         wavelength_m: float = 1550e-9) -> dict:
    """CHECKED: compares vibration-induced phase jitter against the
    ring's own FWHM phase width -- the same "is this bigger than the
    resonance itself" criterion as the thermal case."""
    dphi = vibration_phase_jitter(displacement_amplitude_m, n_eff, wavelength_m)
    fwhm_phase = ring_FWHM_phase(t, a)
    fraction = dphi / fwhm_phase
    return {"phase_jitter_rad": dphi, "fwhm_phase_rad": fwhm_phase,
            "fraction_of_linewidth": fraction, "exceeds_linewidth": bool(fraction > 1.0)}


if __name__ == "__main__":
    print("=== 1. Thermal drift vs. resonance linewidth ===")
    for dT in (0.5, 1.0, 3.0, 5.0, 10.0):
        check = verify_thermal_detuning_vs_linewidth(dT)
        print(f"  dT={dT:>5.1f}K: shift={check['delta_lambda_nm']:.4f}nm, "
              f"FWHM={check['fwhm_lambda_nm']:.4f}nm, "
              f"fraction={check['fraction_of_linewidth']:.2f}, exceeds: {check['exceeds_linewidth']}")

    print("\n=== 2. Radiation: microring (radiation-tolerant) vs. recirculating loop (vulnerable) ===")
    print("  Microring (path length ~63 um):")
    for dose in (0, 10, 50, 100, 500):
        r = microring_finesse_under_radiation(dose)
        print(f"    dose={dose:>4}krad: finesse={r['finesse']:.4f} "
              f"(baseline={r['finesse_baseline']:.4f}, change={r['finesse_relative_change']:.2e})")

    print("\n  Recirculating fiber loop (path length 5 km):")
    for dose in (0, 1, 5, 10, 20):
        r = recirculating_loop_threshold_gain_under_radiation(dose)
        print(f"    dose={dose:>4}krad: threshold_gain={r['threshold_gain_dB']:>7.2f}dB "
              f"(baseline={r['threshold_gain_baseline_dB']:.2f}dB, "
              f"+{r['additional_gain_needed_dB']:.2f}dB needed)")

    print("\n=== 3. Vibration-induced phase noise vs. resonance linewidth ===")
    for dL_nm in (0.1, 1.0, 5.0, 10.0, 20.0):
        check = verify_vibration_jitter_vs_linewidth(dL_nm * 1e-9)
        print(f"  dL={dL_nm:>5.1f}nm: jitter={check['phase_jitter_rad']:.4f}rad, "
              f"FWHM={check['fwhm_phase_rad']:.4f}rad, "
              f"fraction={check['fraction_of_linewidth']:.3f}, exceeds: {check['exceeds_linewidth']}")

    print("\nSame dgs.optical_loops ring model, stressed three physically different ways --")
    print("a compact microring survives radiation dose that would cripple a recirculating")
    print("fiber loop, purely from path length, the same math each time.")
