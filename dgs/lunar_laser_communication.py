"""Lunar optical (laser) communication link budget: why does deep-space
communication move to photonics at all, and roughly how does the
resulting data rate compare to a real, publicly documented demonstration?

Concrete, real reference point: NASA's Lunar Laser Communication
Demonstration (LLCD, flown on LADEE in 2013) publicly demonstrated a
622 Mbps downlink from lunar distance -- the first such optical
communication link to/from the Moon. That figure is used below only as
an order-of-magnitude sanity check against an idealized photon-budget
model built from first principles; this module does NOT claim to know
LLCD's actual transmitter power, aperture, or coding scheme -- those are
treated as illustrative, plausible parameters for a small spacecraft
optical terminal, not asserted as the real hardware's specs.

Reuses dgs.quantum_internet_link_budget's diffraction-limited free-space
geometric loss model directly (same physics, same functions) rather than
reimplementing it, and dgs.special_relativity.C_SI for the speed of
light.
"""

import numpy as np

from dgs.special_relativity import C_SI
from dgs.quantum_internet_link_budget import (
    diffraction_divergence_half_angle_rad,
    free_space_geometric_loss_db,
    transmittance_from_db,
)

PLANCK_H_J_S = 6.62607015e-34   # exact, SI-defined since 2019

# widely-cited mean/perigee/apogee Earth-Moon distances
MOON_MEAN_DISTANCE_M = 384_400e3
MOON_PERIGEE_M = 363_300e3
MOON_APOGEE_M = 405_500e3

LLCD_PUBLIC_DOWNLINK_BPS = 622e6   # publicly documented LLCD (2013) demonstrated downlink rate


def one_way_light_time_s(distance_m):
    """Signal travel time Earth<->Moon: distance/c. At mean distance this
    is close to the widely-cited ~1.28s figure."""
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    return distance_m / C_SI


def photon_energy_j(wavelength_m):
    """E = hc/lambda -- the energy of a single photon at this wavelength."""
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    return PLANCK_H_J_S * C_SI / wavelength_m


def received_power_w(tx_power_w, distance_m, wavelength_m, tx_aperture_m,
                      rx_aperture_m, extra_loss_db=0.0):
    """Received optical power after diffraction-limited free-space geometric
    loss (reusing dgs.quantum_internet_link_budget's model directly) plus
    any additional fixed loss budget (pointing, optics, atmosphere) in dB."""
    if tx_power_w <= 0:
        raise ValueError("tx_power_w must be positive")
    if extra_loss_db < 0:
        raise ValueError("extra_loss_db must be non-negative")
    geometric_loss_db = free_space_geometric_loss_db(
        distance_m, wavelength_m, tx_aperture_m, rx_aperture_m)
    total_loss_db = geometric_loss_db + extra_loss_db
    return tx_power_w * transmittance_from_db(total_loss_db), geometric_loss_db


def idealized_photon_limited_rate_bps(received_power_w_val, wavelength_m, photons_per_bit=1.0):
    """Idealized CEILING on data rate: how many bits/s could be sent if
    every detected photon carried photons_per_bit worth of information,
    with perfect detection and zero coding overhead. This is an upper
    bound, not an engineering claim -- a real system needs many more
    photons/bit (coding overhead, detector inefficiency, background noise)
    to close a link reliably, so a real achievable rate sits well below
    this ceiling."""
    if received_power_w_val <= 0:
        raise ValueError("received_power_w_val must be positive")
    if photons_per_bit <= 0:
        raise ValueError("photons_per_bit must be positive")
    e_photon = photon_energy_j(wavelength_m)
    photon_rate_hz = received_power_w_val / e_photon
    return photon_rate_hz / photons_per_bit


def verify_against_llcd_public_figure(distance_m=MOON_MEAN_DISTANCE_M,
                                       wavelength_m=1550e-9,
                                       tx_power_w=2.0,
                                       tx_aperture_m=0.1,
                                       rx_aperture_m=0.4,
                                       extra_loss_db=6.0,
                                       photons_per_bit=2.0):
    """Build an idealized photon-budget rate from illustrative (not
    asserted-as-real) small-terminal parameters, and compare its order of
    magnitude against the publicly documented LLCD 622 Mbps downlink.
    Expect the idealized ceiling to land ABOVE the real demonstrated rate
    (a real system needs coding overhead and margin this idealized model
    doesn't include), but within a small number of orders of magnitude --
    not wildly off, which would indicate a modeling error rather than a
    reasonable idealization gap."""
    p_rx, geometric_loss_db = received_power_w(
        tx_power_w, distance_m, wavelength_m, tx_aperture_m, rx_aperture_m, extra_loss_db)
    ceiling_bps = idealized_photon_limited_rate_bps(p_rx, wavelength_m, photons_per_bit)
    ratio = ceiling_bps / LLCD_PUBLIC_DOWNLINK_BPS
    return {
        "received_power_w": p_rx,
        "geometric_loss_db": geometric_loss_db,
        "idealized_ceiling_bps": ceiling_bps,
        "llcd_public_downlink_bps": LLCD_PUBLIC_DOWNLINK_BPS,
        "ratio_ceiling_to_llcd": ratio,
        "ceiling_above_real_rate": ceiling_bps > LLCD_PUBLIC_DOWNLINK_BPS,
        "same_order_of_magnitude_regime": 1.0 <= ratio <= 100.0,
    }


def compare_optical_vs_rf_geometric_loss(distance_m, aperture_m,
                                          optical_wavelength_m=1550e-9,
                                          rf_wavelength_m=9.4e-3):
    """Same-aperture comparison of optical vs. Ka-band-RF (32 GHz,
    lambda~9.4mm) diffraction-limited geometric loss, isolating the ONE
    variable that matters here: wavelength. This is NOT a claim that real
    deep-space RF terminals use a small aperture like this -- real systems
    (e.g. NASA's Deep Space Network 34m/70m dishes) compensate for RF's
    much wider diffraction divergence with much larger apertures. Holding
    aperture fixed here isolates why, at EQUAL aperture, going optical
    buys enormously more geometric link margin.

    NOTE: return values are explicitly cast with bool()/float(). An
    earlier version returned numpy's np.bool_/np.float64 straight out of
    the np.log10-based comparison -- value-correct, but np.bool_ is never
    `is True` (numpy scalars aren't identical to the Python singleton),
    which this module's own test caught via `optical_wins is True`."""
    if aperture_m <= 0:
        raise ValueError("aperture_m must be positive")
    optical_loss_db = free_space_geometric_loss_db(
        distance_m, optical_wavelength_m, aperture_m, aperture_m)
    rf_loss_db = free_space_geometric_loss_db(
        distance_m, rf_wavelength_m, aperture_m, aperture_m)
    return {
        "optical_loss_db": float(optical_loss_db),
        "rf_loss_db": float(rf_loss_db),
        "optical_advantage_db": float(rf_loss_db - optical_loss_db),
        "optical_wins": bool(optical_loss_db < rf_loss_db),
    }


if __name__ == "__main__":
    t_light = one_way_light_time_s(MOON_MEAN_DISTANCE_M)
    print("=== 1. Earth-Moon light time ===")
    print(f"  mean distance: {MOON_MEAN_DISTANCE_M/1e3:.0f} km")
    print(f"  one-way light time: {t_light:.3f} s  (widely-cited reference value: ~1.28 s)")

    print("\n=== 2. Idealized photon-limited link budget vs. public LLCD figure ===")
    check = verify_against_llcd_public_figure()
    print(f"  geometric loss: {check['geometric_loss_db']:.1f} dB")
    print(f"  received power: {check['received_power_w']:.3e} W")
    print(f"  idealized rate ceiling: {check['idealized_ceiling_bps']/1e6:.1f} Mbps")
    print(f"  public LLCD (2013) downlink: {check['llcd_public_downlink_bps']/1e6:.0f} Mbps")
    print(f"  ceiling / LLCD ratio: {check['ratio_ceiling_to_llcd']:.2f}x")
    print(f"  ceiling above real demonstrated rate: {check['ceiling_above_real_rate']}")
    print(f"  same order-of-magnitude regime (1x-100x): {check['same_order_of_magnitude_regime']}")

    print("\n=== 3. Why optical: same-aperture loss comparison vs. Ka-band RF ===")
    rf_compare = compare_optical_vs_rf_geometric_loss(MOON_MEAN_DISTANCE_M, aperture_m=0.1)
    print(f"  optical (1550nm) geometric loss: {rf_compare['optical_loss_db']:.1f} dB")
    print(f"  RF (Ka-band, 32GHz) geometric loss: {rf_compare['rf_loss_db']:.1f} dB")
    print(f"  optical advantage: {rf_compare['optical_advantage_db']:.1f} dB "
          f"(at EQUAL, small aperture -- real RF systems close this gap with much bigger dishes)")
