"""Entangled-photon quantum-internet link budget: is direct fiber-based
entanglement distribution between two real, specific sites geometrically
and physically feasible, or does it need a satellite/free-space link (or
quantum repeaters) instead?

Concrete scenario used throughout: UC Merced to UC Riverside, two real UC
campuses, distance computed via the haversine great-circle formula (not
assumed). Two competing physical channels are modeled and compared:

  * TELECOM FIBER: attenuation is EXPONENTIAL in distance (dB loss scales
    linearly with km, so transmittance scales as 10^(-alpha*L/10) --
    exponential decay). Standard C-band (1550 nm) fiber loses ~0.2 dB/km.
  * FREE-SPACE / SATELLITE DOWNLINK: loss is dominated by diffraction
    (Rayleigh criterion beam divergence), which scales as a POWER LAW in
    distance (receiver captures a 1/L^2 fraction of the diverging spot),
    not exponentially. This is why real continental/intercontinental
    quantum-key-distribution demonstrations (e.g. satellite-to-ground
    entanglement distribution) use free-space links for long haul, while
    fiber dominates only for short/metro distances.

Reuses dgs.special_relativity.C_SI for the speed of light rather than
redefining it.
"""

import numpy as np

from dgs.special_relativity import C_SI

EARTH_RADIUS_KM = 6371.0
FIBER_ATTEN_DB_PER_KM_1550NM = 0.2   # standard C-band telecom fiber, e.g. Corning SMF-28 spec
FIBER_CORE_INDEX_1550NM = 1.468       # silica core refractive index at 1550 nm


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two (lat, lon) points in degrees,
    via the haversine formula on a spherical Earth of radius EARTH_RADIUS_KM."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def fiber_route_distance_km(great_circle_km, route_factor=1.4):
    """Actual buried-fiber route length is always longer than the great-circle
    distance (roads, right-of-way, terrain detours). route_factor=1.4 is a
    commonly cited real-world terrestrial-fiber routing overhead estimate."""
    if great_circle_km < 0:
        raise ValueError("great_circle_km must be non-negative")
    if route_factor < 1.0:
        raise ValueError("route_factor must be >= 1.0 (fiber can't be shorter than great-circle)")
    return great_circle_km * route_factor


def fiber_loss_db(distance_km, alpha_db_per_km=FIBER_ATTEN_DB_PER_KM_1550NM):
    """Total fiber attenuation in dB: linear in distance (the defining
    property of exponential-in-distance power loss)."""
    if distance_km < 0:
        raise ValueError("distance_km must be non-negative")
    if alpha_db_per_km <= 0:
        raise ValueError("alpha_db_per_km must be positive")
    return alpha_db_per_km * distance_km


def transmittance_from_db(loss_db):
    """Convert a dB power loss into a linear transmittance fraction (0, 1]."""
    return float(10 ** (-loss_db / 10.0))


def fiber_transit_time_s(distance_km, n_fiber=FIBER_CORE_INDEX_1550NM):
    """Photon transit time through fiber of given length: t = n*L/c, since
    light slows to v = c/n inside the fiber core (NOT the vacuum c)."""
    if distance_km < 0:
        raise ValueError("distance_km must be non-negative")
    if n_fiber < 1.0:
        raise ValueError("n_fiber must be >= 1.0")
    return n_fiber * (distance_km * 1e3) / C_SI


def entangled_pair_detection_rate_hz(source_rate_hz, one_way_loss_db,
                                      detector_efficiency=0.2):
    """Detected coincidence rate for a midpoint entangled-pair source sending
    one photon each way to two end stations, each link independently lossy.
    Both photons must survive AND be detected for a coincidence count, so the
    two independent link transmittances multiply (loss budgets ADD in dB)."""
    if source_rate_hz <= 0:
        raise ValueError("source_rate_hz must be positive")
    if not (0 < detector_efficiency <= 1):
        raise ValueError("detector_efficiency must be in (0, 1]")
    T_link = transmittance_from_db(one_way_loss_db)
    return source_rate_hz * (T_link * detector_efficiency) ** 2


def diffraction_divergence_half_angle_rad(wavelength_m, tx_aperture_diameter_m):
    """Rayleigh-criterion diffraction-limited half-angle divergence of a
    free-space (satellite/ground telescope) transmitter: theta = 1.22*lambda/D.
    This is the SAME diffraction physics as dgs.paraxial_optics_abcd's
    Gaussian-beam treatment, applied here to a link-budget geometry."""
    if wavelength_m <= 0 or tx_aperture_diameter_m <= 0:
        raise ValueError("wavelength_m and tx_aperture_diameter_m must be positive")
    return 1.22 * wavelength_m / tx_aperture_diameter_m


def free_space_geometric_loss_db(distance_m, wavelength_m, tx_aperture_diameter_m,
                                  rx_aperture_diameter_m):
    """Diffraction-limited geometric loss for a free-space link: the
    transmitted beam spreads to a spot of radius L*theta at the receiver,
    which only captures the (D_rx / spot_diameter)^2 fraction of the total
    power -- a POWER LAW (1/L^2) loss, fundamentally different scaling from
    fiber's EXPONENTIAL loss. Returns loss in dB (positive = power lost)."""
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    if rx_aperture_diameter_m <= 0:
        raise ValueError("rx_aperture_diameter_m must be positive")
    theta = diffraction_divergence_half_angle_rad(wavelength_m, tx_aperture_diameter_m)
    spot_diameter_m = 2 * distance_m * theta
    capture_fraction = (rx_aperture_diameter_m / spot_diameter_m) ** 2
    capture_fraction = min(capture_fraction, 1.0)
    return -10 * np.log10(capture_fraction)


def compare_fiber_vs_satellite(ground_distance_km, satellite_altitude_km=500.0,
                                wavelength_m=1550e-9, tx_aperture_m=0.3,
                                rx_aperture_m=1.0, fiber_route_factor=1.4,
                                fiber_alpha_db_per_km=FIBER_ATTEN_DB_PER_KM_1550NM,
                                atmospheric_extra_loss_db=3.0):
    """Head-to-head link budget for the same two ground sites: direct buried
    fiber vs. a satellite relay at satellite_altitude_km overhead. Returns
    both total losses (dB) and transmittances, plus which channel wins."""
    if ground_distance_km <= 0:
        raise ValueError("ground_distance_km must be positive")
    if satellite_altitude_km <= 0:
        raise ValueError("satellite_altitude_km must be positive")

    fiber_km = fiber_route_distance_km(ground_distance_km, fiber_route_factor)
    fiber_db = fiber_loss_db(fiber_km, fiber_alpha_db_per_km)

    # slant range site->satellite->site, each leg ~satellite_altitude_km
    # (a flat-Earth approximation for a satellite roughly overhead of the
    # midpoint; adequate for an order-of-magnitude link-budget comparison)
    slant_range_m = satellite_altitude_km * 1e3
    one_leg_db = free_space_geometric_loss_db(slant_range_m, wavelength_m,
                                               tx_aperture_m, rx_aperture_m)
    satellite_db = 2 * one_leg_db + atmospheric_extra_loss_db  # uplink + downlink legs

    return {
        "ground_distance_km": ground_distance_km,
        "fiber_route_km": fiber_km,
        "fiber_loss_db": fiber_db,
        "fiber_transmittance": transmittance_from_db(fiber_db),
        "satellite_loss_db": satellite_db,
        "satellite_transmittance": transmittance_from_db(satellite_db),
        "satellite_wins": satellite_db < fiber_db,
    }


def repeater_spacing_for_budget_km(total_distance_km, max_span_loss_db,
                                    alpha_db_per_km=FIBER_ATTEN_DB_PER_KM_1550NM):
    """How many equal-length quantum-repeater segments are needed to keep
    each individual fiber span's loss under max_span_loss_db? Returns
    (n_segments, span_length_km, span_loss_db)."""
    if total_distance_km <= 0:
        raise ValueError("total_distance_km must be positive")
    if max_span_loss_db <= 0:
        raise ValueError("max_span_loss_db must be positive")
    max_span_km = max_span_loss_db / alpha_db_per_km
    n_segments = int(np.ceil(total_distance_km / max_span_km))
    span_km = total_distance_km / n_segments
    span_loss_db = fiber_loss_db(span_km, alpha_db_per_km)
    return n_segments, span_km, span_loss_db


if __name__ == "__main__":
    UCM = (37.3661, -120.4269)   # UC Merced
    UCR = (33.9737, -117.3281)   # UC Riverside

    d = haversine_distance_km(*UCM, *UCR)
    result = compare_fiber_vs_satellite(d)
    n_seg, span_km, span_loss = repeater_spacing_for_budget_km(d, max_span_loss_db=20.0)

    print("Entangled-photon quantum internet: UC Merced <-> UC Riverside")
    print(f"great-circle distance:  {d:.1f} km")
    print(f"realistic fiber route:  {result['fiber_route_km']:.0f} km")
    print()
    print(f"FIBER  (0.2 dB/km, 1550 nm):  loss = {result['fiber_loss_db']:.1f} dB, "
          f"transmittance = {result['fiber_transmittance']:.2e}")
    print(f"SATELLITE (500 km LEO relay): loss = {result['satellite_loss_db']:.1f} dB, "
          f"transmittance = {result['satellite_transmittance']:.2e}")
    print()
    if result["satellite_wins"]:
        ratio = result["satellite_transmittance"] / result["fiber_transmittance"]
        print(f"--> satellite relay wins by a factor of {ratio:.2e} in transmittance.")
        print("    Direct fiber over this distance is essentially dead (exponential")
        print("    loss); free-space diffraction loss only grows as a power law, which")
        print("    is why real continental-scale quantum links go via satellite relay.")
    print()
    print(f"Alternative: quantum repeaters every {span_km:.1f} km "
          f"({n_seg} segments, {span_loss:.1f} dB/span) would make the fiber route viable too.")
