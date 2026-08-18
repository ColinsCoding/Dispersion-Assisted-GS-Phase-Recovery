"""Laser safety calculations -- irradiance, beam divergence, Nominal Hazard
Distance (NHZ), and required eyewear optical density (OD).

These are the actual quantitative calculations behind the online/in-person
laser safety training required before working with a Class 3B/4 system
(e.g. a Jalali-lab-style pulsed/CW laser bench): the training explains
WHY a control measure is needed, these functions compute WHERE the hazard
actually extends to and HOW MUCH attenuation eyewear needs to provide.

Formulas follow the standard ANSI Z136.1 non-focusing-beam model. This is
an engineering-estimate tool, not a substitute for a real Laser Safety
Officer's (LSO) sign-off -- see references/laser_safety_training.md for
the full picture (classification, control measures, PPE, procedures).
"""
import numpy as np


def beam_area_cm2(diameter_cm: float) -> float:
    """Circular beam cross-sectional area (cm^2) from its diameter (cm)."""
    if diameter_cm <= 0:
        raise ValueError("diameter_cm must be positive")
    return np.pi / 4 * diameter_cm ** 2


def irradiance_W_cm2(power_W: float, diameter_cm: float) -> float:
    """Irradiance (W/cm^2) = power / beam area -- the quantity compared
    against the Maximum Permissible Exposure (MPE) limit."""
    if power_W <= 0:
        raise ValueError("power_W must be positive")
    return power_W / beam_area_cm2(diameter_cm)


def beam_diameter_at_distance_cm(d0_cm: float, divergence_rad: float, distance_cm: float) -> float:
    """Beam diameter (cm) at a given distance for a simple divergent
    (non-focusing) beam: d(r) = d0 + divergence * r."""
    if d0_cm <= 0:
        raise ValueError("d0_cm must be positive")
    if divergence_rad < 0:
        raise ValueError("divergence_rad must be non-negative")
    if distance_cm < 0:
        raise ValueError("distance_cm must be non-negative")
    return d0_cm + divergence_rad * distance_cm


def nominal_hazard_distance_cm(power_W: float, mpe_W_cm2: float, d0_cm: float, divergence_rad: float) -> float:
    """Nominal Hazard Distance (NHZ, cm): the distance at which irradiance
    falls to the MPE, for a simple divergent beam. Beyond this distance,
    direct/specular viewing is below the exposure limit.

    NHZ = (1/divergence) * [ sqrt(4*power / (pi*MPE)) - d0 ]

    Returns 0.0 if the beam is already at or below MPE at the source
    (no hazard zone extends outward)."""
    if power_W <= 0:
        raise ValueError("power_W must be positive")
    if mpe_W_cm2 <= 0:
        raise ValueError("mpe_W_cm2 must be positive")
    if d0_cm <= 0:
        raise ValueError("d0_cm must be positive")
    if divergence_rad <= 0:
        raise ValueError("divergence_rad must be positive (a perfectly collimated beam has no finite NHZ)")
    source_irradiance = irradiance_W_cm2(power_W, d0_cm)
    if source_irradiance <= mpe_W_cm2:
        return 0.0
    nhz = (np.sqrt(4 * power_W / (np.pi * mpe_W_cm2)) - d0_cm) / divergence_rad
    return max(nhz, 0.0)


def required_optical_density(incident_irradiance_W_cm2: float, mpe_W_cm2: float) -> float:
    """Optical density (OD) an eyewear filter must provide to attenuate
    incident irradiance down to the MPE: OD = log10(incident / MPE).
    Returns 0.0 if incident is already at or below MPE (no attenuation needed)."""
    if incident_irradiance_W_cm2 <= 0:
        raise ValueError("incident_irradiance_W_cm2 must be positive")
    if mpe_W_cm2 <= 0:
        raise ValueError("mpe_W_cm2 must be positive")
    if incident_irradiance_W_cm2 <= mpe_W_cm2:
        return 0.0
    return np.log10(incident_irradiance_W_cm2 / mpe_W_cm2)
