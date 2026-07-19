"""Integrated photonic biosensors / lab-on-chip: the real physics behind
label-free biomolecule detection using silicon microring (whispering-
gallery-mode) resonators, the dominant real 2025/2026 research direction
in on-chip biosensing (per the biosensor/lab-on-chip WebSearch done this
session -- microring/WGM resonators, evanescent-field sensing).

The idea: a biomolecule binding to a functionalized ring surface changes
the local effective refractive index seen by the guided mode's evanescent
tail. That shifts the ring's resonance wavelength. No fluorescent tag is
needed (hence "label-free") -- you just track how far the resonance dip
moves. This module quantifies each step of that chain from first
principles: free spectral range -> Q-factor -> sensitivity (nm shift per
refractive-index-unit) -> the smallest wavelength shift actually
resolvable given noise -> the resulting limit of detection (LOD) in RIU
(refractive index units), the real figure of merit reported in the
biosensor literature (state-of-art silicon microrings: LOD ~ 1e-6-1e-7
RIU, bulk sensitivity ~ 50-100 nm/RIU, Q ~ 1e4-1e5).
"""

import numpy as np


def resonator_fsr_m(wavelength_m, n_group, radius_m):
    """Free spectral range of a ring resonator: FSR = lambda^2 / (n_g * L),
    L = 2*pi*R (round-trip circumference). Sets how many resonance dips
    fit in a given wavelength scan range."""
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    if n_group <= 0:
        raise ValueError("n_group must be positive")
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    L = 2 * np.pi * radius_m
    return wavelength_m ** 2 / (n_group * L)


def resonator_q_factor(wavelength_m, fwhm_m):
    """Loaded Q-factor from the resonance dip's measured linewidth:
    Q = lambda / FWHM. Higher Q means a narrower, more precisely
    locatable resonance -- directly improves detection limit."""
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    if fwhm_m <= 0:
        raise ValueError("fwhm_m must be positive")
    return wavelength_m / fwhm_m


def resonator_finesse(fsr_m, fwhm_m):
    """Finesse = FSR / FWHM: how many linewidths fit between adjacent
    resonances, a resonator-quality figure independent of absolute Q."""
    if fsr_m <= 0:
        raise ValueError("fsr_m must be positive")
    if fwhm_m <= 0:
        raise ValueError("fwhm_m must be positive")
    return fsr_m / fwhm_m


def bulk_sensitivity_m_per_riu(wavelength_m, n_group, confinement_factor):
    """Bulk refractive-index sensitivity S = dlambda/dn = (lambda/n_g) *
    Gamma, where Gamma in [0,1] is the fraction of the guided mode's
    energy that actually overlaps the analyte (the evanescent tail
    sticking out of the waveguide core, not the whole mode). This is
    WHY the waveguide is deliberately under-confined for a sensor --
    more field outside the core means more sensitivity to what's out
    there, at the cost of higher bend/scattering loss."""
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    if n_group <= 0:
        raise ValueError("n_group must be positive")
    if not (0 < confinement_factor <= 1):
        raise ValueError("confinement_factor must be in (0, 1]")
    return (wavelength_m / n_group) * confinement_factor


def minimum_detectable_wavelength_shift_m(fwhm_m, snr):
    """The smallest resonance shift actually resolvable, given the dip's
    linewidth and the achievable signal-to-noise ratio of the wavelength-
    tracking scheme (peak-fitting/lock-in routinely beats the naive
    FWHM/2 "can you tell two dips apart" limit by 2-3 orders of
    magnitude -- SNR here plays that role directly)."""
    if fwhm_m <= 0:
        raise ValueError("fwhm_m must be positive")
    if snr <= 0:
        raise ValueError("snr must be positive")
    return fwhm_m / snr


def limit_of_detection_riu(min_shift_m, sensitivity_m_per_riu):
    """LOD in refractive-index units: the smallest bulk (or, by the same
    ratio, surface-bound analyte) index change that produces a resonance
    shift still bigger than the noise floor. This is the actual number
    biosensor papers report and compare against."""
    if min_shift_m <= 0:
        raise ValueError("min_shift_m must be positive")
    if sensitivity_m_per_riu <= 0:
        raise ValueError("sensitivity_m_per_riu must be positive")
    return min_shift_m / sensitivity_m_per_riu


def biosensor_figure_of_merit(wavelength_m, n_group, radius_m, fwhm_m,
                               confinement_factor, snr):
    """Runs the full chain -- FSR, Q, sensitivity, minimum resolvable
    shift, LOD -- from raw ring-resonator design parameters, so the whole
    label-free sensing argument is one checkable set of numbers rather
    than a qualitative claim."""
    fsr = resonator_fsr_m(wavelength_m, n_group, radius_m)
    Q = resonator_q_factor(wavelength_m, fwhm_m)
    finesse = resonator_finesse(fsr, fwhm_m)
    S = bulk_sensitivity_m_per_riu(wavelength_m, n_group, confinement_factor)
    min_shift = minimum_detectable_wavelength_shift_m(fwhm_m, snr)
    lod = limit_of_detection_riu(min_shift, S)
    return {
        "fsr_nm": fsr * 1e9,
        "Q_factor": Q,
        "finesse": finesse,
        "sensitivity_nm_per_riu": S * 1e9,
        "min_detectable_shift_pm": min_shift * 1e12,
        "lod_riu": lod,
    }


if __name__ == "__main__":
    print("=== Silicon microring biosensor: label-free lab-on-chip detection ===")
    print("A biomolecule binding the ring's functionalized surface shifts n_eff,")
    print("which shifts the resonance wavelength. No fluorescent tag needed.\n")

    # real-scale silicon photonic microring biosensor parameters
    wavelength_m = 1550e-9      # C-band telecom wavelength, standard choice
    n_group = 4.2               # silicon group index near 1550 nm
    radius_m = 10e-6             # 10 um ring radius, typical compact design
    fwhm_m = 0.1e-9              # 0.1 nm linewidth, realistic high-Q silicon ring
    confinement_factor = 0.2     # 20% of the mode overlaps the analyte (evanescent tail)
    snr = 1000.0                  # realistic achievable with peak-fitting/lock-in

    result = biosensor_figure_of_merit(
        wavelength_m, n_group, radius_m, fwhm_m, confinement_factor, snr)

    print(f"free spectral range:        {result['fsr_nm']:.2f} nm")
    print(f"Q factor:                   {result['Q_factor']:.0f}")
    print(f"finesse:                    {result['finesse']:.1f}")
    print(f"bulk sensitivity:           {result['sensitivity_nm_per_riu']:.1f} nm/RIU")
    print(f"min. detectable shift:      {result['min_detectable_shift_pm']:.2f} pm")
    print(f"limit of detection (LOD):   {result['lod_riu']:.2e} RIU")

    print("\n=== Sanity check against real literature ===")
    print("Reported silicon microring biosensors: Q ~ 1e4-1e5, bulk sensitivity")
    print("~ 50-100 nm/RIU, LOD ~ 1e-6-1e-7 RIU. The numbers above land in that")
    print("real range using only first-principles ring-resonator physics --")
    print("not fitted or reverse-engineered to match.")

    print("\n=== Why this is a real lab-on-chip problem, not just optics ===")
    print("The hard part in practice isn't the resonator -- it's microfluidics:")
    print("getting a nanoliter-scale sample to the ring's surface, functionalizing")
    print("that surface with the right antibody/aptamer, and rejecting nonspecific")
    print("binding (false positives from the WRONG molecule sticking). The optics")
    print("above is necessary but not sufficient; the surface chemistry and fluidic")
    print("delivery are equally real, equally hard, and not modeled here.")
