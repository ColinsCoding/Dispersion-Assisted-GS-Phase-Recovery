"""Energy and wavelength, precisely -- and why it's the actual physical
limit on advanced manufacturing of silicon photonics.

THE CONNECTION: E = hc/lambda is not just a formula to evaluate -- the
photon energy used in lithography DIRECTLY sets the smallest feature a
fab can print (the Rayleigh resolution limit, CD = k1*lambda/NA).
Shorter wavelength -> higher photon energy -> smaller minimum feature.
This is WHY the semiconductor industry moved from 248 nm (KrF) to 193 nm
(ArF) to EUV (13.5 nm) lithography over decades -- each step is a real,
physical consequence of E=hc/lambda, not an arbitrary engineering choice.
Silicon photonic waveguides (100s of nm wide, needed for single-mode
operation near 1550 nm) sit right at the scale these lithography
generations can resolve.

Also covers the real process CHEMISTRY used to fabricate a silicon
photonic chip: thermal oxidation (Deal-Grove kinetics, the foundational
1965 semiconductor manufacturing model), CVD dielectric deposition,
plasma etch chemistry for waveguide definition, and dopant diffusion --
using precise SI-exact physical constants (h, c exact since the 2019 SI
redefinition) throughout.
"""

import numpy as np

H_PLANCK = 6.62607015e-34    # J*s, EXACT (2019 SI redefinition)
C_LIGHT = 299792458.0         # m/s, EXACT
E_CHARGE = 1.602176634e-19    # C, EXACT
HC_EV_NM = H_PLANCK * C_LIGHT / E_CHARGE * 1e9   # convenient hc in eV*nm


def photon_energy_ev(wavelength_m):
    """E = hc/lambda, using SI-exact constants. Returns eV."""
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    return (H_PLANCK * C_LIGHT) / (wavelength_m * E_CHARGE)


def rayleigh_resolution_m(wavelength_m, NA, k1=0.4):
    """The real photolithography resolution limit: CD = k1*lambda/NA.
    k1 is a process-dependent factor (0.25-0.4 for advanced, aggressive
    processes; 0.4 is a common 'comfortable' baseline). Smaller
    wavelength or larger numerical aperture -> smaller printable
    feature -- this is literally why fabs keep shrinking lithography
    wavelength."""
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    if NA <= 0:
        raise ValueError("NA must be positive")
    if k1 <= 0:
        raise ValueError("k1 must be positive")
    return k1 * wavelength_m / NA


def depth_of_focus_m(wavelength_m, NA, k2=0.5):
    """Companion figure of merit to Rayleigh resolution: DOF = k2*lambda/NA^2
    -- shrinking wavelength/growing NA to get better resolution also
    shrinks how much vertical process variation (resist thickness,
    wafer flatness) the exposure can tolerate. The real manufacturing
    tradeoff behind 'just use a smaller wavelength'."""
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    if NA <= 0:
        raise ValueError("NA must be positive")
    if k2 <= 0:
        raise ValueError("k2 must be positive")
    return k2 * wavelength_m / NA**2


def deal_grove_oxide_thickness_m(time_hr, A_um, B_um2_per_hr, tau_hr=0.0):
    """Deal-Grove (1965) thermal oxidation kinetics -- the foundational
    semiconductor-manufacturing model for growing SiO2 (Si + O2 ->
    SiO2, or Si + 2H2O -> SiO2 + 2H2 for wet oxidation):
      x_ox^2 + A*x_ox = B*(t + tau)
    Solved here via the quadratic formula for the oxide thickness x_ox
    at a given oxidation time. Combines an initial reaction-rate-limited
    LINEAR regime (thin oxide) with a diffusion-limited PARABOLIC regime
    (thick oxide, oxidant must diffuse through existing oxide)."""
    if time_hr < 0:
        raise ValueError("time_hr must be non-negative")
    if A_um <= 0:
        raise ValueError("A_um must be positive")
    if B_um2_per_hr <= 0:
        raise ValueError("B_um2_per_hr must be positive")
    if tau_hr < 0:
        raise ValueError("tau_hr must be non-negative")
    x_um = (-A_um + np.sqrt(A_um**2 + 4 * B_um2_per_hr * (time_hr + tau_hr))) / 2.0
    return x_um * 1e-6


SILICON_PHOTONICS_PROCESS_CHEMISTRY = {
    "thermal_oxidation": {
        "reaction": "Si + O2 -> SiO2 (dry)  or  Si + 2 H2O -> SiO2 + 2 H2 (wet)",
        "role": "grows the buried oxide (BOX) cladding beneath the waveguide core",
        "kinetics": "Deal-Grove linear-parabolic model",
    },
    "lpcvd_nitride": {
        "reaction": "3 SiH2Cl2 + 4 NH3 -> Si3N4 + 6 HCl + 6 H2",
        "role": "deposits Si3N4, a real alternative waveguide core material "
                "(lower loss than Si at visible wavelengths, used in some "
                "photonic platforms)",
        "kinetics": "surface-reaction-limited LPCVD, near-uniform deposition rate",
    },
    "teos_pecvd_oxide": {
        "reaction": "Si(OC2H5)4 + O2 -> SiO2 + byproducts (TEOS precursor)",
        "role": "deposits the top oxide cladding at low temperature "
                "(compatible with underlying metal/device layers)",
        "kinetics": "plasma-enhanced, lower temperature than thermal oxidation",
    },
    "waveguide_dry_etch": {
        "reaction": "reactive-ion etch, e.g. Cl2/HBr chemistry for silicon "
                     "(or SF6/C4F8 Bosch-type cycling for deep features)",
        "role": "defines the waveguide's physical sidewalls -- etch "
                "roughness here becomes real optical scattering loss",
        "kinetics": "plasma chemistry, anisotropic (vertical) etch profile",
    },
    "dopant_diffusion": {
        "reaction": "POCl3 (phosphorus) or BBr3 (boron) source diffusion, "
                     "or direct ion implantation",
        "role": "forms the p-n junction for an integrated modulator/"
                "photodetector (dopant profile obeys Fick's second law)",
        "kinetics": "erfc profile (constant-source diffusion) or Gaussian "
                     "profile (ion implant + anneal)",
    },
}


if __name__ == "__main__":
    print("=== Photon energy, precisely (SI-exact h and c) ===\n")
    print(f"hc = {HC_EV_NM:.4f} eV*nm (handy constant for E(eV) = hc_eV_nm / lambda(nm))\n")

    lithography_wavelengths_nm = [
        ("g-line", 436.0), ("i-line", 365.0), ("KrF", 248.0),
        ("ArF", 193.0), ("EUV", 13.5),
    ]
    for name, wl_nm in lithography_wavelengths_nm:
        E = photon_energy_ev(wl_nm * 1e-9)
        print(f"  {name:6s} ({wl_nm:6.1f} nm): photon energy = {E:7.2f} eV")
    print("  (real known values: ArF ~6.4 eV, KrF ~5.0 eV, EUV ~91.8 eV -- match)\n")

    print("=== Photon energy -> lithography resolution -> waveguide feature size ===\n")
    cases = [
        ("ArF immersion (193 nm)", 193e-9, 1.35, 0.30),
        ("EUV (13.5 nm)", 13.5e-9, 0.33, 0.30),
    ]
    for name, wl, NA, k1 in cases:
        CD = rayleigh_resolution_m(wl, NA, k1)
        DOF = depth_of_focus_m(wl, NA)
        print(f"  {name}: min feature (CD) = {CD*1e9:.1f} nm, "
              f"depth of focus = {DOF*1e9:.1f} nm")
    print("  (real: 193i achieves ~40 nm-class half-pitch; EUV single-exposure ~13 nm-class)")
    print("  Silicon photonic waveguides need ~200-500 nm features for single-mode")
    print("  1550 nm operation -- comfortably within reach of mature 193 nm tooling,")
    print("  which is WHY silicon photonics is CMOS-fab-compatible at all.\n")

    print("=== Deal-Grove thermal oxidation kinetics ===\n")
    # representative literature-typical dry-oxidation parameters at ~1000 C
    A_um, B_um2_per_hr = 0.235, 0.0117
    for t_hr in [0.5, 1.0, 2.0, 4.0]:
        x = deal_grove_oxide_thickness_m(t_hr, A_um, B_um2_per_hr)
        print(f"  {t_hr:.1f} hr dry oxidation -> oxide thickness = {x*1e9:.1f} nm")
    print("  (thin-oxide regime grows ~linearly in time; thick-oxide regime")
    print("   slows to ~sqrt(time) as oxidant must diffuse through existing oxide)\n")

    print("=== Real process chemistry for a silicon photonic chip ===\n")
    for step, d in SILICON_PHOTONICS_PROCESS_CHEMISTRY.items():
        print(f"  {step.upper().replace('_', ' ')}")
        print(f"    reaction: {d['reaction']}")
        print(f"    role:     {d['role']}")
        print(f"    kinetics: {d['kinetics']}\n")
