"""Biophotonics: Beer-Lambert, photodynamic therapy, optogenetics,
underwater optical communication, and COVID spike biosensing.

THE CENTRAL EQUATION IS BEER-LAMBERT:
  I(z) = I_0 * exp(-mu_t * z)

  mu_t = mu_a + mu_s  (total attenuation = absorption + scattering)

  Every use case in this module is a variant of this one equation:
    - Submarine blue-green laser: mu_t(450 nm) is minimum in seawater
    - PDT (photodynamic therapy): photons must reach tumor depth z
    - Optogenetics: blue light (470 nm) activates channelrhodopsin in tissue
    - COVID spike biosensor: antibody binding changes mu_a at sensor surface
    - Retinal imaging: OCT measures backscattered I(z) to reconstruct depth

PHOTODYNAMIC THERAPY (PDT):
  1. Inject photosensitizer (porphyrin / chlorin) that accumulates in tumor
  2. Illuminate tumor with red/NIR light (630-900 nm, tissue is transparent)
  3. Photosensitizer absorbs photon -> excited singlet state -> intersystem
     crossing -> triplet state -> energy transfer to O2 -> singlet oxygen (1O2)
  4. 1O2 kills tumor cells (oxidative damage) without ionizing radiation
  Dose = fluence * photosensitizer concentration * oxygen availability

OPTOGENETICS:
  Insert channelrhodopsin (ChR2) gene into neurons via viral vector.
  Shine 470 nm blue light -> ChR2 opens -> Na+ flows in -> neuron fires.
  This is how neuroscientists turn individual neurons on/off with light.
  Precision: single-cell, millisecond timescale. Equivalent of a GS algorithm
  for the brain: measure output (firing pattern), adjust input (light pulse).

UNDERWATER OPTICAL COMM:
  US Navy uses 532 nm (green) laser for submarine communication.
  Blue-green window (450-550 nm): seawater absorption minimum ~0.005/m.
  At 1064 nm (IR): absorption ~0.4/m -- signal gone in 10 m.
  At 532 nm: 1/e depth ~200 m -- can reach submarines at periscope depth.
  The signal propagation is the same TS-DFT + GS physics: intensity I(z)
  is measured, and you need to recover the modulated phase from it.
"""
import numpy as np
import sympy as sp


# ── Beer-Lambert law ─────────────────────────────────────────────────

def beer_lambert(I0, mu_total, depth_m):
    """Intensity at depth z: I(z) = I_0 * exp(-mu_t * z).

    Parameters
    ----------
    I0 : float       -- incident intensity (W/m^2 or any power unit)
    mu_total : float -- total attenuation coefficient (1/m); mu_a + mu_s
    depth_m : float  -- depth into medium (m); must be >= 0

    Returns I at depth z, and the 1/e penetration depth.
    """
    if I0 <= 0:
        raise ValueError("I0 must be positive")
    if mu_total < 0:
        raise ValueError("mu_total must be non-negative")
    if depth_m < 0:
        raise ValueError("depth_m must be non-negative")
    I = I0 * np.exp(-mu_total * depth_m)
    depth_1e = 1.0 / mu_total if mu_total > 0 else float("inf")
    return {"I": I, "I0": I0, "fraction_transmitted": I / I0,
            "depth_1e_m": depth_1e, "attenuation_dB": -10 * np.log10(I / I0)}


def penetration_depth(mu_total):
    """1/e penetration depth = 1 / mu_total (metres)."""
    if mu_total <= 0:
        raise ValueError("mu_total must be positive")
    return 1.0 / mu_total


def depth_for_fraction(I0, mu_total, fraction):
    """Depth z at which intensity has fallen to fraction * I0.

    z = -ln(fraction) / mu_total
    """
    if not (0 < fraction < 1):
        raise ValueError("fraction must be in (0, 1)")
    if mu_total <= 0:
        raise ValueError("mu_total must be positive")
    return -np.log(fraction) / mu_total


# ── seawater optical attenuation vs wavelength ───────────────────────

# Approximate total attenuation coefficients for clear open ocean (m^-1)
# Source: Pope & Fry 1997, Smith & Baker 1981 tabulated values
SEAWATER_MU_APPROX = {
    400: 0.018,   # UV -- high absorption
    450: 0.009,   # blue-violet -- window begins
    480: 0.007,   # blue -- near-minimum
    510: 0.006,   # blue-green -- minimum (~0.005-0.007 in clear ocean)
    532: 0.007,   # Nd:YAG 2nd harmonic (submarine comm laser)
    550: 0.010,   # green
    600: 0.025,   # yellow-orange
    650: 0.035,   # red (PDT window in tissue, but not water)
    700: 0.060,   # red
    800: 0.200,   # NIR
    1000: 0.380,  # NIR
    1064: 0.420,  # Nd:YAG fundamental (bad for water)
}


def seawater_transmission(wavelength_nm, depth_m):
    """Approximate transmission of light in clear ocean at given depth.

    Uses tabulated mu values; interpolates linearly between table entries.
    For military blue-green submarine comm (532 nm): 1/e depth ~ 143 m.
    """
    wl_keys = sorted(SEAWATER_MU_APPROX.keys())
    wl_nm = float(wavelength_nm)
    if wl_nm < wl_keys[0] or wl_nm > wl_keys[-1]:
        raise ValueError(f"wavelength_nm must be in [{wl_keys[0]}, {wl_keys[-1]}] nm")
    # linear interpolation between table entries
    for i in range(len(wl_keys) - 1):
        if wl_keys[i] <= wl_nm <= wl_keys[i+1]:
            t = (wl_nm - wl_keys[i]) / (wl_keys[i+1] - wl_keys[i])
            mu = (SEAWATER_MU_APPROX[wl_keys[i]] * (1 - t)
                  + SEAWATER_MU_APPROX[wl_keys[i+1]] * t)
            break
    r = beer_lambert(1.0, mu, depth_m)
    return {**r, "wavelength_nm": wavelength_nm,
            "mu_total_per_m": mu,
            "penetration_depth_m": penetration_depth(mu)}


def find_optimal_wavelength_for_depth(target_depth_m, min_fraction=0.01):
    """Find the wavelength with best transmission at target depth."""
    best_wl = None
    best_frac = 0.0
    for wl in SEAWATER_MU_APPROX:
        mu = SEAWATER_MU_APPROX[wl]
        frac = np.exp(-mu * target_depth_m)
        if frac > best_frac:
            best_frac = frac
            best_wl = wl
    return {"best_wavelength_nm": best_wl,
            "fraction_at_depth": best_frac,
            "target_depth_m": target_depth_m}


# ── photodynamic therapy (PDT) ────────────────────────────────────────

# Tissue optical properties (mu in 1/cm, then convert to 1/m for Beer-Lambert)
# Breast tissue approximation at therapeutic window 630-900 nm
TISSUE_MU_A_PER_CM = {630: 0.04, 700: 0.02, 800: 0.01, 900: 0.008}  # absorption
TISSUE_MU_S_PER_CM = {630: 10.0, 700: 9.0,  800: 8.0,  900: 7.0}    # scattering


def pdt_fluence_at_depth(I0_W_m2, wavelength_nm, depth_cm):
    """Fluence (J/m^2) delivered to tumor at depth in tissue.

    PDT requires a minimum fluence of ~10-50 J/cm^2 at the tumor.
    Optimal window: 630-900 nm (red/NIR) where tissue has low mu_a.
    Blue light (470 nm) is blocked within ~1 mm -- only good for surface PDT.

    Uses simplified diffusion theory: effective attenuation
    mu_eff = sqrt(3 * mu_a * (mu_a + mu_s))
    """
    wl = int(wavelength_nm)
    wl_keys = sorted(TISSUE_MU_A_PER_CM.keys())
    # clamp to nearest tabulated wavelength
    wl_nearest = min(wl_keys, key=lambda k: abs(k - wl))
    mu_a = TISSUE_MU_A_PER_CM[wl_nearest] * 100  # convert cm^-1 -> m^-1
    mu_s = TISSUE_MU_S_PER_CM[wl_nearest] * 100
    mu_eff = np.sqrt(3 * mu_a * (mu_a + mu_s))
    depth_m = depth_cm / 100
    I = I0_W_m2 * np.exp(-mu_eff * depth_m)
    therapeutic = I >= 100  # 10 J/cm^2 = 1000 J/m^2 threshold (rough)
    return {"fluence_W_m2": I, "fluence_mW_cm2": I * 0.1,
            "mu_eff_per_m": mu_eff, "depth_cm": depth_cm,
            "wavelength_nm": wavelength_nm,
            "therapeutic_at_depth": therapeutic,
            "penetration_1e_cm": 100 / mu_eff}


# ── COVID spike protein optical biosensing ────────────────────────────

def spike_biosensor_response(antibody_conc_nM, sensor_area_cm2=1.0,
                              binding_efficiency=0.8,
                              delta_n_per_molecule=1e-7):
    """Model refractive index shift from COVID spike protein binding.

    Surface plasmon resonance (SPR) biosensors detect binding of spike
    protein to ACE2-coated sensor surface by measuring the shift in
    resonance angle (or wavelength) caused by the added mass.

    delta_n = N_bound * delta_n_per_molecule  (refractive index change)

    where N_bound = antibody_conc * binding_efficiency * Avogadro * volume

    A shift of delta_n ~ 1e-4 is detectable with a standard SPR instrument.
    Limit of detection: ~1 pM spike protein (10^9 molecules/mL).

    Parameters
    ----------
    antibody_conc_nM : float  -- spike/antibody concentration in nanomolar
    sensor_area_cm2 : float   -- active sensor surface area
    binding_efficiency : float -- fraction that binds (0-1)
    delta_n_per_molecule : float -- refractive index contribution per molecule

    Returns dict with estimated delta_n and whether detectable.
    """
    if antibody_conc_nM < 0:
        raise ValueError("concentration must be non-negative")
    if not (0 <= binding_efficiency <= 1):
        raise ValueError("binding_efficiency must be in [0,1]")
    avogadro = 6.022e23
    volume_L = sensor_area_cm2 * 1e-4 * 1e-4 * 1e3   # ~100 nm sensing volume
    conc_mol_L = antibody_conc_nM * 1e-9
    n_molecules = conc_mol_L * volume_L * avogadro * binding_efficiency
    delta_n = n_molecules * delta_n_per_molecule
    detectable_threshold = 1e-5   # typical SPR sensitivity
    return {
        "antibody_conc_nM": antibody_conc_nM,
        "n_molecules_bound": n_molecules,
        "delta_n": delta_n,
        "detectable": delta_n >= detectable_threshold,
        "LOD_nM": detectable_threshold / (delta_n / antibody_conc_nM) if antibody_conc_nM > 0 else None,
    }


# ── optogenetics ──────────────────────────────────────────────────────

def optogenetics_activation(peak_wavelength_nm, power_density_mW_mm2,
                             exposure_time_ms, depth_mm=1.0):
    """Model channelrhodopsin-2 (ChR2) activation probability.

    ChR2 absorption peak: 470 nm (blue). Half-activation intensity: ~1 mW/mm^2.
    Tissue attenuation at 470 nm: high (mu_eff ~ 200/m in brain tissue).
    At 1 mm depth in brain: ~18% of surface intensity reaches the cell.

    Activation probability:
      P_act = sigmoid((I_at_depth / I_half) - 1)
      where I_half ~ 1 mW/mm^2 for ChR2.
    """
    if power_density_mW_mm2 <= 0:
        raise ValueError("power_density must be positive")
    mu_eff_brain_per_m = 200.0  # rough value for 470 nm in brain tissue
    depth_m = depth_mm * 1e-3
    I_surface = power_density_mW_mm2
    I_at_depth = I_surface * np.exp(-mu_eff_brain_per_m * depth_m)
    I_half = 1.0   # mW/mm^2 -- ChR2 half-activation
    x = (I_at_depth / I_half) - 1
    P_act = 1.0 / (1.0 + np.exp(-x))   # sigmoid
    photons_per_ms = (I_at_depth * 1e-3 / (6.63e-34 * 3e8 / (peak_wavelength_nm * 1e-9))
                      * 1e-6 * 1e-3)
    return {
        "I_surface_mW_mm2": I_surface,
        "I_at_depth_mW_mm2": I_at_depth,
        "depth_mm": depth_mm,
        "P_activation": P_act,
        "activated": P_act > 0.5,
        "wavelength_nm": peak_wavelength_nm,
    }


# ── SymPy formalism ───────────────────────────────────────────────────

def biophotonics_sympy_5():
    """Five key biophotonics equations."""
    I0, mu, z = sp.symbols('I_0 mu z', positive=True)
    mu_a, mu_s = sp.symbols('mu_a mu_s', positive=True)
    delta_n, N = sp.symbols('Delta_n N', positive=True)
    n_idx = sp.Symbol('n_molecule')
    C, V, Na = sp.symbols('C V N_A', positive=True)

    return {
        "Beer_Lambert":
            sp.Eq(sp.Symbol('I(z)'), I0 * sp.exp(-mu * z)),
        "Effective_attenuation":
            sp.Eq(sp.Symbol('mu_eff'),
                  sp.sqrt(3 * mu_a * (mu_a + mu_s))),
        "Penetration_depth":
            sp.Eq(sp.Symbol('delta_pen'), 1 / mu),
        "SPR_refractive_index_shift":
            sp.Eq(delta_n, N * n_idx),
        "Molecule_count":
            sp.Eq(N, C * V * Na),
    }


if __name__ == "__main__":
    print("=== Beer-Lambert: 532 nm in seawater ===")
    for d in [10, 50, 100, 200]:
        r = seawater_transmission(532, d)
        print(f"  {d:3d} m depth: {r['fraction_transmitted']*100:.1f}% transmitted "
              f"(mu={r['mu_total_per_m']:.3f}/m)")
    print(f"  1/e depth at 532 nm: {seawater_transmission(532,1)['penetration_depth_m']:.0f} m")

    print("\n=== Optimal wavelength for submarine at 100 m ===")
    opt = find_optimal_wavelength_for_depth(100)
    print(f"  Best: {opt['best_wavelength_nm']} nm, "
          f"{opt['fraction_at_depth']*100:.1f}% transmission")

    print("\n=== PDT: tumor at 1 cm depth, 630 nm red light ===")
    pdt = pdt_fluence_at_depth(1000, 630, depth_cm=1.0)
    print(f"  Fluence at tumor: {pdt['fluence_mW_cm2']:.1f} mW/cm^2")
    print(f"  1/e depth at 630 nm: {pdt['penetration_1e_cm']:.2f} cm")
    print(f"  Therapeutic: {pdt['therapeutic_at_depth']}")

    print("\n=== COVID spike biosensor (SPR) ===")
    for conc in [0.001, 0.1, 1.0, 100.0]:
        r = spike_biosensor_response(conc)
        print(f"  {conc:.3f} nM: delta_n={r['delta_n']:.2e}, "
              f"detectable={r['detectable']}")

    print("\n=== Optogenetics: ChR2 at 1 mm depth in brain tissue ===")
    opt_g = optogenetics_activation(470, power_density_mW_mm2=5.0,
                                    exposure_time_ms=10, depth_mm=1.0)
    print(f"  I at 1 mm: {opt_g['I_at_depth_mW_mm2']:.3f} mW/mm^2")
    print(f"  P(activation): {opt_g['P_activation']:.3f}")
    print(f"  Neuron fires: {opt_g['activated']}")

    print("\n=== SymPy 5 ===")
    for k, eq in biophotonics_sympy_5().items():
        print(f"  {k}: {eq}")
