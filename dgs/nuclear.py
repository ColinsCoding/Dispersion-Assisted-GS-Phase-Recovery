"""Nuclear physics: binding energy, radioactive decay, neutron interactions,
particle stopping power, radiation dose, and food preservation.

Coverage maps to Serway Modern Physics Ch 13-14:
  Ch 13: binding energy (Bethe-Weizsacker semi-empirical mass formula),
          radioactive decay (alpha/beta/gamma), half-life, carbon dating
  Ch 14: neutron cross sections, fission Q-value, Bethe-Bloch stopping power,
          radiation dose in Grays, food preservation dose levels

Why Python over MATLAB for this work:
  Python is free, runs on every OS, integrates with torch/numpy/scipy for ML,
  has symbolic algebra (SymPy), version control works naturally (text files),
  and the physics community has moved to Python (astropy, pymatgen, ROOT Python).
  MATLAB requires a license (~$2500/yr), its arrays are 1-indexed (confusing),
  and its package ecosystem is closed.  Every calculation in this module runs
  on py -3.13 with only numpy.

Usage:
    from dgs.nuclear import (
        binding_energy, bethe_weizsacker, half_life_to_decay_constant,
        activity, decay_chain, carbon_dating_age,
        neutron_cross_section_estimate, fission_q_value,
        bethe_bloch_stopping, dose_gray, food_preservation_dose
    )
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

AMU_KG     = 1.66053906660e-27   # unified atomic mass unit (kg)
AMU_MEV    = 931.494            # 1 amu = 931.494 MeV/c^2
M_PROTON   = 1.007276           # proton mass (amu)
M_NEUTRON  = 1.008665           # neutron mass (amu)
M_ELECTRON = 0.000549           # electron mass (amu)
C_LIGHT    = 2.99792458e8       # m/s
Q_E        = 1.602176634e-19    # Coulombs
N_AVOGADRO = 6.02214076e23      # mol^-1

BARN = 1e-28   # 1 barn = 1e-28 m^2 (nuclear cross section unit)

# ---------------------------------------------------------------------------
# Bethe-Weizsacker semi-empirical mass formula (SEMF)
# Serway Ch 13.2 / Krane Ch 3
# ---------------------------------------------------------------------------

# SEMF coefficients (MeV) -- Rohlf 1994 values
A_V  = 15.835   # volume term
A_S  = 18.33    # surface term
A_C  = 0.714    # Coulomb term
A_A  = 23.2     # asymmetry term
A_P  = 11.2     # pairing term


def bethe_weizsacker(Z: int, A: int) -> float:
    """Semi-empirical binding energy B(Z,A) in MeV.

    B = a_V*A - a_S*A^(2/3) - a_C*Z(Z-1)/A^(1/3)
        - a_A*(A-2Z)^2/A + delta(Z,A)

    where delta is the pairing term:
      +a_P / sqrt(A)  for even-even
       0              for even-odd or odd-even
      -a_P / sqrt(A)  for odd-odd

    Parameters
    ----------
    Z : atomic number (proton count)
    A : mass number (proton + neutron count)

    Returns
    -------
    Binding energy B in MeV (positive = bound).
    """
    N = A - Z
    if N < 0 or Z < 1 or A < 1:
        raise ValueError(f"Invalid (Z={Z}, A={A})")

    vol      = A_V * A
    surf     = A_S * A**(2/3)
    coulomb  = A_C * Z * (Z - 1) / A**(1/3)
    asym     = A_A * (A - 2*Z)**2 / A
    if N % 2 == 0 and Z % 2 == 0:
        pairing = +A_P / np.sqrt(A)
    elif N % 2 == 1 and Z % 2 == 1:
        pairing = -A_P / np.sqrt(A)
    else:
        pairing = 0.0

    return vol - surf - coulomb - asym + pairing


def binding_energy_per_nucleon(Z: int, A: int) -> float:
    """B/A in MeV per nucleon.  Peak near Fe-56 (~8.8 MeV/nucleon)."""
    return bethe_weizsacker(Z, A) / A


def mass_excess(Z: int, A: int) -> float:
    """Nuclear mass excess Delta = M(Z,A) - A*amu, in MeV.

    M(Z,A) = Z*m_p + N*m_n - B/c^2
    Delta   = (Z*m_p + N*m_n - A) * AMU_MEV - B
    (using m_p=1.007276, m_n=1.008665 in amu)
    """
    N = A - Z
    B = bethe_weizsacker(Z, A)
    return (Z * M_PROTON + N * M_NEUTRON - A) * AMU_MEV - B


# ---------------------------------------------------------------------------
# Radioactive decay  (Serway Ch 13.4-13.5)
# ---------------------------------------------------------------------------

def half_life_to_decay_constant(t_half: float) -> float:
    """lambda = ln(2) / t_{1/2}  (s^-1, if t_half in seconds)."""
    return np.log(2) / t_half


def activity(N0: float, t_half: float, t: float) -> float:
    """Activity A(t) = lambda * N(t) = lambda * N0 * exp(-lambda*t)  (decays/s).

    Parameters
    ----------
    N0     : initial number of radioactive nuclei
    t_half : half-life in seconds
    t      : elapsed time in seconds

    Returns
    -------
    Activity in Becquerel (1 Bq = 1 decay/s).
    """
    lam = half_life_to_decay_constant(t_half)
    return lam * N0 * np.exp(-lam * t)


def decay_chain(
    N0_list: List[float],
    half_lives: List[float],
    t_arr: np.ndarray,
) -> np.ndarray:
    """Bateman solution for a simple decay chain A -> B -> C -> ... (stable).

    Uses numerical Euler integration (accurate for smooth half-lives).

    Parameters
    ----------
    N0_list    : initial populations [N_A(0), N_B(0), ...]
    half_lives : half-lives of each species except the last (stable)
    t_arr      : time array (seconds)

    Returns
    -------
    N[i, j] = population of species i at time t_arr[j]
    """
    n_species = len(N0_list)
    N = np.zeros((n_species, len(t_arr)))
    N[:, 0] = N0_list
    lambdas = [np.log(2) / th for th in half_lives]

    for k in range(1, len(t_arr)):
        dt = t_arr[k] - t_arr[k-1]
        for i in range(n_species):
            feed = lambdas[i-1] * N[i-1, k-1] if i > 0 else 0.0
            loss = lambdas[i] * N[i, k-1] if i < len(lambdas) else 0.0
            N[i, k] = N[i, k-1] + dt * (feed - loss)

    return N


def carbon_dating_age(
    ratio: float,
    t_half_C14: float = 5730 * 365.25 * 24 * 3600,
) -> float:
    """Estimate age of organic sample from measured C-14/C-12 ratio.

    The atmosphere maintains a constant C-14/C-12 ratio R0 ~ 1.3e-12 while
    an organism is alive.  After death, C-14 decays with t_{1/2} = 5730 yr.

    The measured ratio R = R0 * exp(-lambda * t) gives:
        t = -ln(R/R0) / lambda

    Parameters
    ----------
    ratio    : measured C-14 / C-12 ratio (normalised to R0 = 1.0)
    t_half_C14 : C-14 half-life in seconds (default 5730 yr)

    Returns
    -------
    Age in years.
    """
    if ratio <= 0 or ratio > 1.0:
        raise ValueError("ratio must be in (0, 1] (1.0 = living organism)")
    lam = half_life_to_decay_constant(t_half_C14)
    t_sec = -np.log(ratio) / lam
    return t_sec / (365.25 * 24 * 3600)   # convert to years


# ---------------------------------------------------------------------------
# Neutron interactions  (Serway Ch 14.3)
# ---------------------------------------------------------------------------

def neutron_cross_section_estimate(A: int) -> float:
    """Geometric nuclear cross section sigma ~ pi * R^2  (barns).

    Nuclear radius R = R0 * A^(1/3), R0 = 1.2 fm = 1.2e-15 m.
    This is the geometric cross section; actual cross sections vary
    enormously due to resonances (thermal neutrons on Cd-113: 20,000 barns).

    Returns sigma in barns.
    """
    R0 = 1.2e-15   # meters
    R  = R0 * A**(1/3)
    return np.pi * R**2 / BARN


def neutron_mean_free_path(sigma_barn: float, n_density: float) -> float:
    """Mean free path lambda = 1 / (n * sigma)  (metres).

    Parameters
    ----------
    sigma_barn : total neutron cross section (barns)
    n_density  : number density of target nuclei (m^-3)

    Returns
    -------
    Mean free path in metres.
    """
    sigma = sigma_barn * BARN
    return 1.0 / (n_density * sigma)


def moderation_ratio(A: int) -> float:
    """Average fraction of neutron energy retained per elastic collision.

    For elastic collision with stationary nucleus of mass A (in amu),
    a neutron with mass ~1 retains on average:
        <E_f/E_i> = ((A-1)/(A+1))^2

    Moderating power: hydrogen (A=1) is ideal (average loss = 100%),
    which is why H2O is a common reactor moderator.

    Returns fraction of energy retained (0 = perfect moderation, 1 = none).
    """
    return ((A - 1) / (A + 1))**2


def fission_q_value(Z_parent: int, A_parent: int,
                    Z1: int, A1: int, Z2: int, A2: int,
                    n_neutrons: int = 0) -> float:
    """Q-value of a fission reaction in MeV.

    Q = B(fragments) - B(parent) + B(neutrons absorbed - emitted)
      = B(Z1,A1) + B(Z2,A2) - B(Z_parent, A_parent)

    Positive Q means energy is released.

    Example: U-235 + n -> Ba-141 + Kr-92 + 3n
        fission_q_value(92, 236, 56, 141, 36, 92, n_neutrons=3)
        -> ~200 MeV
    """
    B_parent = bethe_weizsacker(Z_parent, A_parent)
    B1       = bethe_weizsacker(Z1, A1)
    B2       = bethe_weizsacker(Z2, A2)
    return B1 + B2 - B_parent


# ---------------------------------------------------------------------------
# Bethe-Bloch stopping power  (Serway Ch 14.7)
# ---------------------------------------------------------------------------

def bethe_bloch_stopping(
    z: int,
    beta: float,
    Z_target: int,
    A_target: int,
    I_eV: float = None,
    density_g_cm3: float = 1.0,
) -> float:
    """Non-relativistic Bethe-Bloch stopping power -dE/dx  (MeV/cm).

    -dE/dx = (4*pi*e^4 * z^2 * Z) / (m_e * v^2 * A) * n_e * ln(2*m_e*v^2/I)

    In practical units (Bethe formula):
    -dE/dx [MeV g^-1 cm^2] = 0.307 * z^2 * Z/A * 1/beta^2 * (1/2 * ln(...) - beta^2)

    Parameters
    ----------
    z             : charge of incident particle in units of e (e.g., 2 for alpha)
    beta          : v/c of incident particle
    Z_target      : atomic number of target material
    A_target      : mass number of target material
    I_eV          : mean excitation potential (eV); default: 13.5 * Z eV (Bloch estimate)
    density_g_cm3 : density of target material (g/cm^3) for converting to MeV/cm

    Returns
    -------
    Stopping power -dE/dx in MeV/cm.
    """
    if beta <= 0 or beta >= 1:
        raise ValueError("beta must be in (0, 1)")
    if I_eV is None:
        I_eV = 13.5 * Z_target          # Bloch estimate: I ~ 13.5Z eV
    I_MeV = I_eV * 1e-6

    m_e_MeV = 0.511                     # electron rest mass in MeV

    # Bethe-Bloch in MeV g^-1 cm^2 (mass stopping power)
    # This is the standard form; K = 4*pi*N_A*r_e^2*m_e*c^2 = 0.307 MeV cm^2/g
    K = 0.30707   # MeV cm^2 / g

    log_arg = (2 * m_e_MeV * beta**2) / (I_MeV * (1 - beta**2)) - beta**2
    if log_arg <= 0:
        return 0.0

    mass_stop = K * z**2 * (Z_target / A_target) * (1 / beta**2) * (0.5 * np.log(log_arg) - beta**2)
    return max(mass_stop * density_g_cm3, 0.0)   # MeV/cm


def range_estimate(
    KE_MeV: float,
    z: int,
    m_amu: float,
    Z_target: int,
    A_target: int,
    density_g_cm3: float = 1.0,
    n_steps: int = 1000,
) -> float:
    """Estimate range of charged particle in target material by integrating -dE/dx.

    Uses the continuous-slowing-down approximation (CSDA):
        R = int_0^E0 dE / (dE/dx)

    Parameters
    ----------
    KE_MeV        : initial kinetic energy (MeV)
    z             : particle charge (units of e)
    m_amu         : particle mass (amu)
    Z_target, A_target: target material
    density_g_cm3 : target density
    n_steps       : integration steps

    Returns
    -------
    Range in cm.
    """
    m_MeV = m_amu * AMU_MEV
    E = KE_MeV
    R = 0.0
    dE = E / n_steps

    while E > 1e-4:   # stop when below 100 eV
        v = np.sqrt(2 * E / m_MeV) * C_LIGHT   # non-relativistic KE = 1/2 mv^2
        beta = v / C_LIGHT
        if beta <= 0 or beta >= 0.99:
            break
        sp = bethe_bloch_stopping(z, beta, Z_target, A_target,
                                  density_g_cm3=density_g_cm3)
        if sp <= 0:
            break
        dx = min(dE / sp, 0.1)   # cap step size at 1 mm
        R += dx
        E -= sp * dx

    return R


# ---------------------------------------------------------------------------
# Radiation dose  (Serway Ch 14.8)
# ---------------------------------------------------------------------------

def dose_gray(energy_J: float, mass_kg: float) -> float:
    """Absorbed dose in Grays (1 Gy = 1 J/kg).

    Parameters
    ----------
    energy_J : energy deposited in tissue (Joules)
    mass_kg  : mass of tissue irradiated (kg)

    Returns
    -------
    Dose in Gy.
    """
    return energy_J / mass_kg


def dose_sievert(dose_gy: float, quality_factor: float) -> float:
    """Effective dose in Sievert = dose_gray * quality_factor.

    Quality factors (ICRP):
      X-rays, gamma, beta : Q = 1
      Protons             : Q = 5
      Alpha particles     : Q = 20
      Fast neutrons       : Q = 10-20

    Low-dose background: ~3 mSv/yr (cosmic + natural radioactivity).
    Radiation sickness threshold: 1 Sv whole-body acute.
    LD50/30 (lethal to 50% in 30 days): ~4-5 Sv without treatment.
    """
    return dose_gy * quality_factor


# ---------------------------------------------------------------------------
# Food preservation by radiation  (Serway Ch 14.10)
# ---------------------------------------------------------------------------

FOOD_PRESERVATION_DOSES = {
    'low_dose_0.1_kGy': {
        'dose_kGy': 0.1,
        'application': 'Inhibit sprouting of potatoes, onions',
        'organisms_killed': 'Insects, some parasites',
        'FDA_approved': True,
    },
    'medium_dose_1_kGy': {
        'dose_kGy': 1.0,
        'application': 'Extend shelf life of strawberries, delay ripening',
        'organisms_killed': 'Salmonella, E. coli (partial reduction)',
        'FDA_approved': True,
    },
    'medium_dose_3_kGy': {
        'dose_kGy': 3.0,
        'application': 'Poultry, meat pathogens (FDA 1997)',
        'organisms_killed': 'Salmonella, Campylobacter, E. coli O157:H7',
        'FDA_approved': True,
    },
    'high_dose_10_kGy': {
        'dose_kGy': 10.0,
        'application': 'Sterilisation of spices, dried herbs',
        'organisms_killed': 'Virtually all bacteria, spores, mold',
        'FDA_approved': True,
    },
    'ultra_high_25_kGy': {
        'dose_kGy': 25.0,
        'application': 'Medical device sterilisation (not food)',
        'organisms_killed': 'All microorganisms including prions (partial)',
        'FDA_approved': False,  # medical, not food
    },
}


def food_preservation_dose(application_key: str) -> dict:
    """Return dose parameters for a food irradiation application.

    Keys: 'low_dose_0.1_kGy', 'medium_dose_1_kGy', 'medium_dose_3_kGy',
          'high_dose_10_kGy', 'ultra_high_25_kGy'
    """
    if application_key not in FOOD_PRESERVATION_DOSES:
        keys = list(FOOD_PRESERVATION_DOSES.keys())
        raise ValueError(f"Unknown key. Options: {keys}")
    return FOOD_PRESERVATION_DOSES[application_key]


def irradiation_energy_deposited(
    dose_kGy: float,
    mass_kg: float,
    source: str = 'Co-60 gamma',
) -> dict:
    """Compute energy deposited and number of Co-60 decays needed.

    Co-60 emits two gamma photons per decay: 1.173 MeV + 1.332 MeV = 2.505 MeV total.
    Assuming 100% absorption efficiency (upper bound).

    Parameters
    ----------
    dose_kGy : absorbed dose in kGy
    mass_kg  : food mass in kg
    source   : radiation source description (informational)

    Returns
    -------
    dict with energy_J, n_decays_needed, activity_needed_Ci
    """
    dose_gy     = dose_kGy * 1e3                 # J/kg
    energy_J    = dose_gy * mass_kg              # total energy
    E_per_decay = 2.505e6 * Q_E                 # Co-60: 2.505 MeV per decay

    n_decays = energy_J / E_per_decay
    # For 1-second irradiation: A = n_decays / 1s; convert to Curie (1 Ci = 3.7e10 Bq)
    activity_bq = n_decays                       # if 1-second exposure
    activity_ci = activity_bq / 3.7e10

    return {
        'dose_kGy'       : dose_kGy,
        'mass_kg'        : mass_kg,
        'energy_J'       : energy_J,
        'source'         : source,
        'n_decays_needed': n_decays,
        'activity_Ci_for_1s': activity_ci,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("Nuclear Physics Demo")
    print("=" * 55)

    # 1. Binding energy curve
    nuclei = [(2,4,'He-4'), (6,12,'C-12'), (8,16,'O-16'),
              (26,56,'Fe-56'), (82,208,'Pb-208'), (92,235,'U-235'), (92,238,'U-238')]
    print("\n1. Binding energy per nucleon (Bethe-Weizsacker SEMF):")
    print(f"   Note: SEMF is fitted for A >= 12; He-4 is already approximate.")
    print(f"   {'Nucleus':<10}  {'B (MeV)':>10}  {'B/A (MeV)':>12}")
    print(f"   {'-'*36}")
    for Z, A, name in nuclei:
        B   = bethe_weizsacker(Z, A)
        BpA = B / A
        print(f"   {name:<10}  {B:>10.2f}  {BpA:>12.4f}")
    print(f"   Peak is near Fe-56 (~8.8 MeV/nucleon): iron is the endpoint")
    print(f"   of stellar fusion -- heavier nuclei require energy to form.")

    # 2. Radioactive decay: C-14 carbon dating
    print(f"\n2. Carbon dating:")
    for pct in [100, 75, 50, 25, 10, 1]:
        ratio = pct / 100.0
        if ratio == 1.0:
            age = 0.0
        else:
            age = carbon_dating_age(ratio)
        print(f"   C-14 remaining: {pct:3d}%  ->  age = {age:,.0f} years")

    # 3. Neutron moderation
    print(f"\n3. Neutron moderation (fraction of energy retained per collision):")
    for A, mat in [(1,'H (water)'), (2,'D (heavy water)'), (12,'C (graphite)'),
                   (238,'U (fuel)')]:
        f = moderation_ratio(A)
        print(f"   A={A:3d} ({mat:<16}): {f:.4f}  ({(1-f)*100:.1f}% lost per collision)")

    # 4. Fission Q-value: U-235 + n -> Ba-141 + Kr-92 + 3n
    Q = fission_q_value(92, 236, 56, 141, 36, 92)
    print(f"\n4. U-235 fission Q-value (SEMF):")
    # 1 kg U-235: N_atoms = 1000/235 * N_A; E_total = N_atoms * Q * 1e6 * Q_E
    N_atoms = (1000.0 / 235.0) * N_AVOGADRO
    E_total_J = N_atoms * Q * 1e6 * Q_E          # Joules
    coal_GJ_per_ton = 30.0                        # 30 GJ / metric ton of coal
    coal_equiv = E_total_J / (coal_GJ_per_ton * 1e9)
    print(f"   U-236 -> Ba-141 + Kr-92 + 3n: Q = {Q:.1f} MeV")
    print(f"   (Measured: ~202 MeV; SEMF is approximate)")
    print(f"   1 kg U-235 fission energy ~ {E_total_J:.2e} J = {coal_equiv:.0f} metric tons of coal")

    # 5. Stopping power: alpha particle in tissue (Z=2, m=4amu, 5.5 MeV)
    sp = bethe_bloch_stopping(z=2, beta=0.054, Z_target=7, A_target=14,
                               density_g_cm3=1.0)
    R  = range_estimate(KE_MeV=5.5, z=2, m_amu=4.0,
                        Z_target=7, A_target=14, density_g_cm3=1.0)
    print(f"\n5. Alpha particle (5.5 MeV) in tissue (N, rho=1 g/cm^3):")
    print(f"   Stopping power: {sp:.2f} MeV/cm")
    print(f"   Range estimate: {R*10:.2f} mm  (actual: ~40 um; alpha stopped by skin)")
    print(f"   Alpha is dangerous INTERNALLY (lung, GI tract) but harmless externally.")

    # 6. Radiation dose context
    print(f"\n6. Radiation dose reference table:")
    doses = [
        ('Background (annual)',       3e-3,  1,  'mSv'),
        ('Chest X-ray',               0.02e-3, 1, 'uSv'),
        ('CT scan (chest)',            7e-3,  1,  'mSv'),
        ('Radiation therapy (tumor)', 60.0,  1,  'Gy per treatment'),
        ('Acute radiation sickness',  1.0,   1,  'Sv whole body'),
        ('LD50/30 (no treatment)',    4.5,   1,  'Sv whole body'),
    ]
    for label, val, Q_factor, unit in doses:
        print(f"   {label:<35}: {val} {unit}")

    # 7. Food irradiation
    print(f"\n7. Food preservation by radiation (Co-60 gamma):")
    print(f"   {'Application':<40}  {'Dose (kGy)':>12}  {'FDA?':>6}")
    print(f"   {'-'*62}")
    for key, d in FOOD_PRESERVATION_DOSES.items():
        fda = 'Yes' if d['FDA_approved'] else 'No'
        print(f"   {d['application']:<40}  {d['dose_kGy']:>12.1f}  {fda:>6}")

    result = irradiation_energy_deposited(dose_kGy=3.0, mass_kg=1.0)
    print(f"\n   Energy for 3 kGy in 1 kg food: {result['energy_J']:.0f} J")
    print(f"   Co-60 activity needed (1s irradiation): {result['activity_Ci_for_1s']:.0f} Ci")
    print(f"   (Industrial irradiators use ~1e6 Ci sources)")

    print("\nDemo complete.")


if __name__ == "__main__":
    demo()
