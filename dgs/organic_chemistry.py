"""Organic chemistry for EE and photonics -- molecular orbitals, chromophores,
organic semiconductors, and the photon-molecule interaction.

WHY ORGANIC CHEMISTRY FOR EE:
  OLED displays: organic emitters (Alq3, Ir complexes) convert electrons to photons
  Organic solar cells (OPV): HOMO-LUMO gap determines absorbed wavelength
  Photoresists: UV photons break C-C bonds in lithography
  Chromophores: pi-conjugated molecules that absorb/emit at specific wavelengths
  DNA photodamage: UV absorbs in the 260 nm pyrimidine pi* bands
  PDT sensitizers: porphyrins (already in biophotonics.py) are organic molecules

MOLECULAR ORBITAL THEORY (key ideas for EE):
  HOMO = Highest Occupied Molecular Orbital  (like valence band in solid-state)
  LUMO = Lowest Unoccupied Molecular Orbital (like conduction band)
  Gap = LUMO - HOMO energy (in eV) = determines optical properties
  Color: absorbed wavelength lambda (nm) = 1240 / Gap(eV)   [Planck: E = hc/lambda]

  The HOMO-LUMO gap maps directly to the semiconductor bandgap:
    - Large gap (>3 eV): UV-absorbing, colorless to eye (e.g., benzene 5.2 eV)
    - Gap 1.8-3 eV: visible light (colored molecules, OLEDs)
    - Small gap (<1.5 eV): IR-absorbing, photovoltaic use

PI-CONJUGATION INCREASES ORBITAL DELOCALIZATION:
  Single bond: localized sigma bond, large gap
  Alternating double bonds: pi electrons delocalize across the chain
  Longer conjugation -> smaller gap (red-shift) -> longer wavelength absorption
  This is the SAME physics as the particle-in-a-box: L increases -> E_n decreases

HUCKEL THEORY (the FEM of molecular orbitals):
  For a linear pi-conjugated chain of N carbons:
    E_k = alpha + 2*beta * cos(k*pi / (N+1)),  k = 1,2,...,N
  where alpha = Coulomb integral (~-11.4 eV for C), beta = resonance integral (~-2.4 eV)
  HOMO is the N/2 level, LUMO is the N/2 + 1 level (for even N).
  Gap = E_LUMO - E_HOMO = 2*|beta| * [cos(N/2*pi/(N+1)) - cos((N/2+1)*pi/(N+1))]

ORGANIC SEMICONDUCTORS (photonics relevance):
  Pentacene: 5 fused benzene rings, gap ~1.9 eV, OPV donor material
  PCBM ([60]PCBM): fullerene derivative, gap ~1.7 eV, OPV acceptor
  Alq3: Al chelate, OLED green emitter (~520 nm, gap 2.7 eV)
  P3HT: polythiophene, gap ~1.9 eV, flexible electronics
  TIPS-pentacene: soluble, high mobility (5 cm^2/Vs), used in organic TFTs
"""
import numpy as np
import sympy as sp


# ── constants ─────────────────────────────────────────────────────────

hc_eV_nm = 1239.84   # h*c in eV*nm (Planck: E = hc/lambda)
ALPHA_C = -11.4      # eV  Coulomb integral for sp2 carbon
BETA_C  = -2.4       # eV  Resonance integral for C=C pi bond (Huckel)


# ── HOMO-LUMO gap and color ───────────────────────────────────────────

def homo_lumo_to_wavelength(gap_eV):
    """Absorbed wavelength for a chromophore with given HOMO-LUMO gap.

    E = hc/lambda => lambda (nm) = 1239.84 / gap(eV)

    The COMPLEMENTARY color is what you SEE (the unabsorbed portion):
      absorbed 400-450 nm (violet) -> appears yellow
      absorbed 450-495 nm (blue)   -> appears orange
      absorbed 495-570 nm (green)  -> appears red-purple
      absorbed 570-620 nm (yellow) -> appears violet
    """
    if gap_eV <= 0:
        raise ValueError("gap_eV must be positive")
    lambda_abs_nm = hc_eV_nm / gap_eV
    # approximate complementary color
    complementary = _complementary_color(lambda_abs_nm)
    return {
        "gap_eV": gap_eV,
        "lambda_abs_nm": lambda_abs_nm,
        "complementary_color": complementary,
        "visible": 380 < lambda_abs_nm < 700,
    }


def _complementary_color(lambda_nm):
    if lambda_nm < 380 or lambda_nm > 700:
        return "UV/IR (colorless or colorless to eye)"
    if lambda_nm < 450:
        return "yellow-orange"
    if lambda_nm < 495:
        return "orange-red"
    if lambda_nm < 570:
        return "red-purple"
    if lambda_nm < 620:
        return "violet-blue"
    return "blue-green"


# ── Huckel MO theory ─────────────────────────────────────────────────

def huckel_pi_chain(N_carbons, alpha_eV=ALPHA_C, beta_eV=BETA_C):
    """Huckel MO energies for a LINEAR conjugated pi chain of N sp2 carbons.

    E_k = alpha + 2*beta * cos(k*pi / (N+1))  for k = 1,...,N

    HOMO = level N//2, LUMO = level N//2 + 1 (even N).
    Returns energies (eV), HOMO, LUMO, gap.

    This is the tight-binding chain -- the same model as the Kitaev chain
    in quantum_information.py, but for molecular orbitals instead of
    superconducting topological phases.
    """
    if N_carbons < 2:
        raise ValueError("N_carbons must be >= 2")
    k = np.arange(1, N_carbons + 1)
    E_k = alpha_eV + 2 * beta_eV * np.cos(k * np.pi / (N_carbons + 1))
    E_k_sorted = np.sort(E_k)   # ascending energy

    if N_carbons % 2 == 0:
        homo_idx = N_carbons // 2 - 1
        lumo_idx = N_carbons // 2
    else:
        homo_idx = (N_carbons - 1) // 2
        lumo_idx = homo_idx + 1

    E_homo = E_k_sorted[homo_idx]
    E_lumo = E_k_sorted[lumo_idx]
    gap = E_lumo - E_homo

    return {
        "N": N_carbons, "E_k_eV": E_k_sorted,
        "E_homo_eV": E_homo, "E_lumo_eV": E_lumo,
        "gap_eV": gap,
        "lambda_abs_nm": hc_eV_nm / gap if gap > 0 else float("inf"),
        "alpha_eV": alpha_eV, "beta_eV": beta_eV,
    }


def huckel_benzene():
    """Special case: benzene (N=6 carbons in a RING, not a chain).

    Ring boundary condition: E_k = alpha + 2*beta*cos(2*pi*k/N), k=0,1,...,N-1
    Degenerate levels at k=1 and k=N-1 (and k=2 and k=N-2).
    This is the DFT spectrum -- N-point DFT eigenvalues.
    """
    N = 6
    k = np.arange(N)
    E_k = ALPHA_C + 2 * BETA_C * np.cos(2 * np.pi * k / N)
    E_sorted = np.sort(E_k)
    # 6 pi electrons fill 3 levels: E(k=0), and degenerate E(k=1,5)
    E_homo = E_sorted[2]    # 3rd level (0-indexed: 0,1,2)
    E_lumo = E_sorted[3]    # 4th level
    gap = E_lumo - E_homo
    return {
        "molecule": "benzene",
        "N": N, "E_k_eV": E_sorted,
        "E_homo_eV": E_homo, "E_lumo_eV": E_lumo,
        "gap_eV": gap,
        "lambda_abs_nm": hc_eV_nm / gap,
        "note": "benzene absorbs ~250 nm (UV), appears colorless",
    }


# ── functional groups ─────────────────────────────────────────────────

FUNCTIONAL_GROUPS = {
    "alkane":     {"formula": "-CH2-", "reactivity": "low", "example": "ethane C2H6",
                   "ee_relevance": "insulating, low dielectric, paraffin coatings"},
    "alkene":     {"formula": ">C=C<", "reactivity": "addition (HX, H2, Br2)",
                   "example": "ethylene C2H4",
                   "ee_relevance": "polyethylene, spin-on dielectrics"},
    "alkyne":     {"formula": "-C=C-", "reactivity": "addition",
                   "example": "acetylene C2H2",
                   "ee_relevance": "pi-conjugated wires in molecular electronics"},
    "alcohol":    {"formula": "-OH", "reactivity": "dehydration, oxidation",
                   "example": "ethanol",
                   "ee_relevance": "photoresist developers, ALD precursors"},
    "aldehyde":   {"formula": "-CHO", "reactivity": "nucleophilic addition",
                   "example": "formaldehyde HCHO",
                   "ee_relevance": "crosslinker in phenolic resins (PCB laminate)"},
    "ketone":     {"formula": "R-CO-R'", "reactivity": "nucleophilic addition",
                   "example": "acetone",
                   "ee_relevance": "photoacid generator in EUV resists"},
    "carboxylic": {"formula": "-COOH", "reactivity": "esterification, deprotonation",
                   "example": "acetic acid",
                   "ee_relevance": "surface functionalization of metal contacts"},
    "amine":      {"formula": "-NH2", "reactivity": "basicity, acylation",
                   "example": "aniline",
                   "ee_relevance": "organic dopants, OLED hole-transport layers"},
    "halide":     {"formula": "-X (X=F,Cl,Br,I)", "reactivity": "substitution",
                   "example": "chloromethane",
                   "ee_relevance": "organochlorosilanes for surface passivation"},
    "nitro":      {"formula": "-NO2", "reactivity": "reduction to amine",
                   "example": "nitrobenzene",
                   "ee_relevance": "explosive photoacid, UV-sensitive resist component"},
    "ester":      {"formula": "-COO-", "reactivity": "saponification, transesterification",
                   "example": "ethyl acetate",
                   "ee_relevance": "PMMA backbone (electron-beam resist)"},
    "aromatic":   {"formula": "benzene ring C6H6", "reactivity": "electrophilic aromatic sub.",
                   "example": "benzene, naphthalene, anthracene",
                   "ee_relevance": "pi-conjugated backbones of OLED/OPV materials"},
}


def lookup_functional_group(name):
    """Look up a functional group by name."""
    name_lower = name.lower().strip()
    if name_lower not in FUNCTIONAL_GROUPS:
        avail = list(FUNCTIONAL_GROUPS.keys())
        raise ValueError(f"Unknown group '{name}'. Available: {avail}")
    return FUNCTIONAL_GROUPS[name_lower]


# ── organic semiconductor database ───────────────────────────────────

ORGANIC_SEMICONDUCTORS = {
    "pentacene": {
        "formula": "C22H14", "n_rings": 5,
        "gap_eV": 1.9, "lambda_abs_nm": 660,
        "HOMO_eV": -5.0, "LUMO_eV": -3.1,
        "mobility_cm2_Vs": 5.0, "type": "donor",
        "use": "OPV donor, OTFT channel",
    },
    "PCBM": {
        "formula": "C61H23NO2 (C60 derivative)", "n_rings": 1,
        "gap_eV": 1.7, "lambda_abs_nm": 730,
        "HOMO_eV": -6.1, "LUMO_eV": -4.4,
        "mobility_cm2_Vs": 1e-3, "type": "acceptor",
        "use": "OPV acceptor, electron transport",
    },
    "Alq3": {
        "formula": "Al(C9H6NO)3", "n_rings": 9,
        "gap_eV": 2.7, "lambda_abs_nm": 510,
        "HOMO_eV": -5.8, "LUMO_eV": -3.1,
        "mobility_cm2_Vs": 1e-5, "type": "ambipolar",
        "use": "OLED green emitter (first vacuum-deposited OLED, Tang & VanSlyke 1987)",
    },
    "P3HT": {
        "formula": "(C10H14S)n", "n_rings": None,
        "gap_eV": 1.9, "lambda_abs_nm": 550,
        "HOMO_eV": -5.1, "LUMO_eV": -3.2,
        "mobility_cm2_Vs": 0.1, "type": "donor",
        "use": "OPV donor, flexible electronics, thermoelectrics",
    },
    "ITIC": {
        "formula": "C82H66N4O4S3", "n_rings": 12,
        "gap_eV": 1.6, "lambda_abs_nm": 775,
        "HOMO_eV": -5.5, "LUMO_eV": -3.9,
        "mobility_cm2_Vs": 1e-4, "type": "non-fullerene acceptor",
        "use": "High-efficiency OPV acceptor (PCE > 14% in blends)",
    },
}


def organic_semiconductor(name):
    """Look up an organic semiconductor by name."""
    if name not in ORGANIC_SEMICONDUCTORS:
        raise ValueError(f"Unknown semiconductor '{name}'. Options: {list(ORGANIC_SEMICONDUCTORS.keys())}")
    return ORGANIC_SEMICONDUCTORS[name]


# ── Beer-Lambert absorption for chromophores ──────────────────────────

def chromophore_absorption(epsilon_L_mol_cm, conc_mol_L, path_cm):
    """Beer-Lambert law for molecular absorption:
    A = epsilon * c * l  (absorbance, dimensionless)
    T = 10^(-A)  (transmittance)
    Identical in form to I(z) = I0 * exp(-mu*z) from biophotonics.py
    but using molar extinction coefficient epsilon and molar concentration c.

    For photoresists, epsilon ~ 10^3-10^5 L/(mol*cm) at exposure wavelength.
    """
    if epsilon_L_mol_cm < 0 or conc_mol_L < 0 or path_cm < 0:
        raise ValueError("epsilon, concentration, and path must be non-negative")
    A = epsilon_L_mol_cm * conc_mol_L * path_cm
    T = 10 ** (-A)
    return {
        "absorbance_A": A,
        "transmittance_T": T,
        "absorbance_percent": (1 - T) * 100,
        "OD_definition": "A = -log10(T) = epsilon * c * l",
    }


# ── reaction types ────────────────────────────────────────────────────

REACTION_TYPES = {
    "addition":         "two reactants combine to one product; alkenes + HX, H2, Br2",
    "substitution":     "one group replaces another; SN1/SN2 for alkyl halides; EAS for aromatics",
    "elimination":      "removes H and X to form C=C; E1/E2; opposite of addition",
    "condensation":     "two molecules join with loss of water; ester formation, peptide bond",
    "oxidation":        "loss of electrons/H or gain of O; alcohol -> aldehyde -> carboxylic acid",
    "reduction":        "gain of electrons/H; aldehyde -> alcohol; nitro -> amine",
    "radical":          "homolytic cleavage produces radicals; free-radical polymerization",
    "photochemical":    "photon absorbed, excited state reacts; [2+2] cycloaddition, photo-crosslinking",
    "polymerization":   "monomers chain into polymer; addition (styrene->polystyrene) or condensation (nylon)",
}


def list_reaction_types():
    """Print all reaction types with EE applications."""
    return REACTION_TYPES


# ── particle-in-a-box model for conjugated chain ──────────────────────

def pib_conjugated_chain(N_double_bonds, bond_length_nm=0.14):
    """Particle-in-a-box model for a linear conjugated pi chain.

    The pi electrons are delocalized over a box of length L = 2*N*d
    where d = C-C bond length (~0.14 nm for aromatic, ~0.135 nm for vinyl).
    Ground state energy transition (N/2 -> N/2+1 for N pi electrons):

    E_n = n^2 * h^2 / (8 * m_e * L^2)
    Delta_E = (2*N + 1) * h^2 / (8 * m_e * L^2)

    This MODEL predicts the color TREND correctly but is less accurate than
    Huckel for the absolute wavelength.
    """
    if N_double_bonds < 1:
        raise ValueError("N_double_bonds must be >= 1")
    N_pi = 2 * N_double_bonds      # each double bond contributes 2 pi electrons
    L_m = 2 * N_double_bonds * bond_length_nm * 1e-9   # box length in meters
    # physical constants
    h = 6.62607e-34    # J*s
    me = 9.10938e-31   # kg electron mass
    n_homo = N_pi // 2        # HOMO quantum number
    n_lumo = n_homo + 1       # LUMO quantum number
    E_homo = n_homo**2 * h**2 / (8 * me * L_m**2)
    E_lumo = n_lumo**2 * h**2 / (8 * me * L_m**2)
    delta_E_J = E_lumo - E_homo
    delta_E_eV = delta_E_J / 1.60218e-19
    lambda_nm = hc_eV_nm / delta_E_eV
    return {
        "N_double_bonds": N_double_bonds,
        "N_pi_electrons": N_pi,
        "L_nm": L_m * 1e9,
        "gap_eV": delta_E_eV,
        "lambda_abs_nm": lambda_nm,
        "n_homo": n_homo, "n_lumo": n_lumo,
    }


# ── SymPy formalism ───────────────────────────────────────────────────

def organic_chemistry_sympy_5():
    """Five key organic/photonics equations in SymPy."""
    n, N, L, me, h = sp.symbols('n N L m_e h', positive=True)
    alpha_s, beta_s, k_s = sp.symbols('alpha beta k', real=True)
    eps, c_s, ell = sp.symbols('epsilon c l', positive=True)
    E_gap, lam = sp.symbols('E_gap lambda', positive=True)

    return {
        "Huckel_energy":
            sp.Eq(sp.Symbol('E_k'),
                  alpha_s + 2*beta_s * sp.cos(k_s * sp.pi / (N + 1))),
        "Planck_photon_energy":
            sp.Eq(E_gap, sp.Symbol('hc') / lam),
        "PIB_transition_energy":
            sp.Eq(sp.Symbol('DeltaE'),
                  (2*n + 1) * h**2 / (8 * me * L**2)),
        "Beer_Lambert":
            sp.Eq(sp.Symbol('A'), eps * c_s * ell),
        "HOMO_LUMO_gap":
            sp.Eq(lam, sp.Rational(1240, 1) / E_gap),
    }


if __name__ == "__main__":
    print("=== Huckel MO: linear pi chains ===")
    print(f"  {'N':>4}  {'gap(eV)':>10}  {'lambda(nm)':>12}  note")
    for N in [2, 4, 6, 8, 10, 14]:
        r = huckel_pi_chain(N)
        note = "(visible)" if 380 < r["lambda_abs_nm"] < 700 else ""
        print(f"  {N:>4}  {r['gap_eV']:>10.2f}  {r['lambda_abs_nm']:>12.0f}  {note}")

    print("\n=== Benzene ring (Huckel ring) ===")
    b = huckel_benzene()
    print(f"  gap = {b['gap_eV']:.2f} eV, lambda = {b['lambda_abs_nm']:.0f} nm")
    print(f"  note: {b['note']}")

    print("\n=== Particle-in-a-box: beta-carotene (11 double bonds) ===")
    pib = pib_conjugated_chain(11)
    print(f"  gap = {pib['gap_eV']:.2f} eV, lambda = {pib['lambda_abs_nm']:.0f} nm "
          f"(orange absorption, beta-carotene appears orange)")

    print("\n=== Organic semiconductor database ===")
    for name, osc in ORGANIC_SEMICONDUCTORS.items():
        print(f"  {name}: gap={osc['gap_eV']} eV, lambda={osc['lambda_abs_nm']} nm, "
              f"mu={osc['mobility_cm2_Vs']:.1e} cm^2/Vs ({osc['type']})")

    print("\n=== Beer-Lambert: PCBM film 100nm thick, 20 mM, epsilon=30000 ===")
    bl = chromophore_absorption(30000, 0.020, 100e-7)
    print(f"  Absorbance A = {bl['absorbance_A']:.4f}")
    print(f"  Transmittance T = {bl['transmittance_T']:.4f}")
    print(f"  Absorbed = {bl['absorbance_percent']:.2f}%")

    print("\n=== SymPy 5 ===")
    for k, eq in organic_chemistry_sympy_5().items():
        print(f"  {k}: {eq}")
