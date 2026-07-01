"""Feynman Lectures: atomic and molecular physics -- organized by TOPIC, not chapter.

NON-LINEAR ORDER THROUGH FEYNMAN (Vol I Ch.1-2, 38-40; Vol II; Vol III):
  The Feynman Lectures are ordered pedagogically for 1961 Caltech freshmen.
  We re-order by the PHYSICS of what you need to know as an EE/photonics researcher.

  TOPIC ORDER (this module):
    1. ATOMIC HYPOTHESIS           (Vol I, Ch.1) -- the foundation
    2. KINETIC THEORY + GAS LAWS   (Vol I, Ch.39-40) -- statistical mechanics entry
    3. ATOMIC STRUCTURE + BOHR     (Vol I, Ch.38; Vol III) -- quantization of energy
    4. SPECTRAL LINES + SELECTION  (connects to photon emission/absorption)
    5. MOLECULAR BINDING           (Vol I, Ch.12; Vol III) -- covalent bonds
    6. VAN DER WAALS / DISPERSION  (Vol II, Ch.11) -- optical susceptibility root
    7. BLACKBODY / QUANTIZATION    (Vol I, Ch.40) -- Planck distribution
    8. MANY-BODY + BAND THEORY     (Vol III, Ch.13-14) -- solid state bridge

  FEYNMAN'S ATOMIC HYPOTHESIS (1965 Nobel Lecture, also Vol I Ch.1):
    'If, in some cataclysm, all of scientific knowledge were to be destroyed,
    and only one sentence passed on to the next generations of creatures, what
    statement would contain the most information in the fewest words?
    I believe it is the atomic hypothesis: that all things are made of atoms --
    little particles that move around in perpetual motion, attracting each other
    when they are a little distance apart but repelling upon being squeezed into
    one another.'

  CONNECTION TO THIS REPO:
    The GS algorithm is a method to recover the PHASE of an atomic/molecular
    wavefunction (or optical field) from intensity measurements alone.
    Crystallography (the original Gerchberg-Saxton problem!) recovers protein
    atomic positions from X-ray diffraction intensity patterns.
"""
import numpy as np
import sympy as sp


# ── constants ─────────────────────────────────────────────────────────

h_planck = 6.62607e-34   # J*s
hbar      = h_planck / (2*np.pi)
k_B       = 1.38065e-23  # J/K
c_light   = 2.99792e8    # m/s
e_charge  = 1.60218e-19  # C
m_e       = 9.10938e-31  # kg electron mass
a_0       = 5.29177e-11  # m Bohr radius
E_rydberg = 13.60569     # eV Rydberg energy = ionization energy of H


# ── 1. Atomic hypothesis: kinetic gas model ───────────────────────────

def ideal_gas(n_molecules, T_K, V_m3=None, P_Pa=None):
    """Kinetic theory of the ideal gas. PV = NkT.

    Feynman Vol I Ch.39: the pressure of a gas comes from momentum transfer
    during atomic collisions with the wall. This is the FIRST derivation
    of a macroscopic law from atomic assumptions.

    Provide either V_m3 or P_Pa; the other is computed.
    """
    if V_m3 is None and P_Pa is None:
        raise ValueError("Provide either V_m3 or P_Pa")
    if T_K <= 0:
        raise ValueError("Temperature must be positive")
    if V_m3 is not None and V_m3 <= 0:
        raise ValueError("Volume must be positive")

    if V_m3 is None:
        V_m3 = n_molecules * k_B * T_K / P_Pa
    if P_Pa is None:
        P_Pa = n_molecules * k_B * T_K / V_m3

    v_rms = np.sqrt(3 * k_B * T_K / (m_e * 1836))  # for H atom (approx)
    KE_per_atom_eV = 1.5 * k_B * T_K / e_charge

    return {
        "P_Pa": P_Pa, "V_m3": V_m3, "T_K": T_K,
        "n_molecules": n_molecules,
        "KE_per_atom_eV": KE_per_atom_eV,
        "PV_product_J": P_Pa * V_m3,
        "NkT": n_molecules * k_B * T_K,
    }


def maxwell_boltzmann_speed(T_K, m_kg, v_m_s):
    """Maxwell-Boltzmann speed distribution for a gas at temperature T.

    f(v) = 4*pi * (m/2*pi*kT)^(3/2) * v^2 * exp(-m*v^2 / (2*k*T))

    Feynman Vol I Ch.40: this distribution is the SEED of all statistical
    mechanics. The exp(-E/kT) factor is the Boltzmann factor.
    """
    if T_K <= 0 or m_kg <= 0:
        raise ValueError("T and m must be positive")
    A = 4 * np.pi * (m_kg / (2 * np.pi * k_B * T_K))**1.5
    f = A * v_m_s**2 * np.exp(-m_kg * v_m_s**2 / (2 * k_B * T_K))
    v_mp  = np.sqrt(2 * k_B * T_K / m_kg)    # most probable speed
    v_avg = np.sqrt(8 * k_B * T_K / (np.pi * m_kg))  # average speed
    v_rms = np.sqrt(3 * k_B * T_K / m_kg)    # rms speed
    return {
        "f_v": f, "v_mp_ms": v_mp, "v_avg_ms": v_avg, "v_rms_ms": v_rms,
        "T_K": T_K, "m_kg": m_kg,
    }


# ── 2. Atomic structure: Bohr model ──────────────────────────────────

def bohr_hydrogen(n, m_nucleus_kg=None):
    """Bohr model of the hydrogen atom: energy levels and orbital radii.

    E_n = -E_rydberg / n^2    (in eV)
    r_n = n^2 * a_0           (in meters)

    Feynman Vol III Ch.2: Bohr's quantum condition is L = n*hbar.
    This is the quantization of ANGULAR MOMENTUM, which leads to
    DISCRETE ENERGY LEVELS and SPECTRAL LINES.

    The Bohr model fails for multi-electron atoms (no electron-electron
    interaction) but gives exact hydrogen energies.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    E_n_eV = -E_rydberg / n**2
    r_n_m = n**2 * a_0
    return {
        "n": n, "E_n_eV": E_n_eV, "r_n_m": r_n_m,
        "r_n_angstrom": r_n_m * 1e10,
        "E_n_J": E_n_eV * e_charge,
    }


def hydrogen_spectral_line(n_upper, n_lower):
    """Photon wavelength for hydrogen transition n_upper -> n_lower.

    Rydberg formula: 1/lambda = R_inf * (1/n_lower^2 - 1/n_upper^2)
    where R_inf = E_rydberg / (hc) = 1.097e7 m^-1.

    Series:
      Lyman:   n_lower=1, UV (91-121 nm)
      Balmer:  n_lower=2, visible/UV (365-656 nm) -- H-alpha at 656 nm (red)
      Paschen: n_lower=3, near-IR (820-1875 nm) -- relevant to photonics!
      Brackett: n_lower=4, mid-IR
    """
    if n_upper <= n_lower or n_lower < 1:
        raise ValueError("n_upper > n_lower >= 1 required")
    E_upper = bohr_hydrogen(n_upper)["E_n_eV"]
    E_lower = bohr_hydrogen(n_lower)["E_n_eV"]
    delta_E_eV = E_upper - E_lower   # negative: photon emitted
    photon_energy_eV = abs(delta_E_eV)
    lambda_nm = 1239.84 / photon_energy_eV

    series = {1: "Lyman (UV)", 2: "Balmer (visible/near-UV)",
              3: "Paschen (near-IR)", 4: "Brackett (mid-IR)"}.get(n_lower, f"n={n_lower}")

    return {
        "n_upper": n_upper, "n_lower": n_lower,
        "photon_energy_eV": photon_energy_eV,
        "lambda_nm": lambda_nm,
        "series": series,
        "visible": 380 < lambda_nm < 700,
        "E_upper_eV": E_upper, "E_lower_eV": E_lower,
    }


def hydrogen_series(n_lower, n_max=10):
    """All spectral lines in a given hydrogen series."""
    return [hydrogen_spectral_line(n, n_lower) for n in range(n_lower+1, n_max+1)]


# ── 3. Molecular binding ──────────────────────────────────────────────

def hydrogen_molecule_binding():
    """Covalent bond in H2: quantum exchange energy.

    Feynman Vol III Ch.10: the covalent bond arises from the SYMMETRY of the
    wavefunction. The symmetric (bonding) orbital has lower energy than two
    isolated atoms because the electron density is enhanced between the nuclei.

    Key numbers for H2:
      Bond length: 0.74 Angstrom (0.74e-10 m)
      Binding energy: 4.52 eV
      Zero-point energy: 0.27 eV  (quantum zero-point oscillation of nuclei)
      Dissociation energy: 4.52 - 0.27 = 4.26 eV (measurable)

    The LCAO (Linear Combination of Atomic Orbitals) approximation:
      psi_bonding = (phi_A + phi_B) / sqrt(2 + 2S)
      psi_antibonding = (phi_A - phi_B) / sqrt(2 - 2S)
    where S = <phi_A|phi_B> is the overlap integral.
    """
    return {
        "molecule": "H2",
        "bond_length_angstrom": 0.74,
        "binding_energy_eV": 4.52,
        "zero_point_energy_eV": 0.27,
        "dissociation_energy_eV": 4.52 - 0.27,
        "bond_type": "covalent (sigma bond, s-orbital overlap)",
        "LCAO_bonding": "psi = (phi_A + phi_B) / sqrt(2 + 2*S)  (lower energy)",
        "LCAO_antibonding": "psi = (phi_A - phi_B) / sqrt(2 - 2*S)  (higher energy)",
        "feynman_reference": "Vol III, Ch.10: other two-state systems",
    }


# ── 4. Van der Waals / London dispersion forces ───────────────────────

def london_dispersion(alpha_1_A3, alpha_2_A3, I_1_eV, I_2_eV, r_A):
    """London dispersion force energy between two neutral atoms.

    U_London = -(3/2) * alpha_1 * alpha_2 * I_1 * I_2 / ((I_1+I_2) * r^6)

    This is the QUANTUM MECHANICAL origin of Van der Waals forces:
    instantaneous dipole fluctuations in atom 1 induce a correlated
    dipole in atom 2 (vacuum fluctuation coupling).

    The r^-6 dependence creates the attractive well in the Lennard-Jones potential.
    The same POLARIZABILITY alpha that governs London forces governs the
    optical susceptibility chi (linear response to EM fields).
    alpha_optical(omega) reduces to static alpha as omega -> 0.

    Parameters: polarizabilities in Angstrom^3, ionization energies in eV,
                interatomic distance r in Angstrom.
    """
    if r_A <= 0 or alpha_1_A3 <= 0 or alpha_2_A3 <= 0:
        raise ValueError("r, alpha_1, alpha_2 must be positive")
    # convert to SI
    alpha_1 = alpha_1_A3 * 1e-30   # A^3 -> m^3
    alpha_2 = alpha_2_A3 * 1e-30
    I_1 = I_1_eV * e_charge
    I_2 = I_2_eV * e_charge
    r = r_A * 1e-10   # A -> m
    eps_0 = 8.854e-12

    U = -(3/2) * alpha_1 * alpha_2 * I_1 * I_2 / ((I_1 + I_2) * (4*np.pi*eps_0)**2 * r**6)
    U_eV = U / e_charge
    return {
        "U_eV": U_eV, "U_J": U,
        "r_A": r_A, "alpha_1_A3": alpha_1_A3, "alpha_2_A3": alpha_2_A3,
        "force_type": "London dispersion (r^-6)",
        "note": "Same polarizability drives optical chi(omega) at optical frequencies",
    }


# ── 5. Planck blackbody radiation ────────────────────────────────────

def planck_spectrum(T_K, lambda_nm):
    """Planck blackbody spectral radiance B(lambda, T).

    B = (2*h*c^2 / lambda^5) / (exp(hc/lambda*kT) - 1)

    Feynman Vol I Ch.41: Planck's derivation of this formula in 1900 was
    the beginning of quantum mechanics. He quantized the energy of the
    oscillators in the cavity wall: E_n = n*h*nu.

    Returns spectral radiance in W/(m^2 * sr * m) and W/(m^2 * sr * nm).
    """
    if T_K <= 0:
        raise ValueError("T must be positive")
    lam = np.asarray(lambda_nm, dtype=float) * 1e-9   # nm -> m
    x = h_planck * c_light / (lam * k_B * T_K)
    B = (2 * h_planck * c_light**2 / lam**5) / (np.exp(x) - 1)
    # Wien displacement law
    lambda_peak_nm = 2.898e6 / T_K   # nm
    return {
        "B_W_per_m2_sr_m": B,
        "B_W_per_m2_sr_nm": B * 1e-9,
        "lambda_nm": lambda_nm,
        "T_K": T_K,
        "lambda_peak_nm": lambda_peak_nm,
        "x_parameter": x,
    }


# ── 6. Selection rules (what photons can be emitted/absorbed) ─────────

SELECTION_RULES = {
    "electric_dipole": {
        "Delta_l": "+-1 (orbital angular momentum)",
        "Delta_m_l": "0, +-1 (magnetic quantum number)",
        "Delta_s": "0 (spin unchanged in E1)",
        "Delta_J": "0, +-1 but not 0->0",
        "rule": "These transitions are ALLOWED (strong, fast ~ns lifetime)",
        "example": "H: 2p -> 1s (Lyman-alpha, 121.6 nm, ALLOWED)",
    },
    "forbidden": {
        "Violates": "delta_l = 0 or +-2",
        "rule": "Electric quadrupole or magnetic dipole; slow (~ms to s lifetime)",
        "example": "H: 2s -> 1s is FORBIDDEN (2s is metastable)",
    },
    "laser_relevance": {
        "Population_inversion": "Need a FORBIDDEN lower transition to build up population",
        "HeNe_laser": "Ne 3s metastable level (slow decay) enables 633 nm lasing",
        "GS_receiver": "The photon detector doesn't care about selection rules -- it's classical",
    },
}

def selection_rule_check(l_upper, l_lower, Delta_m=0):
    """Check if an electric dipole transition is allowed."""
    dl = abs(l_upper - l_lower)
    dm_ok = abs(Delta_m) <= 1
    allowed = (dl == 1) and dm_ok
    return {
        "l_upper": l_upper, "l_lower": l_lower,
        "Delta_l": dl, "Delta_m": Delta_m,
        "E1_allowed": allowed,
        "reason": "Delta_l must be +-1 and |Delta_m| <= 1 for electric dipole",
    }


# ── SymPy: key equations ──────────────────────────────────────────────

def feynman_atomic_sympy_5():
    """Five atomic/molecular physics equations in SymPy."""
    n_s, R_inf = sp.symbols('n R_inf', positive=True)
    h_s, c_s, lam_s = sp.symbols('h c lambda', positive=True)
    k_s, T_s = sp.symbols('k_B T', positive=True)
    m_s, v_s = sp.symbols('m v', positive=True)
    E_R = sp.Symbol('E_Rydberg', positive=True)

    return {
        "Bohr_energy":
            sp.Eq(sp.Symbol('E_n'), -E_R / n_s**2),
        "Rydberg_formula":
            sp.Eq(1 / lam_s,
                  R_inf * (1 / sp.Symbol('n_1')**2 - 1 / sp.Symbol('n_2')**2)),
        "Planck_distribution":
            sp.Eq(sp.Symbol('B'),
                  2*h_s*c_s**2 / lam_s**5 / (sp.exp(h_s*c_s/(lam_s*k_s*T_s)) - 1)),
        "Maxwell_Boltzmann_factor":
            sp.Eq(sp.Symbol('f(v)'),
                  4*sp.pi * (m_s/(2*sp.pi*k_s*T_s))**sp.Rational(3,2)
                  * v_s**2 * sp.exp(-m_s*v_s**2/(2*k_s*T_s))),
        "LCAO_bonding_orbital":
            sp.Eq(sp.Symbol('psi_bond'),
                  (sp.Symbol('phi_A') + sp.Symbol('phi_B'))
                  / sp.sqrt(2 + 2*sp.Symbol('S'))),
    }


if __name__ == "__main__":
    print("=== Bohr hydrogen: first 5 levels ===")
    for n in range(1, 6):
        r = bohr_hydrogen(n)
        print(f"  n={n}: E={r['E_n_eV']:+.3f} eV,  r={r['r_n_angstrom']:.2f} A")

    print("\n=== Hydrogen spectral lines (Balmer series, n->2) ===")
    for line in hydrogen_series(n_lower=2, n_max=7):
        vis = "VISIBLE" if line["visible"] else "UV"
        print(f"  {line['n_upper']}->2: lambda={line['lambda_nm']:.1f} nm  "
              f"E={line['photon_energy_eV']:.2f} eV  ({vis})")

    print("\n=== Maxwell-Boltzmann (H atom, 300 K): characteristic speeds ===")
    m_H = 1.67e-27   # kg proton mass (approx H)
    mb = maxwell_boltzmann_speed(300, m_H, 1000)  # f(1000 m/s)
    print(f"  v_mp  = {mb['v_mp_ms']:.0f} m/s")
    print(f"  v_avg = {mb['v_avg_ms']:.0f} m/s")
    print(f"  v_rms = {mb['v_rms_ms']:.0f} m/s")

    print("\n=== H2 molecular binding ===")
    hm = hydrogen_molecule_binding()
    print(f"  Bond length: {hm['bond_length_angstrom']} A")
    print(f"  Binding energy: {hm['binding_energy_eV']} eV")
    print(f"  Zero-point E: {hm['zero_point_energy_eV']} eV")

    print("\n=== Planck blackbody: sun T=5778K at 550 nm ===")
    pb = planck_spectrum(5778, 550)
    print(f"  Peak wavelength: {pb['lambda_peak_nm']:.0f} nm  (visible peak ~500 nm)")
    print(f"  B(550nm) = {pb['B_W_per_m2_sr_nm']:.2e} W/(m^2 sr nm)")

    print("\n=== Selection rules ===")
    sc = selection_rule_check(1, 0, 0)   # p -> s: allowed
    print(f"  2p->1s: allowed={sc['E1_allowed']}  (Delta_l={sc['Delta_l']})")
    sc2 = selection_rule_check(0, 0, 0)  # s -> s: forbidden
    print(f"  2s->1s: allowed={sc2['E1_allowed']}  (Delta_l={sc2['Delta_l']})")

    print("\n=== SymPy 5 ===")
    for k, eq in feynman_atomic_sympy_5().items():
        print(f"  {k}: {eq}")
