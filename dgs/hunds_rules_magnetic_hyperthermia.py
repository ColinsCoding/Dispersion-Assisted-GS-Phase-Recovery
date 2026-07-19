"""Hund's rules -- the real "orientation rules" for how electrons fill
degenerate orbitals -- determine an ion's ground-state magnetic moment.
Applied here to Fe3+ and Fe2+, the real active ions in iron-oxide
(magnetite/maghemite) nanoparticles used in magnetic fluid hyperthermia:
a real, clinically-used cancer treatment (e.g. MagForce's NanoTherm
therapy, approved in the EU for glioblastoma) where nanoparticles
delivered via the bloodstream (or injected into a solid tumor) are
heated by an external oscillating magnetic field, killing cancer cells
by localized heat -- sometimes combined with laser/photothermal therapy
in current research, tying together magnetic moment quantum mechanics,
solid-state nanoparticle physics, and a real oncology application.

HUND'S RULES (the 3 orientation rules):
  1. Maximize total spin S -- fill degenerate orbitals with parallel
     (unpaired) spins before pairing any electron (lowest Coulomb
     repulsion between parallel-spin electrons in different orbitals).
  2. Maximize total orbital angular momentum L consistent with rule 1 --
     fill from the highest |m_l| down.
  3. J = |L-S| if the subshell is at most half full, J = L+S if more
     than half full (spin-orbit coupling favors anti-alignment below
     half-filling, alignment above).

Reuses dgs.total_angular_momentum_coupling's term_symbol_string and
lande_g_factor rather than reimplementing them.
"""

import numpy as np

from dgs.total_angular_momentum_coupling import term_symbol_string

MU_B = 9.274e-24        # J/T, Bohr magneton
MU_0 = 4 * np.pi * 1e-7  # T*m/A, vacuum permeability
K_BOLTZMANN = 1.380649e-23  # J/K


def hunds_rules_ground_state(l, n_electrons):
    """Fill a degenerate (2l+1)-orbital subshell per Hund's 3 rules;
    return (S, L, J, term_symbol)."""
    if l < 0:
        raise ValueError("l must be non-negative")
    n_orbitals = 2 * l + 1
    if n_electrons <= 0 or n_electrons > 2 * n_orbitals:
        raise ValueError(f"n_electrons must be in [1, {2 * n_orbitals}] for l={l}")

    m_l_values = list(range(l, -l - 1, -1))   # +l, +l-1, ..., -l (fill order for max L)
    filled_singly = min(n_electrons, n_orbitals)
    remaining = n_electrons - filled_singly    # electrons that must pair up
    occupation = [1] * filled_singly + [0] * (n_orbitals - filled_singly)
    for i in range(remaining):                 # pair starting from HIGHEST m_l
        occupation[i] += 1

    unpaired_count = filled_singly - remaining
    S = unpaired_count / 2.0
    L = sum(m_l * occ for m_l, occ in zip(m_l_values, occupation))
    L = abs(L)   # L is a magnitude; sign only reflects filling-order bookkeeping
    J = abs(L - S) if n_electrons <= n_orbitals else L + S
    term = term_symbol_string(L, S, J)
    return S, L, J, term


def lande_g_factor(L, S, J):
    """g_J = 1 + [J(J+1)+S(S+1)-L(L+1)] / (2*J*(J+1)) -- same formula as
    dgs.torch.stern_gerlach_zeeman_hydrogen.lande_g_factor, reimplemented
    here in plain numpy (not torch) so this module doesn't need the
    py-3.12/torch dependency for a one-line formula."""
    if J <= 0:
        raise ValueError("J must be positive")
    if L < 0 or S < 0:
        raise ValueError("L and S must be non-negative")
    return 1.0 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))


def effective_magnetic_moment_bohr_magnetons(L, S, J):
    """mu_eff = g_J * sqrt(J(J+1)) Bohr magnetons -- the standard
    formula for an ion's effective paramagnetic moment."""
    g_J = lande_g_factor(L, S, J)
    return g_J * np.sqrt(J * (J + 1))


def neel_relaxation_time_s(K_anisotropy_j_per_m3, V_m3, T_kelvin, tau0_s=1e-9):
    """Neel-Arrhenius relaxation time: tau_N = tau0 * exp(K*V/(kB*T)) --
    how long a nanoparticle's internal magnetic moment takes to flip
    against its anisotropy energy barrier at temperature T. tau0 ~ 1e-9 s
    is the standard 'attempt time' order of magnitude for magnetic
    nanoparticles."""
    if K_anisotropy_j_per_m3 <= 0:
        raise ValueError("K_anisotropy_j_per_m3 must be positive")
    if V_m3 <= 0:
        raise ValueError("V_m3 must be positive")
    if T_kelvin <= 0:
        raise ValueError("T_kelvin must be positive")
    if tau0_s <= 0:
        raise ValueError("tau0_s must be positive")
    exponent = K_anisotropy_j_per_m3 * V_m3 / (K_BOLTZMANN * T_kelvin)
    return tau0_s * np.exp(exponent)


def ac_susceptibility_imaginary_part(chi0, omega_rad_s, tau_s):
    """Debye linear-response formula: chi'' = chi0 * omega*tau / (1 + (omega*tau)^2).
    Peaks exactly at omega*tau = 1 -- the resonance condition that
    determines the optimal AC field frequency for a given particle size."""
    if chi0 <= 0:
        raise ValueError("chi0 must be positive")
    if omega_rad_s <= 0:
        raise ValueError("omega_rad_s must be positive")
    if tau_s <= 0:
        raise ValueError("tau_s must be positive")
    wt = omega_rad_s * tau_s
    return chi0 * wt / (1 + wt**2)


def specific_absorption_rate_w_per_kg(chi_double_prime, H0_a_per_m, f_hz, rho_kg_per_m3):
    """SAR = pi * mu0 * chi'' * H0^2 * f / rho -- the real figure of
    merit in magnetic hyperthermia: heating power delivered per unit
    mass of nanoparticles under an oscillating field of amplitude H0 and
    frequency f. Clinically relevant SAR is typically tens to hundreds
    of W/kg."""
    if chi_double_prime <= 0:
        raise ValueError("chi_double_prime must be positive")
    if H0_a_per_m <= 0:
        raise ValueError("H0_a_per_m must be positive")
    if f_hz <= 0:
        raise ValueError("f_hz must be positive")
    if rho_kg_per_m3 <= 0:
        raise ValueError("rho_kg_per_m3 must be positive")
    return np.pi * MU_0 * chi_double_prime * H0_a_per_m**2 * f_hz / rho_kg_per_m3


if __name__ == "__main__":
    print("=== Hund's rules: orientation rules for magnetic moment ===\n")
    for label, l, n in [("Fe3+ (d5, high-spin)", 2, 5), ("Fe2+ (d6, high-spin)", 2, 6)]:
        S, L, J, term = hunds_rules_ground_state(l, n)
        mu_eff = effective_magnetic_moment_bohr_magnetons(L, S, J)
        print(f"  {label}: S={S}, L={L}, J={J}  ->  ground term {term}, "
              f"mu_eff = {mu_eff:.2f} mu_B")
    print("  (real known ground terms: Fe3+ -> 6S5/2, Fe2+ -> 5D4 -- match)\n")

    print("=== Magnetic hyperthermia: real iron-oxide nanoparticle cancer treatment ===\n")
    print("Iron-oxide (Fe3O4/maghemite) nanoparticles, carrying the Fe2+/Fe3+ magnetic")
    print("moments above, are delivered via the bloodstream or injected into a solid")
    print("tumor, then heated by an external oscillating magnetic field -- a real")
    print("clinical technique (e.g. MagForce NanoTherm, EU-approved for glioblastoma).\n")

    # representative literature-typical parameters for magnetite nanoparticles
    K_aniso = 2.0e4    # J/m^3, representative magnetite anisotropy constant
    diameter_nm = 15.0
    V = (4 / 3) * np.pi * (diameter_nm * 1e-9 / 2) ** 3
    T_body = 310.0     # K, body temperature
    tau_N = neel_relaxation_time_s(K_aniso, V, T_body)
    print(f"{diameter_nm} nm nanoparticle: Neel relaxation time = {tau_N * 1e9:.2f} ns")

    f_clinical = 100e3   # Hz, real clinical AC field frequency (e.g. NanoTherm system)
    omega = 2 * np.pi * f_clinical
    chi0 = 2.0   # representative dimensionless initial susceptibility
    chi_pp = ac_susceptibility_imaginary_part(chi0, omega, tau_N)
    print(f"at clinical field frequency {f_clinical/1e3:.0f} kHz: "
          f"omega*tau = {omega*tau_N:.3f}, chi'' = {chi_pp:.4f}")

    H0_clinical = 10e3   # A/m, real clinical field amplitude order of magnitude
    rho_magnetite = 5000.0   # kg/m^3
    sar = specific_absorption_rate_w_per_kg(chi_pp, H0_clinical, f_clinical, rho_magnetite)
    print(f"specific absorption rate (SAR): {sar:.2f} W/kg")
    print("(real reported magnetite SAR values span roughly 1-2000+ W/kg depending")
    print(" heavily on particle size/quality and the chi0 assumed here -- this is a")
    print(" representative-parameter estimate, not a specific product's measured SAR)")
