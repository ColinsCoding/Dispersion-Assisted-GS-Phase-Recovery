"""Physical chemistry: Arrhenius kinetics, Gibbs thermodynamics, Michaelis-Menten,
Clausius-Clapeyron, Boltzmann distribution, phase equilibria.

Connection to physical AI / TS-DFT: Beer-Lambert (thc_spectroscopy.py) is the
measurement layer; this module supplies the underlying chemical physics.
"""
import numpy as np
import sympy as sp

R_GAS = 8.314462   # J/(mol K)
kB    = 1.380649e-23  # J/K
NA    = 6.02214076e23


# ── Kinetics ──────────────────────────────────────────────────────────────────

def arrhenius(A, Ea_kJ_mol, T_K):
    """Rate constant k = A * exp(-Ea/RT).

    Parameters
    ----------
    A           : pre-exponential factor (same units as k)
    Ea_kJ_mol   : activation energy (kJ/mol)
    T_K         : temperature (K)
    """
    T = np.asarray(T_K, float)
    Ea = Ea_kJ_mol * 1e3   # J/mol
    k  = A * np.exp(-Ea / (R_GAS * T))
    Ea_over_kT = Ea / (R_GAS * T)
    return {
        "k": k,
        "Ea_kJ_mol": Ea_kJ_mol,
        "A": A,
        "T_K": T,
        "Ea_over_RT": Ea_over_kT,
        "ln_k": np.log(k),
    }


def arrhenius_two_temp(A, Ea_kJ_mol, T1, T2):
    """Rate ratio k2/k1 from Arrhenius at two temperatures."""
    k1 = arrhenius(A, Ea_kJ_mol, T1)["k"]
    k2 = arrhenius(A, Ea_kJ_mol, T2)["k"]
    return {
        "k1": k1, "k2": k2, "T1": T1, "T2": T2,
        "ratio_k2_k1": k2 / k1,
        "Ea_kJ_mol": Ea_kJ_mol,
    }


def first_order_kinetics(k, C0, t_arr):
    """[A](t) = C0 * exp(-k*t); half-life t_half = ln(2)/k."""
    t   = np.asarray(t_arr, float)
    C   = C0 * np.exp(-k * t)
    t_h = np.log(2) / k
    return {
        "C": C, "t": t, "k": k, "C0": C0,
        "t_half": t_h,
        "tau": 1.0 / k,         # time constant
    }


# ── Thermodynamics ────────────────────────────────────────────────────────────

def gibbs_free_energy(delta_H_kJ, delta_S_J_K, T_K):
    """dG = dH - T*dS; spontaneous if dG < 0; K_eq = exp(-dG/RT)."""
    T    = np.asarray(T_K, float)
    dH   = delta_H_kJ * 1e3      # J/mol
    dG   = dH - T * delta_S_J_K
    K_eq = np.exp(-dG / (R_GAS * T))
    return {
        "dG_J_mol": dG,
        "dG_kJ_mol": dG / 1e3,
        "dH_kJ_mol": delta_H_kJ,
        "dS_J_K":    delta_S_J_K,
        "T_K": T,
        "K_eq": K_eq,
        "spontaneous": dG < 0,
        "T_crossover_K": dH / delta_S_J_K if delta_S_J_K != 0 else np.inf,
    }


def vant_hoff(K1, T1_K, T2_K, delta_H_kJ_mol):
    """ln(K2/K1) = -dH/R * (1/T2 - 1/T1)  (van't Hoff equation)."""
    dH   = delta_H_kJ_mol * 1e3
    lnK2 = np.log(K1) - dH / R_GAS * (1.0/T2_K - 1.0/T1_K)
    K2   = np.exp(lnK2)
    return {
        "K1": K1, "K2": K2,
        "T1_K": T1_K, "T2_K": T2_K,
        "ratio": K2 / K1,
        "delta_H_kJ_mol": delta_H_kJ_mol,
    }


def clausius_clapeyron(T1_K, P1_Pa, T2_K, delta_H_vap_kJ_mol):
    """ln(P2/P1) = -dH_vap/R * (1/T2 - 1/T1).

    Water: dH_vap = 40.7 kJ/mol; at T1=373K, P1=101325 Pa.
    """
    dH  = delta_H_vap_kJ_mol * 1e3
    lnP = np.log(P1_Pa) - dH / R_GAS * (1.0/T2_K - 1.0/T1_K)
    P2  = np.exp(lnP)
    return {
        "P2_Pa": P2,
        "P2_atm": P2 / 101325,
        "P1_Pa": P1_Pa,
        "T2_K": T2_K,
        "ln_ratio": lnP - np.log(P1_Pa),
        "delta_H_vap_kJ_mol": delta_H_vap_kJ_mol,
    }


# ── Statistical thermodynamics ────────────────────────────────────────────────

def boltzmann_distribution(E_levels_J, T_K, g=None):
    """Canonical ensemble populations p_i = g_i*exp(-E_i/kT) / Z.

    Parameters
    ----------
    E_levels_J : energy levels (J) — array, at least 2 elements
    T_K        : temperature (K)
    g          : degeneracy array (default: all 1)
    """
    E = np.asarray(E_levels_J, float)
    if g is None:
        g = np.ones_like(E)
    else:
        g = np.asarray(g, float)
    kT = kB * T_K
    # Shift energies to avoid overflow
    E_shift   = E - E.min()
    weights   = g * np.exp(-E_shift / kT)
    Z         = weights.sum()
    pop       = weights / Z
    U_avg     = float(np.dot(pop, E))
    S_entropy = -kB * float(np.sum(pop * np.log(pop + 1e-300)))
    return {
        "populations": pop,
        "Z_partition": Z,
        "U_avg_J": U_avg,
        "S_J_K": S_entropy,
        "kT_J": kT,
        "T_K": T_K,
        "E_levels_J": E,
    }


# ── Enzyme kinetics ───────────────────────────────────────────────────────────

def michaelis_menten(Vmax, Km, S_arr):
    """v = Vmax * [S] / (Km + [S]).

    Returns reaction rate, efficiency (v/Vmax), and catalytic regime info.
    """
    S  = np.asarray(S_arr, float)
    v  = Vmax * S / (Km + S)
    eff = v / Vmax
    return {
        "v": v,
        "S": S,
        "efficiency": eff,
        "Vmax": Vmax,
        "Km": Km,
        "S_half": Km,          # [S] where v = Vmax/2
        "kcat_over_Km": Vmax / Km,
    }


# ── Phase diagram (water, simplified) ────────────────────────────────────────

def water_phase(T_C, P_atm):
    """Rough water phase: solid / liquid / gas / supercritical."""
    T = T_C + 273.15
    P = P_atm * 101325
    Tc = 647.1   # K
    Pc = 220.6e5  # Pa
    if T > Tc and P > Pc:
        return {"phase": "supercritical", "T_K": T, "P_Pa": P}
    # Clausius-Clapeyron boiling point at P
    bp = clausius_clapeyron(373.15, 101325, T, 40.7)
    bp_T = bp["T2_K"]  # boiling T at P... reversed logic below
    # Simple boundaries
    if T_C < 0.0 and P_atm >= 0.006:
        phase = "solid (ice)"
    elif T_C >= 0.0 and T_C < 100.0 and P_atm >= 0.03:
        phase = "liquid (water)"
    else:
        phase = "gas (steam)"
    return {"phase": phase, "T_C": T_C, "P_atm": P_atm,
            "T_K": T, "P_Pa": P}


# ── SymPy ────────────────────────────────────────────────────────────────────

def physical_chemistry_sympy_5():
    """Five symbolic physical chemistry equations."""
    A_s, Ea, R, T, k_s = sp.symbols("A Ea R T k", positive=True)
    dH, dS, dG, K       = sp.symbols("Delta_H Delta_S Delta_G K")
    Vm, Km_s, S_s       = sp.symbols("V_max K_m S", positive=True)
    dHv, T1, T2, P1, P2 = sp.symbols("Delta_H_vap T1 T2 P1 P2", positive=True)
    kB_s, Ei, Z_s       = sp.symbols("k_B E_i Z", positive=True)
    return {
        "Arrhenius":          sp.Eq(k_s, A_s * sp.exp(-Ea / (R * T))),
        "Gibbs_free_energy":  sp.Eq(dG, dH - T * dS),
        "Michaelis_Menten":   sp.Eq(sp.Symbol("v"),
                                    Vm * S_s / (Km_s + S_s)),
        "Clausius_Clapeyron": sp.Eq(sp.log(P2 / P1),
                                    -dHv / R * (1/T2 - 1/T1)),
        "Boltzmann_pop":      sp.Eq(sp.Symbol("p_i"),
                                    sp.exp(-Ei / (kB_s * T)) / Z_s),
    }


if __name__ == "__main__":
    print("=== Arrhenius: enzyme at 25C and 37C ===")
    a1 = arrhenius(1e13, 50.0, 298.15)
    a2 = arrhenius(1e13, 50.0, 310.15)
    print(f"  k(25C) = {a1['k']:.3e}  k(37C) = {a2['k']:.3e}  "
          f"ratio = {a2['k']/a1['k']:.2f}x")

    print("\n=== Gibbs: exothermic spontaneous reaction ===")
    g = gibbs_free_energy(-100.0, 50.0, 298.15)
    print(f"  dG = {g['dG_kJ_mol']:.2f} kJ/mol  spontaneous: {g['spontaneous']}")
    print(f"  K_eq = {g['K_eq']:.3e}")

    print("\n=== Clausius-Clapeyron: boiling point at 0.5 atm ===")
    cc = clausius_clapeyron(373.15, 101325, 355.0, 40.7)
    print(f"  P(355 K) = {cc['P2_atm']:.4f} atm  (expect ~0.56)")

    print("\n=== Boltzmann: 3 levels at 300 K ===")
    E = np.array([0, 0.025, 0.10]) * 1.6e-19   # 0, 25 meV, 100 meV
    b = boltzmann_distribution(E, 300)
    print(f"  populations: {b['populations'].round(4)}")
    print(f"  partition Z = {b['Z_partition']:.4f}")

    print("\n=== Michaelis-Menten ===")
    mm = michaelis_menten(Vmax=10.0, Km=2.0, S_arr=[0.5, 2.0, 10.0, 50.0])
    for s, v, e in zip(mm["S"], mm["v"], mm["efficiency"]):
        print(f"  [S]={s:.1f}  v={v:.3f}  eff={e:.3f}")

    print("\n=== SymPy 5 ===")
    for k, eq in physical_chemistry_sympy_5().items():
        print(f"  {k}: {eq}")
