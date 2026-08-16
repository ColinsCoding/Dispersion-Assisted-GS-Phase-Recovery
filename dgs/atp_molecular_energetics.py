"""ATP: the same math this repo already built for photosynthesis, applied
to the molecule that actually spends the energy photosynthesis collects.

ATP hydrolysis (ATP + H2O -> ADP + Pi) releases free energy
    dG = dG_standard + R*T*ln([ADP][Pi]/[ATP]),
and because a living cell keeps [ATP] high and [ADP],[Pi] low (the
products are used up almost as fast as they're made), the CELLULAR dG is
substantially more negative than the textbook standard-state value
(dG_standard ~ -30.5 kJ/mol) -- verified numerically below with realistic
intracellular concentrations, not just quoted.

ATP SYNTHASE AS A CYCLIC RATE-MATRIX SYSTEM. Boyer's binding-change
mechanism describes each catalytic site of ATP synthase cycling through
three conformations: Open (empty) -> Loose (ADP+Pi bound) -> Tight
(catalysis, ATP formed) -> back to Open (ATP released). That is a 3-state
unidirectional cycle -- EXACTLY the "population transfer rate matrix"
structure dgs.photosynthesis_energy_transfer already built for FRET energy
hopping between chromophores, just relabeled: reuses
build_rate_matrix/eigen_decomposition_modes/solve_population_dynamics
directly rather than re-deriving the linear-algebra machinery.

A one-way cycle (no reverse rates) reaches a genuine NON-EQUILIBRIUM
STEADY STATE: populations stop changing (dp/dt=0) but probability current
keeps circulating around the cycle -- the physically correct description
of a molecular motor. That circulating flux IS the ATP production rate,
and it must come out the SAME whichever of the three transitions you
measure it at (checked below, not assumed) -- exactly the way current is
the same everywhere around a series circuit loop.

Multiplying that turnover flux by the (cellular) free energy per ATP gives
the actual metabolic power output of a single ATP synthase molecule --
tiny in watts, but a real, dimensionally checked number. py-3.13, NumPy
only (no torch needed for the parts reused here).
"""

from __future__ import annotations
import numpy as np

from dgs.photosynthesis_energy_transfer import build_rate_matrix, eigen_decomposition_modes, solve_population_dynamics

R_GAS = 8.314462618      # J/(mol*K)
N_AVOGADRO = 6.02214076e23
ATP_DG_STANDARD_J_PER_MOL = -30500.0   # textbook standard-state value, ~-30.5 kJ/mol


def atp_hydrolysis_free_energy(ATP_M: float, ADP_M: float, Pi_M: float,
                                T_K: float = 310.15, dG0_J_per_mol: float = ATP_DG_STANDARD_J_PER_MOL) -> dict:
    """dG = dG_standard + R*T*ln([ADP][Pi]/[ATP]) -- the ACTUAL free energy
    release under given (e.g. real cellular) concentrations, not the
    standard-state textbook number alone."""
    if ATP_M <= 0 or ADP_M <= 0 or Pi_M <= 0 or T_K <= 0:
        raise ValueError("concentrations and temperature must be positive")
    dG = dG0_J_per_mol + R_GAS * T_K * np.log((ADP_M * Pi_M) / ATP_M)
    return {"dG_J_per_mol": dG, "dG_kJ_per_mol": dG / 1000.0,
            "dG0_kJ_per_mol": dG0_J_per_mol / 1000.0, "T_K": T_K}


def atp_synthase_rate_matrix(k_OL: float, k_LT: float, k_TO: float) -> np.ndarray:
    """The 3-state Open->Loose->Tight->Open catalytic cycle as a rate
    matrix, built with the SAME dgs.photosynthesis_energy_transfer.build_rate_matrix
    used for FRET hopping -- a one-way cycle is just three transfer_rates
    entries and zero intrinsic decay (nothing leaves the cycle)."""
    if k_OL <= 0 or k_LT <= 0 or k_TO <= 0:
        raise ValueError("all three rate constants must be positive")
    transfer_rates = {(0, 1): k_OL, (1, 2): k_LT, (2, 0): k_TO}   # O=0, L=1, T=2
    decay_rates = [0.0, 0.0, 0.0]                                  # closed cycle: no population loss
    return build_rate_matrix(transfer_rates, decay_rates)


def steady_state_distribution(K: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """The stationary population distribution p_ss (Kp_ss=0, sum(p_ss)=1),
    found via dgs.photosynthesis_energy_transfer.eigen_decomposition_modes
    -- its slowest-decaying (closest-to-zero eigenvalue) mode IS the steady
    state for a closed, population-conserving cycle."""
    eigvals, eigvecs = eigen_decomposition_modes(K)
    if abs(eigvals[0].real) > tol or abs(eigvals[0].imag) > tol:
        raise AssertionError(f"expected a zero eigenvalue (steady state) first, got {eigvals[0]}")
    p_ss = eigvecs[:, 0].real
    p_ss = p_ss / p_ss.sum()
    if np.any(p_ss < -tol):
        raise AssertionError(f"steady-state populations must be non-negative, got {p_ss}")
    return np.clip(p_ss, 0.0, None)


def turnover_flux(p_ss: np.ndarray, k_OL: float, k_LT: float, k_TO: float, tol: float = 1e-6) -> dict:
    """The circulating probability current around the cycle -- the ATP
    production rate (cycles/s, one ATP released per full cycle). CHECKED
    to be the SAME measured at all three transitions (a single unbranched
    cycle has one flux, everywhere), not just computed one way and
    assumed consistent, the way current is identical everywhere around a
    single-loop circuit."""
    J_OL = p_ss[0] * k_OL
    J_LT = p_ss[1] * k_LT
    J_TO = p_ss[2] * k_TO
    spread = max(J_OL, J_LT, J_TO) - min(J_OL, J_LT, J_TO)
    if spread / max(J_OL, J_LT, J_TO) > tol:
        raise AssertionError(f"flux should be identical at all 3 transitions in steady state, "
                              f"got J_OL={J_OL:.6e}, J_LT={J_LT:.6e}, J_TO={J_TO:.6e}")
    return {"J_OL": J_OL, "J_LT": J_LT, "J_TO": J_TO, "flux_per_s": J_TO}


def atp_production_power(flux_per_s: float, dG_J_per_mol: float) -> float:
    """Metabolic power output of ONE ATP synthase molecule: (ATP/s) *
    (energy released per ATP, per MOLECULE = dG_per_mol / N_avogadro).
    dG_J_per_mol is negative (energy released), so this returns a
    negative power (energy leaving the ATP pool, i.e. delivered to work)."""
    if flux_per_s < 0:
        raise ValueError("flux_per_s must be >= 0")
    return flux_per_s * (dG_J_per_mol / N_AVOGADRO)


if __name__ == "__main__":
    print("=== ATP hydrolysis free energy: standard state vs. real cell ===")
    standard = atp_hydrolysis_free_energy(ATP_M=1.0, ADP_M=1.0, Pi_M=1.0)   # 1 M each = standard state
    print(f"  standard state (1 M each): dG = {standard['dG_kJ_per_mol']:.1f} kJ/mol")
    cellular = atp_hydrolysis_free_energy(ATP_M=3e-3, ADP_M=1e-4, Pi_M=3e-3)   # realistic cell concentrations
    print(f"  typical cell ([ATP]=3mM, [ADP]=0.1mM, [Pi]=3mM): dG = {cellular['dG_kJ_per_mol']:.1f} kJ/mol")
    print(f"  cellular dG is MORE negative than standard: {cellular['dG_kJ_per_mol'] < standard['dG_kJ_per_mol']}")

    print("\n=== ATP synthase as a 3-state cyclic rate matrix (Boyer binding-change mechanism) ===")
    k_OL, k_LT, k_TO = 300.0, 500.0, 150.0   # 1/s, chosen to land near the ~100/s literature turnover rate
    K = atp_synthase_rate_matrix(k_OL, k_LT, k_TO)
    print("  K ="); print(K)
    p_ss = steady_state_distribution(K)
    print(f"  steady-state populations [O, L, T] = {np.round(p_ss, 4)}")

    flux = turnover_flux(p_ss, k_OL, k_LT, k_TO)
    print(f"  flux identical at all 3 transitions: J_OL={flux['J_OL']:.2f}, "
          f"J_LT={flux['J_LT']:.2f}, J_TO={flux['J_TO']:.2f} /s")
    print(f"  ATP turnover rate: {flux['flux_per_s']:.1f} ATP/s per synthase "
          f"(literature ballpark: ~100/s per catalytic site)")

    print("\n=== metabolic power of a single ATP synthase molecule ===")
    P = atp_production_power(flux["flux_per_s"], cellular["dG_J_per_mol"])
    print(f"  P = {P:.3e} W per molecule (delivered as mechanical/chemical work)")

    print("\n=== a sanity-check trajectory: relaxation toward the steady state ===")
    p0 = np.array([1.0, 0.0, 0.0])   # start fully in the Open state
    t = np.linspace(0, 0.05, 6)
    p_t = solve_population_dynamics(K, p0, t)
    for ti, row in zip(t, p_t):
        print(f"  t={ti*1000:5.1f} ms: [O,L,T] = {np.round(row, 4)}")
    print(f"  converges to steady state {np.round(p_ss, 4)}: "
          f"{np.allclose(p_t[-1], p_ss, atol=0.05)}")
