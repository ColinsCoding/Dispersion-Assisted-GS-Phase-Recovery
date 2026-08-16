"""molecular_ohms_law.py -- Ohm's law (V=IR, a macroscopic circuit
relation) derived down to the molecular/crystal-structure level via the
Drude model, then applied across real compounds to show that
"molecular manufacturing" (which compound, which doping) is what actually
sets a circuit element's resistance.

THE CHAIN:
  macroscopic:  V = I*R,  R = rho*L/A = L/(sigma*A)   (a wire's geometry)
  microscopic:  sigma = n*e^2*tau/m                    (dgs/solid_state_physics.py's
                                                          drude_conductivity, REUSED
                                                          directly, not reimplemented)
  molecular:    n (carrier density) and tau (mean free time between
                electron-lattice collisions) are set by WHICH COMPOUND you
                manufacture and how it's doped -- not free parameters, but
                a direct consequence of the material's crystal/molecular
                structure. Silicon doped with phosphorus (n-type) has a
                carrier density orders of magnitude higher than intrinsic
                silicon; that difference alone, with everything else held
                fixed, is why a doped-silicon resistor and an intrinsic-
                silicon resistor of the IDENTICAL geometry have wildly
                different resistances (see compound_resistance_table).

COMPOUND_LIBRARY's carrier densities (n) are real, standard textbook
values for the named materials (copper: 8.47e28/m^3 is the same value
already used in dgs/solid_state_physics.py's own demo; intrinsic silicon:
~1.5e16/m^3 at room temperature is the standard textbook figure). The
mean free times (tau) and GaAs effective mass are REPRESENTATIVE
order-of-magnitude values for illustrating the Drude-model dependence,
not measured values for a specific manufactured sample -- verify against
a cited source before using a specific number in a real design.
"""

from __future__ import annotations
from typing import Dict, List, Optional

from dgs.solid_state_physics import drude_conductivity, M_E


# ── 1. Macroscopic Ohm's law: resistance from geometry + conductivity ───────

def resistance_from_conductivity(sigma: float, length_m: float, area_m2: float) -> float:
    """R = L/(sigma*A) -- a wire/bar's resistance from its conductivity and
    geometry (rho = 1/sigma, R = rho*L/A, the standard relation)."""
    if sigma <= 0:
        raise ValueError(f"sigma={sigma}: must be positive")
    if length_m <= 0:
        raise ValueError(f"length_m={length_m}: must be positive")
    if area_m2 <= 0:
        raise ValueError(f"area_m2={area_m2}: must be positive")
    return length_m / (sigma * area_m2)


def ohms_law_voltage(current_A: float, resistance_ohm: float) -> float:
    """V = I*R -- included for completeness/testability alongside
    resistance_from_conductivity, so the full macro->micro->molecular
    chain (V=IR, R=L/(sigma*A), sigma=n*e^2*tau/m) is exercised end to end
    by compound_resistance_table below."""
    if resistance_ohm < 0:
        raise ValueError(f"resistance_ohm={resistance_ohm}: must be non-negative")
    return current_A * resistance_ohm


# ── 2. Real compounds: what "molecular manufacturing" actually changes ──────

COMPOUND_LIBRARY: Dict[str, Dict] = {
    "copper (Cu, metal)": {
        "n_density_m3": 8.47e28,  # standard textbook value, matches dgs/solid_state_physics.py's demo
        "tau_s": 2.5e-14,
        "m_eff": M_E,
        "note": "one free electron per atom, no manufacturing choice involved -- the baseline metal.",
    },
    "intrinsic silicon (Si)": {
        "n_density_m3": 1.5e16,  # standard textbook room-temperature intrinsic carrier density
        "tau_s": 1e-13,
        "m_eff": M_E,
        "note": "undoped silicon -- thermally generated carriers only, 10^12x fewer than copper.",
    },
    "n-doped silicon (Si:P, representative)": {
        "n_density_m3": 1e22,  # representative moderate n-type doping level
        "tau_s": 1e-13,
        "m_eff": M_E,
        "note": "phosphorus dopant atoms manufactured into the silicon lattice donate free "
                "electrons -- n_density set by DOPING CONCENTRATION, an actual manufacturing "
                "process parameter (ion implantation dose, diffusion time/temperature).",
    },
    "GaAs (n-type, representative)": {
        "n_density_m3": 1e23,  # representative doped GaAs
        "tau_s": 2e-13,
        "m_eff": 0.067 * M_E,  # GaAs conduction-band effective mass, standard textbook value
        "note": "a different COMPOUND entirely (III-V, not elemental) -- different lattice, "
                "different effective mass, used where higher electron mobility matters (RF devices).",
    },
}


def compound_resistance_table(length_m: float, area_m2: float,
                               compounds: Optional[Dict[str, Dict]] = None) -> List[Dict]:
    """For each compound in `compounds` (defaults to COMPOUND_LIBRARY),
    compute sigma via dgs.solid_state_physics.drude_conductivity (reused,
    not reimplemented) and R via resistance_from_conductivity, for the
    SAME fixed geometry -- isolating how much of R's variation comes from
    the compound/doping choice alone."""
    compounds = compounds or COMPOUND_LIBRARY
    rows = []
    for name, p in compounds.items():
        sigma = drude_conductivity(p["n_density_m3"], p["tau_s"], p["m_eff"])
        R = resistance_from_conductivity(sigma, length_m, area_m2)
        rows.append({"compound": name, "sigma_S_per_m": sigma, "resistance_ohm": R,
                      "n_density_m3": p["n_density_m3"], "note": p["note"]})
    rows.sort(key=lambda r: r["resistance_ohm"])
    return rows


if __name__ == "__main__":
    L, A = 0.01, 1e-6  # 1 cm length, 1 mm^2 cross-section -- fixed geometry throughout
    print(f"Fixed geometry: L={L*100:.0f} cm, A={A*1e6:.1f} mm^2\n")
    print(f"{'compound':40s} {'sigma (S/m)':>14s} {'R (ohm)':>14s}")
    print("-" * 70)
    for row in compound_resistance_table(L, A):
        print(f"{row['compound']:40s} {row['sigma_S_per_m']:14.3e} {row['resistance_ohm']:14.3e}")

    print("\nSame geometry, same V=IR macroscopic law -- 12+ orders of magnitude")
    print("difference in R, entirely from WHICH COMPOUND (and how it's doped)")
    print("was manufactured into the wire. That's the molecular-manufacturing lever.")

    V = ohms_law_voltage(current_A=0.001, resistance_ohm=354.9)
    print(f"\nOhm's law check: I=1 mA through the n-doped-Si example (R~355 ohm) -> V={V:.3f} V")
