"""The four fundamental forces: how strong, how far, carried by what.

Everything that happens is one of four interactions, and they span an almost absurd
range of strengths:

    force            relative strength   range        carrier          acts on
    ---------------  -----------------   ----------   --------------   -------------
    strong                  1            ~1e-15 m     gluon            color (quarks)
    electromagnetic        ~1/137        infinite     photon           electric charge
    weak                   ~1e-6         ~1e-18 m     W, Z bosons      flavor
    gravity                ~6e-39        infinite     graviton (?)     mass-energy

The strong force binds nuclei (against the electromagnetic repulsion of the protons,
dgs.semi_empirical_mass); the weak force runs the pp-chain's first step
(dgs.proton_proton_chain); electromagnetism is light itself (dgs.laser_physics); and
gravity, by far the weakest, wins at astronomical scales only because it is always
attractive and never cancels.

Two of these strengths come straight out of fundamental constants, and their ratio is
one of the most striking numbers in physics -- the electromagnetic vs gravitational
force between TWO PROTONS (both fall off as 1/r^2, so the ratio is distance-free):
        F_EM / F_grav = k e^2 / (G m_p^2)  ~=  1.2e36.
Gravity is ~10^36 times weaker. This module computes that (and the fine-structure and
gravitational couplings behind it) from constants, and tabulates the four forces'
ranges and carriers. NumPy-free; py-3.13.
"""

import math

G_NEWTON = 6.67430e-11            # m^3 kg^-1 s^-2
HBAR = 1.054571817e-34           # J s
C_LIGHT = 2.99792458e8           # m/s
E_CHARGE = 1.602176634e-19       # C
EPS0 = 8.8541878128e-12          # F/m
M_PROTON = 1.67262192369e-27     # kg
K_COULOMB = 1.0 / (4 * math.pi * EPS0)

# reference data: relative strength, range, carrier, what it couples to
FORCES = {
    "strong": {"relative_strength": 1.0, "range_m": 1e-15,
               "carrier": "gluon", "acts_on": "color charge"},
    "electromagnetic": {"relative_strength": 1 / 137.035999, "range_m": math.inf,
                        "carrier": "photon", "acts_on": "electric charge"},
    "weak": {"relative_strength": 1e-6, "range_m": 1e-18,
             "carrier": "W and Z bosons", "acts_on": "flavor"},
    "gravitational": {"relative_strength": 5.9e-39, "range_m": math.inf,
                      "carrier": "graviton (hypothetical)", "acts_on": "mass-energy"},
}


def em_coupling():
    """The fine-structure constant alpha = e^2/(4 pi eps0 hbar c) ~= 1/137 -- the
    dimensionless strength of the electromagnetic interaction."""
    return K_COULOMB * E_CHARGE ** 2 / (HBAR * C_LIGHT)


def gravitational_coupling(mass=M_PROTON):
    """Dimensionless gravitational coupling alpha_g = G m^2/(hbar c). For two
    protons it is ~5.9e-39 -- the EM coupling's ~1e36-times-smaller cousin. Scales
    as mass^2."""
    if mass <= 0:
        raise ValueError("mass must be positive")
    return G_NEWTON * mass ** 2 / (HBAR * C_LIGHT)


def em_to_gravity_ratio(mass=M_PROTON):
    """The ratio of electromagnetic to gravitational force between two particles of
    the given mass and one elementary charge: k e^2/(G m^2). Distance-independent
    (both ~1/r^2). ~1.2e36 for protons -- how many times weaker gravity is."""
    if mass <= 0:
        raise ValueError("mass must be positive")
    return K_COULOMB * E_CHARGE ** 2 / (G_NEWTON * mass ** 2)


def strongest_to_weakest():
    """The four forces ordered by relative strength, strong -> gravity."""
    return sorted(FORCES, key=lambda f: FORCES[f]["relative_strength"], reverse=True)


def force_range(name):
    """The range of a force in meters (infinite for massless carriers: photon,
    graviton; short for the massive W/Z and the confined gluon)."""
    if name not in FORCES:
        raise ValueError(f"unknown force {name!r}; know {sorted(FORCES)}")
    return FORCES[name]["range_m"]


def force_carrier(name):
    """The exchange particle (gauge boson) that mediates the force."""
    if name not in FORCES:
        raise ValueError(f"unknown force {name!r}; know {sorted(FORCES)}")
    return FORCES[name]["carrier"]


if __name__ == "__main__":
    print("the four forces, strongest to weakest:")
    for f in strongest_to_weakest():
        d = FORCES[f]
        rng = "infinite" if math.isinf(d["range_m"]) else f"{d['range_m']:.0e} m"
        print(f"  {f:16s} strength ~{d['relative_strength']:.1e}  range {rng:9s}  "
              f"carrier {d['carrier']}")

    print("\nfrom fundamental constants:")
    print(f"  fine-structure alpha   = {em_coupling():.6e}  (= 1/{1/em_coupling():.2f})")
    print(f"  gravitational alpha_g  = {gravitational_coupling():.3e}  (two protons)")
    print(f"  F_EM / F_grav (protons) = {em_to_gravity_ratio():.3e}  "
          f"(gravity is that many times weaker)")
    print(f"  consistency: alpha/alpha_g = {em_coupling()/gravitational_coupling():.3e} "
          f"(same ratio)")
