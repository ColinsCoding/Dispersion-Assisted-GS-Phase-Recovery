"""The proton-proton chain: how four hydrogens become sunlight.

The Sun runs on the pp-chain -- the net reaction that fuses four protons into one
helium-4 nucleus:
        4 (1H)  ->  4He + 2 e+ + 2 nu + energy.
Helium-4 is lighter than four hydrogens, and that missing MASS is the energy, by
E = m c^2. Using atomic masses (which carry the electrons, so the positron
annihilation is bookkept automatically):
        Q = (4 m_H - m_He4) c^2  ~=  26.7 MeV  per helium-4.

It happens in three steps (branch pp-I):
    1.  1H + 1H -> 2H + e+ + nu      (the slow, weak-force step; +annihilation)
    2.  2H + 1H -> 3He + gamma
    3.  3He + 3He -> 4He + 1H + 1H
Two runs each of steps 1 and 2 feed one run of step 3; their energies sum to the same
26.7 MeV, a nice internal check.

Why FOUR protons and not more, and why the Sun -- not a supernova -- can't fuse past
iron: the BINDING ENERGY PER NUCLEON rises to a peak at IRON-56 (~8.8 MeV/nucleon)
and falls after. Fusing UP toward iron releases energy (the products are more tightly
bound); past iron it would COST energy. That single curve explains both why fusion
powers stars and why iron is where it ends.

Everything is a mass-defect times c^2, checked against the textbook 26.7 MeV, the
iron peak, and -- from the Sun's luminosity -- the ~6e11 kg/s of hydrogen it fuses and
the ~4e9 kg/s of mass it turns into light. NumPy-free; py-3.13.
"""

U_TO_MEV = 931.49410242          # 1 atomic mass unit in MeV/c^2
MEV_TO_J = 1.602176634e-13       # 1 MeV in joules
C_LIGHT = 2.99792458e8           # m/s
AMU_KG = 1.66053906660e-27       # kg

# atomic masses (u) -- include the atomic electrons
M = {
    "1H": 1.00782503207, "n": 1.00866491588, "2H": 2.01410177785,
    "3He": 3.01602932008, "4He": 4.00260325413,
    "56Fe": 55.93493633, "238U": 238.05078826, "e": 0.00054857990,
}

# pp-I branch: (label, Q in MeV, how many times it runs per helium-4)
PP_I_STEPS = [
    ("1H + 1H -> 2H + e+ + nu", 1.442, 2),   # 0.420 fusion + 1.022 e+ annihilation
    ("2H + 1H -> 3He + gamma", 5.493, 2),
    ("3He + 3He -> 4He + 2 1H", 12.859, 1),
]

NEUTRINO_LOSS_MEV = 0.593        # ~2 neutrinos carry this away (escape the Sun)


def mass_defect_energy(reactant_masses_u, product_masses_u):
    """Energy released Q = (sum m_reactants - sum m_products) * c^2, in MeV.
    Positive Q means the reaction gives up energy (exothermic)."""
    dm = sum(reactant_masses_u) - sum(product_masses_u)
    return dm * U_TO_MEV


def pp_chain_net_energy():
    """The pp-chain net Q = (4 m_H - m_He4) c^2 ~= 26.73 MeV per helium-4,
    including positron annihilation (atomic masses handle the electrons)."""
    return mass_defect_energy([M["1H"]] * 4, [M["4He"]])


def pp_chain_step_sum():
    """Sum the pp-I step energies weighted by how often each runs -- an
    independent route to the net Q that must agree with pp_chain_net_energy()."""
    return sum(q * n for _, q, n in PP_I_STEPS)


def energy_as_light(include_neutrino_loss=True):
    """The energy that actually becomes heat and light: the net Q minus the
    ~0.59 MeV the neutrinos carry straight out of the Sun."""
    q = pp_chain_net_energy()
    return q - NEUTRINO_LOSS_MEV if include_neutrino_loss else q


def binding_energy(Z, N, atomic_mass_u):
    """Total nuclear binding energy B = (Z m_H + N m_n - M_atomic) c^2 in MeV.
    The electrons in m_H and M_atomic cancel, so this is the mass defect of the
    nucleus -- how much energy it took to bind it (and would take to unbind it)."""
    if Z < 0 or N < 0:
        raise ValueError("Z and N must be non-negative")
    dm = Z * M["1H"] + N * M["n"] - atomic_mass_u
    return dm * U_TO_MEV


def binding_energy_per_nucleon(Z, N, atomic_mass_u):
    """B/A -- the tightness of binding. Peaks near iron-56 (~8.8 MeV), the reason
    fusion releases energy up to iron and fission releases it coming down from
    the heavy end."""
    A = Z + N
    if A <= 0:
        raise ValueError("need at least one nucleon")
    return binding_energy(Z, N, atomic_mass_u) / A


def solar_fusion_rate(luminosity_W=3.828e26):
    """From the Sun's luminosity, how fast the pp-chain runs. Returns helium-4
    made per second, protons consumed per second, hydrogen mass fused per second,
    and the mass turned directly into energy per second (L/c^2)."""
    if luminosity_W <= 0:
        raise ValueError("luminosity must be positive")
    q_light_J = energy_as_light() * MEV_TO_J        # energy per He-4 as light
    he4_per_s = luminosity_W / q_light_J
    protons_per_s = 4 * he4_per_s
    hydrogen_kg_per_s = protons_per_s * M["1H"] * AMU_KG
    mass_to_energy_kg_per_s = luminosity_W / C_LIGHT ** 2
    return {
        "he4_per_s": he4_per_s,
        "protons_per_s": protons_per_s,
        "hydrogen_kg_per_s": hydrogen_kg_per_s,
        "mass_to_energy_kg_per_s": mass_to_energy_kg_per_s,
    }


if __name__ == "__main__":
    print("pp-chain: 4 (1H) -> 4He + energy")
    print(f"  net Q = (4 m_H - m_He4) c^2 = {pp_chain_net_energy():.3f} MeV")
    print(f"  step-by-step sum            = {pp_chain_step_sum():.3f} MeV  "
          f"(should match)")
    for label, q, n in PP_I_STEPS:
        print(f"    x{n}: {label:28s} Q = {q:6.3f} MeV")
    print(f"  as light (minus neutrinos)  = {energy_as_light():.3f} MeV")

    print("\nbinding energy per nucleon (the iron peak):")
    for name, Z, N in [("2H", 1, 1), ("4He", 2, 2), ("56Fe", 26, 30), ("238U", 92, 146)]:
        b = binding_energy(Z, N, M[name])
        print(f"  {name:5s}: B = {b:8.2f} MeV,  B/A = {b/(Z+N):.3f} MeV/nucleon")

    print("\nfrom the Sun's luminosity (3.828e26 W):")
    r = solar_fusion_rate()
    print(f"  helium-4 produced : {r['he4_per_s']:.2e} /s")
    print(f"  hydrogen fused    : {r['hydrogen_kg_per_s']:.2e} kg/s "
          f"(~600 million tonnes/s)")
    print(f"  mass -> energy    : {r['mass_to_energy_kg_per_s']:.2e} kg/s "
          f"(L/c^2, ~4 million tonnes/s)")
