"""Solid state physics: crystal structure, X-ray/Bragg diffraction, the free-
electron gas (Fermi energy, density of states, Fermi-Dirac occupation), and
semiconductor carrier concentration -- the modern-physics chapter that applies
the Schrodinger equation (dgs/schrodinger.py) and the Fermi-Dirac statistics
this repo doesn't have yet to real materials.

The throughline: a solid's electrons are a free-electron GAS confined to a box
(same particle-in-a-box math as dgs/schrodinger.py), and packing N of them
into the lowest states up to the Fermi energy is the T=0 limit of the
Fermi-Dirac distribution built here -- band theory and semiconductor
conduction are both just "where does the Fermi level sit relative to a gap."

NumPy only. Education.
"""

import numpy as np

HBAR = 1.054571817e-34   # J s
K_B = 1.380649e-23        # J/K
M_E = 9.1093837015e-31    # electron mass, kg
E_CHARGE = 1.602176634e-19  # J per eV


# -- Crystal structure: packing fraction for the standard cubic lattices ------

def packing_fraction(structure):
    """Fraction of a unit cell's volume filled by hard-sphere atoms touching
    along the cell's close-packed direction, for the three standard cubic
    lattices: 'SC' (simple cubic), 'BCC' (body-centered), 'FCC' (face-centered).
    These are the textbook constants (pi/6, sqrt(3)*pi/8, sqrt(2)*pi/6)."""
    structure = structure.upper()
    if structure == "SC":
        return np.pi / 6
    if structure == "BCC":
        return np.sqrt(3) * np.pi / 8
    if structure == "FCC":
        return np.sqrt(2) * np.pi / 6
    raise ValueError(f"structure must be 'SC', 'BCC', or 'FCC', got {structure!r}")


def unit_cell_positions(structure):
    """Fractional atom coordinates (in [0,1]^3) for a conventional cubic unit
    cell diagram, plus each drawn atom's SHARING WEIGHT (1/8 for a corner
    shared among 8 neighboring cells, 1/2 for a face shared with 1 neighbor,
    1 for a body-center atom exclusive to this cell). Returns
    (positions, weights). Summing weights recovers atoms_per_unit_cell exactly
    -- the consistency check this function's tests actually verify."""
    structure = structure.upper()
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    corner_weights = np.full(len(corners), 1 / 8)

    if structure == "SC":
        return corners, corner_weights
    if structure == "BCC":
        body = np.array([[0.5, 0.5, 0.5]])
        return np.vstack([corners, body]), np.concatenate([corner_weights, [1.0]])
    if structure == "FCC":
        faces = np.array([
            [0.5, 0.5, 0.0], [0.5, 0.5, 1.0],
            [0.5, 0.0, 0.5], [0.5, 1.0, 0.5],
            [0.0, 0.5, 0.5], [1.0, 0.5, 0.5],
        ])
        face_weights = np.full(len(faces), 1 / 2)
        return np.vstack([corners, faces]), np.concatenate([corner_weights, face_weights])
    raise ValueError(f"structure must be 'SC', 'BCC', or 'FCC', got {structure!r}")


def plot_unit_cell(structure, ax=None):
    """3D scatter/wireframe diagram of a cubic unit cell (SC/BCC/FCC): corner
    atoms, a body-center atom for BCC, or face-center atoms for FCC, plus the
    cube's edges for reference. Returns the matplotlib Axes3D used, so the
    caller can add it to a figure or save it."""
    import matplotlib.pyplot as plt

    positions, weights = unit_cell_positions(structure)
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")

    edges = [
        [(0,0,0),(1,0,0)], [(0,0,0),(0,1,0)], [(0,0,0),(0,0,1)],
        [(1,1,1),(0,1,1)], [(1,1,1),(1,0,1)], [(1,1,1),(1,1,0)],
        [(1,0,0),(1,1,0)], [(1,0,0),(1,0,1)], [(0,1,0),(1,1,0)],
        [(0,1,0),(0,1,1)], [(0,0,1),(1,0,1)], [(0,0,1),(0,1,1)],
    ]
    for p0, p1 in edges:
        xs, ys, zs = zip(p0, p1)
        ax.plot(xs, ys, zs, color="gray", lw=1)

    corner_mask = weights < 1.0 / 4  # corners only (weight 1/8), for distinct styling
    ax.scatter(*positions[corner_mask].T, s=200, c="steelblue", edgecolor="k", label="corner (1/8)")
    other_mask = ~corner_mask
    if np.any(other_mask):
        label = "body center (1)" if structure.upper() == "BCC" else "face center (1/2)"
        ax.scatter(*positions[other_mask].T, s=260, c="crimson", edgecolor="k", label=label)

    ax.set_title(f"{structure.upper()} unit cell ({atoms_per_unit_cell(structure)} atoms/cell)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=8)
    return ax


def atoms_per_unit_cell(structure):
    """Number of full atoms per conventional cubic unit cell (corner atoms
    shared 1/8 each, face atoms shared 1/2 each, body atom exclusive)."""
    structure = structure.upper()
    if structure == "SC":
        return 1
    if structure == "BCC":
        return 2
    if structure == "FCC":
        return 4
    raise ValueError(f"structure must be 'SC', 'BCC', or 'FCC', got {structure!r}")


# -- Bragg diffraction (same condition as Davisson-Germer, for X-rays) --------

def bragg_angle(d_spacing, wavelength, order=1):
    """Bragg's law d*sin(theta) = n*lambda, solved for theta (radians).
    Same diffraction condition as the electron-diffraction Davisson-Germer
    check in notebooks/matter_waves_chapter5_sympy_torch.ipynb -- here with
    X-rays and an interplanar spacing instead of electrons and a lattice
    constant, but it's the identical wave condition."""
    if d_spacing <= 0 or wavelength <= 0:
        raise ValueError("d_spacing and wavelength must be positive")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    ratio = order * wavelength / (2 * d_spacing)
    if ratio > 1:
        raise ValueError(f"no diffraction possible: n*lambda/(2d) = {ratio:.3f} > 1")
    return np.arcsin(ratio)


# -- Free electron gas: Fermi energy and density of states --------------------

def fermi_energy(n_density):
    """Fermi energy of a free-electron gas (T=0) at number density n
    (electrons/m^3): E_F = (hbar^2/2m)*(3*pi^2*n)^(2/3). This is the T=0
    limit of Fermi-Dirac occupation -- fill single-particle box states from
    the bottom up until all N electrons are placed; E_F is the top rung."""
    if n_density <= 0:
        raise ValueError(f"n_density must be positive, got {n_density}")
    return (HBAR ** 2 / (2 * M_E)) * (3 * np.pi ** 2 * n_density) ** (2 / 3)


def fermi_temperature(n_density):
    """T_F = E_F/k_B -- the temperature at which thermal energy becomes
    comparable to the Fermi energy (huge for metals, ~10^4-10^5 K, which is
    why the free electron gas stays "degenerate" -- essentially T=0-like --
    at room temperature)."""
    return fermi_energy(n_density) / K_B


def density_of_states_3d(E, n_density):
    """3D free-electron density of states g(E) ~ E^(1/2), normalized so that
    integral_0^E_F g(E) dE = n (the density used to define E_F in the first
    place) -- internal consistency check baked into the normalization."""
    if np.any(np.asarray(E) < 0):
        raise ValueError("E must be non-negative")
    E_F = fermi_energy(n_density)
    E = np.asarray(E, dtype=float)
    g = (3 * n_density) / (2 * E_F ** 1.5) * np.sqrt(np.maximum(E, 0.0))
    return g


# -- Classical (Drude) conduction, the Hall effect ----------------------------

def drude_conductivity(n_density, tau, m=M_E):
    """Classical (Drude) theory of conduction: sigma = n*e^2*tau/m, where tau
    is the mean free time between electron-lattice collisions. Same free
    electron gas as fermi_energy above, but here treated as a classical
    fluid accelerated by E and randomized by collisions -- no quantum
    statistics needed for THIS formula (that's what makes it "classical")."""
    if n_density <= 0:
        raise ValueError(f"n_density must be positive, got {n_density}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return n_density * E_CHARGE ** 2 * tau / m


def hall_coefficient(n_density, q=-E_CHARGE):
    """Hall coefficient R_H = 1/(n*q) -- q negative for electron carriers
    (the usual metal/n-type case), positive for hole carriers (p-type),
    so the SIGN of a measured R_H directly reveals the majority carrier type."""
    if n_density <= 0:
        raise ValueError(f"n_density must be positive, got {n_density}")
    if q == 0:
        raise ValueError("q must be nonzero")
    return 1.0 / (n_density * q)


def hall_voltage(I, B, n_density, thickness, q=-E_CHARGE):
    """Hall voltage V_H = I*B/(n*q*t) across a conductor of thickness t
    carrying current I in field B -- the measurement that DEFINES the Hall
    coefficient above (V_H = R_H * I * B / t)."""
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    return hall_coefficient(n_density, q) * I * B / thickness


# -- Thermal conduction: the quantum (Wiedemann-Franz) bridge -----------------

def lorenz_number():
    """Theoretical Lorenz number L = (pi^2/3)*(k_B/e)^2 from the Sommerfeld
    (quantum) free-electron model -- NOT a fitted constant, derived from the
    same Fermi-Dirac statistics as fermi_energy/density_of_states_3d above.
    Matches the experimental value (~2.44e-8 W*Ohm/K^2) across most metals,
    which is itself evidence that thermal AND electrical conduction share
    the same carriers (the free electron gas), not two separate mechanisms."""
    return (np.pi ** 2 / 3) * (K_B / E_CHARGE) ** 2


def thermal_conductivity_from_electrical(sigma, T):
    """Wiedemann-Franz law: kappa = L*sigma*T -- thermal conductivity
    PREDICTED directly from a measured electrical conductivity, no separate
    thermal measurement needed, because both are carried by the same
    electron gas."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    return lorenz_number() * sigma * T


# -- Superconductivity: flux quantization and the Josephson effect -----------

def flux_quantum():
    """Magnetic flux quantum Phi_0 = h/(2e) -- the smallest unit of magnetic
    flux a superconducting loop can enclose. The factor of 2 (not h/e) is
    itself a measurement: it's direct evidence that the current carriers in
    a superconductor are Cooper PAIRS (charge 2e), not single electrons."""
    h = 2 * np.pi * HBAR
    return h / (2 * E_CHARGE)


def josephson_frequency(V):
    """AC Josephson effect: a Josephson junction biased at DC voltage V
    oscillates at f = 2*e*V/h -- a voltage-to-frequency converter accurate
    enough that the Josephson constant 2e/h is used to REALIZE the volt in
    the SI system (a piece of quantum mechanics is a national measurement
    standard)."""
    h = 2 * np.pi * HBAR
    return 2 * E_CHARGE * V / h


def fermi_dirac(E, E_F, T):
    """Fermi-Dirac occupation probability f(E) = 1/(exp((E-E_F)/(k_B T)) + 1).
    T=0 limit is a hard step (1 below E_F, 0 above) -- handled via a large
    but finite ratio to avoid overflow rather than a special-cased branch."""
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    E = np.asarray(E, dtype=float)
    if T == 0:
        return np.where(E < E_F, 1.0, np.where(E > E_F, 0.0, 0.5))
    x = (E - E_F) / (K_B * T)
    x = np.clip(x, -700, 700)   # avoid exp() overflow; occupation saturates well before this
    return 1.0 / (np.exp(x) + 1.0)


# -- Semiconductors: intrinsic carrier concentration --------------------------

def intrinsic_carrier_concentration(T, E_gap_eV, A=1e21):
    """Intrinsic semiconductor carrier density n_i(T) ~ A*T^(3/2)*exp(-E_g/(2 k_B T)).
    A is a material-dependent prefactor (absorbs effective-mass/density-of-states
    constants); the T^(3/2)*exp(-E_g/2kT) SHAPE is universal and is what makes
    semiconductors conduct better hot and act like insulators cold -- the
    opposite temperature dependence of a metal (whose conductivity falls with T)."""
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if E_gap_eV < 0:
        raise ValueError(f"E_gap_eV must be non-negative, got {E_gap_eV}")
    E_gap_J = E_gap_eV * E_CHARGE
    return A * T ** 1.5 * np.exp(-E_gap_J / (2 * K_B * T))


def classify_material(E_gap_eV):
    """Rough classification by band gap: conductor (E_g<=0, overlapping
    bands), semiconductor (0<E_g<~3 eV, thermally accessible), insulator
    (E_g large, effectively no thermal carriers at room T)."""
    if E_gap_eV <= 0:
        return "conductor"
    if E_gap_eV < 3.0:
        return "semiconductor"
    return "insulator"


if __name__ == "__main__":
    for s in ("SC", "BCC", "FCC"):
        positions, weights = unit_cell_positions(s)
        print(f"{s}: packing fraction = {packing_fraction(s):.4f}, "
              f"atoms/cell = {atoms_per_unit_cell(s)}, "
              f"sum(weights) = {weights.sum():.4f}, "
              f"{len(positions)} atoms drawn")

    # copper: n ~ 8.47e28 /m^3, textbook E_F ~ 7.0 eV
    n_cu = 8.47e28
    E_F_cu = fermi_energy(n_cu)
    print(f"\ncopper: E_F = {E_F_cu/E_CHARGE:.2f} eV (textbook ~7.0 eV), "
          f"T_F = {fermi_temperature(n_cu):.0f} K")

    # silicon band gap 1.12 eV: intrinsic carriers rise sharply with T
    for T in (200.0, 300.0, 400.0):
        n_i = intrinsic_carrier_concentration(T, 1.12)
        print(f"Si at T={T:.0f} K: n_i ~ {n_i:.3e} (arbitrary units, A=1e21)")

    print(f"\nclassify: E_g=0 -> {classify_material(0.0)}, "
          f"E_g=1.12 -> {classify_material(1.12)}, E_g=5.0 -> {classify_material(5.0)}")

    print("\n--- classical/quantum conduction, Hall effect, superconductivity ---")
    sigma_cu = drude_conductivity(n_cu, tau=2.5e-14)
    print(f"Drude sigma(Cu) = {sigma_cu:.3e} S/m (real ~5.96e7 S/m)")
    print(f"Hall coefficient(Cu) = {hall_coefficient(n_cu):.3e} m^3/C (real ~ -5.5e-11)")
    L = lorenz_number()
    print(f"Lorenz number = {L:.4e} W-Ohm/K^2 (experimental ~2.44e-8)")
    print(f"Wiedemann-Franz kappa(Cu, 300K) = {thermal_conductivity_from_electrical(sigma_cu, 300.0):.1f} "
          f"W/(m K) (real ~401)")
    print(f"flux quantum = {flux_quantum():.6e} Wb (known 2.067834e-15 Wb)")
    print(f"Josephson frequency @ 1V = {josephson_frequency(1.0):.6e} Hz (known 4.835979e14 Hz/V)")
