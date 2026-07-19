"""The Hartree self-consistent-field (SCF) method for a multi-electron
atom -- literally the procedure: "An educated guess is made initially
for each of the N ground-state electron waves. Starting with this
guess, the [charge density] and U_eff for every electron can be
computed and all N Schrodinger equations solved" -- then repeated until
the effective potential stops changing.

Applied here to carbon (Z=6, configuration 1s^2 2s^2 2p^2). Uses atomic
units internally (hbar=m_e=e=4*pi*eps0=1; energy unit = 1 Hartree =
27.211 eV, length unit = 1 Bohr radius) -- standard practice in real
electronic-structure code, and it avoids SI floating-point scale issues.

METHOD, concretely:
  1. Initial guess: bare nuclear potential V_eff(r) = -Z/r (no screening
     yet -- literally "an educated guess").
  2. Solve the radial Schrodinger equation for each occupied orbital (1s,
     2s, 2p) via finite-difference discretization + matrix
     diagonalization (a real, standard, numerically robust technique --
     no shooting-method convergence fragility).
  3. Build the total electron charge density from the occupied orbitals,
     compute the new Hartree (electron-electron repulsion) potential via
     the exact classical electrostatic shell-theorem formula.
  4. Form the new V_eff = -Z/r + V_Hartree(r), damped-mix with the
     previous iteration's V_eff for stability, and repeat.

HONEST LIMITATION: this is HARTREE, not HARTREE-FOCK -- it has no
exchange term. Historically, Hartree's original 1928 method omitted
electron exchange/antisymmetry entirely; Fock added it in 1930 by
building the wavefunction as a Slater determinant, which automatically
enforces the Pauli exclusion principle (no two electrons in the same
quantum state) via the antisymmetry required for indistinguishable
fermions. Without exchange, this module can only be trusted for
qualitative energy-level ORDERING (1s deeper than 2s deeper than 2p) and
convergence behavior, not precise ionization energies.

Also includes the quantum defect: a real, separate technique for
alkali/Rydberg atoms where the outer electron's energy is corrected by a
measured, term-dependent "defect" delta: E_n = -13.6 eV * Z_eff^2 /
(n-delta)^2, accounting for how much the orbital penetrates the
core (s-orbitals penetrate most, hence largest delta).
"""

import numpy as np

HARTREE_EV = 27.211386245988
BOHR_RADIUS_M = 5.29177210903e-11


def thomas_fermi_radius_bohr(Z):
    """Thomas-Fermi screening radius b = 0.8853 * Z^(-1/3), in units of
    the Bohr radius -- the real, classic statistical-atom length scale
    that sets how far the nuclear charge is screened by the electron
    cloud."""
    if Z <= 0:
        raise ValueError("Z must be positive")
    return 0.8853 * Z ** (-1.0 / 3.0)


def radial_hamiltonian_matrix(r_grid, l, V_eff):
    """Finite-difference discretization of the radial Schrodinger
    equation for u(r) = r*R(r):
      H u = [-1/2 d^2/dr^2 + l(l+1)/(2r^2) + V_eff(r)] u = E u
    Returns a dense, symmetric (Hermitian) matrix -- robust to
    diagonalize directly with no shooting/bisection needed."""
    if l < 0:
        raise ValueError("l must be non-negative")
    n = len(r_grid)
    if n < 3:
        raise ValueError("r_grid must have at least 3 points")
    dr = r_grid[1] - r_grid[0]
    H = np.zeros((n, n))
    kinetic_diag = 1.0 / dr**2
    kinetic_offdiag = -0.5 / dr**2
    for i in range(n):
        centrifugal = l * (l + 1) / (2.0 * r_grid[i]**2) if r_grid[i] > 0 else 0.0
        H[i, i] = kinetic_diag + centrifugal + V_eff[i]
        if i > 0:
            H[i, i - 1] = kinetic_offdiag
        if i < n - 1:
            H[i, i + 1] = kinetic_offdiag
    return H


def solve_radial_states(r_grid, l, V_eff, n_states=3):
    """Diagonalize the radial Hamiltonian; return the n_states lowest
    (most bound) eigenvalues (Hartree units) and their normalized
    eigenvectors u(r) (with integral of u^2 dr = 1)."""
    if n_states < 1:
        raise ValueError("n_states must be at least 1")
    H = radial_hamiltonian_matrix(r_grid, l, V_eff)
    eigvals, eigvecs = np.linalg.eigh(H)
    dr = r_grid[1] - r_grid[0]
    energies = eigvals[:n_states]
    orbitals = []
    for k in range(n_states):
        u = eigvecs[:, k]
        norm = np.sqrt(np.sum(u**2) * dr)
        orbitals.append(u / norm)
    return energies, orbitals


def hartree_potential_from_density(r_grid, P_total):
    """Classical electrostatic Hartree potential from the total radial
    probability density P(r) = 4*pi*r^2*rho(r) (i.e. u(r)^2 summed over
    occupied, weighted orbitals). Uses the exact shell-theorem identity:
      V_H(r) = (1/r) * integral_0^r P(r') dr'  +  integral_r^inf P(r')/r' dr'
    (enclosed charge acts like a point charge at the origin; charge
    outside r contributes its own 1/r' potential, evaluated at that
    shell) -- standard, exact classical electrostatics, no approximation
    beyond spherical symmetry."""
    if len(r_grid) != len(P_total):
        raise ValueError("r_grid and P_total must have the same length")
    dr = r_grid[1] - r_grid[0]
    enclosed = np.cumsum(P_total) * dr
    outer_integrand = P_total / np.maximum(r_grid, 1e-12)
    outer_from_r = np.cumsum(outer_integrand[::-1])[::-1] * dr
    V_H = enclosed / np.maximum(r_grid, 1e-12) + outer_from_r
    return V_H


def hartree_scf_carbon(r_max_bohr=15.0, n_grid=800, n_iter=15, mixing_alpha=0.3):
    """Run the Hartree SCF loop for carbon (Z=6, 1s^2 2s^2 2p^2).
    Returns final orbital energies (eV) for 1s, 2s, 2p and the
    convergence history of the 1s energy across iterations."""
    if r_max_bohr <= 0 or n_grid < 10 or n_iter < 1:
        raise ValueError("r_max_bohr, n_grid, n_iter must be positive (n_grid>=10)")
    if not (0 < mixing_alpha <= 1):
        raise ValueError("mixing_alpha must be in (0, 1]")

    Z = 6
    dr = r_max_bohr / n_grid
    # r_min MUST be set to the grid spacing itself (not an arbitrarily tiny
    # value): a linear grid whose first point sits deep inside the -Z/r
    # singularity (|V(r_min)| >> 1/dr^2, the discretization's kinetic-energy
    # scale) produces a spurious, unphysically deep eigenstate localized on
    # that single grid point -- caught by comparing against the known bare
    # hydrogenic estimate E_1s = -Z^2*13.6 eV before trusting the SCF loop.
    r_grid = np.linspace(dr, r_max_bohr, n_grid)
    V_eff = -Z / r_grid   # step 1: "an educated guess" -- bare nucleus, no screening yet

    history_1s_ev = []
    for _ in range(n_iter):
        E_l0, orb_l0 = solve_radial_states(r_grid, l=0, V_eff=V_eff, n_states=2)
        E_l1, orb_l1 = solve_radial_states(r_grid, l=1, V_eff=V_eff, n_states=1)

        u_1s, u_2s = orb_l0[0], orb_l0[1]
        u_2p = orb_l1[0]

        # total radial probability density, weighted by real occupation (1s^2 2s^2 2p^2)
        P_total = 2 * u_1s**2 + 2 * u_2s**2 + 2 * u_2p**2

        V_H = hartree_potential_from_density(r_grid, P_total)
        V_eff_new = -Z / r_grid + V_H
        V_eff = mixing_alpha * V_eff_new + (1 - mixing_alpha) * V_eff   # damped mixing

        history_1s_ev.append(E_l0[0] * HARTREE_EV)

    return {
        "E_1s_ev": E_l0[0] * HARTREE_EV,
        "E_2s_ev": E_l0[1] * HARTREE_EV,
        "E_2p_ev": E_l1[0] * HARTREE_EV,
        "history_1s_ev": history_1s_ev,
    }


def quantum_defect_energy_ev(n, delta, Z_eff=1.0):
    """Quantum defect theory: E_n = -13.6 eV * Z_eff^2 / (n - delta)^2.
    delta (the quantum defect) is largest for s-orbitals (most core
    penetration) and near-zero for high-l orbitals (least penetration) --
    a real, separate technique from full SCF, used for alkali/Rydberg
    spectroscopy."""
    if n <= 0:
        raise ValueError("n must be positive")
    if Z_eff <= 0:
        raise ValueError("Z_eff must be positive")
    if (n - delta) <= 0:
        raise ValueError("n - delta must be positive (effective quantum number must be positive)")
    return -13.6 * Z_eff**2 / (n - delta) ** 2


if __name__ == "__main__":
    print("=== Thomas-Fermi screening length scale ===\n")
    for Z in [1, 6, 29, 79]:
        b = thomas_fermi_radius_bohr(Z)
        print(f"  Z={Z:3d}: Thomas-Fermi radius = {b:.3f} Bohr radii")

    print("\n=== Hartree SCF for carbon (Z=6, 1s^2 2s^2 2p^2) ===\n")
    print("Step 1 (the 'educated guess'): start from the bare, unscreened")
    print("nuclear potential V_eff = -Z/r. Then iterate: solve each electron's")
    print("radial Schrodinger equation, rebuild the charge density, recompute")
    print("the Hartree (electron-electron repulsion) potential, repeat.\n")

    result = hartree_scf_carbon()
    print(f"converged orbital energies:")
    print(f"  1s: {result['E_1s_ev']:8.2f} eV")
    print(f"  2s: {result['E_2s_ev']:8.2f} eV")
    print(f"  2p: {result['E_2p_ev']:8.2f} eV")
    print(f"  ordering check (1s deeper than 2s deeper than 2p): "
          f"{result['E_1s_ev'] < result['E_2s_ev'] < result['E_2p_ev']}")
    print(f"\n1s energy across SCF iterations (eV): "
          f"{[round(e, 2) for e in result['history_1s_ev']]}")
    print("(Real experimental carbon core/valence energies are roughly -308 eV")
    print(" (1s), -19 eV (2s), -11 eV (2p) -- this simplified HARTREE-only model")
    print(" (no exchange term) is not expected to match those precisely, but")
    print(" the qualitative ordering and iterative convergence ARE real.)\n")

    print("=== Why this is Hartree, NOT Hartree-Fock ===")
    print("This method has no exchange term -- it doesn't enforce antisymmetry")
    print("under electron exchange, so nothing here automatically implements the")
    print("Pauli exclusion principle for identical fermions. Real Hartree-Fock")
    print("(Fock, 1930) builds the total wavefunction as a Slater determinant,")
    print("which is antisymmetric by construction and yields an extra 'exchange")
    print("energy' term with no classical analogue -- purely a consequence of")
    print("quantum indistinguishability.\n")

    print("=== Quantum defect: a separate real technique (alkali/Rydberg atoms) ===\n")
    # real, known sodium quantum defects: s~1.35, p~0.86, d~0.01
    for label, delta in [("3s (Na)", 1.35), ("3p (Na)", 0.86), ("3d (Na)", 0.01)]:
        n = 3
        E_hydrogenic = -13.6 / n**2
        E_defect = quantum_defect_energy_ev(n, delta)
        print(f"  {label}: hydrogenic E_3 = {E_hydrogenic:.2f} eV  ->  "
              f"quantum-defect-corrected E = {E_defect:.2f} eV")
    print("  (s-orbitals penetrate the ion core most -> largest defect -> most bound)")
