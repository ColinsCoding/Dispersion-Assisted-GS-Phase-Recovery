"""Test the Hartree self-consistent-field radial solver: the finite-
difference Hamiltonian (checked against the known bare hydrogenic
formula E_n=-Z^2*13.6/n^2 eV), the classical Hartree-potential
integral, and the carbon SCF loop's convergence/ordering. Also confirms
the specific grid-singularity bug caught and fixed this session (an
r_min set too close to zero produces a spurious, unphysically deep
eigenstate)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import hartree_scf_atom as hscf

# 1. Thomas-Fermi radius shrinks with increasing Z (more nuclear charge
#    screens over a shorter distance)
b_low_Z = hscf.thomas_fermi_radius_bohr(1)
b_high_Z = hscf.thomas_fermi_radius_bohr(79)
assert b_high_Z < b_low_Z

# 2. radial_hamiltonian_matrix + solve_radial_states reproduce the known
#    bare hydrogenic 1s energy E_1 = -Z^2 * 13.6 eV, WHEN the grid's r_min
#    is set correctly (r_min = dr, not an arbitrarily tiny value) --
#    this is exactly the bug caught and fixed this session
Z = 6
r_max, n_grid = 15.0, 800
dr = r_max / n_grid
r_grid_correct = np.linspace(dr, r_max, n_grid)
V_bare = -Z / r_grid_correct
E, orb = hscf.solve_radial_states(r_grid_correct, l=0, V_eff=V_bare, n_states=1)
E_1s_ev = E[0] * hscf.HARTREE_EV
expected_bare_ev = -(Z**2) * 13.6
assert abs(E_1s_ev - expected_bare_ev) / abs(expected_bare_ev) < 0.02   # within 2%

# 3. the BUGGY grid choice (r_min too close to zero) gives a wildly wrong,
#    much-too-negative energy -- confirms the fix actually matters
r_grid_buggy = np.linspace(1e-4, r_max, n_grid)
V_buggy = -Z / r_grid_buggy
E_buggy, _ = hscf.solve_radial_states(r_grid_buggy, l=0, V_eff=V_buggy, n_states=1)
E_buggy_ev = E_buggy[0] * hscf.HARTREE_EV
assert E_buggy_ev < 10 * expected_bare_ev   # far more (spuriously) bound than physical

# 4. orbitals returned by solve_radial_states are normalized:
#    integral of u^2 dr = 1
u = orb[0]
norm = np.sum(u**2) * dr
assert abs(norm - 1.0) < 1e-6

# 5. hartree_potential_from_density: a delta-like density near r=0 gives a
#    potential that looks like a point charge (~1/r) far away
r_grid = np.linspace(0.01, 20.0, 2000)
dr2 = r_grid[1] - r_grid[0]
P = np.exp(-((r_grid - 0.5) ** 2) / (2 * 0.05**2))
P = P / (np.sum(P) * dr2)   # normalize to represent 1 electron
V_H = hscf.hartree_potential_from_density(r_grid, P)
# far from the charge, V_H(r) should behave like 1/r (total enclosed charge = 1)
far_idx = -1
assert abs(V_H[far_idx] - 1.0 / r_grid[far_idx]) / (1.0 / r_grid[far_idx]) < 0.05

# 6. hartree_scf_carbon: correct energy ORDERING (1s most bound, then 2s, then 2p)
result = hscf.hartree_scf_carbon(n_iter=15)
assert result["E_1s_ev"] < result["E_2s_ev"] < result["E_2p_ev"] < 0

# 7. convergence: the 1s energy history should settle down (later steps
#    change less than earlier steps)
history = result["history_1s_ev"]
early_change = abs(history[1] - history[0])
late_change = abs(history[-1] - history[-2])
assert late_change < early_change

# 8. same order of magnitude as the real experimental carbon 1s core
#    energy (~-308 eV) -- not exact (no exchange term), but not off by
#    orders of magnitude either
assert -1000 < result["E_1s_ev"] < -50

# 9. quantum_defect_energy_ev: zero defect reduces exactly to the plain
#    hydrogenic formula; nonzero defect makes the state MORE bound
E_no_defect = hscf.quantum_defect_energy_ev(3, 0.0)
assert abs(E_no_defect - (-13.6 / 9)) < 1e-9
E_with_defect = hscf.quantum_defect_energy_ev(3, 1.35)
assert E_with_defect < E_no_defect   # more bound (more negative)

# 10. input validation
for bad_call in [
    lambda: hscf.thomas_fermi_radius_bohr(-1),
    lambda: hscf.radial_hamiltonian_matrix(np.linspace(0.1, 10, 5), l=-1, V_eff=np.zeros(5)),
    lambda: hscf.radial_hamiltonian_matrix(np.linspace(0.1, 10, 2), l=0, V_eff=np.zeros(2)),
    lambda: hscf.solve_radial_states(np.linspace(0.1, 10, 100), 0, np.zeros(100), n_states=0),
    lambda: hscf.hartree_potential_from_density(np.zeros(5), np.zeros(4)),
    lambda: hscf.hartree_scf_carbon(r_max_bohr=-1.0),
    lambda: hscf.hartree_scf_carbon(mixing_alpha=0.0),
    lambda: hscf.hartree_scf_carbon(mixing_alpha=1.5),
    lambda: hscf.quantum_defect_energy_ev(-1, 0.5),
    lambda: hscf.quantum_defect_energy_ev(1, 0.5, Z_eff=-1.0),
    lambda: hscf.quantum_defect_energy_ev(1, 1.0),   # n - delta = 0
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.hartree_scf_atom tests passed")
