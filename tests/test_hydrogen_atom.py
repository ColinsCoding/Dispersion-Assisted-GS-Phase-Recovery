"""Test dgs.hydrogen_atom: Bohr energies & Z-scaling, shell degeneracy n^2, quantum-number rules,
angular-momentum magnitude/projection quantization, the centrifugal minimum, and spectral lines."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from dgs import hydrogen_atom as H

# 1. energy levels: -13.6/n^2 for H, Z^2 scaling for hydrogen-like ions
assert math.isclose(H.energy_level(1), -13.605693122, rel_tol=1e-9)
assert math.isclose(H.energy_level(2), -13.605693122/4, rel_tol=1e-9)
assert math.isclose(H.energy_level(1, Z=2), 4*H.energy_level(1), rel_tol=1e-12)   # He+ = 4x
assert math.isclose(H.energy_level(1, Z=3), 9*H.energy_level(1), rel_tol=1e-12)   # Li2+ = 9x
# antihydrogen: identical spectrum (the function is charge-magnitude based -> same numbers)
assert H.energy_level(3) == H.energy_level(3)

# 2. orbital radius r_n = n^2 a0 / Z
assert math.isclose(H.orbital_radius(1), H.A0, rel_tol=1e-12)
assert math.isclose(H.orbital_radius(2), 4*H.A0, rel_tol=1e-12)
assert math.isclose(H.orbital_radius(1, Z=2), H.A0/2, rel_tol=1e-12)

# 3. quantum-number ranges and validity
assert H.allowed_l(3) == [0, 1, 2]
assert H.allowed_m(2) == [-2, -1, 0, 1, 2]
assert H.valid_state(3, 2, -1) and H.valid_state(1, 0, 0)
assert not H.valid_state(2, 2, 0)      # l must be < n
assert not H.valid_state(3, 1, 2)      # |m| must be <= l
assert not H.valid_state(0, 0, 0)      # n >= 1

# 4. degeneracy = n^2 (and 2n^2 with spin = periodic-table shell capacities)
for n in range(1, 6):
    assert H.degeneracy(n) == n**2
    assert len(H.shell_states(n)) == n**2                 # the (l,m) count equals n^2
    assert sum(2*l + 1 for l in H.allowed_l(n)) == n**2   # sum of (2l+1) telescopes to n^2
assert [H.degeneracy(n, include_spin=True) for n in (1,2,3,4)] == [2, 8, 18, 32]

# 5. angular momentum: |L| = sqrt(l(l+1)), L_z = m, and space quantization
assert H.angular_momentum_magnitude(0) == 0
assert math.isclose(H.angular_momentum_magnitude(1), math.sqrt(2), rel_tol=1e-12)
assert math.isclose(H.angular_momentum_magnitude(2), math.sqrt(6), rel_tol=1e-12)
assert H.angular_momentum_z(-1) == -1
# |L| strictly exceeds its largest projection l -> L never lies on the axis
for l in (1, 2, 3):
    assert H.angular_momentum_magnitude(l) > l
    angles = H.space_quantization_angles(l)
    assert len(angles) == 2*l + 1                         # 2l+1 orientations
    assert min(angles) > 0                                # smallest tilt is nonzero
    # cos(theta) = m / sqrt(l(l+1))
    assert math.isclose(math.cos(min(angles)), l/H.angular_momentum_magnitude(l), rel_tol=1e-12)
assert H.space_quantization_angles(0) == []              # l=0: no defined direction

# 6. centrifugal barrier: l=0 has no minimum; l>0 minimum at l(l+1) a0/Z
assert H.centrifugal_minimum_radius(0) is None
assert math.isclose(H.centrifugal_minimum_radius(1), 2*H.A0, rel_tol=1e-12)      # l=1 -> 2 a0
assert math.isclose(H.centrifugal_minimum_radius(2), 6*H.A0, rel_tol=1e-12)      # l=2 -> 6 a0
# verify numerically that V_eff actually bottoms out there for l=1
r = np.linspace(0.2*H.A0, 20*H.A0, 20000)
V = H.effective_potential(r, l=1)
assert math.isclose(r[np.argmin(V)], 2*H.A0, rel_tol=2e-2)
# l=0 is monotonically rising (attractive, no interior minimum)
V0 = H.effective_potential(r, l=0)
assert np.all(np.diff(V0) > 0)

# 7. spectral lines (cross-check Rydberg): Lyman-alpha and Balmer-alpha
assert math.isclose(H.transition_energy(2, 1), 13.605693122*(1 - 1/4), rel_tol=1e-12)  # 10.2 eV
assert math.isclose(H.transition_wavelength_nm(2, 1), 121.5, abs_tol=0.5)              # Lyman-a
assert math.isclose(H.transition_wavelength_nm(3, 2), 656.3, abs_tol=1.0)              # Balmer-a (red)
# Z^2 scaling of transitions; He+ Lyman-alpha is 4x the energy
assert math.isclose(H.transition_energy(2, 1, Z=2), 4*H.transition_energy(2, 1), rel_tol=1e-12)

# 8. bounds
for bad in (lambda: H.energy_level(0), lambda: H.energy_level(1, Z=0),
            lambda: H.allowed_l(0), lambda: H.allowed_m(-1),
            lambda: H.transition_energy(1, 2)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_hydrogen_atom: all checks passed")
