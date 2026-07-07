"""Test dgs.vibration_modes: the one second-difference operator underlying both
a spring chain and the Schrodinger kinetic term; normal modes matching the
closed form and satisfying K v = omega^2 M v (mass-orthonormal); particle-in-a-
box energies matching n^2 pi^2/2; and the 64-vs-128-bit demo where a near-
degenerate mode splitting cancels in float64 but survives in decimal128."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import vibration_modes as vm

# 1. the second-difference matrix: symmetric tridiagonal [1,-2,1]/dx^2
D2 = vm.second_difference_matrix(5, dx=1.0)
assert np.allclose(D2, D2.T)
assert np.allclose(np.diag(D2), -2) and np.allclose(np.diag(D2, 1), 1)
assert np.allclose(vm.second_difference_matrix(5, dx=2.0), D2 / 4)   # 1/dx^2 scaling

# 2. THE UNIFICATION: a uniform spring chain's stiffness IS -k*(d^2/dx^2), the
#    same operator (up to a constant) as the Schrodinger kinetic term
n = 6
M, K = vm.uniform_spring_chain(n, m=1.0, k=1.0)
assert np.allclose(K, -1.0 * vm.second_difference_matrix(n, 1.0))
assert np.allclose(M, np.eye(n))
H_kin = vm.schrodinger_hamiltonian(np.zeros(n), dx=1.0, hbar=1.0, mass=1.0)
# stiffness K and kinetic operator are proportional (both = second difference)
scale = K[0, 0] / H_kin[0, 0]
assert np.allclose(K, scale * H_kin)

# 3. normal modes match the closed form and are mass-orthonormal
omega, modes = vm.normal_modes(M, K)
assert np.allclose(omega, vm.analytic_chain_frequencies(n, 1.0, 1.0))
assert np.allclose(modes.T @ M @ modes, np.eye(n), atol=1e-9)   # V^T M V = I
# each column really solves K v = omega^2 M v
for i in range(n):
    v = modes[:, i]
    assert np.allclose(K @ v, omega[i]**2 * (M @ v), atol=1e-9)

# 4. heavier masses lower the frequencies (omega ~ sqrt(k/m))
_, _ = vm.uniform_spring_chain(n, m=4.0, k=1.0)
omega_heavy = vm.normal_modes(*vm.uniform_spring_chain(n, m=4.0, k=1.0))[0]
assert np.allclose(omega_heavy, omega / 2, atol=1e-9)           # 4x mass -> half omega

# 5. QUANTUM: particle-in-a-box energies match E_n = n^2 pi^2 hbar^2/(2mL^2)
E, E_exact = vm.particle_in_box(n_grid=600, L=1.0, n_levels=5)
assert np.allclose(E, E_exact, rtol=1e-2)
assert np.isclose(E_exact[0], np.pi**2 / 2)                    # ground state
assert np.all(E <= E_exact + 1e-9)                            # finite-diff underestimates
# a finer grid gets closer to the analytic ground state
E_fine, _ = vm.particle_in_box(n_grid=1500, L=1.0, n_levels=1)
E_coarse, _ = vm.particle_in_box(n_grid=200, L=1.0, n_levels=1)
assert abs(E_fine[0] - np.pi**2/2) < abs(E_coarse[0] - np.pi**2/2)

# 6. schrodinger_hamiltonian is symmetric (Hermitian, real) with V on the diagonal
V = np.linspace(0, 3, 20)
H = vm.schrodinger_hamiltonian(V, dx=0.1)
assert np.allclose(H, H.T)
assert np.allclose(np.diag(H) - V, np.diag(vm.schrodinger_hamiltonian(np.zeros(20), 0.1)))

# 7. 64 vs 128 bit: Python precision facts
rep = vm.precision_report()
assert rep["python_float"]["bits"] == 64
assert np.isclose(rep["python_float"]["eps"], 2.220446049250313e-16)
assert 15.5 < rep["python_float"]["decimal_digits"] < 16.5
assert rep["decimal128"]["significant_digits"] == 34
assert isinstance(rep["numpy_longdouble"]["extends_float64"], bool)

# 8. the near-degenerate mode splitting: float64 cancels, decimal128 survives
ps = vm.mode_splitting_precision(coupling_ratio=1e-13, m=1.0, k=1.0)
assert np.isclose(ps["splitting"], 1e-13, rtol=1e-3)           # ~ the coupling
assert ps["float64_correct_digits"] < 6                       # cancellation ruins it
assert ps["decimal128_correct_digits"] > 20                   # 128-bit keeps ~26
assert ps["decimal128_correct_digits"] > ps["float64_correct_digits"] + 15
# less coupling => worse float64 cancellation (fewer correct digits)
loose = vm.mode_splitting_precision(coupling_ratio=1e-6)
assert loose["float64_correct_digits"] > ps["float64_correct_digits"]

# 9. kwarg bounds
for bad in (lambda: vm.second_difference_matrix(1),
            lambda: vm.second_difference_matrix(4, dx=0),
            lambda: vm.uniform_spring_chain(1),
            lambda: vm.uniform_spring_chain(4, m=-1),
            lambda: vm.particle_in_box(n_grid=5),
            lambda: vm.mode_splitting_precision(coupling_ratio=2.0)):
    try:
        bad()
        assert False, "expected ValueError"
    except ValueError:
        pass

print("test_vibration_modes: all checks passed")
