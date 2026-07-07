"""Test dgs.quantum_oscillator: the ladder E_n=(n+1/2)hbar*omega reached three
ways -- numeric diagonalization of the trapped-atom Hamiltonian, the ladder-
operator algebra ([a,a-dagger]=1, a-dagger adds a quantum), and thermal occupation
crossing from the zero-point to the classical kT -- plus a C partition-sum check."""
import sys, pathlib, math, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import quantum_oscillator as qo

# 1. the ladder and its constant quantum of energy hbar*omega
assert np.allclose(qo.energy_levels(4), [0.5, 1.5, 2.5, 3.5, 4.5])
assert np.allclose(np.diff(qo.energy_levels(10)), 1.0)          # equally spaced
assert qo.level_spacing(omega=2.0) == 2.0                       # gap = hbar*omega
assert qo.zero_point_energy(omega=3.0) == 1.5                   # (1/2)hbar*omega
assert np.allclose(qo.energy_levels(3, omega=2.0), [1, 3, 5, 7])

# 2. put the atom in the well: FD eigenvalues land on (n+1/2)hbar*omega
E = qo.numeric_energies(n_levels=6, n_grid=1000, x_max=9.0)
assert np.allclose(E[:4], [0.5, 1.5, 2.5, 3.5], atol=2e-3)      # low levels tight
assert np.allclose(E, qo.energy_levels(5), rtol=2e-3)          # whole set
assert np.allclose(np.diff(E), 1.0, atol=3e-3)                 # spacing = hbar*omega

# 3. ladder-operator algebra
N = 12
a = qo.annihilation_operator(N)
adag = qo.creation_operator(N)
assert np.allclose(adag, a.T)
assert np.allclose(qo.commutator_top_block(N), np.eye(N - 1))   # [a, a-dagger] = 1
# a-dagger adds a quantum: a-dagger|n> = sqrt(n+1)|n+1>; a|n> = sqrt(n)|n-1>
for n in (0, 2, 5):
    en = np.eye(N)[n]
    assert np.allclose(adag @ en, math.sqrt(n + 1) * np.eye(N)[n + 1])
    if n > 0:
        assert np.allclose(a @ en, math.sqrt(n) * np.eye(N)[n - 1])
# number operator counts quanta; ladder Hamiltonian == the analytic ladder
assert np.allclose(np.diag(qo.number_operator(N)), np.arange(N))
H = qo.hamiltonian_from_ladder(6, omega=1.0)
assert np.allclose(np.diag(H), qo.energy_levels(5))
assert np.allclose(np.sort(np.linalg.eigvalsh(H)), qo.energy_levels(5))

# 4. thermal occupation and the quantum -> classical crossover
assert qo.mean_occupation(1.0, 0.05) < 1e-8                    # frozen near T=0
assert np.isclose(qo.mean_occupation(1.0, 1.0), 1/(math.e - 1))   # Bose-Einstein
assert np.isclose(qo.mean_energy(1.0, 0.01), 0.5, atol=1e-3)   # -> zero-point
assert np.isclose(qo.mean_energy(1.0, 1000.0) / 1000.0, 1.0, atol=1e-3)  # -> kT (classical)
# partition function closed form == explicit sum over the ladder
x = 0.7                                                         # hbar*omega/kT
Z_sum = np.sum(np.exp(-(np.arange(400) + 0.5) * x))
assert np.isclose(qo.partition_function(1.0, 1.0 / x), Z_sum, rtol=1e-9)

# 5. C partition sum matches the closed forms (gcc-guarded)
if qo.gcc_available():
    with tempfile.TemporaryDirectory() as d:
        c = qo.compile_and_run_c(d)
    assert np.isclose(c["Z"], qo.partition_function(1.0, 1.0), rtol=1e-6)   # x=1
    assert np.isclose(c["mean_energy_over_hw"], 0.5 / math.tanh(0.5), rtol=1e-6)
    print("test_quantum_oscillator: all checks passed (incl. C sum)")
else:
    print("test_quantum_oscillator: all checks passed (C sum skipped: no gcc)")

# 6. kwarg bounds
for bad in (lambda: qo.energy_levels(-1),
            lambda: qo.level_spacing(omega=0),
            lambda: qo.numeric_energies(n_levels=0),
            lambda: qo.annihilation_operator(1),
            lambda: qo.mean_occupation(1.0, 0),
            lambda: qo.partition_function(0, 1.0)):
    try:
        bad(); assert False
    except ValueError:
        pass
