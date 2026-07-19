"""Test photosynthesis_energy_transfer: Forster rate at R0, rate-matrix
eigenmodes, population conservation/decay, and the gradient-descent inverse
fit recovering a known FRET rate from a noisy fluorescence decay curve."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import photosynthesis_energy_transfer as pet

# 1. Forster rate equals 1/tau_donor exactly at r = R0 (the defining property)
assert abs(pet.forster_rate(r=5.0, r0=5.0, tau_donor=2.0) - 0.5) < 1e-9

# 2. rate-matrix eigenvalues are real and non-positive for this 2-level system
K = pet.build_rate_matrix({(0, 1): 2.0}, decay_rates=[0.1, 0.3])
eigvals, _ = pet.eigen_decomposition_modes(K)
assert np.all(eigvals.real <= 1e-9)
assert np.allclose(eigvals.imag, 0)

# 3. population dynamics: starts at p0, never exceeds 1, monotonically decays
t = np.linspace(0, 5, 50)
p0 = [1.0, 0.0]
p_t = pet.solve_population_dynamics(K, p0, t)
total = p_t.sum(axis=1)
assert abs(total[0] - 1.0) < 1e-6
assert np.all(np.diff(total) <= 1e-9)   # monotonically non-increasing
assert np.all(total <= 1.0 + 1e-9)

# 4. fluorescence signal is non-negative
F = pet.fluorescence_signal(p_t, radiative_rates=[0.1, 0.3])
assert np.all(F >= 0)

# 5. the ML inverse fit recovers a known FRET rate from noisy data
true_k = 2.0
K_true = pet.build_rate_matrix({(0, 1): true_k}, decay_rates=[0.1, 0.3])
p_t_true = pet.solve_population_dynamics(K_true, p0, t)
F_true = pet.fluorescence_signal(p_t_true, [0.1, 0.3])
rng = np.random.default_rng(0)
F_noisy = F_true + 0.01 * rng.standard_normal(F_true.shape)

fit = pet.fit_transfer_rate_from_decay(t, F_noisy, p0, [0.1, 0.3], n_steps=500, lr=0.1)
assert abs(fit["k_fit"] - true_k) / true_k < 0.1

print("test_photosynthesis_energy_transfer: all checks passed")
