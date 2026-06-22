"""Test engineering eigenvalues: torch normal modes of a mass-spring chain."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from dgs import eigen_modes as em

N, k, m = 12, 4.0, 0.5
D = em.chain_dynamical_matrix(N, k, m)

# 1. the dynamical matrix is symmetric (real eigenvalues guaranteed)
assert torch.allclose(D, D.T)

# 2. torch eigenvalue frequencies match the closed form to machine precision
omega, modes = em.normal_modes(D)
ref = em.chain_frequencies_analytic(N, k, m)
assert np.max(np.abs(omega.numpy() - ref)) < 1e-12

# 3. eigenvalues are real, non-negative, and ascending
w = omega.numpy()
assert np.all(w >= 0) and np.all(np.diff(w) > 0)

# 4. eigenvectors are the standing-sine mode shapes
for j in range(1, 4):
    v = modes[:, j - 1].numpy()
    shape = em.chain_mode_shape(N, j)
    corr = abs(np.corrcoef(v, shape)[0, 1])
    assert corr > 0.9999, (j, corr)

# 5. mode j has exactly j-1 interior nodes (sign changes) -- fundamental is node-free
for j in range(1, 6):
    v = modes[:, j - 1].numpy()
    sign_changes = np.sum(np.diff(np.sign(v)) != 0)
    assert sign_changes == j - 1, (j, sign_changes)

# 6. frequencies scale as sqrt(k/m): quadrupling k doubles every frequency
D2 = em.chain_dynamical_matrix(N, 4 * k, m)
omega2, _ = em.normal_modes(D2)
assert np.allclose(omega2.numpy(), 2 * w, rtol=1e-9)

# 7. the spectrum saturates (acoustic-branch dispersion): omega_max < 2 sqrt(k/m)*...
#    the highest mode approaches 2 sqrt(k/m) but never exceeds it
assert w[-1] < 2 * np.sqrt(k / m) + 1e-9

print(f"TEST PASS  (torch eigh frequencies == analytic to {np.max(np.abs(w-ref)):.1e}; "
      f"modes are standing sines; mode j has j-1 nodes; omega ~ sqrt(k/m); "
      f"fundamental T1={2*np.pi/w[0]:.2f}s)")
