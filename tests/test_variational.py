"""Test the variational principle (Griffiths Ch.7): <H> >= E_ground, minimized at truth."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import quantum as q

# Harmonic oscillator (hbar=m=omega=1): H = -1/2 d^2/dx^2 + 1/2 x^2, ground state E0 = 1/2
x = np.linspace(-12, 12, 6000)
V = 0.5 * x**2

def gaussian(alpha):
    return np.exp(-alpha * x**2)

# 1. the exact ground state is a Gaussian with alpha=1/2 -> <H> = E0 = 1/2
assert abs(q.variational_energy(gaussian(0.5), x, V) - 0.5) < 1e-3

# 2. variational bound: <H> >= E0 for EVERY trial alpha (never beats the truth)
alphas = np.linspace(0.15, 1.5, 40)
energies = np.array([q.variational_energy(gaussian(a), x, V) for a in alphas])
assert np.all(energies >= 0.5 - 1e-3), energies.min()

# 3. the minimum over alpha sits at the true alpha = 1/2
a_best = alphas[np.argmin(energies)]
assert abs(a_best - 0.5) < 0.05, a_best
assert abs(energies.min() - 0.5) < 1e-3

# 4. a deliberately-bad trial (too wide / too narrow) gives a HIGHER energy
assert q.variational_energy(gaussian(0.15), x, V) > 0.5
assert q.variational_energy(gaussian(1.5), x, V) > 0.5

print(f"TEST PASS  (variational: min <H>={energies.min():.4f} at alpha={a_best:.3f} "
      f"= E0=1/2; bound <H> >= E0 holds for all trials)")
