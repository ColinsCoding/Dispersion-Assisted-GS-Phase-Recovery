"""Test quantum_plots: square well/HO normalization, uncertainty-chirp trend,
Legendre/Bessel special functions, hard-sphere phase shifts, Born cross section."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import quantum_plots as qp

# 1. square well eigenfunctions are normalized
x = np.linspace(0, 1, 400)
psi = qp.square_well_eigenfunction(3, x)
assert abs(np.trapezoid(psi ** 2, x) - 1.0) < 1e-3

# 2. harmonic oscillator eigenfunctions are normalized
xo = np.linspace(-8, 8, 2000)
psi_ho = qp.harmonic_oscillator_eigenfunction(3, xo)
assert abs(np.trapezoid(psi_ho ** 2, xo) - 1.0) < 1e-3

# 3. a superposition snapshot preserves total probability
snap = qp.superposition_snapshot([1, 2j, -1], [1, 2, 3], x, t=0.7)
assert abs(np.trapezoid(np.abs(snap) ** 2, x) - 1.0) < 1e-3

# 4. Fourier uncertainty grows away from the unchirped (minimum-product) pulse
products = qp.uncertainty_vs_chirp([0.0, 10.0, 25.0])
assert products[0] < products[-1], products

# 5. Legendre P_0=1, P_1=x
xl = np.linspace(-1, 1, 11)
assert np.allclose(qp.legendre(0, xl), 1.0)
assert np.allclose(qp.legendre(1, xl), xl)

# 6. spherical Bessel j_0(x) = sin(x)/x
xb = np.linspace(0.5, 10, 20)
assert np.allclose(qp.spherical_bessel_jl(0, xb), np.sin(xb) / xb)

# 7. hard-sphere phase shifts are finite for all ka
delta = qp.hard_sphere_phase_shift(2, np.linspace(0.2, 8, 30))
assert np.all(np.isfinite(delta))

# 8. Born cross section is non-negative
theta = np.linspace(0.01, np.pi, 30)
sigma = qp.born_square_well_cross_section(theta, k=1.5, V0=2.0, a=0.8)
assert np.all(sigma >= 0)

print("test_quantum_plots: all checks passed")
