"""Test path_integral_qkd: free-particle composition law, PIMC harmonic
oscillator convergence to the exact QHO thermal values, BB84 QBER vs analytic."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import path_integral_qkd as pq

# 1. free-particle propagator composition law: splitting time into two slices
#    and integrating out the midpoint reproduces the one-slice analytic kernel
K_two_slice, K_analytic = pq.free_particle_propagator_two_slice_check(2.0, 0.0, 1.0)
assert abs(K_two_slice - K_analytic) / abs(K_analytic) < 1e-6

# 2. analytic QHO thermal values reduce to the T->0 ground state at large beta
big_beta = 50.0
assert abs(pq.qho_thermal_energy(big_beta, omega=1.0) - 0.5) < 1e-6     # E_0 = hbar*omega/2
assert abs(pq.qho_thermal_x2(big_beta, m=1.0, omega=1.0) - 0.5) < 1e-6  # <x^2>_0 = hbar/(2 m omega)

# 3. PIMC harmonic oscillator converges toward the analytic thermal values
res = pq.pimc_harmonic_oscillator(beta=2.0, n_slices=16, n_sweeps=20000, burn_in=2000, step_size=1.0, seed=3)
assert abs(res["x2_mc"] - res["x2_analytic"]) / res["x2_analytic"] < 0.15
assert abs(res["H_mc"] - res["H_analytic"]) / res["H_analytic"] < 0.15

# 4. BB84 with no eavesdropper: zero QBER
clean = pq.bb84_intercept_resend_qber(n_bits=40000, eavesdrop=False, seed=0)
assert clean["qber_mc"] == 0.0

# 5. BB84 with intercept-resend: QBER near the analytic 25%
eve = pq.bb84_intercept_resend_qber(n_bits=40000, eavesdrop=True, seed=0)
assert abs(eve["qber_mc"] - 0.25) < 0.02

print("test_path_integral_qkd: all checks passed")
