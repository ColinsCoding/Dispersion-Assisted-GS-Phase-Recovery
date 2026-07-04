"""Test curl and divergence in modern (20th-century) physics: quantum
probability current continuity, the curl(grad)=0 identity underlying
gauge invariance, and the Aharonov-Bohm effect's topological curl-free-
but-not-gradient loophole."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import curl_div_modern_physics as cdmp

# 1. quantum continuity: d|psi|^2/dt + dJ/dx ~ 0 for a real time-evolved
#    free-particle wave packet, small relative to the scale of the terms
x = np.linspace(-5e-9, 5e-9, 400)
t = np.linspace(0, 1e-15, 60)
residual, rho, J = cdmp.verify_quantum_continuity(x, t, x0=0.0, k0=5e9, sigma0=1e-9)
interior = (slice(5, -5), slice(5, -5))
max_residual = np.max(np.abs(residual[interior]))
max_scale = np.max(np.abs(rho)) / (t[1] - t[0])
assert max_residual / max_scale < 1e-3

# 2. probability is normalized (integral of rho stays ~constant over time --
#    a real physical requirement, not something continuity alone guarantees
#    unless the wavefunction is correctly normalized)
integral_rho_over_time = np.trapezoid(rho, x, axis=1)
assert np.max(np.abs(integral_rho_over_time - integral_rho_over_time[0])) / integral_rho_over_time[0] < 1e-2

# 3. curl(grad(chi)) = 0 symbolically, for a GENERIC scalar chi
is_zero, curl_of_grad = cdmp.curl_of_gradient_is_zero_symbolic()
assert is_zero is True

# 4. solenoid vector potential: continuous at r=R (inside formula meets
#    outside formula)
R, B0 = 1e-6, 0.5
A_inside_at_R = cdmp.solenoid_vector_potential(np.array([R - 1e-12]), R, B0)[0]
A_outside_at_R = cdmp.solenoid_vector_potential(np.array([R + 1e-12]), R, B0)[0]
assert abs(A_inside_at_R - A_outside_at_R) / A_inside_at_R < 1e-4

# 5. curl(A) outside the solenoid is exactly 0 (matches B=0 there), at
#    several different radii -- not just one lucky point
r_test = np.linspace(1.5 * R, 10 * R, 10)
curl_outside = cdmp.curl_of_solenoid_A_outside(r_test, R, B0)
assert np.max(np.abs(curl_outside)) < 1e-6

# 6. circulation of A around loops of DIFFERENT radii (all outside the
#    solenoid) is IDENTICAL -- the topological signature: curl=0 along
#    the whole path, yet a nonzero, path-independent circulation
circ_a = cdmp.circulation_of_A_outside(1.5 * R, R, B0)
circ_b = cdmp.circulation_of_A_outside(4.0 * R, R, B0)
circ_c = cdmp.circulation_of_A_outside(9.0 * R, R, B0)
assert abs(circ_a - circ_b) / circ_a < 1e-9
assert abs(circ_b - circ_c) / circ_b < 1e-9

# 7. that circulation equals the TOTAL enclosed flux exactly
expected_flux = B0 * np.pi * R ** 2
assert abs(circ_a - expected_flux) / expected_flux < 1e-9

# 8. Aharonov-Bohm phase scales linearly with charge and with flux (both
#    physically required linear dependencies)
phase_1 = cdmp.aharonov_bohm_phase(1.0, B0, R)
phase_2 = cdmp.aharonov_bohm_phase(2.0, B0, R)
assert abs(phase_2 / phase_1 - 2.0) < 1e-9
phase_double_B = cdmp.aharonov_bohm_phase(1.0, 2 * B0, R)
assert abs(phase_double_B / phase_1 - 2.0) < 1e-9

# 9. input validation
for bad_call in [
    lambda: cdmp.free_particle_wavepacket(x, t, 0.0, 5e9, -1.0),
    lambda: cdmp.solenoid_vector_potential(np.array([-1.0]), R, B0),
    lambda: cdmp.solenoid_vector_potential(np.array([1e-6]), -1.0, B0),
    lambda: cdmp.curl_of_solenoid_A_outside(np.array([R * 0.5]), R, B0),  # inside, invalid
    lambda: cdmp.circulation_of_A_outside(R * 0.5, R, B0),  # inside, invalid
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.curl_div_modern_physics tests passed")
