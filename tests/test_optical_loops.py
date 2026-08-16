"""Test dgs/optical_loops.py: ring-resonator steady state (finesse, critical
coupling), the recirculating fiber loop's dispersion-multiplication identity
(checked against dgs.gs_core.disperse directly, not just trusted algebra),
and the ring-as-single-pole-IIR-filter recursion converging to the closed
form."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math
import numpy as np
from dgs.optical_loops import (
    round_trip_phase, cross_coupling_from_self, through_port_transmission,
    circulating_buildup_factor, ring_finesse, ring_FWHM_phase,
    critical_coupling_residual, round_trip_net_dB, power_survival_from_dB,
    loop_threshold_gain_dB, accumulated_dispersion, simulate_recirculating_loop,
    verify_accumulated_dispersion_equals_single_pass, iir_pole_from_ring,
    simulate_ring_buildup_recursion, verify_recursion_converges_to_closed_form,
)

# 1. round_trip_phase / cross_coupling_from_self: basic relationships
assert abs(round_trip_phase(f=1e9, FSR=1e9) - 2 * math.pi) < 1e-12
assert abs(cross_coupling_from_self(1.0) - 0.0) < 1e-12   # t=1 -> no coupling -> kappa=0
assert abs(cross_coupling_from_self(0.6) - 0.8) < 1e-12   # 0.6-0.8 Pythagorean triple

for bad in (0.0, -0.5, 1.5):
    try:
        cross_coupling_from_self(bad)
        raise AssertionError(f"expected ValueError for t={bad}")
    except ValueError:
        pass

# 2. through_port_transmission: T(phi=0) with t=a is EXACTLY 0 (critical coupling)
for a in (0.7, 0.85, 0.99):
    resid = critical_coupling_residual(a)
    assert resid < 1e-20, f"critical coupling residual not ~0: {resid}"

# T=1 (all power reflected) when a=1 (lossless ring) at phi = pi (off resonance,
# t - a*e^{i*pi} = t + a, and with a=1 the denominator/numerator have equal magnitude)
# simpler robust check: on resonance (phi=0) with NO loss (a=1), T(0) = ((t-1)/(1-t))^2 = 1
T_no_loss_on_res = through_port_transmission(t=0.9, a=1.0, phi=0.0)
assert abs(T_no_loss_on_res - 1.0) < 1e-12

# 3. ring_finesse / ring_FWHM_phase: higher t*a -> higher finesse, narrower FWHM
F_low = ring_finesse(t=0.7, a=0.95)
F_high = ring_finesse(t=0.95, a=0.98)
assert F_high > F_low
assert ring_FWHM_phase(t=0.95, a=0.98) < ring_FWHM_phase(t=0.7, a=0.95)

# 4. circulating_buildup_factor: on resonance, |buildup|^2 must exceed off-resonance
t, a = 0.9, 0.98
build_on_res = abs(circulating_buildup_factor(t, a, phi=0.0))**2
build_off_res = abs(circulating_buildup_factor(t, a, phi=math.pi))**2
assert build_on_res > build_off_res

# 5. round_trip_net_dB / power_survival_from_dB / loop_threshold_gain_dB:
#    at threshold gain, net round-trip dB is exactly 0 -> power survival = 1
g_th = loop_threshold_gain_dB(fiber_loss_dB_per_km=0.2, length_km=10.0, coupler_loss_dB=1.0)
net_dB_at_threshold = round_trip_net_dB(0.2, 10.0, 1.0, amplifier_gain_dB=g_th)
assert abs(net_dB_at_threshold) < 1e-12
assert abs(power_survival_from_dB(net_dB_at_threshold) - 1.0) < 1e-12
assert abs(power_survival_from_dB(0.0) - 1.0) < 1e-12
assert abs(power_survival_from_dB(10.0) - 10.0) < 1e-9   # +10 dB = 10x power

for bad in [(-1, 10, 1), (0.2, -1, 1), (0.2, 10, -1)]:
    try:
        round_trip_net_dB(*bad, amplifier_gain_dB=0.0)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass

# 6. accumulated_dispersion: exact linearity
assert accumulated_dispersion(D_per_pass=-50.0, N=12) == -600.0
assert accumulated_dispersion(D_per_pass=123.4, N=0) == 0.0
try:
    accumulated_dispersion(-50.0, N=-1)
    raise AssertionError("expected ValueError for N<0")
except ValueError:
    pass

# 7. simulate_recirculating_loop: N=0 returns just [E] unchanged; length is N+1
rng = np.random.default_rng(0)
E0 = np.exp(1j * rng.uniform(0, 2 * np.pi, 128))
snaps0 = simulate_recirculating_loop(E0, D_per_pass=-50.0, N=0)
assert len(snaps0) == 1
assert np.allclose(snaps0[0], E0)
snaps5 = simulate_recirculating_loop(E0, D_per_pass=-50.0, N=5, net_dB_per_pass=-1.0)
assert len(snaps5) == 6
# lossy loop (net_dB<0): power must strictly decrease each pass
powers = [float(np.sum(np.abs(s)**2)) for s in snaps5]
assert all(powers[i] > powers[i + 1] for i in range(len(powers) - 1))

# 8. verify_accumulated_dispersion_equals_single_pass: the core physics
#    identity this module exists to demonstrate -- must match to near
#    machine precision, checked at several (D_per_pass, N)
for D_per_pass, N in [(-50.0, 12), (137.0, 7), (-5000.0, 3), (10.0, 1)]:
    check = verify_accumulated_dispersion_equals_single_pass(E0, D_per_pass, N)
    assert check["matches"] is True, f"mismatch at D={D_per_pass}, N={N}: {check}"
    assert check["max_abs_diff"] < 1e-8

# 9. iir_pole_from_ring: |pole| = t*a always < 1 (t,a in (0,1]) -> unconditionally stable
for t, a, phi in [(0.9, 0.98, 0.3), (0.5, 0.5, 1.0), (1.0, 1.0, 0.0)]:
    z0 = iir_pole_from_ring(t, a, phi)
    assert abs(abs(z0) - t * a) < 1e-12
    if t < 1.0 or a < 1.0:
        assert abs(z0) < 1.0

# 10. simulate_ring_buildup_recursion: E_circ[0] = 0 always (ring starts empty)
E_circ = simulate_ring_buildup_recursion(E_in=1.0, t=0.9, a=0.98, phi=0.0, n_round_trips=50)
assert E_circ[0] == 0.0
assert len(E_circ) == 51

# 11. verify_recursion_converges_to_closed_form: must converge for a lossy
#     (t*a < 1) ring, and converge FASTER (fewer round trips needed) for
#     lower finesse than higher finesse -- both checked
conv_low_F = verify_recursion_converges_to_closed_form(1.0, t=0.5, a=0.9, phi=0.0, n_round_trips=100)
conv_high_F = verify_recursion_converges_to_closed_form(1.0, t=0.95, a=0.99, phi=0.0, n_round_trips=500)
assert conv_low_F["converged"] is True
assert conv_high_F["converged"] is True
assert conv_low_F["settling_round_trips_estimate"] < conv_high_F["settling_round_trips_estimate"]

# with too few round trips, a high-finesse ring should NOT yet have converged
conv_high_F_early = verify_recursion_converges_to_closed_form(1.0, t=0.95, a=0.99, phi=0.0,
                                                                n_round_trips=3, tol=1e-6)
assert conv_high_F_early["converged"] is False

print("all dgs.optical_loops tests passed")
