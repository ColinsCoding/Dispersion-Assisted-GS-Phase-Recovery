"""Test the quantum-internet link budget: real UC Merced/Riverside geography,
fiber's exponential-in-distance loss vs. free-space diffraction's power-law
loss, and quantum-repeater spacing."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import quantum_internet_link_budget as qi

# 1. real geography: UC Merced to UC Riverside great-circle distance
UCM = (37.3661, -120.4269)
UCR = (33.9737, -117.3281)
d = qi.haversine_distance_km(*UCM, *UCR)
assert abs(d - 469.7) < 1.0, f"expected ~469.7 km, got {d}"

# symmetry: order of points shouldn't matter
d_rev = qi.haversine_distance_km(*UCR, *UCM)
assert abs(d - d_rev) < 1e-9

# same point -> zero distance
assert qi.haversine_distance_km(*UCM, *UCM) < 1e-9

# 2. fiber route distance is always >= great-circle
fiber_km = qi.fiber_route_distance_km(d, route_factor=1.4)
assert fiber_km > d
assert abs(fiber_km - d * 1.4) < 1e-9

# 3. fiber loss is LINEAR in distance (the defining exponential-power property)
loss_100 = qi.fiber_loss_db(100.0)
loss_200 = qi.fiber_loss_db(200.0)
assert abs(loss_200 - 2 * loss_100) < 1e-9
assert abs(loss_100 - 100.0 * qi.FIBER_ATTEN_DB_PER_KM_1550NM) < 1e-9

# 4. dB <-> transmittance conversion: 10 dB loss = factor of 10 in power
assert abs(qi.transmittance_from_db(0.0) - 1.0) < 1e-12
assert abs(qi.transmittance_from_db(10.0) - 0.1) < 1e-9
assert abs(qi.transmittance_from_db(20.0) - 0.01) < 1e-9

# 5. fiber transit time: v = c/n inside the core, slower than vacuum c
t_fiber = qi.fiber_transit_time_s(fiber_km)
t_vacuum = (fiber_km * 1e3) / qi.C_SI
assert t_fiber > t_vacuum
assert abs(t_fiber / t_vacuum - qi.FIBER_CORE_INDEX_1550NM) < 1e-9

# 6. entangled-pair detection rate: two independent lossy links multiply
#    (loss budgets ADD in dB, transmittances MULTIPLY)
rate = qi.entangled_pair_detection_rate_hz(1e6, one_way_loss_db=20.0, detector_efficiency=0.5)
T_link = qi.transmittance_from_db(20.0)
expected = 1e6 * (T_link * 0.5) ** 2
assert abs(rate - expected) < 1e-6
# doubling the one-way loss (in dB) should reduce the rate by (10x)^2 = 100x per photon pair...
# actually doubling loss_db doubles total dB, meaning transmittance^2 -> transmittance^4 equivalent scaling check:
rate_double_loss = qi.entangled_pair_detection_rate_hz(1e6, one_way_loss_db=40.0, detector_efficiency=0.5)
assert rate_double_loss < rate  # more loss must mean fewer detected pairs
assert rate_double_loss / rate < 1e-2  # 20 dB extra loss per link, squared, is a huge additional suppression

# 7. diffraction divergence: larger transmit aperture -> smaller divergence angle
theta_small_tx = qi.diffraction_divergence_half_angle_rad(1550e-9, 0.1)
theta_large_tx = qi.diffraction_divergence_half_angle_rad(1550e-9, 1.0)
assert theta_large_tx < theta_small_tx

# 8. free-space geometric loss grows with distance but as a POWER LAW,
#    not exponentially: doubling distance should ~quadruple the loss FACTOR
#    (add ~6.02 dB, i.e. 20*log10(2)), not double-then-square the dB value
loss_L = qi.free_space_geometric_loss_db(500e3, 1550e-9, 0.3, 1.0)
loss_2L = qi.free_space_geometric_loss_db(1000e3, 1550e-9, 0.3, 1.0)
assert abs((loss_2L - loss_L) - 20 * np.log10(2)) < 1e-6

# 9. head-to-head comparison for the real Merced-Riverside distance:
#    fiber's exponential loss over ~470-660 km should be catastrophic
#    (>100 dB), while a satellite relay at LEO altitude should win
result = qi.compare_fiber_vs_satellite(d)
assert result["fiber_loss_db"] > 100.0, "fiber loss over this distance should be enormous"
assert result["satellite_wins"] is True or result["satellite_loss_db"] < result["fiber_loss_db"]
assert result["fiber_transmittance"] < 1e-10

# 10. quantum repeater spacing: more segments -> lower per-span loss,
#     and the reconstructed total should match the original distance
n_seg, span_km, span_loss = qi.repeater_spacing_for_budget_km(d, max_span_loss_db=20.0)
assert n_seg >= 1
assert abs(n_seg * span_km - d) < 1e-6
assert span_loss <= 20.0 + 1e-9
# a tighter per-span budget requires MORE segments
n_seg_tight, _, _ = qi.repeater_spacing_for_budget_km(d, max_span_loss_db=5.0)
assert n_seg_tight >= n_seg

# 11. input validation
for bad_call in [
    lambda: qi.fiber_route_distance_km(-1.0),
    lambda: qi.fiber_route_distance_km(100.0, route_factor=0.5),
    lambda: qi.fiber_loss_db(-1.0),
    lambda: qi.fiber_loss_db(10.0, alpha_db_per_km=0.0),
    lambda: qi.fiber_transit_time_s(-1.0),
    lambda: qi.fiber_transit_time_s(10.0, n_fiber=0.5),
    lambda: qi.entangled_pair_detection_rate_hz(-1.0, 10.0),
    lambda: qi.entangled_pair_detection_rate_hz(1e6, 10.0, detector_efficiency=1.5),
    lambda: qi.diffraction_divergence_half_angle_rad(-1e-6, 0.3),
    lambda: qi.diffraction_divergence_half_angle_rad(1550e-9, 0.0),
    lambda: qi.free_space_geometric_loss_db(-1.0, 1550e-9, 0.3, 1.0),
    lambda: qi.compare_fiber_vs_satellite(-1.0),
    lambda: qi.repeater_spacing_for_budget_km(-1.0, 20.0),
    lambda: qi.repeater_spacing_for_budget_km(100.0, 0.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.quantum_internet_link_budget tests passed")
print(f"UC Merced -> UC Riverside great-circle distance: {d:.1f} km")
print(f"realistic fiber route: {fiber_km:.0f} km, loss: {result['fiber_loss_db']:.1f} dB "
      f"(transmittance {result['fiber_transmittance']:.2e})")
print(f"satellite relay ({500.0:.0f} km altitude): loss {result['satellite_loss_db']:.1f} dB "
      f"(transmittance {result['satellite_transmittance']:.2e})")
print(f"satellite wins: {result['satellite_wins']}")
print(f"repeater spacing for 20 dB/span budget: {n_seg} segments of {span_km:.1f} km each")
