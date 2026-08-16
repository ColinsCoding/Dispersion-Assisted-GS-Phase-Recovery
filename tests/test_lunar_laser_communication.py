"""Test dgs/lunar_laser_communication.py: Earth-Moon light time, the
diffraction-limited link budget (reusing dgs.quantum_internet_link_budget
directly, not reimplemented), the idealized photon-limited rate ceiling
checked against the publicly documented LLCD 622 Mbps figure, and the
optical-vs-RF geometric loss comparison."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.lunar_laser_communication import (
    one_way_light_time_s, photon_energy_j, received_power_w,
    idealized_photon_limited_rate_bps, verify_against_llcd_public_figure,
    compare_optical_vs_rf_geometric_loss, MOON_MEAN_DISTANCE_M,
    MOON_PERIGEE_M, MOON_APOGEE_M, LLCD_PUBLIC_DOWNLINK_BPS,
)

# 1. one_way_light_time_s: mean-distance light time matches the
#    widely-cited ~1.28s figure, and perigee/apogee bracket it correctly
t_mean = one_way_light_time_s(MOON_MEAN_DISTANCE_M)
assert abs(t_mean - 1.282) < 0.01

t_perigee = one_way_light_time_s(MOON_PERIGEE_M)
t_apogee = one_way_light_time_s(MOON_APOGEE_M)
assert t_perigee < t_mean < t_apogee

try:
    one_way_light_time_s(-1.0)
    raise AssertionError("expected ValueError for negative distance")
except ValueError:
    pass

print("dgs.lunar_laser_communication: light-time checks passed")

# 2. photon_energy_j: known value at 1550nm (telecom C-band) --
#    E = hc/lambda ~ 1.28e-19 J
e_1550 = photon_energy_j(1550e-9)
assert abs(e_1550 - 1.282e-19) / 1.282e-19 < 1e-3

# shorter wavelength -> higher photon energy (inverse relationship)
e_800 = photon_energy_j(800e-9)
assert e_800 > e_1550

print("dgs.lunar_laser_communication: photon energy checks passed")

# 3. received_power_w: more tx power -> more received power (linear);
#    farther distance -> less received power (geometric loss grows)
p_near, loss_near = received_power_w(1.0, MOON_PERIGEE_M, 1550e-9, 0.1, 0.4)
p_far, loss_far = received_power_w(1.0, MOON_APOGEE_M, 1550e-9, 0.1, 0.4)
assert p_near > p_far
assert loss_near < loss_far

p_double, _ = received_power_w(2.0, MOON_MEAN_DISTANCE_M, 1550e-9, 0.1, 0.4)
p_single, _ = received_power_w(1.0, MOON_MEAN_DISTANCE_M, 1550e-9, 0.1, 0.4)
assert abs(p_double - 2 * p_single) / p_single < 1e-9

for bad in [dict(tx_power_w=-1.0, distance_m=MOON_MEAN_DISTANCE_M, wavelength_m=1550e-9,
                  tx_aperture_m=0.1, rx_aperture_m=0.4),
            dict(tx_power_w=1.0, distance_m=MOON_MEAN_DISTANCE_M, wavelength_m=1550e-9,
                  tx_aperture_m=0.1, rx_aperture_m=0.4, extra_loss_db=-1.0)]:
    try:
        received_power_w(**bad)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass

print("dgs.lunar_laser_communication: received-power checks passed")

# 4. idealized_photon_limited_rate_bps: more received power -> higher
#    rate; more photons required per bit -> lower rate (both directions)
rate_more_power = idealized_photon_limited_rate_bps(1e-9, 1550e-9, photons_per_bit=1.0)
rate_less_power = idealized_photon_limited_rate_bps(1e-10, 1550e-9, photons_per_bit=1.0)
assert rate_more_power > rate_less_power

rate_1ppb = idealized_photon_limited_rate_bps(1e-9, 1550e-9, photons_per_bit=1.0)
rate_4ppb = idealized_photon_limited_rate_bps(1e-9, 1550e-9, photons_per_bit=4.0)
assert abs(rate_1ppb - 4 * rate_4ppb) / rate_4ppb < 1e-9

print("dgs.lunar_laser_communication: idealized-rate checks passed")

# 5. verify_against_llcd_public_figure: the idealized ceiling should sit
#    ABOVE the real publicly documented rate (an idealization is optimistic)
#    but within a modest number of orders of magnitude -- not wildly off,
#    which would indicate a modeling error rather than a reasonable
#    idealization gap (coding overhead, detector inefficiency, margin)
llcd_check = verify_against_llcd_public_figure()
assert llcd_check["ceiling_above_real_rate"] is True
assert llcd_check["same_order_of_magnitude_regime"] is True
assert llcd_check["llcd_public_downlink_bps"] == LLCD_PUBLIC_DOWNLINK_BPS

print("dgs.lunar_laser_communication: LLCD public-figure comparison checks passed")

# 6. compare_optical_vs_rf_geometric_loss: at EQUAL aperture, optical
#    (1550nm) must have dramatically lower geometric loss than Ka-band RF
#    (9.4mm) -- the actual physical reason deep-space links go optical
rf_compare = compare_optical_vs_rf_geometric_loss(MOON_MEAN_DISTANCE_M, aperture_m=0.1)
assert rf_compare["optical_wins"] is True
assert rf_compare["optical_advantage_db"] > 50.0   # a large, not marginal, advantage

try:
    compare_optical_vs_rf_geometric_loss(MOON_MEAN_DISTANCE_M, aperture_m=-1.0)
    raise AssertionError("expected ValueError for negative aperture")
except ValueError:
    pass

print("dgs.lunar_laser_communication: optical-vs-RF comparison checks passed")
print("all dgs.lunar_laser_communication tests passed")
