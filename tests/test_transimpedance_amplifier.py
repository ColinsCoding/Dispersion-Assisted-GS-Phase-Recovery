"""Test dgs.transimpedance_amplifier: responsivity, transimpedance gain, the R_f*f_3dB budget,
the shot/thermal/amplifier noise terms, SNR/NEP/sensitivity, and the noise-limit regimes."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from dgs import transimpedance_amplifier as tia

q, k = tia.Q_E, tia.K_B

# 1. responsivity: ideal ~1.25 A/W at 1550 nm, scales with eta and lambda
assert math.isclose(tia.responsivity(1550), 1.249, abs_tol=2e-3)
assert math.isclose(tia.responsivity(1550, 0.8), 0.8 * tia.responsivity(1550), rel_tol=1e-12)
assert tia.responsivity(1550) > tia.responsivity(850)          # longer lambda -> higher R
# R = eta q lambda/(h c) closed form
assert math.isclose(tia.responsivity(1310, 0.9),
                    0.9 * q * 1310e-9 / (tia.H_PL * tia.C_LIGHT), rel_tol=1e-12)

# 2. photocurrent and output voltage
R = tia.responsivity(1550, 0.8)
assert math.isclose(tia.photocurrent(1e-6, R), R * 1e-6, rel_tol=1e-12)
assert math.isclose(tia.output_voltage(2e-6, 1e4), 2e-6 * 1e4, rel_tol=1e-12)   # V = I R_f

# 3. bandwidth and the fixed gain-bandwidth budget R_f * f_3dB = 1/(2 pi C)
C = 0.5e-12
assert math.isclose(tia.bandwidth_3db(1e4, C), 1/(2*np.pi*1e4*C), rel_tol=1e-12)
budget = tia.gain_bandwidth_product(C)
for R_f in (1e3, 1e4, 1e5, 1e6):
    assert math.isclose(R_f * tia.bandwidth_3db(R_f, C), budget, rel_tol=1e-9)
# bigger R_f -> proportionally smaller bandwidth
assert math.isclose(tia.bandwidth_3db(1e3, C) / tia.bandwidth_3db(1e4, C), 10.0, rel_tol=1e-9)

# 4. noise terms match closed forms and their scalings
B = 100e6
assert math.isclose(tia.shot_noise_current(1e-3, B), math.sqrt(2*q*1e-3*B), rel_tol=1e-12)
assert math.isclose(tia.thermal_noise_current(1e4, B), math.sqrt(4*k*300*B/1e4), rel_tol=1e-12)
assert math.isclose(tia.amplifier_noise_current(2e-9, C, B),
                    2*np.pi*C*2e-9*math.sqrt(B**3/3), rel_tol=1e-12)
# thermal noise falls as 1/sqrt(R_f): 100x R_f -> 10x quieter
assert math.isclose(tia.thermal_noise_current(1e2, B)/tia.thermal_noise_current(1e4, B),
                    10.0, rel_tol=1e-9)
# shot noise grows as sqrt(current)
assert math.isclose(tia.shot_noise_current(4e-3, B)/tia.shot_noise_current(1e-3, B),
                    2.0, rel_tol=1e-9)

# 5. total noise is the quadrature sum; SNR rises with optical power
P, R_f, e_n = 1e-6, 1e4, 2e-9
i_tot = tia.total_noise_current(P, R, R_f, C, B, e_n=e_n)
i_shot = tia.shot_noise_current(tia.photocurrent(P, R), B)
i_therm = tia.thermal_noise_current(R_f, B)
i_amp = tia.amplifier_noise_current(e_n, C, B)
assert math.isclose(i_tot, math.sqrt(i_shot**2 + i_therm**2 + i_amp**2), rel_tol=1e-12)
assert tia.snr(2e-6, R, R_f, C, B, e_n=e_n) > tia.snr(1e-6, R, R_f, C, B, e_n=e_n)

# 6. shot-noise-limited vs thermal-limited regimes
# huge R_f + strong signal -> thermal negligible -> shot dominates (near shot limit)
strong = tia.total_noise_current(1e-3, R, 1e6, C, B)   # 1 mW, R_f=1M, no amp noise
shot_only = tia.shot_noise_current(tia.photocurrent(1e-3, R), B)
assert math.isclose(strong, shot_only, rel_tol=0.05)   # within 5% of shot limit
# tiny signal + small R_f -> thermal dominates
weak_total = tia.total_noise_current(1e-12, R, 1e3, C, B)
assert math.isclose(weak_total, tia.thermal_noise_current(1e3, B), rel_tol=0.05)

# 7. SNR(power) = 1 exactly at the NEP optical power (P-independent noise regime)
nep = tia.noise_equivalent_power(R, R_f, C, B, e_n=e_n)
# at P = NEP, I_ph = i_n(P=0); with signal shot added SNR is slightly under 1
assert tia.snr(nep, R, R_f, C, B, e_n=e_n) <= 1.0 + 1e-9
assert tia.snr(nep, R, R_f, C, B, e_n=e_n) > 0.9
assert nep > 0

# 8. sensitivity: the returned power actually delivers the requested SNR
for S in (5.0, 7.0, 10.0):
    Pmin = tia.sensitivity(S, R, R_f, C, B, e_n=e_n)
    assert math.isclose(tia.snr(Pmin, R, R_f, C, B, e_n=e_n), S, rel_tol=1e-6)
# a higher SNR target needs more power
assert tia.sensitivity(10, R, R_f, C, B) > tia.sensitivity(5, R, R_f, C, B)

# 9. kwarg bounds
for bad in (lambda: tia.responsivity(0),
            lambda: tia.responsivity(1550, 1.5),
            lambda: tia.bandwidth_3db(0, C),
            lambda: tia.thermal_noise_current(1e3, 0),
            lambda: tia.sensitivity(0, R, R_f, C, B)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_transimpedance_amplifier: all checks passed")
