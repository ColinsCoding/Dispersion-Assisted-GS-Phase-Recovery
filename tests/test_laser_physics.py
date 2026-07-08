"""Test dgs.laser_physics: thermal populations never invert (Boltzmann < 1, and a
real inversion is a NEGATIVE temperature), the Einstein A/B ~ nu^3 and the tiny
stimulated/spontaneous ratio at optical frequencies, exponential gain <-> decibels,
and the threshold condition g_th = alpha + ln(1/(R1 R2))/2L (round-trip gain = 1)."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import laser_physics as lp

h, k, c = lp.H_PLANCK, lp.K_BOLTZ, lp.C_LIGHT
freq = c / 632.8e-9                      # HeNe-like
dE = h * freq

# 1. Boltzmann: N2/N1 < 1 for T>0, and -> 1 as T -> infinity
assert lp.boltzmann_population_ratio(dE, 300) < 1e-30      # optical, room T: ~0
assert lp.boltzmann_population_ratio(dE, 300) < lp.boltzmann_population_ratio(dE, 3000)
assert lp.boltzmann_population_ratio(dE, 1e12) > 0.99      # very hot -> nearly equal
assert lp.boltzmann_population_ratio(dE, 300) == math.exp(-dE / (k * 300))

# 2. inverting the Boltzmann law: round-trip, and inversion = NEGATIVE temperature
T = 500.0
ratio = lp.boltzmann_population_ratio(dE, T)
assert math.isclose(lp.temperature_from_ratio(ratio, dE), T, rel_tol=1e-9)
assert lp.temperature_from_ratio(2.0, dE) < 0             # N2>N1 -> T < 0
assert lp.temperature_from_ratio(0.5, dE) > 0             # normal thermal
assert lp.temperature_from_ratio(1.0, dE) == math.inf     # equal populations
assert lp.is_inverted(N2=10, N1=5) and not lp.is_inverted(N2=3, N1=8)

# 3. Einstein A/B = 8 pi h nu^3 / c^3, scaling as nu^3
assert math.isclose(lp.einstein_A_over_B(freq), 8*math.pi*h*freq**3/c**3)
assert math.isclose(lp.einstein_A_over_B(2*freq) / lp.einstein_A_over_B(freq), 8.0)
# stimulated/spontaneous is tiny at optical/room T, and ~ Boltzmann for h nu >> kT
assert lp.stimulated_over_spontaneous(freq, 300) < 1e-30
assert math.isclose(lp.stimulated_over_spontaneous(freq, 300),
                    lp.boltzmann_population_ratio(dE, 300), rel_tol=1e-6)

# 4. gain: sign, exponential growth, and the dB (log) identities
assert lp.gain_coefficient(1e-20, N2=1e24, N1=1e23) > 0     # inverted -> amplifies
assert lp.gain_coefficient(1e-20, N2=1e23, N1=1e24) < 0     # not inverted -> absorbs
assert math.isclose(lp.intensity_after_gain(1.0, 0.05, 0.3), math.exp(0.05*0.3))
assert math.isclose(lp.gain_dB(100e-3, 1e-3), 20.0)         # 100x power = 20 dB
# the dB gain of a gain medium matches 10 log10 of its intensity ratio
g, L = 0.05, 0.3
assert math.isclose(lp.small_signal_gain_dB(g, L),
                    lp.gain_dB(lp.intensity_after_gain(1.0, g, L), 1.0))

# 5. threshold condition and the round-trip identity
alpha, R1, R2, L = 0.01, 1.0, 0.98, 0.3
gth = lp.threshold_gain(alpha, R1, R2, L)
assert math.isclose(gth, alpha + math.log(1/(R1*R2))/(2*L))
assert math.isclose(lp.round_trip_gain(gth, alpha, R1, R2, L), 1.0)   # self-sustaining
# perfect mirrors leave only the internal loss; better mirrors lower the threshold
assert math.isclose(lp.threshold_gain(alpha, 1.0, 1.0, L), alpha)
assert lp.threshold_gain(alpha, 1.0, 0.99, L) < lp.threshold_gain(alpha, 1.0, 0.90, L)
# below threshold the round trip loses energy, above it gains
assert lp.round_trip_gain(gth - 0.01, alpha, R1, R2, L) < 1
assert lp.round_trip_gain(gth + 0.01, alpha, R1, R2, L) > 1

# 6. kwarg bounds
for bad in (lambda: lp.boltzmann_population_ratio(-1, 300),
            lambda: lp.temperature_from_ratio(-1, dE),
            lambda: lp.einstein_A_over_B(0),
            lambda: lp.gain_dB(0, 1),
            lambda: lp.threshold_gain(0.01, 1.2, 0.9, 0.3),   # R > 1
            lambda: lp.threshold_gain(0.01, 1.0, 0.9, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_laser_physics: all checks passed")
