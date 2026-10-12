"""Test dgs.fire_imaging: the noiseless round trip is exact on an orthogonal
tag grid, the lock-in equals the FFT bin, cutting the line short produces real
crosstalk, the fluorophore low-pass has the RC-filter values, the shot-noise
formula matches a Monte Carlo (including the multiplex penalty), and bad
inputs (aliased tags above all) are rejected rather than silently wrong."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import fire_imaging as fi

N, F0, DF = 16, 20e6, 1e6
d = fi.design_line(N, F0, DF)
freqs, fs, n = d["freqs_hz"], d["fs_hz"], d["n_samples"]
truth = np.linspace(5.0, 50.0, N)

# 1. tag grid and throughput: pixel rate equals the RF bandwidth used
assert np.allclose(fi.tag_frequencies(4, 10e6, 2e6), [10e6, 12e6, 14e6, 16e6])
tp = fi.throughput(N, DF)
assert tp["line_time_s"] == 1 / DF and tp["line_rate_hz"] == DF
assert tp["pixel_rate_hz"] == tp["rf_bandwidth_hz"] == N * DF
assert math.isclose(d["line_time_s"], 1 / DF) and fs > 2 * freqs[-1]
assert math.isclose(n / fs, d["line_time_s"])                 # whole line, whole cycles

# 2. noiseless round trip on the orthogonal grid is exact, for both estimators
clean = fi.simulate_line(truth, freqs, fs, n)
assert np.allclose(fi.demodulate_line(clean, freqs, fs), truth, atol=1e-9)
assert np.allclose(fi.demodulate_line(clean, freqs, fs, method="magnitude"), truth, atol=1e-9)
assert math.isclose(clean.mean(), truth.sum())                # DC level = all pixels summed

# 3. the lock-in IS the FFT when tags sit on bins
bins = np.rint(freqs * n / fs).astype(int)
assert np.allclose(fi.tone_amplitudes(clean, freqs, fs), 2 / n * np.fft.fft(clean)[bins])

# 4. a pixel that is dark stays dark: no leakage from bright neighbours
one_hot = np.zeros(N); one_hot[5] = 40.0
rec = fi.demodulate_line(fi.simulate_line(one_hot, freqs, fs, n), freqs, fs)
assert np.allclose(rec, one_hot, atol=1e-9)

# 5. cut the line to a non-integer number of beat periods -> real crosstalk
n_short = int(0.63 * n)
leaky = fi.demodulate_line(fi.simulate_line(one_hot, freqs, fs, n_short), freqs, fs)
assert np.max(np.abs(np.delete(leaky, 5))) > 1.0              # photons/sample leaked into dark pixels

# 6. modulation depth below 1 is divided back out
half = fi.simulate_line(truth, freqs, fs, n, mod_depth=0.5)
assert np.allclose(fi.demodulate_line(half, freqs, fs, mod_depth=0.5), truth, atol=1e-9)

# 7. fluorophore = first-order low-pass (RC filter with RC = tau)
tau = 4e-9
M, lag = fi.lifetime_response(1 / (2 * math.pi * tau), tau)   # corner frequency
assert math.isclose(M, 1 / math.sqrt(2)) and math.isclose(lag, math.pi / 4)
assert fi.lifetime_response(50e6, 0.0) == (1.0, 0.0)          # no lifetime -> no filtering
assert math.isclose(fi.lifetime_from_phase(lag, 1 / (2 * math.pi * tau)), tau)

# 8. with a lifetime: correcting recovers truth; NOT correcting dims the pixels by
#    exactly M(f) * cos(lag) -- and the phase of each tag gives tau back
sig = fi.simulate_line(truth, freqs, fs, n, tau_s=tau)
assert np.allclose(fi.demodulate_line(sig, freqs, fs, tau_s=tau), truth, atol=1e-9)
Mf, lagf = fi.lifetime_response(freqs, tau)
assert np.allclose(fi.demodulate_line(sig, freqs, fs, tau_s=0.0), truth * Mf * np.cos(lagf), atol=1e-9)
assert np.allclose(fi.demodulate_line(sig, freqs, fs, tau_s=0.0, method="magnitude"), truth * Mf, atol=1e-9)
assert np.allclose(fi.measure_phase_lag(sig, freqs, fs), lagf, atol=1e-9)
assert np.allclose(fi.lifetime_from_phase(fi.measure_phase_lag(sig, freqs, fs), freqs), tau, rtol=1e-9)

# 9. shot noise: Monte Carlo matches sqrt(2 N_total)/(n m), and the estimate is unbiased
rng = np.random.default_rng(0)
trials = np.array([fi.demodulate_line(fi.simulate_line(truth, freqs, fs, n, rng=rng), freqs, fs)
                   for _ in range(3000)])
sigma = fi.predicted_sigma(truth.sum(), n)
assert np.allclose(trials.std(axis=0), sigma, rtol=0.08)
assert np.allclose(trials.mean(axis=0), truth, atol=5 * sigma / math.sqrt(3000))
assert math.isclose(fi.predicted_snr(truth[3], truth.sum(), n), truth[3] / sigma)

# 10. multiplex penalty: the same dim pixel is noisier beside bright neighbours
dim_alone = np.zeros(N); dim_alone[0] = 5.0
crowded = np.full(N, 50.0); crowded[0] = 5.0
rng = np.random.default_rng(1)
s_alone = np.std([fi.demodulate_line(fi.simulate_line(dim_alone, freqs, fs, n, rng=rng), freqs, fs)[0]
                  for _ in range(3000)])
s_crowd = np.std([fi.demodulate_line(fi.simulate_line(crowded, freqs, fs, n, rng=rng), freqs, fs)[0]
                  for _ in range(3000)])
expected_ratio = math.sqrt(crowded.sum() / dim_alone.sum())
assert math.isclose(s_crowd / s_alone, expected_ratio, rel_tol=0.08)
assert math.isclose(fi.predicted_sigma(4 * truth.sum(), n), 2 * sigma)   # 4x light -> 2x noise

# 11. reproducible with a seed; noiseless when rng is None
a = fi.simulate_line(truth, freqs, fs, n, rng=np.random.default_rng(7))
b = fi.simulate_line(truth, freqs, fs, n, rng=np.random.default_rng(7))
assert np.array_equal(a, b) and not np.array_equal(a, clean)
assert np.all(a == np.round(a)) and np.all(a >= 0)             # photon counts are whole and non-negative

# 12. full image: exact without noise, close with it, and timing follows the design
yy, xx = np.mgrid[0:24, 0:32]
image = np.exp(-((xx - 10) ** 2 + (yy - 12) ** 2) / 30.0) + 0.5 * (np.abs(yy - 0.5 * xx) < 1.5)
rec0, info = fi.simulate_image(image, 20e6, 1e6, peak_photons_per_sample=100.0)
assert rec0.shape == image.shape and np.allclose(rec0, info["truth"], atol=1e-8)
assert math.isclose(info["frame_time_s"], 24 * 1e-6) and info["pixel_rate_hz"] == 32e6
rec1, _ = fi.simulate_image(image, 20e6, 1e6, peak_photons_per_sample=100.0, seed=0)
rec2, _ = fi.simulate_image(image, 20e6, 1e6, peak_photons_per_sample=100.0, seed=0)
assert np.array_equal(rec1, rec2)
assert np.corrcoef(rec1.ravel(), info["truth"].ravel())[0, 1] > 0.95

# 13. bad inputs are refused, aliasing first of all
bad_calls = [
    lambda: fi.simulate_line(truth, freqs, 2 * freqs[-1], n),            # tag at/above Nyquist
    lambda: fi.demodulate_line(clean, freqs, 60e6),                      # same, on the receive side
    lambda: fi.simulate_line(-truth, freqs, fs, n),                      # negative photons
    lambda: fi.simulate_line(truth, freqs[:-1], fs, n),                  # length mismatch
    lambda: fi.simulate_line(truth, freqs, fs, n, mod_depth=1.5),
    lambda: fi.simulate_line(truth, freqs, fs, n, tau_s=-1e-9),
    lambda: fi.demodulate_line(clean, freqs, fs, method="guess"),
    lambda: fi.design_line(N, 20.5e6, DF),                               # off-grid start: not orthogonal
    lambda: fi.design_line(N, F0, DF, oversample=2.0),
    lambda: fi.tag_frequencies(0, F0, DF),
    lambda: fi.throughput(N, 0.0),
    lambda: fi.lifetime_from_phase(math.pi / 2, 1e6),
    lambda: fi.predicted_sigma(100.0, n, mod_depth=1.0, modulation=1.2),
    lambda: fi.simulate_image(np.zeros((4, 4)), F0, DF, 10.0),
    lambda: fi.simulate_image(np.ones(8), F0, DF, 10.0),
]
for bad in bad_calls:
    try:
        bad(); assert False, "should have raised"
    except ValueError:
        pass

print("test_fire_imaging: all checks passed")
