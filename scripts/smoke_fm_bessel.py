"""Verify FM spectrum sidebands match Bessel J_n(beta), and the carrier null."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import bessel as bz

# build an FM tone and measure its sideband amplitudes from the FFT
fs = 8000           # sample rate
T = 2.0             # seconds (long -> sharp spectral lines)
t = np.arange(int(fs*T)) / fs
fc, fm = 1000.0, 100.0      # carrier and modulation frequencies
beta = 3.0                  # modulation index

s = np.cos(2*np.pi*fc*t + beta*np.sin(2*np.pi*fm*t))
S = np.abs(np.fft.rfft(s)) / (len(s)/2)
freqs = np.fft.rfftfreq(len(s), 1/fs)

print(f"FM tone fc={fc}, fm={fm}, beta={beta}: sideband amplitudes vs |J_n(beta)|")
maxerr = 0.0
for n in range(-5, 6):
    f_line = fc + n*fm
    idx = np.argmin(np.abs(freqs - f_line))
    measured = S[idx]
    predicted = abs(float(bz.fm_sideband_amplitudes(beta, 5)[n]))
    maxerr = max(maxerr, abs(measured - predicted))
    print(f"  n={n:+d}  f={f_line:5.0f} Hz   measured={measured:.3f}   |J_n|={predicted:.3f}")
print("max |measured - |J_n||:", round(maxerr, 4))
assert maxerr < 0.02, "FM sidebands do not match Bessel"

# carrier null: at beta = 2.405 (first zero of J_0) the carrier vanishes
betas = bz.carrier_null_indices(3)
print("\ncarrier-null modulation indices (zeros of J_0):", np.round(betas, 4))
for b in betas:
    print(f"  beta={b:.4f}: J_0={float(bz.fm_sideband_amplitudes(b,0)[0]):.4f} (~0 -> carrier off)")
assert abs(bz.fm_sideband_amplitudes(betas[0], 0)[0]) < 1e-6

# power conservation: sum of J_n(beta)^2 over all n = 1 (Bessel identity)
amps = bz.fm_sideband_amplitudes(4.0, 40)
total = sum(v**2 for v in amps.values())
print(f"\nsum J_n(4)^2 over n=-40..40 = {total:.5f} (Parseval: -> 1)")
assert abs(total - 1) < 1e-3

for bad in [lambda: bz.fm_sideband_amplitudes(-1, 5)]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", e)
print("SMOKE PASS")
