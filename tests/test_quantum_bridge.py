"""Test the quantum bridge: Born rule, phase ambiguity, shot noise, time-bandwidth."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import quantum_bridge as qb

# 1. Born rule: |A e^{i phi}|^2 = A^2 (phase discarded)
A = np.array([1.0, 2.0, 3.0]); phi = np.array([0.5, -1.2, 2.0])
psi = A * np.exp(1j * phi)
assert np.allclose(qb.born_rule(psi), A**2)

# 2. phase ambiguity: same intensity, genuinely different fields (the retrieval problem)
I = np.array([1.0, 4.0, 9.0])
a, b = qb.phase_ambiguity(I, np.zeros(3), np.array([0.3, -0.7, 1.1]))
assert np.allclose(qb.born_rule(a), I) and np.allclose(qb.born_rule(b), I)
assert not np.allclose(a, b)                                   # different phase -> different field
assert np.allclose(np.abs(a), np.abs(b))                       # but identical magnitude

# 3. partial-wave angular momentum = n hbar
hbar = 1.054571817e-34
assert np.isclose(qb.partial_wave_angular_momentum(3), 3 * hbar)
assert np.isclose(qb.partial_wave_angular_momentum(1, hbar=1.0), 1.0)

# 4. shot noise: SNR = sqrt(N); 4x the photons -> 2x the SNR (the sqrt(N) law)
assert np.isclose(qb.shot_noise_snr(1e4), 100.0)
assert np.isclose(qb.shot_noise_snr(4 * 1e4) / qb.shot_noise_snr(1e4), 2.0)

# 5. time-bandwidth: a Gaussian reaches the ~0.5 minimum; a CHIRP raises it (excess bandwidth)
t = np.linspace(-20, 20, 4096)
g = np.exp(-t**2 / 2)
tbp_min = qb.time_bandwidth_product(t, g)
assert abs(tbp_min - 0.5) < 0.02                               # minimum-uncertainty pulse
chirped = g * np.exp(1j * 1.0 * t**2)                          # same duration, more bandwidth
assert qb.time_bandwidth_product(t, chirped) > 1.0             # chirp lifts it above the 0.5 min
# a wider Gaussian is narrower in frequency -> product stays ~0.5 (still minimum)
g2 = np.exp(-t**2 / 8)
assert abs(qb.time_bandwidth_product(t, g2) - 0.5) < 0.05

print(f"TEST PASS  (Born rule |psi|^2 drops phase; phase ambiguity same I diff field; "
      f"L=n hbar; shot-noise SNR=sqrt(N) (4x->2x); Gaussian TBP={tbp_min:.3f}~0.5 min, "
      f"chirp raises it)")
