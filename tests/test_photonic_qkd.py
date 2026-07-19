"""Test photonic_qkd: Malus's law, deterministic matching-basis detection,
50/50 mismatched-basis detection, and the optical BB84 QBER matching both
the no-eavesdropper (0%) and intercept-resend (25%) analytic predictions."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import photonic_qkd as pq

# 1. Malus's law itself: P = cos^2(angle offset)
assert abs(pq.malus_law_curve([0.0])[0] - 1.0) < 1e-9
assert abs(pq.malus_law_curve([np.pi / 2])[0] - 0.0) < 1e-9
assert abs(pq.malus_law_curve([np.pi / 4])[0] - 0.5) < 1e-9

# 2. measuring in the SAME basis the photon was encoded in is deterministic
for bit in (0, 1):
    for basis in (0, 1):
        photon = pq.encode_bit(bit, basis)
        p = pq.detection_probability(photon, basis)
        assert abs(p - bit) < 1e-9

# 3. measuring in the WRONG basis (45 degree mismatch) gives exactly 50/50
photon = pq.encode_bit(0, 0)
p_mismatch = pq.detection_probability(photon, 1)
assert abs(p_mismatch - 0.5) < 1e-9

# 4. full optical protocol: no eavesdropper -> 0% QBER
clean = pq.bb84_optical_intercept_resend(n_bits=80_000, eavesdrop=False, seed=0)
assert clean["qber_mc"] == 0.0

# 5. full optical protocol: intercept-resend -> QBER near the analytic 25%,
#    derived here from real Jones-calculus optics, not assumed
eve = pq.bb84_optical_intercept_resend(n_bits=80_000, eavesdrop=True, seed=0)
assert abs(eve["qber_mc"] - 0.25) < 0.02

print("test_photonic_qkd: all checks passed")
