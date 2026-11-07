"""Test dgs.precision_phase_retrieval: the IEEE-754 facts (24/53-bit mantissas,
matching eps), the FFT round-trip floor tracking eps, and the headline result --
running the SAME noiseless GS retrieval in float32 vs float64 gives error floors
~7 orders of magnitude apart, set purely by the storage format. NumPy only."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import precision_phase_retrieval as pr

# 1. the two formats' hard numbers (from numpy.finfo, not quoted)
s = pr.ieee754_summary()
assert s["float32"]["mantissa_bits"] == 24
assert s["float64"]["mantissa_bits"] == 53
assert np.isclose(s["float32"]["eps"], 2.0 ** -23, rtol=1e-6)   # ~1.19e-7
assert np.isclose(s["float64"]["eps"], 2.0 ** -52, rtol=1e-6)   # ~2.22e-16
# ~7 vs ~16 decimal digits; ADC-style dynamic range 6.02 dB/bit
assert 7.0 < s["float32"]["decimal_digits"] < 7.5
assert 15.5 < s["float64"]["decimal_digits"] < 16.5
assert np.isclose(s["float64"]["dynamic_range_dB"], 6.02 * 53)

# 2. FFT round-trip floor tracks machine epsilon of the dtype
f32 = pr.fft_roundtrip_floor(dtype=np.float32)
f64 = pr.fft_roundtrip_floor(dtype=np.float64)
assert 1e-8 < f32 < 1e-5           # single precision: ~1e-7
assert f64 < 1e-12                 # double precision: ~1e-15
assert f32 / f64 > 1e5             # ~9 orders apart

# 3. dtype guard rejects anything but the two supported precisions
try:
    pr.disperse_p(np.ones(8), 5000.0, dtype=np.int32)
    assert False
except ValueError:
    pass

# 4. THE HEADLINE: identical noiseless GS retrieval, float32 vs float64
cmp = pr.compare_precisions(n_iter=200)
e32, e64 = cmp["float32"], cmp["float64"]
# float64 converges essentially to machine zero on noiseless data
assert e64["amp_error"] < 1e-10
assert e64["phase_error"] < 1e-10
# float32 is floor-limited near its epsilon, ~1e-6..1e-7 -- NOT a bug, the format
assert 1e-8 < e32["amp_error"] < 1e-4
assert 1e-8 < e32["phase_error"] < 1e-4
# the floors are ~a million-plus apart, set purely by the mantissa
assert cmp["floor_ratio"] > 1e5
assert e32["phase_error"] > 1e4 * e64["phase_error"]

# 5. both actually converged from a much worse start (the loop is doing work)
for d in (e32, e64):
    assert d["errors"][-1] < 0.1 * d["errors"][0]
# float64's error keeps dropping well past where float32 has already flatlined:
# by iteration 120 float64 is comfortably below float32's final floor and
# still falling, while float32 has been stuck at its epsilon for ~100 iters
assert e64["errors"][120] < e32["amp_error"]
assert e64["errors"][-1] < e64["errors"][120]

# 6. per-precision retrieval returns a full-length phase array; without
#    phi_true the phase_error is None
from dgs import gs_core
d = gs_core.make_qpsk_measurements(n_symbols=32, sps=8, snr_db=np.inf)
phi, errs, perr = pr.retrieve_phase_p(d["I1"], d["I2"], d["D1"], d["D2"],
                                      n_iter=10, dtype=np.float64)
assert len(phi) == len(d["I1"]) and len(errs) == 10 and perr is None

print("test_precision_phase_retrieval: all checks passed")
