"""Test dgs.phase_retrieval_ambiguities: the three trivial ambiguities preserve |FFT|, the exact
FFT identities behind them, global-phase alignment / invariant distance, canonicalization, and that
a relative-phase change is NOT invisible."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from dgs import phase_retrieval_ambiguities as pa

rng = np.random.default_rng(1)
x = rng.standard_normal(32) + 1j * rng.standard_normal(32)
X = np.fft.fft(x)

# 1. global phase leaves the magnitude invariant, and only multiplies X by e^{i phi}
for phi in (0.3, 1.7, -2.1):
    y = pa.apply_global_phase(x, phi)
    assert pa.leaves_magnitude_invariant(x, y)
    assert np.allclose(np.fft.fft(y), np.exp(1j*phi)*X)

# 2. conjugate/twin: FFT(y) = conj(X) exactly, so same magnitude
tw = pa.conjugate_reflection(x)
assert np.allclose(np.fft.fft(tw), np.conj(X))
assert pa.leaves_magnitude_invariant(x, tw)
# applying it twice returns the original
assert np.allclose(pa.conjugate_reflection(tw), x)

# 3. translation: FFT gets a linear phase e^{-i k n0}, magnitude unchanged
for s in (1, 5, 13):
    y = pa.translate(x, s)
    assert pa.leaves_magnitude_invariant(x, y)
    k = np.arange(len(x))
    assert np.allclose(np.fft.fft(y), X * np.exp(-2j*np.pi*k*s/len(x)))

# 4. any composition of the three is still magnitude-invariant
combo = pa.translate(pa.apply_global_phase(pa.conjugate_reflection(x), 0.9), 7)
assert pa.leaves_magnitude_invariant(x, combo)

# 5. global_phase_align recovers a signal that differs only by a global phase
y = pa.apply_global_phase(x, 2.4)
aligned = pa.global_phase_align(x, y)
assert np.linalg.norm(x - aligned) < 1e-9          # exactly recovered
assert np.linalg.norm(x - y) > 1                   # ...whereas the naive difference is large

# 6. phase-invariant distance: 0 for global-phase copies, >0 for genuinely different, symmetric
assert pa.phase_invariant_distance(x, pa.apply_global_phase(x, 1.1)) < 1e-9
w = rng.standard_normal(32) + 1j*rng.standard_normal(32)
assert pa.phase_invariant_distance(x, w) > 0
assert math.isclose(pa.phase_invariant_distance(x, w), pa.phase_invariant_distance(w, x), rel_tol=1e-12)
# matches the closed form ||a||^2+||b||^2-2|<a,b>|
aa, bb = np.vdot(x, x).real, np.vdot(w, w).real
assert math.isclose(pa.phase_invariant_distance(x, w),
                    math.sqrt(aa + bb - 2*abs(np.vdot(x, w))), rel_tol=1e-12)
# distance to a scaled-and-rephased copy: invariant to phase but not to scale
assert pa.phase_invariant_distance(x, 2*pa.apply_global_phase(x, 0.5)) > 0

# 7. canonicalization: global-phase-equivalent signals share one canonical representative
for phi in (0.0, 0.6, -1.9, 3.0):
    assert np.allclose(pa.canonicalize_phase(x), pa.canonicalize_phase(pa.apply_global_phase(x, phi)))
# the canonical form's largest-magnitude sample is real and positive
xc = pa.canonicalize_phase(x)
i = int(np.argmax(np.abs(xc)))
assert abs(xc[i].imag) < 1e-9 and xc[i].real > 0

# 8. a RELATIVE phase change (one sample) is NOT invisible -- it changes the magnitude
z = x.copy(); z[4] *= np.exp(1j*0.8)
assert not pa.leaves_magnitude_invariant(x, z)
assert pa.phase_invariant_distance(x, z) > 1e-6     # genuinely different physics

# 9. real, nonnegative x is still ambiguous (twin + shift), a classic support caveat
xr = np.abs(rng.standard_normal(16))
assert pa.leaves_magnitude_invariant(xr, pa.conjugate_reflection(xr))
assert pa.leaves_magnitude_invariant(xr, pa.translate(xr, 3))

print("test_phase_retrieval_ambiguities: all checks passed")
