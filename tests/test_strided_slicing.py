"""Test dgs.strided_slicing: a slice equals its selection matrix, nested
strides multiply, decimation folds the spectrum (aliasing theorem), the
x[0::2]/x[1::2] slice pair rebuilds the full FFT, and the Cooley-Tukey
butterfly is proven SYMBOLICALLY (residuals exactly zero, twiddle half-turn
= -1). NumPy + SymPy."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from dgs import strided_slicing as ss

# 1. a slice IS a linear operator: x[sl] == S @ x for several slice forms
x = np.arange(12.0) ** 2 - 3 * np.arange(12.0)
for sl in (np.s_[::2], np.s_[1::2], np.s_[3:9], np.s_[2::3]):
    native, viamat = ss.apply_slice_as_matrix(x, sl)
    assert np.allclose(native, viamat)
# S has exactly one 1 per row and correct shape
S = ss.slice_matrix(12, np.s_[1::2])
assert S.shape == (6, 12)
assert np.array_equal(S.sum(axis=1), np.ones(6))

# 2. stride composition: (x[a::s])[b::t] == x[a+b*s :: s*t]
for a, s, b, t, N in [(1, 2, 1, 3, 40), (0, 3, 2, 2, 30), (2, 5, 1, 1, 50)]:
    nested, single = ss.compose_slices(a, s, b, t, N)
    assert np.array_equal(nested, single)
    assert np.array_equal(single, np.arange(N)[a + b * s:: s * t])

# 3. decimation <-> aliasing theorem, to machine precision.
#    A tone below f_s/2M survives; folding is exact regardless of aliasing.
rng = np.random.default_rng(1)
for sig in (rng.standard_normal(24), np.cos(2 * np.pi * 3 * np.arange(24) / 24)):
    for M in (2, 3, 4):
        Yd, Yf, err = ss.decimation_spectrum_identity(sig, M)
        assert err < 1e-10
        assert len(Yd) == len(sig) // M

# 4. an aliasing DEMONSTRATION: two tones that collide under x[::2].
#    bins 2 and 2 + N/2 = 10 both fold onto output bin 2, so downsampling
#    cannot tell them apart -- the folded spectrum proves the loss.
N = 16
hi = np.exp(2j * np.pi * 10 * np.arange(N) / N)   # bin 10
lo = np.exp(2j * np.pi * 2 * np.arange(N) / N)    # bin 2
assert np.allclose(hi[::2], lo[::2])              # identical after decimation

# 5. the FFT is the slice pair: decimation_in_time == np.fft.fft
for n in (2, 8, 16, 32):
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert np.allclose(ss.decimation_in_time(v), np.fft.fft(v))
    # and equals the repo's recursive radix-2 too
    from dgs.fourier_tools import fft_radix2
    assert np.allclose(ss.decimation_in_time(v), fft_radix2(v))

# 6. THE SYMBOLIC PROOF (the "maxima" part): butterfly == direct DFT exactly
for n in (2, 4, 8):
    res = ss.butterfly_symbolic(n)
    assert res["proven"]
    assert all(r == 0 for r in res["residuals"])
    assert len(res["X"]) == n
# X[0] is always the DC sum x0 + x1 + ... (symbolically)
res4 = ss.butterfly_symbolic(4)
x0, x1, x2, x3 = sp.symbols("x0 x1 x2 x3")
assert sp.simplify(res4["X"][0] - (x0 + x1 + x2 + x3)) == 0

# 7. the twiddle half-turn that powers the +/- butterfly: W^(k+N/2) = -W^k
assert ss.twiddle_half_turn_symbolic() == -1

# 8. kwarg bounds
for bad in (lambda: ss.slice_matrix(0, np.s_[::2]),
            lambda: ss.slice_matrix(8, "not a slice"),
            lambda: ss.compose_slices(1, 0, 1, 2, 10),
            lambda: ss.decimate(x, 0),
            lambda: ss.decimation_spectrum_identity(x[:10], 3),  # 3 does not divide 10
            lambda: ss.decimation_in_time(x[:7]),                # odd length
            lambda: ss.butterfly_symbolic(6)):                   # not a power of 2
    try:
        bad()
        assert False, "expected an exception"
    except (ValueError, TypeError):
        pass

print("test_strided_slicing: all checks passed")
