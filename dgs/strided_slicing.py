"""Strided slicing x[start:stop:step] -- a linear operator hiding in syntax.

Python's `x[start:stop:step]` looks like mere indexing, but every slice is a
LINEAR MAP y = S @ x for a fixed 0/1 selection matrix S: it picks a subset of
samples, so it commutes with addition and scaling like any matrix. Writing S
down turns four throwaway bits of syntax into operators you can compose and
Fourier-analyze:
    x[x::y]  start x, every y-th sample      (a decimation)
    x[i::J]  start i, every J-th sample       (a phase of a polyphase split)
    x[p:l]   contiguous window p..l, step 1   (a projection / crop)
    x[b::d]  start b, every d-th sample        (another decimation phase)

Two identities make slicing an algebra rather than a notation:
  * COMPOSITION. Slicing a slice multiplies the strides:
        (x[a::s])[b::t] == x[a + b*s :: s*t].
    Strided access is closed under composition -- exactly why nested strides
    in NumPy never need a copy.
  * DECIMATION <-> ALIASING. Downsampling y[n]=x[M*n] folds the spectrum:
        Y[k] = (1/M) * sum_{m=0}^{M-1} X[k + m*L],   L = N/M.
    Taking every M-th SAMPLE sums M equally spaced COPIES of the spectrum --
    the aliasing theorem, and the reason `x[::2]` can destroy information.

The famous payoff: the radix-2 FFT is nothing but the slice pair x[0::2]
(even) and x[1::2] (odd). Decimation-in-time splits the DFT into those two
sub-DFTs recombined by the butterfly X[k] = E[k] + W^k O[k], X[k+L] = E[k] -
W^k O[k]. This module proves that butterfly SYMBOLICALLY with SymPy (the
repo's Maxima; see dgs.cas_tools) -- for symbolic x0..xN the sliced-and-
recombined result simplifies to the direct DFT exactly, not just numerically.

Ties to dgs.even_odd / dgs.eye_diagram (x[0::2] and x[1::2] ARE the even/odd
decimation the parity operator diagonalizes) and reuses
dgs.fourier_tools.fft_radix2. NumPy + SymPy. Education.
"""

import numpy as np
import sympy as sp


# ----------------------------------------------------------------------
# A slice is a matrix: y = S @ x
# ----------------------------------------------------------------------

def slice_matrix(N, sl):
    """The selection matrix S (shape L x N) such that S @ x == x[sl] for any
    length-N vector, where `sl` is a Python `slice`. Each row is a one-hot
    picking one selected index; L = len(range(*sl.indices(N))). This makes
    concrete that x[start:stop:step] is a linear operator."""
    if N <= 0:
        raise ValueError("N must be positive")
    if not isinstance(sl, slice):
        raise TypeError("sl must be a Python slice, e.g. np.s_[1::2]")
    idx = range(*sl.indices(N))
    S = np.zeros((len(idx), N))
    for row, col in enumerate(idx):
        S[row, col] = 1.0
    return S


def apply_slice_as_matrix(x, sl):
    """Return (x[sl], S @ x): the native slice and the matrix product, which
    are equal. The point is that they are equal -- slicing IS a matmul."""
    x = np.asarray(x, float)
    S = slice_matrix(len(x), sl)
    return x[sl], S @ x


def compose_slices(a, s, b, t, N):
    """Verify the stride-composition identity (x[a::s])[b::t] == x[a+b*s::s*t].
    Returns (nested, single) which are equal -- strided access is closed
    under composition, so NumPy fuses nested strides without copying.
    Requires positive strides s, t."""
    if s <= 0 or t <= 0:
        raise ValueError("strides s, t must be positive")
    if not (0 <= a < N and 0 <= b):
        raise ValueError("need 0 <= a < N and b >= 0")
    x = np.arange(N)
    nested = x[a::s][b::t]
    single = x[a + b * s:: s * t]
    return nested, single


# ----------------------------------------------------------------------
# Decimation and its spectral shadow: aliasing
# ----------------------------------------------------------------------

def decimate(x, M):
    """Keep every M-th sample: x[::M]. The prototypical strided slice."""
    x = np.asarray(x)
    if M < 1:
        raise ValueError("M must be >= 1")
    return x[::M]


def decimation_spectrum_identity(x, M):
    """Numerically confirm the aliasing theorem for y = x[::M]:
        DFT_L(y)[k] == (1/M) * sum_{m=0}^{M-1} DFT_N(x)[k + m*L],  L = N/M.
    Taking every M-th sample sums M equally spaced copies of the spectrum;
    when those copies overlap (input not band-limited to < f_s/2M), the sum
    is aliased and x[::M] has thrown information away. Requires M | N.
    Returns (Y_direct, Y_folded, max_abs_error)."""
    x = np.asarray(x, complex)
    N = len(x)
    if M < 1:
        raise ValueError("M must be >= 1")
    if N % M != 0:
        raise ValueError("len(x) must be divisible by M")
    L = N // M
    X = np.fft.fft(x)
    Y_direct = np.fft.fft(x[::M])
    folded = sum(X[np.arange(L) + m * L] for m in range(M)) / M
    return Y_direct, folded, float(np.max(np.abs(Y_direct - folded)))


# ----------------------------------------------------------------------
# The slice pair that is the FFT: x[0::2], x[1::2]
# ----------------------------------------------------------------------

def decimation_in_time(x):
    """One radix-2 Cooley-Tukey stage spelled out as two slices. Splits x
    into even = x[0::2] and odd = x[1::2], DFTs each (via numpy here for the
    sub-transform), and recombines with the butterfly
        X[k]   = E[k] + W_N^k * O[k]
        X[k+L] = E[k] - W_N^k * O[k],     W_N = exp(-2*pi*i/N),  L = N/2.
    Returns X equal to np.fft.fft(x). N must be even. The whole recursive
    FFT is dgs.fourier_tools.fft_radix2 -- this exposes a single stage so the
    slices are visible."""
    x = np.asarray(x, complex)
    N = len(x)
    if N < 2 or N % 2:
        raise ValueError("N must be even (>= 2)")
    L = N // 2
    E = np.fft.fft(x[0::2])
    O = np.fft.fft(x[1::2])
    tw = np.exp(-2j * np.pi * np.arange(L) / N)
    return np.concatenate([E + tw * O, E - tw * O])


# ----------------------------------------------------------------------
# The Maxima part: prove the butterfly SYMBOLICALLY, not just numerically
# ----------------------------------------------------------------------

def _sym_dft(x):
    """Direct symbolic DFT: X[k] = sum_n x[n] exp(-2*pi*i*k*n/N)."""
    N = len(x)
    return [sp.expand(sum(x[n] * sp.exp(-2 * sp.I * sp.pi * k * n / N)
                          for n in range(N))) for k in range(N)]


def _sym_fft(x):
    """Symbolic radix-2 FFT via the same x[0::2]/x[1::2] slices, recombined
    by the butterfly -- the exact operations decimation_in_time performs, but
    carried out on SymPy symbols so nothing is rounded."""
    N = len(x)
    if N == 1:
        return [x[0]]
    E = _sym_fft(x[0::2])
    O = _sym_fft(x[1::2])
    out = [sp.Integer(0)] * N
    for k in range(N // 2):
        w = sp.exp(-2 * sp.I * sp.pi * k / N)
        out[k] = sp.expand(E[k] + w * O[k])
        out[k + N // 2] = sp.expand(E[k] - w * O[k])
    return out


def butterfly_symbolic(N=4):
    """Prove Cooley-Tukey exactly: for symbolic inputs x0..x_{N-1}, the
    slice-and-butterfly FFT equals the direct DFT term for term. Returns
    dict with the symbolic X[k] (from the sliced FFT), the per-bin residuals
    fft[k] - dft[k] (all simplify to 0), and `proven` True. N a power of 2."""
    if N < 2 or (N & (N - 1)):
        raise ValueError("N must be a power of 2 (>= 2)")
    x = list(sp.symbols(f"x0:{N}"))
    fft = _sym_fft(x)
    dft = _sym_dft(x)
    residuals = [sp.simplify(fft[k] - dft[k]) for k in range(N)]
    return {
        "X": [sp.simplify(v) for v in fft],
        "residuals": residuals,
        "proven": all(r == 0 for r in residuals),
    }


def twiddle_half_turn_symbolic():
    """The one identity that makes the butterfly's +/- work: the twiddle
    factor at k+N/2 is the NEGATIVE of the one at k,
        W_N^{k+N/2} = -W_N^k,      W_N = exp(-2*pi*i/N),
    because exp(-i*pi) = -1. So E[k] +/- W_N^k O[k] already covers both
    output halves -- one multiply feeds two outputs, halving the work.
    Returned symbolically simplified to exactly -W^k."""
    k, N = sp.symbols("k N", positive=True)
    # combine the exponents first: (-2*pi*i/N)*(k+N/2) - (-2*pi*i/N)*k = -pi*i,
    # so the ratio is exp(-pi*i) = -1. (Dividing symbolic powers W**a/W**b does
    # not auto-collapse in SymPy; forming the single exponent does.)
    exponent = -2 * sp.I * sp.pi / N * (k + N / 2) - (-2 * sp.I * sp.pi / N * k)
    return sp.simplify(sp.exp(sp.simplify(exponent)))       # == -1


if __name__ == "__main__":
    # 1. a slice really is a matmul
    x = np.arange(8.0)
    native, viamat = apply_slice_as_matrix(x, np.s_[1::2])
    print("x[1::2]      =", native, " == S @ x ?", np.allclose(native, viamat))

    # 2. nested strides multiply
    nested, single = compose_slices(1, 2, 1, 3, 40)
    print("x[1::2][1::3] == x[7::6] ?", np.array_equal(nested, single),
          " ->", single)

    # 3. decimation folds the spectrum (aliasing theorem)
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(16)
    _, _, err = decimation_spectrum_identity(sig, 2)
    print(f"DFT(x[::2]) == folded spectrum sum? residual = {err:.1e}")

    # 4. the slice pair x[0::2]/x[1::2] reconstructs the full FFT
    print("decimation_in_time == np.fft.fft ?",
          np.allclose(decimation_in_time(sig), np.fft.fft(sig)))

    # 5. and it's provably exact, symbolically (the Maxima part)
    res = butterfly_symbolic(4)
    print("butterfly proven for N=4 (residuals all 0)?", res["proven"])
    print("  X[1] =", res["X"][1])
    print("twiddle half-turn W^(k+N/2)/W^k simplifies to",
          twiddle_half_turn_symbolic(), "(the +/- in the butterfly)")
