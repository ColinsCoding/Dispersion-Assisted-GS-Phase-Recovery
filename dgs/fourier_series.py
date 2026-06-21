"""Fourier series -- building any periodic signal from sines and cosines.

A periodic function on [-L, L] is a sum of harmonics:
    f(x) = a0/2 + sum_{n>=1} [ a_n cos(n pi x/L) + b_n sin(n pi x/L) ],
    a_n = (1/L) int_{-L}^{L} f(x) cos(n pi x/L) dx,
    b_n = (1/L) int_{-L}^{L} f(x) sin(n pi x/L) dx.

The link to dgs.even_odd is the whole point: cosine is even, sine is odd, so
  * an EVEN function has only cosine terms (all b_n = 0),
  * an ODD function has only sine terms (all a_n = 0).
Computing a coefficient is just projecting f onto a harmonic -- an integral over a
symmetric interval, where (even x odd) integrands vanish. The coefficients are taken
numerically (dgs.numerical_methods), so this works for any f you can sample, and it
exposes the Gibbs overshoot (~9%) at jumps. NumPy. Education.
"""

import numpy as np
from dgs import numerical_methods as nm


def fourier_coefficients(f, L=np.pi, N=20, n_grid=8001):
    """Return (a, b): arrays of cosine/sine coefficients a[0..N], b[0..N] for the
    period-2L Fourier series of f. a[0] is the DC term (constant = a[0]/2); b[0]=0.
    Coefficients are projections of f onto each harmonic, integrated numerically."""
    x = np.linspace(-L, L, n_grid)
    fx = f(x)
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    a[0] = nm.trapezoid(fx, x) / L                       # DC: constant term is a0/2
    for n in range(1, N + 1):
        a[n] = nm.trapezoid(fx * np.cos(n * np.pi * x / L), x) / L
        b[n] = nm.trapezoid(fx * np.sin(n * np.pi * x / L), x) / L
    return a, b


def reconstruct(x, a, b, L=np.pi):
    """Evaluate the partial Fourier sum a0/2 + sum a_n cos + b_n sin at point(s) x."""
    x = np.asarray(x, float)
    out = np.full_like(x, a[0] / 2)
    for n in range(1, len(a)):
        out = out + a[n] * np.cos(n * np.pi * x / L) + b[n] * np.sin(n * np.pi * x / L)
    return out


def harmonic_energy(a, b):
    """Per-harmonic energy (Parseval): the (a_n^2 + b_n^2) spectrum over n>=1."""
    return a[1:] ** 2 + b[1:] ** 2


def gibbs_overshoot(a, b, L, x_jump=0.0, halfwidth=0.4, n=4000):
    """Peak of the partial sum just past a jump discontinuity. The Fourier series
    overshoots a jump by ~8.95% of the JUMP HEIGHT no matter how many terms you add
    (the Gibbs phenomenon) -- the lobe only narrows, never shrinks. For sign(x)
    (jump = 2 from -1 to +1) the peak approaches 1 + 0.0895*2 ~ 1.179."""
    x = np.linspace(x_jump + 1e-3, x_jump + halfwidth, n)
    return float(np.max(reconstruct(x, a, b, L)))


if __name__ == "__main__":
    L = np.pi
    # odd square wave sign(x): only sine terms, b_n = 4/(n pi) for odd n
    a_sq, b_sq = fourier_coefficients(np.sign, L, N=15)
    print("square wave (ODD): max|a_n| =", f"{np.max(np.abs(a_sq)):.2e}",
          " b_1 =", f"{b_sq[1]:.4f}", " (4/pi =", f"{4/np.pi:.4f})")
    # even triangle |x|: only cosine terms
    a_tr, b_tr = fourier_coefficients(np.abs, L, N=15)
    print("triangle  (EVEN): max|b_n| =", f"{np.max(np.abs(b_tr)):.2e}",
          " a_1 =", f"{a_tr[1]:.4f}")
    # Gibbs overshoot of the square wave (~1.0895, i.e. ~9% past the jump to 1)
    print("Gibbs overshoot (square wave, N=15) =",
          f"{gibbs_overshoot(a_sq, b_sq, L):.4f}  (~1.179 = 1 + 9% of the jump height 2)")
