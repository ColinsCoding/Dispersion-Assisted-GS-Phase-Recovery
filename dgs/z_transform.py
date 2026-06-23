"""The Z-transform -- calculus for discrete (sampled) signals, and digital filters.

Sampled data needs DISCRETE calculus, and the Z-transform is its engine -- the
discrete analog of the Laplace transform:
    X(z) = sum_n x[n] z^{-n}.
It turns difference equations into algebra, exactly as Laplace turns differential
equations into algebra. The dictionary between continuous and discrete calculus:

    derivative  d/dt   <->  s        <->  backward difference (1 - z^{-1})
    integral    int dt <->  1/s      <->  accumulator        1/(1 - z^{-1})

and just as d/dt and the integral undo each other (the Fundamental Theorem), the
difference and the accumulator multiply to (1 - z^{-1}) * 1/(1 - z^{-1}) = 1 -- the
DISCRETE FTC. A digital filter is a difference equation; its transfer function
H(z) = B(z)/A(z) is a ratio of polynomials whose POLES (roots of A) must lie INSIDE the
unit circle for stability (the discrete version of 'poles in the left half-plane').
This is the math under every DSP receiver. NumPy. Education.
"""

import numpy as np

# canonical operators as (b, a) difference-equation coefficients (sum a_k y[n-k] = sum b_k x[n-k])
DIFFERENCE = ([1.0, -1.0], [1.0])     # y[n] = x[n] - x[n-1]  -> H(z) = 1 - z^{-1}  (discrete d/dt)
ACCUMULATOR = ([1.0], [1.0, -1.0])    # y[n] = y[n-1] + x[n]  -> H(z) = 1/(1 - z^{-1}) (discrete integral)


def z_transform(x, z):
    """X(z) = sum_n x[n] z^{-n} for a finite causal sequence x (n = 0, 1, ...). Accepts a
    scalar or array z. The discrete Laplace transform."""
    x = np.asarray(x, complex)
    n = np.arange(len(x))
    z = np.asarray(z, complex)
    if z.ndim == 0:
        return complex(np.sum(x * z ** (-n)))
    return np.array([np.sum(x * zz ** (-n)) for zz in z])


def filter_response(b, a, z):
    """Transfer function H(z) = B(z)/A(z) = (sum b_k z^{-k}) / (sum a_k z^{-k}) of the
    difference equation with coefficients (b, a). Scalar or array z."""
    b, a = np.asarray(b, complex), np.asarray(a, complex)

    def H(zz):
        num = np.sum(b * zz ** (-np.arange(len(b))))
        den = np.sum(a * zz ** (-np.arange(len(a))))
        return num / den

    z = np.asarray(z, complex)
    return H(z) if z.ndim == 0 else np.array([H(zz) for zz in z])


def frequency_response(b, a, omega):
    """H(e^{j omega}) -- the digital filter's response at normalized frequency omega in
    [0, pi] (omega = pi is Nyquist). The discrete version of the Fourier response."""
    return filter_response(b, a, np.exp(1j * np.asarray(omega, float)))


def poles_zeros(b, a):
    """(poles, zeros) of H(z): zeros are roots of B(z), poles roots of A(z) (as
    polynomials in z, highest power first). Stability and shape both live here."""
    return np.roots(np.asarray(a, complex)), np.roots(np.asarray(b, complex))


def is_stable(a):
    """A digital filter is stable iff every POLE (root of A(z)) lies INSIDE the unit
    circle, |z| < 1 -- the discrete analog of 'all poles in the left half s-plane'."""
    poles = np.roots(np.asarray(a, complex))
    return bool(len(poles) == 0 or np.all(np.abs(poles) < 1.0 - 1e-12))


def apply_filter(b, a, x):
    """Run the difference equation on a signal x (direct form): the time-domain filter
    whose transfer function is H(z) = B(z)/A(z)."""
    b, a = np.asarray(b, float), np.asarray(a, float)
    b = b / a[0]; a = a / a[0]
    y = np.zeros(len(x))
    for n in range(len(x)):
        acc = sum(b[k] * x[n - k] for k in range(len(b)) if n - k >= 0)
        acc -= sum(a[k] * y[n - k] for k in range(1, len(a)) if n - k >= 0)
        y[n] = acc
    return y


if __name__ == "__main__":
    # Z-transform of shifted deltas: delta[n-2] -> z^{-2}
    print("Z{delta[n-2]} at z=2 :", z_transform([0, 0, 1], 2.0), " (= 2^-2 = 0.25)")
    # the discrete derivative kills DC (z=1) and emphasizes high frequency
    print("difference H(z=1) (DC) :", filter_response(*DIFFERENCE, 1.0), " (= 0, a difference removes DC)")
    # discrete FTC: difference then accumulate = identity, H = 1
    z = 1.7 + 0.3j
    Hd = filter_response(*DIFFERENCE, z); Ha = filter_response(*ACCUMULATOR, z)
    print("difference * accumulator =", round((Hd * Ha).real, 10), " (discrete FTC: = 1)")
    # stability: a pole at 0.9 is stable, at 1.1 is not
    print("pole 0.9 stable:", is_stable([1, -0.9]), "  pole 1.1 stable:", is_stable([1, -1.1]))
