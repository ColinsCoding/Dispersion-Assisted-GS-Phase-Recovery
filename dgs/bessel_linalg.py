"""The linear algebra of Bessel functions -- eigenvalues, orthogonality, recurrence.

Bessel's equation is an EIGENVALUE problem in disguise. With the substitution
u = sqrt(r) * y, the radial operator becomes Schrodinger-like,
    -u'' + (n^2 - 1/4)/r^2 * u = k^2 u ,   u(0) = u(R) = 0,
so discretizing it gives a symmetric tridiagonal MATRIX whose eigenvalues are the
squared Bessel zeros (alpha_m/R)^2 and whose eigenvectors are the Bessel functions
J_n(alpha_m r/R) (times sqrt(r)). A differential operator turned into a matrix -- the
core move of numerical linear algebra.

Two more linear-algebra facts: the Bessel functions are an ORTHOGONAL basis
(Fourier-Bessel series, expansion by inner products, like Fourier), and the
three-term RECURRENCE J_{n-1} + J_{n+1} = (2n+1)/x J_n is a linear relation -- the
same recurrence that builds the spherical Bessel functions in the SEALS/Mie code.
NumPy only. Education.
"""

import numpy as np


def bessel_operator_matrix(order, R=1.0, N=1200):
    """Matrix for the radial Bessel operator  -y'' - (1/r)y' + n^2/r^2 y  on (0,R],
    Dirichlet y(R)=0, and at the center y'(0)=0 (n=0) or y(0)=0 (n>0). Returns
    (M, r). Eigenvalues are (alpha_m/R)^2, eigenvectors are J_n(alpha_m r/R)."""
    dr = R / N
    r = np.arange(1, N) * dr                        # nodes r_1..r_{N-1}; r_N=R is Dirichlet
    Nn = len(r)
    M = np.zeros((Nn, Nn))
    for i in range(Nn):
        ri = r[i]
        M[i, i] = 2.0 / dr**2 + order**2 / ri**2
        if i < Nn - 1:
            M[i, i + 1] = -1.0 / dr**2 - 1.0 / (2 * ri * dr)
        if i > 0:
            M[i, i - 1] = -1.0 / dr**2 + 1.0 / (2 * ri * dr)
    if order == 0:                                  # Neumann at center: y_0 = y_1
        M[0, 0] += -1.0 / dr**2 + 1.0 / (2 * r[0] * dr)
    return M, r


def _sorted_eig(M, k, symmetric=False):
    eig, vec = np.linalg.eig(M)
    eig = eig.real
    idx = np.argsort(eig)[:k]
    return eig[idx], vec[:, idx].real


def bessel_zeros(order, k=5, R=1.0, N=1200):
    """First k positive zeros of J_order, found as sqrt(eigenvalues)*R of the radial
    operator matrix -- the Bessel zeros ARE the eigenvalues of a differential operator."""
    M, _ = bessel_operator_matrix(order, R, N)
    eig, _ = _sorted_eig(M, k)
    return np.sqrt(eig) * R


def bessel_modes(order, k=3, R=1.0, N=1200):
    """Return (r, zeros, modes): the first k eigenvectors, each proportional to
    J_order(alpha_m r/R). Normalized to peak 1, sign-fixed."""
    M, r = bessel_operator_matrix(order, R, N)
    eig, vec = _sorted_eig(M, k)
    zeros = np.sqrt(eig) * R
    modes = []
    for m in range(k):
        y = vec[:, m]
        y = y / y[np.argmax(np.abs(y))]
        if y[0] < 0:
            y = -y
        modes.append(y)
    return r, zeros, np.array(modes)


def spherical_jn(nmax, x):
    """Spherical Bessel functions j_0..j_nmax at scalar x, by the three-term upward
    recurrence j_{n+1} = (2n+1)/x j_n - j_{n-1} (stable for n <~ x). j_0 = sin x / x,
    j_1 = sin x / x^2 - cos x / x. These are the radial functions in Mie scattering."""
    j = np.zeros(nmax + 1)
    j[0] = np.sin(x) / x
    if nmax >= 1:
        j[1] = np.sin(x) / x**2 - np.cos(x) / x
    for n in range(1, nmax):
        j[n + 1] = (2 * n + 1) / x * j[n] - j[n - 1]
    return j


def fourier_bessel_coeffs(f_vals, r, zeros, R=1.0):
    """Fourier-Bessel coefficients: project f(r) onto J_0(alpha_m r/R) using the
    orthogonality inner product <f,J_m> = int_0^R f J_0(alpha_m r/R) r dr, normalized
    by ||J_m||^2 = (R^2/2) J_1(alpha_m)^2. Returns the coefficient array."""
    from numpy import trapezoid as _tz
    coeffs = []
    for a in zeros:
        basis = _besselj0(a * r / R)
        num = _tz(f_vals * basis * r, r)
        norm = _tz(basis**2 * r, r)
        coeffs.append(num / norm)
    return np.array(coeffs)


def fourier_bessel_reconstruct(coeffs, r, zeros, R=1.0):
    """Sum the Fourier-Bessel series back: f(r) ~ sum c_m J_0(alpha_m r/R)."""
    return sum(c * _besselj0(a * r / R) for c, a in zip(coeffs, zeros))


def _besselj0(x):
    """J_0 via its integral 1/pi int_0^pi cos(x sin t) dt (vectorized, no scipy)."""
    t = np.linspace(0, np.pi, 400)
    x = np.asarray(x)
    return np.trapezoid(np.cos(np.outer(x, np.sin(t))), t, axis=-1) / np.pi


if __name__ == "__main__":
    known = {0: [2.4048, 5.5201, 8.6537], 1: [3.8317, 7.0156], 2: [5.1356, 8.4172]}
    for n, ref in known.items():
        z = bessel_zeros(n, k=len(ref))
        print(f"J_{n} zeros: matrix eig {np.round(z,4)}  vs known {ref}")
    print("\nspherical Bessel j_n(2.5):", np.round(spherical_jn(4, 2.5), 5))
    print("recurrence check j0+j2 vs (3/x)j1:",
          spherical_jn(2, 2.5)[0] + spherical_jn(2, 2.5)[2],
          "vs", 3 / 2.5 * spherical_jn(2, 2.5)[1])
