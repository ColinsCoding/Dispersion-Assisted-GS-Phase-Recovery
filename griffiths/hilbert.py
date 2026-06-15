"""Hilbert spaces: inner products, orthogonality, and basis expansions.

The single abstraction under Fourier series, Legendre/spherical-harmonic
expansions, Bessel (Fourier-Bessel) series, and quantum states: a vector space
with an inner product, an orthonormal basis, and the projection theorem --
expansion coefficients are inner products. SymPy-based, so orthogonality of the
classical families is *verified*, not assumed.
"""

import sympy as sp


def inner_product(f, g, var, a, b, weight=1):
    """<f, g> = integral_a^b conj(f) g w(x) dx (the L^2 inner product)."""
    return sp.integrate(sp.conjugate(f) * g * weight, (var, a, b))


def norm(f, var, a, b, weight=1):
    """||f|| = sqrt(<f, f>)."""
    return sp.sqrt(sp.simplify(inner_product(f, f, var, a, b, weight)))


def gram_matrix(basis, var, a, b, weight=1):
    """Matrix G_ij = <e_i, e_j>. Diagonal  =>  orthogonal basis;
    identity  =>  orthonormal."""
    n = len(basis)
    return sp.Matrix(n, n, lambda i, j:
                     sp.simplify(inner_product(basis[i], basis[j], var, a, b, weight)))


def is_orthogonal(basis, var, a, b, weight=1):
    """True iff the Gram matrix is diagonal (basis vectors mutually perpendicular)."""
    G = gram_matrix(basis, var, a, b, weight)
    return G == sp.diag(*[G[i, i] for i in range(G.rows)])


def expand(f, basis, var, a, b, weight=1):
    """Project f onto the (orthogonal) basis: c_i = <e_i, f> / <e_i, e_i>.

    Returns (coeffs, reconstruction). For an orthonormal complete basis the
    reconstruction converges to f -- the projection theorem in action.
    """
    coeffs, recon = [], sp.Integer(0)
    for e in basis:
        num = inner_product(e, f, var, a, b, weight)
        den = inner_product(e, e, var, a, b, weight)
        c = sp.simplify(num / den)
        coeffs.append(c)
        recon += c * e
    return coeffs, sp.simplify(recon)


# ── phasors: the simplest Hilbert space ─────────────────────────────
def phasor(amplitude, phase):
    """A phasor A e^{i phi}: a point in the 1-D complex Hilbert space C, the
    standing-still representation of the sinusoid A cos(wt + phi)."""
    return amplitude * sp.exp(sp.I * phase)


def phasor_inner(z, w):
    """Hermitian inner product on C^n: <z, w> = sum conj(z_i) w_i. For phasors,
    Re<z, w> is proportional to the time-averaged product (power/correlation)."""
    z = sp.Matrix(z) if hasattr(z, "__len__") else sp.Matrix([z])
    w = sp.Matrix(w) if hasattr(w, "__len__") else sp.Matrix([w])
    if z.shape != w.shape:
        raise ValueError("phasor vectors must have equal length")
    return sp.simplify(sum(sp.conjugate(z[i]) * w[i] for i in range(z.rows)))
