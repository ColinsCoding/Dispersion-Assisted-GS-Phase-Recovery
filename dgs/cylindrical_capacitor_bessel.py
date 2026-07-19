"""Where Bessel functions, a cylindrical capacitor, and coefficient-finding
all genuinely meet: the classic "potential inside a grounded cylindrical can"
boundary-value problem (Griffiths/Jackson). Laplace's EQUATION (not the
transform) in cylindrical coordinates, solved via separation of variables --
the Fourier-Bessel series this repo's diffraction grating / cylindrical
tunnel work already used J0 for, now actually solving an electrostatics BVP.

SETUP: a cylindrical can of radius a, height L. The curved side wall and the
bottom are grounded (V=0); the top cap is held at V=V0. Inside:

    Laplace's equation:  (1/r) d/dr(r dV/dr) + d^2V/dz^2 = 0

Separating V(r,z)=R(r)Z(z) gives R'' + R'/r + k^2 R = 0 (Bessel's equation,
order 0) and Z'' - k^2 Z = 0 (exponential/hyperbolic in z). Requiring
V=0 on the side wall (r=a) forces k = k_n/a where k_n are the ZEROS of J0
(so J0(k_n)=0 there); requiring V=0 at the bottom (z=0) picks sinh(k z)
over cosh. The general solution automatically satisfying both grounded
boundaries is

    V(r,z) = sum_n A_n * J0(k_n r/a) * sinh(k_n z/a)

and the coefficients A_n are FOUND (not guessed) by matching the one
remaining boundary condition V(r,L)=V0, using Bessel-function orthogonality:

    A_n = 2*V0 / (k_n * J1(k_n) * sinh(k_n*L/a))

NumPy/SciPy for the reference computation; Torch (GPU, batched over all n
terms in one shot) for the same calculation, cross-checked against it.
"""

import numpy as np
from scipy.special import j0, j1, jn_zeros

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False

EPS0 = 8.8541878128e-12  # F/m


def simple_cylindrical_capacitance(a, b, L, eps=EPS0):
    """The OTHER, simpler cylindrical capacitor: two infinite (or long)
    coaxial cylinders of radii a<b, length L: C = 2*pi*eps*L/ln(b/a).
    Not the same problem as the Bessel-series one below (no z-dependence,
    no Bessel functions needed) -- included since 'cylindrical capacitor'
    most often means this textbook formula first."""
    if not (0 < a < b):
        raise ValueError(f"need 0 < a < b, got a={a}, b={b}")
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    return 2 * np.pi * eps * L / np.log(b / a)


def bessel_coefficients(V0, a, L, n_terms=20):
    """A_n = 2*V0/(k_n*J1(k_n)*sinh(k_n*L/a)) for n=1..n_terms, k_n = the
    n-th zero of J0. Returns (k_n array, A_n array)."""
    if a <= 0 or L <= 0:
        raise ValueError("a and L must be positive")
    if n_terms < 1:
        raise ValueError(f"n_terms must be >= 1, got {n_terms}")
    k_n = jn_zeros(0, n_terms)
    A_n = 2 * V0 / (k_n * j1(k_n) * np.sinh(k_n * L / a))
    return k_n, A_n


def potential(r, z, V0, a, L, n_terms=20):
    """V(r,z) = sum_n A_n * J0(k_n r/a) * sinh(k_n z/a) -- the actual
    electrostatic potential inside the can, truncated to n_terms."""
    k_n, A_n = bessel_coefficients(V0, a, L, n_terms)
    r = np.asarray(r, dtype=float)
    total = np.zeros_like(r)
    for k, A in zip(k_n, A_n):
        total = total + A * j0(k * r / a) * np.sinh(k * z / a)
    return total


def bessel_coefficients_torch(V0, a, L, n_terms=20, device=None):
    """Same coefficient calculation, batched on GPU via torch -- all n_terms
    computed in one vectorized pass instead of a Python loop. torch is
    py-3.12 ONLY in this repo; falls back to the NumPy/SciPy path (with a
    printed notice) if torch isn't installed."""
    if not _TORCH_AVAILABLE:
        print("torch not available in this environment (py-3.12 only) -- "
              "falling back to bessel_coefficients (NumPy/SciPy).")
        return bessel_coefficients(V0, a, L, n_terms)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    k_n_np, _ = bessel_coefficients(V0, a, L, n_terms)   # zeros + J1 need scipy either way
    j1_vals = j1(k_n_np)
    k_n_t = torch.tensor(k_n_np, dtype=torch.float64, device=dev)
    j1_t = torch.tensor(j1_vals, dtype=torch.float64, device=dev)
    A_n_t = 2 * V0 / (k_n_t * j1_t * torch.sinh(k_n_t * L / a))
    return k_n_t.cpu().numpy(), A_n_t.cpu().numpy()


if __name__ == "__main__":
    a, L, V0 = 1.0, 2.0, 100.0

    C = simple_cylindrical_capacitance(a, 2*a, L)
    print(f"simple coaxial-cylinder capacitance (a={a}, b={2*a}, L={L}): {C:.4e} F")

    print(f"\nGrounded can, V0={V0} at top, n_terms=20:")
    k_n, A_n = bessel_coefficients(V0, a, L, n_terms=20)
    print("first 5 k_n (zeros of J0):", np.round(k_n[:5], 4))
    print("first 5 A_n:", np.round(A_n[:5], 4))

    # verify the series reconstructs V(r,L) ~= V0 across the interior (not
    # right at r=a, where the actual boundary condition has a discontinuity
    # -- Gibbs phenomenon there is real physics, not a bug)
    r_test = np.linspace(0, 0.95*a, 50)
    V_at_top = potential(r_test, L, V0, a, L, n_terms=50)
    print(f"\nreconstructed V(r,L) for r in [0, 0.95a], n_terms=50:")
    print(f"  should be ~{V0} (the top boundary condition) -- ", end="")
    print(f"min={V_at_top.min():.2f}, max={V_at_top.max():.2f}, mean={V_at_top.mean():.2f}")

    print(f"\ntorch available: {_TORCH_AVAILABLE}")
    k_n_t, A_n_t = bessel_coefficients_torch(V0, a, L, n_terms=20)
    print(f"torch vs numpy/scipy coefficients agree: {np.allclose(A_n_t, A_n)}")
