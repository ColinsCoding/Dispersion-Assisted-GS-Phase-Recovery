"""Band theory of solids: the Kronig-Penney model, derived (not assumed) from
Bloch's theorem + wavefunction matching across a periodic delta-function
potential, then swept numerically to find allowed bands vs forbidden gaps --
the actual first-principles answer to "why do conductors/semiconductors/
insulators differ" (dgs.solid_state_physics.classify_material takes the band
gap as a given number; this module is where that gap comes FROM).

DERIVATION (see derive_dispersion_relation() -- this is executed SymPy, not
copied from a textbook):
  Period a, delta potential V(x) = (hbar^2 P)/(m a) * delta(x) at each cell
  boundary. In 0<x<a: psi(x) = A sin(alpha x) + B cos(alpha x), alpha^2 = 2mE/hbar^2.
  Bloch's theorem gives psi just left of x=0 in terms of psi at x=a (one
  period back): psi(0^-) = exp(-ika) psi(a^-). Matching continuity of psi
  and the derivative JUMP the delta function forces at x=0 gives a 2x2
  homogeneous linear system in (A,B); a nontrivial solution needs det=0,
  which SymPy reduces to:

      cos(k*a) = cos(alpha*a) + (P/(alpha*a)) * sin(alpha*a)

  The right-hand side is a real number that can exceed [-1,1] -- wherever
  it does, there is NO real k solving the equation, i.e. no propagating
  Bloch wave exists at that energy. That's a band GAP, falling directly out
  of the algebra, not asserted.

NumPy for the sweep; Torch (GPU) batched version included, guarded import
since torch is py-3.12 only in this repo (falls back to NumPy on py-3.13).
"""

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


def derive_dispersion_relation():
    """Executes the SymPy boundary-matching derivation described in the
    module docstring and returns (rhs_expr, alpha, a, P) where the dispersion
    relation is cos(k*a) = rhs_expr. This IS the derivation, not a comment
    describing one -- run it and see cos(alpha*a) + P*sin(alpha*a)/(alpha*a)
    fall out of a 2x2 determinant."""
    import sympy as sp

    A, B, alpha, a, k, P = sp.symbols('A B alpha a k P', real=True)
    I = sp.I

    psi_at_a = A * sp.sin(alpha * a) + B * sp.cos(alpha * a)
    dpsi_at_a = alpha * A * sp.cos(alpha * a) - alpha * B * sp.sin(alpha * a)
    psi_0_minus = sp.exp(-I * k * a) * psi_at_a          # Bloch's theorem
    dpsi_0_minus = sp.exp(-I * k * a) * dpsi_at_a
    psi_0_plus = B
    dpsi_0_plus = alpha * A

    eq1 = psi_0_plus - psi_0_minus                                    # continuity
    eq2 = (dpsi_0_plus - dpsi_0_minus) - (2 * P / a) * psi_0_plus      # delta-function jump

    M = sp.linear_eq_to_matrix([eq1, eq2], [A, B])[0]
    det_cleared = sp.expand(M.det() * sp.exp(I * k * a)).rewrite(sp.cos)
    det_cleared = sp.simplify(det_cleared)

    rhs = sp.solve(sp.Eq(det_cleared, 0), sp.cos(k * a))[0]
    rhs = sp.simplify(rhs)
    return rhs, alpha, a, P


def kronig_penney_rhs(alpha_a, P):
    """Numeric RHS of the dispersion relation cos(ka) = rhs, as a function of
    the dimensionless variable alpha*a (proportional to sqrt(E)) and the
    dimensionless barrier strength P. |rhs| <= 1 -> allowed band (a real k
    exists); |rhs| > 1 -> forbidden gap (no propagating Bloch state)."""
    alpha_a = np.asarray(alpha_a, dtype=float)
    near_zero = np.abs(alpha_a) < 1e-9
    safe = np.where(near_zero, 1.0, alpha_a)
    sinc_term = np.where(near_zero, 1.0, np.sin(safe) / safe)   # limit at alpha_a->0 is 1
    return np.cos(alpha_a) + P * sinc_term


def find_bands(P, alpha_a_max=10.0, n_points=20000):
    """Sweep alpha*a in (0, alpha_a_max] and classify each point as ALLOWED
    (|rhs|<=1, a propagating Bloch state exists) or FORBIDDEN (a gap).
    Returns (alpha_a_grid, rhs_values, allowed_mask)."""
    if alpha_a_max <= 0:
        raise ValueError(f"alpha_a_max must be positive, got {alpha_a_max}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    alpha_a = np.linspace(1e-6, alpha_a_max, n_points)
    rhs = kronig_penney_rhs(alpha_a, P)
    allowed = np.abs(rhs) <= 1.0
    return alpha_a, rhs, allowed


def gap_edges(P, alpha_a_max=10.0, n_points=20000):
    """Locations (in alpha*a) where the allowed/forbidden classification
    flips -- the band edges. Free-particle case (P=0) has NONE (rhs=cos is
    always in [-1,1], one continuous band, no gaps at all)."""
    alpha_a, _, allowed = find_bands(P, alpha_a_max, n_points)
    flips = np.where(np.diff(allowed.astype(int)) != 0)[0]
    return alpha_a[flips]


def find_bands_torch(P, alpha_a_max=10.0, n_points=20000, device=None):
    """Same sweep as find_bands, batched on GPU via torch if available.
    torch is py-3.12 ONLY in this repo -- run this under py -3.12. Falls
    back to the NumPy path (with a printed notice) if torch isn't installed,
    rather than silently returning nothing."""
    if not _TORCH_AVAILABLE:
        print("torch not available in this environment (py-3.12 only) -- "
              "falling back to find_bands (NumPy).")
        return find_bands(P, alpha_a_max, n_points)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    alpha_a = torch.linspace(1e-6, alpha_a_max, n_points, dtype=torch.float64, device=dev)
    near_zero = alpha_a.abs() < 1e-9
    safe = torch.where(near_zero, torch.ones_like(alpha_a), alpha_a)
    sinc_term = torch.where(near_zero, torch.ones_like(alpha_a), torch.sin(safe) / safe)
    rhs = torch.cos(alpha_a) + P * sinc_term
    allowed = rhs.abs() <= 1.0
    return alpha_a.cpu().numpy(), rhs.cpu().numpy(), allowed.cpu().numpy()


if __name__ == "__main__":
    rhs_expr, alpha, a, P = derive_dispersion_relation()
    print("Derived dispersion relation: cos(k*a) =", rhs_expr)

    print("\nP=0 (no potential): free particle, should have ZERO gaps")
    gaps_free = gap_edges(0.0)
    print(f"  gap edges found: {len(gaps_free)} (expect 0)")

    print("\nP=5 (real periodic potential): gaps should open at alpha*a ~ n*pi")
    gaps_P5 = gap_edges(5.0)
    print(f"  gap edges found near: {np.round(gaps_P5, 2)}")
    print(f"  (compare to n*pi = {np.round(np.pi*np.arange(1, len(gaps_P5)//2+2), 2)})")

    print(f"\ntorch available: {_TORCH_AVAILABLE}")
    alpha_a_t, rhs_t, allowed_t = find_bands_torch(5.0, n_points=2000)
    alpha_a_n, rhs_n, allowed_n = find_bands(5.0, n_points=2000)
    print(f"torch vs numpy sweep agree: {np.allclose(rhs_t, rhs_n)}")
