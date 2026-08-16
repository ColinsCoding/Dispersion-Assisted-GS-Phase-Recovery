"""Test dgs/em_lagrangian_action.py: deriving Maxwell's equations from the
electromagnetic Lagrangian density symbolically, verifying each algebraic
step (field tensor vs. E/B, Lagrangian decomposition, canonical momentum,
Euler-Lagrange field equation, Bianchi identity) rather than assuming any
of them."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from dgs.em_lagrangian_action import (
    COORDS, ETA, EPS0, MU0, C, MU0_FROM_EPS0,
    four_potential_symbols, four_current_symbols,
    contravariant_and_covariant_potential, field_strength_tensor,
    E_and_B_from_potentials, verify_field_tensor_matches_E_B,
    free_lagrangian_density, verify_lagrangian_reduces_to_field_energy,
    canonical_momentum_density, verify_canonical_momentum_matches_F,
    euler_lagrange_maxwell_equation, verify_gauss_and_ampere_maxwell,
    verify_bianchi_identity, derive_maxwell_from_lagrangian,
)

V, Ax, Ay, Az = four_potential_symbols()
rho, Jx, Jy, Jz = four_current_symbols()
A_up, A_lo = contravariant_and_covariant_potential(V, Ax, Ay, Az)
J_up = [C * rho, Jx, Jy, Jz]

F_lo, F_up = field_strength_tensor(A_lo)
E, B = E_and_B_from_potentials(V, Ax, Ay, Az)

# 1. field_strength_tensor: F must be antisymmetric (F_{mu nu} = -F_{nu mu}),
#    and every diagonal entry must be exactly 0 -- structural properties of
#    F = dA that hold for ANY potential, checked directly rather than assumed
for mu in range(4):
    assert sp.simplify(F_lo[mu, mu]) == 0, f"F_lo[{mu},{mu}] should be 0"
    for nu in range(4):
        assert sp.simplify(F_lo[mu, nu] + F_lo[nu, mu]) == 0, \
            f"F_lo[{mu},{nu}] should be -F_lo[{nu},{mu}]"

# 2. verify_field_tensor_matches_E_B: must succeed (returns True) for the
#    generic potential, and must NOT silently pass on a broken definition
assert verify_field_tensor_matches_E_B(F_lo, E, B) is True

broken_F = sp.zeros(4, 4)  # all-zero tensor won't match nonzero E, B
try:
    verify_field_tensor_matches_E_B(broken_F, E, B)
    raise AssertionError("expected AssertionError for a field tensor that doesn't match E, B")
except AssertionError as e:
    assert "did not simplify to 0" in str(e)

# 3. free_lagrangian_density / verify_lagrangian_reduces_to_field_energy
L_free = free_lagrangian_density(F_lo, F_up)
assert verify_lagrangian_reduces_to_field_energy(L_free, E, B) is True

# a Lagrangian off by a factor of 2 must be caught, not pass silently
try:
    verify_lagrangian_reduces_to_field_energy(2 * L_free, E, B)
    raise AssertionError("expected AssertionError for a Lagrangian off by a factor of 2")
except AssertionError as e:
    assert "did not reduce to" in str(e)

# 4. canonical_momentum_density / verify_canonical_momentum_matches_F
pi_generic, F_up_generic = canonical_momentum_density()
assert verify_canonical_momentum_matches_F(pi_generic, F_up_generic) is True

wrong_pi = pi_generic + sp.ones(4, 4)  # perturb every entry
try:
    verify_canonical_momentum_matches_F(wrong_pi, F_up_generic)
    raise AssertionError("expected AssertionError for a perturbed canonical momentum")
except AssertionError as e:
    assert "!=" in str(e)

# pi^{mu nu} must itself be antisymmetric, inherited directly from F^{mu nu}
for mu in range(4):
    for nu in range(4):
        assert sp.simplify(pi_generic[mu, nu] + pi_generic[nu, mu]) == 0

# 5. euler_lagrange_maxwell_equation / verify_gauss_and_ampere_maxwell:
#    the RESIDUAL d_mu F^(mu nu) - mu0 J^nu must vanish identically ONLY
#    once J^nu is exactly the source implied by Gauss/Ampere-Maxwell -- i.e.
#    div_F (not the residual against an arbitrary J) is what verify_* checks
div_F, residual = euler_lagrange_maxwell_equation(F_up, J_up)
assert len(div_F) == 4
assert verify_gauss_and_ampere_maxwell(div_F, E, B) is True

# a div_F entry that's off (e.g. missing the x1 partial) must be caught
broken_div_F = list(div_F)
broken_div_F[0] = div_F[0] + sp.Symbol("stray_leftover_term")
try:
    verify_gauss_and_ampere_maxwell(broken_div_F, E, B)
    raise AssertionError("expected AssertionError for a broken div_F entry")
except AssertionError as e:
    assert "did not match its familiar EM form" in str(e)

# 6. verify_bianchi_identity: must hold for F = dA (built from a potential),
#    and must NOT hold for an arbitrary antisymmetric tensor not derived
#    from any potential (confirms the check is actually discriminating)
assert verify_bianchi_identity(F_lo) is True

x0, x1, x2, x3 = COORDS
generic_antisym = sp.zeros(4, 4)
funcs = {}
for mu in range(4):
    for nu in range(mu + 1, 4):
        f = sp.Function(f"G_{mu}{nu}")(*COORDS)
        generic_antisym[mu, nu] = f
        generic_antisym[nu, mu] = -f
try:
    verify_bianchi_identity(generic_antisym)
    raise AssertionError("expected AssertionError: an arbitrary antisymmetric tensor "
                          "need not satisfy the Bianchi identity")
except AssertionError as e:
    assert "Bianchi identity failed" in str(e)

# 7. derive_maxwell_from_lagrangian: the full orchestration, every flag True
result = derive_maxwell_from_lagrangian()
for key in ("field_tensor_matches_E_B", "lagrangian_reduces_to_field_energy",
            "canonical_momentum_matches_F", "maxwell_inhomogeneous_verified",
            "bianchi_identity_verified"):
    assert result[key] is True, f"{key} was not True"

# 8. sanity: mu0 * eps0 * c^2 = 1 is the identity this whole derivation
#    leans on (Gauss's law and the field-energy decomposition both use it)
assert sp.simplify(MU0_FROM_EPS0 * EPS0 * C**2 - 1) == 0

print("all dgs.em_lagrangian_action tests passed")
