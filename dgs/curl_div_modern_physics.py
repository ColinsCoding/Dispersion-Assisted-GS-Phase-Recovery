"""Curl and divergence, in 20th-century physics rather than classical E&M:

  1. DIVERGENCE -- quantum probability current. Schrodinger's equation
     implies a CONTINUITY equation for probability itself,
     d|psi|^2/dt + dJ/dx = 0, exactly the classical charge-conservation
     structure (dgs.causality.continuity_residual, reused here rather than
     reimplemented) but now for a probability density that has no
     classical analog -- probability doesn't leak, it only flows.

  2. CURL -- gauge invariance and the Aharonov-Bohm effect. curl(grad(chi))
     is IDENTICALLY zero for any scalar chi -- this single vector-calculus
     identity is why the vector potential A can be changed (A -> A+grad(chi))
     without changing the physical B field, the seed of all of modern
     gauge theory (QED, the Standard Model). But the CONVERSE -- "curl-free
     implies a pure gradient" -- only holds in a SIMPLY CONNECTED region.
     Outside an infinite solenoid, curl(A)=0 (matching B=0 there) even
     though A is NOT a pure gradient (the region has a hole), and this
     topological loophole is exactly why an electron circling a solenoid
     picks up a real, measurable phase shift proportional to the enclosed
     flux -- the Aharonov-Bohm effect (predicted 1959, confirmed by
     Tonomura's 1986 electron holography experiment).
"""

import numpy as np
import sympy as sp

from dgs.causality import continuity_residual

HBAR = 1.0545718e-34
M_ELECTRON = 9.1093837e-31


def free_particle_wavepacket(x, t, x0, k0, sigma0, hbar=HBAR, m=M_ELECTRON):
    """The EXACT analytic solution for a free-particle Gaussian wave
    packet's time evolution under the Schrodinger equation (closed form,
    not numerically integrated) -- a genuinely time-dependent quantum
    state to test the continuity equation against."""
    if sigma0 <= 0:
        raise ValueError("sigma0 must be positive")
    X, T = np.meshgrid(np.asarray(x, dtype=float), np.asarray(t, dtype=float))
    sigma_t = sigma0 * np.sqrt(1 + (hbar * T / (2 * m * sigma0 ** 2)) ** 2)
    a_t = sigma0 ** 2 + 1j * hbar * T / (2 * m)
    norm = (2 * np.pi * sigma0 ** 2) ** (-0.25)
    # NOTE: denominator is 4*a_t (not 4*sigma0*a_t -- that extra sigma0 factor
    # was a bug caught by the continuity-equation check below failing at 50%
    # relative residual instead of ~0)
    envelope = np.exp(-(X - x0 - hbar * k0 * T / m) ** 2 / (4 * a_t))
    phase = np.exp(1j * (k0 * X - hbar * k0 ** 2 * T / (2 * m)))
    # NOTE: sqrt(sigma0**2/a_t), not sqrt(sigma0/a_t) -- the missing square on
    # sigma0 was the second half of the same normalization bug
    normalization = np.sqrt(sigma0 ** 2 / a_t)
    psi = norm * normalization * envelope * phase
    return psi, X, T, sigma_t


def probability_density(psi):
    """rho = |psi|^2."""
    return np.abs(psi) ** 2


def probability_current(psi, x):
    """J = (hbar/m) * Im(psi* d(psi)/dx) -- the quantum probability
    current, the SAME role classical J plays in charge continuity, now
    for probability density instead of charge density."""
    hbar, m = HBAR, M_ELECTRON
    dpsi_dx = np.gradient(psi, x, axis=1)
    return (hbar / m) * np.imag(np.conj(psi) * dpsi_dx)


def verify_quantum_continuity(x, t, x0, k0, sigma0):
    """Verify d|psi|^2/dt + dJ/dx ~ 0 for the free-particle wave packet,
    using dgs.causality.continuity_residual (REUSED, not reimplemented --
    the same equation classical charge conservation obeys)."""
    psi, X, T, sigma_t = free_particle_wavepacket(x, t, x0, k0, sigma0)
    rho = probability_density(psi)
    J = probability_current(psi, x)
    residual = continuity_residual(rho, J, x, t)
    return residual, rho, J


def curl_of_gradient_is_zero_symbolic():
    """The identity curl(grad(chi)) = 0, verified symbolically for a
    GENERIC scalar field chi(x,y,z) -- not a specific example, the general
    case, which is what makes gauge transformations A -> A+grad(chi)
    universally safe."""
    x, y, z = sp.symbols("x y z", real=True)
    chi = sp.Function("chi")(x, y, z)
    grad_chi = sp.Matrix([sp.diff(chi, x), sp.diff(chi, y), sp.diff(chi, z)])
    curl_of_grad = sp.Matrix([
        sp.diff(grad_chi[2], y) - sp.diff(grad_chi[1], z),
        sp.diff(grad_chi[0], z) - sp.diff(grad_chi[2], x),
        sp.diff(grad_chi[1], x) - sp.diff(grad_chi[0], y),
    ])
    is_zero = all(sp.simplify(component) == 0 for component in curl_of_grad)
    return is_zero, curl_of_grad


def solenoid_vector_potential(r, R, B0):
    """Azimuthal vector potential of an infinite solenoid (radius R,
    interior field B0 along the axis): A_phi = B0*r/2 inside, B0*R^2/(2r)
    outside -- continuous at r=R by construction, chosen to reproduce the
    correct B field via curl."""
    r = np.asarray(r, dtype=float)
    if R <= 0 or np.any(r < 0):
        raise ValueError("R must be positive, r must be non-negative")
    return np.where(r < R, B0 * r / 2, B0 * R ** 2 / (2 * np.where(r == 0, 1, r)))


def curl_of_solenoid_A_outside(r, R, B0, dr=None):
    """(curl A)_z = (1/r) d(r*A_phi)/dr outside the solenoid, computed
    NUMERICALLY (central difference) -- should be ~0, matching B=0 there,
    even though A_phi itself is nonzero."""
    if dr is None:
        dr = r * 1e-6
    if np.any(r <= R):
        raise ValueError("this check is only valid outside the solenoid (r > R)")
    rA_plus = (r + dr) * solenoid_vector_potential(r + dr, R, B0)
    rA_minus = (r - dr) * solenoid_vector_potential(r - dr, R, B0)
    return (rA_plus - rA_minus) / (2 * dr) / r


def aharonov_bohm_phase(charge, B0, R, hbar=HBAR):
    """The AB phase shift for a charge circling OUTSIDE the solenoid:
    Delta_phi = (q/hbar) * (closed-loop integral of A.dl) = (q/hbar)*Phi_B,
    where Phi_B = B0*pi*R^2 is the TOTAL flux enclosed -- independent of
    the loop radius (as long as it's outside R), which is the whole point:
    curl(A)=0 along the entire path, yet the circulation is nonzero and
    purely topological (depends only on what's enclosed, not the path)."""
    flux = B0 * np.pi * R ** 2
    return charge * flux / hbar


def circulation_of_A_outside(r_loop, R, B0):
    """Direct line-integral circulation of A around a circle of radius
    r_loop > R: A_phi * 2*pi*r_loop -- verified to equal the total
    enclosed flux, INDEPENDENT of r_loop, confirming the topological
    (path-independent-given-what's-enclosed) nature of the effect."""
    if r_loop <= R:
        raise ValueError("r_loop must be outside the solenoid (r_loop > R)")
    A_phi = solenoid_vector_potential(np.array([r_loop]), R, B0)[0]
    return A_phi * 2 * np.pi * r_loop


if __name__ == "__main__":
    print("=== 1. Quantum probability current: the continuity equation ===")
    x = np.linspace(-5e-9, 5e-9, 400)
    t = np.linspace(0, 1e-15, 60)
    residual, rho, J = verify_quantum_continuity(x, t, x0=0.0, k0=5e9, sigma0=1e-9)
    interior = (slice(5, -5), slice(5, -5))
    max_residual = np.max(np.abs(residual[interior]))
    max_scale = np.max(np.abs(rho)) / (t[1] - t[0])
    print(f"max |d(rho)/dt + dJ/dx| (interior): {max_residual:.3e}")
    print(f"typical scale of d(rho)/dt alone:   {max_scale:.3e}")
    print(f"relative residual: {max_residual/max_scale:.2e}  (should be small: probability is conserved)")

    print("\n=== 2. curl(grad(chi)) = 0, symbolically, for a GENERIC chi ===")
    is_zero, curl_of_grad = curl_of_gradient_is_zero_symbolic()
    print(f"curl(grad(chi)) == 0 for arbitrary chi(x,y,z): {is_zero}")

    print("\n=== 3. Aharonov-Bohm effect: curl-free but not gradient-free ===")
    R, B0 = 1e-6, 0.5   # solenoid radius (m), interior field (T)
    r_test = np.linspace(1.5 * R, 10 * R, 20)
    curl_outside = curl_of_solenoid_A_outside(r_test, R, B0)
    print(f"max |curl(A)| outside the solenoid: {np.max(np.abs(curl_outside)):.3e}  (should be ~0, matches B=0)")

    circulations = [circulation_of_A_outside(r, R, B0) for r in [1.5*R, 3*R, 10*R]]
    flux = B0 * np.pi * R ** 2
    print(f"circulation of A at r=1.5R, 3R, 10R: {circulations}")
    print(f"total enclosed flux Phi_B = {flux:.4e} Wb  (matches all three -- path-independent)")

    q_electron = -1.602176634e-19
    phase = aharonov_bohm_phase(q_electron, B0, R)
    print(f"\nAharonov-Bohm phase shift for an electron circling this solenoid: {phase:.4f} rad")
    print("Measurable via electron interference (Tonomura 1986) -- direct experimental")
    print("confirmation that A is physically real, not just a mathematical convenience,")
    print("precisely because curl(A)=0 outside does NOT make A a pure gradient there.")
