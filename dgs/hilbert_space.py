"""Hilbert space formalism: bra-ket, operators, uncertainty, projection.

HILBERT SPACE = complete inner product space.
  L²[a,b]: square-integrable functions. Inner product: <f|g> = int f*(x)g(x) dx.
  C^n: finite-dim. Inner product: <u|v> = u† v (conjugate transpose dot).
  L²(R, exp(-x²)): weighted space — Hermite polynomial basis.

TRIANGLE INEQUALITY IN HILBERT SPACE:
  ||f + g|| <= ||f|| + ||g||    (triangle inequality)
  |<f|g>|   <= ||f|| * ||g||   (Cauchy-Schwarz)
  These are NOT approximations — they are EXACT in any inner product space.

  "Triangle approximation" for operators:
    ||A + B|| <= ||A|| + ||B||  (operator norm triangle inequality)
  Key insight: the uncertainty principle IS Cauchy-Schwarz applied to operators.

UNCERTAINTY PRINCIPLE FROM CAUCHY-SCHWARZ:
  For observables A, B with [A,B] = iC:
    sigma_A * sigma_B >= |<C>| / 2
  Heisenberg: [x,p] = i*hbar -> sigma_x * sigma_p >= hbar/2
  Time-bandwidth: [t,omega] pairs in Fourier sense -> Delta_t * Delta_omega >= 1/2
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict, List, Optional, Tuple


HBAR = 1.0546e-34   # J·s


# ════════════════════════════════════════════════════════════════════════════
# §1  INNER PRODUCT AND NORMS
# ════════════════════════════════════════════════════════════════════════════

def inner_product(psi: np.ndarray, phi: np.ndarray,
                   x: Optional[np.ndarray] = None,
                   weight: Optional[np.ndarray] = None) -> complex:
    """<psi|phi> in L² or C^n.

    If x is provided: continuous <psi|phi> = int psi*(x) phi(x) w(x) dx.
    If x is None: discrete <psi|phi> = psi† @ phi.
    """
    if x is None:
        return complex(np.conj(psi) @ phi)
    w = weight if weight is not None else np.ones_like(x)
    integrand = np.conj(psi) * phi * w
    return complex(np.trapezoid(integrand, x))


def norm_L2(psi: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    """L² norm: ||psi|| = sqrt(<psi|psi>)."""
    return float(np.sqrt(np.abs(inner_product(psi, psi, x))))


def normalize(psi: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
    """Return normalized |psi> / ||psi||."""
    n = norm_L2(psi, x)
    return psi / (n + 1e-300)


def gram_schmidt_L2(funcs: List[np.ndarray],
                     x: np.ndarray) -> List[np.ndarray]:
    """Gram-Schmidt orthonormalization in L²[x].

    Input:  list of functions on grid x
    Output: orthonormal basis {phi_i} such that <phi_i|phi_j> = delta_ij
    """
    basis = []
    for f in funcs:
        v = f.copy().astype(complex)
        for phi in basis:
            v -= inner_product(phi, v, x) * phi
        n = norm_L2(v, x)
        if n > 1e-12:
            basis.append(v / n)
    return basis


def triangle_inequality_check(f: np.ndarray, g: np.ndarray,
                                x: np.ndarray) -> Dict:
    """Verify ||f+g|| <= ||f||+||g|| and |<f,g>| <= ||f||*||g|| in L²."""
    nf  = norm_L2(f, x)
    ng  = norm_L2(g, x)
    nfg = norm_L2(f + g, x)
    ip  = inner_product(f, g, x)
    return {
        "norm_f":          nf,
        "norm_g":          ng,
        "norm_f_plus_g":   nfg,
        "sum_norms":       nf + ng,
        "triangle_holds":  bool(nfg <= nf + ng + 1e-10),
        "inner_product":   ip,
        "CS_lhs":          abs(ip),
        "CS_rhs":          nf * ng,
        "cauchy_schwarz":  bool(abs(ip) <= nf * ng + 1e-10),
        "correlation":     float(np.real(ip) / (nf * ng + 1e-300)),
    }


# ════════════════════════════════════════════════════════════════════════════
# §2  QUANTUM STATES AND OPERATORS
# ════════════════════════════════════════════════════════════════════════════

def qubit_state(theta: float, phi: float) -> np.ndarray:
    """Bloch sphere parameterization: |psi> = cos(theta/2)|0> + e^{i*phi}*sin(theta/2)|1>.

    |0> = north pole (spin up), |1> = south pole (spin down).
    theta in [0, pi], phi in [0, 2*pi).
    """
    return np.array([np.cos(theta/2),
                     np.exp(1j*phi) * np.sin(theta/2)])


def expectation_value(psi: np.ndarray, A: np.ndarray,
                       x: Optional[np.ndarray] = None) -> complex:
    """<A> = <psi|A|psi> for operator A (matrix or function-space operator).

    Matrix case: <psi|A|psi> = psi† @ A @ psi.
    """
    if A.ndim == 2:
        return complex(np.conj(psi) @ A @ psi)
    # Diagonal operator: A is a 1D array (multiplication operator)
    return inner_product(psi, A * psi, x)


def variance_operator(psi: np.ndarray, A: np.ndarray,
                       x: Optional[np.ndarray] = None) -> float:
    """Var_A = <A²> - <A>² = <(A - <A>)²>."""
    mean_A  = expectation_value(psi, A, x)
    mean_A2 = expectation_value(psi, A @ A if A.ndim == 2 else A**2, x)
    return float(np.real(mean_A2 - mean_A**2))


def commutator_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """[A, B] = AB - BA."""
    return A @ B - B @ A


# ════════════════════════════════════════════════════════════════════════════
# §3  UNCERTAINTY PRINCIPLE
# ════════════════════════════════════════════════════════════════════════════

def heisenberg_uncertainty(psi_x: np.ndarray,
                            x: np.ndarray) -> Dict:
    """Compute sigma_x * sigma_p for a wavefunction psi(x) and verify >= hbar/2.

    p_hat = -i*hbar * d/dx (momentum operator).
    sigma_x = sqrt(<x²> - <x>²)
    sigma_p = sqrt(<p²> - <p>²)
    """
    dx = x[1] - x[0]
    # Normalize
    psi = normalize(psi_x, x)

    # <x> and <x²>
    x_mean  = float(np.real(inner_product(psi, x * psi, x)))
    x2_mean = float(np.real(inner_product(psi, x**2 * psi, x)))
    sigma_x = np.sqrt(max(0, x2_mean - x_mean**2))

    # Momentum via FFT: p-space wavefunction
    # phi(k) = FT[psi(x)]; p = hbar*k
    n = len(x)
    psi_k = np.fft.fft(psi) * dx / np.sqrt(2*np.pi)
    k = np.fft.fftfreq(n, d=dx) * 2 * np.pi
    dk = k[1] - k[0] if n > 1 else 1.0

    norm_k = float(np.trapezoid(np.abs(psi_k)**2, k))
    psi_k_norm = psi_k / np.sqrt(abs(norm_k) + 1e-300)

    k_mean  = float(np.real(np.trapezoid(k * np.abs(psi_k_norm)**2, k)))
    k2_mean = float(np.real(np.trapezoid(k**2 * np.abs(psi_k_norm)**2, k)))
    sigma_k = np.sqrt(max(0, k2_mean - k_mean**2))
    sigma_p = HBAR * sigma_k

    product = sigma_x * sigma_p
    lower_bound = HBAR / 2

    return {
        "sigma_x":       sigma_x,
        "sigma_p":       sigma_p,
        "sigma_k":       sigma_k,
        "x_mean":        x_mean,
        "k_mean":        k_mean,
        "product":       product,
        "lower_bound":   lower_bound,
        "uncertainty_satisfied": bool(product >= lower_bound * 0.99),
        "ratio":         float(product / lower_bound),
        "gaussian_saturates": (sigma_x * sigma_k >= 0.49),  # Gaussian achieves minimum
    }


def time_bandwidth_product(t: np.ndarray, E_t: np.ndarray) -> Dict:
    """Time-bandwidth product Delta_t * Delta_omega >= 1/2 for optical pulse.

    Connects directly to TS-DFT: dispersive Fourier transform maps time -> frequency.
    Minimum TBP = 0.5 for transform-limited (unchirped) Gaussian pulse.
    """
    # Intensity envelope
    I_t = np.abs(E_t)**2
    I_t_norm = I_t / (np.trapezoid(I_t, t) + 1e-300)

    # RMS pulse duration
    t_mean  = float(np.trapezoid(t * I_t_norm, t))
    t2_mean = float(np.trapezoid(t**2 * I_t_norm, t))
    delta_t = np.sqrt(max(0, t2_mean - t_mean**2))

    # Spectrum
    dt = t[1] - t[0]
    n = len(t)
    E_omega = np.fft.fft(E_t) * dt
    omega = np.fft.fftfreq(n, d=dt) * 2 * np.pi
    I_omega = np.abs(E_omega)**2
    I_omega_norm = I_omega / (np.trapezoid(I_omega, omega) + 1e-300)

    omega_mean  = float(np.real(np.trapezoid(omega * I_omega_norm, omega)))
    omega2_mean = float(np.real(np.trapezoid(omega**2 * I_omega_norm, omega)))
    delta_omega = np.sqrt(max(0, omega2_mean - omega_mean**2))

    tbp = delta_t * delta_omega
    return {
        "delta_t":      delta_t,
        "delta_omega":  delta_omega,
        "TBP":          tbp,
        "TBP_limit":    0.5,
        "transform_limited": bool(tbp < 0.6),   # near min TBP
        "ratio":        tbp / 0.5,
        "chirp_estimate": tbp / 0.5 - 1.0,  # 0 = unchirped
    }


# ════════════════════════════════════════════════════════════════════════════
# §4  PROJECTION OPERATORS AND SPECTRAL DECOMPOSITION
# ════════════════════════════════════════════════════════════════════════════

def projection_operator(phi: np.ndarray) -> np.ndarray:
    """Projection onto |phi>: P = |phi><phi| (outer product).

    P² = P (idempotent), P† = P (Hermitian).
    """
    phi = phi / (np.linalg.norm(phi) + 1e-300)
    return np.outer(phi, np.conj(phi))


def spectral_decomposition(A: np.ndarray) -> Dict:
    """Spectral theorem: A = sum_i lambda_i |phi_i><phi_i| for Hermitian A.

    Also checks: is A Hermitian, compute projectors, reconstruct A.
    """
    is_herm = bool(np.allclose(A, A.conj().T, atol=1e-10))
    if is_herm:
        vals, vecs = np.linalg.eigh(A)
    else:
        vals, vecs = np.linalg.eig(A)

    n = A.shape[0]
    A_recon = np.zeros_like(A)
    projectors = []
    for i in range(n):
        P = projection_operator(vecs[:, i])
        A_recon += vals[i] * P
        projectors.append(P)

    return {
        "eigenvalues":  vals,
        "eigenvectors": vecs,
        "projectors":   projectors,
        "A_reconstructed": A_recon,
        "recon_error":  float(np.linalg.norm(A - A_recon)),
        "hermitian":    is_herm,
        "trace_check":  bool(abs(np.sum(vals) - np.trace(A)) < 1e-8),
    }


def density_matrix(psi: np.ndarray) -> np.ndarray:
    """Pure state density matrix: rho = |psi><psi|."""
    psi = psi / (np.linalg.norm(psi) + 1e-300)
    return np.outer(psi, np.conj(psi))


def von_neumann_entropy(rho: np.ndarray) -> float:
    """S = -Tr(rho * ln(rho)) for density matrix rho.

    Pure state: S = 0.  Maximally mixed: S = ln(n).
    """
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 1e-15]
    return float(-np.sum(vals * np.log(vals)))


# ════════════════════════════════════════════════════════════════════════════
# §5  HARMONIC OSCILLATOR IN HILBERT SPACE (LADDER OPERATORS)
# ════════════════════════════════════════════════════════════════════════════

def harmonic_oscillator_states(n_states: int = 5,
                                n_pts: int = 500,
                                x_max: float = 5.0) -> Dict:
    """Quantum harmonic oscillator: psi_n(x) = H_n(x)*exp(-x²/2)/sqrt(2^n * n! * sqrt(pi)).

    Uses dimensionless units: hbar = m = omega = 1.
    E_n = (n + 1/2) * hbar * omega.
    """
    x = np.linspace(-x_max, x_max, n_pts)
    dx = x[1] - x[0]
    psi_list = []
    energies = []

    # Build psi_n using recurrence on Hermite polynomials
    # H_0 = 1, H_1 = 2x, H_{n+1} = 2x*H_n - 2n*H_{n-1}
    H_prev = np.ones(n_pts)
    H_curr = 2 * x
    gauss = np.exp(-x**2 / 2)

    psi0 = gauss / np.pi**(1/4)
    psi1 = np.sqrt(2) * x * gauss / np.pi**(1/4)
    psi_list = [psi0, psi1]
    energies = [0.5, 1.5]

    for n in range(2, n_states):
        H_next = 2 * x * H_curr - 2 * (n-1) * H_prev
        import math
        norm = np.sqrt(2**n * float(math.factorial(n)) * np.sqrt(np.pi))
        psi_n = H_next * gauss / norm
        psi_list.append(psi_n)
        energies.append(n + 0.5)
        H_prev, H_curr = H_curr, H_next

    # Verify orthonormality
    ortho_errors = []
    for i in range(n_states):
        for j in range(i, min(i+3, n_states)):
            ip = float(np.real(np.trapezoid(psi_list[i] * psi_list[j], x)))
            expected = 1.0 if i == j else 0.0
            ortho_errors.append(abs(ip - expected))

    return {
        "x":           x,
        "psi":         psi_list[:n_states],
        "energies":    energies[:n_states],
        "n_states":    n_states,
        "max_ortho_error": float(max(ortho_errors)),
        "orthonormal": bool(max(ortho_errors) < 0.01),
    }


# ════════════════════════════════════════════════════════════════════════════
# §6  SYMPY: 5 HILBERT SPACE EQUATIONS
# ════════════════════════════════════════════════════════════════════════════

def hilbert_space_sympy_5() -> Dict:
    """5 key Hilbert space equations in SymPy."""
    # 1. Inner product
    eq1 = sp.Eq(sp.Symbol("<psi|phi>"),
                sp.Symbol("integral_psi*_phi_dx"))
    # 2. Triangle inequality
    norm_f, norm_g = sp.symbols("||f|| ||g||", positive=True)
    eq2 = sp.Eq(sp.Symbol("||f+g||"),
                sp.Le(sp.Symbol("||f+g||_val"), norm_f + norm_g))
    # 3. Cauchy-Schwarz
    eq3 = sp.Eq(sp.Symbol("|<f|g>|"),
                sp.Le(sp.Symbol("|<f|g>|_val"), norm_f * norm_g))
    # 4. Heisenberg uncertainty
    hbar = sp.Symbol("hbar", positive=True)
    sigma_x, sigma_p = sp.symbols("sigma_x sigma_p", positive=True)
    eq4 = sp.Eq(sp.Symbol("Heisenberg"),
                sp.Ge(sigma_x * sigma_p, hbar / 2))
    # 5. Spectral theorem A = sum lambda_i |phi_i><phi_i|
    lam_i = sp.Symbol("lambda_i")
    eq5 = sp.Eq(sp.Symbol("A"),
                sp.Sum(lam_i * sp.Symbol("|phi_i><phi_i|"),
                       (sp.Symbol("i"), 0, sp.Symbol("n"))))
    return {
        "inner_product":       eq1,
        "triangle_inequality": eq2,
        "cauchy_schwarz":      eq3,
        "heisenberg":          eq4,
        "spectral_theorem":    eq5,
    }


if __name__ == "__main__":
    import numpy as np

    print("=== Triangle Inequality in L² ===")
    x = np.linspace(0, 2*np.pi, 500)
    res = triangle_inequality_check(np.sin(x), np.cos(x), x)
    print(f"  ||sin|| = {res['norm_f']:.4f}, ||cos|| = {res['norm_g']:.4f}")
    print(f"  ||sin+cos|| = {res['norm_f_plus_g']:.4f} <= {res['sum_norms']:.4f}? {res['triangle_holds']}")
    print(f"  Cauchy-Schwarz: {res['cauchy_schwarz']}")

    print("\n=== Heisenberg Uncertainty: Gaussian psi ===")
    x = np.linspace(-10, 10, 2048)
    sigma0 = 1.0
    psi = np.exp(-x**2 / (4*sigma0**2)) / (np.pi*sigma0**2)**0.25
    res = heisenberg_uncertainty(psi, x)
    print(f"  sigma_x = {res['sigma_x']:.4f} m")
    print(f"  sigma_p = {res['sigma_p']:.4e} kg*m/s")
    print(f"  sigma_x*sigma_p = {res['ratio']:.3f} * hbar/2")
    print(f"  Uncertainty satisfied: {res['uncertainty_satisfied']}")

    print("\n=== QHO States ===")
    qho = harmonic_oscillator_states(5)
    print(f"  E_n = {qho['energies']}")
    print(f"  Max ortho error = {qho['max_ortho_error']:.2e}")
    print(f"  Orthonormal: {qho['orthonormal']}")

    print("\n=== Spectral Decomposition of Pauli-X ===")
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sd = spectral_decomposition(sigma_x)
    print(f"  Eigenvalues: {np.real(sd['eigenvalues'])}")
    print(f"  Recon error: {sd['recon_error']:.2e}")
    print(f"  Hermitian: {sd['hermitian']}")

    print("\n=== Time-Bandwidth Product ===")
    t = np.linspace(-100e-15, 100e-15, 4096)  # 100 fs window
    T0 = 10e-15  # 10 fs pulse
    E_t = np.exp(-t**2 / (2*T0**2))
    tbp = time_bandwidth_product(t, E_t)
    print(f"  Delta_t = {tbp['delta_t']*1e15:.2f} fs")
    print(f"  Delta_omega = {tbp['delta_omega']/1e12:.2f} Trad/s")
    print(f"  TBP = {tbp['TBP']:.4f}  (limit 0.5)")
    print(f"  Transform-limited: {tbp['transform_limited']}")
