"""
_repl_qm_honors.py
==================
Upper-division quantum mechanics -- honors treatment.
No music. Full SymPy + scipy + matplotlib.

S1: The 6 Postulates of QM (precise statement + math)
S2: Operators, eigenvalues, commutators  [x, p] = i*hbar
S3: Harmonic oscillator via ladder operators a+, a-
    - n-th eigenstate, energy spectrum, matrix elements
S4: Hydrogen atom
    - radial equation, Bohr radius, energy levels
    - SymPy: psi_100, psi_210 wavefunctions
    - matplotlib: radial probability densities
S5: Spin-1/2 and Pauli matrices
    - Pauli algebra, spinor rotation, Bloch sphere
    - Stern-Gerlach measurement, spin-z eigenvalues
S6: Time-independent perturbation theory
    - First-order energy correction E_n^(1) = <n|H'|n>
    - First-order state correction
    - Example: linear Stark effect on harmonic oscillator
S7: Entanglement and Bell states
    - Composite Hilbert space, tensor product
    - 4 Bell states, density matrix, partial trace
    - CHSH inequality violation: quantum bound = 2*sqrt(2) > 2

Output: repl/_out_qm_honors.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import special as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sympy as sp
from sympy import (symbols, sqrt, exp, pi, oo, I, Rational, Matrix,
                   integrate, diff, simplify, conjugate, Abs,
                   factorial, assoc_laguerre, hermite, latex,
                   KroneckerDelta, eye, zeros)
import os

OUT = os.path.join(os.path.dirname(__file__), "_out_qm_honors.png")

HBAR = 1.0545718e-34  # J*s
M_E  = 9.10938e-31    # kg electron mass
E_C  = 1.60218e-19    # C
A0   = 5.29177e-11    # Bohr radius [m]
EV   = E_C            # 1 eV in Joules

SEP = "=" * 65

# ==========================================================
# S1: THE 6 POSTULATES OF QM
# ==========================================================
print(SEP)
print("SECTION 1: THE 6 POSTULATES OF QM")
print(SEP)

postulates = [
    ("1. State space",
     "The state of a quantum system is a ray |psi> in a\n"
     "   complex Hilbert space H. |psi> and c|psi> (c != 0)\n"
     "   describe the same physical state.\n"
     "   Normalization: <psi|psi> = 1."),

    ("2. Observables",
     "Every measurable physical quantity A is represented\n"
     "   by a Hermitian operator A_hat on H: A_hat = A_hat^dag.\n"
     "   Hermitian => real eigenvalues (measurable numbers).\n"
     "   Eigenvalue eq: A_hat |a_n> = a_n |a_n>"),

    ("3. Measurement",
     "Measuring A on |psi> = SUM_n c_n |a_n> yields\n"
     "   eigenvalue a_n with probability P(a_n) = |c_n|^2.\n"
     "   After measurement: |psi> collapses to |a_n>.\n"
     "   Expectation: <A> = <psi|A_hat|psi> = SUM_n a_n|c_n|^2"),

    ("4. Heisenberg uncertainty",
     "For any two observables A, B:\n"
     "   sigma_A * sigma_B >= (1/2)|<[A,B]>|\n"
     "   [x, p] = i*hbar  =>  sigma_x*sigma_p >= hbar/2\n"
     "   This is NOT instrument limitation -- it is fundamental."),

    ("5. Time evolution",
     "i*hbar * d|psi>/dt = H_hat |psi>    (Schrodinger equation)\n"
     "   Formal solution: |psi(t)> = U(t)|psi(0)>\n"
     "   where U(t) = exp(-i H_hat t / hbar)  (unitary).\n"
     "   For energy eigenstate: |psi_n(t)> = e^{-i E_n t/hbar}|n>"),

    ("6. Identical particles",
     "Bosons (integer spin): |psi> symmetric under exchange.\n"
     "   Fermions (half-integer spin): |psi> antisymmetric.\n"
     "   Fermions obey Pauli exclusion: no two in same state.\n"
     "   Electrons are fermions -> periodic table structure."),
]

for title, body in postulates:
    print(f"\n  POSTULATE {title}")
    for line in body.split("\n"):
        print(f"  {line}")

# ==========================================================
# S2: OPERATORS, EIGENVALUES, COMMUTATORS
# ==========================================================
print(f"\n{SEP}")
print("SECTION 2: OPERATORS, EIGENVALUES, COMMUTATORS")
print(SEP)

x_s, p_s, hbar_s, m_s, omega_s = symbols("x p hbar m omega", real=True, positive=True)
psi_s = symbols("psi", cls=sp.Function)

print("""
  POSITION AND MOMENTUM OPERATORS (position representation):
    x_hat: psi(x) -> x * psi(x)          [multiply by x]
    p_hat: psi(x) -> -i*hbar * d/dx      [differential operator]

  CANONICAL COMMUTATION RELATION:
    [x_hat, p_hat] psi = x_hat(p_hat psi) - p_hat(x_hat psi)
                       = x(-i hbar dpsi/dx) - (-i hbar)(d/dx)(x psi)
                       = -i hbar x dpsi/dx - (-i hbar)(psi + x dpsi/dx)
                       = -i hbar x dpsi/dx + i hbar psi + i hbar x dpsi/dx
                       = i hbar psi
    => [x_hat, p_hat] = i hbar   (canonical)
""")

# SymPy verify commutator via test function
x_var = symbols("x")
hb    = symbols("hbar", positive=True)
f     = sp.Function("f")(x_var)

xp_f  = x_var * (-I * hb * sp.diff(f, x_var))    # x*(p*f)
px_f  = -I * hb * sp.diff(x_var * f, x_var)       # p*(x*f)
comm  = sp.expand(xp_f - px_f)
print(f"  SymPy commutator [x,p]f = {comm}  =>  [x,p] = i*hbar  CHECK: {sp.simplify(comm / (I*hb*f))}")

print("""
  KEY COMMUTATOR ALGEBRA:
    [A, B] = -[B, A]                  (antisymmetry)
    [A, BC] = [A,B]C + B[A,C]         (product rule)
    [AB, C] = A[B,C] + [A,C]B
    [x^n, p] = i*hbar*n*x^(n-1)       (derived from canonical)
    [x, p^n] = i*hbar*n*p^(n-1)

  OBSERVABLE CONSEQUENCES:
    [x,p] = i*hbar  => sigma_x * sigma_p >= hbar/2
    [Lx,Ly] = i*hbar*Lz  => cannot know Lx,Ly simultaneously
    [H, L^2] = 0  => energy and L^2 share eigenstates (H atom)
    [H, Lz] = 0   => energy and Lz share eigenstates
    [Lx, Ly] != 0 => cannot specify all 3 angular momentum components
""")

# Ehrenfest theorem
print("  EHRENFEST THEOREM (quantum <-> classical connection):")
print("    d<x>/dt = <p>/m            (quantum = classical velocity)")
print("    d<p>/dt = -<dV/dx>         (quantum = classical force, if V linear)")
print("    For SHO (V quadratic): exact. For anharmonic: corrections appear.")

# ==========================================================
# S3: HARMONIC OSCILLATOR -- LADDER OPERATORS
# ==========================================================
print(f"\n{SEP}")
print("SECTION 3: HARMONIC OSCILLATOR -- LADDER OPERATORS")
print(SEP)

print("""
  HAMILTONIAN:
    H = p^2/(2m) + (1/2)*m*omega^2*x^2

  DEFINE LADDER OPERATORS:
    a_-  = sqrt(m*omega/(2*hbar)) * (x + i*p/(m*omega))   [lowering]
    a_+  = sqrt(m*omega/(2*hbar)) * (x - i*p/(m*omega))   [raising]
    a_+  = (a_-)^dag  (Hermitian conjugate)

  IN TERMS OF x AND p:
    x = sqrt(hbar/(2*m*omega)) * (a_+ + a_-)
    p = i*sqrt(m*omega*hbar/2) * (a_+ - a_-)

  NUMBER OPERATOR:
    N = a_+ a_-   (N hat)
    N |n> = n |n>   (counts quanta)

  COMMUTATORS:
    [a_-, a_+] = 1         (fundamental ladder relation)
    [N, a_+]   = +a_+      (a_+ raises by 1)
    [N, a_-]   = -a_-      (a_- lowers by 1)

  HAMILTONIAN REWRITTEN:
    H = hbar*omega*(N + 1/2) = hbar*omega*(a_+ a_- + 1/2)
    H|n> = hbar*omega*(n + 1/2)|n> = E_n|n>

  ENERGY SPECTRUM:
    E_n = hbar*omega*(n + 1/2),   n = 0, 1, 2, 3, ...
    Zero-point energy E_0 = hbar*omega/2  (cannot be zero -- uncertainty!)

  LADDER ACTIONS:
    a_- |n> = sqrt(n)   |n-1>    (lowering)
    a_+ |n> = sqrt(n+1) |n+1>    (raising)
    a_- |0> = 0                   (vacuum: no state below ground)
    |n>     = (a_+)^n / sqrt(n!) |0>  (build all states from vacuum)
""")

# Matrix elements in finite basis {|0>, |1>, |2>, |3>, |4>}
N_basis = 5
a_minus = np.zeros((N_basis, N_basis))
for n in range(1, N_basis):
    a_minus[n-1, n] = np.sqrt(n)   # <n-1|a_-|n>
a_plus = a_minus.T

N_op = a_plus @ a_minus            # number operator
H_op = N_op + 0.5 * np.eye(N_basis)  # H / (hbar*omega)

print("  MATRIX REPRESENTATION (basis |0>...|4>, units of hbar*omega):")
print(f"\n  a_- matrix:")
for row in a_minus:
    print("   ", "  ".join(f"{v:5.3f}" for v in row))
print(f"\n  H/(hbar*omega) = N + 1/2 diagonal:")
for i in range(N_basis):
    print(f"    |{i}>: {H_op[i,i]:.1f}")

# Position matrix elements <m|x|n> = sqrt(hbar/2mw) * (<m|a_+|n> + <m|a_-|n>)
# = sqrt(hbar/2mw) * (sqrt(n+1)*delta(m,n+1) + sqrt(n)*delta(m,n-1))
print("\n  POSITION MATRIX ELEMENTS <m|x|n> (units sqrt(hbar/2m*omega)):")
x_mat = a_plus + a_minus
for i in range(N_basis):
    row_str = "  ".join(f"{v:5.3f}" for v in x_mat[i])
    print(f"    <{i}|x|n>:  {row_str}")
print("  -> Only off-diagonal by 1: selection rule Delta_n = +/-1")

# Wavefunctions
xi_arr = np.linspace(-4, 4, 800)   # dimensionless xi = x*sqrt(mw/hbar)

print("\n  WAVEFUNCTIONS psi_n(xi):")
psi_n = {}
for n in range(5):
    Hn    = sc.hermite(n)(xi_arr)
    norm  = (np.pi**0.25 * np.sqrt(2**n * sc.factorial(n)))
    psi_n[n] = Hn * np.exp(-xi_arr**2 / 2) / norm
    print(f"    n={n}: E={n+0.5:.1f} hbar*omega,  max|psi|={np.max(np.abs(psi_n[n])):.4f}")

# ==========================================================
# S4: HYDROGEN ATOM
# ==========================================================
print(f"\n{SEP}")
print("SECTION 4: HYDROGEN ATOM")
print(SEP)

print("""
  SCHRODINGER EQUATION (spherical coords):
    [-hbar^2/(2m) nabla^2 - e^2/(4*pi*eps0*r)] psi = E psi

  SEPARATION: psi(r,theta,phi) = R(r) * Y_l^m(theta,phi)
    Y_l^m = spherical harmonics (eigenfunctions of L^2, Lz)
    L^2 Y_l^m = hbar^2 l(l+1) Y_l^m
    Lz  Y_l^m = hbar*m * Y_l^m
    l = 0,1,2,...,n-1   (s,p,d,f...)
    m = -l,-l+1,...,+l

  RADIAL EQUATION -> LAGUERRE POLYNOMIALS:
    R_nl(r) = -sqrt((2/n*a0)^3 * (n-l-1)! / (2n*(n+l)!^3))
              * exp(-r/(n*a0)) * (2r/(n*a0))^l
              * L_{n-l-1}^{2l+1}(2r/(n*a0))
    a0 = 4*pi*eps0*hbar^2/(m*e^2) = 0.52918 Angstrom (Bohr radius)

  ENERGY LEVELS:
    E_n = -m*e^4/(8*eps0^2*h^2) * 1/n^2
        = -13.6057 eV / n^2
    n = 1: E_1 = -13.606 eV  (ground state)
    n = 2: E_2 =  -3.401 eV  (first excited)
    n = 3: E_3 =  -1.512 eV
    Ionization energy = 13.606 eV

  DEGENERACY:
    For each n: l = 0..n-1, m = -l..+l  => n^2 states per n
    (2*n^2 with spin)
""")

E_n = lambda n: -13.6057 / n**2

print("  ENERGY LEVELS:")
for n in range(1, 6):
    degen = 2 * n**2
    print(f"    n={n}: E={E_n(n):8.4f} eV,  degeneracy 2n^2={degen},  "
          f"subshells: {', '.join(str(l) for l in range(n))} (l=0..{n-1})")

print(f"\n  SPECTRAL SERIES (emission lines):")
series = {"Lyman (UV, nf=1)": 1, "Balmer (vis, nf=2)": 2, "Paschen (IR, nf=3)": 3}
for name, nf in series.items():
    print(f"\n    {name}:")
    for ni in range(nf+1, nf+5):
        dE   = E_n(nf) - E_n(ni)          # photon energy (positive)
        lam  = 1240.0 / abs(dE)            # wavelength nm (E in eV)
        print(f"      {ni}->{nf}: dE={abs(dE):.4f} eV,  lambda={lam:.1f} nm")

# SymPy: ground state wavefunction
print("\n  SYMPY GROUND STATE psi_100(r):")
r_s, a_s = symbols("r a0", positive=True)
psi_100 = (1/sqrt(pi)) * (1/a_s)**sp.Rational(3,2) * exp(-r_s/a_s)
print(f"    psi_100 = {psi_100}")

# Normalization check: INT |psi|^2 * 4*pi*r^2 dr = 1
norm_integrand = psi_100**2 * 4 * pi * r_s**2
norm_result    = integrate(norm_integrand, (r_s, 0, oo))
print(f"    Normalization: INT |psi_100|^2 4pi r^2 dr = {simplify(norm_result)}")

# Expectation values
r_exp    = integrate(r_s * psi_100**2 * 4*pi*r_s**2, (r_s, 0, oo))
r2_exp   = integrate(r_s**2 * psi_100**2 * 4*pi*r_s**2, (r_s, 0, oo))
print(f"    <r>   = {simplify(r_exp)}  (= 3/2 * a0)")
print(f"    <r^2> = {simplify(r2_exp)}  (= 3 * a0^2)")

# psi_210
print("\n  SYMPY psi_210 (n=2, l=1, m=0):")
theta_s = symbols("theta", positive=True)
psi_210 = (1/(4*sqrt(2*pi))) * (1/a_s)**sp.Rational(3,2) * (r_s/a_s) * exp(-r_s/(2*a_s)) * sp.cos(theta_s)
print(f"    psi_210 = {psi_210}")

# Radial probability: scipy
r_arr = np.linspace(0, 30, 1000)   # in Bohr radii

def R_nl(n, l, r):
    """Radial wavefunction R_nl(r/a0), r in Bohr radii."""
    rho   = 2*r/n
    norm  = -np.sqrt((2/n)**3 * sc.factorial(n-l-1) /
                     (2*n*sc.factorial(n+l)**3))
    L     = sc.genlaguerre(n-l-1, 2*l+1)(rho)
    return norm * np.exp(-rho/2) * rho**l * L

states = [(1,0,"1s"), (2,0,"2s"), (2,1,"2p"), (3,0,"3s"), (3,1,"3p"), (3,2,"3d")]
print("\n  RADIAL PROBABILITY DENSITY P(r) = |R_nl|^2 * r^2:")
for n, l, name in states:
    R    = R_nl(n, l, r_arr)
    P    = R**2 * r_arr**2
    r_pk = r_arr[np.argmax(P)]
    print(f"    {name}: peak at r={r_pk:.2f} a0,  <r>~{3*n**2/2 - l*(l+1)/2:.2f} a0")

# ==========================================================
# S5: SPIN-1/2 AND PAULI MATRICES
# ==========================================================
print(f"\n{SEP}")
print("SECTION 5: SPIN-1/2 AND PAULI MATRICES")
print(SEP)

# Pauli matrices
sigma_x = Matrix([[0, 1], [1, 0]])
sigma_y = Matrix([[0, -I], [I, 0]])
sigma_z = Matrix([[1, 0], [0, -1]])
I2      = eye(2)

print("""
  SPIN OPERATORS: S = (hbar/2) * sigma
    sigma_x = [[0,1],[1,0]]
    sigma_y = [[0,-i],[i,0]]
    sigma_z = [[1,0],[0,-1]]
""")
print(f"  sigma_x:\n{sigma_x}")
print(f"\n  sigma_y:\n{sigma_y}")
print(f"\n  sigma_z:\n{sigma_z}")

# Pauli algebra
sx2 = simplify(sigma_x * sigma_x)
sy2 = simplify(sigma_y * sigma_y)
sz2 = simplify(sigma_z * sigma_z)
print(f"\n  PAULI ALGEBRA:")
print(f"    sigma_x^2 = {sx2}  (= I)")
print(f"    sigma_y^2 = {sy2}  (= I)")
print(f"    sigma_z^2 = {sz2}  (= I)")

comm_xy = simplify(sigma_x*sigma_y - sigma_y*sigma_x)
comm_yz = simplify(sigma_y*sigma_z - sigma_z*sigma_y)
comm_zx = simplify(sigma_z*sigma_x - sigma_x*sigma_z)
print(f"    [sx,sy] = {comm_xy}  (= 2i*sz)")
print(f"    [sy,sz] = {comm_yz}  (= 2i*sx)")
print(f"    [sz,sx] = {comm_zx}  (= 2i*sy)")
print(f"    sigma_i * sigma_j = delta_ij * I + i * eps_ijk * sigma_k")

# Eigenstates
print("""
  SPIN-Z EIGENSTATES:
    |+z> = [1, 0]^T   (spin up,   eigenvalue +hbar/2)
    |-z> = [0, 1]^T   (spin down, eigenvalue -hbar/2)

  SPIN-X EIGENSTATES:
    |+x> = (1/sqrt(2))[1, 1]^T   (eigenvalue +hbar/2)
    |-x> = (1/sqrt(2))[1,-1]^T   (eigenvalue -hbar/2)

  MEASUREMENT:
    System in |+z>. Measure Sx.
    P(+hbar/2) = |<+x|+z>|^2 = |1/sqrt(2)|^2 = 1/2
    P(-hbar/2) = |<-x|+z>|^2 = |1/sqrt(2)|^2 = 1/2
    -> Fully random outcome (x perpendicular to z)
""")

# Spinor rotation: Rz(theta) = exp(-i theta/2 * sigma_z)
# = cos(theta/2)*I - i*sin(theta/2)*sigma_z
theta_val = sp.Symbol("theta")
Rz = sp.cos(theta_val/2)*I2 - I*sp.sin(theta_val/2)*sigma_z
print(f"  ROTATION OPERATOR Rz(theta) = exp(-i*theta/2*sigma_z):")
print(f"    Rz = {Rz}")
print(f"    Rz(2*pi) = {simplify(Rz.subs(theta_val, 2*pi))}  (= -I !)")
print(f"    SPINORS pick up -1 under 2*pi rotation (need 4*pi for identity)")
print(f"    This is the DOUBLE COVER: SU(2) -> SO(3), spinors see SU(2)")

# Bloch sphere
print("""
  BLOCH SPHERE:
    Any spin-1/2 pure state: |psi> = cos(theta/2)|+z> + e^(i*phi)*sin(theta/2)|-z>
    theta in [0, pi]: polar angle (north pole = |+z>, south = |-z>)
    phi   in [0, 2pi): azimuthal angle
    Mixed state: density matrix rho = (I + r dot sigma)/2, |r| <= 1
    Pure state: |r|=1 (on sphere surface)
    Completely mixed: r=0 (center, rho = I/2)
""")

# Stern-Gerlach
print("  STERN-GERLACH:")
print("    Magnetic moment: mu = -g*mu_B*S/hbar  (g~2 for electron)")
print("    Force: F = nabla(mu dot B) = -g*mu_B*(dBz/dz)*m_s")
print("    m_s = +1/2: deflected up")
print("    m_s = -1/2: deflected down")
print("    Sequential SG devices verify quantum measurement postulate")

# ==========================================================
# S6: TIME-INDEPENDENT PERTURBATION THEORY
# ==========================================================
print(f"\n{SEP}")
print("SECTION 6: TIME-INDEPENDENT PERTURBATION THEORY")
print(SEP)

print("""
  SETUP:
    H = H_0 + lambda*H'     (lambda << 1, small perturbation)
    H_0 |n^0> = E_n^0 |n^0>   (known exact solution)
    Expand: |n> = |n^0> + lambda|n^1> + lambda^2|n^2> + ...
            E_n = E_n^0 + lambda*E_n^1 + lambda^2*E_n^2 + ...

  FIRST-ORDER ENERGY CORRECTION:
    E_n^1 = <n^0| H' |n^0>
    = expectation value of perturbation in unperturbed state
    Physical: energy shifts = diagonal matrix element of H'

  FIRST-ORDER STATE CORRECTION:
    |n^1> = SUM_{m != n} <m^0|H'|n^0> / (E_n^0 - E_m^0) * |m^0>
    = perturbed state mixes in nearby levels
    Denominator: E_n^0 - E_m^0  (diverges if degenerate -> need degen PT)

  SECOND-ORDER ENERGY:
    E_n^2 = SUM_{m != n} |<m^0|H'|n^0>|^2 / (E_n^0 - E_m^0)
    Always negative for ground state (ground state always pushed down)

  VALIDITY:
    |<m|H'|n>| << |E_n^0 - E_m^0|  for all m != n
    Perturbation must be small compared to level spacing.
""")

# Example: linear Stark effect on 1D harmonic oscillator
# H' = q*E*x = epsilon * x  (external electric field)
# E_n^1 = <n|x|n> = 0  (x is odd, |n> has definite parity -> zero)
# E_n^2 = sum_{m!=n} |<m|x|n>|^2 / (E_n^0 - E_m^0)
# <m|x|n> = sqrt(hbar/2mw) * (sqrt(n)*delta(m,n-1) + sqrt(n+1)*delta(m,n+1))
# Only m=n+/-1 contribute

print("  EXAMPLE: STARK EFFECT ON HARMONIC OSCILLATOR")
print("    H' = epsilon * x    (uniform electric field, charge q=1)")
print("    E_n^0 = hbar*omega*(n + 1/2)")
print()
print("    First-order: E_n^1 = <n|x|n> = 0  (parity argument)")
print("    x has odd parity; |n> has parity (-1)^n;")
print("    <n|x|n> integrand = even * odd * even = odd -> integral = 0")
print()
print("    Second-order:")
print("    <m|x|n> = sqrt(hbar/(2mw)) * [sqrt(n)*d(m,n-1) + sqrt(n+1)*d(m,n+1)]")
print("    Only m=n-1 and m=n+1 contribute:")
print("    E_n^2 = eps^2 * hbar/(2mw) * [n/(E_n^0-E_{n-1}^0) + (n+1)/(E_n^0-E_{n+1}^0)]")
print("          = eps^2 * hbar/(2mw) * [n/(+hbar*w) + (n+1)/(-hbar*w)]")
print("          = eps^2/(2mw^2) * [n - n - 1]")
print("          = -eps^2 / (2*m*omega^2)")
print()
print("    => E_n^2 is INDEPENDENT of n! All levels shift by same amount.")
print("    Physical: electric field displaces equilibrium, shifts all levels equally.")
print("    This is exact (can verify by completing the square in H).")
print()

# Numerical verify: complete the square
# H = p^2/2m + mw^2 x^2/2 + eps*x
#   = p^2/2m + mw^2/2*(x + eps/(mw^2))^2 - eps^2/(2mw^2)
# New equilibrium at x0 = -eps/(mw^2), energy shifted by -eps^2/(2mw^2)
print("    EXACT VERIFICATION (complete the square):")
print("    H = p^2/(2m) + (mw^2/2)*(x + eps/(mw^2))^2 - eps^2/(2mw^2)")
print("    => Same SHO with x0 displaced, E_n = hbar*w*(n+1/2) - eps^2/(2mw^2)")
print("    => E_n^(2) = -eps^2/(2mw^2)  (perturbation theory gives EXACT result)")

# ==========================================================
# S7: ENTANGLEMENT AND BELL STATES
# ==========================================================
print(f"\n{SEP}")
print("SECTION 7: ENTANGLEMENT AND BELL STATES")
print(SEP)

print("""
  COMPOSITE HILBERT SPACE:
    Two spin-1/2 particles: H = H_A (x) H_B
    Basis: |++>, |+->, |-+>, |-->
    Dimension = 2 * 2 = 4

  PRODUCT STATE (NOT entangled):
    |psi> = |+>_A (x) |->_B = |+->
    Subsystem A is always +, B always -, regardless of other.

  ENTANGLED STATE:
    |psi> = (1/sqrt(2)) * (|+-> + |-+>)
    CANNOT be written as |phi_A> (x) |phi_B> for ANY |phi_A>, |phi_B>
    Measuring A as +: B instantly collapses to -  (no matter distance)
    Measuring A as -: B instantly collapses to +
""")

# Bell states as numpy vectors  |++>=0, |+->=1, |-+>=2, |-->=3
Phi_plus  = np.array([1, 0, 0, 1]) / np.sqrt(2)   # (|++> + |-->)/sqrt(2)
Phi_minus = np.array([1, 0, 0,-1]) / np.sqrt(2)   # (|++> - |-->)/sqrt(2)
Psi_plus  = np.array([0, 1, 1, 0]) / np.sqrt(2)   # (|+-> + |-+>)/sqrt(2)
Psi_minus = np.array([0, 1,-1, 0]) / np.sqrt(2)   # (|+-> - |-+>)/sqrt(2)

bell_states = [
    ("Phi+", Phi_plus,  "(|++> + |-->)/sqrt(2)"),
    ("Phi-", Phi_minus, "(|++> - |-->)/sqrt(2)"),
    ("Psi+", Psi_plus,  "(|+-> + |-+>)/sqrt(2)"),
    ("Psi-", Psi_minus, "(|+-> - |-+>)/sqrt(2)"),
]

print("  FOUR BELL STATES (maximally entangled, form orthonormal basis):")
for name, vec, desc in bell_states:
    norm = np.dot(vec, vec)
    print(f"    |{name}> = {desc},  norm={norm:.4f}")

# Orthogonality check
print("\n  ORTHOGONALITY <Bell_i|Bell_j>:")
names = [b[0] for b in bell_states]
vecs  = [b[1] for b in bell_states]
for i in range(4):
    row = "  ".join(f"{np.dot(vecs[i], vecs[j]):+.0f}" for j in range(4))
    print(f"    |{names[i]}>:  {row}")
print("    -> Identity matrix: orthonormal basis confirmed")

# Density matrix of Psi_minus
print("\n  DENSITY MATRIX rho = |Psi-> <Psi-|:")
rho = np.outer(Psi_minus, Psi_minus)
print("    rho (4x4):")
for row in rho:
    print("     ", "  ".join(f"{v:+.4f}" for v in row))
print(f"    Tr(rho)   = {np.trace(rho):.4f}  (= 1, normalized)")
print(f"    Tr(rho^2) = {np.trace(rho @ rho):.4f}  (= 1: pure state)")

# Partial trace over B -> reduced density matrix for A
# rho_A = Tr_B(rho): trace out B indices
# In |++>,|+->,|-+>|--> basis: B=+: rows/cols 0,2; B=-: rows/cols 1,3
rho_A = np.zeros((2, 2), complex)
rho_A[0,0] = rho[0,0] + rho[1,1]   # A=+, Tr over B
rho_A[0,1] = rho[0,2] + rho[1,3]
rho_A[1,0] = rho[2,0] + rho[3,1]
rho_A[1,1] = rho[2,2] + rho[3,3]
print(f"\n  REDUCED DENSITY MATRIX rho_A = Tr_B(rho):")
print(f"    rho_A = {rho_A.real}")
print(f"    Tr(rho_A)   = {np.trace(rho_A).real:.4f}")
print(f"    Tr(rho_A^2) = {np.trace(rho_A @ rho_A).real:.4f}  (< 1: MIXED state!)")
print(f"    => Entanglement makes subsystem A maximally MIXED (I/2)")
print(f"    => Cannot describe A alone with a pure state -- fundamental")

# CHSH inequality
print("""
  BELL / CHSH INEQUALITY:
    Local hidden variables (classical): |E(a,b) - E(a,b') + E(a',b) + E(a',b')| <= 2
    where E(a,b) = <sigma_a (x) sigma_b> = correlation between measurements

    For |Psi-> and optimal angles a=0, b=pi/4, a'=pi/2, b'=3*pi/4:
    E(a,b)  = -cos(b-a)   (QM prediction for singlet)
""")

angles = [(0, np.pi/4), (0, 3*np.pi/4), (np.pi/2, np.pi/4), (np.pi/2, 3*np.pi/4)]
E_vals = [-np.cos(b-a) for a,b in angles]
CHSH_QM = abs(E_vals[0] - E_vals[1] + E_vals[2] + E_vals[3])
print(f"    E(0,pi/4)     = {E_vals[0]:+.6f}")
print(f"    E(0,3pi/4)    = {E_vals[1]:+.6f}")
print(f"    E(pi/2,pi/4)  = {E_vals[2]:+.6f}")
print(f"    E(pi/2,3pi/4) = {E_vals[3]:+.6f}")
print(f"    CHSH = |E1 - E2 + E3 + E4| = {CHSH_QM:.6f}")
print(f"    Classical limit: <= 2.000000")
print(f"    Quantum maximum: 2*sqrt(2) = {2*np.sqrt(2):.6f}")
print(f"    VIOLATION: {CHSH_QM:.4f} > 2  => NO local hidden variable theory possible")
print(f"    Experimentally confirmed: Aspect 1982, Zeilinger 2022 Nobel Prize")

# ==========================================================
# MATPLOTLIB -- 6-PANEL FIGURE
# ==========================================================
print(f"\n{SEP}")
print("BUILDING FIGURE...")
print(SEP)

fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor("#F8F8F0")
gs0 = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38,
                        top=0.93, bottom=0.05, left=0.06, right=0.97)

ax_ho    = fig.add_subplot(gs0[0, 0])   # HO wavefunctions
ax_prob  = fig.add_subplot(gs0[0, 1])   # HO probability densities
ax_energy= fig.add_subplot(gs0[0, 2])   # Energy level diagram
ax_H     = fig.add_subplot(gs0[1, 0])   # Hydrogen radial prob
ax_bloch = fig.add_subplot(gs0[1, 1])   # Bloch sphere (2D projection)
ax_bell  = fig.add_subplot(gs0[1, 2])   # Bell state density matrix
ax_pt    = fig.add_subplot(gs0[2, 0])   # Perturbation theory diagram
ax_comm  = fig.add_subplot(gs0[2, 1])   # Commutator / uncertainty
ax_post  = fig.add_subplot(gs0[2, 2])   # Postulates summary table

fig.suptitle("Honors Quantum Mechanics  |  Operators, HO, H Atom, Spin, Entanglement",
             fontsize=14, fontweight="bold", color="#1a1a2e")

colors_n = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]

# ---- AX_HO: wavefunctions ----
ax = ax_ho
ax.set_facecolor("#F0F4FF")
for n in range(5):
    offset = n * 1.2
    ax.plot(xi_arr, psi_n[n] + offset, color=colors_n[n], lw=1.5,
            label=f"n={n}, E={n+0.5:.1f}hw")
    ax.axhline(offset, color=colors_n[n], lw=0.5, ls="--", alpha=0.4)
ax.set_xlabel(r"$\xi = x\sqrt{m\omega/\hbar}$", fontsize=9)
ax.set_ylabel(r"$\psi_n(\xi)$ + offset", fontsize=9)
ax.set_title("HO Wavefunctions $\\psi_n$", fontsize=10)
ax.legend(fontsize=7, loc="upper right")
ax.grid(alpha=0.2)

# ---- AX_PROB: probability densities ----
ax = ax_prob
ax.set_facecolor("#F0F4FF")
for n in range(5):
    prob = psi_n[n]**2
    ax.plot(xi_arr, prob, color=colors_n[n], lw=1.5, label=f"n={n}")
    ax.fill_between(xi_arr, prob, alpha=0.15, color=colors_n[n])
ax.set_xlabel(r"$\xi$", fontsize=9)
ax.set_ylabel(r"$|\psi_n|^2$", fontsize=9)
ax.set_title("HO Probability Densities", fontsize=10)
ax.legend(fontsize=7, loc="upper right")
ax.grid(alpha=0.2)
ax.text(0.02, 0.97,
        r"$\sigma_x\sigma_p = \hbar/2$ at $n=0$",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(fc="#ffffcc", ec="#bbb", pad=2))

# ---- AX_ENERGY: energy level diagram ----
ax = ax_energy
ax.set_facecolor("#FAFAF0")
ax.set_xlim(-2, 5)
ax.set_ylim(-0.5, 7)
ax.axis("off")
ax.set_title("Energy Level Diagrams", fontsize=10)

# HO (left side)
ax.text(0.3, 6.7, "SHO", fontsize=9, ha="center", fontweight="bold", color="#1f77b4")
for n in range(6):
    En = n + 0.5
    ax.plot([-0.5, 1.0], [En, En], color=colors_n[n % 5], lw=2)
    ax.text(1.1, En, f"n={n}, {En:.1f}hw", fontsize=7, va="center", color=colors_n[n % 5])
ax.text(-0.8, 0.5, r"$E_n=(n+\frac{1}{2})\hbar\omega$",
        fontsize=7.5, rotation=90, va="center", color="#1f77b4")

# Hydrogen (right side)
ax.text(3.5, 6.7, "H atom", fontsize=9, ha="center", fontweight="bold", color="#d62728")
for n in range(1, 6):
    En_H = -13.606 / n**2
    y_H  = (En_H + 13.606) / 13.606 * 6    # scale to [0,6]
    ax.plot([2.8, 4.2], [y_H, y_H], color="#d62728", lw=2, alpha=0.8)
    ax.text(4.3, y_H, f"n={n}: {En_H:.1f}eV", fontsize=6.5, va="center", color="#d62728")

ax.text(2.5, 0.5, r"$E_n=-\frac{13.6}{n^2}$eV",
        fontsize=7.5, va="center", color="#d62728")

# ---- AX_H: hydrogen radial probability ----
ax = ax_H
ax.set_facecolor("#F0FFF0")
state_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
for i, (n, l, name) in enumerate(states):
    R    = R_nl(n, l, r_arr)
    P    = R**2 * r_arr**2
    P   /= np.max(P) + 1e-15
    ax.plot(r_arr, P, color=state_colors[i], lw=1.5, label=name)
ax.set_xlabel(r"$r / a_0$", fontsize=9)
ax.set_ylabel(r"$P(r) = |R_{nl}|^2 r^2$  (normalized)", fontsize=9)
ax.set_title("Hydrogen Radial Probability Densities", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right")
ax.set_xlim(0, 25)
ax.grid(alpha=0.2)

# ---- AX_BLOCH: Bloch sphere 2D projection ----
ax = ax_bloch
ax.set_facecolor("#FFF5FF")
ax.set_aspect("equal")
ax.set_title("Bloch Sphere (2D section)", fontsize=10)

theta_circ = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(theta_circ), np.sin(theta_circ), "k-", lw=1.2, alpha=0.4)
ax.axhline(0, color="#999", lw=0.7, ls="--")
ax.axvline(0, color="#999", lw=0.7, ls="--")

# Key states on Bloch sphere (z-x plane, phi=0)
states_bloch = [
    (0,          1,    r"$|+z\rangle$", "#1f77b4"),
    (0,         -1,    r"$|-z\rangle$", "#d62728"),
    (1,          0,    r"$|+x\rangle$", "#2ca02c"),
    (-1,         0,    r"$|-x\rangle$", "#ff7f0e"),
    (1/np.sqrt(2), 1/np.sqrt(2), r"$|+\rangle=|{+z}\rangle+i|{-z}\rangle$... $|+y\rangle$", "#9467bd"),
]
for bx, bz, label, c in states_bloch:
    ax.annotate("", xy=(bx, bz), xytext=(0,0),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.8))
    ax.text(bx*1.15, bz*1.15, label, fontsize=7.5, color=c,
            ha="center", va="center")

ax.text(0.02, 0.02,
        r"$|\psi\rangle=\cos(\theta/2)|{+z}\rangle+e^{i\phi}\sin(\theta/2)|-z\rangle$",
        transform=ax.transAxes, fontsize=7.5,
        bbox=dict(fc="white", ec="#bbb", pad=2))
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
ax.set_xlabel("x (Re part)", fontsize=8)
ax.set_ylabel("z", fontsize=8)
ax.grid(alpha=0.15)

# ---- AX_BELL: Bell state density matrix ----
ax = ax_bell
ax.set_facecolor("#FFF8F0")
rho_show = np.abs(rho)
im = ax.imshow(rho_show, cmap="YlOrRd", vmin=0, vmax=0.5, aspect="auto")
ax.set_xticks([0,1,2,3])
ax.set_yticks([0,1,2,3])
ax.set_xticklabels(["|++>","|+->","|-+>","|--->"], fontsize=8)
ax.set_yticklabels(["|++>","|+->","|-+>","|--->"], fontsize=8)
ax.set_title(r"Bell State $|\Psi^-\rangle$ Density Matrix $|\rho_{ij}|$", fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{rho_show[i,j]:.2f}", ha="center", va="center",
                fontsize=8, color="black" if rho_show[i,j] < 0.35 else "white")
ax.text(0.02, 0.02, r"$\mathrm{Tr}(\rho^2)=1$ (pure)$\quad\mathrm{Tr}(\rho_A^2)=0.5$ (mixed)",
        transform=ax.transAxes, fontsize=7.5,
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ---- AX_PT: perturbation theory ----
ax = ax_pt
ax.set_facecolor("#F0FFF8")
ax.axis("off")
ax.set_title("Perturbation Theory Summary", fontsize=10)
pt_lines = [
    ("H = H_0 + lam*H'", "#222"),
    ("", ""),
    ("E_n^(1) = <n^0|H'|n^0>", "#1a5c9c"),
    ("  (diagonal matrix element)", "#555"),
    ("", ""),
    ("|n^(1)> = SUM_{m!=n} <m|H'|n>", "#1a5c9c"),
    ("          -----------------  |m>", "#1a5c9c"),
    ("           E_n^0 - E_m^0", "#1a5c9c"),
    ("", ""),
    ("E_n^(2) = SUM_{m!=n} |<m|H'|n>|^2", "#d62728"),
    ("          -------------------------", "#d62728"),
    ("               E_n^0 - E_m^0", "#d62728"),
    ("", ""),
    ("Stark (HO): E_n^(1) = 0 (parity)", "#2c7c2c"),
    ("            E_n^(2) = -eps^2/(2mw^2)", "#2c7c2c"),
    ("            (indep. of n, exact!)", "#2c7c2c"),
    ("", ""),
    ("Validity: |<m|H'|n>| << |E_n-E_m|", "#888"),
]
for k, (line, color) in enumerate(pt_lines):
    if not line:
        continue
    ax.text(0.03, 0.97 - k*0.056, line, transform=ax.transAxes,
            fontsize=8, color=color if color else "#333", va="top", fontfamily="monospace")

# ---- AX_COMM: commutators and uncertainty ----
ax = ax_comm
ax.set_facecolor("#FFF0F0")
ax.axis("off")
ax.set_title("Commutators & Uncertainty", fontsize=10)
comm_lines = [
    ("[x, p]   = i*hbar         (canonical)", "#1a1a8c"),
    ("[Lx, Ly] = i*hbar*Lz      (angular)", "#1a1a8c"),
    ("[L^2, Lz]= 0              (shared eig)", "#2c7c2c"),
    ("[H, L^2] = 0  (central force)", "#2c7c2c"),
    ("", ""),
    ("Generalized uncertainty:", "#555"),
    ("  sigma_A * sigma_B >= |<[A,B]>|/2", "#d62728"),
    ("", ""),
    ("[x,p] => sigma_x*sigma_p >= hbar/2", "#d62728"),
    ("n=0 SHO: sigma_x*sigma_p = hbar/2", "#2c7c2c"),
    ("  (ground state SATURATES bound)", "#2c7c2c"),
    ("", ""),
    ("Energy-time uncertainty:", "#9467bd"),
    ("  sigma_E * Delta_t >= hbar/2", "#9467bd"),
    ("  (Delta_t = lifetime of state)", "#9467bd"),
    ("  Natural linewidth: Gamma=hbar/tau", "#9467bd"),
    ("", ""),
    ("Pauli: [sx,sy] = 2i*sz  (spin)", "#8c564b"),
    ("CHSH violation: 2sqrt(2) > 2", "#8c564b"),
]
for k, (line, color) in enumerate(comm_lines):
    if not line:
        continue
    ax.text(0.03, 0.97 - k*0.048, line, transform=ax.transAxes,
            fontsize=7.8, color=color if color else "#333", va="top", fontfamily="monospace")

# ---- AX_POST: postulates quick ref ----
ax = ax_post
ax.set_facecolor("#F8F0FF")
ax.axis("off")
ax.set_title("QM Postulates Quick Reference", fontsize=10)
post_lines = [
    ("1. State", "|psi> in Hilbert space H", "#1f77b4"),
    ("2. Observables", "Hermitian A_hat = A_hat^dag", "#ff7f0e"),
    ("3. Measurement", "P(a_n) = |<a_n|psi>|^2", "#2ca02c"),
    ("4. Uncertainty", "sig_A*sig_B >= |<[A,B]>|/2", "#d62728"),
    ("5. Dynamics", "i*hbar*d|psi>/dt = H|psi>", "#9467bd"),
    ("6. Identical", "Bosons sym / Fermions antisym", "#8c564b"),
    ("", "", ""),
    ("CONNECTING TO ML:", "", "#333"),
    ("H_hat eigenvecs", "-> PCA / attention K,Q,V", "#1a4e8c"),
    ("[A,B]=0 => sim.", "-> commuting ops share basis", "#1a4e8c"),
    ("Density matrix", "-> mixed states / dropout", "#1a4e8c"),
    ("Partition Z", "-> softmax normalization", "#1a4e8c"),
    ("Unitary U(t)", "-> weight matrix constraints", "#1a4e8c"),
    ("Entanglement", "-> correlations in embedding", "#1a4e8c"),
]
for k, (label, value, color) in enumerate(post_lines):
    y = 0.97 - k*0.063
    if label == "":
        ax.axhline(y + 0.01, color="#ccc", lw=0.5,
                   xmin=0.03, xmax=0.97)
        continue
    ax.text(0.03, y, label, transform=ax.transAxes,
            fontsize=7.8, color=color if color else "#333", va="top", fontweight="bold")
    ax.text(0.38, y, value, transform=ax.transAxes,
            fontsize=7.8, color="#333", va="top", fontfamily="monospace")

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {OUT}")

print(f"\n{SEP}")
print("Done.")
print(SEP)
