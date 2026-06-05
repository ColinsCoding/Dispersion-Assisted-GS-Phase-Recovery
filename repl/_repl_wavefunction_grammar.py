"""
repl/_repl_wavefunction_grammar.py
Wavefunction vocabulary and grammar. Griffiths Ch 1-3.
Even potential -> parity eigenstates. Connects odd/even -> QM -> GS.
"""
import math
import numpy as np
import sympy as sp
sp.init_printing(use_unicode=False, wrap_line=False)

print("=" * 60)
print("WAVEFUNCTION GRAMMAR  (Griffiths Ch 1-3)")
print("=" * 60)
print()

# ============================================================
# VOCABULARY TABLE
# ============================================================
print("=== VOCABULARY ===")
print("""
Symbol       Reads as              Means
-------      ---------             -----
psi(x,t)     'psi of x,t'         wavefunction: complex number at each (x,t)
|psi|^2      'mod psi squared'     probability density: P(x) dx = |psi|^2 dx
<x>          'expectation of x'    average position: integral x|psi|^2 dx
<H>          'expectation of H'    average energy
psi*         'psi star'            complex conjugate
<f|g>        'f inner product g'   integral f*(x)*g(x) dx
|n>          'ket n'               state n (Dirac notation)
<n|          'bra n'               dual of |n>
<m|n>        'braket m,n'          overlap integral = delta_mn if orthonormal
H|psi>       'H acting on psi'     apply Hamiltonian operator to state
[A,B]        'commutator A,B'      A*B - B*A  (zero = simultaneous eigenstates)

GRAMMAR:
  psi is a function, not a number
  |psi|^2 is a number (probability density)
  <x> is a number (expectation value)
  H|psi> is a function (new state after operator acts)
  operators act on kets: A|v> -> |w>
  brackets give numbers: <u|A|v> -> scalar
""")

# ============================================================
# 1. Normalization grammar
# ============================================================
print("=== 1. Normalization: int |psi|^2 dx = 1 ===")
x, a_s = sp.symbols('x a', positive=True)
n_sym = sp.Symbol('n', integer=True, positive=True)

# particle in a box: psi_n = sqrt(2/L) * sin(n*pi*x/L)
L = sp.Symbol('L', positive=True)
psi_n = sp.sqrt(2/L) * sp.sin(n_sym * sp.pi * x / L)

# verify normalization
norm = sp.integrate(sp.conjugate(psi_n) * psi_n, (x, 0, L))
norm_simplified = sp.simplify(norm)
print(f"psi_n = sqrt(2/L) * sin(n*pi*x/L)")
print(f"int |psi_n|^2 dx = {norm_simplified}  (must be 1)")

# orthogonality
for m_val, n_val in [(1,1),(1,2),(2,3)]:
    psi_m = sp.sqrt(2/L)*sp.sin(m_val*sp.pi*x/L)
    psi_nv = sp.sqrt(2/L)*sp.sin(n_val*sp.pi*x/L)
    overlap = sp.simplify(sp.integrate(psi_m*psi_nv, (x,0,L)))
    print(f"  <{m_val}|{n_val}> = {overlap}  ({'delta_mn' if m_val==n_val else '0 (orthogonal)'})")
print()

# ============================================================
# 2. Expectation values: the grammar of <O>
# ============================================================
print("=== 2. Expectation values ===")
print("""
<O> = integral psi*(x) * O_hat * psi(x) dx

O_hat = x:          O_hat*psi = x*psi(x)           (multiply by x)
O_hat = p:          O_hat*psi = -i*hbar*dpsi/dx     (differentiate)
O_hat = H:          O_hat*psi = -hbar^2/2m*d^2psi/dx^2 + V*psi
""")

hbar, m_s = sp.symbols('hbar m', positive=True)

# <x> for ground state of particle in box (n=1, L=1)
psi_1 = sp.sqrt(2) * sp.sin(sp.pi * x)
x_exp = sp.integrate(psi_1 * x * psi_1, (x, 0, 1))
x2_exp = sp.integrate(psi_1 * x**2 * psi_1, (x, 0, 1))
sigma_x = sp.sqrt(sp.simplify(x2_exp - x_exp**2))
print(f"Particle in box n=1, L=1:")
print(f"  <x>     = {sp.simplify(x_exp)}")
print(f"  <x^2>   = {sp.simplify(x2_exp)}")
print(f"  sigma_x = {sp.simplify(sigma_x)}")
print()

# ============================================================
# 3. Even potential -> parity eigenstates (Griffiths 2.1)
# ============================================================
print("=== 3. Even potential -> parity eigenstates ===")
print("""
If V(x) = V(-x)  (even potential)
then parity operator P: P*psi(x) = psi(-x)

[H, P] = 0  (H and P commute)
-> they share eigenstates
-> every eigenstate is purely even OR purely odd

Proof sketch:
  H*psi = E*psi
  H*(P*psi) = P*(H*psi) = P*(E*psi) = E*(P*psi)
  -> P*psi is also an eigenstate with same energy E
  -> psi_even = psi + P*psi  (even combination)
  -> psi_odd  = psi - P*psi  (odd  combination)
  -> can always choose eigenstates with definite parity

Examples:
  V = 0 (free):          sin(kx) odd,  cos(kx) even
  V = (1/2)kx^2 (SHO):  psi_0,2,4... even,  psi_1,3,5... odd
  V = delta(x):          even solution only (ground state)
  V = infinite well [-L,L]: cos(n*pi*x/2L) even, sin(n*pi*x/2L) odd

NOT even potential:
  V = (1/2)kx^2 + alpha*x  (displaced SHO)
  -> breaks parity -> eigenstates are neither even nor odd
""")

# numerical: SHO wavefunctions parity
print("SHO: psi_n(-x) = (-1)^n * psi_n(x)")
x_arr = np.linspace(-4, 4, 200)
from numpy.polynomial.hermite import hermval

def sho_psi(n, x):
    """Normalized SHO wavefunction (hbar=m=omega=1)."""
    coeffs = np.zeros(n+1); coeffs[n] = 1
    H_n = hermval(x, coeffs)
    norm = (2**n * math.factorial(n) * np.sqrt(np.pi))**(-0.5)
    return norm * np.exp(-x**2/2) * H_n

print(f"  {'n':>3}  {'parity':>8}  {'max|psi(-x)-(-1)^n*psi(x)|':>30}")
for n in range(5):
    psi_pos = sho_psi(n, x_arr)
    psi_neg = sho_psi(n, -x_arr)
    parity_check = np.max(np.abs(psi_neg - (-1)**n * psi_pos))
    kind = 'even' if n%2==0 else 'odd'
    print(f"  {n:>3}  {kind:>8}  {parity_check:.2e}")
print()

# ============================================================
# 4. Selection rules from parity
# ============================================================
print("=== 4. Selection rules from parity ===")
print("""
Electric dipole transition matrix element:
  <m|x|n> = integral psi_m*(x) * x * psi_n(x) dx

  x is ODD.
  If psi_m and psi_n have SAME parity:
    psi_m* * x * psi_n = even*odd*even = odd  -> integral = 0
  If psi_m and psi_n have DIFFERENT parity:
    psi_m* * x * psi_n = even*odd*odd  = even -> integral != 0

SELECTION RULE: transitions only between states of OPPOSITE parity
  SHO: Delta_n = odd (n=0->1, 1->2, 2->3, ...)
  Hydrogen: Delta_l = +/-1  (same origin: parity of angular wavefunctions)
""")

# verify numerically for SHO
print("SHO dipole matrix <m|x|n>:")
print(f"  {'(m,n)':>8}  {'<m|x|n>':>12}  {'allowed?'}")
for m in range(4):
    for n in range(4):
        psi_m_arr = sho_psi(m, x_arr)
        psi_n_arr = sho_psi(n, x_arr)
        dx = x_arr[1]-x_arr[0]
        me = np.trapezoid(psi_m_arr * x_arr * psi_n_arr, x_arr)
        allowed = abs(m-n) == 1
        if abs(me) > 1e-6 or allowed:
            print(f"  ({m},{n}):  {me:>12.4f}  {'YES' if abs(me)>1e-6 else 'no '}")
print()

# ============================================================
# 5. Connection to GS phase recovery
# ============================================================
print("=== 5. Wavefunction grammar -> GS grammar ===")
print("""
QM                              GS phase recovery
---------------------------     ---------------------------
psi(t)  complex wavefunction    E(t)  complex field
|psi|^2 probability density     |E|^2 intensity (what you measure)
<x>     expectation value       <phi> mean phase
normalization: int|psi|^2=1     unit amplitude: |E(t)|=1 (constraint)
Born rule: P = |<x|psi>|^2      detector: I = |<H|E>|^2

Measurement destroys psi        Detector destroys phase
GS restores phi from I1,I2      <- solving the Born rule inversion

Parity in GS:
  If E(t) is even (symmetric pulse) -> I1, I2 are also even
  -> GS should exploit this symmetry to halve the search space
  -> rfft instead of fft: use Hermitian symmetry (2x speedup)
""")
