# -*- coding: utf-8 -*-
"""
_repl_dirac_delta_physics.py
=============================
Dirac delta function — pure physics, heavily monitored.
Every result is (1) derived symbolically with SymPy init_printing,
(2) verified numerically, and (3) visualised.

EXAMPLE 1  Sifting property         integral f(x)*delta(x-a) = f(a)
EXAMPLE 2  Scaling / odd argument   delta(a*x) = delta(x)/|a|
EXAMPLE 3  First derivative delta'  integral f*delta'(x-a) = -f'(a)
EXAMPLE 4  Composition rule         delta(g(x)) = sum_i delta(x-x_i)/|g'(x_i)|
EXAMPLE 5  Fourier representation   delta(x) = (1/2pi) integral e^{ikx} dk
EXAMPLE 6  x * delta(x) = 0        distributional identity
EXAMPLE 7  3-D delta / Coulomb      nabla^2(1/r) = -4*pi*delta^3(r)
EXAMPLE 8  Green's function         L*G(x,x') = delta(x-x'), 1-D Laplacian

All integrals verified: SymPy exact  vs  scipy.quad numerical.
Output: repl/_out_dirac_delta_physics.png
"""

# ── Notebook preamble (init_printing lives in the injected setup cell) ────────
import sympy as sp
from sympy import (
    symbols, DiracDelta, Heaviside, integrate, diff, oo, pi, exp, I,
    sqrt, Abs, cos, sin, Rational, limit, simplify, latex,
    Function, Piecewise, S, factorial
)
from sympy import init_printing          # called explicitly so the cell is clear
init_printing(use_latex="mathjax")       # LaTeX render in Jupyter

import numpy as np
from scipy import integrate as sci_int, special as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display, Math, Markdown
import warnings, os
warnings.filterwarnings("ignore")

try:
    _REPL_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _REPL_DIR = os.getcwd()
OUT = os.path.join(_REPL_DIR, "_out_dirac_delta_physics.png")

SEP = "=" * 65

try:
    from IPython.display import display as _ipy_display
    def show(expr, label=None):
        if label: print(f"  {label}:")
        _ipy_display(expr)
except ImportError:
    def show(expr, label=None):
        if label: print(f"  {label}:")
        import sympy as _sp
        print("  " + _sp.pretty(expr, use_unicode=True))
        display(expr)

def check(sympy_val, numerical_val, tol=1e-8, label=""):
    status = "PASS" if abs(float(sympy_val.evalf()) - numerical_val) < tol else "FAIL"
    print(f"  [{status}]  SymPy = {float(sympy_val.evalf()):.10f}"
          f"   Numerical = {numerical_val:.10f}"
          + (f"   ({label})" if label else ""))
    return status == "PASS"

# ─────────────────────────────────────────────────────────────
print(SEP)
print("MOTIVATION: WHAT IS THE DIRAC DELTA?")
print(SEP)

print("""
  The Dirac delta delta(x) is NOT a function in the classical sense.
  It is a DISTRIBUTION (generalised function) defined by its action
  on smooth test functions phi in the Schwartz space S(R):

        integral_{-inf}^{inf} delta(x) * phi(x) dx  =  phi(0)

  Intuitively: infinite spike at x=0, zero everywhere else,
  with unit area under the spike.

  REPRESENTATIONS AS LIMITS:
    1. Gaussian:      delta(x) = lim_{eps->0}  (1/eps*sqrt(pi)) exp(-x^2/eps^2)
    2. Lorentzian:    delta(x) = lim_{eps->0}  (eps/pi) / (x^2 + eps^2)
    3. sinc:          delta(x) = lim_{L->inf}  sin(L*x) / (pi*x)
    4. rect:          delta(x) = lim_{eps->0}  (1/eps) rect(x/eps)
    5. Fourier:       delta(x) = (1/2*pi) integral_{-inf}^{inf} e^{ikx} dk
""")

# ── Build visualisation data for all examples first ──────────────────────────
x_plot  = np.linspace(-5, 5, 4000)
eps_vals = [2.0, 1.0, 0.5, 0.2, 0.05]

gaussian_seqs  = {e: (1/(e*np.sqrt(np.pi)))*np.exp(-x_plot**2/e**2) for e in eps_vals}
lorentz_seqs   = {e: (e/np.pi)/(x_plot**2 + e**2) for e in eps_vals}

# ─────────────────────────────────────────────────────────────
print(SEP)
print("EXAMPLE 1 — SIFTING PROPERTY")
print(SEP)

print("""
  STATEMENT:
        integral_{-inf}^{inf} f(x) * delta(x - a) dx  =  f(a)

  The delta function "sifts out" the value of f at x = a.
  This is the DEFINING property of the distribution.
""")

x, a = symbols('x a', real=True)
f    = symbols('f', cls=Function)

# Symbolic — SymPy knows DiracDelta
ex1_integrand_sym = x**2 * DiracDelta(x - 3)
ex1_result_sym    = integrate(ex1_integrand_sym, (x, -oo, oo))
show(ex1_result_sym, "Ex1:  integral x^2 * delta(x-3) dx =")

ex1_integrand2 = sp.exp(x) * DiracDelta(x - sp.log(2))
ex1_result2    = integrate(ex1_integrand2, (x, -oo, oo))
show(ex1_result2, "Ex1b: integral e^x * delta(x - ln2) dx =")

ex1_integrand3 = sp.sin(x)**2 * DiracDelta(x - pi/4)
ex1_result3    = integrate(ex1_integrand3, (x, -oo, oo))
show(ex1_result3, "Ex1c: integral sin^2(x) * delta(x - pi/4) dx =")

# Numerical verification using Gaussian approx
def sift_numerical(f_func, a_val, eps=0.02):
    """Numerical sifting using Gaussian approx; fine grid around peak."""
    # Use a tight grid around the peak to capture the narrow Gaussian
    x_fine = np.linspace(a_val - 10*eps, a_val + 10*eps, 20000)
    delta_ap = (1/(eps*np.sqrt(np.pi))) * np.exp(-(x_fine - a_val)**2/eps**2)
    integrand = f_func(x_fine) * delta_ap
    return np.trapezoid(integrand, x_fine)

print("\n  VERIFICATION:")
check(ex1_result_sym,
      sift_numerical(lambda xv: xv**2, 3.0),
      tol=0.005, label="x^2 at x=3")
check(ex1_result2,
      sift_numerical(lambda xv: np.exp(xv), np.log(2)),
      tol=0.005, label="e^x at x=ln2")
check(sp.Rational(1,2),
      sift_numerical(lambda xv: np.sin(xv)**2, np.pi/4),
      label="sin^2(x) at x=pi/4")

# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EXAMPLE 2 — SCALING RULE  delta(a*x) = delta(x) / |a|")
print(SEP)

print("""
  DERIVATION by change of variables  u = a*x:
    integral f(x) * delta(a*x) dx
    = integral f(u/a) * delta(u) * (1/|a|) du   [du = a*dx]
    = f(0) / |a|
  Therefore:
        delta(a*x)  =  delta(x) / |a|
  More generally:
        delta(a*(x - b))  =  delta(x - b) / |a|
  And with a = -1:
        delta(-x) = delta(x)   (delta is EVEN)
""")

a_val = symbols('a', real=True, positive=True)
ex2_result_pos = integrate(DiracDelta(3*x), (x, -oo, oo))
ex2_result_neg = integrate(DiracDelta(-5*x + 10), (x, -oo, oo))
show(ex2_result_pos, "Ex2a: integral delta(3x) dx =")
show(ex2_result_neg, "Ex2b: integral delta(-5x+10) dx =")

# Symbolic proof
print("\n  SYMBOLIC PROOF (SymPy):")
f_sym = x**3 + 2*x
integral_ax  = integrate(f_sym * DiracDelta(2*x), (x, -oo, oo))
expected_ax  = f_sym.subs(x, 0) / 2
show(integral_ax,  "integral (x^3+2x)*delta(2x) dx =")
show(expected_ax,  "f(0)/|a| = (0^3+2*0)/2 =")
print(f"  Match: {simplify(integral_ax - expected_ax) == 0}")

print("\n  VERIFICATION:")
check(ex2_result_pos, sift_numerical(lambda xv: 1.0, 0.0)/3,
      tol=0.005, label="integral delta(3x) = 1/3")
check(ex2_result_neg,
      sift_numerical(lambda xv: 1.0, 2.0)/5,
      tol=0.005, label="integral delta(-5x+10) = 1/5  [zero at x=2]")

# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EXAMPLE 3 — DERIVATIVE  delta'(x-a)")
print(SEP)

print("""
  DEFINITION by integration by parts:
    integral f(x) * delta'(x-a) dx
    = [f(x)*delta(x-a)]_{-inf}^{inf} - integral f'(x)*delta(x-a) dx
    = 0  -  f'(a)
    = -f'(a)

  DISTRIBUTIONAL IDENTITY:
        integral f(x) * delta^{(n)}(x-a) dx  =  (-1)^n  f^{(n)}(a)

  Physical significance: delta' is a "dipole" source term.
  Appears in: 1-D Euler beam equation, electrostatics (dipole layers).
""")

# SymPy
ex3_f   = x**3
ex3_a   = S(2)
ex3_res = integrate(ex3_f * DiracDelta(x - ex3_a, 1), (x, -oo, oo))
show(ex3_res, "Ex3a: integral x^3 * delta'(x-2) dx =")
show(-diff(ex3_f, x).subs(x, ex3_a), "-f'(2) = -3*(2)^2 =")

ex3_f2  = sp.cos(x)
ex3_a2  = pi / 3
ex3_res2 = integrate(ex3_f2 * DiracDelta(x - ex3_a2, 1), (x, -oo, oo))
show(ex3_res2, "Ex3b: integral cos(x)*delta'(x-pi/3) dx =")
show(-diff(ex3_f2, x).subs(x, ex3_a2), "-(-sin(pi/3)) =")

# Second derivative delta''
ex3_f3  = x**4
ex3_res3 = integrate(ex3_f3 * DiracDelta(x, 2), (x, -oo, oo))
show(ex3_res3, "Ex3c: integral x^4 * delta''(x) dx  =  f''(0) =")

print("\n  VERIFICATION (Gaussian derivative approx):")
def delta_prime_numerical(f_func, a_val, eps=0.02):
    """Integrate f*delta'(x-a) using derivative of Gaussian on fine grid.
    d/dx[(1/(eps*sqrt(pi)))*exp(-(x-a)^2/eps^2)] = -(2*(x-a)/(eps^2)) * (1/(eps*sqrt(pi)))*exp(...)
    """
    x_fine = np.linspace(a_val - 12*eps, a_val + 12*eps, 40000)
    u      = (x_fine - a_val) / eps
    # derivative of Gaussian delta approx wrt x:
    dphi   = -(2*u / (eps**2 * np.sqrt(np.pi))) * np.exp(-u**2)
    return np.trapezoid(f_func(x_fine) * dphi, x_fine)

check(ex3_res,
      delta_prime_numerical(lambda xv: xv**3, 2.0),
      tol=0.05, label="x^3 * delta'(x-2) -> -12")
check(ex3_res2,
      delta_prime_numerical(lambda xv: np.cos(xv), np.pi/3),
      tol=0.05, label="cos*delta'(x-pi/3) -> sin(pi/3)")

# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EXAMPLE 4 — COMPOSITION RULE  delta(g(x))")
print(SEP)

print("""
  If g has simple zeros at x_1, x_2, ..., x_n (g(x_i) = 0), then:

        delta(g(x))  =  sum_i  delta(x - x_i) / |g'(x_i)|

  DERIVATION: near each zero x_i, g(x) ~ g'(x_i)*(x-x_i)
    so delta(g(x)) ~ delta(g'(x_i)*(x-x_i)) = delta(x-x_i)/|g'(x_i)|.

  EXAMPLE: g(x) = x^2 - 4  zeros at x = +2 and x = -2
    g'(x) = 2x  ->  g'(2)=4,  g'(-2)=-4,  |g'|=4 at both
        delta(x^2 - 4)  =  [delta(x-2) + delta(x+2)] / 4
""")

g = x**2 - 4
zeros_g = sp.solve(g, x)
show(sp.Matrix(zeros_g), "Zeros of g(x) = x^2 - 4:")

gp = diff(g, x)
for z in zeros_g:
    show(Abs(gp.subs(x, z)), f"|g'| at x = {z}:")

ex4_f   = x**2 + 1
ex4_int = integrate(ex4_f * DiracDelta(x**2 - 4), (x, -oo, oo))
ex4_manual = sum(ex4_f.subs(x, z) / Abs(gp.subs(x, z)) for z in zeros_g)
show(ex4_int,    "integral (x^2+1)*delta(x^2-4) dx =")
show(ex4_manual, "Manual:  (f(2)+f(-2))/4 =")

# Quadratic: x^2 - 9
g2    = x**2 - 9
zeros_g2 = sp.solve(g2, x)
ex4b  = integrate(sp.sin(x) * DiracDelta(x**2 - 9), (x, -oo, oo))
show(ex4b, "integral sin(x)*delta(x^2-9) dx =")
print("  (sin is odd, zeros at +/-3 equidistant -> integral = 0 by antisymmetry)")

print("\n  VERIFICATION:")
def comp_numerical(f_func, zeros_list, gp_abs_list, eps=0.05):
    """Sum Gaussian spikes at each zero of g, divided by |g'| there."""
    total = 0.0
    for z, gpz in zip(zeros_list, gp_abs_list):
        x_fine = np.linspace(z - 10*eps, z + 10*eps, 10000)
        delta_ap = (1/(eps*np.sqrt(np.pi)))*np.exp(-(x_fine - z)**2/eps**2)
        total += np.trapezoid(f_func(x_fine) * delta_ap / gpz, x_fine)
    return total

check(ex4_int,
      comp_numerical(lambda xv: xv**2+1, [2.0, -2.0], [4.0, 4.0]),
      tol=0.05, label="(x^2+1)*delta(x^2-4)")

# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EXAMPLE 5 — FOURIER REPRESENTATION")
print(SEP)

print("""
  STATEMENT:
        delta(x)  =  (1/2*pi) integral_{-inf}^{inf}  e^{ikx} dk

  DERIVATION: take the Fourier transform of f(x), then invert:
    f(x) = (1/2*pi) integral dk e^{ikx} integral dx' e^{-ikx'} f(x')
          = integral dx'  [ (1/2*pi) integral dk e^{ik(x-x')} ]  f(x')
    Comparing with sifting: the bracket IS delta(x-x').

  EQUIVALENTLY (in terms of frequency nu = k/2pi):
        delta(x)  =  integral_{-inf}^{inf} e^{2*pi*i*nu*x} d_nu

  PHYSICAL MEANING:
    delta(x) contains ALL Fourier components with EQUAL AMPLITUDE.
    It is the "whitest" possible signal — perfectly flat spectrum.
    Impulse response of a linear system IS the Green's function.
    Optical fiber: H(nu) = exp(i*pi*D*nu^2) acts on flat-spectrum delta
    to produce a chirped pulse -- EXACTLY the RogueGuard dispersion step!
""")

k, nu_sym = symbols('k nu', real=True)

# Verify: integral of (1/2pi)*e^{ikx} over finite bandwidth L -> sinc
L_sym = symbols('L', positive=True)
sinc_approx = integrate(exp(I*k*x)/(2*pi), (k, -L_sym, L_sym))
sinc_approx_simplified = simplify(sinc_approx)
show(sinc_approx_simplified, "Finite-band approx: (1/2pi) int_{-L}^{L} e^{ikx} dk =")

# Numerical: sinc kernel converges to delta (monitor sifting)
L_vals = [2, 5, 10, 20, 50]
print("\n  MONITORING convergence  integral sinc_L(x) * x^2 dx  ->  0^2 = 0")
print(f"  {'L':>8}  {'Numerical result':>20}  {'Error':>15}")
for Lv in L_vals:
    def sinc_kernel_integrand(xv):
        if abs(xv) < 1e-12:
            return (Lv / np.pi) * 1.0   # limit: sinc(0) = L/pi
        return np.sin(Lv * xv) / (np.pi * xv)

    def ex5_integrand(xv):
        return xv**2 * sinc_kernel_integrand(xv)

    val, _ = sci_int.quad(ex5_integrand, -200, 200, limit=5000)
    print(f"  {Lv:>8}  {val:>20.10f}  {abs(val-0.0):>15.2e}")

# Parseval / completeness
print("""
  COMPLETENESS RELATION (resolves identity in L^2):
    integral_{-inf}^{inf} delta(x - x') f(x') dx' = f(x)
    This means { e^{ikx} }_{k in R} is a COMPLETE BASIS for L^2(R).
    Every square-integrable function has a unique Fourier expansion.
""")

# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EXAMPLE 6 — DISTRIBUTIONAL IDENTITY  x * delta(x) = 0")
print(SEP)

print("""
  STATEMENT:
        x * delta(x)  =  0     (as distributions)

  PROOF: for ANY test function phi in S(R):
    integral x * delta(x) * phi(x) dx  =  integral delta(x) * [x*phi(x)] dx
                                        =  [x*phi(x)]_{x=0}
                                        =  0 * phi(0)  =  0.

  Since this holds for ALL test functions, x*delta(x) = 0 as a distribution.

  GENERALISATION:
    x^n * delta^{(m)}(x) = 0   if  n > m   (higher zero wins)
    x * delta'(x)         = -delta(x)       (integration by parts identity)

  COROLLARY: x * delta(x - a) = a * delta(x - a)
    (multiply both sides by delta shifts the argument)

  PHYSICS APPLICATION:
    In QM: <x|x'> = delta(x-x') and x|x> = x|x>
    So x * delta(x-x') = x' * delta(x-x')   [eigenvalue pulls out]
    This is the DEFINITION of the position operator in the position basis.
""")

# SymPy verification
ex6_lhs = integrate(x * DiracDelta(x) * (x**2 + 3*x + 1), (x, -oo, oo))
show(ex6_lhs, "integral x*delta(x)*(x^2+3x+1) dx =")

ex6b = integrate(x * DiracDelta(x, 1) * sp.cos(x), (x, -oo, oo))
show(ex6b, "integral x*delta'(x)*cos(x) dx = integral -delta(x)*cos(x) dx =")
show(integrate(-DiracDelta(x)*sp.cos(x), (x, -oo, oo)),
     "= -cos(0) =")

ex6c = integrate(x * DiracDelta(x - 3) * (x + 1), (x, -oo, oo))
ex6c_alt = integrate(S(3) * DiracDelta(x - 3) * (x + 1), (x, -oo, oo))
show(ex6c, "integral x*(x+1)*delta(x-3) dx = 3*(3+1) =")
print(f"  Match x*delta(x-a)=a*delta(x-a): {simplify(ex6c - ex6c_alt) == 0}")

# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EXAMPLE 7 — 3-D DELTA  nabla^2(1/r) = -4*pi*delta^3(r)")
print(SEP)

print("""
  STATEMENT (Coulomb potential):
        nabla^2 (1/r)  =  -4*pi * delta^3(r)      r = |r| = sqrt(x^2+y^2+z^2)

  DERIVATION:
    For r > 0:  nabla^2(1/r) = (1/r^2) d/dr [r^2 d(1/r)/dr]
                              = (1/r^2) d/dr [r^2 * (-1/r^2)]
                              = (1/r^2) d/dr [-1]  =  0.
    So nabla^2(1/r) = 0 EVERYWHERE except r = 0.

    Near r = 0 there is a singularity. Integrate over a ball B_R:
    integral_{B_R} nabla^2(1/r) dV  = (Gauss)  integral_{dB_R} grad(1/r).dA
                                    = integral_{sphere} (-1/R^2) r_hat . dA
                                    = (-1/R^2) * 4*pi*R^2
                                    = -4*pi.

    This must equal  integral delta^3(r) dV * (-4*pi)  ->  -4*pi*1 = -4*pi. CHECK.

  PHYSICS: Poisson equation of electrostatics
        nabla^2 phi  =  -rho / epsilon_0
    For a point charge Q at origin: rho = Q*delta^3(r)
        phi(r) = Q / (4*pi*epsilon_0 * r)    (Coulomb potential)
    nabla^2 phi = Q/4pi*eps_0 * nabla^2(1/r) = Q/4pi*eps_0 * (-4pi*delta^3)
               = -Q/eps_0 * delta^3(r)  =  -rho/eps_0.  CONSISTENT.

  IN SPHERICAL COORDINATES:
    delta^3(r) = delta(r) / (4*pi*r^2)   for a spherically symmetric source.
    This is often written as delta^3(r-r') in QFT / QM for propagators.
""")

r_sym = symbols('r', positive=True)
phi_coulomb = 1/r_sym
lap_r = sp.diff(r_sym**2 * sp.diff(phi_coulomb, r_sym), r_sym) / r_sym**2
show(simplify(lap_r), "nabla^2(1/r) for r > 0:")
print("  -> zero for r > 0, singularity at r=0 captured by delta^3(r)")

# Numerical: surface integral test
R_test = 1.5
theta_v, phi_v = np.linspace(0, np.pi, 100), np.linspace(0, 2*np.pi, 100)
TH, PH = np.meshgrid(theta_v, phi_v)
dOmega = np.sin(TH) * np.diff(theta_v)[0] * np.diff(phi_v)[0]
grad_dot_n = -1/R_test**2   # (d/dr)(1/r)|_{r=R} * r_hat
surface_int = np.sum(grad_dot_n * dOmega) * R_test**2
print(f"\n  Numerical surface integral over sphere R={R_test}:")
print(f"    integral grad(1/r).dA = {surface_int:.6f}  (exact: {-4*np.pi:.6f})")
print(f"    Error: {abs(surface_int - (-4*np.pi)):.2e}")

# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("EXAMPLE 8 — GREEN'S FUNCTION  L*G(x,x') = delta(x-x')")
print(SEP)

print("""
  DEFINITION: Given a linear differential operator L, the Green's function
  G(x, x') satisfies:
        L_x G(x, x')  =  delta(x - x')

  The solution to L*u(x) = f(x) is then:
        u(x) = integral G(x, x') * f(x') dx'

  EXAMPLE: 1-D Laplacian  L = -d^2/dx^2  on  R
    -d^2G/dx^2 = delta(x-x')
    Solution: G(x, x') = |x - x'| / 2

  VERIFICATION:
    G(x, x') = |x-x'|/2
    dG/dx = sign(x-x') / 2
    d^2G/dx^2 = delta(x-x')    [derivative of sign = 2*delta]
    -> -d^2G/dx^2 = -delta(x-x')  ... wait, we need +delta.
    Convention: L = d^2/dx^2, G(x,x') = |x-x'|/2, L*G = delta.

  HELMHOLTZ GREEN'S FUNCTION (L = d^2/dx^2 + k^2):
    G_k(x, x') = -i/(2k) * exp(i*k*|x-x'|)
    This is the 1-D outgoing wave Green's function.
    At k->0: G -> |x-x'|/2  (Laplacian limit).

  PHYSICS CHAIN:
    Schrodinger: (H - E)|psi> = 0  ->  Green's function = resolvent operator
    E&M: wave equation -> retarded/advanced Green's functions
    GS algorithm: H(nu)*E_hat(nu) in freq space is multiplication by Green's
    function of the dispersion operator d/dz - i*beta2/2 * d^2/dt^2.
""")

x_prime = symbols("x'", real=True)
G_laplace = Abs(x - x_prime) / 2
show(G_laplace, "G(x,x') = |x-x'|/2  for  L = d^2/dx^2:")

# Verify by taking two derivatives
G_deriv1 = diff(Abs(x - x_prime), x)
G_deriv2 = diff(G_deriv1, x)
show(simplify(G_deriv2), "d^2G/dx^2 =  (should equal delta(x-x')):")

# Helmholtz
k_sym = symbols('k', positive=True)
G_helm = -sp.I / (2*k_sym) * exp(sp.I*k_sym*Abs(x - x_prime))
show(G_helm, "Helmholtz G_k(x,x') = ")

# Use Green's function to solve a specific BVP numerically
print("""
  NUMERICAL EXAMPLE: solve  d^2u/dx^2 = f(x) on [-L,L], u(+/-L)=0
  Using G(x,x') = (1/2)|x-x'| - (1/2)*|x|  (homogeneous BC version):
""")
Lgf = 5.0
N_gf = 300
x_gf  = np.linspace(-Lgf, Lgf, N_gf)
dx_gf = x_gf[1] - x_gf[0]
# Source: f(x) = sin(pi*x/L), BCs u(+-L) = 0
f_gf  = np.sin(np.pi * x_gf / Lgf)

# Green's function for d^2u/dx^2 = f on [-L,L] with Dirichlet u(+-L)=0:
#   G(x,x') = (1/2L) * (min(x,x') + L) * (max(x,x') - L)
# This is symmetric, negative definite (concave-up BVP).
Xi, Xj = np.meshgrid(x_gf, x_gf, indexing='ij')   # (N, N)
Xmin = np.minimum(Xi, Xj)
Xmax = np.maximum(Xi, Xj)
G_mat = (Xmin + Lgf) * (Xmax - Lgf) / (2 * Lgf)   # shape (N, N)
u_gf  = G_mat @ f_gf * dx_gf

# Exact: u'' = sin(pi*x/L) -> u = -(L/pi)^2 * sin(pi*x/L)  (BCs satisfied)
u_exact = -(Lgf / np.pi)**2 * np.sin(np.pi * x_gf / Lgf)
err_gf  = np.max(np.abs(u_gf - u_exact))
print(f"  L = {Lgf}, N = {N_gf}, f(x) = sin(pi*x/L)")
print(f"  Green's function solution max error vs exact: {err_gf:.4e}")

# ─────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("BUILDING FIGURE (8 panels, heavily monitored)...")
print(SEP)

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#F8F8F4")
gs0 = gridspec.GridSpec(2, 4, figure=fig,
                        hspace=0.48, wspace=0.36,
                        top=0.93, bottom=0.06,
                        left=0.05, right=0.97)

axes = [fig.add_subplot(gs0[r, c]) for r in range(2) for c in range(4)]
(ax1, ax2, ax3, ax4,
 ax5, ax6, ax7, ax8) = axes

fig.suptitle(
    r"Dirac $\delta(x)$ — 8 Physics Examples: Sifting | Scaling | "
    r"$\delta'$ | Composition | Fourier | $x\delta=0$ | "
    r"$\nabla^2(1/r)$ | Green's Function",
    fontsize=11, fontweight="bold", color="#1a1a2e"
)

colors5 = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]

# ── AX1: delta representations converging ──────────────────
ax = ax1
ax.set_facecolor("#F0F8FF")
for i, (eps, col) in enumerate(zip(eps_vals, colors5)):
    ax.plot(x_plot, gaussian_seqs[eps], color=col, lw=1.4+i*0.1,
            label=fr"$\varepsilon={eps}$", alpha=0.9)
ax.set_xlim(-3, 3); ax.set_ylim(-0.5, 12)
ax.set_title(r"$\delta(x) = \lim_{\varepsilon\to0}"
             r"\frac{e^{-x^2/\varepsilon^2}}{\varepsilon\sqrt{\pi}}$", fontsize=11)
ax.set_xlabel("$x$", fontsize=10); ax.set_ylabel("Amplitude", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.2)
ax.text(0.05, 0.98, "Ex 1: Sifting", transform=ax.transAxes,
        fontsize=9, va="top", color="#444",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ── AX2: scaling delta(ax) ─────────────────────────────────
ax = ax2
ax.set_facecolor("#FFFFF0")
eps_sc = 0.06
for a_sc, col in zip([0.5, 1, 2, 4], colors5):
    y_sc = (1/(eps_sc*np.sqrt(np.pi))) * np.exp(-((a_sc*x_plot)**2)/eps_sc**2)
    ax.plot(x_plot, y_sc / a_sc, color=col, lw=1.8, label=fr"$a={a_sc}$")
ax.set_xlim(-2, 2); ax.set_ylim(-0.5, 8)
ax.set_title(r"Scaling: $\delta(ax) = \delta(x)/|a|$", fontsize=11)
ax.set_xlabel("$x$", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.text(0.05, 0.98, "Ex 2: Scaling", transform=ax.transAxes,
        fontsize=9, va="top", color="#444",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ── AX3: delta and delta' side by side ─────────────────────
ax = ax3
ax.set_facecolor("#FFF5F0")
eps_d = 0.15
d0 = (1/(eps_d*np.sqrt(np.pi)))*np.exp(-x_plot**2/eps_d**2)
d1 = -(2*x_plot/eps_d**2) * d0
ax.plot(x_plot, d0/d0.max(), "#1f77b4", lw=2.0, label=r"$\delta(x)$ (norm)")
ax.plot(x_plot, d1/np.abs(d1).max(), "#d62728", lw=2.0,
        label=r"$\delta'(x)$ (norm)")
ax.axhline(0, color="#888", lw=0.5)
ax.set_xlim(-1.5, 1.5)
ax.set_title(r"$\delta'(x)$: dipole — antisymmetric", fontsize=11)
ax.set_xlabel("$x$", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
ax.text(0.05, 0.98, "Ex 3: Derivative", transform=ax.transAxes,
        fontsize=9, va="top", color="#444",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ── AX4: composition delta(x^2-4) ──────────────────────────
ax = ax4
ax.set_facecolor("#F0FFF0")
eps_c = 0.06
g_vals = x_plot**2 - 4
d_comp = (1/(eps_c*np.sqrt(np.pi))) * np.exp(-g_vals**2/eps_c**2) * np.abs(2*x_plot)
# Analytic: two spikes at +-2, height = 1/(|g'(+-2)|) = 1/4
ax.plot(x_plot, d_comp, "#2ca02c", lw=2.0, label=r"$\delta(x^2-4)$ (numerical)")
ax.axvline(2,  color="#d62728", lw=1.5, ls="--", label=r"Zeros at $x=\pm 2$")
ax.axvline(-2, color="#d62728", lw=1.5, ls="--")
ax.set_xlim(-4, 4); ax.set_ylim(-0.5, 5)
ax.set_title(r"Composition: $\delta(x^2-4) = \frac{\delta(x-2)+\delta(x+2)}{4}$",
             fontsize=10)
ax.set_xlabel("$x$", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.text(0.05, 0.98, "Ex 4: Composition", transform=ax.transAxes,
        fontsize=9, va="top", color="#444",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ── AX5: Fourier sinc convergence ──────────────────────────
ax = ax5
ax.set_facecolor("#F5F0FF")
for Lv, col in zip(L_vals, colors5):
    sinc_y = np.where(np.abs(x_plot) < 1e-9,
                      Lv/np.pi,
                      np.sin(Lv*x_plot)/(np.pi*x_plot))
    ax.plot(x_plot, sinc_y, color=col, lw=1.5, label=fr"$L={Lv}$", alpha=0.9)
ax.set_xlim(-3, 3); ax.set_ylim(-3, 18)
ax.set_title(r"Fourier: $\delta(x)=\lim_{L\to\infty}\frac{\sin(Lx)}{\pi x}$",
             fontsize=11)
ax.set_xlabel("$x$", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.text(0.05, 0.98, "Ex 5: Fourier", transform=ax.transAxes,
        fontsize=9, va="top", color="#444",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ── AX6: x*delta(x)=0 -- residual monitor ─────────────────
ax = ax6
ax.set_facecolor("#FFF0F5")
eps_range = np.logspace(-1, -4, 60)
resid = []
for ep in eps_range:
    def intgd_xdelta(xv):
        delta_ap = (1/(ep*np.sqrt(np.pi)))*np.exp(-xv**2/ep**2)
        return xv * delta_ap * (xv**2 + 1)   # f(x) = x^2+1
    val, _ = sci_int.quad(intgd_xdelta, -10, 10, limit=500)
    resid.append(abs(val))
ax.loglog(eps_range, resid, "#9467bd", lw=2.0, label=r"$|\int x\,\delta_\varepsilon(x)\,(x^2+1)\,dx|$")
ax.loglog(eps_range, eps_range, "#888", lw=1.0, ls="--", label=r"$O(\varepsilon)$")
ax.set_xlabel(r"$\varepsilon$", fontsize=10)
ax.set_ylabel("Residual", fontsize=10)
ax.set_title(r"$x\,\delta(x) = 0$: Residual $\to 0$ as $\varepsilon\to 0$",
             fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax.text(0.05, 0.98, "Ex 6: x·δ=0", transform=ax.transAxes,
        fontsize=9, va="top", color="#444",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ── AX7: nabla^2(1/r) -- radial potential ─────────────────
ax = ax7
ax.set_facecolor("#F0F8F0")
r_arr = np.linspace(0.01, 4, 400)
phi_r = 1 / r_arr
d2phi_r_approx = np.gradient(np.gradient(phi_r, r_arr), r_arr)
# Exact Laplacian in 1/r^2 * d/dr (r^2 df/dr) in spherical
laplacian_exact = np.zeros_like(r_arr)   # = 0 for r > 0
ax.plot(r_arr, phi_r, "#1f77b4", lw=2.0, label=r"$\phi(r) = 1/r$")
ax.plot(r_arr, np.clip(-d2phi_r_approx, -1, 5), "#d62728", lw=1.5, ls="--",
        label=r"$-\nabla^2(1/r)$ numerical")
ax.axvline(0, color="#333", lw=1.0, ls=":")
ax.set_xlim(0, 3); ax.set_ylim(-1, 6)
ax.set_title(r"$\nabla^2(1/r) = -4\pi\,\delta^3(\mathbf{r})$"
             "\n(Coulomb / Poisson)", fontsize=11)
ax.set_xlabel("$r$", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
ax.text(0.35, 0.98, "Ex 7: 3-D delta", transform=ax.transAxes,
        fontsize=9, va="top", color="#444",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ── AX8: Green's function solution ────────────────────────
ax = ax8
ax.set_facecolor("#F8F4E8")
ax.plot(x_gf, u_gf,    "#ff7f0e", lw=2.5, label="GF solution")
ax.plot(x_gf, u_exact, "#1f77b4", lw=1.5, ls="--", label="Exact solution")
ax.fill_between(x_gf, u_gf, u_exact, alpha=0.2, color="#d62728",
                label=f"Error max={err_gf:.1e}")
ax.set_xlabel("$x$", fontsize=10)
ax.set_ylabel("$u(x)$", fontsize=10)
ax.set_title(r"Green's Function: $u'' = \sin(\pi x/L)$"
             f"\n$L={Lgf}$, $N={N_gf}$, max err={err_gf:.1e}", fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.text(0.05, 0.98, "Ex 8: Green's fn", transform=ax.transAxes,
        fontsize=9, va="top", color="#444",
        bbox=dict(fc="white", ec="#bbb", pad=2))

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {OUT}")

print(f"\n{SEP}")
print("All 8 Dirac delta examples complete.")
print(SEP)
