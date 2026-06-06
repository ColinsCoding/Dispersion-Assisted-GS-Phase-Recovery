"""
_repl_calc_qec_bessel.py
========================
Filling schooling gaps: vector calculus, power/log/exp calculus,
Bessel functions, distribution theory, Dirac delta proofs,
quantum error correction (QEC) in qubits.

S1: Power functions and rules -- x^2, x^3, x^-1
    Derivatives, integrals, second derivatives, Taylor expansions
    Logarithmic differentiation (product/quotient of many terms)
    Exponential integrals: INT e^(ax), INT x*e^x, INT e^(x^2)

S2: Vector calculus
    Gradient, divergence, curl, Laplacian -- definitions + SymPy
    Del operator algebra identities
    Stokes theorem, divergence (Gauss) theorem -- statement + meaning
    Connection to Maxwell's equations

S3: Bessel functions
    Bessel ODE derivation from Laplacian in cylindrical coords
    Series solution J_n(x) -- radius of convergence = inf
    J_0, J_1 zeros (crucial for waveguide modes)
    Y_n (Neumann, singular at 0), modified I_n, K_n
    Applications: LP modes in optical fiber, drum modes

S4: Distribution theory (rigorous)
    Test function space D, Schwartz space S
    Linear functional definition of distribution
    Formal proof: delta is a distribution (linear + continuous)
    Tempered distributions, Fourier transform of delta

S5: Dirac delta -- proofs and derivations
    Sifting property proof (Gaussian sequence)
    Scaling: delta(ax) = delta(x)/|a| -- rigorous proof
    Derivative: INT f*delta' = -f'(a) -- integration by parts proof
    Composition: delta(g(x)) = SUM delta(x-xi)/|g'(xi)| -- proof
    Fourier representation proof: FT[delta]=1, inverse -> delta

S6: Quantum error correction (QEC)
    Why errors happen: decoherence, gate errors, measurement noise
    The no-cloning theorem -- why classical repetition fails
    Stabilizer formalism -- Pauli group, stabilizer codes
    3-qubit bit-flip code, 3-qubit phase-flip code
    Shor 9-qubit code = bit-flip + phase-flip
    Steane [[7,1,3]] code -- encodes 1 logical in 7 physical
    Fault tolerance threshold: ~1% physical error rate
    Surface code: most promising, threshold ~1%, 2D nearest-neighbor
    RogueGuard optical QEC connection

Output: repl/_out_calc_qec_bessel.png
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
from sympy import (symbols, Function, diff, integrate, exp, log, sin, cos,
                   sqrt, pi, oo, I, Rational, simplify, series, latex,
                   DiracDelta, Heaviside, limit, factorial, besselj, bessely,
                   besseli, besselk, atan2, Matrix, eye, zeros)
import os

OUT = os.path.join(os.path.dirname(__file__), "_out_calc_qec_bessel.png")
SEP = "=" * 65

# ============================================================
# S1: POWER FUNCTIONS, LOG DIFF, EXPONENTIAL INTEGRALS
# ============================================================
print(SEP)
print("SECTION 1: POWER / LOG / EXPONENTIAL CALCULUS")
print(SEP)

x, a, n_s, t = symbols("x a n t", real=True)
x_pos = symbols("x", positive=True)

print("""
  POWER RULE (the engine of calculus):
    d/dx [x^n] = n * x^(n-1)    for all real n != 0
    INT x^n dx = x^(n+1)/(n+1) + C    for n != -1

  KEY CASES:
""")

cases = [
    ("x^0 = 1",    "0",             "x + C"),
    ("x^1",        "1",             "x^2/2 + C"),
    ("x^2",        "2x",            "x^3/3 + C"),
    ("x^3",        "3x^2",          "x^4/4 + C"),
    ("x^(-1)=1/x", "x^(-2) = -1/x^2","ln|x| + C  (SPECIAL CASE n=-1)"),
    ("x^(-2)",     "-2x^(-3)",      "-x^(-1) + C = -1/x + C"),
    ("x^(1/2)",    "(1/2)x^(-1/2)", "(2/3)x^(3/2) + C"),
    ("x^(-1/2)",   "(-1/2)x^(-3/2)","2x^(1/2) + C"),
]

print(f"  {'f(x)':<20} {'f_prime(x)':<25} {'INT f(x) dx'}")
print(f"  {'-'*65}")
for f_, fp, intf in cases:
    print(f"  {f_:<20} {fp:<25} {intf}")

# SymPy verification
print("\n  SYMPY VERIFICATION:")
for expr, label in [(x**2,"x^2"), (x**3,"x^3"), (1/x,"1/x"), (x**(-2),"x^-2")]:
    d  = diff(expr, x)
    it = integrate(expr, x)
    print(f"    d/dx [{label}] = {d},   INT [{label}] dx = {it}")

# WHY x^-1 is special
print("""
  WHY x^-1 IS SPECIAL:
    INT x^n dx = x^(n+1)/(n+1)  fails at n=-1 (denominator = 0).
    Instead: INT (1/x) dx = ln|x| + C.
    Proof via FTC: d/dx [ln|x|] = 1/x  for x != 0.
    This is why the natural log appears everywhere -- it is the
    "missing" antiderivative that completes the power rule.

  SECOND DERIVATIVES (curvature):
    d^2/dx^2 [x^2] = 2     (constant: parabola has constant curvature)
    d^2/dx^2 [x^3] = 6x    (linear: inflection at x=0)
    d^2/dx^2 [1/x] = 2/x^3 (always positive for x>0: concave up)
""")

# Logarithmic differentiation
print("  LOGARITHMIC DIFFERENTIATION:")
print("""
    Take ln of both sides, then differentiate:
    If y = f1(x)^a * f2(x)^b / (f3(x)^c)
    ln(y) = a*ln(f1) + b*ln(f2) - c*ln(f3)
    y'/y  = a*f1'/f1 + b*f2'/f2 - c*f3'/f3
    y'    = y * [a*f1'/f1 + b*f2'/f2 - c*f3'/f3]

    POWER FUNCTION x^x (cannot use power or exponential rule alone):
    y = x^x  =>  ln(y) = x*ln(x)
    y'/y = ln(x) + x*(1/x) = ln(x) + 1
    y' = x^x * (ln(x) + 1)
""")

y_xx = x_pos**x_pos
dy_xx = diff(y_xx, x_pos)
print(f"  SymPy d/dx[x^x] = {dy_xx}")

# More examples
print("\n  MORE LOG DIFF EXAMPLES:")
examples_ld = [
    (x_pos**3 * (x_pos+1)**2 / (x_pos-1), "x^3*(x+1)^2/(x-1)"),
    (sqrt(x_pos) * exp(x_pos) / (x_pos**2 + 1), "sqrt(x)*e^x/(x^2+1)"),
]
for expr, label in examples_ld:
    dy = simplify(diff(expr, x_pos))
    print(f"    d/dx [{label}] = {dy}")

# Exponential integrals
print("\n  EXPONENTIAL INTEGRALS:")
exp_integrals = [
    (exp(a*x),        "e^(ax)",    f"e^(ax)/a"),
    (x*exp(x),        "x*e^x",    "e^x*(x-1)  [IBP]"),
    (x**2*exp(x),     "x^2*e^x",  "e^x*(x^2-2x+2)  [IBP twice]"),
    (exp(-x**2),      "e^(-x^2)", "sqrt(pi)/2 * erf(x)  [no closed form]"),
    (x*exp(-x**2),    "x*e^(-x^2)","-(1/2)*e^(-x^2)  [substitution]"),
    (exp(x)/x,        "e^x/x",    "Ei(x)  [exponential integral, no elementary form]"),
]
print(f"  {'f(x)':<25} {'INT f dx'}")
print(f"  {'-'*55}")
for expr, label, antideriv in exp_integrals:
    print(f"  {label:<25} {antideriv}")

# SymPy check key ones
print("\n  SymPy verification:")
for expr, label in [(exp(a*x), "e^(ax)"), (x*exp(x), "x*e^x"),
                    (x**2*exp(x), "x^2*e^x")]:
    it = simplify(integrate(expr, x))
    print(f"    INT {label} dx = {it}")

# Gaussian
gauss_result = integrate(exp(-x**2), (x, -oo, oo))
print(f"    INT e^(-x^2) from -inf to inf = {gauss_result}  (Gaussian integral)")

# ============================================================
# S2: VECTOR CALCULUS
# ============================================================
print(f"\n{SEP}")
print("SECTION 2: VECTOR CALCULUS")
print(SEP)

print("""
  DEL OPERATOR (nabla):
    In Cartesian 3D:  nabla = x_hat d/dx + y_hat d/dy + z_hat d/dz

  THE THREE MAIN OPERATIONS:
  1. GRADIENT (scalar -> vector):
     grad(f) = nabla f = (df/dx, df/dy, df/dz)
     Points in direction of STEEPEST ASCENT of f.
     |grad f| = rate of change in that direction.
     grad(f) is perpendicular to level surfaces f = const.

  2. DIVERGENCE (vector -> scalar):
     div(F) = nabla . F = dFx/dx + dFy/dy + dFz/dz
     Measures OUTWARD FLUX per unit volume (source density).
     div(F) > 0: source (field spreading out)
     div(F) < 0: sink (field converging in)
     div(F) = 0: incompressible (no sources/sinks)

  3. CURL (vector -> vector):
     curl(F) = nabla x F = (dFz/dy-dFy/dz, dFx/dz-dFz/dx, dFy/dx-dFx/dy)
     Measures rotation/circulation of F.
     curl(F) = 0: irrotational (conservative field, path-independent)
     Right-hand rule: curl direction = axis of rotation.

  LAPLACIAN (scalar -> scalar):
     Laplacian(f) = div(grad(f)) = d^2f/dx^2 + d^2f/dy^2 + d^2f/dz^2
     Laplacian(f) > 0 at a point: f is below average of neighbors (concave up).
     Laplacian(f) = 0: HARMONIC function (averages its neighbors exactly).
""")

# SymPy vector calculus
x_v, y_v, z_v = symbols("x y z", real=True)

# Example scalar field
f_scal = x_v**2 * y_v + y_v**2 * z_v + z_v**2 * x_v

grad_f = [diff(f_scal, var) for var in (x_v, y_v, z_v)]
print("  EXAMPLE: f = x^2*y + y^2*z + z^2*x")
print(f"    grad(f) = ({grad_f[0]}, {grad_f[1]}, {grad_f[2]})")

lap_f = sum(diff(f_scal, var, 2) for var in (x_v, y_v, z_v))
print(f"    Laplacian(f) = {simplify(lap_f)}")

# Example vector field
Fx = x_v**2 * z_v
Fy = y_v**2 * x_v
Fz = z_v**2 * y_v
div_F = diff(Fx,x_v) + diff(Fy,y_v) + diff(Fz,z_v)
print(f"\n  EXAMPLE: F = (x^2*z, y^2*x, z^2*y)")
print(f"    div(F) = dFx/dx + dFy/dy + dFz/dz = {simplify(div_F)}")

curlx = diff(Fz,y_v) - diff(Fy,z_v)
curly = diff(Fx,z_v) - diff(Fz,x_v)
curlz = diff(Fy,x_v) - diff(Fx,y_v)
print(f"    curl(F) = ({simplify(curlx)}, {simplify(curly)}, {simplify(curlz)})")

print("""
  FUNDAMENTAL IDENTITIES (memorize these):
    div(curl F) = 0         (curl of anything has no divergence)
    curl(grad f) = 0        (gradient has no curl -- conservative)
    Laplacian(f) = div(grad f)
    div(f*F) = grad(f).F + f*div(F)    (product rule)
    curl(f*F) = grad(f) x F + f*curl(F)
    curl(curl F) = grad(div F) - Laplacian(F)   [vector Laplacian]
""")

# Verify div(curl F) = 0
dc = (diff(curlz,y_v) - diff(curly,z_v) +
      diff(curlx,z_v) - diff(curlz,x_v) +  # this is wrong structure but checking
      diff(curly,x_v) - diff(curlx,y_v))
div_curl = diff(curlx,x_v) + diff(curly,y_v) + diff(curlz,z_v)
print(f"  SymPy: div(curl F) = {simplify(div_curl)}  (= 0, verified)")

curl_grad = [diff(diff(f_scal,z_v),y_v) - diff(diff(f_scal,y_v),z_v),
             diff(diff(f_scal,x_v),z_v) - diff(diff(f_scal,z_v),x_v),
             diff(diff(f_scal,y_v),x_v) - diff(diff(f_scal,x_v),y_v)]
print(f"  SymPy: curl(grad f) = ({simplify(curl_grad[0])}, "
      f"{simplify(curl_grad[1])}, {simplify(curl_grad[2])})  (= 0, verified)")

print("""
  INTEGRAL THEOREMS:
  1. GRADIENT THEOREM (FTC in 3D):
     INT_C grad(f).dr = f(b) - f(a)
     Line integral of gradient = change in f.
     Path-independent: only endpoints matter.

  2. STOKES THEOREM:
     INT_S curl(F).dA = OINT_C F.dr
     Surface integral of curl = line integral around boundary.
     "Circulation = total rotation through surface"
     Maxwell: Faraday's law = Stokes applied to E field.

  3. DIVERGENCE THEOREM (Gauss):
     INT_V div(F) dV = OOINT_S F.dA
     Volume integral of divergence = flux through boundary surface.
     "Total sources inside = net flux out"
     Maxwell: Gauss's law = divergence theorem applied to E field.

  MAXWELL EQUATIONS (in terms of div/curl):
    div(E) = rho/eps0         [Gauss's law: charge is source of E]
    div(B) = 0                [no magnetic monopoles: B has no sources]
    curl(E) = -dB/dt          [Faraday: changing B creates E]
    curl(B) = mu0*J + mu0*eps0*dE/dt  [Ampere-Maxwell]
    Wave equation from curl of Faraday:
    curl(curl E) = -d/dt(curl B) = -mu0*eps0*d^2E/dt^2
    => Laplacian(E) = mu0*eps0 * d^2E/dt^2  (c^2 = 1/(mu0*eps0))
""")

# ============================================================
# S3: BESSEL FUNCTIONS
# ============================================================
print(f"\n{SEP}")
print("SECTION 3: BESSEL FUNCTIONS")
print(SEP)

print("""
  ORIGIN: LAPLACIAN IN CYLINDRICAL COORDINATES
    nabla^2 f = d^2f/dr^2 + (1/r)*df/dr + (1/r^2)*d^2f/dphi^2 + d^2f/dz^2

  SEPARATION: f(r,phi,z) = R(r)*Phi(phi)*Z(z)
    Phi'' = -m^2 * Phi  =>  Phi = e^(im*phi), m integer (periodicity)
    Z'' = k_z^2 * Z     =>  Z = e^(k_z*z)
    Radial equation:
    r^2 R'' + r R' + (k_r^2 * r^2 - m^2) R = 0   [BESSEL EQUATION]
    Let x = k_r*r:
    x^2 R'' + x R' + (x^2 - m^2) R = 0

  BESSEL EQUATION OF ORDER m:
    x^2 y'' + x y' + (x^2 - m^2) y = 0

  SOLUTIONS (via Frobenius method -- power series about x=0):
    J_m(x) = SUM_{k=0}^{inf} (-1)^k / (k! * Gamma(m+k+1)) * (x/2)^(2k+m)
    (Bessel function of FIRST kind -- finite at x=0)

    Y_m(x) = [J_m(x)*cos(m*pi) - J_{-m}(x)] / sin(m*pi)
    (Bessel function of SECOND kind / Neumann -- diverges at x=0)

  GENERAL SOLUTION:
    R(r) = A*J_m(k_r*r) + B*Y_m(k_r*r)
    For problems with r=0 included: B=0 (Y diverges there).
    For annular domain (hollow cylinder): keep both.

  MODIFIED BESSEL FUNCTIONS (imaginary argument):
    I_m(x) = i^(-m) J_m(ix)   [modified first kind -- exponentially growing]
    K_m(x) = (pi/2) * i^(m+1) * (J_m(ix) + i*Y_m(ix))  [decaying]
    For evanescent fields outside waveguide: K_m.
""")

# Series for J_0 and J_1
print("  J_0(x) SERIES (m=0):")
print("    J_0(x) = 1 - x^2/4 + x^4/64 - x^6/2304 + ...")
print("           = SUM_{k=0}^inf (-1)^k / (k!)^2 * (x/2)^(2k)")
print("\n  J_1(x) SERIES (m=1):")
print("    J_1(x) = x/2 - x^3/16 + x^5/384 - ...")
print("           = SUM_{k=0}^inf (-1)^k / (k! * (k+1)!) * (x/2)^(2k+1)")

# Key properties
print("""
  KEY PROPERTIES:
    Recurrence:  J_{m-1}(x) + J_{m+1}(x) = (2m/x) J_m(x)
                 J_{m-1}(x) - J_{m+1}(x) = 2 J_m'(x)
    Derivative:  d/dx [x^m J_m(x)] = x^m J_{m-1}(x)
                 d/dx [x^(-m) J_m(x)] = -x^(-m) J_{m+1}(x)
    J_0'(x) = -J_1(x)
    Orthogonality:
    INT_0^a r J_m(lambda_i r) J_m(lambda_j r) dr = 0  if i != j
    where lambda_i = zeros of J_m(lambda*a) / a

    ZEROS of J_0 (crucial for waveguide cutoff):
""")

zeros_J0 = sc.jn_zeros(0, 6)
zeros_J1 = sc.jn_zeros(1, 6)
zeros_J2 = sc.jn_zeros(2, 6)
print(f"    J_0 zeros: {[f'{z:.4f}' for z in zeros_J0]}")
print(f"    J_1 zeros: {[f'{z:.4f}' for z in zeros_J1]}")
print(f"    J_2 zeros: {[f'{z:.4f}' for z in zeros_J2]}")

print("""
  WAVEGUIDE / OPTICAL FIBER APPLICATION:
    LP modes (linearly polarized) in step-index fiber:
    Core (r < a):  E ~ J_m(k_r * r) * e^(im*phi) * e^(i*beta*z)
    Cladding (r>a): E ~ K_m(kappa * r) * e^(im*phi) * e^(i*beta*z)
    k_r = sqrt(n_core^2 * k0^2 - beta^2)    [transverse wavenumber in core]
    kappa = sqrt(beta^2 - n_clad^2 * k0^2)  [decay in cladding]
    Boundary conditions at r=a: match E, H tangential
    => eigenvalue equation: V-number cutoff at J_{m-1}(V) = 0
    V = (2*pi*a/lambda) * sqrt(n_core^2 - n_clad^2)
    Single-mode: V < 2.405 (first zero of J_0)

  DRUM MODES:
    Circular membrane: wave equation with J_m(k_mn*r)*cos(m*phi)
    Eigenfrequencies: omega_mn = (x_mn/a) * sqrt(T/sigma)
    x_mn = nth zero of J_m
    Lowest mode (m=0,n=1): x_01 = 2.4048 (J_0 first zero)
""")

# SymPy Bessel
x_bes = symbols("x", positive=True)
print("\n  SymPy Bessel examples:")
print(f"    J_0(0)  = {besselj(0, 0)}  (= 1)")
print(f"    J_1(0)  = {besselj(1, 0)}  (= 0)")
dJ0 = diff(besselj(0, x_bes), x_bes)
print(f"    d/dx J_0(x) = {simplify(dJ0)}  (= -J_1(x))")
print(f"    J_0 series: {series(besselj(0, x_bes), x_bes, 0, 8)}")

# ============================================================
# S4: DISTRIBUTION THEORY (RIGOROUS)
# ============================================================
print(f"\n{SEP}")
print("SECTION 4: DISTRIBUTION THEORY (RIGOROUS)")
print(SEP)

print("""
  THE PROBLEM WITH CLASSICAL FUNCTIONS:
    1/x is not integrable near 0.
    delta(x) has no pointwise definition yet INT delta = 1.
    d/dx |x| is not differentiable at 0 classically.
    Need a larger space that includes these as valid objects.

  TEST FUNCTION SPACES:
  D = C_c^inf(R): smooth functions with COMPACT SUPPORT.
    phi in D: infinitely differentiable, zero outside some bounded interval.
    Example: phi(x) = exp(-1/(1-x^2)) for |x|<1, 0 otherwise.
    phi is C^inf but NOT analytic at |x|=1 (Taylor series fails there).
    D is a vector space under pointwise addition and scalar mult.

  S = Schwartz space: smooth functions with rapid decay.
    phi in S: all derivatives decay faster than any polynomial.
    phi in S iff: sup |x^a * d^b phi/dx^b| < inf for all a,b >= 0.
    D is a SUBSET of S: D subset S subset L^2.
    FT maps S to S (Fourier transform stable on Schwartz space).

  DISTRIBUTIONS (generalized functions):
    A distribution T is a continuous linear functional on D:
    T: D -> R  (or C)
    satisfying:
    1. LINEARITY:    T[alpha*phi + beta*psi] = alpha*T[phi] + beta*T[psi]
    2. CONTINUITY:   phi_n -> phi in D => T[phi_n] -> T[phi]
    Written: T[phi] = <T, phi> = INT T(x)*phi(x) dx  (formal)

  REGULAR DISTRIBUTIONS (from ordinary functions):
    Any locally integrable function f defines a distribution T_f:
    <T_f, phi> = INT f(x) * phi(x) dx
    Consistent: if f is a classical function, T_f gives same results.

  SINGULAR DISTRIBUTIONS (no ordinary function):
    DELTA: <delta, phi> = phi(0)
    This is clearly linear: <delta, alpha*phi+beta*psi> = alpha*phi(0)+beta*psi(0)
    Continuity: if phi_n -> phi in D, then phi_n(0) -> phi(0) (pointwise).
    So delta IS a distribution, despite not being a function.

  DERIVATIVE OF A DISTRIBUTION:
    Define: <T', phi> = -<T, phi'>
    Motivation: integration by parts (boundary terms vanish for phi in D):
    INT T'(x)*phi(x) dx = [T*phi]_boundary - INT T(x)*phi'(x) dx = -INT T*phi'
    This extends differentiation to ALL distributions:
    EVERY distribution is infinitely differentiable!
    => delta'(x) defined by: <delta', phi> = -<delta, phi'> = -phi'(0)

  TEMPERED DISTRIBUTIONS (S' = dual of S):
    Larger space: polynomials, e^x, etc. become distributions on S.
    FT well-defined on S': FT[T] defined by <FT[T], phi> = <T, FT[phi]>
    => FT[delta] = 1  (the constant function 1).

  SUPPORT OF A DISTRIBUTION:
    supp(T) = complement of largest open set where T=0.
    supp(delta) = {0}  (delta is zero away from origin).
""")

# ============================================================
# S5: DIRAC DELTA -- PROOFS AND DERIVATIONS
# ============================================================
print(f"\n{SEP}")
print("SECTION 5: DIRAC DELTA -- PROOFS AND DERIVATIONS")
print(SEP)

print("""
  PROOF 1: SIFTING PROPERTY FROM GAUSSIAN SEQUENCE
  -------------------------------------------------
  Define: delta_eps(x) = (1/(eps*sqrt(pi))) * exp(-x^2/eps^2)
  Claim: lim_{eps->0} INT delta_eps(x)*f(x) dx = f(0)

  Proof:
    INT delta_eps(x)*f(x) dx
    = INT (1/(eps*sqrt(pi))) * exp(-x^2/eps^2) * f(x) dx
    Let u = x/eps, so x = eps*u, dx = eps*du:
    = INT (1/sqrt(pi)) * exp(-u^2) * f(eps*u) * du
    As eps -> 0: f(eps*u) -> f(0)  [f continuous]
    = f(0) * INT (1/sqrt(pi)) * exp(-u^2) du
    = f(0) * (1/sqrt(pi)) * sqrt(pi)
    = f(0)  QED

  PROOF 2: SCALING PROPERTY  delta(ax) = delta(x)/|a|
  ----------------------------------------------------
  Claim: INT delta(ax)*f(x) dx = f(0)/|a|  for a != 0.

  Proof (case a > 0):
    Let u = ax, so x = u/a, dx = du/a:
    INT delta(ax)*f(x) dx = INT delta(u)*f(u/a)*(du/a)
    = (1/a) * INT delta(u)*f(u/a) du
    = (1/a) * f(0/a)         [sifting at u=0]
    = (1/a) * f(0)

  Proof (case a < 0):  |a| = -a
    Let u = ax, dx = du/a, limits FLIP when a<0:
    INT_{-inf}^{inf} delta(ax)*f(x) dx
    = INT_{+inf}^{-inf} delta(u)*f(u/a) (du/a)
    = (-1/a) * INT_{-inf}^{inf} delta(u)*f(u/a) du  [flip limits = sign change]
    = (-1/a) * f(0) = (1/|a|) * f(0)

  Combined: delta(ax) = delta(x)/|a|  for all a != 0.  QED

  PROOF 3: DERIVATIVE  INT f(x)*delta'(x-a) dx = -f'(a)
  -------------------------------------------------------
  Definition of distributional derivative: <delta', phi> = -<delta, phi'>
  So: INT f(x)*delta'(x-a) dx  with g(x) = delta'(x-a), phi = f
  = <delta'(.-a), f> = -<delta(.-a), f'> = -f'(a)

  Alternatively, integration by parts (formal):
    INT_{-inf}^{inf} delta'(x-a)*f(x) dx
    = [delta(x-a)*f(x)]_{-inf}^{inf} - INT delta(x-a)*f'(x) dx
    = 0 - f'(a)    [boundary = 0 since delta=0 at +/-inf]
    = -f'(a)  QED

  PROOF 4: COMPOSITION  delta(g(x)) = SUM_i delta(x-xi)/|g'(xi)|
  ---------------------------------------------------------------
  where xi are simple zeros of g (g(xi) = 0, g'(xi) != 0).

  Proof:
    Near each zero xi, g(x) ~ g'(xi)*(x-xi)  [Taylor, first order]
    So: delta(g(x)) ~ delta(g'(xi)*(x-xi))
    Apply scaling: delta(a*u) = delta(u)/|a| with a = g'(xi), u = x-xi:
    delta(g'(xi)*(x-xi)) = delta(x-xi)/|g'(xi)|
    Sum contributions from all zeros:
    delta(g(x)) = SUM_i delta(x-xi) / |g'(xi)|  QED

  EXAMPLE: delta(x^2 - a^2) with a > 0
    Zeros: x = +a (g'(+a) = 2a) and x = -a (g'(-a) = -2a)
    => delta(x^2 - a^2) = delta(x-a)/(2a) + delta(x+a)/(2a)
    INT delta(x^2-4) dx = 1/(2*2) + 1/(2*2) = 1/2  CHECK.

  PROOF 5: FOURIER REPRESENTATION  delta(x) = (1/2pi) INT e^(ikx) dk
  -------------------------------------------------------------------
  Claim: FT[delta(x)](k) = INT delta(x)*e^(-ikx) dx = e^(-ik*0) = 1
  (sifting property with f(x) = e^(-ikx) evaluated at x=0)

  Inverse FT:
    f(x) = (1/2pi) INT F(k)*e^(ikx) dk
  With F(k) = 1 (the FT of delta):
    (1/2pi) INT 1 * e^(ikx) dk = delta(x)

  This is the FOURIER INVERSION THEOREM applied to the trivial case.
  Formal meaning: integrate e^(ikx) over all k at fixed x:
  - At x=0: all oscillations are 1, integral = +inf (delta peak)
  - At x!=0: oscillations cancel (Riemann-Lebesgue lemma), integral = 0

  PROOF 6: x*delta(x) = 0
  ------------------------
  For any test function f:
  <x*delta, f> = <delta, x*f>  [x is a smooth mult, dual action]
               = (x*f)(0)
               = 0 * f(0) = 0
  So x*delta = 0 as a distribution.  QED
  Corollary: delta(x) = 0 for x != 0, but is NOT the zero function.
""")

# SymPy proofs numerical checks
print("  NUMERICAL CHECKS (Gaussian approximation, eps=0.01):")
eps = 0.01
x_arr = np.linspace(-5, 5, 100000)
delta_eps = np.exp(-x_arr**2/eps**2) / (eps * np.sqrt(np.pi))

# Sifting: INT delta_eps * f = f(0)
for f_func, f_name, f_at_0 in [
    (np.cos(x_arr), "cos(x)", 1.0),
    (x_arr**2 + 1,  "x^2+1", 1.0),
    (np.exp(-x_arr**2), "e^(-x^2)", 1.0),
]:
    result = np.trapezoid(delta_eps * f_func, x_arr)
    print(f"    INT delta_eps*{f_name} = {result:.6f}  (exact: {f_at_0:.6f})")

# Scaling check
delta_3x = np.exp(-(3*x_arr)**2/eps**2) * 3 / (eps*np.sqrt(np.pi))  # delta_eps(3x)*|3|
result_scale = np.trapezoid(delta_3x * (x_arr**2 + 1), x_arr)
print(f"    INT delta(3x)*(x^2+1) = {result_scale:.6f}  (exact: (0+1)/3 = 0.333)")

# ============================================================
# S6: QUANTUM ERROR CORRECTION
# ============================================================
print(f"\n{SEP}")
print("SECTION 6: QUANTUM ERROR CORRECTION (QEC)")
print(SEP)

print("""
  WHY QEC IS NECESSARY:
    Quantum computers are fragile. Qubits interact with environment:
    - Decoherence: |0> -> alpha|0> + beta|1> (entangles with environment)
    - Bit-flip error: |0> -> |1>  (X error)
    - Phase-flip error: |+> -> |->  (Z error, alpha|0>+beta|1> -> alpha|0>-beta|1>)
    - Both (Y error = iXZ)
    Gate errors: each 2-qubit gate ~0.1-1% error rate (2024 state of art).
    T1 (energy relaxation): ~100-500 us (superconducting), ~minutes (trapped ion).
    T2 (dephasing): ~T1 (best case).

  THE NO-CLONING THEOREM:
    Cannot copy unknown quantum state: |psi>|0> =/> |psi>|psi>
    Proof: U(|psi>|0>) = |psi>|psi> for all |psi>
    Try: |psi> = (|0>+|1>)/sqrt(2)
    U(|psi>|0>) = |psi>|psi> = (|00>+|01>+|10>+|11>)/2
    But U is LINEAR:
    U(|psi>|0>) = U(|0>|0>+|1>|0>)/sqrt(2)
               = (|0>|0> + |1>|1>)/sqrt(2) = (|00>+|11>)/sqrt(2)
    Contradiction: (|00>+|01>+|10>+|11>)/2 != (|00>+|11>)/sqrt(2)
    => No universal quantum cloner exists.
    => Classical repetition code (copy bit 3 times) FAILS quantumly.

  SOLUTION: ENCODE, NOT COPY
    Instead of copying, use entanglement:
    Logical |0_L> and |1_L> are entangled states of many physical qubits.
    Errors change physical qubits but not the logical state.
    SYNDROME measurement: measure without collapsing logical qubit.

  3-QUBIT BIT-FLIP CODE:
    Encode: |0_L> = |000>,  |1_L> = |111>
    alpha|0_L> + beta|1_L> = alpha|000> + beta|111>
    Bit-flip on qubit 1: alpha|100> + beta|011>
    Syndrome: measure Z1Z2 and Z2Z3:
      Z1Z2 = +1 -> qubits 1,2 agree;  -1 -> they disagree
      Z2Z3 = +1 -> qubits 2,3 agree;  -1 -> they disagree
    Error syndromes:
      No error:    Z1Z2=+1, Z2Z3=+1  -> do nothing
      Flip on 1:   Z1Z2=-1, Z2Z3=+1  -> apply X_1
      Flip on 2:   Z1Z2=-1, Z2Z3=-1  -> apply X_2
      Flip on 3:   Z1Z2=+1, Z2Z3=-1  -> apply X_3
    Corrects ANY SINGLE bit-flip error.
    Does NOT correct phase-flip errors (Z errors).

  3-QUBIT PHASE-FLIP CODE:
    Encode in Hadamard basis:
    |0_L> = |+++>,  |1_L> = |-->
    (|+> = (|0>+|1>)/sqrt(2),  |-> = (|0>-|1>)/sqrt(2))
    Syndrome: measure X1X2 and X2X3.
    Corrects ANY SINGLE phase-flip (Z) error.
    Does NOT correct bit-flip (X) errors.

  SHOR 9-QUBIT CODE (first quantum error correcting code):
    Concatenate: phase-flip code around bit-flip code.
    |0_L> = (|000>+|111>)^3 / 2sqrt(2)
    |1_L> = (|000>-|111>)^3 / 2sqrt(2)
    Uses 9 physical qubits to encode 1 logical qubit.
    Corrects ANY single-qubit error (X, Y, or Z on any of 9 qubits).
    Distance d=3: can detect 2 errors, correct 1.
    Parameters: [[9, 1, 3]] -- n=9 physical, k=1 logical, d=3.

  STABILIZER FORMALISM:
    Pauli group G_n: {I,X,Y,Z}^n with phases {+1,-1,+i,-i}
    Stabilizer S: abelian subgroup of G_n (all elements commute).
    Code space C(S): all states |psi> with g|psi> = |psi> for all g in S.
    Dimension: 2^(n-k) stabilizer generators -> 2^k logical qubit states.
    Error detection: error E commutes with S -> undetectable; anticommutes -> syndrome.

  STEANE [[7,1,3]] CODE:
    n=7 physical qubits, k=1 logical, distance d=3.
    Based on classical [7,4] Hamming code.
    Generator matrix H (6 generators, each acts on 7 qubits):
    X-type: IIXXXXX, IXXIIXX, XIXIXIX (bit parity checks)
    Z-type: IIZZZZ,  IZZIZ,   ZIZIZIZ (phase parity checks)
    Logical operators:
    X_L = XXXXXXX, Z_L = ZZZZZZZ
    Threshold: if physical error rate p < ~1%, code reduces errors.
    Resources: 7 physical per logical.

  SURFACE CODE (most practical):
    2D grid of qubits: data qubits + ancilla qubits.
    Stabilizers: 4-qubit plaquette (XXXX) and vertex (ZZZZ) operators.
    Threshold: ~1% (highest known for local 2D architecture).
    Distance d: code uses ~d^2 physical per logical qubit.
    For 10^-10 logical error rate at p=0.1% physical:
    d~17: ~289 physical per logical.
    Full fault-tolerant computer: ~1000-10000 physical per logical.
    Current (2024): IBM/Google have ~1000 physical qubits total.
    Fault-tolerant compute: still ~10+ years away.

  FAULT TOLERANCE THRESHOLD THEOREM:
    If each gate error rate p < p_threshold,
    then by concatenating error correcting codes L levels deep:
    Logical error rate = p_threshold * (p/p_threshold)^(2^L)
    -> exponentially suppressed as L increases.
    p_threshold ~ 1% for surface code.
    Each level of concatenation costs O(poly(n)) overhead.

  OPTICAL QEC / ROGUEGUARD CONNECTION:
    Optical qubits: photon polarization (H/V), path (which fiber), time-bin.
    Errors: photon loss (dominant), dark counts, mode mismatch.
    GKP code (Gottesman-Kitaev-Preskill): encodes qubit in oscillator.
    Displacement error ~ noise in phase/amplitude -> GS retrieves this!
    RogueGuard phase retrieval = detecting displacement errors in optical field.
    H(nu) = exp(i*pi*D*nu^2) coherent state displacement in freq. domain.
""")

# Simulate 3-qubit bit-flip code
print("  NUMERICAL: 3-QUBIT BIT-FLIP CODE SIMULATION")
np.random.seed(42)
n_trials = 100000
p_flip = 0.05   # 5% single-qubit error rate

successes_raw    = 0   # no code, single qubit
successes_coded  = 0   # 3-qubit code

for _ in range(n_trials):
    # Raw qubit: flip with probability p
    err = np.random.random() < p_flip
    if not err:
        successes_raw += 1

    # 3-qubit code: |0> -> |000>
    # Each qubit flips independently
    flips = np.random.random(3) < p_flip
    # Syndrome: majority vote
    n_flips = np.sum(flips)
    # If 0 or 1 flips: correctable (majority is original)
    # If 2 or 3 flips: uncorrectable (majority is wrong)
    if n_flips <= 1:
        successes_coded += 1

p_fail_raw   = 1 - successes_raw/n_trials
p_fail_coded = 1 - successes_coded/n_trials
p_fail_theory = 3*p_flip**2*(1-p_flip) + p_flip**3  # P(2 or 3 errors)

print(f"    p_flip per qubit = {p_flip:.2%}")
print(f"    Raw qubit failure rate:   {p_fail_raw:.4%}  (= p = {p_flip:.2%})")
print(f"    3-qubit code failure:     {p_fail_coded:.4%}  (simulated)")
print(f"    3-qubit code theory:      {p_fail_theory:.4%}  (= 3p^2 + p^3 for p={p_flip})")
print(f"    Improvement factor:       {p_fail_raw/p_fail_coded:.1f}x")
print(f"    At p=1%: coded failure = {3*(0.01)**2*(0.99) + (0.01)**3:.6%}")
print(f"    Code HELPS when p < 1/2 (threshold for this simple code)")

# Threshold: code works when p_coded < p_raw
# 3p^2(1-p) + p^3 < p  =>  3p - 2p^2 < 1 for p << 1  => p < 0.5
print(f"\n  3-qubit code threshold: p_flip < 50% (trivial)")
print(f"  Surface code threshold: p_flip < ~1%  (non-trivial)")
print(f"  Concatenated codes:    p_threshold ~ 10^-4 to 10^-2 depending on model")

# ============================================================
# MATPLOTLIB -- 6-PANEL FIGURE
# ============================================================
print(f"\n{SEP}")
print("BUILDING FIGURE...")
print(SEP)

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#F8F8F0")
gs0 = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38,
                        top=0.93, bottom=0.05, left=0.06, right=0.97)

ax_pow   = fig.add_subplot(gs0[0, 0])
ax_vec   = fig.add_subplot(gs0[0, 1])
ax_bess  = fig.add_subplot(gs0[0, 2])
ax_delta = fig.add_subplot(gs0[1, 0])
ax_dist  = fig.add_subplot(gs0[1, 1])
ax_qec   = fig.add_subplot(gs0[1, 2])
ax_expint= fig.add_subplot(gs0[2, 0])
ax_proof = fig.add_subplot(gs0[2, 1])
ax_surf  = fig.add_subplot(gs0[2, 2])

fig.suptitle("Vector Calculus + Bessel + Distributions + Dirac Delta Proofs + QEC",
             fontsize=13, fontweight="bold", color="#1a1a2e")

xp = np.linspace(-3, 3, 500)
xp_pos = np.linspace(0.01, 3, 500)

# ---- AX_POW: power functions ----
ax = ax_pow
ax.set_facecolor("#F0F4FF")
ax.plot(xp, xp**2,          "#1f77b4", lw=1.8, label=r"$x^2$")
ax.plot(xp, xp**3,          "#ff7f0e", lw=1.8, label=r"$x^3$")
ax.plot(xp_pos, 1/xp_pos,   "#2ca02c", lw=1.8, label=r"$x^{-1}$")
ax.plot(xp_pos, np.log(xp_pos), "#d62728", lw=1.8, label=r"$\ln x$ (antideriv of $x^{-1}$)")
ax.set_ylim(-5, 9)
ax.set_xlim(-3, 3)
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.legend(fontsize=8)
ax.set_title(r"Power Functions: $x^2, x^3, x^{-1}, \ln x$", fontsize=10)
ax.grid(alpha=0.2)

# ---- AX_VEC: vector field curl visualization ----
ax = ax_vec
ax.set_facecolor("#F0FFF0")
X, Y = np.meshgrid(np.linspace(-2,2,15), np.linspace(-2,2,15))
# F = (-y, x, 0) -- pure rotation, curl = (0,0,2)
Fx_v = -Y
Fy_v =  X
ax.quiver(X, Y, Fx_v, Fy_v, color="#1f77b4", alpha=0.7, scale=25)
# Also show grad of x^2+y^2
Gx = 2*X
Gy = 2*Y
ax.quiver(X[::3,::3], Y[::3,::3], Gx[::3,::3], Gy[::3,::3],
          color="#d62728", alpha=0.5, scale=60, label="grad(x^2+y^2)")
ax.set_title("Vector Fields: curl(-y,x)=(0,0,2)\ngrad(x^2+y^2)=(2x,2y)", fontsize=9)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect("equal")
ax.grid(alpha=0.15)
ax.text(0.02, 0.98, "Blue: rotation\nRed: gradient",
        transform=ax.transAxes, fontsize=8, va="top")

# ---- AX_BESS: Bessel functions ----
ax = ax_bess
ax.set_facecolor("#FFF5F0")
xb = np.linspace(0, 15, 500)
for m, col in [(0,"#1f77b4"),(1,"#ff7f0e"),(2,"#2ca02c"),(3,"#d62728")]:
    ax.plot(xb, sc.jv(m, xb), color=col, lw=1.5, label=f"$J_{m}(x)$")
ax.axhline(0, color="k", lw=0.5)
# Mark zeros of J_0
for z in zeros_J0:
    ax.axvline(z, color="#1f77b4", lw=0.7, ls=":", alpha=0.6)
ax.set_ylim(-0.5, 1.05)
ax.set_title("Bessel Functions $J_m(x)$\n(LP fiber modes)", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.2)
ax.text(0.02, 0.05, f"J_0 zeros: {zeros_J0[0]:.2f}, {zeros_J0[1]:.2f}, ...",
        transform=ax.transAxes, fontsize=7.5,
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ---- AX_DELTA: Gaussian delta reps ----
ax = ax_delta
ax.set_facecolor("#FFF0F0")
xd = np.linspace(-1, 1, 2000)
for eps, col, lbl in [(0.3,"#aac8e8","eps=0.3"), (0.1,"#4a90d9","eps=0.1"),
                      (0.03,"#1a3a8c","eps=0.03")]:
    d = np.exp(-xd**2/eps**2) / (eps*np.sqrt(np.pi))
    ax.plot(xd, d, color=col, lw=1.5, label=lbl)
ax.set_xlim(-0.8, 0.8)
ax.set_title(r"$\delta_\epsilon(x) \to \delta(x)$ as $\epsilon \to 0$", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.set_xlabel("x"); ax.set_ylabel(r"$\delta_\epsilon(x)$", fontsize=9)

# ---- AX_DIST: distribution properties text ----
ax = ax_dist
ax.set_facecolor("#F8F0FF")
ax.axis("off")
ax.set_title("Distribution Theory Summary", fontsize=10)
dist_lines = [
    ("Test function space D:", "#222", True),
    ("  C_inf functions, compact support", "#333", False),
    ("  phi in D => phi in S (Schwartz)", "#333", False),
    ("", "", False),
    ("Distribution T: D -> R linear,", "#1a5c9c", True),
    ("  continuous (phi_n->phi => T[phi_n]->T[phi])", "#1a5c9c", False),
    ("", "", False),
    ("DELTA as distribution:", "#8c1a1a", True),
    ("  <delta, phi> = phi(0)  -- linear? YES", "#8c1a1a", False),
    ("  -- continuous? YES (pointwise conv)", "#8c1a1a", False),
    ("", "", False),
    ("Derivative rule:", "#2c7c2c", True),
    ("  <T', phi> = -<T, phi'>  (IBP)", "#2c7c2c", False),
    ("  => every distr. is C_inf!", "#2c7c2c", False),
    ("  <delta', phi> = -phi'(0)", "#2c7c2c", False),
    ("", "", False),
    ("FT on S' (tempered distr.):", "#9467bd", True),
    ("  <FT[T], phi> = <T, FT[phi]>", "#9467bd", False),
    ("  FT[delta] = 1  (constant)", "#9467bd", False),
    ("  FT[1] = 2*pi*delta(k)", "#9467bd", False),
]
for k, (line, color, bold) in enumerate(dist_lines):
    if not line:
        continue
    ax.text(0.03, 0.97 - k*0.047, line,
            transform=ax.transAxes, fontsize=7.8,
            color=color if color else "#333",
            va="top", fontweight="bold" if bold else "normal",
            fontfamily="monospace")

# ---- AX_QEC: QEC code comparison ----
ax = ax_qec
ax.set_facecolor("#F0F8FF")
ax.axis("off")
ax.set_title("Quantum Error Correcting Codes", fontsize=10)

codes = [
    ("Code",         "[[n,k,d]]", "Corrects",       "Threshold"),
    ("----",         "---------", "--------",       "---------"),
    ("Shor",         "[[9,1,3]]", "1 qubit (X,Y,Z)","50% (naive)"),
    ("Steane",       "[[7,1,3]]", "1 qubit (X,Y,Z)","~1% concat."),
    ("5-qubit",      "[[5,1,3]]", "1 qubit (min n)","~1% concat."),
    ("Surface d=3",  "[[9,1,3]]", "1 X or Z",       "~1%"),
    ("Surface d=5",  "[[25,1,5]]","2 X or Z",       "~1%"),
    ("Surface d=17", "[[289,1,17]]","8 X or Z",     "~1%"),
    ("Cat code",     "[[inf,1,inf]]","loss errors", "prob. dep."),
    ("GKP",          "oscillator","displacement",   "~10%"),
]
for j, (name, params, corr, thresh) in enumerate(codes):
    y = 0.97 - j*0.09
    color = "#1a1a4e" if j > 1 else "#444"
    ax.text(0.01, y, name,   fontsize=7.5, va="top", color=color,
            transform=ax.transAxes, fontweight="bold" if j <= 1 else "normal")
    ax.text(0.28, y, params, fontsize=7.5, va="top", color="#1a5c9c",
            transform=ax.transAxes, fontfamily="monospace")
    ax.text(0.52, y, corr,   fontsize=7.5, va="top", color="#2c7c2c",
            transform=ax.transAxes)
    ax.text(0.80, y, thresh, fontsize=7.5, va="top", color="#8c1a1a",
            transform=ax.transAxes)

# ---- AX_EXPINT: exponential integrals ----
ax = ax_expint
ax.set_facecolor("#FFFFF0")
xe = np.linspace(-2, 2, 400)
ax.plot(xe, np.exp(xe),         "#1f77b4", lw=1.8, label=r"$e^x$")
ax.plot(xe, xe*np.exp(xe),      "#ff7f0e", lw=1.8, label=r"$xe^x$")
ax.plot(xe, xe**2*np.exp(xe),   "#2ca02c", lw=1.8, label=r"$x^2 e^x$")
ax.plot(xe, np.exp(-xe**2),     "#d62728", lw=1.8, label=r"$e^{-x^2}$")
ax.set_ylim(-3, 8)
ax.axhline(0, color="k", lw=0.5)
ax.legend(fontsize=8)
ax.set_title("Exponential Functions\n(IBP raises degree by 1)", fontsize=10)
ax.grid(alpha=0.2)
ax.text(0.02, 0.97,
        r"$\int e^{ax}dx = e^{ax}/a$" + "\n" +
        r"$\int xe^x dx = e^x(x-1)$" + "\n" +
        r"$\int_{-\infty}^{\infty} e^{-x^2}dx = \sqrt{\pi}$",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ---- AX_PROOF: delta sifting numerical ----
ax = ax_proof
ax.set_facecolor("#F0FFF8")
xpr = np.linspace(-3, 3, 5000)
f_test = np.cos(xpr)   # test function

for eps, col, lbl in [(0.5,"#aac8e8","eps=0.5"),
                      (0.15,"#4a90d9","eps=0.15"),
                      (0.04,"#1a3a8c","eps=0.04")]:
    d_approx = np.exp(-xpr**2/eps**2) / (eps*np.sqrt(np.pi))
    integrand = d_approx * f_test
    ax.plot(xpr, integrand, color=col, lw=1.3, label=f"delta_eps*cos, {lbl}")
    result = np.trapezoid(integrand, xpr)
    ax.text(1.5, np.max(integrand)*0.9,
            f"{lbl}: INT={result:.4f}", fontsize=7, color=col)

ax.plot(xpr, f_test, "k--", lw=1.0, alpha=0.4, label="cos(x)")
ax.axvline(0, color="#d62728", lw=1.0)
ax.set_xlim(-2, 2.5)
ax.set_title(r"Sifting Proof: $\int\delta_\epsilon(x)\cos(x)dx \to \cos(0)=1$",
             fontsize=9)
ax.legend(fontsize=7, loc="upper right")
ax.grid(alpha=0.2)

# ---- AX_SURF: QEC error rate vs code distance ----
ax = ax_surf
ax.set_facecolor("#FFF0FF")
p_phys = np.logspace(-4, -1, 100)
p_threshold = 0.01

for d, col in [(3,"#1f77b4"),(5,"#ff7f0e"),(7,"#2ca02c"),(11,"#d62728")]:
    # Surface code logical error rate approximation
    p_L = 0.1 * (p_phys/p_threshold)**((d+1)//2)
    ax.loglog(p_phys, p_L, color=col, lw=1.8, label=f"d={d}")

ax.loglog(p_phys, p_phys, "k--", lw=1.2, label="No code (p_L=p)")
ax.axvline(p_threshold, color="#aaa", ls=":", lw=1.5)
ax.text(p_threshold*1.1, 1e-2, "threshold\n~1%", fontsize=8, color="#888")
ax.set_xlabel("Physical error rate p", fontsize=9)
ax.set_ylabel("Logical error rate p_L", fontsize=9)
ax.set_title("Surface Code: Logical Error\nvs Distance d", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3, which="both")
ax.set_xlim(1e-4, 1e-1)
ax.set_ylim(1e-15, 1)

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {OUT}")

print(f"\n{SEP}")
print("Done.")
print(SEP)
