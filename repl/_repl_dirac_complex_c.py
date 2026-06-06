"""
_repl_dirac_complex_c.py
========================
Dirac delta + complex exponential + C language deep reference.

S1: Complex exponential e^(ix)
    - Euler's formula, Taylor series proof, phasors
    - e^(i*pi) + 1 = 0 verified
    - SymPy: Re/Im, abs, arg, de Moivre, roots of unity
    - Connection to Fourier transform kernel

S2: Dirac delta function
    - Definition by action on test functions
    - 5 representations (Gaussian, Lorentzian, sinc^2, box, Fourier)
    - Key properties (sifting, scaling, derivative, composition)
    - SymPy DiracDelta: integrals, derivatives
    - matplotlib: 4 representations converging to delta

S3: Modern physics connections
    - QM: position eigenstate <x'|x> = delta(x-x')
    - Completeness: INT |x><x| dx = I
    - Green's functions: L*G(x,x') = delta(x-x')
    - Propagators: K(x,t; x0,0) = <x|e^(-iHt/hbar)|x0>
    - RogueGuard: H(nu) = e^(i*pi*D*nu^2) is a chirp -- complex exp kernel

S4: C operators (complete precedence table)
    - All 15 precedence levels
    - Arithmetic, relational, logical, bitwise, pointer, assignment, etc.
    - sizeof, cast, ternary, comma

S5: C control flow
    - if / else if / else
    - switch / case / default / fall-through
    - for / while / do-while
    - break / continue / goto / return
    - Code examples for each

S6: C standard library
    - stdio.h, stdlib.h, string.h, math.h, time.h, assert.h, errno.h
    - Key functions with signatures and usage notes

Output: repl/_out_dirac_complex_c.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sympy as sp
from sympy import (symbols, exp, I, pi, oo, cos, sin, sqrt, Rational,
                   re, im, Abs, arg, simplify, integrate, diff,
                   DiracDelta, Heaviside, series, factorial, conjugate,
                   fourier_transform, inverse_fourier_transform)
import os

OUT = os.path.join(os.path.dirname(__file__), "_out_dirac_complex_c.png")
SEP = "=" * 65

# ============================================================
# S1: COMPLEX EXPONENTIAL
# ============================================================
print(SEP)
print("SECTION 1: COMPLEX EXPONENTIAL  e^(ix)")
print(SEP)

x_s, t_s, omega_s = symbols("x t omega", real=True)
z_s = symbols("z")
a_s, b_s = symbols("a b", real=True)

print("""
  EULER'S FORMULA:
    e^(ix) = cos(x) + i*sin(x)

  PROOF VIA TAYLOR SERIES:
    e^(ix) = 1 + ix + (ix)^2/2! + (ix)^3/3! + (ix)^4/4! + ...
           = 1 + ix - x^2/2! - ix^3/3! + x^4/4! + ix^5/5! - ...
    Real part:  1 - x^2/2! + x^4/4! - ... = cos(x)
    Imag part:  x - x^3/3! + x^5/5! - ... = sin(x)
    => e^(ix) = cos(x) + i*sin(x)  QED
""")

# SymPy Taylor series
eix = exp(I*x_s)
series_eix = series(eix, x_s, 0, 10)
print(f"  SymPy Taylor e^(ix) to order 9:")
print(f"    {series_eix}")

# Euler's identity
euler_check = sp.N(exp(I*pi) + 1)
print(f"\n  Euler's identity: e^(i*pi) + 1 = {euler_check}  (= 0 exactly)")

print("""
  RECTANGULAR <-> POLAR FORM:
    z = a + ib = r * e^(i*phi)
    r   = |z| = sqrt(a^2 + b^2)      [modulus]
    phi = arg(z) = atan2(b, a)        [argument, angle]
    Re(z) = r*cos(phi)
    Im(z) = r*sin(phi)
    z*    = a - ib = r*e^(-i*phi)     [complex conjugate]
    z*z   = |z|^2 = r^2               [always real, non-negative]
""")

# SymPy verify
z_ex = 3 + 4*I
print(f"  Example z = 3 + 4i:")
print(f"    |z|   = {Abs(z_ex)}  (= 5)")
print(f"    arg(z)= {sp.atan2(4,3).evalf():.6f} rad = {float(sp.atan2(4,3)*180/pi):.2f} deg")
print(f"    z*    = {conjugate(z_ex)}")
print(f"    z*z   = {simplify(z_ex * conjugate(z_ex))}  (= |z|^2 = 25)")

print("""
  DE MOIVRE'S THEOREM:
    (e^(i*theta))^n = e^(i*n*theta)
    (cos(theta) + i*sin(theta))^n = cos(n*theta) + i*sin(n*theta)

  N-TH ROOTS OF UNITY:
    z^n = 1  =>  z_k = e^(2*pi*i*k/n),  k = 0, 1, ..., n-1
    These are n equally-spaced points on the unit circle.
    Sum of all roots = 0  (centroid = origin)
    Product of all roots = (-1)^(n+1)
""")

# Roots of unity
n_roots = 6
roots = [np.exp(2j*np.pi*k/n_roots) for k in range(n_roots)]
print(f"  6th roots of unity (z^6 = 1):")
for k, r in enumerate(roots):
    print(f"    k={k}: e^(2pi*i*{k}/6) = {r.real:+.4f} + {r.imag:+.4f}i  "
          f"|z|={abs(r):.4f}")
print(f"  Sum = {sum(roots).real:+.6f} + {sum(roots).imag:+.6f}i  (= 0)")

print("""
  PHASORS AND OSCILLATIONS:
    Physical signal: A*cos(omega*t + phi)
    Complex phasor:  A*e^(i*(omega*t + phi))
    Take Re part to get physical signal.
    Advantage: multiplication replaces trig identities.
    Two signals: A*e^(i*omega*t) * B*e^(i*phi) = AB*e^(i*(omega*t + phi))
    => amplitude multiplies, phase adds.

  FOURIER TRANSFORM KERNEL:
    F(omega) = INT f(t) * e^(-i*omega*t) dt
    Kernel: e^(-i*omega*t) = cos(omega*t) - i*sin(omega*t)
    At each omega: project f(t) onto that frequency's phasor.
    F(omega) = amplitude and phase of frequency omega in f(t).

  ROGUEGUARD / DISPERSION CONNECTION:
    Transfer function: H(nu) = e^(i*pi*D*nu^2)
    This IS a complex exponential with quadratic phase.
    |H(nu)| = 1 for all nu  (all-pass: no amplitude change)
    arg H(nu) = pi*D*nu^2   (phase shifts quadratically with freq)
    => Chirp: different frequencies arrive at different times.
    => Time-stretch: maps frequency axis to time axis (Jalali/STEAM).
    GS algorithm: recovers arg(H*F(nu)) = pi*D*nu^2 + phi_signal(nu)
""")

# ============================================================
# S2: DIRAC DELTA FUNCTION
# ============================================================
print(f"\n{SEP}")
print("SECTION 2: DIRAC DELTA FUNCTION")
print(SEP)

print("""
  DEFINITION (distributional):
    delta(x) is NOT a function -- it is a distribution (generalized function).
    Defined by its ACTION on smooth test functions f(x):

    INT_{-inf}^{inf}  f(x) * delta(x - a)  dx  =  f(a)

    This is the SIFTING property: delta "sifts out" f at the point x=a.
    Informally: delta(x) = 0 for x != 0,  delta(0) = +inf,
                INT delta(x) dx = 1  (unit area, but infinite height)

  REPRESENTATIONS (limits of ordinary functions):

  1. GAUSSIAN:
     delta(x) = lim_{eps->0} (1/(eps*sqrt(pi))) * exp(-x^2/eps^2)
     width~eps, height~1/eps, area=1 always.

  2. LORENTZIAN (Cauchy):
     delta(x) = lim_{eps->0} (1/pi) * eps / (x^2 + eps^2)
     Natural linewidth in spectroscopy is Lorentzian (energy broadening).

  3. SINC^2:
     delta(x) = lim_{N->inf} N * sinc^2(N*x) / pi
     where sinc(x) = sin(x)/x

  4. BOX (rectangle):
     delta(x) = lim_{eps->0} (1/(2*eps)) * rect(x/(2*eps))
     Simplest: box of width 2*eps, height 1/(2*eps).

  5. FOURIER REPRESENTATION:
     delta(x) = (1/(2*pi)) * INT_{-inf}^{inf} e^(i*k*x) dk
     = (1/(2*pi)) * INT e^(ikx) dk
     This is the MOST IMPORTANT representation.
     Says: delta is the function with FLAT Fourier spectrum (all frequencies equal).
     Proof: FT[delta(x)](k) = INT delta(x)*e^(-ikx) dx = e^(-ik*0) = 1
            -> inverse FT of 1 = delta(x).
""")

# SymPy DiracDelta
x_d = symbols("x", real=True)
a_d = symbols("a", real=True)

print("  SYMPY DiracDelta:")
# Sifting
expr1 = integrate(sp.Function("f")(x_d) * DiracDelta(x_d - a_d),
                  (x_d, -oo, oo))
print(f"    INT f(x)*delta(x-a) dx = {expr1}  (sifting)")

expr2 = integrate((x_d**2 + 1) * DiracDelta(x_d - 3), (x_d, -oo, oo))
print(f"    INT (x^2+1)*delta(x-3) dx = {expr2}  (= 3^2+1 = 10)")

expr3 = integrate(sp.exp(x_d) * DiracDelta(x_d), (x_d, -oo, oo))
print(f"    INT exp(x)*delta(x) dx = {expr3}  (= exp(0) = 1)")

print("""
  KEY PROPERTIES:
    1. SIFTING:          INT f(x)*delta(x-a) dx = f(a)
    2. NORMALIZATION:    INT delta(x) dx = 1
    3. SYMMETRY:         delta(-x) = delta(x)         [even function]
    4. SCALING:          delta(a*x) = delta(x)/|a|
    5. COMPOSITION:      delta(g(x)) = SUM_i delta(x-x_i)/|g'(x_i)|
                         where x_i are roots of g(x) = 0
    6. DERIVATIVE:       INT f(x)*delta'(x-a) dx = -f'(a)
                         delta'(x) = -delta'(-x)  [odd]
    7. x*delta(x) = 0    [multiplication by zero factor]
    8. HEAVISIDE:        d/dx Theta(x) = delta(x)
                         Theta(x) = INT_{-inf}^{x} delta(t) dt
""")

# SymPy: scaling property
print("  SymPy scaling: delta(a*x) = delta(x)/|a|")
# INT delta(3*x) dx = 1/3
expr4 = integrate(DiracDelta(3*x_d), (x_d, -oo, oo))
print(f"    INT delta(3*x) dx = {expr4}  (= 1/3)")

# Derivative
expr5 = integrate(x_d**3 * DiracDelta(x_d - 2, 1), (x_d, -oo, oo))
print(f"    INT x^3 * delta'(x-2) dx = {expr5}  (= -(d/dx x^3)|_{{x=2}} = -12)")

# Composition
print("  Composition example: delta(x^2 - a^2) = [delta(x-a) + delta(x+a)] / (2|a|)")
expr6 = integrate(DiracDelta(x_d**2 - 4), (x_d, -oo, oo))
print(f"    INT delta(x^2 - 4) dx = {expr6}  (= 1/(2*2) + 1/(2*2) = 0.5)")

# Heaviside connection
print("\n  Heaviside step function: d/dx Theta(x) = delta(x)")
print(f"    SymPy: diff(Heaviside(x), x) = {diff(Heaviside(x_d), x_d)}")

# ============================================================
# S3: MODERN PHYSICS CONNECTIONS
# ============================================================
print(f"\n{SEP}")
print("SECTION 3: MODERN PHYSICS CONNECTIONS")
print(SEP)

print("""
  QM: POSITION EIGENSTATES
    In continuous position basis:
    x_hat |x'> = x' |x'>           (position eigenvalue equation)
    <x|x'>     = delta(x - x')     (orthogonality -- continuous version)
    |psi> = INT psi(x) |x> dx      (any state as superposition)
    psi(x) = <x|psi>               (wavefunction = overlap with |x>)

    COMPLETENESS RELATION:
    INT |x><x| dx = I_hat           (identity operator)
    Proof: (INT |x><x| dx)|psi> = INT |x> <x|psi> dx
                                 = INT psi(x) |x> dx = |psi>  QED
    This is the RESOLUTION OF IDENTITY -- key tool in QM.

  CONNECTION TO FOURIER TRANSFORM:
    Momentum eigenstates: <x|p> = (1/sqrt(2*pi*hbar)) * e^(i*p*x/hbar)
    Momentum wavefunction: phi(p) = <p|psi>
    Completeness: INT |p><p| dp = I
    Switching bases:
      psi(x) = <x|psi> = INT <x|p><p|psi> dp
             = INT (1/sqrt(2*pi*hbar)) e^(ipx/hbar) phi(p) dp
    => psi(x) and phi(p) are FOURIER TRANSFORM PAIRS.
    Uncertainty: delta_x * delta_p >= hbar/2  is the FT bandwidth theorem.

  GREEN'S FUNCTIONS:
    For differential operator L (e.g., L = -d^2/dx^2 + k^2):
    L * G(x, x') = delta(x - x')    [Green's function equation]
    Then: solution to L*u = f is
    u(x) = INT G(x, x') * f(x') dx'
    G is the impulse response: response at x due to unit source at x'.

    EXAMPLE: Free-space 1D Green's function
    (-d^2/dx^2 - k^2) G(x,x') = delta(x-x')
    Solution: G(x,x') = (i/2k) * e^(ik|x-x'|)
    (Outgoing wave from point source -- used in scattering theory)

    FEM CONNECTION (from _repl_imaging_dew.py):
    Stiffness matrix K comes from discretizing L.
    K*u = f is the discrete version of L*u = f.
    K_ij = INT phi_i' * phi_j' dx  (basis function overlaps)

  PROPAGATOR (FEYNMAN PATH INTEGRAL):
    K(x_f, t_f; x_i, t_i) = <x_f | e^(-iHt/hbar) | x_i>
    = amplitude for particle to go from x_i at t_i to x_f at t_f
    At t=0: K(x_f, 0; x_i, 0) = <x_f|x_i> = delta(x_f - x_i)
    For free particle (H = p^2/2m):
    K = sqrt(m/(2*pi*i*hbar*t)) * exp(i*m*(x_f-x_i)^2 / (2*hbar*t))
    This is a CHIRPED GAUSSIAN -- same form as our H(nu)!

  OPTICAL / ROGUEGUARD CONNECTION:
    H(nu) = exp(i*pi*D*nu^2)         [dispersion transfer function]
    Compare free-particle propagator:
    K(x,t) ~ exp(i*m*x^2/(2*hbar*t))
    Both are QUADRATIC PHASE exponentials (chirps).
    D in optics <-> t/(2m*hbar) in QM.
    Dispersion = quantum-mechanical free propagation in frequency space.
    GS phase recovery = measuring the wavefunction of the optical field.
    This is why the math is identical -- same underlying Hilbert space structure.
""")

# Numerical: free-particle propagator as chirp
t_vals = [0.01, 0.1, 1.0]
x_arr  = np.linspace(-5, 5, 1000)
m_qm, hb = 1.0, 1.0   # natural units

print("  Free-particle propagator |K(x,t)| (natural units m=hbar=1):")
print(f"  {'t':<8} {'|K| (const)':<15} {'phase at x=1 (rad)'}")
for t in t_vals:
    K_amp   = np.sqrt(m_qm / (2*np.pi*hb*t))
    phase_1 = m_qm * 1.0**2 / (2*hb*t)
    print(f"  {t:<8.2f} {K_amp:<15.6f} {phase_1:.4f}")

print("\n  => As t->0: amplitude->inf, phase->inf: K->delta(x)")
print("     This is why the free propagator IS the Dirac delta at t=0.")

# ============================================================
# S4: C OPERATORS (COMPLETE PRECEDENCE TABLE)
# ============================================================
print(f"\n{SEP}")
print("SECTION 4: C OPERATORS -- COMPLETE PRECEDENCE TABLE")
print(SEP)

print("""
  C has 15 precedence levels (1=highest, 15=lowest).
  Associativity: L=left-to-right, R=right-to-left.

  Prec  Assoc  Operators            Category
  ----  -----  -------------------  --------------------------""")

prec_table = [
    (1,  "L", "()  []  .  ->  i++  i--",       "Postfix: call, subscript, member, post-incr"),
    (2,  "R", "++i  --i  +  -  !  ~  (T)  *  &  sizeof",
                                                "Prefix: unary, cast, deref, addr-of, size"),
    (3,  "L", "*  /  %",                        "Multiplicative"),
    (4,  "L", "+  -",                            "Additive"),
    (5,  "L", "<<  >>",                          "Bitwise shift"),
    (6,  "L", "<  <=  >  >=",                    "Relational"),
    (7,  "L", "==  !=",                          "Equality"),
    (8,  "L", "&",                               "Bitwise AND"),
    (9,  "L", "^",                               "Bitwise XOR"),
    (10, "L", "|",                               "Bitwise OR"),
    (11, "L", "&&",                              "Logical AND (short-circuit)"),
    (12, "L", "||",                              "Logical OR  (short-circuit)"),
    (13, "R", "?:",                              "Ternary conditional"),
    (14, "R", "=  +=  -=  *=  /=  %=  &=  |=  ^=  <<=  >>=",
                                                "Assignment (all compound forms)"),
    (15, "L", ",",                               "Comma (sequence, evaluate both)"),
]

for p, assoc, ops, cat in prec_table:
    print(f"  {p:>4}  {assoc:<5}  {ops:<36} {cat}")

print("""
  COMMON TRAPS:
    a & b == c       parsed as  a & (b == c)   [== beats &]
    a || b && c      parsed as  a || (b && c)   [&& beats ||]
    *p++             parsed as  *(p++)           [postfix ++ beats *]
    sizeof x + 1     parsed as  (sizeof x) + 1  [sizeof beats +]
    a = b = c = 0    R-assoc:   a = (b = (c=0)) [ok, all get 0]
    f(a,b)           comma in arg list != comma operator!

  BITWISE vs LOGICAL:
    &   bitwise AND  -- operates on every bit
    &&  logical AND  -- short-circuits: if left is false, skip right
    |   bitwise OR
    ||  logical OR   -- short-circuits
    ~   bitwise NOT  (unary, flips all bits)
    !   logical NOT  (unary, returns 0 or 1)

  POINTER OPERATORS:
    *ptr             dereference: value at address ptr
    &var             address-of: pointer to var
    ptr->member      dereference + member: (*ptr).member
    arr[i]           subscript:  *(arr + i)  (always)

  SIZEOF:
    sizeof(int)      size in bytes of type (compile-time constant)
    sizeof expr      size of expression's type (no evaluation of expr!)
    sizeof(arr)/sizeof(arr[0])  = number of elements in array
""")

# ============================================================
# S5: C CONTROL FLOW
# ============================================================
print(f"\n{SEP}")
print("SECTION 5: C CONTROL FLOW")
print(SEP)

c_code = r"""
  /* ---- if / else if / else ---- */
  if (x > 0) {
      printf("positive\n");
  } else if (x < 0) {
      printf("negative\n");
  } else {
      printf("zero\n");
  }
  /* Dangling else: binds to nearest if -- always use braces */

  /* ---- switch / case / default ---- */
  switch (c) {
      case 'a': case 'e': case 'i': case 'o': case 'u':
          printf("vowel\n");
          break;            /* WITHOUT break: FALLS THROUGH to next case */
      case '\n':
          ++lines;
          break;
      default:
          ++others;
          break;            /* good practice: break on default too */
  }
  /* switch expression must be integral (int, char, enum). NOT float, NOT string. */

  /* ---- for loop ---- */
  for (int i = 0; i < n; i++) {
      if (arr[i] < 0) continue;   /* skip negatives */
      sum += arr[i];
  }
  /* All 3 clauses optional: for(;;) = infinite loop */

  /* ---- while loop ---- */
  while (fgets(line, sizeof(line), fp) != NULL) {
      process(line);
  }
  /* Condition checked BEFORE body. May execute 0 times. */

  /* ---- do-while loop ---- */
  do {
      printf("Enter 1-10: ");
      scanf("%d", &n);
  } while (n < 1 || n > 10);
  /* Body executes AT LEAST ONCE. Condition checked AFTER body. */

  /* ---- break ---- */
  for (int i = 0; i < n; i++) {
      if (arr[i] == target) {
          found = i;
          break;   /* exits innermost loop only */
      }
  }

  /* ---- continue ---- */
  for (int i = 0; i < n; i++) {
      if (arr[i] < 0) continue;    /* jump to i++ */
      total += arr[i];
  }

  /* ---- goto ---- */
  /* LEGITIMATE USE: error cleanup in deep nesting */
  if (!open_file())   goto cleanup;
  if (!alloc_mem())   goto cleanup;
  if (!read_data())   goto cleanup;
  process();
  result = 0;           /* success */
cleanup:
  free(mem);
  fclose(fp);
  return result;
  /* goto cannot jump INTO a block or past a declaration with initializer */

  /* ---- return ---- */
  int max(int a, int b) {
      return (a > b) ? a : b;    /* ternary + return */
  }
  void print_arr(int *a, int n) {
      if (!a || n <= 0) return;  /* early return (void function) */
      for (int i=0; i<n; i++) printf("%d ", a[i]);
  }
"""
print(c_code)

print("""
  CONTROL FLOW SUMMARY:
  Statement   Checks cond   Body executes     Exit early
  ----------  -----------   ----------------  ----------
  if          before        0 or 1 times      N/A
  while       before body   0 or more times   break
  for         before body   0 or more times   break
  do-while    after body    1 or more times   break
  switch      once          matching case     break (per case)
  goto        N/A           jumps directly    N/A
  return      N/A           exits function    N/A

  TERNARY:  cond ? expr_true : expr_false
    value = (x > 0) ? x : -x;   /* abs value */
    /* Not a statement -- it's an EXPRESSION. Has a value. */
""")

# ============================================================
# S6: C STANDARD LIBRARY
# ============================================================
print(f"\n{SEP}")
print("SECTION 6: C STANDARD LIBRARY")
print(SEP)

stdlib_table = {
    "<stdio.h>  -- I/O": [
        ("printf(fmt, ...)",      "Formatted output to stdout"),
        ("scanf(fmt, ...)",       "Formatted input from stdin"),
        ("fprintf(fp, fmt, ...)", "Formatted output to file"),
        ("fscanf(fp, fmt, ...)",  "Formatted input from file"),
        ("fopen(path, mode)",     "Open file; mode: r,w,a,rb,wb"),
        ("fclose(fp)",            "Close file, flush buffers"),
        ("fgets(buf, n, fp)",     "Read line; safe (limits n)"),
        ("fputs(str, fp)",        "Write string to file"),
        ("fread(ptr,size,n,fp)",  "Binary read n items of size bytes"),
        ("fwrite(ptr,size,n,fp)", "Binary write"),
        ("fseek(fp, off, whence)","Seek: SEEK_SET/CUR/END"),
        ("ftell(fp)",             "Current file position"),
        ("sprintf(buf, fmt, ...)", "Formatted write to string"),
        ("snprintf(buf,n,fmt,...)", "Safe: limits to n bytes"),
        ("perror(str)",           "Print str + errno description"),
        ("EOF",                   "End-of-file sentinel (-1)"),
    ],
    "<stdlib.h> -- Memory, process, conversion": [
        ("malloc(size)",          "Allocate size bytes, uninit; returns NULL on fail"),
        ("calloc(n, size)",       "Allocate n*size bytes, zero-initialized"),
        ("realloc(ptr, size)",    "Resize allocation"),
        ("free(ptr)",             "Release allocation; free(NULL) is safe no-op"),
        ("exit(status)",          "Exit process; runs atexit(), flushes stdio"),
        ("abort()",               "Abnormal exit; raises SIGABRT, no cleanup"),
        ("atexit(fn)",            "Register fn() to call at normal exit"),
        ("atoi(str)",             "String to int (no error check -- prefer strtol)"),
        ("atof(str)",             "String to double (prefer strtod)"),
        ("strtol(str,end,base)",  "String to long, detects errors via end/errno"),
        ("strtod(str, end)",      "String to double, robust"),
        ("rand()",                "Pseudo-random int [0, RAND_MAX]"),
        ("srand(seed)",           "Seed RNG; srand(time(NULL)) for different sequence"),
        ("qsort(arr,n,size,cmp)", "Sort array; cmp(a,b): neg/0/pos"),
        ("bsearch(key,arr,n,size,cmp)", "Binary search sorted array"),
        ("abs(n)",                "Integer absolute value"),
        ("labs(n)",               "Long absolute value"),
        ("div(a, b)",             "Returns {quot, rem} as div_t"),
        ("getenv(name)",          "Get environment variable string"),
        ("system(cmd)",           "Run shell command"),
    ],
    "<string.h> -- String and memory": [
        ("strlen(s)",             "Length excluding NUL"),
        ("strcpy(dst, src)",      "Copy string (unsafe if dst too small)"),
        ("strncpy(dst,src,n)",    "Copy at most n bytes; may not NUL-terminate"),
        ("strcat(dst, src)",      "Append src to dst (unsafe)"),
        ("strncat(dst,src,n)",    "Append at most n chars, always NUL-terminates"),
        ("strcmp(s1, s2)",        "Compare: neg/0/pos"),
        ("strncmp(s1,s2,n)",      "Compare at most n chars"),
        ("strchr(s, c)",          "Find first c in s; returns ptr or NULL"),
        ("strrchr(s, c)",         "Find last c"),
        ("strstr(hay, needle)",   "Find first occurrence of needle in hay"),
        ("strtok(s, delim)",      "Tokenize; modifies string; not thread-safe"),
        ("memcpy(dst,src,n)",     "Copy n bytes (no overlap allowed)"),
        ("memmove(dst,src,n)",    "Copy n bytes (overlap safe)"),
        ("memset(ptr,val,n)",     "Fill n bytes with val"),
        ("memcmp(a,b,n)",         "Compare n bytes"),
    ],
    "<math.h>  -- link with -lm": [
        ("fabs(x)",               "Floating absolute value"),
        ("sqrt(x)",               "Square root"),
        ("pow(x, y)",             "x^y"),
        ("exp(x)",                "e^x"),
        ("log(x)",                "Natural log ln(x)"),
        ("log2(x)",               "Log base 2"),
        ("log10(x)",              "Log base 10"),
        ("sin/cos/tan(x)",        "Trig (radians)"),
        ("asin/acos/atan(x)",     "Inverse trig"),
        ("atan2(y, x)",           "4-quadrant atan(y/x); use this not atan"),
        ("ceil/floor/round(x)",   "Round up/down/nearest"),
        ("fmod(x, y)",            "Float remainder"),
        ("hypot(x, y)",           "sqrt(x^2+y^2), no overflow"),
        ("HUGE_VAL",              "Overflow result sentinel"),
        ("M_PI",                  "3.14159... (POSIX, not C standard)"),
        ("isnan/isinf/isfinite(x)","Check IEEE 754 special values (<math.h>)"),
    ],
    "<time.h>   -- timing": [
        ("time(NULL)",            "Current time as time_t (seconds since epoch)"),
        ("clock()",               "CPU time used; divide by CLOCKS_PER_SEC"),
        ("difftime(t2, t1)",      "Difference in seconds (double)"),
        ("localtime(&t)",         "time_t -> struct tm (local timezone)"),
        ("gmtime(&t)",            "time_t -> struct tm (UTC)"),
        ("strftime(buf,n,fmt,tm)","Format time to string"),
        ("mktime(&tm)",           "struct tm -> time_t"),
    ],
    "<assert.h> / <errno.h>": [
        ("assert(expr)",          "Abort with message if expr is false (debug)"),
        ("NDEBUG",                "Define to disable all asserts"),
        ("errno",                 "Global error code set by library functions"),
        ("strerror(errno)",       "String description of errno"),
        ("EINVAL/ENOENT/ENOMEM",  "Common errno values (invalid arg/no file/no mem)"),
    ],
}

for header, funcs in stdlib_table.items():
    print(f"\n  {header}")
    print(f"  {'-'*62}")
    for name, desc in funcs:
        print(f"    {name:<35} {desc}")

print("""
  SAFE PATTERNS:
    Always check malloc return:
      int *p = malloc(n * sizeof *p);
      if (!p) { perror("malloc"); exit(EXIT_FAILURE); }

    Prefer strtol over atoi:
      char *end;
      errno = 0;
      long val = strtol(str, &end, 10);
      if (errno || end == str || *end) { /* error */ }

    Prefer snprintf over sprintf:
      snprintf(buf, sizeof(buf), "value=%d", x);  /* never overflows */

    Free every malloc; NULL after free:
      free(ptr);  ptr = NULL;  /* double-free protection */

    String size: always n+1 bytes for string of length n (NUL terminator).
""")

# ============================================================
# MATPLOTLIB -- 4-PANEL FIGURE
# ============================================================
print(f"\n{SEP}")
print("BUILDING FIGURE...")
print(SEP)

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("#F8F8F0")
gs0 = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                        top=0.92, bottom=0.06, left=0.06, right=0.97)

ax_euler  = fig.add_subplot(gs0[0, 0])   # Euler / unit circle
ax_delta  = fig.add_subplot(gs0[0, 1])   # 4 delta representations
ax_chirp  = fig.add_subplot(gs0[0, 2])   # Complex chirp H(nu)
ax_roots  = fig.add_subplot(gs0[1, 0])   # Roots of unity
ax_prop   = fig.add_subplot(gs0[1, 1])   # Free-particle propagator
ax_c      = fig.add_subplot(gs0[1, 2])   # C precedence quick-ref

fig.suptitle("Dirac Delta + Complex Exponential + C Language Reference",
             fontsize=14, fontweight="bold", color="#1a1a2e")

# ---- AX_EULER: unit circle ----
ax = ax_euler
ax.set_facecolor("#F0F4FF")
theta = np.linspace(0, 2*np.pi, 400)
ax.plot(np.cos(theta), np.sin(theta), "k-", lw=1.2, alpha=0.5)
# Phasor at theta = pi/3
phi0 = np.pi/3
ax.annotate("", xy=(np.cos(phi0), np.sin(phi0)), xytext=(0,0),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2.0))
ax.plot([np.cos(phi0), np.cos(phi0)], [0, np.sin(phi0)], "#2ca02c", ls="--", lw=1.2)
ax.plot([0, np.cos(phi0)], [0, 0], "#d62728", ls="--", lw=1.2)
ax.text(np.cos(phi0)+0.05, np.sin(phi0)+0.05, r"$e^{i\theta}$", fontsize=12, color="#1f77b4")
ax.text(np.cos(phi0)/2, -0.12, r"$\cos\theta$", fontsize=9, color="#d62728", ha="center")
ax.text(np.cos(phi0)+0.08, np.sin(phi0)/2, r"$\sin\theta$", fontsize=9, color="#2ca02c")
# Euler labels
ax.axhline(0, color="#999", lw=0.6)
ax.axvline(0, color="#999", lw=0.6)
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
ax.set_aspect("equal")
ax.set_title(r"Euler: $e^{i\theta} = \cos\theta + i\sin\theta$", fontsize=10)
ax.grid(alpha=0.2)
ax.text(-1.3, 1.2,
        r"$e^{i\pi}+1=0$", fontsize=12, color="#8c1a1a",
        fontweight="bold",
        bbox=dict(fc="#fff8f0", ec="#bbb", pad=3))

# ---- AX_DELTA: 4 representations ----
ax = ax_delta
ax.set_facecolor("#FFF0F0")
x_arr = np.linspace(-2, 2, 2000)
eps_vals = [0.5, 0.2, 0.08]
colors_e = ["#aac8e8", "#4a90d9", "#1a3a8c"]

for eps, col in zip(eps_vals, colors_e):
    gauss = np.exp(-x_arr**2/eps**2) / (eps * np.sqrt(np.pi))
    ax.plot(x_arr, gauss, color=col, lw=1.3, alpha=0.9,
            label=f"Gauss eps={eps:.2f}")

# Lorentzian for smallest eps
eps_lor = 0.08
lor = (1/np.pi) * eps_lor / (x_arr**2 + eps_lor**2)
ax.plot(x_arr, lor, "#d62728", lw=1.3, ls="--", label=f"Lorentz eps={eps_lor}")

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.5, 8)
ax.set_xlabel("x", fontsize=9)
ax.set_ylabel(r"$\delta_\epsilon(x)$", fontsize=9)
ax.set_title(r"Dirac $\delta(x)$: Gaussian + Lorentzian reps", fontsize=10)
ax.legend(fontsize=7, loc="upper right")
ax.grid(alpha=0.2)
ax.text(0.02, 0.95,
        r"$\lim_{\epsilon\to 0}\frac{1}{\epsilon\sqrt{\pi}}e^{-x^2/\epsilon^2}=\delta(x)$",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ---- AX_CHIRP: complex chirp H(nu) ----
ax = ax_chirp
ax.set_facecolor("#F0FFF0")
nu = np.linspace(-5, 5, 1000)
D  = 2000.0
H  = np.exp(1j * np.pi * D * nu**2 / 1e6)   # scaled D for visibility
ax.plot(nu, H.real, "#1f77b4", lw=1.5, label=r"Re[$H(\nu)$] = $\cos(\pi D\nu^2)$")
ax.plot(nu, H.imag, "#ff7f0e", lw=1.5, label=r"Im[$H(\nu)$] = $\sin(\pi D\nu^2)$")
ax.plot(nu, np.abs(H), "k--", lw=1.0, label=r"|H(\nu)| = 1 (all-pass)")
ax.set_xlabel(r"$\nu$ (frequency)", fontsize=9)
ax.set_ylabel(r"$H(\nu) = e^{i\pi D\nu^2}$", fontsize=9)
ax.set_title("Dispersion Chirp = Complex Exponential", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(alpha=0.2)
ax.text(0.02, 0.05,
        "GS: phase = arg(H) = pi*D*nu^2\n"
        "All-pass: |H|=1, no amplitude change\n"
        "= free-particle QM propagator",
        transform=ax.transAxes, fontsize=7.5,
        bbox=dict(fc="#fffff0", ec="#bbb", pad=2))

# ---- AX_ROOTS: roots of unity ----
ax = ax_roots
ax.set_facecolor("#FFF5FF")
ax.set_aspect("equal")
theta_c = np.linspace(0, 2*np.pi, 300)
ax.plot(np.cos(theta_c), np.sin(theta_c), "k-", lw=0.8, alpha=0.3)
ax.axhline(0, color="#999", lw=0.5)
ax.axvline(0, color="#999", lw=0.5)

for n_r, col in [(4,"#1f77b4"),(6,"#d62728"),(8,"#2ca02c")]:
    roots_r = [np.exp(2j*np.pi*k/n_r) for k in range(n_r)]
    rx = [r.real for r in roots_r]
    ry = [r.imag for r in roots_r]
    ax.scatter(rx, ry, color=col, s=50, zorder=5, label=f"n={n_r}")
    # connect with polygon
    rx_c = rx + [rx[0]]
    ry_c = ry + [ry[0]]
    ax.plot(rx_c, ry_c, color=col, lw=0.8, alpha=0.5)

ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
ax.set_title(r"Roots of Unity: $z^n=1$, $z_k=e^{2\pi ik/n}$", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.15)
ax.text(-1.3, -1.25,
        r"Sum = 0, $|z_k|=1$", fontsize=8, color="#555")

# ---- AX_PROP: free-particle propagator ----
ax = ax_prop
ax.set_facecolor("#F0F8FF")
x_p = np.linspace(-6, 6, 1000)
for t_p, col, ls in [(0.1,"#d62728","-"), (0.5,"#ff7f0e","-"),
                     (1.0,"#1f77b4","-"), (3.0,"#2ca02c","--")]:
    K = np.sqrt(1/(2*np.pi*t_p)) * np.exp(1j * x_p**2 / (2*t_p))
    ax.plot(x_p, K.real, color=col, lw=1.3, ls=ls, label=f"t={t_p}, Re[K]")
ax.set_xlabel("x", fontsize=9)
ax.set_ylabel("Re[K(x,t)]", fontsize=9)
ax.set_title(r"Free-particle Propagator $K\propto e^{imx^2/2\hbar t}$", fontsize=10)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.2)
ax.text(0.02, 0.97,
        "t->0: K->delta(x)\nSame chirp as H(nu)!",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ---- AX_C: C precedence quick-ref table ----
ax = ax_c
ax.set_facecolor("#FFFFF0")
ax.axis("off")
ax.set_title("C Operator Precedence (1=highest)", fontsize=10)

short_table = [
    ("1", "L", "()  []  .  ->  x++  x--"),
    ("2", "R", "++x --x  +x -x  ! ~  (T)x  *x  &x  sizeof"),
    ("3", "L", "*   /   %"),
    ("4", "L", "+   -"),
    ("5", "L", "<<  >>"),
    ("6", "L", "<  <=  >  >="),
    ("7", "L", "==  !="),
    ("8", "L", "&   (bitwise AND)"),
    ("9", "L", "^   (bitwise XOR)"),
    ("10","L", "|   (bitwise OR)"),
    ("11","L", "&&  (logical AND, short-circuit)"),
    ("12","L", "||  (logical OR,  short-circuit)"),
    ("13","R", "?:  (ternary)"),
    ("14","R", "=  +=  -=  *=  /=  etc."),
    ("15","L", ",   (comma/sequence)"),
]

col_colors = ["#ffe8e8","#fff8e8","#e8ffe8","#e8f0ff","#f8e8ff"]
for k, (prec, assoc, ops) in enumerate(short_table):
    y = 0.97 - k*0.061
    bg = col_colors[k % len(col_colors)]
    ax.text(0.02, y, prec,    fontsize=7.5, va="top", color="#555",
            transform=ax.transAxes, fontweight="bold")
    ax.text(0.08, y, assoc,   fontsize=7.5, va="top", color="#888",
            transform=ax.transAxes)
    ax.text(0.14, y, ops,     fontsize=7.2, va="top", color="#1a1a4e",
            transform=ax.transAxes, fontfamily="monospace")

ax.text(0.02, 0.02,
        "TRAP: & < == < && < ||  (not left-to-right!)\n"
        "Always parenthesize bitwise ops: (a & b) == c",
        transform=ax.transAxes, fontsize=7.5,
        bbox=dict(fc="#fff0f0", ec="#d62728", pad=2))

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {OUT}")

print(f"\n{SEP}")
print("Done.")
print(SEP)
