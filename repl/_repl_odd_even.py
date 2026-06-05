"""
repl/_repl_odd_even.py
Odd and even functions. Mathematical maturity: symmetry everywhere.
"""
import numpy as np
import sympy as sp
sp.init_printing(use_unicode=False, wrap_line=False)

print("=" * 55)
print("ODD AND EVEN FUNCTIONS")
print("=" * 55)
print()

# ============================================================
# RESULT FIRST
# ============================================================
print("""RESULT FIRST:
  Even: f(-x) =  f(x)   symmetric about y-axis   cos, x^2, |x|
  Odd:  f(-x) = -f(x)   antisymmetric             sin, x,   x^3

  Any function = even part + odd part:
    f(x) = [f(x)+f(-x)]/2  +  [f(x)-f(-x)]/2
              even                 odd

  Integrals:
    integral of odd  over [-a,a]  = 0   (cancels)
    integral of even over [-a,a]  = 2 * integral over [0,a]

  Fourier:
    even f(x) -> purely real transform F(k)
    odd  f(x) -> purely imaginary transform F(k)
    real signal  -> F(-k) = F(k)*  (Hermitian symmetry)
""")

# ============================================================
# 1. SymPy: verify algebraically
# ============================================================
print("=== 1. SymPy verification ===")
x = sp.Symbol('x', real=True)

tests = [
    ('cos(x)',   sp.cos(x)),
    ('sin(x)',   sp.sin(x)),
    ('x^2',      x**2),
    ('x^3',      x**3),
    ('exp(x)',   sp.exp(x)),
    ('|x|',      sp.Abs(x)),
    ('x*cos(x)', x*sp.cos(x)),
    ('x^2+x',   x**2 + x),
]

print(f"  {'f(x)':15s}  {'f(-x)=f(x)?':12s}  {'f(-x)=-f(x)?':13s}  {'type'}")
print("  " + "-"*55)
for name, f in tests:
    fm   = f.subs(x, -x)
    even = sp.simplify(fm - f) == 0
    odd  = sp.simplify(fm + f) == 0
    kind = 'even' if even else ('odd' if odd else 'neither')
    print(f"  {name:15s}  {str(even):12s}  {str(odd):13s}  {kind}")
print()

# decompose exp(x) into even and odd parts
f_exp    = sp.exp(x)
even_exp = sp.simplify((f_exp + f_exp.subs(x,-x)) / 2)
odd_exp  = sp.simplify((f_exp - f_exp.subs(x,-x)) / 2)
print(f"exp(x) = even part + odd part:")
print(f"  even part = (exp(x)+exp(-x))/2 = "); sp.pprint(even_exp)
print(f"  odd  part = (exp(x)-exp(-x))/2 = "); sp.pprint(odd_exp)
print(f"  -> cosh(x) + sinh(x)  (exact Euler-like decomposition)")
print()

# ============================================================
# 2. Integrals exploit symmetry
# ============================================================
print("=== 2. Integrals: symmetry kills work ===")
a = sp.Symbol('a', positive=True)

integrals = [
    ("x^3 on [-a,a]",    x**3,           -a, a,   "odd -> 0"),
    ("cos(x) on [-pi,pi]", sp.cos(x),  -sp.pi, sp.pi, "even -> 2*integral[0,pi]"),
    ("sin(x) on [-pi,pi]", sp.sin(x),  -sp.pi, sp.pi, "odd  -> 0"),
    ("x*sin(x) on [-pi,pi]", x*sp.sin(x), -sp.pi, sp.pi, "even (odd*odd=even)"),
]
for desc, f, lo, hi, note in integrals:
    val = sp.integrate(f, (x, lo, hi))
    print(f"  int({desc}) = {sp.simplify(val)}   [{note}]")
print()

# ============================================================
# 3. FFT symmetry: real signal -> Hermitian spectrum
# ============================================================
print("=== 3. FFT symmetry ===")
N = 16
rng = np.random.default_rng(0)
x_real = rng.normal(size=N)                    # real signal
x_imag = rng.normal(size=N) + 1j*rng.normal(size=N)  # complex signal

X_real = np.fft.fft(x_real)
X_imag = np.fft.fft(x_imag)

herm_real = np.allclose(X_real[1:N//2], np.conj(X_real[N//2+1:][::-1]))
herm_imag = np.allclose(X_imag[1:N//2], np.conj(X_imag[N//2+1:][::-1]))
print(f"  Real input  -> X[-k]=X[k]*  (Hermitian): {herm_real}")
print(f"  Complex input -> Hermitian:              {herm_imag}")
print()
print(f"  Consequence for rfft:")
print(f"    fft(real, N)  -> N complex values, but only N/2+1 unique")
print(f"    rfft(real, N) -> N/2+1 values  (discards redundant half)")
print(f"    2x memory saving, sqrt(2) speed saving")
print()

# even/odd decomposition of a signal
sig = rng.normal(size=32)
even_sig = (sig + sig[::-1]) / 2
odd_sig  = (sig - sig[::-1]) / 2
print(f"  Signal decomposition check:")
print(f"    max|sig - (even+odd)| = {np.max(np.abs(sig - (even_sig+odd_sig))):.2e}")
print(f"    even part is_even: {np.allclose(even_sig, even_sig[::-1])}")
print(f"    odd  part is_odd:  {np.allclose(odd_sig,  -odd_sig[::-1])}")
print()

# ============================================================
# 4. GS connection: odd/even in dispersion
# ============================================================
print("=== 4. GS: H(nu) symmetry ===")
print("""
H(nu) = exp(i*pi*D*nu^2)

  nu^2 is EVEN in nu  ->  H(nu) is EVEN
  |H(nu)| = 1         ->  all-pass, Hermitian for real signals

  For real input E(t):
    FFT(E)[k] and FFT(E)[-k] are conjugates
    H[k] = H[-k]  (even)  ->  output stays real-valued

  This is why disperse() preserves real-signal structure:
    E_real -> fft -> multiply by even H -> ifft -> E_real (still real)

  If D=0: H=1 everywhere, identity.
  If D!=0: quadratic phase ramp, spreads pulse in time.
  Even symmetry of H guarantees no DC offset injection.
""")

D = 5000; N = 512
nu = np.fft.fftfreq(N)
H = np.exp(1j*np.pi*D*nu**2)
print(f"  H is even: H[k]=H[-k]? {np.allclose(H, H[::-1])}")
print(f"  H is Hermitian: H[k]=conj(H[-k])? {np.allclose(H, np.conj(H[::-1]))}")
print(f"  -> H is real-valued and even: cos(pi*D*nu^2) + i*sin(pi*D*nu^2)")
print(f"     but NOT Hermitian (imaginary part is not antisymmetric)")
print()
print(f"  |H[k]|: min={np.abs(H).min():.6f}  max={np.abs(H).max():.6f}  (all-pass confirmed)")
