"""
repl/_repl_griffiths_notation.py
Griffiths notation in SymPy. Write once, run anytime.
Covers Griffiths Ch 1-4: wavefunctions, operators, expectation values,
hydrogen atom, spin. Plus exp(-i*phase) complex analysis.
"""
import numpy as np
import sympy as sp
from sympy import (symbols, Function, integrate, oo, sqrt, exp, I,
                   pi, conjugate, diff, simplify, latex, Abs, cos, sin,
                   Matrix, eye)
sp.init_printing(use_unicode=False, wrap_line=False)

print("=" * 60)
print("GRIFFITHS NOTATION -- SymPy Reference")
print("=" * 60)
print()

# ============================================================
# 1. Wavefunction normalization (Griffiths 1.3)
# ============================================================
print("=== Griffiths 1.3: Normalization ===")
x, t, a_s, k_s = symbols('x t a k', real=True)
hbar, m_s, omega_s = symbols('hbar m omega', positive=True)

# Gaussian wavepacket
psi = exp(-a_s * x**2)
norm2 = integrate(psi * conjugate(psi), (x, -oo, oo))
print("psi =", sp.pretty(psi))
print("int|psi|^2 dx =", sp.pretty(simplify(norm2)))
A = 1 / sqrt(norm2)
print("Normalization A =", sp.pretty(simplify(A)))
psi_n = A * psi
print("psi_normalized =", sp.pretty(psi_n))
print()

# expectation value <x>
x_exp = integrate(conjugate(psi_n) * x * psi_n, (x, -oo, oo))
x2_exp = integrate(conjugate(psi_n) * x**2 * psi_n, (x, -oo, oo))
sigma_x = sqrt(simplify(x2_exp - x_exp**2))
print("<x>  =", sp.pretty(simplify(x_exp)))
print("<x^2>=", sp.pretty(simplify(x2_exp)))
print("sigma_x =", sp.pretty(simplify(sigma_x)))
print()

# ============================================================
# 2. Operators: momentum and Hamiltonian (Griffiths 1.5, 2.1)
# ============================================================
print("=== Griffiths 1.5 / 2.1: Operators ===")
print("""
Momentum operator:   p_hat = -i*hbar * d/dx
Kinetic energy:      T_hat = p_hat^2 / 2m = -hbar^2/(2m) * d^2/dx^2
Hamiltonian:         H_hat = T_hat + V(x)

Commutator:          [x, p] = x*p - p*x = i*hbar
""")

# verify [x,p] = i*hbar symbolically
f = Function('f')
xp_f = x * (-I*hbar * diff(f(x), x))
px_f = -I*hbar * diff(x * f(x), x)
commutator = simplify(xp_f - px_f)
print("[x,p]f(x) = xp*f - px*f =", sp.pretty(commutator))
print("= i*hbar * f(x)  confirmed")
print()

# ============================================================
# 3. Infinite square well (Griffiths 2.2)
# ============================================================
print("=== Griffiths 2.2: Infinite Square Well ===")
n_sym, L_s = symbols('n L', positive=True, integer=True)
L_s = symbols('L', positive=True)
n_sym = symbols('n', positive=True, integer=True)

psi_n_well = sqrt(2/L_s) * sin(n_sym * pi * x / L_s)
E_n_well   = n_sym**2 * pi**2 * hbar**2 / (2 * m_s * L_s**2)

print("psi_n(x) =", sp.pretty(psi_n_well))
print("E_n      =", sp.pretty(E_n_well))
print()

# orthonormality check n=1,2
L_n = sp.Symbol('L_val', positive=True)
for n1, n2 in [(1,1),(1,2),(2,2)]:
    psi1 = sqrt(2) * sin(n1 * pi * x)   # L=1 for simplicity
    psi2 = sqrt(2) * sin(n2 * pi * x)
    overlap = integrate(psi1 * psi2, (x, 0, 1))
    print(f"  <psi_{n1}|psi_{n2}> = {simplify(overlap)}")
print()

# ============================================================
# 4. Harmonic oscillator ladder operators (Griffiths 2.3)
# ============================================================
print("=== Griffiths 2.3: Harmonic Oscillator ===")
print("""
a_+ = 1/sqrt(2*hbar*m*omega) * (-i*p + m*omega*x)   (raising)
a_- = 1/sqrt(2*hbar*m*omega) * (+i*p + m*omega*x)   (lowering)

H = hbar*omega*(a_+*a_- + 1/2) = hbar*omega*(N + 1/2)

E_n = hbar*omega*(n + 1/2)    n = 0,1,2,...

a_- |n> = sqrt(n)   |n-1>
a_+ |n> = sqrt(n+1) |n+1>
""")
n_ho = symbols('n', nonneg=True, integer=True)
E_ho = hbar * omega_s * (n_ho + sp.Rational(1,2))
print("E_n =", sp.pretty(E_ho))
for n in range(5):
    E_val = f"hbar*omega*{n + 0.5}"
    print(f"  n={n}  E = {n+0.5}*hbar*omega")
print()

# ============================================================
# 5. exp(-i*phase): complex analysis
# ============================================================
print("=== Complex Analysis: exp(-i*phase) ===")
phi_s = symbols('phi', real=True)

euler = exp(I*phi_s)
euler_neg = exp(-I*phi_s)

print("Euler's formula:")
print("  exp(i*phi)  =", sp.pretty(sp.expand_complex(euler)))
print("  exp(-i*phi) =", sp.pretty(sp.expand_complex(euler_neg)))
print()

# GS uses exp(i*phi) for field, exp(-i*phi) for conjugate
print("In GS phase recovery:")
print("  E(t)    = exp(i*phi(t))       field  (unit amplitude)")
print("  E*(t)   = exp(-i*phi(t))      conjugate")
print("  |E|^2   = E*E* = 1            intensity (unit amplitude)")
print("  angle(E)= phi(t)              recovered phase")
print()

# Cauchy-Riemann equations
z = symbols('z')
z_r, z_i = symbols('x y', real=True)
f_re = z_r**2 - z_i**2   # Re(z^2)
f_im = 2*z_r*z_i          # Im(z^2)

CR1 = simplify(diff(f_re, z_r) - diff(f_im, z_i))
CR2 = simplify(diff(f_re, z_i) + diff(f_im, z_r))
print("Cauchy-Riemann for f(z)=z^2: u=x^2-y^2, v=2xy")
print(f"  du/dx - dv/dy = {CR1}  (should be 0)")
print(f"  du/dy + dv/dx = {CR2}  (should be 0)")
print("  f(z)=z^2 is analytic everywhere (entire function)")
print()

# residue theorem application
print("Residue theorem (key for Fourier integrals):")
print("""
  integral exp(i*k*x) / (x^2 + a^2) dx  from -inf to +inf

  Poles at x = +/- i*a  (close upper half-plane for k>0)
  Residue at x=+ia: exp(-k*a) / (2*i*a)
  Result: pi/a * exp(-k*a)   (Lorentzian FT = exponential)

  This is why Lorentzian lineshape FT gives exponential decay:
  spectral line <-> exponential ringdown in time domain.
""")

# numerical verification
a_n = 2.0
k_n = 1.5
x_n = np.linspace(-100, 100, 100000)
dx  = x_n[1]-x_n[0]
integrand = np.exp(1j*k_n*x_n) / (x_n**2 + a_n**2)
result_num = np.trapezoid(integrand, x_n)
result_exact = np.pi/a_n * np.exp(-k_n*a_n)
print(f"Numerical: {result_num.real:.6f}  Exact: {result_exact:.6f}  "
      f"err={abs(result_num.real-result_exact):.2e}")
print()

# ============================================================
# 6. Spin-1/2: Pauli matrices (Griffiths 4.4)
# ============================================================
print("=== Griffiths 4.4: Spin-1/2, Pauli Matrices ===")
sx = Matrix([[0,1],[1,0]]) / 2
sy = Matrix([[0,-I],[I,0]]) / 2
sz = Matrix([[1,0],[0,-1]]) / 2

print("S_x =", sp.pretty(sx * 2), "* hbar/2")
print("S_y =", sp.pretty(sy * 2), "* hbar/2")
print("S_z =", sp.pretty(sz * 2), "* hbar/2")
print()

# commutation [Sx, Sy] = i*hbar*Sz
comm_xy = 2*sx*2*sy - 2*sy*2*sx
print("[sigma_x, sigma_y] =", sp.pretty(simplify(comm_xy)))
print("= 2*i*sigma_z  confirmed  (so [Sx,Sy] = i*hbar*Sz)")
print()

# eigenvalues of Sz
eigs = sz.eigenvects()
print("Eigenvalues of S_z (hbar/2 units):")
for eigenval, mult, vecs in eigs:
    print(f"  lambda={eigenval*2} (hbar/2)  eigenvector={sp.pretty(vecs[0].T)}")
print()

# ============================================================
# 7. Verilog / C / numpy / sympy: same operation
# ============================================================
print("=== Same operation: Verilog / C / numpy / SymPy ===")
print("""
Operation: phase rotation by phi

Verilog (fixed point, 16-bit):
  reg signed [15:0] re_out, im_out;
  re_out = (re_in * cos_phi - im_in * sin_phi) >>> 15;
  im_out = (re_in * sin_phi + im_in * cos_phi) >>> 15;

C (float):
  float re_out = re_in*cos(phi) - im_in*sin(phi);
  float im_out = re_in*sin(phi) + im_in*cos(phi);

numpy:
  E_out = E_in * np.exp(1j * phi)

SymPy:
  E_out = E_in * exp(I * phi)

All equivalent. numpy/SymPy use complex exponential directly.
Verilog/C use trig decomposition (CORDIC on hardware).
""")

phi_n = np.pi/3
E_in  = 1.0 + 0.5j
E_out_np  = E_in * np.exp(1j * phi_n)
E_out_trig = complex(E_in.real*np.cos(phi_n) - E_in.imag*np.sin(phi_n),
                     E_in.real*np.sin(phi_n) + E_in.imag*np.cos(phi_n))
print(f"numpy:  {E_out_np:.6f}")
print(f"trig:   {E_out_trig:.6f}")
print(f"match:  {np.allclose(E_out_np, E_out_trig)}")
