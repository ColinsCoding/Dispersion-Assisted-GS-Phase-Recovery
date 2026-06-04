"""
_repl_sympy_physics.py -- SymPy init_printing, loops, physics intersections
2D -> 3D geometry, Euler-Lagrange, GS fixed point, uncertainty principle
"""
import sympy as sp
sp.init_printing(use_unicode=False, wrap_line=False)

print("=== SymPy init_printing active ===")
print()

# ============================================================
# 1. Basic symbols, print nicely
# ============================================================
x, y, z, t, k, w = sp.symbols('x y z t k omega', real=True)
a, b, c_s = sp.symbols('a b c', positive=True)
hbar, m, E_s, V = sp.symbols('hbar m E V', positive=True)

expr = sp.sqrt(x**2 + y**2)
print("2D distance:", sp.pretty(expr))
print()

expr3 = sp.sqrt(x**2 + y**2 + z**2)
print("3D distance:", sp.pretty(expr3))
print()

# ============================================================
# 2. Loop over physics expressions, print each
# ============================================================
print("=== Physics expression zoo ===")
exprs = [
    ("Gaussian",          sp.exp(-x**2 / (2*a**2))),
    ("Dispersion kernel", sp.exp(sp.I * sp.pi * k * x**2)),
    ("de Broglie",        hbar / (m * sp.sqrt(2*E_s/m))),
    ("Lorentzian",        a / (sp.pi * (x**2 + a**2))),
    ("Sinc",              sp.sinc(x)),
    ("Heaviside deriv",   sp.diff(sp.Heaviside(x), x)),
    ("Bessel J0",         sp.besselj(0, x)),
]

for name, expr in exprs:
    print(f"  {name}:")
    print("  " + sp.pretty(expr))
    print()

# ============================================================
# 3. 2D -> 3D: gradient and Laplacian
# ============================================================
print("=== Gradient: 2D -> 3D ===")
f2 = sp.exp(-(x**2 + y**2) / (2*a**2))
f3 = sp.exp(-(x**2 + y**2 + z**2) / (2*a**2))

grad2 = [sp.diff(f2, v) for v in [x, y]]
grad3 = [sp.diff(f3, v) for v in [x, y, z]]

print("f(x,y) = exp(-(x^2+y^2)/2a^2)")
print("  grad_x =", sp.pretty(sp.simplify(grad2[0])))
print("  grad_y =", sp.pretty(sp.simplify(grad2[1])))
print()
print("f(x,y,z) = exp(-(x^2+y^2+z^2)/2a^2)")
print("  grad_x =", sp.pretty(sp.simplify(grad3[0])))
print()

lap2 = sum(sp.diff(f2, v, 2) for v in [x, y])
lap3 = sum(sp.diff(f3, v, 2) for v in [x, y, z])
print("Laplacian 2D:", sp.pretty(sp.simplify(lap2)))
print("Laplacian 3D:", sp.pretty(sp.simplify(lap3)))
print()

# ============================================================
# 4. Euler-Lagrange -> GS fixed point
# ============================================================
print("=== Euler-Lagrange equation ===")
print("""
Lagrangian for GS phase recovery:
  L[phi] = (I_measured - |E_dispersed|^2)^2

Euler-Lagrange:  d/dphi ( dL/d(dphi) ) - dL/dphi = 0

For unit-amplitude signal E = exp(i*phi):
  dL/dphi = 0  at the GS fixed point
  -> phase phi satisfies: angle(H * exp(i*phi)) = angle(H * E_true)
  -> GS iteration IS the gradient step on this functional
""")

phi, phi_t, D_s = sp.symbols('phi phi_t D', real=True)
I_m = sp.Symbol('I_m', positive=True)

# simplified 1D version: L = (cos(phi - phi_t))^2 error analog
L = (1 - sp.cos(phi - phi_t))**2
dL = sp.diff(L, phi)
fixed = sp.solve(dL, phi)
print("Simplified loss L = (1 - cos(phi - phi_t))^2")
print("dL/dphi =", sp.pretty(dL))
print("Fixed points:", [sp.pretty(f) for f in fixed[:4]])
print("-> phi = phi_t + 2*pi*n  (correct) and phi = phi_t + pi + 2*pi*n (trap)")
print()

# ============================================================
# 5. Uncertainty principle: SymPy Fourier + sigma product
# ============================================================
print("=== Heisenberg uncertainty: sigma_x * sigma_k >= 1/2 ===")

# Gaussian saturates the bound
sigma = sp.Symbol('sigma', positive=True)
xi = sp.Symbol('xi', real=True)

psi    = sp.exp(-xi**2 / (4*sigma**2))
norm2  = sp.integrate(psi**2, (xi, -sp.oo, sp.oo))
psi_n  = psi / sp.sqrt(norm2)

x2_avg = sp.integrate(xi**2 * psi_n**2, (xi, -sp.oo, sp.oo))
sigma_x = sp.sqrt(x2_avg)

print("Gaussian psi = exp(-x^2/4sigma^2)")
print("  sigma_x =", sp.pretty(sp.simplify(sigma_x)))
print("  sigma_k = 1/(2*sigma)  (Fourier dual)")
print("  product = sigma_x * sigma_k =", sp.pretty(sp.simplify(sigma_x * 1/(2*sigma))))
print("  = 1/2  (saturates Heisenberg bound, Gaussian is minimum uncertainty)")
print()

# ============================================================
# 6. Loop: Taylor series for common physics functions
# ============================================================
print("=== Taylor series loop ===")
funcs = [
    ("exp(x)",      sp.exp(x),           6),
    ("sin(x)",      sp.sin(x),           7),
    ("cos(x)",      sp.cos(x),           6),
    ("1/(1-x)",     1/(1-x),             5),
    ("sqrt(1+x)",   sp.sqrt(1+x),        4),
    ("log(1+x)",    sp.log(1+x),         5),
]

for name, f, order in funcs:
    series = sp.series(f, x, 0, order).removeO()
    print(f"  {name:15s} = {sp.pretty(series)}")
print()
print("sqrt(1+x) ~ 1 + x/2  ->  non-relativistic KE = mc^2 * (1 + p^2/2m^2c^2)")
print("exp(ix)   = cos+i*sin ->  GS dispersion kernel H = exp(i*pi*D*nu^2)")
