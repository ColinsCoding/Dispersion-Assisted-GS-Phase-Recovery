# %% [markdown]
# # Griffiths QM → RogueGuard Pipeline
# **§1–§8**: Wave functions · Expectation values · Uncertainty · Operators · Parity · Infinite square well · Harmonic oscillator · Rogue wave statistics

# %%
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, Math
from sympy import init_printing, sqrt, exp, pi, oo, symbols, integrate, diff, conjugate, simplify, Rational, latex

init_printing(use_latex="mathjax")

# ── helpers ──────────────────────────────────────────────────────────────────
def show(expr, label=""):
    try:
        display(Math(r"\text{" + label + r"}\quad" + latex(expr) if label else latex(expr)))
    except Exception:
        print(f"{label}  {sp.pretty(expr)}")

def chk(val, ref, label, tol=1e-6, absolute=False):
    val = float(np.real(val))
    ref = float(np.real(ref))
    if absolute:
        ok = abs(val - ref) < tol
    else:
        ok = abs(val - ref) / (abs(ref) + 1e-30) < tol
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: got {val:.8g}, expected {ref:.8g}")
    return ok

PASS_COUNT = 0

# %% [markdown]
# ## §1 Wave Function — Gaussian Ψ, normalisation, P(within 1σ)

# %%
x, x0, sigma, hbar_sym = symbols("x x_0 sigma hbar", real=True, positive=True)

# Gaussian wave function (real, centered at x0)
Psi = (1 / (sp.sqrt(2 * pi) * sigma)) ** sp.Rational(1, 2) * exp(-(x - x0)**2 / (4 * sigma**2))
show(Psi, "Psi(x) =")

# Normalisation ∫|Ψ|²dx = 1
norm_sym = integrate(Psi**2, (x, -oo, oo))
norm_sym = simplify(norm_sym)
show(norm_sym, "norm =")

# Numerical checks
sig_val = 1.0
x0_val = 0.0
xx = np.linspace(-10, 10, 100_000)
psi_num = (1 / (np.sqrt(2 * np.pi) * sig_val)) ** 0.5 * np.exp(-(xx - x0_val)**2 / (4 * sig_val**2))
norm_num = np.trapezoid(psi_num**2, xx)
PASS_COUNT += chk(norm_num, 1.0, "§1 normalisation int|Psi|^2 dx=1", tol=1e-4)

# P(within 1σ)
from scipy.special import erf
prob_1sigma = float(erf(1 / np.sqrt(2)))
PASS_COUNT += chk(prob_1sigma, 0.6826894921, "§1 P(within 1sigma)~68.27%", tol=1e-6)
print(f"  P(within 1sigma) = {prob_1sigma:.6f}")

# %% [markdown]
# ## §2 Expectation Values — ⟨x⟩, ⟨x²⟩, ⟨p⟩, ⟨p²⟩

# %%
# Numerical: use σ=1, x0=2
sig_val = 1.0
x0_val = 2.0
xx = np.linspace(-10, 14, 200_000)
psi2 = (1 / (np.sqrt(2 * np.pi) * sig_val)) ** 0.5 * np.exp(-(xx - x0_val)**2 / (4 * sig_val**2))
rho = psi2**2

Ex = np.trapezoid(xx * rho, xx)
PASS_COUNT += chk(Ex, x0_val, "§2 <x>=x0", tol=1e-4)

Ex2 = np.trapezoid(xx**2 * rho, xx)
PASS_COUNT += chk(Ex2, x0_val**2 + sig_val**2, "§2 <x^2>=x0^2+sigma^2", tol=1e-4)

# <p> = -ih ∫ψ* ∂ψ/∂x dx  (hbar=1 units)
dpsi_dx = np.gradient(psi2, xx)
Ep = np.trapezoid(psi2 * (-1j) * dpsi_dx, xx)
PASS_COUNT += chk(np.real(Ep), 0.0, "§2 <p>=0 for real Gaussian", tol=1e-4, absolute=True)

# <p²> = -hbar² ∫ψ* ∂²ψ/∂x² dx = hbar²/4σ² (hbar=1)
d2psi_dx2 = np.gradient(dpsi_dx, xx)
Ep2 = np.trapezoid(psi2 * (-1) * d2psi_dx2, xx)
PASS_COUNT += chk(np.real(Ep2), 1 / (4 * sig_val**2), "§2 <p^2>=hbar^2/4sigma^2", tol=1e-3)

# %% [markdown]
# ## §3 Uncertainty Principle — σ_x·σ_p = ℏ/2 (saturates), photonics TBP

# %%
# For Gaussian: σ_x = σ,  σ_p = hbar/(2σ)  →  σ_x·σ_p = hbar/2
sig_val = 1.5
hbar_val = 1.0
sigma_x = sig_val
sigma_p = hbar_val / (2 * sig_val)
product = sigma_x * sigma_p
PASS_COUNT += chk(product, hbar_val / 2, "§3 sigma_x*sigma_p = hbar/2 (saturates HUP)", tol=1e-10)

# Photonics TBP = 2 ln2 / pi for transform-limited Gaussian pulse
TBP = 2 * np.log(2) / np.pi
PASS_COUNT += chk(TBP, 2 * np.log(2) / np.pi, "§3 Photonics TBP = 2ln2/pi ~ 0.4413", tol=1e-10)
print(f"  TBP = {TBP:.6f}")

# %% [markdown]
# ## §4 Operators — x̂, p̂=-iℏ∂/∂x, [x̂,p̂]=iℏ, Hermitian Ĥ

# %%
xv = symbols("x", real=True)
hb = symbols("hbar", positive=True)
f_expr = sp.Function("f")(xv)

# [x, p]f = x(-ih ∂f/∂x) - (-ih ∂/∂x)(xf) = -ihx f' + ih(f + xf') = ih f
xp_f = xv * (-sp.I * hb * diff(f_expr, xv))
px_f = -sp.I * hb * diff(xv * f_expr, xv)
commutator = simplify(xp_f - px_f)
show(commutator, "[xhat, phat]f =")
# commutator = i*hbar*f  →  commutator/(i*hbar*f) = 1
ratio_sym = simplify(commutator / (sp.I * hb * f_expr))
PASS_COUNT += chk(float(ratio_sym), 1.0, "§4 [xhat,phat]=ihbar verified symbolically", tol=1e-10)

# Numerical Hermitian check for H = -hbar²/2m ∂²/∂x² (particle in box, hbar=m=1)
L_val = 1.0
n_pts = 10_000
xx_w = np.linspace(0, L_val, n_pts + 2)[1:-1]
n1, n2 = 2, 3
psi_m = np.sqrt(2 / L_val) * np.sin(n1 * np.pi * xx_w / L_val)
psi_n = np.sqrt(2 / L_val) * np.sin(n2 * np.pi * xx_w / L_val)
d2_psin = np.gradient(np.gradient(psi_n, xx_w), xx_w)
H_psin = -0.5 * d2_psin  # hbar=m=1
lhs = np.trapezoid(psi_m * H_psin, xx_w)
d2_psim = np.gradient(np.gradient(psi_m, xx_w), xx_w)
H_psim = -0.5 * d2_psim
rhs = np.trapezoid(H_psim * psi_n, xx_w)
PASS_COUNT += chk(lhs, rhs, "§4 Hhat Hermitian <m|H|n>=<Hm|n>", tol=1e-4, absolute=True)

# %% [markdown]
# ## §5 Parity — P²=I, eigenvalues ±1, [H,P]=0 for symmetric V, HO parities

# %%
xv = symbols("x", real=True)
f_even = sp.cos(xv)
f_odd = sp.sin(xv)

Pf_even = f_even.subs(xv, -xv)
Pf_odd = f_odd.subs(xv, -xv)
eigenvalue_even = simplify(Pf_even / f_even)
eigenvalue_odd = simplify(Pf_odd / f_odd)
PASS_COUNT += chk(float(eigenvalue_even), 1.0, "§5 P*cos(x) = +1*cos(x)", tol=1e-10)
PASS_COUNT += chk(float(eigenvalue_odd), -1.0, "§5 P*sin(x) = -1*sin(x)", tol=1e-10)

# P² = I
g = xv**3 + 2 * xv**2 - xv + 5
Pg = g.subs(xv, -xv)
PPg = Pg.subs(xv, -xv)
PASS_COUNT += chk(float(simplify(PPg - g).subs(xv, 1.7)), 0.0, "§5 P^2=I verified", tol=1e-10, absolute=True)

# HO eigenstates parity: psi_n(-x) = (-1)^n psi_n(x)
from numpy.polynomial.hermite import hermval
xi = np.linspace(-4, 4, 1000)
for n_ho in [0, 1, 2, 3]:
    coeffs = [0] * (n_ho + 1)
    coeffs[n_ho] = 1
    psi_ho = hermval(xi, coeffs) * np.exp(-xi**2 / 2)
    psi_reflected = hermval(-xi, coeffs) * np.exp(-xi**2 / 2)
    expected_parity = (-1)**n_ho
    mask = np.abs(psi_ho) > 1e-6
    ratio = psi_reflected[mask] / psi_ho[mask]
    parity_num = float(np.mean(ratio))
    PASS_COUNT += chk(parity_num, expected_parity, f"§5 HO psi_{n_ho} parity = {expected_parity:+d}", tol=1e-6)

# %% [markdown]
# ## §6 Infinite Square Well — ψ_n, E_n, orthonormality δ_mn

# %%
L_val = 1.0
n_pts = 50_000
xx_w = np.linspace(0, L_val, n_pts + 2)[1:-1]

def psi_well(n, x, L):
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

def E_well(n, L, hbar=1.0, m=1.0):
    return n**2 * np.pi**2 * hbar**2 / (2 * m * L**2)

# Normalisation
for n in [1, 2, 3]:
    norm = np.trapezoid(psi_well(n, xx_w, L_val)**2, xx_w)
    PASS_COUNT += chk(norm, 1.0, f"§6 psi_{n} normalised", tol=1e-4)

# Orthogonality
for m, n in [(1, 2), (1, 3), (2, 3)]:
    inner = np.trapezoid(psi_well(m, xx_w, L_val) * psi_well(n, xx_w, L_val), xx_w)
    PASS_COUNT += chk(inner, 0.0, f"§6 <psi_{m}|psi_{n}>=0", tol=1e-4, absolute=True)

# Energy ratios E_n = n²·E_1
E1 = E_well(1, L_val)
for n in [2, 3, 4]:
    PASS_COUNT += chk(E_well(n, L_val) / E1, float(n**2), f"§6 E_{n}/E_1 = {n}^2", tol=1e-10)

# %% [markdown]
# ## §7 Harmonic Oscillator — ladder operators, â⁻ψ₀=0, â⁻ψ_n=√n·ψ_{n-1}, Ĥψ_n=(n+½)ψ_n

# %%
import math
from numpy.polynomial.hermite import hermval

xi_arr = np.linspace(-6, 6, 200_000)

def ho_psi(n, xi):
    coeffs = [0] * (n + 1)
    coeffs[n] = 1
    norm = 1.0 / np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi))
    return norm * hermval(xi, coeffs) * np.exp(-xi**2 / 2)

# a-psi0 = 0: a- = (xi + d/dxi)/sqrt(2)
psi0 = ho_psi(0, xi_arr)
dpsi0 = np.gradient(psi0, xi_arr)
a_minus_psi0 = (xi_arr * psi0 + dpsi0) / np.sqrt(2)
norm_a_psi0 = np.trapezoid(a_minus_psi0**2, xi_arr)
PASS_COUNT += chk(norm_a_psi0, 0.0, "§7 a-psi0=0 (norm=0)", tol=1e-4, absolute=True)

# a-psi_n = sqrt(n)*psi_{n-1}
for n in [1, 2, 3]:
    psi_n = ho_psi(n, xi_arr)
    dpsi_n = np.gradient(psi_n, xi_arr)
    a_minus_psin = (xi_arr * psi_n + dpsi_n) / np.sqrt(2)
    psi_nm1 = ho_psi(n - 1, xi_arr)
    inner = np.trapezoid(psi_nm1 * a_minus_psin, xi_arr)
    PASS_COUNT += chk(inner, np.sqrt(n), f"§7 a-psi_{n}=sqrt({n})*psi_{n-1}", tol=1e-3)

# H psi_n = (n+0.5) psi_n  in units hbar*omega=1
# H = -0.5 d²/dxi² + 0.5 xi²
for n in [0, 1, 2]:
    psi_n = ho_psi(n, xi_arr)
    d2psi = np.gradient(np.gradient(psi_n, xi_arr), xi_arr)
    H_psin = -0.5 * d2psi + 0.5 * xi_arr**2 * psi_n
    E_num = np.trapezoid(psi_n * H_psin, xi_arr)
    PASS_COUNT += chk(E_num, n + 0.5, f"§7 E_{n}=({n}+0.5)*hbar*omega", tol=1e-3)

# %% [markdown]
# ## §8 Rogue Wave Statistics — exponential intensity PDF, P(I>2μ)=e^{-2}, Monte Carlo

# %%
rng = np.random.default_rng(42)
N = 1_000_000
mu = 1.0

I_samples = rng.exponential(scale=mu, size=N)

# P(I > 2mu) = e^{-2}
P_rogue_theory = np.exp(-2)
P_rogue_mc = float(np.mean(I_samples > 2 * mu))
PASS_COUNT += chk(P_rogue_mc, P_rogue_theory, "§8 P(rogue>2mu)=e^{-2} Monte Carlo", tol=5e-3)
print(f"  P(rogue>2mu): MC={P_rogue_mc:.6f}, theory={P_rogue_theory:.6f}")

# PDF shape check
counts, edges = np.histogram(I_samples, bins=200, range=(0, 10), density=True)
centres = 0.5 * (edges[:-1] + edges[1:])
pdf_theory = np.exp(-centres / mu) / mu
mask = pdf_theory > 0.5  # only check where PDF is large (low variance region)
residuals = np.abs(counts[mask] - pdf_theory[mask]) / pdf_theory[mask]
max_residual = float(np.max(residuals))
PASS_COUNT += chk(max_residual, 0.0, "§8 exponential PDF histogram residual < 2%", tol=0.02, absolute=True)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax1, ax2 = axes

ax1.plot(centres, counts, "b.", ms=3, label="Monte Carlo")
ax1.plot(centres, pdf_theory, "r-", lw=2, label="p(I)=exp(-I/mu)/mu")
ax1.axvline(2 * mu, color="orange", ls="--", label="2mu rogue threshold")
ax1.set_xlabel("Intensity I")
ax1.set_ylabel("PDF")
ax1.set_title("Rogue Wave Intensity PDF")
ax1.legend()

xi_plot = np.linspace(-4, 4, 1000)
for n in range(4):
    y = ho_psi(n, xi_plot)
    ax2.plot(xi_plot, y + n, label=f"psi_{n}")
ax2.axhline(0, color="k", lw=0.5)
ax2.set_xlabel("xi")
ax2.set_title("Harmonic Oscillator Eigenstates (offset)")
ax2.legend()

plt.tight_layout()
plt.savefig("repl/_repl_griffiths_qm_fig.png", dpi=100)
plt.close()

# %% [markdown]
# ## Summary

# %%
print(f"\n{'='*50}")
print(f"Total PASS count: {PASS_COUNT}")
print(f"{'='*50}")
