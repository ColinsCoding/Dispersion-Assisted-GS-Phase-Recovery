# %% [markdown]
# # Griffiths Quantum Mechanics + Dirac Delta — Between the Lines
# `init_printing(use_latex="mathjax")` throughout.
#
# **What this covers**: every place Griffiths writes "it can be shown" or
# uses a δ identity without proof. Every Taylor expansion he applies to an
# operator. Every symmetry argument he makes faster than you can follow.
#
# **The thread**: symmetry is more fundamental than geometry.
# Geometry tells you what space looks like.
# Symmetry tells you what is conserved — and conservation is what makes
# physics computable. Noether's theorem is the deepest fact in all of physics.
#
# **Structure:**
# §1  Continuous Fourier transform — the one Griffiths uses (not DFT)
# §2  Dirac delta in Griffiths: orthonormality of momentum eigenstates
# §3  Dirac delta: completeness relation  ⟨x|x′⟩ = δ(x−x′)
# §4  Taylor series in QM: e^{iHt/ℏ}, translation operator, BCH
# §5  Momentum operator from translation symmetry
# §6  Noether's theorem: symmetry → conservation law (the real reason)
# §7  Uncertainty principle from Fourier — not statistics, analysis
# §8  Griffiths Chapter 2: infinite square well — all the δ moves
# §9  Griffiths Chapter 3: Dirac notation, Hermitian operators, spectra
# §10 Dispersive wave packets: group velocity, phase velocity, spreading

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import sympy as sp
from sympy import (
    symbols, DiracDelta, integrate, diff, exp, sqrt, pi, cos, sin,
    oo, Eq, simplify, I, conjugate, Abs, Rational, ln, factorial,
    fourier_transform, inverse_fourier_transform, Function,
    Heaviside, limit, series, Matrix, eye, re, im
)
from sympy import init_printing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

init_printing(use_latex='mathjax')

try:
    from IPython.display import display as _D, Math, Markdown
    def show(expr, label=None):
        if label: _D(Markdown(f"**{label}**"))
        _D(expr)
    def tex(s): _D(Math(s))
except ImportError:
    def show(expr, label=None):
        if label: print(f"\n  {label}")
        print("  " + sp.pretty(expr, use_unicode=True))
    def tex(s): print(f"  [{s}]")

def hdr(s):
    bar = '─' * 64
    print(f'\n{bar}\n  {s}\n{bar}')

def chk(val, ref, label, tol=1e-8, absolute=False):
    try: v, r = float(val), float(ref)
    except: print(f'  [FAIL]  {label}  (not float)'); return
    err = abs(v-r) if (absolute or r==0) else abs(v-r)/(abs(r)+1e-30)
    print(f"  [{'PASS' if err<tol else 'FAIL'}]  {label}  got={v:.8g}  ref={r:.8g}")

print("=== Griffiths QM + Dirac Delta: Between the Lines ===")

# %% [markdown]
# ---
# ## §1 · The Continuous Fourier Transform — What Griffiths Actually Uses
#
# Griffiths uses the **symmetric** convention (physics standard):
#
# $$\tilde{\psi}(k) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\psi(x)\,e^{-ikx}\,dx$$
#
# $$\psi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\tilde{\psi}(k)\,e^{ikx}\,dk$$
#
# **Why symmetric**: makes Parseval's theorem clean:
# $$\int|\psi|^2\,dx = \int|\tilde{\psi}|^2\,dk$$
#
# **NOT the DFT**: this is a continuous integral over all of ℝ.
# The DFT sums N terms. The CFT is a limit N→∞ with Δk→0.
# In QM, k is continuous (free particle) or discrete (bound state).
#
# **Connection to momentum**: p = ℏk → k = p/ℏ. So ψ̃(k) is the
# momentum-space wavefunction φ(p) = ψ̃(p/ℏ)/√ℏ.

# %%
hdr("§1 — Continuous Fourier transform: Griffiths convention")

x, k, p, t_sym, a_sym = symbols('x k p t a', real=True)
hbar = symbols('hbar', positive=True)

tex(r"\tilde{\psi}(k) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\psi(x)\,e^{-ikx}\,dx")
tex(r"\psi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\tilde{\psi}(k)\,e^{ikx}\,dk")
tex(r"\int_{-\infty}^{\infty}|\psi|^2\,dx = \int_{-\infty}^{\infty}|\tilde\psi|^2\,dk \quad\text{(Parseval)}")

# Gaussian wavepacket: ψ(x) = (1/πa²)^{1/4} e^{ik₀x} e^{-x²/2a²}
# FT: ψ̃(k) = (a²/π)^{1/4} e^{-a²(k-k₀)²/2}
a_pos = symbols('a', positive=True)
k0 = symbols('k_0', real=True)

psi_gauss = (1/(sp.pi*a_pos**2))**sp.Rational(1,4) * exp(I*k0*x) * exp(-x**2/(2*a_pos**2))
psi_tilde = 1/sqrt(2*pi) * integrate(psi_gauss * exp(-I*k*x), (x, -oo, oo))
psi_tilde_s = simplify(psi_tilde)
show(psi_tilde_s, "ψ̃(k) for Gaussian wavepacket:")

# Verify Parseval
norm_x  = integrate(Abs(psi_gauss)**2, (x, -oo, oo))
norm_k  = integrate(Abs(psi_tilde_s)**2, (k, -oo, oo))
show(Eq(sp.Symbol('∫|ψ|²dx'), simplify(norm_x)), "Normalisation in x-space:")
show(Eq(sp.Symbol('∫|ψ̃|²dk'), simplify(norm_k)), "Normalisation in k-space:")
chk(float(simplify(norm_x - norm_k)), 0,
    "Parseval: ∫|ψ|²dx = ∫|ψ̃|²dk", tol=1e-8, absolute=True)

# Width product: Δx · Δk = 1/2  (Gaussian saturates uncertainty principle)
# σ_x = a/√2,  σ_k = 1/(a√2)
sigma_x = a_pos / sqrt(2)
sigma_k = 1/(a_pos * sqrt(2))
delta_x_delta_k = simplify(sigma_x * sigma_k)
show(Eq(sp.Symbol('Δx·Δk'), delta_x_delta_k), "Width product Δx·Δk (Gaussian):")
chk(float(delta_x_delta_k), 0.5, "Gaussian saturates: Δx·Δk = 1/2")

# %% [markdown]
# ---
# ## §2 · Orthonormality of Momentum Eigenstates — Griffiths Eq. 3.32
#
# Momentum eigenstates $f_p(x) = \frac{1}{\sqrt{2\pi\hbar}}\,e^{ipx/\hbar}$ satisfy:
#
# $$\langle f_{p'}|f_p\rangle = \int_{-\infty}^{\infty} f_{p'}^*(x)\,f_p(x)\,dx
#   = \delta(p-p')$$
#
# **Proof**: the integral is
# $$\frac{1}{2\pi\hbar}\int e^{i(p-p')x/\hbar}\,dx = \delta(p-p')$$
# using the Fourier representation of δ:
# $$\delta(\xi) = \frac{1}{2\pi}\int_{-\infty}^{\infty} e^{i\xi x}\,dx$$
#
# This is the most important δ identity in all of QM.

# %%
hdr("§2 — Momentum eigenstate orthonormality: ⟨p′|p⟩ = δ(p−p′)")

tex(r"\langle f_{p'}|f_p\rangle = \frac{1}{2\pi\hbar}\int e^{i(p-p')x/\hbar}dx = \delta(p-p')")
tex(r"\delta(\xi) = \frac{1}{2\pi}\int_{-\infty}^{\infty}e^{i\xi x}dx \quad\text{(Fourier rep of }\delta\text{)}")

p_sym, pp_sym = symbols('p p_prime', real=True)
hbar_n = 1.0  # natural units

# Fourier representation of delta: ∫e^{iξx}dx/(2π) = δ(ξ)
# Verify via Gaussian nascent delta:
xi_sym = symbols('xi', real=True)
eps_p = symbols('eps', positive=True)

# As ε→0: (1/2π)∫e^{iξx} e^{-εx²/2}dx = (1/√(2πε)) e^{-ξ²/2ε} → δ(ξ)
regulator = 1/(2*pi) * integrate(exp(I*xi_sym*x) * exp(-eps_p*x**2/2), (x,-oo,oo))
regulator_s = simplify(regulator)
show(Eq(sp.Symbol('I_ε(ξ)'), regulator_s), "Regulated Fourier δ:")

# Verify: as ε→0, acts as δ (unit area, unit sifting)
norm_reg = simplify(integrate(regulator_s, (xi_sym,-oo,oo)))
chk(float(norm_reg.evalf()), 1.0, "∫I_ε(ξ)dξ = 1 for all ε")

# At specific p-p' values (numerical verification via oscillatory integral limit)
# Use the identity: ∫_{-L}^{L} e^{iξx}dx/(2π) → δ(ξ) as L→∞
for xi_val in [0.0, 1.0, 2.0]:
    # As a distribution: acts on test function φ via sifting
    phi = exp(-xi_sym**2)  # test function
    if xi_val == 0.0:
        # ∫δ(ξ)φ(ξ)dξ = φ(0) = 1
        # Verify via the regulated version at ε=0.001
        r = float(integrate(
            1/(2*pi) * integrate(exp(I*xi_sym*x)*exp(-0.001*x**2/2),(x,-oo,oo))
            * phi, (xi_sym,-oo,oo)).evalf())
        chk(r, float(phi.subs(xi_sym,0).evalf()),
            "Fourier δ sifts φ at ξ=0", tol=0.01)

# Numerical: two different momenta → overlap → 0 (as distribution: via Gaussian envelope)
# Use ∫e^{iξx}e^{-εx²}dx/(2π) = e^{-ξ²/(4ε)}/(2√(πε)) → 0 for ξ≠0 as ε→0
# Test with small ε=0.001
eps_n = 0.001
x_arr = np.linspace(-200, 200, 200000)
dx_n  = x_arr[1] - x_arr[0]
for dp in [1.0, 2.0, 5.0]:
    integrand = np.exp(1j * dp * x_arr) * np.exp(-eps_n * x_arr**2) / (2*np.pi)
    result = np.abs(np.sum(integrand) * dx_n)
    theory = np.exp(-dp**2/(4*eps_n)) / (2*np.sqrt(np.pi*eps_n))
    chk(result, theory, f"(1/2π)∫e^{{i·{dp}·x-εx²}}dx ≈ 0 (p≠p', ε=0.001)",
        tol=1e-10, absolute=True)

# %% [markdown]
# ---
# ## §3 · Completeness: ⟨x|x′⟩ = δ(x−x′) and ∫|x⟩⟨x|dx = 1
#
# The position eigenstates $|x\rangle$ satisfy:
#
# $$\langle x'|x\rangle = \delta(x-x')$$
# $$\int_{-\infty}^{\infty}|x\rangle\langle x|\,dx = \hat{1}$$
#
# **What this means**: any state |ψ⟩ can be expanded:
# $$|\psi\rangle = \int\langle x|\psi\rangle\,|x\rangle\,dx = \int\psi(x)|x\rangle\,dx$$
#
# Inserting the completeness relation is the single most-used trick in QM.
# Whenever you see Griffiths write ⟨x|A|ψ⟩, he's inserting ∫|x'⟩⟨x'|dx' somewhere.

# %%
hdr("§3 — Completeness: ∫|x⟩⟨x|dx = 1̂ and the insertion trick")

tex(r"\langle x'|x\rangle = \delta(x-x')")
tex(r"\int |x\rangle\langle x|\,dx = \hat{1} \quad\Rightarrow\quad \langle x|\hat{p}|\psi\rangle = -i\hbar\,\frac{\partial\psi}{\partial x}")
tex(r"\langle x|\hat{p}^2|\psi\rangle = -\hbar^2\,\frac{\partial^2\psi}{\partial x^2}")

# The insertion: ⟨x|p|ψ⟩ = ∫⟨x|p|x'⟩⟨x'|ψ⟩dx'
# ⟨x|p|x'⟩ = -iℏ δ'(x-x')  (position-space matrix element of momentum)
# → ⟨x|p|ψ⟩ = ∫(-iℏ)δ'(x-x')ψ(x')dx' = -iℏ ψ'(x)  [δ' rule]
print("  Derivation of p̂ in position space:")
print("  ⟨x|p̂|ψ⟩ = ∫⟨x|p̂|x'⟩⟨x'|ψ⟩dx'")
print("          = ∫(-iℏ)δ'(x-x')ψ(x')dx'")
print("          = -iℏ ψ'(x)  [by δ' sifting rule ∫f δ'(x-x') dx' = f'(x)]")
print("  → p̂ = -iℏ d/dx  ✓")

# Verify: p̂ acting on Gaussian → correct momentum
psi_test = exp(-x**2/2) * exp(I*2*x)   # Gaussian with momentum k₀=2
p_on_psi = -I * diff(psi_test, x)      # ℏ=1 units
show(Eq(sp.Symbol('p̂ψ'), simplify(p_on_psi)), "p̂ψ = -i·d/dx ψ:")

# Expected value ⟨p⟩ = k₀ = 2 (momentum of centre of wavepacket)
psi_norm = integrate(Abs(psi_test)**2, (x,-oo,oo))
p_expect = integrate(conjugate(psi_test) * p_on_psi, (x,-oo,oo)) / psi_norm
show(Eq(sp.Symbol('⟨p⟩'), simplify(p_expect)), "⟨p⟩:")
chk(float(re(p_expect.evalf())), 2.0, "⟨p⟩ = k₀ = 2 (momentum of Gaussian)")
chk(float(Abs(im(p_expect)).evalf()), 0, "⟨p⟩ is real (Hermitian operator)", absolute=True)

# Completeness check: insert 1 = ∫|x⟩⟨x|dx → identity on any ψ
# ∫ψ*(x')δ(x-x')ψ(x')dx' = ψ*(x)ψ(x) = |ψ|²
phi_test = exp(-x**2)
for x_val in [0.0, 1.0, -0.5]:
    r = float(integrate(
        phi_test.subs(x, symbols('xp')) * DiracDelta(x - symbols('xp')) *
        phi_test.subs(x, symbols('xp')),
        (symbols('xp'), -oo, oo)).subs(x, x_val).evalf())
    ref = float(phi_test.subs(x,x_val)**2)
    chk(r, ref, f"completeness: ∫|ψ(x')|²δ(x-x')dx' = |ψ(x)|² at x={x_val}")

# %% [markdown]
# ---
# ## §4 · Taylor Series in QM — Operator Expansions
#
# **Time evolution operator**:
# $$\hat{U}(t) = e^{-i\hat{H}t/\hbar} = \sum_{n=0}^{\infty}\frac{1}{n!}\left(\frac{-i\hat{H}t}{\hbar}\right)^n$$
#
# **Translation operator**:
# $$\hat{T}(a)\psi(x) = \psi(x+a) = e^{a\,\partial/\partial x}\psi(x) = \sum_{n=0}^\infty \frac{a^n}{n!}\frac{d^n\psi}{dx^n}$$
#
# **Baker-Campbell-Hausdorff (BCH)**:
# $$e^A e^B = e^{A+B+[A,B]/2+\cdots}$$
# If [A, B] = c (a number): $e^A e^B = e^{A+B}\,e^{c/2}$
#
# **Why it matters**: [x̂, p̂] = iℏ. So the commutator is a number,
# and BCH gives exact results for position-momentum operations.

# %%
hdr("§4 — Taylor series in QM: U(t)=e^{-iHt/ℏ}, translation, BCH")

tex(r"\hat{U}(t) = e^{-i\hat{H}t/\hbar} = \sum_{n=0}^{\infty}\frac{(-i\hat{H}t/\hbar)^n}{n!}")
tex(r"\hat{T}(a)\psi(x) = e^{a\partial_x}\psi = \psi(x+a) = \sum_{n=0}^\infty\frac{a^n}{n!}\psi^{(n)}(x)")
tex(r"[\hat{x},\hat{p}] = i\hbar \;\Rightarrow\; e^{\hat{x}}e^{\hat{p}} = e^{\hat{x}+\hat{p}}\,e^{i\hbar/2}")

# Verify Taylor series of translation:  ψ(x+a) = e^{a·d/dx}ψ(x)
psi_T = sp.sin(x) + x**3   # test function
a_val_n = 0.3

# Exact: psi(x+a)
psi_shifted_exact = psi_T.subs(x, x + a_val_n)

# Taylor series up to N=10 terms
N_taylor = 10
psi_taylor = sum(a_val_n**n / factorial(n) * diff(psi_T, x, n)
                 for n in range(N_taylor+1))
psi_taylor_s = simplify(psi_taylor - psi_shifted_exact)

# Check at several x values
print(f"  Translation ψ(x+{a_val_n}) = e^(a·d/dx)ψ, N={N_taylor} terms:")
for x_v in [0.0, 0.5, 1.0, -0.5]:
    exact_v = float(psi_shifted_exact.subs(x, x_v).evalf())
    taylor_v = float(psi_taylor.subs(x, x_v).evalf())
    chk(taylor_v, exact_v, f"Taylor translation at x={x_v}", tol=1e-8)

# BCH: [x̂, p̂] = iℏ  →  e^{αx}e^{βp} = e^{αx+βp+αβ[x,p]/2} = e^{αx+βp}·e^{iℏαβ/2}
# Verify numerically in matrix representation (finite basis)
hbar_n = 1.0
N_dim = 20   # basis size
# Create x̂ and p̂ in harmonic oscillator basis
a_op  = np.diag(np.sqrt(np.arange(1, N_dim)), 1)    # lowering
ad_op = a_op.T                                         # raising
x_op  = (a_op + ad_op) / np.sqrt(2)                   # position (units where ω=m=ℏ=1)
p_op  = 1j * (ad_op - a_op) / np.sqrt(2)              # momentum

# Commutator [x̂, p̂] = iℏ·1
comm = x_op @ p_op - p_op @ x_op
expected_comm = 1j * hbar_n * np.eye(N_dim)
chk(np.max(np.abs(comm - expected_comm)[:-2,:-2]), 0,
    "[x̂, p̂] = iℏ·1̂  (matrix rep, interior)", tol=0.01, absolute=True)

# BCH: e^{αx}e^{βp} = e^{αx+βp}·e^{iℏαβ/2}
from scipy.linalg import expm as mat_exp
alpha, beta = 0.1, 0.2
lhs = mat_exp(alpha*x_op) @ mat_exp(beta*p_op)
rhs = mat_exp(alpha*x_op + beta*p_op) * np.exp(1j*hbar_n*alpha*beta/2)
chk(np.max(np.abs(lhs - rhs)[:-3,:-3]), 0,
    "BCH: e^{αx}e^{βp} = e^{αx+βp}·e^{iℏαβ/2}", tol=0.01, absolute=True)

# Time evolution: |ψ(t)⟩ = e^{-iHt/ℏ}|ψ(0)⟩
# Harmonic oscillator: H = ℏω(n̂ + 1/2)
omega_n = 1.0
H_op = hbar_n * omega_n * (np.diag(np.arange(N_dim)) + 0.5*np.eye(N_dim))

# Initial state: coherent state |α=1⟩ (Gaussian wavepacket)
alpha_coh = 1.0
n_arr = np.arange(N_dim)
coh_state = np.exp(-abs(alpha_coh)**2/2) * alpha_coh**n_arr / np.sqrt(np.array([float(sp.factorial(n)) for n in range(N_dim)]))
coh_state /= np.linalg.norm(coh_state)

t_test = np.pi / 2  # quarter period
U_t = mat_exp(-1j * H_op * t_test / hbar_n)
psi_t = U_t @ coh_state

# After T=2π/ω, state returns to initial (up to phase)
T_period = 2*np.pi/omega_n
U_T = mat_exp(-1j * H_op * T_period / hbar_n)
psi_T_state = U_T @ coh_state
overlap = abs(np.dot(np.conj(coh_state), psi_T_state))
chk(overlap, 1.0, "U(T)|ψ⟩ = e^{iφ}|ψ⟩ (period T=2π/ω, |overlap|=1)", tol=0.01)

# %% [markdown]
# ---
# ## §5 · Momentum from Translation Symmetry
#
# **The deep point Griffiths makes implicitly**:
# Momentum is not a fundamental quantity — it is the *generator* of spatial translations.
#
# The translation operator by ε:
# $$\hat{T}(\varepsilon)\psi(x) = \psi(x+\varepsilon)
#   = \left(1 + \varepsilon\frac{\partial}{\partial x} + \cdots\right)\psi(x)
#   \approx \left(1 + \frac{i\varepsilon\hat{p}}{\hbar}\right)\psi(x)$$
#
# Therefore: $\hat{T}(\varepsilon) = e^{i\varepsilon\hat{p}/\hbar}$
# → comparing: $\hat{p} = -i\hbar\,\frac{\partial}{\partial x}$
#
# **Noether's theorem** (classical): if the Lagrangian is invariant under
# $x \to x + \varepsilon$, the conserved charge is:
# $$p = \frac{\partial\mathcal{L}}{\partial\dot{x}} = m\dot{x}$$
# In QM: the same symmetry → [H, p̂] = 0 → momentum is conserved.

# %%
hdr("§5 — Momentum = generator of translations (why p = -iℏ∂/∂x)")

tex(r"\hat{T}(\varepsilon) = e^{i\varepsilon\hat{p}/\hbar} \;\Rightarrow\; \hat{p} = -i\hbar\frac{\partial}{\partial x}")
tex(r"\text{If }[H,\hat{p}]=0 \;\Leftrightarrow\; V(x)=\text{const} \;\Rightarrow\; \langle p\rangle\text{ conserved}")

# Verify: -iℏ d/dx (e^{ikx}) = ℏk · e^{ikx}  (momentum eigenstate)
k_val = 3.0; hbar_v = 1.0
f_k = exp(I * k_val * x)
p_f_k = -I * hbar_v * diff(f_k, x)
eigenvalue = simplify(p_f_k / f_k)
chk(float(eigenvalue), k_val, f"p̂ e^{{ikx}} = ℏk·e^{{ikx}}, k={k_val}")

# Check translation: T(ε)ψ = e^{iεp̂/ℏ}ψ for various ψ
eps_val = 0.2
for psi_expr, label in [(exp(-x**2), "Gaussian"), (sin(x), "sin x"), (x**2 + 1, "x²+1")]:
    # Taylor expansion of translation
    T_eps = sum(eps_val**n / factorial(n) * diff(psi_expr, x, n)
                for n in range(8))
    exact_shift = psi_expr.subs(x, x + eps_val)
    err = float(Abs(simplify(T_eps - exact_shift)).subs(x, 0.5).evalf())
    chk(err, 0, f"T(ε)·{label} = {label}(x+ε)  [8 terms]", tol=1e-5, absolute=True)

# Conservation: [H, p̂] = 0 for free particle V=0
# H = p²/2m → [p², p] = 0 trivially
# Check in matrix form: [H_free, p] where H_free = p²/2m
H_free = (p_op @ p_op) / 2
comm_Hp = H_free @ p_op - p_op @ H_free
chk(np.max(np.abs(comm_Hp[:-2,:-2])), 0,
    "[H_free, p̂] = 0 (free particle conserves p)", tol=1e-10, absolute=True)

# %% [markdown]
# ---
# ## §6 · Noether's Theorem — Symmetry IS More Important Than Geometry
#
# **Statement**: Every continuous symmetry of the action has a corresponding
# conserved quantity.
#
# | Symmetry | Conserved quantity |
# |----------|--------------------|
# | Time translation t→t+ε | Energy E |
# | Space translation x→x+ε | Momentum p |
# | Rotation θ→θ+ε | Angular momentum L |
# | Phase ψ→e^{iα}ψ | Electric charge Q |
# | Gauge A→A+∇χ | (more subtle) |
#
# **This is deeper than geometry**: You don't need to know the geometry of
# spacetime to derive conservation of energy. You just need the Lagrangian
# to be unchanged under time shifts.
#
# **In QM**: $[\hat{H}, \hat{A}] = 0 \Leftrightarrow$ A is conserved
# $\Leftrightarrow$ H has the symmetry generated by A.

# %%
hdr("§6 — Noether's theorem: symmetry → conservation (deepest fact)")

tex(r"\frac{d}{dt}\langle A\rangle = \frac{i}{\hbar}\langle[\hat{H},\hat{A}]\rangle + \left\langle\frac{\partial A}{\partial t}\right\rangle")
tex(r"[\hat{H},\hat{A}] = 0 \;\Leftrightarrow\; \frac{d}{dt}\langle A\rangle = 0 \;\Leftrightarrow\; A\text{ is conserved}")

# Ehrenfest theorem: d⟨x⟩/dt = ⟨p⟩/m,  d⟨p⟩/dt = -⟨dV/dx⟩
# Verify [H, x̂] = -iℏ p̂/m  (in units m=ℏ=1: [H,x] = -ip)
# H = p²/2 + V(x), [H,x] = [p²/2, x] = p[p,x]/2 + [p,x]p/2
# [p,x] = -iℏ → [p²/2, x] = -iℏp  (in ℏ=1 units: = -ip)
H_ho = (p_op @ p_op)/2 + (x_op @ x_op)/2   # harmonic oscillator
comm_Hx = H_ho @ x_op - x_op @ H_ho
expected_comm_Hx = -1j * p_op                # [H,x] = -ip (ℏ=m=1)
chk(np.max(np.abs(comm_Hx - expected_comm_Hx)[:-2,:-2]), 0,
    "[H, x̂] = -ip̂/m  (Ehrenfest: d⟨x⟩/dt = ⟨p⟩/m)", tol=0.01, absolute=True)

comm_Hp_ho = H_ho @ p_op - p_op @ H_ho
expected_comm_Hp = 1j * x_op                 # [H,p] = +ix (harmonic oscillator, ω=k=1)
chk(np.max(np.abs(comm_Hp_ho - expected_comm_Hp)[:-2,:-2]), 0,
    "[H_HO, p̂] = +ix̂  (Ehrenfest: d⟨p⟩/dt = -⟨x⟩ = -⟨dV/dx⟩)", tol=0.01, absolute=True)

# Energy conservation: [H, H] = 0 always
chk(np.max(np.abs(H_ho @ H_ho - H_ho @ H_ho)), 0,
    "[H, H] = 0 (energy always conserved)", tol=1e-15, absolute=True)

# Angular momentum: [Lz, H_spherical] = 0 for spherically symmetric V
# Verify in 2D: Lz = xpy - ypx, [Lz, x²+y²] = 0
print("\n  Angular momentum [Lz, r²] = 0 for rotationally symmetric H:")
x_s, y_s, px_s, py_s = symbols('x_s y_s px py', real=True)
Lz = x_s*py_s - y_s*px_s  # as differential operator action
r2 = x_s**2 + y_s**2
# [Lz, r²]f = Lz(r²f) - r²(Lz f)
f_test = Function('f')(x_s, y_s)
Lz_op = lambda g: x_s*diff(g,y_s) - y_s*diff(g,x_s)
comm_Lz_r2 = simplify(Lz_op(r2 * f_test) - r2 * Lz_op(f_test))
chk(float(comm_Lz_r2.subs(f_test,1).evalf()), 0, "[Lz, r²] = 0", absolute=True)

# %% [markdown]
# ---
# ## §7 · Uncertainty Principle from Fourier — Not Statistics
#
# The Heisenberg uncertainty principle is a theorem about Fourier transforms,
# not a statement about measurement disturbing the system.
#
# **Robertson uncertainty relation**:
# $$\sigma_A\sigma_B \geq \frac{1}{2}|\langle[\hat{A},\hat{B}]\rangle|$$
#
# **For x and p**:
# $$\sigma_x\sigma_p \geq \frac{\hbar}{2}$$
#
# **Fourier proof** (no operators needed):
# For any square-integrable f, the Fourier width product satisfies
# $\Delta x \cdot \Delta k \geq 1/2$.
# With p = ℏk: $\sigma_x\sigma_p \geq \hbar/2$.
# Equality iff f is a Gaussian — the **minimum uncertainty state**.

# %%
hdr("§7 — Uncertainty principle = Fourier theorem, not measurement")

tex(r"\sigma_x\sigma_p \geq \frac{\hbar}{2} \quad\text{(Fourier width theorem)}")
tex(r"\text{Equality iff }\psi\text{ is Gaussian: }\psi = Ae^{ikx}e^{-x^2/4\sigma_x^2}")

a_val = symbols('a', positive=True)

# Compute σ_x and σ_p for general Gaussian ψ = (2πa²)^{-1/4} e^{ik₀x} e^{-x²/4a²}
psi_g = (2*pi*a_val**2)**sp.Rational(-1,4) * exp(I*k0*x) * exp(-x**2/(4*a_val**2))
psi_g_norm = integrate(Abs(psi_g)**2, (x,-oo,oo))

# σ_x² = ⟨x²⟩ - ⟨x⟩²
x_expect = integrate(Abs(psi_g)**2 * x, (x,-oo,oo))
x2_expect = integrate(Abs(psi_g)**2 * x**2, (x,-oo,oo))
sigma_x_sq = simplify(x2_expect - x_expect**2)
sigma_x_sym = simplify(sqrt(sigma_x_sq))
show(Eq(sp.Symbol('σ_x'), sigma_x_sym), "σ_x:")
chk(float(sigma_x_sym.subs(a_val,1).evalf()), 1.0, "σ_x = a at a=1")

# σ_p in ℏ=1 units: p̂ = -i d/dx
p_psi = -I * diff(psi_g, x)
p_expect = simplify(integrate(conjugate(psi_g)*p_psi, (x,-oo,oo)))
p2_psi = -diff(psi_g, x, 2)
p2_expect = simplify(integrate(conjugate(psi_g)*p2_psi, (x,-oo,oo)))
sigma_p_sq = simplify(p2_expect - p_expect**2)
sigma_p_sym = simplify(sqrt(sigma_p_sq))
show(Eq(sp.Symbol('σ_p'), sigma_p_sym), "σ_p (ℏ=1):")

# Uncertainty product
UP = simplify(sigma_x_sym * sigma_p_sym)
show(Eq(sp.Symbol('σ_x·σ_p'), UP), "σ_x · σ_p =")
chk(float(UP.subs(a_val,1).evalf()), 0.5,
    "Gaussian: σ_x·σ_p = 1/2 = ℏ/2  (saturates UP)", tol=1e-6)

# Verify for several a values
for a_n in [0.5, 1.0, 2.0, 3.0]:
    prod = float(UP.subs(a_val, a_n).evalf())
    chk(prod, 0.5, f"σ_x·σ_p = ℏ/2 at a={a_n}")

# Numerical check: non-Gaussian violates equality (σ_x·σ_p > 1/2)
N_up = 10000
x_np = np.linspace(-10, 10, N_up)
dx_np = x_np[1] - x_np[0]

# Top-hat: ψ = 1/√(2L) for |x| < L
L_hat = 2.0
psi_hat = np.where(np.abs(x_np) < L_hat, 1/np.sqrt(2*L_hat), 0.0)
sigma_x_hat = np.sqrt(np.sum(x_np**2 * psi_hat**2) * dx_np)
psi_hat_k = np.fft.fftshift(np.fft.fft(psi_hat)) * dx_np / np.sqrt(2*np.pi)
k_np = np.fft.fftshift(np.fft.fftfreq(N_up, dx_np)) * 2*np.pi
dk_np = k_np[1]-k_np[0]
norm_k = np.sum(np.abs(psi_hat_k)**2) * dk_np
psi_hat_k_n = psi_hat_k / np.sqrt(norm_k)
k_expect_hat = np.sum(k_np * np.abs(psi_hat_k_n)**2) * dk_np
k2_expect_hat = np.sum(k_np**2 * np.abs(psi_hat_k_n)**2) * dk_np
sigma_p_hat = np.sqrt(max(k2_expect_hat - k_expect_hat**2, 0))
UP_hat = sigma_x_hat * sigma_p_hat
chk(UP_hat > 0.5, 1,
    f"top-hat: σ_x·σ_p = {UP_hat:.3f} > 1/2 (non-Gaussian violates equality)",
    tol=1e-9, absolute=True)
print(f"  top-hat σ_x·σ_p = {UP_hat:.4f}  (> 0.5 ✓)")

# %% [markdown]
# ---
# ## §8 · Infinite Square Well — All the δ Moves (Griffiths Ch. 2)
#
# The eigenstates:
# $$\psi_n(x) = \sqrt{\frac{2}{L}}\sin\!\left(\frac{n\pi x}{L}\right), \quad
#   E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$$
#
# **δ identities used by Griffiths in Chapter 2** (without proof):
#
# 1. **Orthonormality**: $\int_0^L\psi_m^*\psi_n\,dx = \delta_{mn}$
# 2. **Completeness**: $\sum_{n=1}^\infty\psi_n^*(x')\psi_n(x) = \delta(x-x')$
# 3. **Expansion coefficients**: $c_n = \int_0^L\psi_n^*(x)\Psi(x,0)\,dx$
#    (sifting the initial condition through the eigenbasis)

# %%
hdr("§8 — Infinite square well: orthonormality, completeness, expansion")

tex(r"\psi_n(x) = \sqrt{\frac{2}{L}}\sin\!\left(\frac{n\pi x}{L}\right)")
tex(r"\int_0^L\psi_m\psi_n\,dx = \delta_{mn} \quad\text{(orthonormality = δ Kronecker)}")
tex(r"\sum_n\psi_n(x)\psi_n(x') = \delta(x-x') \quad\text{(completeness = δ Dirac)}")

x_sym2 = symbols('x', real=True, positive=True)
L_sym = symbols('L', positive=True)

def psi_n(n, x_s, L_s):
    return sqrt(2/L_s) * sin(n*pi*x_s/L_s)

L_val = sp.pi   # so n²π² terms are clean

# Orthonormality
print("  Orthonormality ∫₀ᴸ ψₘ ψₙ dx = δₘₙ:")
for m_v in range(1, 5):
    for n_v in range(m_v, 5):
        r = float(integrate(
            psi_n(m_v, x_sym2, L_val) * psi_n(n_v, x_sym2, L_val),
            (x_sym2, 0, L_val)).evalf())
        ref = 1.0 if m_v == n_v else 0.0
        chk(r, ref, f"⟨ψ_{m_v}|ψ_{n_v}⟩ = {int(ref)}", tol=1e-8, absolute=True)

# Completeness: partial sum of ψₙ(x)ψₙ(x') → δ(x-x') as N→∞
# Test: ∫[Σₙψₙ(x)ψₙ(x')] f(x') dx' = f(x)
f_test_val = lambda xv: float(sp.sin(xv).evalf())
x0_test = float(L_val / 3)   # = π/3

N_complete = 50
x_arr2 = np.linspace(0.01, float(L_val)-0.01, 2000)
dx2 = x_arr2[1]-x_arr2[0]
f_arr = np.sin(x_arr2)  # test function

f_reconstructed = np.zeros_like(f_arr)
for n_v in range(1, N_complete+1):
    psi_n_arr = np.sqrt(2/float(L_val)) * np.sin(n_v*np.pi*x_arr2/float(L_val))
    c_n = np.sum(psi_n_arr * f_arr) * dx2
    f_reconstructed += c_n * psi_n_arr

chk(np.max(np.abs(f_reconstructed - f_arr)), 0,
    f"completeness: Σ|ψₙ⟩⟨ψₙ|·f = f  (N={N_complete})", tol=0.02, absolute=True)

# Expansion coefficients for Gaussian initial state
psi_0 = np.exp(-(x_arr2 - float(L_val)/2)**2 / 0.1)  # Gaussian centred at L/2
psi_0 /= np.sqrt(np.sum(psi_0**2)*dx2)  # normalise

c_ns = np.array([np.sum(np.sqrt(2/float(L_val))*np.sin(n_v*np.pi*x_arr2/float(L_val))*psi_0)*dx2
                 for n_v in range(1, 21)])
prob_total = np.sum(np.abs(c_ns)**2)
chk(prob_total, 1.0, "Σ|cₙ|² = 1  (Born rule normalisation)", tol=0.01)

# Time evolution: Ψ(x,t) = Σ cₙ ψₙ(x) e^{-iEₙt/ℏ}
hbar_m = 1.0; m_val = 0.5
L_num = float(L_val)
t_ev = 1.0
E_ns = np.array([(n_v*np.pi)**2*hbar_m**2/(2*m_val*L_num**2) for n_v in range(1,21)])
psi_ns_arr = np.array([np.sqrt(2/L_num)*np.sin(n_v*np.pi*x_arr2/L_num) for n_v in range(1,21)])

psi_t = np.sum(c_ns[:,None]*psi_ns_arr*np.exp(-1j*E_ns[:,None]*t_ev/hbar_m), axis=0)
norm_t = np.sum(np.abs(psi_t)**2)*dx2
chk(norm_t, 1.0, "|Ψ(x,t)|² normalised for all t (unitary evolution)", tol=0.01)

# %% [markdown]
# ---
# ## §9 · Dirac Notation Deep Cuts — What Griffiths Chapter 3 Assumes
#
# **The δ hidden in Dirac notation**:
#
# Every ket expansion $|\psi\rangle = \int \psi(x)|x\rangle\,dx$ contains a δ:
# when you compute $\langle x'|\psi\rangle$, you get
# $\int\psi(x)\langle x'|x\rangle\,dx = \int\psi(x)\delta(x-x')\,dx = \psi(x')$.
#
# The Hermitian adjoint satisfies:
# $$\langle\phi|\hat{A}|\psi\rangle = \langle\hat{A}^\dagger\phi|\psi\rangle^*$$
#
# For p̂ = −iℏ∂/∂x: p̂† = p̂ (Hermitian).
# Proof: integrate by parts, boundary terms vanish for square-integrable ψ.

# %%
hdr("§9 — Dirac notation: the δ Griffiths hides, Hermitian proof")

tex(r"\langle x'|\psi\rangle = \int\psi(x)\delta(x-x')\,dx = \psi(x')")
tex(r"\langle\phi|\hat{p}|\psi\rangle = \langle\hat{p}\phi|\psi\rangle^* \;\Rightarrow\; \hat{p}^\dagger = \hat{p}")

# Verify p̂ is Hermitian: ⟨φ|p̂ψ⟩ = ⟨p̂φ|ψ⟩*
phi_test = exp(-x**2/2) * cos(x)
psi_test2 = exp(-x**2/2) * sin(2*x)

# Hermitian: ⟨φ|p̂ψ⟩ = ⟨p̂φ|ψ⟩  means ∫φ*(p̂ψ)dx = ∫(p̂φ)*ψ dx
lhs = integrate(conjugate(phi_test) * (-I*diff(psi_test2,x)), (x,-oo,oo))
rhs = integrate(conjugate(-I*diff(phi_test,x)) * psi_test2, (x,-oo,oo))
lhs_n = complex(lhs.evalf())
rhs_n = complex(rhs.evalf())
chk(abs(lhs_n - rhs_n), 0,
    "⟨φ|p̂ψ⟩ = ⟨p̂φ|ψ⟩  (p̂ Hermitian: ∫φ*(p̂ψ) = ∫(p̂φ)*ψ)", tol=1e-8, absolute=True)
chk(abs(lhs_n.real), 0, "⟨φ|p̂ψ⟩ is purely imaginary (p odd operator)", tol=1e-6, absolute=True)

# Spectral theorem: H Hermitian → real eigenvalues
H_test = Matrix([[1,I],[-I,3]])   # Hermitian: H†=H
show(H_test, "H = [[1,i],[-i,3]]  (Hermitian):")
eigenvals_H = H_test.eigenvals()
print(f"  Eigenvalues: {list(eigenvals_H.keys())}")
for ev in eigenvals_H.keys():
    chk(float(im(ev)), 0, f"eigenvalue {ev} is real", tol=1e-10, absolute=True)

# Commutator → uncertainty via Robertson
# ΔA·ΔB ≥ |⟨[A,B]⟩|/2
# For [x̂,p̂] = iℏ: Δx·Δp ≥ ℏ/2
# Already verified numerically — show the algebra
print("\n  Robertson: ΔA·ΔB ≥ |⟨[A,B]⟩|/2")
print("  [x̂,p̂] = iℏ  →  Δx·Δp ≥ ℏ/2")
print("  Proof: Cauchy-Schwarz on (Â|ψ⟩)·(B̂|ψ⟩), A=x-⟨x⟩, B=p-⟨p⟩")

# %% [markdown]
# ---
# ## §10 · Dispersive Wave Packets — Continuous Fourier, Not Discrete
#
# A free-particle wave packet:
# $$\Psi(x,t) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}
#   \phi(k)\,e^{i(kx - \omega(k)t)}\,dk$$
#
# **Dispersion relation** for free particle: $\omega(k) = \hbar k^2/(2m)$
#
# **Group velocity** (where the packet travels):
# $$v_g = \frac{d\omega}{dk}\bigg|_{k=k_0} = \frac{\hbar k_0}{m} = \frac{p_0}{m}$$
#
# **Packet spreading**: the width grows as
# $$\sigma(t) = \sigma_0\sqrt{1 + \left(\frac{\hbar t}{2m\sigma_0^2}\right)^2}$$
# → inevitable from uncertainty principle.

# %%
hdr("§10 — Dispersive wave packet: group velocity, spreading, CFT")

tex(r"\Psi(x,t) = \frac{1}{\sqrt{2\pi}}\int\phi(k)\,e^{i(kx-\omega(k)t)}\,dk")
tex(r"\omega(k) = \frac{\hbar k^2}{2m},\quad v_g = \frac{d\omega}{dk}\bigg|_{k_0} = \frac{\hbar k_0}{m}")
tex(r"\sigma(t) = \sigma_0\sqrt{1+\left(\frac{\hbar t}{2m\sigma_0^2}\right)^2} \quad\text{(spreading)}")

hbar_wp = 1.0; m_wp = 1.0
k0_wp   = 5.0   # central momentum
sigma0  = 1.0   # initial width

# φ(k) = Gaussian centred at k₀ (→ ψ(x,0) is Gaussian)
N_wp = 2048
k_arr = np.linspace(-20, 30, N_wp)
dk_wp = k_arr[1]-k_arr[0]
phi_k = (2*np.pi*sigma0**2)**(-0.25) * np.exp(-(k_arr-k0_wp)**2/(4*sigma0**2/(hbar_wp**2)))
phi_k /= np.sqrt(np.sum(np.abs(phi_k)**2)*dk_wp)

# Dispersion: ω(k) = ℏk²/(2m)
omega_k = hbar_wp * k_arr**2 / (2*m_wp)

x_wp = np.linspace(-5, 30, N_wp)
dx_wp = x_wp[1]-x_wp[0]

# Group velocity
v_g = hbar_wp * k0_wp / m_wp
print(f"  k₀={k0_wp}, v_g = ℏk₀/m = {v_g:.2f}")

# Measure actual σ₀ from ψ(x,0) numerically (φ_k determines real σ₀)
psi_t0 = np.array([np.sum(phi_k*np.exp(1j*k_arr*xi))*dk_wp/np.sqrt(2*np.pi) for xi in x_wp])
prob_t0 = np.abs(psi_t0)**2
norm_t0 = np.sum(prob_t0)*dx_wp
xmean_t0 = np.sum(x_wp*prob_t0)*dx_wp/norm_t0
sigma_0_measured = np.sqrt(np.sum((x_wp-xmean_t0)**2*prob_t0)*dx_wp/norm_t0)
print(f"  measured σ(t=0) = {sigma_0_measured:.4f}")
sigma_theory = lambda t: sigma_0_measured * np.sqrt(1 + (hbar_wp*t/(2*m_wp*sigma_0_measured**2))**2)

# Evaluate ψ(x,t) at several times via numerical CFT
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
t_plot = [0.0, 0.2, 0.5, 1.0]

for ax, t_v in zip(axes, t_plot):
    # ψ(x,t) = (1/√2π) ∫ φ(k) e^{i(kx-ωt)} dk
    psi_xt = np.zeros(N_wp, dtype=complex)
    for i_x, xi in enumerate(x_wp):
        phase = np.exp(1j*(k_arr*xi - omega_k*t_v))
        psi_xt[i_x] = np.sum(phi_k * phase) * dk_wp / np.sqrt(2*np.pi)

    prob = np.abs(psi_xt)**2
    ax.plot(x_wp, prob, 'C0', linewidth=2)
    peak_pos = x_wp[np.argmax(prob)]
    ax.axvline(v_g*t_v, color='r', linestyle='--', alpha=0.7, label=f'v_g·t={v_g*t_v:.1f}')
    ax.set_title(f't={t_v}\npeak≈{peak_pos:.1f}, σ={sigma_theory(t_v):.2f}', fontsize=9)
    ax.legend(fontsize=7); ax.grid(True,alpha=0.3)
    ax.set_xlabel('x')

plt.suptitle('Free-particle wave packet: dispersive spreading (CFT, not DFT)', fontsize=10)
plt.tight_layout()
plt.savefig('repl/_fig_wave_packet.png', dpi=110, bbox_inches='tight')
plt.close()
print("  Saved: repl/_fig_wave_packet.png")

# Verify: peak tracks v_g*t
for t_v in [0.2, 0.5]:
    psi_xt_v = np.array([
        np.sum(phi_k * np.exp(1j*(k_arr*xi - omega_k*t_v))) * dk_wp / np.sqrt(2*np.pi)
        for xi in x_wp])
    peak = x_wp[np.argmax(np.abs(psi_xt_v)**2)]
    chk(peak, v_g*t_v, f"peak at v_g·t={v_g*t_v:.1f}  at t={t_v}", tol=0.5)

# Verify spreading
for t_v in [0.5, 1.0]:
    psi_xt_v = np.array([
        np.sum(phi_k*np.exp(1j*(k_arr*xi - omega_k*t_v)))*dk_wp/np.sqrt(2*np.pi)
        for xi in x_wp])
    prob_v = np.abs(psi_xt_v)**2
    norm_v = np.sum(prob_v)*dx_wp
    x_mean = np.sum(x_wp*prob_v)*dx_wp/norm_v
    sigma_meas = np.sqrt(np.sum((x_wp-x_mean)**2*prob_v)*dx_wp/norm_v)
    sigma_th = sigma_theory(t_v)
    chk(sigma_meas, sigma_th, f"σ(t={t_v}) measured={sigma_meas:.3f} vs theory={sigma_th:.3f}",
        tol=0.15)

# Parseval at t=0: normalisation preserved
psi_0_wp = np.array([np.sum(phi_k*np.exp(1j*k_arr*xi))*dk_wp/np.sqrt(2*np.pi) for xi in x_wp])
norm_x0 = np.sum(np.abs(psi_0_wp)**2)*dx_wp
chk(norm_x0, 1.0, "∫|ψ(x,0)|²dx = 1 (CFT preserves norm, Parseval)", tol=0.02)

hdr("Done — Griffiths QM + Dirac Delta: Between the Lines")
print("  §1 CFT  §2 momentum eigenstates  §3 completeness  §4 Taylor/BCH")
print("  §5 p=generator  §6 Noether  §7 uncertainty=Fourier  §8 square well")
print("  §9 Dirac notation  §10 dispersive wave packet")
