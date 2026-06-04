"""
repl/_repl_energy_lagrangian.py
Work-energy theorem, Lagrangian mechanics, CUDA torch verification.
"""
import numpy as np
import sympy as sp
import torch
sp.init_printing(use_unicode=False, wrap_line=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print()

# ============================================================
# 1. Work-energy theorem (SymPy)
# ============================================================
print("=== Work-Energy Theorem ===")
x, v, t_s, m_s, F_s, k_s = sp.symbols('x v t m F k', positive=True)

# W = integral F dx = delta KE
# For constant F:
W  = sp.Symbol('W', real=True)
KE = sp.Rational(1,2) * m_s * v**2

print("KE = (1/2)*m*v^2:", sp.pretty(KE))
dKE = sp.diff(KE, v)
print("dKE/dv = m*v (momentum):", sp.pretty(dKE))
print()

# Spring: F = -kx, W = -integral(kx dx) = -kx^2/2 = -PE
PE_spring = sp.Rational(1,2) * k_s * x**2
F_spring  = -sp.diff(PE_spring, x)
print("Spring PE = (1/2)kx^2:", sp.pretty(PE_spring))
print("F = -dPE/dx =", sp.pretty(F_spring))
print("Total E = KE + PE = constant (conservative force)")
print()

# ============================================================
# 2. Lagrangian: L = KE - PE
# ============================================================
print("=== Lagrangian Mechanics ===")
print("""
L(q, q_dot, t) = T - V     T=kinetic, V=potential

Euler-Lagrange equation:
  d/dt (dL/d(q_dot)) - dL/dq = 0

Harmonic oscillator: q=x, T=(1/2)m*x_dot^2, V=(1/2)k*x^2
""")

q, q_d = sp.symbols('q q_dot', real=True)
m_L, k_L = sp.symbols('m k', positive=True)

T_L = sp.Rational(1,2) * m_L * q_d**2
V_L = sp.Rational(1,2) * k_L * q**2
L_osc = T_L - V_L

dL_dq     = sp.diff(L_osc, q)
dL_dqdot  = sp.diff(L_osc, q_d)

print("L =", sp.pretty(L_osc))
print("dL/dq =", sp.pretty(dL_dq))
print("dL/d(q_dot) =", sp.pretty(dL_dqdot))
print("EL eq: d/dt(m*q_dot) - (-k*q) = 0  =>  m*q_ddot + k*q = 0")
print("=> omega_0 = sqrt(k/m)")
print()

# Pendulum
theta, g_s, l_s = sp.symbols('theta g l', positive=True)
T_pend = sp.Rational(1,2) * m_L * l_s**2 * theta**2
V_pend = m_L * g_s * l_s * (1 - sp.cos(theta))
L_pend = T_pend - V_pend

EL_pend = sp.diff(sp.diff(L_pend, theta), theta) - sp.diff(L_pend, theta)
print("Pendulum EL equation:", sp.pretty(sp.simplify(EL_pend)))
print("Small angle sin(theta)~theta: omega = sqrt(g/l)")
print()

# ============================================================
# 3. Lagrangian -> GS connection
# ============================================================
print("=== Lagrangian view of GS ===")
print("""
GS minimizes a functional (action):

  S[phi] = ||  |H1*exp(i*phi)|^2 - I1  ||^2
         + ||  |H2*exp(i*phi)|^2 - I2  ||^2

Since |H1*exp(i*phi)|^2 = 1 for unit-amplitude signal:
  S[phi] = ||1 - I1||^2 + ||1 - I2||^2   <- trivially zero if I1=I2=1

For non-unit amplitude (OOK, PAM4):
  S[phi] = ||sqrt(I1)*exp(i*angle(H1*E)) - H1*E||^2

GS iteration = gradient descent step on S[phi] with step size = 1
  (projection is exact, not approximate gradient)

Noether: S is invariant under phi -> phi + C  (global phase)
=> conserved quantity = total phase (unobservable)
=> GS always has a global phase ambiguity
""")

# ============================================================
# 4. CUDA torch: energy conservation in neural forward pass
# ============================================================
print("=== CUDA: energy conservation in FNO spectral layer ===")

N = 512
modes = 32

# simulate SpectralConv1d forward pass
x_in = torch.randn(1, 1, N, device=device)
E_in = float(torch.sum(x_in**2))

# FFT -> truncate to modes -> IFFT (what SpectralConv1d does)
X = torch.fft.rfft(x_in, dim=-1)
X_trunc = torch.zeros_like(X)
X_trunc[..., :modes] = X[..., :modes]
x_out = torch.fft.irfft(X_trunc, n=N, dim=-1)
E_out = float(torch.sum(x_out**2))

print(f"Input energy:    {E_in:.4f}")
print(f"Output energy:   {E_out:.4f}  (truncated to {modes} modes)")
print(f"Energy ratio:    {E_out/E_in:.4f}  <- fraction in first {modes} modes")
print(f"Parseval check:  {float(torch.sum(torch.abs(X)**2))/N:.4f} vs {E_in:.4f}")
print()

# sweep: how much energy is in the first M modes?
print("Energy fraction in first M spectral modes (Parseval decomposition):")
X_full = torch.fft.rfft(x_in, dim=-1)
E_total = float(torch.sum(torch.abs(X_full)**2)) / N

for M in [4, 8, 16, 32, 64, 128, 256]:
    E_M = float(torch.sum(torch.abs(X_full[..., :M])**2)) / N
    print(f"  M={M:4d}  fraction={E_M/E_total:.4f}  "
          f"{'[FNO default]' if M==32 else ''}")
print()

# work-energy in optimization: loss decrease per step
print("=== Work-energy in SGD: loss decrease per gradient step ===")
print("""
Gradient descent:  theta_{k+1} = theta_k - lr * grad(L)

Analogy:
  Force     F = -grad(L)    (negative gradient = restoring force)
  Work      W = F . delta_theta = -||grad||^2 * lr  (always negative = loss decreases)
  Power     P = dL/dt = -lr * ||grad||^2

For GS:
  No explicit gradient — projection does exact step
  Work per iteration = ||P_C1(E) - E||^2 + ||P_C2(E) - E||^2
  Decreases monotonically when constraint sets are convex
""")

# numerical demo: SGD loss curve as "power dissipation"
torch.manual_seed(0)
w = torch.randn(32, requires_grad=True, device=device)
target = torch.zeros(32, device=device)
opt = torch.optim.SGD([w], lr=0.1)

powers = []
for step in range(30):
    loss = ((w - target)**2).sum()
    opt.zero_grad()
    loss.backward()
    grad_norm = float(w.grad.norm())
    powers.append(grad_norm**2 * 0.1)
    opt.step()

print("SGD 'power' (lr * ||grad||^2) per step:")
for i, p in enumerate(powers):
    bar = '#' * int(p * 2)
    print(f"  step {i:2d}  {p:8.4f}  {bar}")
