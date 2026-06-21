"""Griffiths Problem 7.7 (sliding bar) -- numeric torch sim; verify all KE -> heat."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch

# bar of mass m on rails (separation l), resistor R, field B, initial speed v0
m, l, R, B, v0 = 0.10, 0.50, 2.0, 0.80, 5.0
alpha = B**2 * l**2 / (m * R)               # decay rate (part c)

T = 12.0 / alpha; n = 40000
t = torch.linspace(0, T, n + 1, dtype=torch.float64)
dt = float(t[1] - t[0])

# RK4 integrate the equation of motion  m dv/dt = -(B^2 l^2 / R) v  ->  dv/dt = -alpha v
v = torch.zeros(n + 1, dtype=torch.float64); v[0] = v0
f = lambda vv: -alpha * vv
for i in range(n):
    k1 = f(v[i]); k2 = f(v[i] + 0.5*dt*k1); k3 = f(v[i] + 0.5*dt*k2); k4 = f(v[i] + dt*k3)
    v[i+1] = v[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# 1. matches the analytic solution v = v0 exp(-alpha t)  (part c)
v_exact = v0 * torch.exp(-alpha * t)
assert torch.max(torch.abs(v - v_exact)) < 1e-6

# 2. current I = Blv/R and force F = B^2 l^2 v / R (parts a, b) at t=0
I0 = B * l * v0 / R
assert abs(I0 - 1.0) < 1e-9                  # B l v0 / R = 0.8*0.5*5/2 = 1.0 A
assert abs((B**2 * l**2 * v0 / R) - (B * l * I0)) < 1e-12   # F = B l I

# 3. THE POINT (part d): energy into the resistor = integral I^2 R dt = 1/2 m v0^2
I = B * l * v / R
P = I**2 * R
W = float(torch.trapezoid(P, t))
KE0 = 0.5 * m * v0**2
assert abs(W - KE0) / KE0 < 1e-3, (W, KE0)   # all the kinetic energy becomes heat

print(f"TEST PASS  (v=v0 e^(-alpha t), alpha={alpha:.2f}; energy to R = {W:.4f} J "
      f"= 1/2 m v0^2 = {KE0:.4f} J -- all KE -> heat)")
