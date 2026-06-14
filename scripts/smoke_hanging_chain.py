"""Verify the hanging-chain (J_0) frequencies against a finite-difference eigensolver."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import bessel as bz

L, g = 1.0, 9.81

# Bessel prediction: omega_n = (alpha_n/2) sqrt(g/L)
freqs, zeros = bz.hanging_chain_frequencies(L, g, 5)
print("J_0 zeros:", np.round(zeros, 4))
print("Bessel frequencies omega_n:", np.round(freqs, 4))

# Finite-difference check: -(x u')' u = lambda u,  lambda = omega^2/g,  u(L)=0
# Sturm-Liouville with p(x)=x, weight 1. Discretize on [0, L].
N = 4000
x = np.linspace(0, L, N + 1)
h = x[1] - x[0]
xmid = 0.5 * (x[:-1] + x[1:])            # p at half-points
# interior unknowns u_1..u_{N-1}; u_0 free (natural), u_N = 0 (Dirichlet)
n = N
main = np.zeros(n)
lower = np.zeros(n - 1)
upper = np.zeros(n - 1)
for i in range(n):                       # node i (0..N-1), u_N=0 enforced by omission
    pL = xmid[i - 1] if i > 0 else 0.0   # p at i-1/2 (0 at the singular bottom)
    pR = xmid[i]                         # p at i+1/2
    main[i] = (pL + pR) / h**2
    if i > 0:
        lower[i - 1] = -pL / h**2
    if i < n - 1:
        upper[i] = -pR / h**2
A = np.diag(main) + np.diag(lower, -1) + np.diag(upper, 1)
lam = np.sort(np.linalg.eigvalsh(A))[:5]
omega_fd = np.sqrt(g * lam)
print("FD frequencies:           ", np.round(omega_fd, 4))
print("max rel error vs Bessel:   ",
      np.max(np.abs(omega_fd - np.array(freqs)) / np.array(freqs)))

# mode shape: J_0(alpha_n sqrt(x/L)); fixed top (xfrac=1 -> 0), free bottom (xfrac=0 -> 1)
u = bz.hanging_chain_modeshape(zeros[0], np.linspace(0, 1, 11))
print("\nfundamental mode J_0 values (bottom->top):", np.round(u, 3))
print("  top value (should be ~0):", round(u[-1], 4), " bottom (should be 1):", round(u[0], 4))

# simple-pendulum limit sanity: a point mass has omega=sqrt(g/L); chain fundamental is higher
print(f"\nsimple pendulum omega = sqrt(g/L) = {np.sqrt(g/L):.3f}")
print(f"chain fundamental    = {freqs[0]:.3f}  (ratio {freqs[0]/np.sqrt(g/L):.3f} = alpha_1/2)")

for bad in [lambda: bz.hanging_chain_frequencies(-1, g, 3),
            lambda: bz.hanging_chain_frequencies(L, g, 0)]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", e)
print("SMOKE PASS")
