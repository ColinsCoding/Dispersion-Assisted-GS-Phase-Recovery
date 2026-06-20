"""Smoke-test griffiths.hilbert: orthogonality of the classical bases + expansion."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from griffiths import hilbert as hb

x = sp.Symbol("x", real=True)

# 1. Fourier sines orthogonal on [-pi, pi]: <sin nx, sin mx> = pi delta_nm
sines = [sp.sin(n*x) for n in (1, 2, 3)]
G = hb.gram_matrix(sines, x, -sp.pi, sp.pi)
print("Fourier-sine Gram matrix on [-pi,pi]:")
sp.pprint(G)
print("  orthogonal?", hb.is_orthogonal(sines, x, -sp.pi, sp.pi))

# 2. Legendre orthogonal on [-1,1]: <P_l, P_m> = 2/(2l+1) delta_lm
legs = [sp.legendre(l, x) for l in range(4)]
Gl = hb.gram_matrix(legs, x, -1, 1)
print("\nLegendre Gram diagonal:", [Gl[i, i] for i in range(4)],
      " (expect 2, 2/3, 2/5, 2/7)")
print("  orthogonal?", hb.is_orthogonal(legs, x, -1, 1))

# 3. Hermite orthogonal with weight e^{-x^2} on (-inf, inf)
herms = [sp.hermite(n, x) for n in range(3)]
Gh = hb.gram_matrix(herms, x, -sp.oo, sp.oo, weight=sp.exp(-x**2))
print("\nHermite Gram diagonal:", [Gh[i, i] for i in range(3)],
      " (expect sqrt(pi)*2^n*n!)")
print("  orthogonal? (weighted)", hb.is_orthogonal(herms, x, -sp.oo, sp.oo, weight=sp.exp(-x**2)))

# 4. expansion = projection: expand x on [-pi,pi] in sines (Fourier sine series)
coeffs, recon = hb.expand(x, sines, x, -sp.pi, sp.pi)
print("\nFourier-sine coefficients of f(x)=x:", coeffs, " (expect 2, -1, 2/3 ...)")
# check the n-th coefficient is 2*(-1)^(n+1)/n
for n, c in zip((1, 2, 3), coeffs):
    assert sp.simplify(c - 2*(-1)**(n+1)/n) == 0

# 5. phasors: the simplest Hilbert space
A, ph = sp.symbols("A phi", real=True, positive=True)
p = hb.phasor(A, ph)
print("\nphasor A e^{i phi} =", p, " |phasor|^2 =", sp.simplify(hb.phasor_inner([p], [p])))

# validation
try:
    hb.phasor_inner([1, 2], [1])
except ValueError as e:
    print("err ok:", e)
print("SMOKE PASS")
