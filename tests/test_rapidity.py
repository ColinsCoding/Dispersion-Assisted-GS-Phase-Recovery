"""Smoke-test rapidity = spacetime rotation: boosts compose by adding rapidities."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from griffiths import relativity as r

t1, t2 = sp.symbols("theta_1 theta_2", real=True)
b1, b2 = sp.symbols("beta_1 beta_2", real=True)

# 1. a boost is a HYPERBOLIC rotation: boost2 preserves the Minkowski interval
#    Lambda^T eta Lambda = eta, eta = diag(1, -1) for (ct, x)
eta = sp.diag(1, -1)
L = r.boost2(t1)
assert sp.simplify(L.T * eta * L - eta) == sp.zeros(2, 2)
# (compare: rotation2 preserves x^2 + y^2, i.e. R^T R = I)
R = r.rotation2(t1)
assert sp.simplify(R.T * R - sp.eye(2)) == sp.zeros(2, 2)

# 2. boosts COMPOSE by adding rapidities: boost2(t1) boost2(t2) = boost2(t1+t2)
assert sp.simplify(r.boost2(t1) * r.boost2(t2) - r.boost2(t1 + t2)) == sp.zeros(2, 2)

# 3. rapidity <-> velocity round trip: tanh(atanh(beta)) = beta
assert sp.simplify(r.velocity_from_rapidity(r.rapidity(b1)) - b1) == 0

# 4. THE PUNCHLINE: Einstein velocity addition is just rapidity addition.
#    tanh(rapidity(b1) + rapidity(b2)) == (b1+b2)/(1+b1 b2)
combined = sp.expand_trig(r.velocity_from_rapidity(r.add_rapidities(b1, b2)))
assert sp.simplify(combined - r.add_velocities(b1, b2)) == 0

# 5. numeric: two boosts of beta=0.6 do NOT give 1.2 (would exceed c); rapidities add
v = float(r.velocity_from_rapidity(r.add_rapidities(sp.Rational(3, 5), sp.Rational(3, 5))))
assert abs(v - (0.6 + 0.6) / (1 + 0.36)) < 1e-12 and v < 1.0      # 0.882 < c
th = float(r.rapidity(sp.Rational(3, 5)))
assert abs(float(r.add_rapidities(sp.Rational(3,5), sp.Rational(3,5))) - 2 * th) < 1e-12

print(f"SMOKE PASS  (boost = hyperbolic rotation, composes by adding rapidity; "
      f"0.6c (+) 0.6c = {v:.3f}c < 1, because rapidities add not velocities)")
