"""Smoke-test quantum phase estimation -- the 'phase ADC'."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import qubits as q

# 1. exactly representable phases are read out exactly, with probability 1
for phi, n in [(0.25, 3), (0.125, 4), (0.375, 3), (0.5, 2), (0.0, 3)]:
    phi_hat, p = q.estimate_phase(phi, n)
    assert abs(phi_hat - phi) < 1e-12, (phi, phi_hat)
    assert abs(p - 1.0) < 1e-9, (phi, p)

# 2. it really is the BINARY EXPANSION of phi (the ADC view): phi=0.25 -> m=2 -> '010'
state = q.phase_estimation(0.25, 3)
m = int(np.argmax(q.measure_probs(state)))
assert format(m, "03b") == "010"                 # 0.010_2 = 1/4
assert format(int(np.argmax(q.measure_probs(q.phase_estimation(0.375, 3)))), "03b") == "011"

# 3. a non-representable phase lands on the nearest n-bit value with high prob
phi = 0.1
phi_hat, p = q.estimate_phase(phi, 6)
nearest = round(phi * 2**6) / 2**6               # = 6/64 = 0.09375
assert abs(phi_hat - nearest) < 1e-12
assert p > 0.5                                    # dominant outcome

# 4. more counting qubits -> finer resolution (closer to the true phi)
errs = [abs(q.estimate_phase(0.1, n)[0] - 0.1) for n in (3, 6, 9)]
assert errs[0] >= errs[1] >= errs[2] and errs[2] < 0.01

# 5. the QPE register state stays normalized (unitary throughout)
assert abs(np.linalg.norm(q.phase_estimation(0.3, 5)) - 1.0) < 1e-10

# 6. validation
try:
    q.phase_estimation(0.25, 0)
except ValueError:
    pass
else:
    raise AssertionError("n_counting < 1 should raise")

print(f"SMOKE PASS  (phi=0.25 -> '010'=exact; phi=0.1 (n=6) -> {q.estimate_phase(0.1,6)[0]:.5f} "
      f"p={q.estimate_phase(0.1,6)[1]:.2f}; more qubits -> finer: {[round(e,4) for e in errs]})")
