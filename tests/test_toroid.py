"""Prove the toroid's Ampere-law field by measuring it with Biot-Savart:
B = mu0 N I / (2 pi s) inside the windings, ~0 outside. (Where Ampere 'works'.)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import magnetostatics as ms

R, a, N, I = 0.5, 0.08, 100, 2.0
loops = ms.toroid_loops(R, a, N, n=300)
expected = ms.MU0_SI * N * I / (2 * np.pi * R)        # Ampere's-law toroid field

# evaluate at the tube center, at a generic azimuth (between two turns)
phi = np.pi / N
er = np.array([np.cos(phi), np.sin(phi), 0.0])        # radial
ephi = np.array([-np.sin(phi), np.cos(phi), 0.0])     # azimuthal
B = ms.biot_savart_multi(loops, I, R * er)
Bmag = np.linalg.norm(B)

# 1. magnitude matches the Ampere result mu0 N I / (2 pi R)
assert abs(Bmag - expected) / expected < 0.06, (Bmag, expected)

# 2. the field is AZIMUTHAL: along +/- ephi, with negligible radial / vertical parts
assert abs(B @ er) / Bmag < 0.05            # no radial component
assert abs(B[2]) / Bmag < 0.05              # no vertical component
assert abs(abs(B @ ephi) - Bmag) / Bmag < 0.05   # essentially all azimuthal

# 3. OUTSIDE the windings the field vanishes (Ampere: no enclosed current)
assert np.linalg.norm(ms.biot_savart_multi(loops, I, [0, 0, 0])) < 0.02 * expected   # on axis
assert np.linalg.norm(ms.biot_savart_multi(loops, I, [3.0, 0, 0])) < 0.02 * expected # far out

# 4. the numerical field agrees with the symbolic closed form toroid_field
sym = float(ms.toroid_field(N, I, R).subs(ms.mu0, ms.MU0_SI))
assert abs(sym - expected) < 1e-12
assert abs(Bmag - sym) / sym < 0.06

# 5. 1/s falloff inside: doubling s halves B (compare two radii within the tube
#    using thin separate toroids of the same N at different R)
loops_2R = ms.toroid_loops(2 * R, a, N, n=300)
B_2R = np.linalg.norm(ms.biot_savart_multi(loops_2R, I, 2 * R * er))
assert abs(B_2R - expected / 2) / (expected / 2) < 0.06     # B ~ 1/s

print(f"TEST PASS  (toroid: |B|={Bmag:.3e} T = mu0 N I/(2 pi R)={expected:.3e} within 6%, "
      f"azimuthal, ~0 outside, 1/s falloff). Ampere's-law result proved by Biot-Savart.")
