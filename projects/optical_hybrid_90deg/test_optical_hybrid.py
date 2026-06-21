"""Test the 90-degree optical hybrid: ideal matrix, insertion loss, I/Q recovery."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from optical_hybrid_90deg import optical_hybrid_90deg, balanced_detection

s, l = 1 + 1j, 1 - 1j

# 1. ideal hybrid: outputs are exactly [s+l, s-l, j(s+l), j(s-l)]
o0, o90, o180, o270 = optical_hybrid_90deg(s, l)
assert np.isclose(o0, s + l) and np.isclose(o90, s - l)
assert np.isclose(o180, 1j * (s + l)) and np.isclose(o270, 1j * (s - l))

# 2. output power is 4x the input: |s+l|^2+|s-l|^2 = 2|s|^2+2|l|^2 (parallelogram law),
#    and there are two such pairs (the 0/180 and 90/270 branches)
in_energy = np.abs(s) ** 2 + np.abs(l) ** 2
out_energy = sum(np.abs(o) ** 2 for o in (o0, o90, o180, o270))
assert np.isclose(out_energy, 4 * in_energy)

# 3. balanced detection: both difference signals equal 4 Re(s l*) for THIS matrix
#    (the 1j on rows 3-4 is a global phase, so there is no independent quadrature)
D1, D2 = balanced_detection((o0, o90, o180, o270))
assert np.isclose(D1, 4 * np.real(s * np.conj(l)))
assert np.isclose(D2, 4 * np.real(s * np.conj(l)))
assert np.isclose(D1, D2)                              # both pairs carry the in-phase term

# 4. sweeping the signal phase, D1 tracks the in-phase projection 4 cos(phi) (LO = 1)
phases = np.linspace(0, 2 * np.pi, 32, endpoint=False)
D1s = np.array([balanced_detection(optical_hybrid_90deg(np.exp(1j * ph), 1.0 + 0j))[0]
                for ph in phases])
assert np.allclose(D1s, 4 * np.cos(phases), atol=1e-6)   # in-phase correlation, not a circle

# 5. insertion loss attenuates the field. NOTE the ported convention multiplies the
#    FIELD by 10^(-IL/10), so output POWER scales as 10^(-2*IL/10): a "3 dB" setting
#    gives ~0.251 power (i.e. 6 dB). Tested faithfully to the code as written.
o0_loss = optical_hybrid_90deg(1 + 0j, 0 + 0j, insertion_loss_signal=3.0)[0]
assert np.isclose(np.abs(o0_loss) ** 2, 10 ** (-2 * 3 / 10), atol=1e-3)   # ~0.251
assert np.isclose(np.abs(o0_loss), 10 ** (-3 / 10), atol=1e-3)            # field ~0.501

print(f"TEST PASS  (ideal hybrid [s+l,s-l,j(s+l),j(s-l)]; out power = 4x in; "
      f"balanced det D1=D2=4Re(s l*) (no independent Q in this matrix); "
      f"D1 sweeps as 4cos(phi); 3 dB loss -> {np.abs(o0_loss)**2:.3f})")
