"""Test dgs.zeeman: Lande g, anomalous line patterns, THz<->wavelength cases.
(Normal Zeeman via griffiths.atomic is covered in test_zeeman.py.)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import zeeman as zm

# 1. THz <-> nm: c/f both ways, involution, telecom anchor 193.41 THz ~ 1550 nm
assert abs(float(zm.thz_to_wavelength_nm(193.4145)) - 1550.0) < 0.02
assert abs(float(zm.wavelength_nm_to_thz(1550.0)) - 193.4145) < 0.001
for lam in (532.0, 589.0, 632.8, 850.0, 1064.0, 1310.0, 1550.0):
    assert abs(float(zm.thz_to_wavelength_nm(zm.wavelength_nm_to_thz(lam))) - lam) < 1e-9
bands = zm.common_bands()
assert bands["HeNe red"][1] == 473.76 and bands["sodium D2"][1] == 508.99

# 2. the lambda^2 lever arm: 100 GHz = 0.801 nm at 1550 nm (DWDM grid fact)
dlam = float(zm.wavelength_shift_nm(1550.0, 100e9))
assert abs(abs(dlam) - 0.801) < 0.001
assert dlam < 0  # higher f -> shorter lambda
# at 589 nm the same 100 GHz moves lambda less by exactly (589/1550)^2
ratio = float(zm.wavelength_shift_nm(589.0, 100e9)) / dlam
assert abs(ratio - (589.0 / 1550.0) ** 2) < 1e-6

# 3. Lande g-factors: the textbook table
assert abs(zm.lande_g(0, 0.5, 0.5) - 2.0) < 1e-12        # 2S_1/2: pure spin
assert abs(zm.lande_g(1, 0.5, 0.5) - 2 / 3) < 1e-12      # 2P_1/2
assert abs(zm.lande_g(1, 0.5, 1.5) - 4 / 3) < 1e-12      # 2P_3/2
assert abs(zm.lande_g(1, 0, 1) - 1.0) < 1e-12            # S=0 -> g=1 always
assert abs(zm.lande_g(2, 0.5, 2.5) - 1.2) < 1e-12        # 2D_5/2
assert zm.lande_g(0, 0, 0) == 0.0                        # J=0: no splitting

# 4. splitting scale: mu_B/h = 13.996 GHz per tesla (g=1); ESR g=2 doubles it
assert abs(zm.splitting_frequency_hz(1.0, 1.0) - 13.996e9) < 5e6
assert abs(zm.splitting_frequency_hz(2.0, 0.5) - zm.splitting_frequency_hz(1.0, 1.0)) < 1
m_J, dE = zm.zeeman_sublevels(4 / 3, 1.5, 1.0)
assert len(m_J) == 4 and np.allclose(m_J, [-1.5, -0.5, 0.5, 1.5])
assert np.allclose(dE, -dE[::-1])                        # symmetric about zero
assert np.allclose(np.diff(dE), (4 / 3) * zm.MU_B * 1.0) # equal spacing g*mu_B*B

# 5. normal Zeeman (S=0 both terms): exactly 3 lines at -1, 0, +1 * mu_B B/h
s3 = zm.transition_shifts_hz(1, 0, 1, 0, 0, 0, 1.0)
assert len(s3) == 3
assert np.allclose(s3, np.array([-1, 0, 1]) * zm.MU_B_HZ_PER_T)

# 6. anomalous Zeeman 2P_3/2 -> 2S_1/2: 6 lines at +/-{1/3, 1, 5/3} * mu_B B/h
s6 = zm.transition_shifts_hz(1, 0.5, 1.5, 0, 0.5, 0.5, 1.0)
assert len(s6) == 6
expect = np.array([-5 / 3, -1, -1 / 3, 1 / 3, 1, 5 / 3]) * zm.MU_B_HZ_PER_T
assert np.allclose(s6, expect)

# 7. sodium D2 at 1 T: full spread = 2*(5/3)*13.996 GHz -> ~54 pm at 589 nm
rep = zm.zeeman_wavelength_report(589.0, 1, 0.5, 1.5, 0, 0.5, 0.5, 1.0)
assert rep["n_lines"] == 6
spread_hz = 2 * (5 / 3) * zm.MU_B_HZ_PER_T
expect_pm = (589e-9) ** 2 * spread_hz / zm.C * 1e12
assert abs(rep["spread_pm"] - expect_pm) < 0.01 and 50 < rep["spread_pm"] < 60

# 8. kwarg bounds: bad inputs raise ValueError with clear messages
for bad in (lambda: zm.thz_to_wavelength_nm(0),
            lambda: zm.wavelength_nm_to_thz(-1),
            lambda: zm.lande_g(1, 0.5, 3.0),             # J > L+S
            lambda: zm.lande_g(-1, 0.5, 0.5),
            lambda: zm.splitting_frequency_hz(2.0, -1.0),
            lambda: zm.zeeman_sublevels(2.0, 0.7, 1.0),  # J not half-integer
            lambda: zm.transition_shifts_hz(1, 0, 1, 0, 0, 0, -0.1)):
    try:
        bad()
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

print(f"TEST PASS  (193.41 THz<->1550 nm; 100 GHz=0.801 nm; g: 2S=2, 2P3/2=4/3; "
      f"normal=3 lines, anomalous=6; Na D2 @1T spread={rep['spread_pm']:.1f} pm)")
