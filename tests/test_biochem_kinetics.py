"""Test biochem_kinetics: QSSA validity scales with E0/S0, Hill reduces to MM
at n=1, competitive inhibition preserves Vmax, Lineweaver-Burk recovers
known constants."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import biochem_kinetics as bk

# 1. QSSA (the MM reduction) is far more accurate when E0 << S0 than E0 ~ S0
t = np.linspace(0, 50, 1500)
res_good = bk.qssa_validity(E0=0.01, S0=10.0, k1=1.0, k_minus1=1.0, k2=1.0, t=t)
res_bad = bk.qssa_validity(E0=8.0, S0=10.0, k1=1.0, k_minus1=1.0, k2=1.0, t=t)
assert res_good["rms_relative_error"] < res_bad["rms_relative_error"] / 10

# 2. Hill equation with n=1 reduces exactly to Michaelis-Menten
S = np.linspace(0.1, 20, 50)
mm = bk.michaelis_menten_rate(S, Vmax=5.0, Km=3.0)
hill1 = bk.hill_equation(S, Vmax=5.0, K=3.0, n=1.0)
assert np.allclose(mm, hill1)

# 3. competitive inhibition: Vmax is preserved at saturating S, Km is not
v_no_inhib = bk.competitive_inhibition_rate(S=1e5, I=0.0, Vmax=5.0, Km=3.0, Ki=2.0)
v_inhib = bk.competitive_inhibition_rate(S=1e5, I=10.0, Vmax=5.0, Km=3.0, Ki=2.0)
assert abs(v_no_inhib - 5.0) < 1e-3
assert abs(v_inhib - 5.0) < 1e-3   # both approach Vmax at very high S

# 4. Lineweaver-Burk recovers known Vmax, Km from noiseless synthetic data
true_Vmax, true_Km = 7.5, 2.2
S2 = np.linspace(0.5, 20, 30)
v2 = bk.michaelis_menten_rate(S2, true_Vmax, true_Km)
fit = bk.lineweaver_burk_fit(S2, v2)
assert abs(fit["Vmax"] - true_Vmax) < 1e-3
assert abs(fit["Km"] - true_Km) < 1e-3

# 5. RK4 integrates a known linear decay dy/dt=-y accurately against e^-t
y = bk.integrate_rk4(lambda t, y: -y, np.array([1.0]), np.linspace(0, 5, 200))[:, 0]
assert abs(y[-1] - np.exp(-5)) < 1e-4

print("test_biochem_kinetics: all checks passed")
