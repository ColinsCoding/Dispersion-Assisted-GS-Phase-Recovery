"""Smoke-test the Rayleigh-Plesset sonoluminescence solver (torch, py-3.12)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import sonoluminescence as sl

# 1. no acoustic drive -> bubble sits at its ambient radius (equilibrium)
quiet = sl.simulate(steps_per_period=8000, n_periods=2.0, Pa=0.0)
assert not np.isnan(quiet["R"]).any()
assert np.allclose(quiet["R"], quiet["R0"], rtol=0.02), "undriven bubble must stay ~R0"

# 2. driven bubble: balloons then collapses, no blow-up
sol = sl.simulate(steps_per_period=20000, n_periods=2.5)
assert not np.isnan(sol["R"]).any() and (sol["R"] > 0).all()
s = sl.collapse_summary(sol)
assert s["R_max"] > sol["R0"], "must expand above ambient"
assert s["R_min"] < sol["R0"], "must collapse below ambient"
assert s["expansion_ratio"] > 3.0                       # strong ballooning
assert s["collapse_ratio"] > 5.0                        # violent collapse

# 3. the collapse heats the gas into the flash regime (thousands of K)
assert s["flash_T"] > 3000.0, s["flash_T"]

# 4. flash_temperature is the adiabatic law T0 (R0/Rmin)^{3(kappa-1)}
assert abs(sl.flash_temperature(1e-6, 4e-6, kappa=1.4, T0=300.0)
           - 300.0 * 4.0**(3 * 0.4)) < 1e-6
# stronger drive -> smaller R_min -> hotter flash (monotonic)
hot = sl.collapse_summary(sl.simulate(steps_per_period=20000, n_periods=2.5, Pa=1.35*101325))
mild = sl.collapse_summary(sl.simulate(steps_per_period=20000, n_periods=2.5, Pa=1.15*101325))
assert hot["flash_T"] >= mild["flash_T"]

# 5. numerical convergence: doubling the step count barely moves the collapse
fine = sl.collapse_summary(sl.simulate(steps_per_period=40000, n_periods=2.5))
assert abs(fine["collapse_ratio"] - s["collapse_ratio"]) / s["collapse_ratio"] < 0.05

# 6. validation
try:
    sl.simulate(Pa=-1.0)
except ValueError:
    pass
else:
    raise AssertionError("negative drive should raise")
try:
    sl.flash_temperature(-1.0, 4e-6)
except ValueError:
    pass
else:
    raise AssertionError("negative radius should raise")

print(f"SMOKE PASS  (expand {s['expansion_ratio']:.1f}x -> collapse {s['collapse_ratio']:.1f}x "
      f"-> flash ~{s['flash_T']:.0f} K at t={s['flash_time']*1e6:.1f} us)")
