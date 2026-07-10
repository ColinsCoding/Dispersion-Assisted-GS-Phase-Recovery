"""Test dgs.flicker_noise: the symbolic ln(f2/f1) band power / equal power per octave, and
the synthesized-noise PSD slope = -alpha for white/pink/brown noise."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from dgs import flicker_noise as fn

# 1. closed-form band power = A ln(f2/f1), depends only on the ratio
assert math.isclose(fn.pink_power_in_band(1.0, 2.0), math.log(2.0))
assert math.isclose(fn.pink_power_in_band(10.0, 20.0), math.log(2.0))   # same octave, higher f
assert math.isclose(fn.pink_power_in_band(1.0, 10.0), math.log(10.0))
assert math.isclose(fn.pink_power_in_band(3.0, 30.0), math.log(10.0))
# equal power per octave / decade helpers
assert math.isclose(fn.power_per_octave(), math.log(2.0))
assert math.isclose(fn.power_per_decade(), math.log(10.0))
# scaling with amplitude
assert math.isclose(fn.pink_power_in_band(1.0, 2.0, amplitude=5.0), 5.0 * math.log(2.0))

# 2. symbolic derivation: octave power is A*log(2) and no longer depends on f1
import sympy as sp
band, octave = fn.symbolic_band_power()
A = sp.symbols("A", positive=True)
assert sp.simplify(octave - A * sp.log(2)) == 0
assert sp.Symbol("f1", positive=True) not in octave.free_symbols   # independent of position
# the band integral really is A*ln(f2/f1)
f1, f2 = sp.symbols("f1 f2", positive=True)
assert sp.simplify(band - A * sp.log(f2 / f1)) == 0

# 3. synthesized noise has the right PSD slope -alpha (averaged over realizations)
N = 2 ** 16
for alpha in (0.0, 1.0, 2.0):
    slope = np.mean([fn.estimate_psd_slope(fn.generate_colored_noise(N, alpha=alpha, seed=s))
                     for s in range(8)])
    assert abs(slope - (-alpha)) < 0.15, f"alpha={alpha}: slope {slope}"

# pink noise slope sits between white (0) and brown (-2)
s_white = np.mean([fn.estimate_psd_slope(fn.generate_colored_noise(N, 0.0, s)) for s in range(6)])
s_pink  = np.mean([fn.estimate_psd_slope(fn.generate_colored_noise(N, 1.0, s)) for s in range(6)])
s_brown = np.mean([fn.estimate_psd_slope(fn.generate_colored_noise(N, 2.0, s)) for s in range(6)])
assert s_white > s_pink > s_brown

# 4. output is real, zero-mean, right length; reproducible with a seed
x = fn.generate_colored_noise(1024, alpha=1.0, seed=42)
assert x.shape == (1024,) and np.isrealobj(x)
assert abs(x.mean()) < 1e-9
assert np.allclose(x, fn.generate_colored_noise(1024, alpha=1.0, seed=42))

# 5. the color table is ordered by exponent
assert fn.NOISE_COLORS["white"] == 0.0 and fn.NOISE_COLORS["pink"] == 1.0
assert fn.NOISE_COLORS["violet"] < fn.NOISE_COLORS["white"] < fn.NOISE_COLORS["brown"]

# 6. kwarg bounds
for bad in (lambda: fn.pink_power_in_band(0, 2),
            lambda: fn.pink_power_in_band(2, 1),
            lambda: fn.generate_colored_noise(2)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_flicker_noise: all checks passed")
