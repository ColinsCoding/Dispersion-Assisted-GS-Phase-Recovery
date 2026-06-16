"""Smoke-test the neon domain-colouring: mapping correctness + figure render."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from matplotlib.colors import rgb_to_hsv
import neon_photonics as neon

# 1. RGB output is well-formed and in range
phi = np.linspace(-np.pi, np.pi, 64)
rgb = neon.phase_to_neon_rgb(phi)
assert rgb.shape == (64, 3)
assert rgb.min() >= 0.0 and rgb.max() <= 1.0

# 2. hue is cyclic: phi and phi+2pi give the same colour
a = neon.phase_to_neon_rgb(np.array([0.7]))
b = neon.phase_to_neon_rgb(np.array([0.7 + 2 * np.pi]))
assert np.allclose(a, b, atol=1e-6), "hue must be 2pi-periodic"

# 3. distinct phases -> distinct hues, monotone around the wheel
hsv = rgb_to_hsv(neon.phase_to_neon_rgb(phi))
hues = hsv[:, 0]
assert len(np.unique(np.round(hues, 3))) > 50, "phases should spread across the wheel"

# 4. brightness tracks intensity: brighter where |E|^2 is larger
I = np.array([0.0, 0.25, 1.0])
val = rgb_to_hsv(neon.phase_to_neon_rgb(np.zeros(3), I))[:, 2]
assert val[0] < val[1] < val[2], "brightness must increase with intensity"
assert val[0] > 0.0, "neon floor: dark stays faintly lit"

# 5. negative intensity rejected
try:
    neon.phase_to_neon_rgb(np.zeros(3), np.array([-1.0, 0, 0]))
    raise AssertionError("should reject negative intensity")
except ValueError:
    pass

# 6. STFT shape sanity
S, ti, fi = neon.stft(np.random.randn(1024) + 0j, win=128, hop=16)
assert S.shape == (128, len(ti)) and len(fi) == 128

# 7. full render writes a file
out = neon.render(N=1024, out="figures/_smoke_neon.png")
assert pathlib.Path(out).exists()
pathlib.Path(out).unlink()  # clean up the smoke artifact

print("SMOKE PASS")
