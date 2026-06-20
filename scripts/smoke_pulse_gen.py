"""Smoke-test the one-shot pulse generator + I1/I2 measurement."""
import sys, pathlib, tempfile, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import pulse_gen as pg

# 1. pulse shape is well-formed
t, x, A, phi = pg.generate_pulse(N=512, chirp=18.0, seed=1)
assert x.shape == (512,) and np.iscomplexobj(x)
assert np.allclose(np.abs(x), A)                    # |x| = amplitude envelope
assert A.max() > 0

# 2. measurement: I1, I2 non-negative; diversity grows with |D2-D1|
I1, I2, corr = pg.measure(x, D1=0.0, D2=6000.0)
assert np.all(I1 >= 0) and np.all(I2 >= 0)
assert -1 <= corr <= 1
_, _, corr_lo = pg.measure(x, D1=0.0, D2=300.0)      # little dispersion
_, _, corr_hi = pg.measure(x, D1=0.0, D2=9000.0)     # lots of dispersion
assert corr_lo > corr_hi, (corr_lo, corr_hi)         # more dispersion -> less redundancy
# energy is conserved by dispersion (it's unitary): sum I1 ~ sum I2 (noiseless)
assert abs(I1.sum() - I2.sum()) / I1.sum() < 1e-6

# 3. diversity guard: too-small |D2-D1| is rejected
try:
    pg.measure(x, D1=0.0, D2=50.0)
except ValueError:
    pass
else:
    raise AssertionError("should reject low diversity")

# 4. low-light Poisson path is noisier than noiseless, mean roughly preserved
I1n, I2n, _ = pg.measure(x, D1=0.0, D2=6000.0, photons=50.0, seed=3)
assert np.std(I1n - I1) > 0
assert abs(I1n.mean() - I1.mean()) / I1.mean() < 0.1

# 5. sparkline: right length, only block glyphs
s = pg.sparkline(I1, width=70)
assert len(s) == 70 and all(ch in pg._BLOCKS for ch in s)

# 6. CLI entry returns a result dict and writes CSV
with tempfile.TemporaryDirectory() as d:
    csv_path = os.path.join(d, "shot.csv")
    r = pg.main(["--N", "256", "--chirp", "12", "--D2", "7000", "--ascii", "--csv", csv_path])
    assert set(r) >= {"I1", "I2", "corr", "D1", "D2"}
    assert os.path.exists(csv_path)
    assert len(open(csv_path).read().splitlines()) == 257   # header + 256 rows

# 7. input validation
for bad in (["--N", "8"], ["--photons", "-5"]):
    try:
        pg.main(bad)
    except (ValueError, SystemExit):
        pass
    else:
        raise AssertionError(f"should reject {bad}")

print(f"SMOKE PASS  (corr@D2=6000={corr:.3f}, lo={corr_lo:.3f}>hi={corr_hi:.3f})")
