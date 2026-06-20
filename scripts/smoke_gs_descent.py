"""Smoke-test torch gradient-descent phase retrieval vs GS (run with py -3.12)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import dispersion_gs_prototype as dg
from dgs import dispersion_gs_descent as gd

t, x, A, phi = dg.make_field(2048, seed=0)
I1 = np.abs(x)**2

print("gradient-descent vs GS on the dispersion phase problem (RMS rad):")
for D in (12000, 20000):
    I2 = np.abs(dg.disperse(x, D))**2
    corr = np.corrcoef(I1, I2)[0, 1]
    # GS baseline
    xr, _ = dg.gerchberg_saxton(I1, I2, D, n_iter=400, seed=0)
    gs_rms, _ = dg.compare_phase(np.angle(xr), phi, A**2)
    # torch GD, no prior and with smoothness prior
    p0, _ = gd.torch_phase_retrieval(I1, I2, D, reg=0.0, n_iter=2000, seed=0)
    gd0, _ = gd.compare_phase(p0, phi, A**2)
    pr, hist = gd.torch_phase_retrieval(I1, I2, D, reg=0.5, n_iter=2000, seed=0)
    gdr, _ = gd.compare_phase(pr, phi, A**2)
    print(f"  D={D:>5} corr={corr:.3f}: GS={gs_rms:.3f}  GD(no prior)={gd0:.3f}  "
          f"GD(+smooth)={gdr:.3f}   data-loss {hist[0]:.2e}->{hist[-1]:.2e}")

assert gd0 < gs_rms, "gradient descent should beat GS on the hard case"

for bad in [lambda: gd.torch_phase_retrieval(I1, I1, 1, reg=-1)]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", e)
print("SMOKE PASS")
