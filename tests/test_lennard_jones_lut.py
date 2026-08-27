"""Test dgs/lennard_jones_lut.py: the LUT interpolation actually converges
to the exact lj_potential/lj_force_magnitude as table size grows, fixed-
point quantization error shrinks with bit width the way real hardware
memory would predict, and the LUT-based pair_forces_lut is a genuine
drop-in for dgs.lennard_jones.pair_forces (same physics, small controlled
error) -- not just "runs without crashing"."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.lennard_jones import lj_potential, lj_force_magnitude, hex_cluster, pair_forces
from dgs.lennard_jones_lut import (
    build_lj_lut, lut_lookup, verify_lut_converges, quantize_table,
    lut_quantization_error, pair_forces_lut,
)

# 1. A dense LUT reproduces the exact potential closely at a grid point itself
r2_grid, V_table, FoverR_table = build_lj_lut(r2_min=0.85, r2_max=9.0, n_entries=2000)
r_test = 1.5
V_exact = lj_potential(r_test)
V_lut = lut_lookup(r_test ** 2, r2_grid, V_table)
assert abs(V_lut - V_exact) < 1e-4

# 2. Interpolation error shrinks monotonically as table size grows (the
#    actual claim this module exists to make)
conv = verify_lut_converges(n_entries_list=(16, 64, 256, 1024))
assert conv["monotonically_improving"]
assert conv["rms_error"][-1] < conv["rms_error"][0] / 100  # >=100x improvement, 16->1024 entries

# 3. Out-of-range queries raise, not silently extrapolate
try:
    lut_lookup(100.0, r2_grid, V_table)
    assert False, "should have raised ValueError"
except ValueError:
    pass

# 4. Construction validates its inputs
for bad_call in [
    lambda: build_lj_lut(r2_min=1.0, r2_max=0.5, n_entries=10),
    lambda: build_lj_lut(r2_min=-1.0, r2_max=5.0, n_entries=10),
    lambda: build_lj_lut(r2_min=0.5, r2_max=5.0, n_entries=1),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

# 5. Quantization: more bits -> strictly smaller error (finer fixed-point steps)
quant = lut_quantization_error(n_bits_list=(4, 8, 12, 16))
errs = quant["rms_error"]
assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))

# 6. Quantization error roughly halves per extra bit (each bit halves the
#    quantization step) -- checked as an order-of-magnitude claim, not exact
ratio_4_to_8 = errs[0] / errs[1]     # 4 extra bits -> ~16x smaller error
assert 8.0 < ratio_4_to_8 < 32.0

try:
    quantize_table(V_table, n_bits=0)
    assert False, "should have raised ValueError"
except ValueError:
    pass

try:
    quantize_table(np.array([1.0, 1.0, 1.0]), n_bits=8)
    assert False, "should have raised ValueError for a constant (zero-range) table"
except ValueError:
    pass

# 7. pair_forces_lut is a genuine drop-in: same cluster, closely matching
#    energy and forces against the exact evaluation (small, bounded error)
pos = hex_cluster(n_rings=1)
F_exact, U_exact = pair_forces(pos)
r2_grid_fine, V_table_fine, FoverR_fine = build_lj_lut(r2_min=0.7, r2_max=9.0, n_entries=2000)
F_lut, U_lut = pair_forces_lut(pos, r2_grid_fine, FoverR_fine, V_table_fine)
assert abs(U_lut - U_exact) < 1e-3
assert np.max(np.abs(F_lut - F_exact)) < 1e-2

print("all dgs.lennard_jones_lut tests passed")
