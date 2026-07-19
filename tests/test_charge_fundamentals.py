"""Test Griffiths' three opening charge facts: quantization, conservation,
and the Coulomb-sign-rule-is-literally-XNOR truth table."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import charge_fundamentals as cf

# 1. quantization: exact integer multiples of e pass, non-integer multiples fail
for n in range(-5, 6):
    ok, n_found = cf.is_quantized(n * cf.E_CHARGE)
    assert ok and n_found == n
ok, _ = cf.is_quantized(1.5 * cf.E_CHARGE)
assert not ok
ok, _ = cf.is_quantized(0.33 * cf.E_CHARGE)
assert not ok

# 2. total_charge is a plain sum, and conservation holds when total is
#    preserved across a rearrangement, fails when it isn't
before = [2*cf.E_CHARGE, -1*cf.E_CHARGE, 3*cf.E_CHARGE]
after_conserved = [1*cf.E_CHARGE, 0*cf.E_CHARGE, 3*cf.E_CHARGE]     # rearranged, same total
after_violated = [1*cf.E_CHARGE, 0*cf.E_CHARGE, 2*cf.E_CHARGE]      # 1e vanished -- not physical
assert abs(cf.total_charge(before) - 4*cf.E_CHARGE) < 1e-30
assert cf.charge_conserved(before, after_conserved)
assert not cf.charge_conserved(before, after_violated)

# 3. coulomb_force_sign: like charges (++ or --) repel (+1), opposite attract (-1)
assert cf.coulomb_force_sign(1, 1) == 1
assert cf.coulomb_force_sign(-1, -1) == 1
assert cf.coulomb_force_sign(1, -1) == -1
assert cf.coulomb_force_sign(-1, 1) == -1

# 4. THE central claim: the charge truth table matches XNOR on all 4 rows,
#    not just qualitatively -- exact boolean equality every time
table = cf.charge_truth_table()
assert len(table) == 4
for (b1, b2), (repulsive, xnor_val) in table.items():
    assert repulsive == xnor_val, (b1, b2, repulsive, xnor_val)
    assert repulsive == (b1 == b2)   # double-check against the raw XNOR definition directly

# 5. input validation
for bad_call in [
    lambda: cf.coulomb_force_sign(0, 1),
    lambda: cf.coulomb_force_sign(1, 2),
    lambda: cf.coulomb_force_sign(1.5, -1),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.charge_fundamentals tests passed")
