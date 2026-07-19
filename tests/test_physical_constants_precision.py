"""Test data-type selection for physical constants: exact (2019 SI
redefinition) constants deserve full float64 precision, poorly-measured
constants (G) are fine in float32, and well-measured ones (QED's
fine-structure constant alpha) genuinely need float64 -- verified by
comparing float32's OWN roundoff error against each constant's real
physical uncertainty, not asserted."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import physical_constants_precision as pcp

# 1. exact constants (2019 SI redefinition) report unlimited justified digits
for name in ["c (speed of light)", "h (Planck constant)", "e (elementary charge)",
             "k_B (Boltzmann constant)", "N_A (Avogadro constant)"]:
    value, rel_unc, is_exact, unit = pcp.CONSTANTS[name]
    assert is_exact is True
    assert rel_unc == 0.0
    assert pcp.significant_digits_justified(rel_unc) is None

# 2. G (poorly measured) justifies far fewer digits than alpha (extremely
#    well measured) -- confirms the "not all constants are equally known" claim
digits_G = pcp.significant_digits_justified(pcp.CONSTANTS["G (gravitational constant)"][1])
digits_alpha = pcp.significant_digits_justified(pcp.CONSTANTS["alpha (fine-structure, QED coupling)"][1])
assert digits_G < digits_alpha
assert digits_G <= 5   # G's real uncertainty is only ~5 significant digits
assert digits_alpha >= 8  # alpha is known to 8+ significant digits

# 3. dtype recommendation: G is float32-sufficient, alpha needs float64
rec_G = pcp.recommend_dtype(pcp.CONSTANTS["G (gravitational constant)"][1], is_exact=False)
rec_alpha = pcp.recommend_dtype(pcp.CONSTANTS["alpha (fine-structure, QED coupling)"][1], is_exact=False)
assert "float32 is SUFFICIENT" in rec_G
assert "float64 needed" in rec_alpha

# 4. THE central verified claim: float32's own roundoff error, for alpha,
#    EXCEEDS alpha's real physical uncertainty (float32 would add MORE
#    error than experiment itself is uncertain about) -- while for G, the
#    roundoff is comfortably SMALLER than G's real (much looser) uncertainty
roundoff_alpha, unc_alpha, exact_alpha = pcp.float32_roundoff_vs_true_uncertainty(
    "alpha (fine-structure, QED coupling)")
assert exact_alpha is False
assert roundoff_alpha > unc_alpha    # float32 error EXCEEDS the real uncertainty

roundoff_G, unc_G, exact_G = pcp.float32_roundoff_vs_true_uncertainty("G (gravitational constant)")
assert exact_G is False
assert roundoff_G < unc_G            # float32 error is well within G's real uncertainty

# 5. for an EXACT constant (c), float32 roundoff is nonzero -- meaning it
#    genuinely destroys information that exists (the true value has no
#    uncertainty at all, so ANY roundoff is new, avoidable error)
roundoff_c, unc_c, exact_c = pcp.float32_roundoff_vs_true_uncertainty("c (speed of light)")
assert exact_c is True
assert unc_c == 0.0
assert roundoff_c > 0.0

# 6. precision_waste_ratio: G's uncertainty is many orders of magnitude
#    larger than float64's epsilon (float64 is massive overkill for G)
waste_ratio_G_f64 = pcp.precision_waste_ratio(pcp.CONSTANTS["G (gravitational constant)"][1], pcp.FLOAT64_EPS)
assert waste_ratio_G_f64 > 1e8   # G's uncertainty is ~1e11x larger than float64's epsilon

# 7. input validation
for bad_call in [
    lambda: pcp.significant_digits_justified(-1.0),
    lambda: pcp.precision_waste_ratio(0.0, pcp.FLOAT64_EPS),
    lambda: pcp.precision_waste_ratio(-1.0, pcp.FLOAT64_EPS),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.physical_constants_precision tests passed")
