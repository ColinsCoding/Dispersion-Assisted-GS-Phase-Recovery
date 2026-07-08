"""Test dgs.fundamental_forces: the fine-structure and gravitational couplings from
constants, the famous ~1.2e36 electromagnetic-to-gravity ratio between two protons
(with the internal consistency alpha/alpha_g = that ratio), the strength ordering
strong>EM>weak>gravity, and the ranges/carriers."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs import fundamental_forces as ff

mp = ff.M_PROTON

# 1. fine-structure constant alpha ~= 1/137.036
assert math.isclose(ff.em_coupling(), 1 / 137.035999, rel_tol=1e-4)
assert math.isclose(ff.em_coupling(),
                    ff.K_COULOMB * ff.E_CHARGE**2 / (ff.HBAR * ff.C_LIGHT))

# 2. gravitational coupling for two protons ~5.9e-39, scaling as mass^2
assert math.isclose(ff.gravitational_coupling(mp), 5.9e-39, rel_tol=0.02)
assert math.isclose(ff.gravitational_coupling(2 * mp),
                    4 * ff.gravitational_coupling(mp))          # ~ m^2

# 3. THE headline number: F_EM/F_grav between two protons ~= 1.2e36
ratio = ff.em_to_gravity_ratio(mp)
assert math.isclose(ratio, 1.24e36, rel_tol=0.02)
assert math.isclose(ratio, ff.K_COULOMB * ff.E_CHARGE**2 / (ff.G_NEWTON * mp**2))
# internal consistency: the force ratio equals alpha / alpha_g
assert math.isclose(ratio, ff.em_coupling() / ff.gravitational_coupling(mp))
# distance-independent, and scales as 1/mass^2 (heavier -> gravity catches up)
assert math.isclose(ff.em_to_gravity_ratio(2 * mp), ratio / 4)
assert ratio > 1e30                                            # gravity absurdly weaker

# 4. strength ordering strong > EM > weak > gravity
assert ff.strongest_to_weakest() == ["strong", "electromagnetic", "weak", "gravitational"]
s = ff.FORCES
assert (s["strong"]["relative_strength"] > s["electromagnetic"]["relative_strength"]
        > s["weak"]["relative_strength"] > s["gravitational"]["relative_strength"])

# 5. ranges: strong and weak are short; EM and gravity are infinite (massless carriers)
assert ff.force_range("strong") < 1e-14
assert ff.force_range("weak") < ff.force_range("strong")       # even shorter
assert math.isinf(ff.force_range("electromagnetic"))
assert math.isinf(ff.force_range("gravitational"))

# 6. carriers (the exchange bosons)
assert ff.force_carrier("electromagnetic") == "photon"
assert ff.force_carrier("strong") == "gluon"
assert "W" in ff.force_carrier("weak")
assert "graviton" in ff.force_carrier("gravitational")

# 7. kwarg bounds
for bad in (lambda: ff.gravitational_coupling(0),
            lambda: ff.em_to_gravity_ratio(-1),
            lambda: ff.force_range("magnetism"),
            lambda: ff.force_carrier("nuclear")):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_fundamental_forces: all checks passed")
