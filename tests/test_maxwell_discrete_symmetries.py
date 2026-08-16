"""Test dgs/maxwell_discrete_symmetries.py's parity and time-reversal
derivations for Maxwell's equations. Each field-transformation claim is
derived directly from the field's defining law (Coulomb, Biot-Savart, the
chain rule under t->-t), not asserted -- these tests re-verify the same
symbolic derivations the module performs, guarding against a future edit
silently breaking one of them."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.maxwell_discrete_symmetries import (
    coulomb_field_parity_check, biot_savart_field_parity_check,
    time_reversal_velocity_parity, time_reversal_acceleration_parity,
    maxwell_parity_consistency, maxwell_time_reversal_consistency,
    combined_pt_type,
)

# 1. E must be a polar vector, derived from Coulomb's law
assert coulomb_field_parity_check() is True

# 2. B must be an axial vector (pseudovector), derived from Biot-Savart
assert biot_savart_field_parity_check() is True

# 3. Velocity must be T-odd (chain rule on x(-t))
assert time_reversal_velocity_parity() is True

# 4. Acceleration must be T-even (chain rule, two derivatives cancel the flip)
assert time_reversal_acceleration_parity() is True

# 5. All four Maxwell equations must be internally parity-consistent
p_check = maxwell_parity_consistency()
assert all(p_check.values()), f"parity-inconsistent equation(s): {p_check}"
assert len(p_check) == 4, "expected exactly 4 Maxwell equations checked"

# 6. All four Maxwell equations must be internally time-reversal-consistent
t_check = maxwell_time_reversal_consistency()
assert all(t_check.values()), f"time-reversal-inconsistent equation(s): {t_check}"
assert len(t_check) == 4, "expected exactly 4 Maxwell equations checked"

# 7. Combined PT: both E and B must come out PT-odd (a consequence of 1-4,
#    not a separate assumption -- P-odd*T-even=-1 for E, P-even*T-odd=-1 for B)
pt = combined_pt_type()
assert pt["E"] == -1
assert pt["B"] == -1

print("all dgs.maxwell_discrete_symmetries tests passed")
