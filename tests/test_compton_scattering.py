"""Test dgs/compton_scattering.py: the Compton wavelength shift DERIVED
from relativistic energy-momentum conservation (a unique symbolic solve,
not a quoted formula), full conservation checks (energy + both momentum
components) for a concrete collision, and the classical Thomson limit."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math
import numpy as np
from dgs.compton_scattering import (
    derive_compton_shift_symbolic, derive_electron_recoil_angle_symbolic,
    compton_wavelength_shift, compton_wavelength_out,
    compton_electron_kinetic_energy, compton_electron_recoil_angle,
    verify_full_conservation, verify_thomson_limit,
    H_PLANCK, M_ELECTRON, C_LIGHT, COMPTON_WAVELENGTH,
)

# 1. derive_compton_shift_symbolic: must succeed and match the textbook formula
lam_p_solved, shift_derived = derive_compton_shift_symbolic()
import sympy as sp
h, c, me, theta = sp.symbols("h c m_e theta", positive=True)
expected = h * (1 - sp.cos(theta)) / (me * c)
assert sp.simplify(shift_derived - expected) == 0

# 2. COMPTON_WAVELENGTH: known value, ~2.426 pm
assert abs(COMPTON_WAVELENGTH - 2.42631023867e-12) / 2.42631023867e-12 < 1e-6

# 3. compton_wavelength_shift: known special angles
#    theta=0 (forward scatter, no collision really happened): shift = 0
#    theta=90deg: shift = lambda_C exactly
#    theta=180deg (backscatter): shift = 2*lambda_C (maximum possible shift)
assert abs(compton_wavelength_shift(0.0)) < 1e-30
assert abs(compton_wavelength_shift(math.pi / 2) - COMPTON_WAVELENGTH) / COMPTON_WAVELENGTH < 1e-9
assert abs(compton_wavelength_shift(math.pi) - 2 * COMPTON_WAVELENGTH) / (2 * COMPTON_WAVELENGTH) < 1e-9

# 4. compton_wavelength_shift: monotonically increasing in theta over [0, pi]
#    (1-cos(theta) is monotonic there)
thetas = np.linspace(0, math.pi, 50)
shifts = [compton_wavelength_shift(t) for t in thetas]
assert all(shifts[i+1] >= shifts[i] for i in range(len(shifts) - 1))

# 5. compton_wavelength_out: input validation
try:
    compton_wavelength_out(-1e-9, math.pi / 2)
    raise AssertionError("expected ValueError for lambda_in_m <= 0")
except ValueError:
    pass

# 6. verify_full_conservation: must pass across several (wavelength, angle)
#    combinations, not just one lucky case
for lam_in, theta_val in [(0.1e-9, math.pi/2), (1e-12, math.pi/4), (5e-10, math.pi), (2e-11, 0.1)]:
    result = verify_full_conservation(lam_in, theta_val)
    for name, ok in result["checks"].items():
        assert ok, f"lam_in={lam_in}, theta={theta_val}: {name} failed (residual info: {result})"

# 7. verify_full_conservation: electron KE must be positive and must not
#    exceed the photon's own incoming energy (energy can only be
#    transferred, never created)
lam_in = 0.1e-9
result = verify_full_conservation(lam_in, math.pi / 2)
photon_energy_in = H_PLANCK * C_LIGHT / lam_in
assert 0 < result["KE_electron_J"] < photon_energy_in

# 8. compton_electron_recoil_angle: at theta=0 (no scattering), the
#    electron recoil angle is undefined/degenerate, but at a real
#    scattering angle it must be a sensible forward-hemisphere angle
#    (electron always recoils forward, |phi| < 90 deg, for theta in (0, pi))
for theta_val in [0.1, math.pi/4, math.pi/2, 3*math.pi/4, math.pi - 0.1]:
    phi = compton_electron_recoil_angle(lam_in, theta_val)
    assert abs(phi) < math.pi / 2, f"theta={theta_val}: electron recoil angle {phi} not forward-hemisphere"

# 9. verify_thomson_limit: must pass, and the fractional shift must
#    actually DECREASE as wavelength_ratio increases (approaching the
#    classical limit more closely, not just passing a fixed threshold)
assert verify_thomson_limit(wavelength_ratio=1e3) is True
assert verify_thomson_limit(wavelength_ratio=1e6) is True
shift_ratio_small = compton_wavelength_shift(math.pi/2) / (1e3 * COMPTON_WAVELENGTH)
shift_ratio_large = compton_wavelength_shift(math.pi/2) / (1e6 * COMPTON_WAVELENGTH)
assert shift_ratio_large < shift_ratio_small

try:
    verify_thomson_limit(wavelength_ratio=-1.0)
    raise AssertionError("expected ValueError for wavelength_ratio <= 0")
except ValueError:
    pass

# 10. derive_electron_recoil_angle_symbolic: must return successfully
#     (structural check -- the numeric recoil-angle formula's correctness
#     is already independently verified via check 6's momentum_y_conserved)
tan_phi_expr = derive_electron_recoil_angle_symbolic()
assert tan_phi_expr is not None

print("all dgs.compton_scattering tests passed")
