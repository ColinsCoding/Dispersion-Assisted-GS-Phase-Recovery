"""Test dgs/wind_tunnel_aerodynamics.py: basic aero quantities, the
Reynolds-number wind-tunnel scaling problem (checked with real numbers,
not asserted), the Blasius boundary layer (solved by shooting, checked
against the classical f''(0)=0.33206 constant), and thin-airfoil lift vs.
angle of attack."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.wind_tunnel_aerodynamics import (
    dynamic_pressure, reynolds_number, mach_number, drag_force, lift_force,
    required_model_velocity_for_Re_match, required_model_density_for_Re_match,
    demonstrate_scaling_problem, solve_blasius, boundary_layer_edge_eta,
    boundary_layer_thickness_m, thin_airfoil_lift_coefficient,
    RHO_SEA_LEVEL, MU_AIR, SPEED_OF_SOUND_SEA_LEVEL,
)

# 1. dynamic_pressure / reynolds_number / mach_number: known relationships
assert abs(dynamic_pressure(1.225, 10.0) - 61.25) < 1e-9
assert abs(reynolds_number(1.225, 10.0, 1.0, 1.81e-5) - 1.225 * 10.0 / 1.81e-5) < 1e-6
assert abs(mach_number(343.0) - 1.0) < 1e-9   # V = speed of sound -> M=1 exactly

for fn, args in [(dynamic_pressure, (-1.0, 10.0)), (reynolds_number, (1.225, 10.0, -1.0, 1.81e-5)),
                  (mach_number, (10.0, -1.0))]:
    try:
        fn(*args)
        raise AssertionError(f"expected ValueError for {fn.__name__}{args}")
    except ValueError:
        pass

# 2. drag_force / lift_force: F = q*A*C
q = dynamic_pressure(1.225, 20.0)
assert abs(drag_force(1.225, 20.0, 2.0, 0.5) - q * 2.0 * 0.5) < 1e-9
assert abs(lift_force(1.225, 20.0, 2.0, 1.2) - q * 2.0 * 1.2) < 1e-9

# 3. required_model_velocity_for_Re_match / required_model_density_for_Re_match:
#    round-trip consistency -- computing Re back from the returned V (or
#    rho) must reproduce Re_target exactly
Re_target, L_model = 1e6, 0.3
V_needed = required_model_velocity_for_Re_match(Re_target, L_model)
assert abs(reynolds_number(RHO_SEA_LEVEL, V_needed, L_model) - Re_target) / Re_target < 1e-9

rho_needed = required_model_density_for_Re_match(Re_target, L_model, V_model=50.0)
assert abs(reynolds_number(rho_needed, 50.0, L_model) - Re_target) / Re_target < 1e-9

# 4. demonstrate_scaling_problem: a small model of a large, fast object
#    should need an unrealistic (supersonic) tunnel speed at atmospheric
#    density -- the actual documented claim, checked with real numbers
result = demonstrate_scaling_problem(L_full=10.0, V_full=60.0, L_model=0.5)
assert result["Mach_required_at_atmospheric_density"] > 1.0, "expected a supersonic requirement for this scale ratio"
assert result["needs_pressurized_or_cryogenic_tunnel"] is True
assert result["pressure_ratio_needed_to_match_Re_at_full_scale_velocity"] > 1.0

# a MILD scale reduction (model nearly as big as full-scale) should NOT
# require going supersonic -- contrast case, confirms the flag isn't
# always True regardless of input
mild_result = demonstrate_scaling_problem(L_full=10.0, V_full=10.0, L_model=8.0)
assert mild_result["needs_pressurized_or_cryogenic_tunnel"] is False

for bad_kwargs in [dict(L_full=10.0, V_full=60.0, L_model=15.0),   # model bigger than full-scale
                    dict(L_full=10.0, V_full=-1.0, L_model=1.0)]:
    try:
        demonstrate_scaling_problem(**bad_kwargs)
        raise AssertionError(f"expected ValueError for {bad_kwargs}")
    except ValueError:
        pass

print("dgs.wind_tunnel_aerodynamics: basic + scaling checks passed")

# 5. solve_blasius: the shooting method must recover the classical
#    f''(0)=0.33206 constant (Blasius 1908) to several decimal places --
#    an OUTPUT of the numerical solve, not hardcoded
profile = solve_blasius()
assert abs(profile["fpp0"] - 0.33206) < 1e-4
assert profile["fprime"][0] < 1e-9   # f'(0)=0, the no-slip condition
assert abs(profile["fprime"][-1] - 1.0) < 1e-6   # f'(eta_max)=1, free-stream recovered
assert np.all(np.diff(profile["fprime"]) >= -1e-9), "f' should be monotonically non-decreasing"

# 6. boundary_layer_edge_eta: must land near the classical ~5.0 constant
eta_edge = boundary_layer_edge_eta(profile)
assert 4.5 < eta_edge < 5.5, f"eta_edge={eta_edge}, expected close to the classical ~5.0"

try:
    boundary_layer_edge_eta(profile, threshold=1.5)   # f' never reaches 1.5
    raise AssertionError("expected RuntimeError for an unreachable threshold")
except RuntimeError:
    pass

# 7. boundary_layer_thickness_m: known scaling -- thickness grows with
#    sqrt(x) and shrinks with higher Re (higher V)
d1 = boundary_layer_thickness_m(x_m=1.0, rho=RHO_SEA_LEVEL, V=60.0, eta_edge=eta_edge)
d4 = boundary_layer_thickness_m(x_m=4.0, rho=RHO_SEA_LEVEL, V=60.0, eta_edge=eta_edge)
assert abs(d4 / d1 - 2.0) < 1e-6, "thickness should scale as sqrt(x): x=4 -> 2x the thickness of x=1"

d_fast = boundary_layer_thickness_m(x_m=1.0, rho=RHO_SEA_LEVEL, V=120.0, eta_edge=eta_edge)
assert d_fast < d1, "higher velocity (higher Re) should give a THINNER boundary layer"

try:
    boundary_layer_thickness_m(x_m=-1.0, rho=RHO_SEA_LEVEL, V=60.0)
    raise AssertionError("expected ValueError for x_m <= 0")
except ValueError:
    pass

print("dgs.wind_tunnel_aerodynamics: Blasius boundary-layer checks passed")

# 8. thin_airfoil_lift_coefficient: C_L(0)=0, small-angle C_L ~ 2*pi*alpha_rad,
#    monotonically increasing pre-stall, decreasing post-stall
assert abs(thin_airfoil_lift_coefficient(0.0)) < 1e-9
C_L_5 = thin_airfoil_lift_coefficient(5.0)
expected_small_angle = 2 * np.pi * np.radians(5.0)
assert abs(C_L_5 - expected_small_angle) / expected_small_angle < 0.02   # small-angle approx holds near alpha=5deg

alphas_pre_stall = np.array([0.0, 5.0, 10.0, 14.9])
C_Ls_pre_stall = thin_airfoil_lift_coefficient(alphas_pre_stall, stall_angle_deg=15.0)
assert np.all(np.diff(C_Ls_pre_stall) > 0), "C_L should increase monotonically before stall"

C_L_at_stall = thin_airfoil_lift_coefficient(15.0, stall_angle_deg=15.0)
C_L_past_stall = thin_airfoil_lift_coefficient(25.0, stall_angle_deg=15.0)
assert C_L_past_stall < C_L_at_stall, "C_L should drop off past the stall angle"

# symmetry: negative alpha should give (approximately) negative C_L, pre-stall
assert abs(thin_airfoil_lift_coefficient(-5.0) + thin_airfoil_lift_coefficient(5.0)) < 1e-9

# array input returns an array, scalar input returns a scalar
arr_out = thin_airfoil_lift_coefficient(np.array([0.0, 10.0]))
assert isinstance(arr_out, np.ndarray) and arr_out.shape == (2,)
scalar_out = thin_airfoil_lift_coefficient(10.0)
assert isinstance(scalar_out, float)

for bad_stall in (-5.0, 0.0, 90.0, 120.0):
    try:
        thin_airfoil_lift_coefficient(10.0, stall_angle_deg=bad_stall)
        raise AssertionError(f"expected ValueError for stall_angle_deg={bad_stall}")
    except ValueError:
        pass

print("dgs.wind_tunnel_aerodynamics: thin-airfoil checks passed")
print("all dgs.wind_tunnel_aerodynamics tests passed")
