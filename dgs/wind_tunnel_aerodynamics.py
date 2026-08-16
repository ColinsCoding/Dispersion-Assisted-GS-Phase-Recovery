"""Wind tunnel aerodynamics: the physics a wind-tunnel test actually has to
get right, not just "blow air at a model."

Three genuinely separate pieces:

1. BASIC QUANTITIES: dynamic pressure, Reynolds number, Mach number, and
   the lift/drag forces a balance measures from lift/drag COEFFICIENTS
   (the coefficients themselves are what a tunnel run is trying to
   determine -- everything here is the surrounding bookkeeping).

2. THE CENTRAL WIND-TUNNEL DESIGN PROBLEM: dynamic similarity requires
   matching Reynolds number between a scale model and the full-scale
   object. Naively cranking up the model's tunnel velocity to match Re
   at atmospheric density introduces a NEW mismatch (Mach number) --
   demonstrated numerically below with real numbers, not asserted -- which
   is the actual engineering reason pressurized or cryogenic wind tunnels
   (denser test gas instead of faster flow) exist.

3. THE BLASIUS LAMINAR BOUNDARY LAYER: solved numerically via a shooting
   method (not looked up), recovering the classical f''(0)=0.33206
   constant and the delta_99 ~ 5.0*x/sqrt(Re_x) boundary-layer-thickness
   rule as OUTPUTS of the numerical solve, not inputs to it.

Plus thin-airfoil lift vs. angle of attack (explicit degrees->radians
handling, since every aerodynamic coefficient formula needs radians but
every wind-tunnel angle-of-attack sweep is set up in degrees).

NumPy + SciPy only (scipy.integrate.solve_ivp for the Blasius ODE,
scipy.optimize.brentq for the shooting method). Education.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

RHO_SEA_LEVEL = 1.225      # kg/m^3, standard atmosphere
MU_AIR = 1.81e-5           # Pa*s, dynamic viscosity of air at ~15C
SPEED_OF_SOUND_SEA_LEVEL = 343.0   # m/s


# ── 1. Basic aerodynamic quantities ──────────────────────────────────────────

def dynamic_pressure(rho: float, V: float) -> float:
    """q = 0.5*rho*V^2."""
    if rho <= 0 or V < 0:
        raise ValueError(f"rho must be > 0 and V >= 0, got rho={rho}, V={V}")
    return 0.5 * rho * V**2


def reynolds_number(rho: float, V: float, L: float, mu: float = MU_AIR) -> float:
    """Re = rho*V*L/mu -- the ratio of inertial to viscous forces, the
    single most important dimensionless number in wind-tunnel testing."""
    if rho <= 0 or V < 0 or L <= 0 or mu <= 0:
        raise ValueError(f"rho, L, mu must be > 0 and V >= 0, got rho={rho}, V={V}, L={L}, mu={mu}")
    return rho * V * L / mu


def mach_number(V: float, speed_of_sound: float = SPEED_OF_SOUND_SEA_LEVEL) -> float:
    """M = V/a."""
    if speed_of_sound <= 0 or V < 0:
        raise ValueError(f"speed_of_sound must be > 0 and V >= 0, got a={speed_of_sound}, V={V}")
    return V / speed_of_sound


def drag_force(rho: float, V: float, A: float, C_D: float) -> float:
    """F_D = q*A*C_D."""
    if A <= 0:
        raise ValueError(f"A must be > 0, got {A}")
    return dynamic_pressure(rho, V) * A * C_D


def lift_force(rho: float, V: float, A: float, C_L: float) -> float:
    """F_L = q*A*C_L."""
    if A <= 0:
        raise ValueError(f"A must be > 0, got {A}")
    return dynamic_pressure(rho, V) * A * C_L


# ── 2. Reynolds-number scaling: the central wind-tunnel design problem ──────

def required_model_velocity_for_Re_match(Re_target: float, L_model: float,
                                          rho: float = RHO_SEA_LEVEL, mu: float = MU_AIR) -> float:
    """The tunnel velocity a scale model needs (at a given test-gas
    density) to match a target Reynolds number: V = Re*mu/(rho*L)."""
    if Re_target <= 0 or L_model <= 0 or rho <= 0 or mu <= 0:
        raise ValueError("Re_target, L_model, rho, mu must all be > 0")
    return Re_target * mu / (rho * L_model)


def required_model_density_for_Re_match(Re_target: float, L_model: float, V_model: float,
                                        mu: float = MU_AIR) -> float:
    """The test-gas density a scale model needs (at a FIXED, achievable
    tunnel velocity) to match a target Reynolds number: rho = Re*mu/(V*L)
    -- the pressurized/cryogenic-tunnel solution to the scaling problem,
    instead of the (often unrealistic) high-velocity solution above."""
    if Re_target <= 0 or L_model <= 0 or V_model <= 0 or mu <= 0:
        raise ValueError("Re_target, L_model, V_model, mu must all be > 0")
    return Re_target * mu / (V_model * L_model)


def demonstrate_scaling_problem(L_full: float, V_full: float, L_model: float,
                                rho: float = RHO_SEA_LEVEL, mu: float = MU_AIR,
                                speed_of_sound: float = SPEED_OF_SOUND_SEA_LEVEL,
                                mach_incompressible_limit: float = 0.3) -> dict:
    """CHECKED, not assumed: computes the full-scale Reynolds number, then
    the tunnel velocity a scale model needs (AT ATMOSPHERIC DENSITY) to
    match it -- and shows that velocity's implied Mach number, flagging
    whether it exceeds the incompressible-flow validity limit (Mach~0.3).
    A small model matching a large object's Reynolds number at atmospheric
    density routinely needs an UNREALISTIC (transonic/supersonic) tunnel
    speed -- the actual reason pressurized/cryogenic tunnels
    (required_model_density_for_Re_match instead) are used in practice."""
    if L_full <= 0 or V_full <= 0 or L_model <= 0 or L_model >= L_full:
        raise ValueError(f"need 0 < L_model < L_full and V_full > 0, got "
                          f"L_full={L_full}, V_full={V_full}, L_model={L_model}")
    Re_full = reynolds_number(rho, V_full, L_full, mu)
    V_model_at_same_density = required_model_velocity_for_Re_match(Re_full, L_model, rho, mu)
    Mach_required = mach_number(V_model_at_same_density, speed_of_sound)
    needs_pressurized_tunnel = Mach_required > mach_incompressible_limit

    rho_model_at_achievable_V = required_model_density_for_Re_match(
        Re_full, L_model, V_full, mu)   # keep model tunnel speed == full-scale speed
    pressure_ratio_needed = rho_model_at_achievable_V / rho   # ideal gas, fixed T: rho ~ P

    return {"Re_full": Re_full, "scale_ratio": L_model / L_full,
            "V_model_required_at_atmospheric_density": V_model_at_same_density,
            "Mach_required_at_atmospheric_density": Mach_required,
            "needs_pressurized_or_cryogenic_tunnel": needs_pressurized_tunnel,
            "pressure_ratio_needed_to_match_Re_at_full_scale_velocity": pressure_ratio_needed}


# ── 3. Blasius laminar boundary layer, solved by shooting ──────────────────

def _blasius_rhs(eta, y):
    """f''' + 0.5*f*f'' = 0, as a first-order system y=[f, f', f'']."""
    f, fp, fpp = y
    return [fp, fpp, -0.5 * f * fpp]


def _blasius_shoot_residual(fpp0: float, eta_max: float) -> float:
    """f'(eta_max) - 1, for a trial f''(0)=fpp0 -- the shooting-method
    residual driven to zero to satisfy the true boundary condition
    f'(eta->infinity)=1 (free-stream velocity recovered far from the
    plate)."""
    sol = solve_ivp(_blasius_rhs, [0, eta_max], [0.0, 0.0, fpp0],
                     max_step=0.01, rtol=1e-10, atol=1e-12)
    return sol.y[1, -1] - 1.0


def solve_blasius(eta_max: float = 10.0) -> dict:
    """Solves the Blasius laminar-flat-plate boundary-layer ODE by
    shooting: finds f''(0) via scipy.optimize.brentq such that the
    integrated solution satisfies f'(eta_max)=1, then returns the full
    (eta, f, f', f'') profile. The recovered f''(0) should match the
    classical literature constant 0.33206 -- an OUTPUT of this numerical
    solve, not hardcoded into it."""
    fpp0 = brentq(_blasius_shoot_residual, 0.1, 1.0, args=(eta_max,), xtol=1e-12)
    sol = solve_ivp(_blasius_rhs, [0, eta_max], [0.0, 0.0, fpp0],
                     max_step=0.01, rtol=1e-10, atol=1e-12, dense_output=True)
    eta = np.linspace(0, eta_max, 2000)
    f, fp, fpp = sol.sol(eta)
    return {"fpp0": fpp0, "eta": eta, "f": f, "fprime": fp, "fdoubleprime": fpp}


def boundary_layer_edge_eta(profile: dict, threshold: float = 0.99) -> float:
    """eta at which f'(eta) first reaches `threshold` (99% of free-stream
    velocity, the conventional boundary-layer EDGE definition) --
    recovered from the numerical profile, should be close to the
    classical delta_99 constant ~5.0."""
    idx = np.argmax(profile["fprime"] >= threshold)
    if idx == 0 and profile["fprime"][0] < threshold:
        raise RuntimeError(f"f' never reached {threshold} within eta_max={profile['eta'][-1]}")
    return float(profile["eta"][idx])


def boundary_layer_thickness_m(x_m: float, rho: float, V: float, mu: float = MU_AIR,
                               eta_edge: float = None) -> float:
    """delta_99(x) = eta_edge * x / sqrt(Re_x), Re_x = rho*V*x/mu -- the
    physical (dimensional) laminar boundary-layer thickness at distance x
    along a flat plate. `eta_edge` defaults to the value recovered by
    boundary_layer_edge_eta(solve_blasius()) if not supplied, so the
    classical "5.0" constant is DERIVED, not assumed, unless the caller
    wants to skip re-solving the ODE."""
    if x_m <= 0:
        raise ValueError(f"x_m must be > 0, got {x_m}")
    if eta_edge is None:
        eta_edge = boundary_layer_edge_eta(solve_blasius())
    Re_x = reynolds_number(rho, V, x_m, mu)
    return eta_edge * x_m / np.sqrt(Re_x)


# ── 4. Thin-airfoil lift vs. angle of attack ────────────────────────────────

def thin_airfoil_lift_coefficient(alpha_deg, stall_angle_deg: float = 15.0,
                                  post_stall_decay: float = 0.6):
    """Thin-airfoil theory: C_L = 2*pi*sin(alpha) for |alpha| <=
    stall_angle_deg (explicit degrees->radians conversion -- every
    coefficient formula needs radians, every wind-tunnel angle-of-attack
    sweep is set up in degrees). Beyond stall, a simple empirical
    post-stall decay model (C_L falls off linearly past the stall angle,
    NOT the thin-airfoil formula, which has no stall physics at all) --
    flagged separately so the two regimes aren't silently conflated."""
    alpha_deg = np.atleast_1d(np.asarray(alpha_deg, dtype=float))
    if stall_angle_deg <= 0 or stall_angle_deg >= 90:
        raise ValueError(f"stall_angle_deg must be in (0, 90), got {stall_angle_deg}")
    alpha_rad = np.radians(alpha_deg)
    C_L_attached = 2 * np.pi * np.sin(alpha_rad)

    stall_rad = np.radians(stall_angle_deg)
    C_L_at_stall = 2 * np.pi * np.sin(np.sign(alpha_rad) * stall_rad)
    beyond_stall = np.abs(alpha_rad) > stall_rad
    decay = np.maximum(0.0, 1.0 - post_stall_decay * (np.abs(alpha_rad) - stall_rad))
    C_L = np.where(beyond_stall, C_L_at_stall * decay, C_L_attached)

    return float(C_L[0]) if C_L.size == 1 else C_L


if __name__ == "__main__":
    print("=== 1. Basic quantities ===")
    rho, V, L = RHO_SEA_LEVEL, 60.0, 1.0
    print(f"  q = {dynamic_pressure(rho, V):.1f} Pa, Re = {reynolds_number(rho, V, L):.3e}, "
          f"M = {mach_number(V):.3f}")

    print("\n=== 2. The Reynolds-number scaling problem ===")
    result = demonstrate_scaling_problem(L_full=10.0, V_full=60.0, L_model=0.5)
    print(f"  full-scale Re = {result['Re_full']:.3e}  (scale ratio {result['scale_ratio']:.3f})")
    print(f"  model velocity needed at atmospheric density: {result['V_model_required_at_atmospheric_density']:.1f} m/s")
    print(f"  implied Mach number: {result['Mach_required_at_atmospheric_density']:.2f}  "
          f"(needs pressurized/cryogenic tunnel: {result['needs_pressurized_or_cryogenic_tunnel']})")
    print(f"  ALTERNATIVE: pressure ratio needed to match Re at the SAME (full-scale) velocity: "
          f"{result['pressure_ratio_needed_to_match_Re_at_full_scale_velocity']:.2f}x atmospheric")

    print("\n=== 3. Blasius boundary layer, solved by shooting ===")
    profile = solve_blasius()
    print(f"  f''(0) = {profile['fpp0']:.5f}  (classical literature value: 0.33206)")
    eta_edge = boundary_layer_edge_eta(profile)
    print(f"  eta at f'=0.99 (boundary-layer edge): {eta_edge:.3f}  (classical delta_99 constant: ~5.0)")
    delta = boundary_layer_thickness_m(x_m=1.0, rho=rho, V=V, eta_edge=eta_edge)
    print(f"  physical boundary-layer thickness at x=1.0m, V=60m/s: {delta*1000:.3f} mm")

    print("\n=== 4. Thin-airfoil lift vs. angle of attack ===")
    for alpha in (0.0, 5.0, 10.0, 15.0, 20.0, 90.0):
        C_L = thin_airfoil_lift_coefficient(alpha)
        print(f"  alpha={alpha:>5.1f} deg -> C_L = {C_L:.3f}")
