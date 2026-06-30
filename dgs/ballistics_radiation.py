"""Ballistics, radiation, quadrupole, and antenna theory.

AP PHYSICS C BRIDGE:
  Mechanics: projectile motion (constant-acceleration ODE -> parabola)
  E&M:       accelerating charge radiates (Larmor formula)
             dipole antenna (oscillating current = oscillating acceleration)
             quadrupole (two opposite dipoles -- much weaker radiation)

THE DIRAC DELTA AS IMPULSE:
  A bat hits a ball: F(t) = J * delta(t) where J is the impulse (N*s).
  Newton: m * a = J * delta(t) -> change in momentum = J (instantaneous).
  Same math as the GS algorithm's amplitude-replacement step: it is an
  instantaneous projection (impulse in function space) onto the constraint set.

QUADRUPOLE RADIATION:
  A monopole (charge q) does not radiate (charge conservation).
  A dipole (two opposite charges oscillating) radiates as (d^2p/dt^2)^2.
  A quadrupole (two opposite dipoles) radiates as (d^3Q/dt^3)^2.
  Gravitational waves are quadrupole radiation -- the LIGO detection was
  the quadrupole radiation from two merging black holes.

ANTENNA + IMPEDANCE MATCHING:
  A half-wave dipole (copper wire, length L = lambda/2) has radiation
  resistance R_rad = 73 Ohm. The feed line is typically 50 Ohm coax.
  Maximum power transfer: Z_antenna = Z_source* (conjugate match).
  Mismatch -> reflected power -> standing wave (VSWR).
  Skin depth: high-frequency current only flows in the outer layer of
  the copper wire -- delta = sqrt(2*rho / (omega*mu)).
"""
import numpy as np
import sympy as sp


# ── constants ─────────────────────────────────────────────────────────

g_m_s2 = 9.80665
c_m_s = 2.99792458e8
eps0 = 8.854187817e-12
mu0 = 4e-7 * np.pi
q_e = 1.602176634e-19
rho_copper = 1.68e-8    # resistivity of copper (Ohm*m)


# ── projectile motion ─────────────────────────────────────────────────

def projectile_range(v0_m_s, angle_deg, g=g_m_s2):
    """Horizontal range of a projectile launched at angle theta from flat ground.

    R = v0^2 * sin(2*theta) / g     (symmetric parabola, no drag)
    Max range at theta = 45 deg.
    """
    if v0_m_s < 0:
        raise ValueError("v0 must be non-negative")
    if not (0 < angle_deg < 90):
        raise ValueError("angle_deg must be in (0, 90)")
    theta = np.radians(angle_deg)
    R = v0_m_s**2 * np.sin(2 * theta) / g
    T = 2 * v0_m_s * np.sin(theta) / g
    H = v0_m_s**2 * np.sin(theta)**2 / (2 * g)
    return {"range_m": R, "time_of_flight_s": T, "max_height_m": H,
            "v0": v0_m_s, "angle_deg": angle_deg}


def projectile_trajectory(v0_m_s, angle_deg, g=g_m_s2, n_pts=500):
    """x(t) and y(t) arrays for a projectile until it hits the ground."""
    r = projectile_range(v0_m_s, angle_deg, g)
    T = r["time_of_flight_s"]
    t = np.linspace(0, T, n_pts)
    theta = np.radians(angle_deg)
    vx = v0_m_s * np.cos(theta)
    vy = v0_m_s * np.sin(theta)
    x = vx * t
    y = vy * t - 0.5 * g * t**2
    return {"t": t, "x": x, "y": y, "v0": v0_m_s, "angle_deg": angle_deg}


def symmetry_of_trajectory(v0_m_s, angle_deg, g=g_m_s2, n_pts=500):
    """Verify the left-right symmetry of the parabolic trajectory.

    The trajectory y(x) is symmetric about x = R/2 (the midpoint).
    y(x) = y(R - x) for all x. This is the same parity symmetry as
    even functions: the integrand is even, so we can integrate 0 to R/2
    and double it.
    Returns dict with max asymmetry (should be ~0 numerically).
    """
    tr = projectile_trajectory(v0_m_s, angle_deg, g, n_pts)
    x = tr["x"]
    y = tr["y"]
    R = x[-1]
    # for each x, find y at x and y at R-x
    from scipy.interpolate import interp1d
    y_of_x = interp1d(x, y, bounds_error=False, fill_value=0)
    x_test = np.linspace(0, R, 200)
    asymmetry = np.abs(y_of_x(x_test) - y_of_x(R - x_test))
    return {"max_asymmetry_m": float(np.max(asymmetry)),
            "is_symmetric": float(np.max(asymmetry)) < 1e-4 * R}


# ── impulse as Dirac delta ────────────────────────────────────────────

def impulse_delta(J_N_s, m_kg, dt=1e-3, t_max=0.1, n_pts=1000):
    """Simulate an impulsive force J * delta(t=0) on a mass m.

    The impulse approximation: replace F(t) by a very narrow rectangular
    pulse of height J/dt and width dt. As dt -> 0, this converges to
    J * delta(t). The resulting velocity change is delta_v = J/m.
    Returns t, v(t) showing the instantaneous jump.
    """
    if m_kg <= 0:
        raise ValueError("m_kg must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    t = np.linspace(-t_max/2, t_max, n_pts)
    v = np.zeros_like(t)
    # velocity step at t=0 (Heaviside response to delta input)
    v[t >= 0] = J_N_s / m_kg
    return {"t": t, "v": t, "delta_v": J_N_s / m_kg,
            "v_after": J_N_s / m_kg,
            "interpretation": "instantaneous velocity change = J/m (Newton impulse)"}


# ── Larmor radiation formula ──────────────────────────────────────────

def larmor_power(acceleration_m_s2, charge_C=q_e):
    """Power radiated by an accelerating charge (non-relativistic Larmor formula).

    P = q^2 * a^2 / (6 * pi * eps0 * c^3)

    An electron in a copper antenna oscillating at 1 GHz with amplitude
    1 nm has a = omega^2 * x ~ (2*pi*1e9)^2 * 1e-9 ~ 4e10 m/s^2.
    Each electron radiates ~1e-30 W. But 1 cm of copper wire has ~10^22
    free electrons contributing coherently -> macroscopic radiation.
    """
    P = charge_C**2 * acceleration_m_s2**2 / (6 * np.pi * eps0 * c_m_s**3)
    return {"power_W": P, "charge_C": charge_C,
            "acceleration_m_s2": acceleration_m_s2}


# ── dipole antenna ────────────────────────────────────────────────────

def dipole_radiation_resistance(length_m, wavelength_m):
    """Radiation resistance of a short dipole antenna.

    R_rad = 80 * pi^2 * (L/lambda)^2   (Ohm, short dipole L << lambda)
    For half-wave dipole (L = lambda/2): R_rad ~ 73 Ohm (numerical).
    """
    if length_m <= 0 or wavelength_m <= 0:
        raise ValueError("length_m and wavelength_m must be positive")
    ratio = length_m / wavelength_m
    if ratio < 0.4:
        # short dipole approximation
        R_rad = 80 * np.pi**2 * ratio**2
    else:
        # half-wave dipole numerical result
        R_rad = 73.0 * (np.sinc(ratio - 0.5) + 0.5)**2  # rough fit
    return {"R_rad_ohm": R_rad, "L_over_lambda": ratio,
            "regime": "short dipole" if ratio < 0.4 else "half-wave"}


def impedance_matching_vswr(Z_source, Z_load):
    """Voltage Standing Wave Ratio (VSWR) and reflection coefficient.

    Gamma = (Z_L - Z_S) / (Z_L + Z_S)   (reflection coefficient)
    VSWR = (1 + |Gamma|) / (1 - |Gamma|)
    Power delivered = (1 - |Gamma|^2) * P_available

    Perfect match: Z_L = Z_S* -> Gamma = 0 -> VSWR = 1 -> 100% power transfer.
    50 Ohm coax feeding 73 Ohm dipole: VSWR = 1.46 -> 96.5% power delivered.
    """
    Z_S = complex(Z_source)
    Z_L = complex(Z_load)
    if abs(Z_S + Z_L) == 0:
        raise ValueError("Z_S + Z_L cannot be zero")
    gamma = (Z_L - Z_S) / (Z_L + Z_S)
    mag_gamma = abs(gamma)
    if mag_gamma >= 1:
        vswr = float("inf")
    else:
        vswr = (1 + mag_gamma) / (1 - mag_gamma)
    power_fraction = 1 - mag_gamma**2
    return {"reflection_coefficient": gamma,
            "mag_gamma": mag_gamma,
            "VSWR": vswr,
            "power_fraction_delivered": power_fraction,
            "return_loss_dB": -20 * np.log10(mag_gamma) if mag_gamma > 0 else float("inf")}


def skin_depth_copper(freq_Hz):
    """Skin depth in copper at given frequency: delta = sqrt(2*rho / (omega*mu)).

    At high frequency, current only flows in the outer layer of the wire.
    Copper wire at 1 GHz: skin depth ~ 2.1 um. The wire behaves as a hollow tube.
    This increases effective resistance and is why RF engineers care about
    conductor surface finish and plating.
    """
    if freq_Hz <= 0:
        raise ValueError("freq_Hz must be positive")
    mu_copper = mu0   # approximately mu0 for copper (non-magnetic)
    omega = 2 * np.pi * freq_Hz
    delta = np.sqrt(2 * rho_copper / (omega * mu_copper))
    return {"skin_depth_m": delta, "skin_depth_um": delta * 1e6,
            "freq_Hz": freq_Hz, "rho_copper": rho_copper}


# ── quadrupole moment ─────────────────────────────────────────────────

def quadrupole_moment_sympy():
    """Electric quadrupole moment tensor in SymPy (Griffiths Ch 3 notation).

    Q_ij = integral (3*r_i*r_j - r^2 * delta_ij) * rho(r) dV

    The quadrupole radiation power goes as (d^3 Q/dt^3)^2 vs the dipole
    (d^2 p/dt^2)^2. Quadrupoles radiate much weaker (extra factor of v/c).
    Gravitational waves are quadrupole radiation because there is no
    gravitational dipole (conservation of momentum -> dipole term = 0).
    """
    x, y, z, r = sp.symbols('x y z r', real=True)
    rho = sp.Function('rho')(x, y, z)
    dV = sp.Symbol('dV')

    # 3x3 quadrupole tensor component Q_xx
    Q_xx = sp.Symbol('integral') * (3*x**2 - r**2) * rho * dV
    Q_xy = sp.Symbol('integral') * (3*x*y) * rho * dV

    return {
        "Q_xx": sp.Eq(sp.Symbol('Q_xx'), (3*x**2 - r**2) * rho * dV),
        "Q_xy": sp.Eq(sp.Symbol('Q_xy'), 3*x*y * rho * dV),
        "Dipole_power":
            sp.Eq(sp.Symbol('P_dipole'),
                  sp.Symbol('p_ddot')**2 /
                  (6*sp.pi*sp.Symbol('epsilon_0')*sp.Symbol('c')**3)),
        "Quadrupole_radiate_weaker":
            sp.Eq(sp.Symbol('P_quad/P_dip'),
                  (sp.Symbol('v')/sp.Symbol('c'))**2),
    }


def ballistics_radiation_sympy_5():
    """Five key equations: projectile, impulse, Larmor, antenna, quadrupole."""
    v0, theta, g_s = sp.symbols('v_0 theta g', positive=True)
    J_s, m_s = sp.symbols('J m', positive=True)
    q_s, a_s, c_s = sp.symbols('q a c', positive=True)
    L, lam = sp.symbols('L lambda', positive=True)
    Z_L, Z_S = sp.symbols('Z_L Z_S')

    return {
        "Projectile_range":
            sp.Eq(sp.Symbol('R'), v0**2 * sp.sin(2*theta) / g_s),
        "Impulse_delta":
            sp.Eq(sp.Symbol('Delta_v'), J_s / m_s),
        "Larmor_power":
            sp.Eq(sp.Symbol('P_rad'),
                  q_s**2 * a_s**2 /
                  (6 * sp.pi * sp.Symbol('epsilon_0') * c_s**3)),
        "Short_dipole_R_rad":
            sp.Eq(sp.Symbol('R_rad'),
                  80 * sp.pi**2 * (L/lam)**2),
        "Reflection_coefficient":
            sp.Eq(sp.Symbol('Gamma'),
                  (Z_L - Z_S) / (Z_L + Z_S)),
    }


if __name__ == "__main__":
    print("=== Projectile motion ===")
    for angle in [30, 45, 60]:
        r = projectile_range(100, angle)
        print(f"  v0=100 m/s, theta={angle} deg: R={r['range_m']:.1f} m, "
              f"H={r['max_height_m']:.1f} m")

    print("\n=== Trajectory symmetry check ===")
    from scipy.interpolate import interp1d
    sym = symmetry_of_trajectory(50, 45)
    print(f"  max asymmetry = {sym['max_asymmetry_m']:.2e} m  "
          f"-> symmetric: {sym['is_symmetric']}")

    print("\n=== Larmor radiation: oscillating electron at 1 GHz ===")
    omega = 2 * np.pi * 1e9
    x_amp = 1e-9   # 1 nm oscillation amplitude
    a = omega**2 * x_amp
    lp = larmor_power(a)
    print(f"  a = {a:.2e} m/s^2,  P = {lp['power_W']:.2e} W per electron")

    print("\n=== Half-wave dipole at 2.4 GHz (WiFi) ===")
    lam_wifi = c_m_s / 2.4e9
    L_dipole = lam_wifi / 2
    dr = dipole_radiation_resistance(L_dipole, lam_wifi)
    print(f"  L = {L_dipole*100:.1f} cm, R_rad = {dr['R_rad_ohm']:.1f} Ohm")

    print("\n=== Impedance matching: 50 Ohm coax -> 73 Ohm dipole ===")
    vm = impedance_matching_vswr(50, 73)
    print(f"  VSWR = {vm['VSWR']:.2f}, power delivered = {vm['power_fraction_delivered']*100:.1f}%")

    print("\n=== Skin depth in copper at 1 GHz ===")
    sd = skin_depth_copper(1e9)
    print(f"  delta = {sd['skin_depth_um']:.2f} um")

    print("\n=== SymPy 5 ===")
    for k, eq in ballistics_radiation_sympy_5().items():
        print(f"  {k}: {eq}")
