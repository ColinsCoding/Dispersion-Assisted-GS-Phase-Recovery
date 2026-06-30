"""Spring-mass-damper suspension mechanics.

Quarter-car model, transfer functions, frequency response, resonance.

PHYSICS:
  Single DOF:  m*x'' + c*x' + k*x = F(t)
  Natural frequency:  omega_n = sqrt(k/m)
  Damping ratio:      zeta = c / (2*sqrt(k*m))
  Damped frequency:   omega_d = omega_n*sqrt(1 - zeta^2)

  Transfer function (Laplace domain):
    H(s) = X(s)/F(s) = 1 / (m*s^2 + c*s + k)
           = (1/m) / (s^2 + 2*zeta*omega_n*s + omega_n^2)

  Frequency response:  H(i*omega) = 1/k * 1/sqrt((1-r^2)^2 + (2*zeta*r)^2)
  where r = omega/omega_n (frequency ratio)

  QUARTER-CAR MODEL:
    Two masses: sprung (body m_s) on suspension spring k_s, damper c_s
                unsprung (wheel m_u) on tire spring k_t
    Road input y_r(t) excites the system.

  CONNECTION TO DISPERSIVE FOURIER TRANSFORM (JALALI):
    The suspension is a mechanical analog of the GVD fiber:
    - Road roughness spectrum  <->  optical pulse spectrum
    - Transfer function H(s)  <->  GVD H(omega)=exp(i*beta2*L*omega^2/2)
    - Ride quality measurement <->  temporal DFT measurement
    - Active suspension (feedback) <-> GS phase recovery (iterative correction)
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple


# ── Single DOF spring-mass-damper ─────────────────────────────────────────────
def sdof_params(k: float, m: float, c: float) -> Dict:
    """Compute natural frequency, damping ratio, Q for single-DOF system.

    k : spring stiffness (N/m)
    m : mass (kg)
    c : viscous damping (N*s/m)
    """
    omega_n = np.sqrt(k / m)
    zeta    = c / (2 * np.sqrt(k * m))
    omega_d = omega_n * np.sqrt(max(0, 1 - zeta**2))
    Q       = 1 / (2 * zeta) if zeta > 0 else np.inf
    f_n     = omega_n / (2 * np.pi)
    T_n     = 1 / f_n
    # Static deflection under gravity
    delta_st = m * 9.81 / k
    return {
        "omega_n":   omega_n,     # natural frequency (rad/s)
        "f_n":       f_n,         # natural frequency (Hz)
        "T_n":       T_n,         # natural period (s)
        "zeta":      zeta,        # damping ratio
        "omega_d":   omega_d,     # damped natural frequency (rad/s)
        "Q":         Q,           # quality factor
        "c_cr":      2 * np.sqrt(k * m),   # critical damping
        "delta_st":  delta_st,    # static deflection (m)
        "k": k, "m": m, "c": c,
        "regime": ("underdamped" if zeta < 1 else
                   "critically_damped" if abs(zeta-1) < 1e-10 else "overdamped"),
    }


def frequency_response(omega_n: float, zeta: float,
                        omega_arr: np.ndarray) -> Dict:
    """Frequency response function H(omega) for SDOF system.

    |H(omega)| = 1/k * 1/sqrt((1-r^2)^2 + (2*zeta*r)^2)
    phase(omega) = -arctan(2*zeta*r / (1-r^2))

    r = omega/omega_n (frequency ratio)
    """
    r = omega_arr / omega_n
    denom_sq = (1 - r**2)**2 + (2 * zeta * r)**2
    H_mag   = 1.0 / np.sqrt(denom_sq)   # normalized (k=1)
    H_phase = -np.arctan2(2 * zeta * r, 1 - r**2)
    # Resonance peak: at r=sqrt(1-2*zeta^2) for zeta < 1/sqrt(2)
    if zeta < 1/np.sqrt(2):
        r_peak  = np.sqrt(1 - 2*zeta**2)
        H_peak  = 1 / (2 * zeta * np.sqrt(1 - zeta**2))
    else:
        r_peak, H_peak = 0.0, 1.0   # overdamped: no resonance peak
    return {
        "omega":   omega_arr,
        "r":       r,
        "H_mag":   H_mag,
        "H_mag_dB": 20 * np.log10(H_mag + 1e-300),
        "H_phase": H_phase,
        "r_peak":  r_peak,
        "H_peak":  H_peak,
        "H_peak_dB": 20 * np.log10(H_peak),
        "omega_peak": r_peak * omega_n,
    }


def step_response_sdof(k: float, m: float, c: float,
                        F0: float = 1.0,
                        t_end: float = None,
                        n_pts: int = 1000) -> Dict:
    """Step response x(t) for unit step force F = F0 * u(t).

    Exact analytic solution (no numerical integration needed).
    """
    p = sdof_params(k, m, c)
    omega_n, zeta, omega_d = p["omega_n"], p["zeta"], p["omega_d"]
    x_st = F0 / k    # static equilibrium

    if t_end is None:
        t_end = 10 / omega_n if omega_n > 0 else 10.0
    t = np.linspace(0, t_end, n_pts)

    if zeta < 1:
        # Underdamped: x(t) = x_st * [1 - e^{-zeta*wn*t}*(cos(wd*t) + zeta/sqrt(1-zeta^2)*sin(wd*t))]
        C = zeta / np.sqrt(1 - zeta**2)
        x = x_st * (1 - np.exp(-zeta*omega_n*t) *
                    (np.cos(omega_d*t) + C*np.sin(omega_d*t)))
    elif abs(zeta - 1) < 1e-10:
        # Critically damped: x(t) = x_st * [1 - (1+omega_n*t)*e^{-omega_n*t}]
        x = x_st * (1 - (1 + omega_n*t) * np.exp(-omega_n*t))
    else:
        # Overdamped: x(t) = x_st * [1 - ...] using hyperbolic functions
        beta = omega_n * np.sqrt(zeta**2 - 1)
        r1   = -zeta*omega_n + beta
        r2   = -zeta*omega_n - beta
        x = x_st * (1 - (r2*np.exp(r1*t) - r1*np.exp(r2*t)) / (r2 - r1))

    v = np.gradient(x, t)
    E_kinetic = 0.5 * m * v**2
    E_potential = 0.5 * k * (x - x_st)**2
    return {
        "t": t, "x": x, "v": v,
        "x_static": x_st,
        "overshoot": (x.max() - x_st) / x_st if zeta < 1 else 0.0,
        "E_kinetic": E_kinetic,
        "E_potential": E_potential,
        **p,
    }


# ── Quarter-car model ─────────────────────────────────────────────────────────
def quarter_car_params(m_s: float = 300.0,
                        m_u: float = 30.0,
                        k_s: float = 15000.0,
                        c_s: float = 1500.0,
                        k_t: float = 150000.0) -> Dict:
    """Quarter-car 2-DOF model parameters.

    m_s  : sprung mass (car body, kg)   default 300 kg
    m_u  : unsprung mass (wheel+axle, kg) default 30 kg
    k_s  : suspension spring (N/m)      default 15 kN/m
    c_s  : suspension damper (N*s/m)    default 1.5 kN*s/m
    k_t  : tire stiffness (N/m)         default 150 kN/m

    Returns natural frequencies, damping ratios, ride/handling trade.
    """
    # Uncoupled natural frequencies (approximate decoupling)
    omega_body  = np.sqrt(k_s / m_s)   # body bounce ~1-2 Hz
    omega_wheel = np.sqrt((k_s + k_t) / m_u)  # wheel hop ~10-15 Hz
    zeta_body   = c_s / (2 * np.sqrt(k_s * m_s))
    # Full 2x2 eigenvalue problem
    # EOM: [m_s  0 ] [x_s''] + [c_s  -c_s] [x_s'] + [k_s    -k_s  ] [x_s] = [0    ]
    #      [0   m_u] [x_u'']   [-c_s  c_s] [x_u']   [-k_s  k_s+k_t] [x_u]   [k_t*y]
    M = np.array([[m_s, 0], [0, m_u]])
    C = np.array([[c_s, -c_s], [-c_s, c_s]])
    K = np.array([[k_s, -k_s], [-k_s, k_s+k_t]])
    # Eigenvalues (undamped): det(K - omega^2*M) = 0
    from numpy.linalg import eig
    Minv = np.diag(1/np.diag(M))
    vals, vecs = eig(Minv @ K)
    omega_modes = np.sqrt(np.real(vals))
    return {
        "m_s": m_s, "m_u": m_u, "k_s": k_s, "c_s": c_s, "k_t": k_t,
        "omega_body":     omega_body,
        "omega_wheel":    omega_wheel,
        "f_body_Hz":      omega_body / (2*np.pi),
        "f_wheel_Hz":     omega_wheel / (2*np.pi),
        "zeta_body":      zeta_body,
        "omega_modes":    np.sort(omega_modes),
        "f_modes_Hz":     np.sort(omega_modes) / (2*np.pi),
        "ride_quality":   "good" if 0.2 <= zeta_body <= 0.4 else "poor",
        "handling":       "stiff" if c_s > 2000 else "soft",
    }


def road_psd_response(omega_n_body: float,
                       zeta: float,
                       omega_arr: np.ndarray,
                       road_roughness_class: str = "B") -> Dict:
    """RMS acceleration response to road roughness PSD.

    ISO 8608 road roughness PSD: G_z(omega) = G_0 * (omega/omega_0)^{-w}
    Class A (smooth highway), B (good road), C (average), D (poor)

    G_0 * 1e-6 m^2/(rad/m):
    A: 1, B: 4, C: 16, D: 64

    RMS vertical acceleration = sqrt(int |H_a(omega)|^2 * G_z(omega) dOmega)
    """
    G0_map = {"A": 1e-6, "B": 4e-6, "C": 16e-6, "D": 64e-6}
    G0 = G0_map.get(road_roughness_class, 4e-6)

    # Road PSD (spatial, converted to temporal via vehicle speed v=20 m/s)
    v_kmh = 20.0   # m/s
    G_road = G0 * (omega_arr / (2*np.pi))**(-2)  # von Karman model

    # Body acceleration transfer function H_a = -omega^2 * H_x
    r = omega_arr / omega_n_body
    H_x = 1.0 / np.sqrt((1 - r**2)**2 + (2*zeta*r)**2)
    H_a = omega_arr**2 * H_x   # acceleration magnification

    PSD_acc = H_a**2 * G_road
    domega  = omega_arr[1] - omega_arr[0]
    rms_acc = float(np.sqrt(np.trapezoid(PSD_acc, omega_arr)))

    return {
        "omega":       omega_arr,
        "G_road":      G_road,
        "H_mag":       H_x,
        "H_accel":     H_a,
        "PSD_acc":     PSD_acc,
        "rms_acc_g":   rms_acc / 9.81,
        "road_class":  road_roughness_class,
        "ISO_limit_g": 0.315,   # ISO 2631 comfort limit (8hr exposure)
        "comfortable": rms_acc / 9.81 < 0.315,
    }


# ── Transfer function (Laplace domain) ───────────────────────────────────────
def transfer_function_sympy() -> Dict:
    """Symbolic transfer function H(s) = X(s)/F(s) for SDOF system."""
    s, m, c, k = sp.symbols("s m c k", positive=True)
    omega_n = sp.sqrt(k / m)
    zeta    = c / (2 * sp.sqrt(k * m))
    # Denominator: m*s^2 + c*s + k
    denom   = m*s**2 + c*s + k
    H_s     = sp.Rational(1, 1) / denom
    # Normalized form
    H_norm  = (1/m) / (s**2 + 2*zeta*omega_n*s + omega_n**2)
    # Poles (characteristic equation roots)
    poles   = sp.solve(denom, s)
    return {
        "H_s":      sp.Eq(sp.Symbol("H(s)"), H_s),
        "H_norm":   sp.Eq(sp.Symbol("H_norm(s)"), H_norm),
        "poles":    poles,
        "omega_n":  sp.Eq(sp.Symbol("omega_n"), omega_n),
        "zeta":     sp.Eq(sp.Symbol("zeta"), zeta),
    }


def suspension_sympy_5() -> Dict:
    """5 key suspension / vibration equations."""
    m, c, k, s = sp.symbols("m c k s", positive=True)
    omega, omega_n, zeta, r = sp.symbols("omega omega_n zeta r", positive=True)
    t = sp.Symbol("t", positive=True)
    x = sp.Function("x")
    # 1. EOM
    eq1 = sp.Eq(m*x(t).diff(t,2) + c*x(t).diff(t) + k*x(t), sp.Function("F")(t))
    # 2. Natural frequency
    eq2 = sp.Eq(omega_n, sp.sqrt(k/m))
    # 3. Transfer function
    eq3 = sp.Eq(sp.Symbol("H(s)"), 1/(m*s**2 + c*s + k))
    # 4. Frequency response magnitude
    eq4 = sp.Eq(sp.Symbol("|H(omega)|"),
                1/(k*sp.sqrt((1-r**2)**2 + (2*zeta*r)**2)))
    # 5. Resonance peak (at r = sqrt(1 - 2*zeta^2))
    eq5 = sp.Eq(sp.Symbol("H_peak"),
                1/(2*zeta*sp.sqrt(1-zeta**2)*k))
    return {
        "EOM":              eq1,
        "Natural_freq":     eq2,
        "Transfer_fn_H(s)": eq3,
        "Freq_response":    eq4,
        "Resonance_peak":   eq5,
    }


if __name__ == "__main__":
    sp.init_printing(use_latex=False)

    print("=== Quarter-Car Suspension Model ===")
    qc = quarter_car_params()
    print(f"  Body bounce:  f = {qc['f_body_Hz']:.2f} Hz, "
          f"zeta = {qc['zeta_body']:.3f} ({qc['ride_quality']})")
    print(f"  Wheel hop:    f = {qc['f_wheel_Hz']:.2f} Hz")
    print(f"  Normal modes: {qc['f_modes_Hz'].round(2)} Hz")

    print("\n=== SDOF Step Response (BMW-class suspension) ===")
    res = step_response_sdof(k=15000, m=300, c=1500)
    print(f"  omega_n = {res['omega_n']:.2f} rad/s = {res['f_n']:.2f} Hz")
    print(f"  zeta    = {res['zeta']:.3f}  ({res['regime']})")
    print(f"  Q       = {res['Q']:.2f}")
    print(f"  Overshoot = {res['overshoot']*100:.1f}%")

    print("\n=== Frequency Response at resonance ===")
    omega_arr = np.linspace(0.1, 40, 2000)
    fr = frequency_response(res["omega_n"], res["zeta"], omega_arr)
    print(f"  Peak at r = {fr['r_peak']:.3f}, H_peak = {fr['H_peak']:.2f} = {fr['H_peak_dB']:.1f} dB")

    print("\n=== Road PSD Ride Quality ===")
    road = road_psd_response(res["omega_n"], res["zeta"],
                              np.linspace(0.5, 50, 500), "B")
    print(f"  Road class B: RMS accel = {road['rms_acc_g']:.4f} g")
    print(f"  ISO 2631 comfortable: {road['comfortable']}")

    print("\n=== 5 SymPy Equations ===")
    for name, eq in suspension_sympy_5().items():
        print(f"  [{name}]  {eq}")
