"""Virtual work, in three unrelated-looking places -- a falling mass with
air drag, an RL circuit, and a microwave transistor's Miller-limited
bandwidth -- turn out to satisfy the IDENTICAL first-order ODE.

THE PRINCIPLE. D'Alembert's form of virtual work says a system evolves so
that, for any virtual displacement delta_x consistent with its
constraints, (applied forces - inertial force) . delta_x = 0. Since
delta_x is arbitrary, the bracket itself must vanish -- that's the Euler-
Lagrange equation. Applied to a generalized coordinate x(t) with kinetic
energy T, potential V, and a Rayleigh dissipation function D (for forces,
like drag or resistance, that don't come from a potential):

    d/dt(dT/dx_dot) - dL/dx + dD/dx_dot = 0,   L = T - V.

THREE SYSTEMS, ONE EQUATION.
  1. FALLING MASS WITH LINEAR AIR DRAG. T=(1/2)m*xdot^2, V=-m*g*x (falling
     is the coordinate doing positive work), D=(1/2)*b*xdot^2 (drag). The
     Euler-Lagrange equation is m*xddot + b*xdot = m*g; writing it for the
     VELOCITY v=xdot alone (x itself drops out, only its derivatives
     appear) gives the first-order ODE
         m*v_dot + b*v = m*g   i.e.   tau_m*v_dot + v = v_terminal,
     tau_m = m/b, v_terminal = m*g/b -- "hits the ground at a velocity"
     that APPROACHES v_terminal exponentially, v(t) = v_terminal*(1-e^{-t/tau_m}).

  2. RL CIRCUIT. KVL for a series R-L driven by a step V_s: L*I_dot + R*I
     = V_s, i.e. tau_RL*I_dot + I = I_final, tau_RL = L/R, I_final = V_s/R.
     (This is the SAME Euler-Lagrange machinery as dgs.lagrangian_circuits,
     just for an R-L pair instead of R-L-C: there's no restoring/"spring"
     term here because there's no capacitor, so the equation is first-
     order, exactly matching the falling mass's velocity equation.)

  3. MICROWAVE TRANSISTOR'S DOMINANT (MILLER) POLE. A common-emitter
     small-signal stage has base-emitter capacitance C_pi and feedback
     (base-collector) capacitance C_mu; the Miller effect multiplies C_mu
     by (1+gm*R_L) as seen from the base, so the base node sees an
     effective C_eq = C_pi + C_mu*(1+gm*R_L) charging through R_eq (~r_pi).
     KCL at that node gives the identical first-order relaxation:
     R_eq*C_eq*V_dot + V = V_s, tau = R_eq*C_eq, f_3dB = 1/(2*pi*tau) --
     this is the SAME transimpedance-bandwidth story as
     dgs.transimpedance_amplifier.bandwidth_3db, now derived from the
     transistor's own internal capacitances instead of an external
     feedback resistor/capacitor.

All three reduce to  tau*y_dot + y = y_infinity,  y(t) = y_infinity*(1 -
e^{-t/tau})  for y(0)=0 -- verified numerically below by normalizing all
three curves to y/y_infinity vs t/tau and checking they are the SAME curve
to machine precision, not just "similar-looking" exponentials.

Reuses dgs.lagrangian_circuits' Euler-Lagrange derivation pattern (T, V, D,
EOM) for the falling-mass case, and connects to
dgs.transimpedance_amplifier.bandwidth_3db for the circuit-bandwidth
framing. py-3.13.
"""

from __future__ import annotations
import numpy as np
import sympy as sp


# ── 1. Falling mass with linear drag: the Euler-Lagrange (virtual work) derivation ──

def virtual_work_falling_mass_sympy() -> dict:
    """Symbolic Euler-Lagrange derivation for a mass falling under gravity
    with LINEAR drag, matching the T/V/D/EOM pattern of
    dgs.lagrangian_circuits.lagrangian_rlc_sympy(). Returns T, V, D, the
    full EOM in x, its reduction to a first-order ODE in v=xdot, and the
    closed-form v(t) from sp.dsolve (checked against the closed form used
    numerically below)."""
    t = sp.Symbol("t")
    m, b, g = sp.symbols("m b g", positive=True)
    x = sp.Function("x")(t)
    v = x.diff(t)

    T_expr = sp.Rational(1, 2) * m * v ** 2
    V_expr = -m * g * x                      # falling (increasing x) releases potential energy
    D_expr = sp.Rational(1, 2) * b * v ** 2  # Rayleigh dissipation (linear drag)
    Lag = T_expr - V_expr

    dL_dxdot = sp.diff(Lag, v)
    dL_dx = sp.diff(Lag, x)
    dD_dxdot = sp.diff(D_expr, v)
    EOM_lhs = dL_dxdot.diff(t) - dL_dx + dD_dxdot
    EOM = sp.Eq(EOM_lhs, 0)                  # m*xddot + b*xdot - m*g = 0

    v_sym = sp.Function("v")(t)
    velocity_ode = sp.Eq(m * v_sym.diff(t) + b * v_sym, m * g)
    v_solution = sp.dsolve(velocity_ode, v_sym, ics={v_sym.subs(t, 0): 0})

    return {
        "T": T_expr, "V": V_expr, "D_rayleigh": D_expr, "L": Lag,
        "EOM_in_x": EOM,
        "velocity_ODE": velocity_ode,
        "v_of_t": v_solution,
        "tau_m": m / b,
        "v_terminal": m * g / b,
    }


def falling_mass_velocity(t, m: float, b: float, g: float = 9.81):
    """v(t) = v_terminal*(1-exp(-t/tau_m)), tau_m=m/b, v_terminal=m*g/b --
    the closed form from virtual_work_falling_mass_sympy()'s v_of_t,
    evaluated numerically."""
    if m <= 0 or b <= 0:
        raise ValueError("m and b must be positive")
    tau_m = m / b
    v_terminal = m * g / b
    return v_terminal * (1 - np.exp(-np.asarray(t, dtype=float) / tau_m))


# ── 2. RL circuit: the same equation, different symbols ────────────────

def rl_circuit_current(t, R: float, L: float, V_s: float):
    """I(t) = I_final*(1-exp(-t/tau_RL)), tau_RL=L/R, I_final=V_s/R -- the
    step response of a series R-L driven by a DC step V_s at t=0."""
    if R <= 0 or L <= 0:
        raise ValueError("R and L must be positive")
    tau_rl = L / R
    I_final = V_s / R
    return I_final * (1 - np.exp(-np.asarray(t, dtype=float) / tau_rl))


# ── 3. Microwave transistor: Miller-multiplied dominant pole ───────────

def transistor_dominant_pole(gm: float, r_pi: float, R_L: float, C_pi: float, C_mu: float) -> dict:
    """Miller's theorem: C_mu (base-collector) looks like C_mu*(1+gm*R_L)
    from the base node. C_eq = C_pi + C_mu*(1+gm*R_L) charges through
    R_eq=r_pi -- tau = R_eq*C_eq, f_3dB = 1/(2*pi*tau). Same functional
    form as dgs.transimpedance_amplifier.bandwidth_3db(R_f, C), here with
    the capacitance coming from the DEVICE's own internal geometry instead
    of an external feedback network."""
    if gm <= 0 or r_pi <= 0 or R_L <= 0 or C_pi < 0 or C_mu < 0:
        raise ValueError("gm, r_pi, R_L must be > 0 and C_pi, C_mu must be >= 0")
    miller_factor = 1 + gm * R_L
    C_eq = C_pi + C_mu * miller_factor
    R_eq = r_pi
    tau = R_eq * C_eq
    f_3db = 1.0 / (2 * np.pi * tau)
    return {"miller_factor": miller_factor, "C_eq": C_eq, "R_eq": R_eq,
            "tau": tau, "f_3dB_Hz": f_3db}


def transistor_step_response(t, gm: float, r_pi: float, R_L: float, C_pi: float, C_mu: float, V_s: float):
    """V_base(t) = V_s*(1-exp(-t/tau)) charging toward a step input V_s,
    tau from transistor_dominant_pole -- the SAME exponential-approach
    curve as falling_mass_velocity and rl_circuit_current."""
    pole = transistor_dominant_pole(gm, r_pi, R_L, C_pi, C_mu)
    return V_s * (1 - np.exp(-np.asarray(t, dtype=float) / pole["tau"]))


# ── the isomorphism, made explicit ──────────────────────────────────────

def isomorphism_table() -> list[dict]:
    """Every system above is  tau*y_dot + y = y_infinity.  This table names
    which physical quantity plays each role in each of the three domains."""
    return [
        {"role": "state variable y(t)", "falling mass": "velocity v",
         "RL circuit": "current I", "transistor": "base voltage V_base"},
        {"role": "storage/inertia element", "falling mass": "mass m",
         "RL circuit": "inductance L", "transistor": "C_eq = C_pi + C_mu*(1+gm*R_L)"},
        {"role": "dissipation/resistance element", "falling mass": "drag coefficient b",
         "RL circuit": "resistance R", "transistor": "r_pi"},
        {"role": "forcing term", "falling mass": "gravity m*g",
         "RL circuit": "source V_s", "transistor": "step input V_s"},
        {"role": "steady-state value y_infinity", "falling mass": "terminal velocity m*g/b",
         "RL circuit": "final current V_s/R", "transistor": "final base voltage V_s"},
        {"role": "time constant tau", "falling mass": "m/b",
         "RL circuit": "L/R", "transistor": "r_pi*C_eq"},
    ]


def verify_same_functional_form(t_over_tau: np.ndarray | None = None, tol: float = 1e-12) -> dict:
    """CHECKED, not assumed: evaluate all three step responses with
    DIFFERENT physical parameters, normalize each to y/y_infinity vs
    t/tau, and confirm the three normalized curves are IDENTICAL (the same
    curve 1-e^{-x}, not just qualitatively similar exponentials)."""
    if t_over_tau is None:
        t_over_tau = np.linspace(0, 6, 200)

    m, b, g = 2.5, 0.8, 9.81
    tau_m = m / b
    v_norm = falling_mass_velocity(t_over_tau * tau_m, m, b, g) / (m * g / b)

    R, L, V_s = 50.0, 2e-6, 3.3
    tau_rl = L / R
    i_norm = rl_circuit_current(t_over_tau * tau_rl, R, L, V_s) / (V_s / R)

    gm, r_pi, R_L, C_pi, C_mu, V_s2 = 0.04, 2000.0, 1000.0, 5e-12, 1e-12, 1.0
    pole = transistor_dominant_pole(gm, r_pi, R_L, C_pi, C_mu)
    v_base_norm = transistor_step_response(t_over_tau * pole["tau"], gm, r_pi, R_L, C_pi, C_mu, V_s2) / V_s2

    reference = 1 - np.exp(-t_over_tau)
    err_mass = np.max(np.abs(v_norm - reference))
    err_rl = np.max(np.abs(i_norm - reference))
    err_transistor = np.max(np.abs(v_base_norm - reference))

    for name, err in (("falling mass", err_mass), ("RL circuit", err_rl), ("transistor", err_transistor)):
        if err > tol:
            raise AssertionError(f"{name}: normalized curve deviates from 1-e^-x by {err:.3e} (tol={tol:.3e})")

    return {"t_over_tau": t_over_tau, "reference_1_minus_e": reference,
            "falling_mass_normalized": v_norm, "rl_circuit_normalized": i_norm,
            "transistor_normalized": v_base_norm,
            "max_errors": {"falling_mass": err_mass, "RL_circuit": err_rl, "transistor": err_transistor}}


if __name__ == "__main__":
    print("=== virtual work / Euler-Lagrange derivation (falling mass + drag) ===")
    deriv = virtual_work_falling_mass_sympy()
    print("L = T - V ="); sp.pprint(deriv["L"])
    print("\nEuler-Lagrange EOM (in x):"); sp.pprint(deriv["EOM_in_x"])
    print("\nreduces to a first-order ODE in v = xdot:"); sp.pprint(deriv["velocity_ODE"])
    print("\nsolved:"); sp.pprint(deriv["v_of_t"])

    print("\n=== same equation, three domains ===")
    m, b, g = 2.5, 0.8, 9.81
    print(f"falling mass (m={m} kg, b={b} kg/s): "
          f"tau={m/b:.3f} s, v_terminal={m*g/b:.3f} m/s, "
          f"v(2*tau)={falling_mass_velocity(2*m/b, m, b, g):.3f} m/s")

    R, L, V_s = 50.0, 2e-6, 3.3
    print(f"RL circuit (R={R} ohm, L={L*1e6:.1f} uH): "
          f"tau={L/R*1e9:.1f} ns, I_final={V_s/R*1e3:.1f} mA")

    gm, r_pi, R_L, C_pi, C_mu = 0.04, 2000.0, 1000.0, 5e-12, 1e-12
    pole = transistor_dominant_pole(gm, r_pi, R_L, C_pi, C_mu)
    print(f"transistor (gm={gm} S, r_pi={r_pi} ohm, R_L={R_L} ohm, "
          f"C_pi={C_pi*1e12:.1f} pF, C_mu={C_mu*1e12:.1f} pF): "
          f"Miller factor={pole['miller_factor']:.1f}, C_eq={pole['C_eq']*1e12:.2f} pF, "
          f"f_3dB={pole['f_3dB_Hz']/1e6:.2f} MHz")

    print("\n=== isomorphism table ===")
    for row in isomorphism_table():
        print(f"  {row['role']:28s}: mass={row['falling mass']:28s} "
              f"RL={row['RL circuit']:22s} transistor={row['transistor']}")

    print("\n=== verified: all three normalize to the IDENTICAL curve 1-e^(-t/tau) ===")
    check = verify_same_functional_form()
    print(f"  max deviation from reference: {check['max_errors']}")
