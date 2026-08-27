"""A floating boat as a Lagrangian mechanics problem: heave and roll.

A boat bobbing at anchor doesn't need free-body diagrams either -- the same
recipe as dgs.lagrangian applies once the restoring force is written as a
potential energy. Two one-degree-of-freedom oscillators come out of it,
both standard naval-architecture results, both derived here rather than
just quoted:

HEAVE (vertical bobbing), coordinate x(t), positive up from the equilibrium
waterline. Displace the hull up by x and it loses submerged volume A_wp*x
(A_wp = waterplane area, the cross-section at the waterline) -- by
Archimedes' principle that's a *restoring* buoyancy deficit rho*g*A_wp*x,
linear in x for small x. Integrating force over displacement gives a
harmonic potential V = 1/2 rho g A_wp x^2, exactly like a spring with
k_eff = rho*g*A_wp. The kinetic energy carries the ADDED MASS m_added: a
hull accelerating vertically also has to accelerate some of the water
around it, so the effective inertia is (m + m_added), not just the hull's
dry mass -- a real correction (typically 0.1-1.0x the displaced mass for
heave), not a modeling nicety.

ROLL (rocking about the fore-aft axis), coordinate theta(t). The restoring
moment for a small heel angle theta is rho*g*nabla*GM*theta (nabla =
displaced volume, GM = metacentric height -- the classic stability margin:
GM > 0 means the righting moment pushes the boat back upright, GM < 0 means
it capsizes). V = 1/2 rho g nabla GM theta^2 is the corresponding potential;
T = 1/2 I_roll theta'^2 with I_roll the roll moment of inertia (about the
roll axis, through the center of gravity).

Both reduce to simple harmonic motion, and both natural frequencies are
verified two ways below: once from the Lagrangian via
dgs.lagrangian.euler_lagrange, once from the standard closed-form omega =
sqrt(k_eff/inertia) -- the two must agree exactly, since they're the same
physics taken through different but equivalent derivations.
"""

import numpy as np
import sympy as sp

from dgs.lagrangian import equation_of_motion, euler_lagrange

G_STANDARD = 9.80665  # m/s^2
RHO_SEAWATER = 1025.0  # kg/m^3
RHO_FRESHWATER = 1000.0  # kg/m^3


def _check_positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


# -- Heave: vertical bobbing --

def heave_lagrangian(x, t, m, m_added, rho, g, A_wp):
    """L = T - V for heave: T = 1/2 (m + m_added) x'^2,
    V = 1/2 rho g A_wp x^2 (linearized buoyancy restoring force)."""
    T = sp.Rational(1, 2) * (m + m_added) * x.diff(t) ** 2
    V = sp.Rational(1, 2) * rho * g * A_wp * x ** 2
    return T - V


def heave_natural_frequency(m, m_added, rho, g, A_wp):
    """omega_heave = sqrt(rho g A_wp / (m + m_added)), the k_eff/inertia
    form -- inertia includes added mass, k_eff is the buoyancy restoring
    stiffness rho*g*A_wp."""
    _check_positive(m, "m")
    if m_added < 0:
        raise ValueError(f"m_added must be non-negative, got {m_added}")
    _check_positive(rho, "rho")
    _check_positive(g, "g")
    _check_positive(A_wp, "A_wp")
    return np.sqrt(rho * g * A_wp / (m + m_added))


def verify_heave_eom(m_val=5000.0, m_added_val=800.0, rho_val=RHO_SEAWATER,
                      g_val=G_STANDARD, A_wp_val=12.0):
    """Derive the heave EOM from the Lagrangian via Euler-Lagrange, extract
    its implied omega^2 (coefficient of x divided by coefficient of x''),
    and check it matches heave_natural_frequency's closed form exactly --
    the symbolic route and the closed-form formula must describe the same
    physics, not just look similar."""
    t = sp.Symbol("t")
    m, m_added, rho, g, A_wp = sp.symbols("m m_added rho g A_wp", positive=True)
    x = sp.Function("x")(t)

    L = heave_lagrangian(x, t, m, m_added, rho, g, A_wp)
    eom = euler_lagrange(L, x, t)  # (m+m_added)*x'' + rho*g*A_wp*x  (= 0)

    xdd = x.diff(t, 2)
    coeff_xdd = eom.coeff(xdd)
    coeff_x = sp.simplify((eom - coeff_xdd * xdd).coeff(x))
    omega_sq_symbolic = sp.simplify(coeff_x / coeff_xdd)

    subs = {m: m_val, m_added: m_added_val, rho: rho_val, g: g_val, A_wp: A_wp_val}
    omega_from_symbolic = float(sp.sqrt(omega_sq_symbolic.subs(subs)))
    omega_from_closed_form = heave_natural_frequency(m_val, m_added_val, rho_val, g_val, A_wp_val)

    matches = bool(abs(omega_from_symbolic - omega_from_closed_form) < 1e-9)
    return {
        "eom": eom,
        "omega_symbolic_rad_s": omega_from_symbolic,
        "omega_closed_form_rad_s": omega_from_closed_form,
        "matches": matches,
    }


# -- Roll: rocking about the fore-aft axis --

def roll_lagrangian(theta, t, I_roll, rho, g, nabla, GM):
    """L = T - V for small-angle roll: T = 1/2 I_roll theta'^2,
    V = 1/2 rho g nabla GM theta^2 (linearized righting moment,
    valid for small theta where the righting arm ~ GM*sin(theta) ~ GM*theta)."""
    T = sp.Rational(1, 2) * I_roll * theta.diff(t) ** 2
    V = sp.Rational(1, 2) * rho * g * nabla * GM * theta ** 2
    return T - V


def roll_natural_frequency(I_roll, rho, g, nabla, GM):
    """omega_roll = sqrt(rho g nabla GM / I_roll). GM must be strictly
    positive: GM <= 0 means the vessel has no static roll stability (it
    doesn't oscillate back to upright -- it heels further or is neutrally
    stable), so there is no restoring potential to speak of."""
    _check_positive(I_roll, "I_roll")
    _check_positive(rho, "rho")
    _check_positive(g, "g")
    _check_positive(nabla, "nabla")
    if GM <= 0:
        raise ValueError(
            f"GM must be positive for a stable, oscillating roll mode, got {GM} "
            "(GM <= 0 means the vessel lacks static roll stability)"
        )
    return np.sqrt(rho * g * nabla * GM / I_roll)


def roll_period_from_radius_of_gyration(k_roll, GM, g=G_STANDARD):
    """The textbook naval-architecture shortcut T_roll = 2*pi*k/sqrt(g*GM),
    valid when I_roll = m*k^2 and m = rho*nabla (Archimedes: displaced mass
    equals the vessel's own mass), so the mass cancels out of
    2*pi/roll_natural_frequency entirely. k_roll is the roll radius of
    gyration (m), GM the metacentric height (m)."""
    _check_positive(k_roll, "k_roll")
    _check_positive(g, "g")
    if GM <= 0:
        raise ValueError(f"GM must be positive, got {GM}")
    return 2 * np.pi * k_roll / np.sqrt(g * GM)


def verify_roll_period_shortcut(k_roll_val=3.0, GM_val=1.2, g_val=G_STANDARD,
                                 rho_val=RHO_SEAWATER, nabla_val=400.0):
    """Cross-check: compute the roll period two ways -- (1) via
    roll_natural_frequency with I_roll = m*k^2, m = rho*nabla (so the mass
    is built explicitly, not assumed away), and (2) via the textbook
    shortcut roll_period_from_radius_of_gyration, which claims the mass
    cancels. They must agree, confirming the cancellation is real algebra,
    not a coincidence of the chosen numbers."""
    m = rho_val * nabla_val  # Archimedes: displaced mass = vessel mass
    I_roll_val = m * k_roll_val ** 2
    omega = roll_natural_frequency(I_roll_val, rho_val, g_val, nabla_val, GM_val)
    period_from_omega = 2 * np.pi / omega
    period_from_shortcut = roll_period_from_radius_of_gyration(k_roll_val, GM_val, g_val)
    matches = bool(abs(period_from_omega - period_from_shortcut) < 1e-9)
    return {
        "period_from_omega_s": period_from_omega,
        "period_from_shortcut_s": period_from_shortcut,
        "matches": matches,
    }


if __name__ == "__main__":
    print("=== Heave: vertical bobbing ===")
    heave_check = verify_heave_eom()
    print(f"Euler-Lagrange EOM: {heave_check['eom']} = 0")
    print(f"omega_heave (symbolic route)   = {heave_check['omega_symbolic_rad_s']:.5f} rad/s")
    print(f"omega_heave (closed-form)      = {heave_check['omega_closed_form_rad_s']:.5f} rad/s")
    print(f"match: {heave_check['matches']}")
    T_heave = 2 * np.pi / heave_check["omega_closed_form_rad_s"]
    print(f"heave period: {T_heave:.2f} s\n")

    print("=== Roll: rocking about the fore-aft axis ===")
    roll_check = verify_roll_period_shortcut()
    print(f"period from omega (I_roll = m k^2 built explicitly): {roll_check['period_from_omega_s']:.3f} s")
    print(f"period from 2*pi*k/sqrt(g*GM) shortcut:              {roll_check['period_from_shortcut_s']:.3f} s")
    print(f"match: {roll_check['matches']}")

    print("\n=== Sanity: fresh vs. seawater changes the restoring force, not the shape of the physics ===")
    A_wp = 12.0
    for rho, label in [(RHO_FRESHWATER, "freshwater"), (RHO_SEAWATER, "seawater")]:
        w = heave_natural_frequency(m=5000.0, m_added=800.0, rho=rho, g=G_STANDARD, A_wp=A_wp)
        print(f"{label:>10s} (rho={rho:.0f} kg/m^3): omega_heave = {w:.5f} rad/s, T = {2*np.pi/w:.2f} s")
