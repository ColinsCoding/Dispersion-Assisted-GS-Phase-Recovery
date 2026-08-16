"""The free rigid body's Euler equations (dgs.gyroscopes), derived from a
Lagrangian instead of assumed -- and Noether's theorem applied directly to
find out WHICH conserved quantities dgs.gyroscopes.rotational_energy_and_momentum
actually comes from.

The body-frame angular velocity components (omega1, omega2, omega3) are
NOT themselves time-derivatives of any coordinate -- they're related to the
Euler angles (phi, theta, psi) and their rates by the standard kinematic
relations. Substituting those into T = (1/2)(I1*w1^2+I2*w2^2+I3*w3^2) gives
a genuine Lagrangian L(phi,theta,psi,phi',theta',psi') that
dgs.lagrangian.euler_lagrange can be handed directly -- no separate
"rigid body" machinery needed, just the SAME Euler-Lagrange recipe reused.

The point of doing this instead of just trusting dgs.gyroscopes' Euler
equations: a coordinate the Lagrangian doesn't depend on (only its rate
does) is CYCLIC, and Noether's theorem says its conjugate momentum is
exactly conserved -- for a specific, checkable reason, not a numerical
coincidence:

  * phi (precession about the space-fixed axis) is cyclic for ANY I1,I2,I3
    -- reflecting the free body's angular momentum being fixed in space,
    true regardless of the body's own shape.
  * psi (spin about the body's own figure axis) is ONLY cyclic when
    I1==I2 (a symmetric top) -- an ADDITIONAL conserved quantity that
    exists only because of the body's own material symmetry, and
    disappears the moment I1, I2, I3 are all distinct (the tennis-racket
    theorem's asymmetric top).
"""

import sympy as sp

from .lagrangian import euler_lagrange


def euler_angle_kinematics(t=None):
    """Body-frame angular velocity components in terms of the 3-1-3 Euler
    angles (phi, theta, psi) and their rates -- the standard kinematic
    relations (Goldstein's convention)."""
    t = t or sp.Symbol("t")
    phi = sp.Function("phi")(t)
    theta = sp.Function("theta")(t)
    psi = sp.Function("psi")(t)

    phid, thetad, psid = phi.diff(t), theta.diff(t), psi.diff(t)
    w1 = phid * sp.sin(theta) * sp.sin(psi) + thetad * sp.cos(psi)
    w2 = phid * sp.sin(theta) * sp.cos(psi) - thetad * sp.sin(psi)
    w3 = phid * sp.cos(theta) + psid
    return {"t": t, "phi": phi, "theta": theta, "psi": psi, "omega1": w1, "omega2": w2, "omega3": w3}


def free_rigid_body_lagrangian(I1, I2, I3, t=None):
    """L = T = (1/2)(I1*w1^2 + I2*w2^2 + I3*w3^2) in Euler angles (V=0,
    no external torque -- the same free-body setup as
    dgs.gyroscopes.euler_rigid_body_rhs, now as a genuine Lagrangian)."""
    kin = euler_angle_kinematics(t)
    L = sp.Rational(1, 2) * (I1 * kin["omega1"] ** 2 + I2 * kin["omega2"] ** 2 + I3 * kin["omega3"] ** 2)
    kin["L"] = sp.simplify(L)
    return kin


def noether_cyclic_check(L, q, t):
    """Is coordinate q cyclic in L (i.e. dL/dq == 0)? If so, Noether's
    theorem guarantees its conjugate momentum p_q = dL/d(q') is conserved
    -- confirmed here by showing the Euler-Lagrange equation for q
    (reusing dgs.lagrangian.euler_lagrange directly, not a separate
    derivation) reduces to exactly d(p_q)/dt = 0."""
    is_cyclic = sp.simplify(sp.diff(L, q)) == 0
    p_q = sp.simplify(sp.diff(L, q.diff(t)))
    el_equation = euler_lagrange(L, q, t)   # d/dt(dL/dq') - dL/dq
    # If q is cyclic, dL/dq=0, so the EL equation IS d(p_q)/dt exactly
    matches_dp_dt = is_cyclic and sp.simplify(el_equation - sp.diff(p_q, t)) == 0
    return {
        "cyclic": is_cyclic,
        "conserved_momentum": p_q,
        "euler_lagrange_equation": el_equation,
        "noether_conservation_confirmed": bool(matches_dp_dt),
    }


if __name__ == "__main__":
    print("=== Asymmetric top (I1, I2, I3 all distinct symbols) ===\n")
    I1, I2, I3 = sp.symbols("I1 I2 I3", positive=True)
    kin = free_rigid_body_lagrangian(I1, I2, I3)
    t, phi, psi = kin["t"], kin["phi"], kin["psi"]

    phi_check = noether_cyclic_check(kin["L"], phi, t)
    print(f"phi cyclic (space-fixed symmetry, should ALWAYS hold): {phi_check['cyclic']}")
    print(f"  Noether conservation confirmed via Euler-Lagrange: {phi_check['noether_conservation_confirmed']}")
    print(f"  conserved p_phi = {phi_check['conserved_momentum']}")

    psi_check = noether_cyclic_check(kin["L"], psi, t)
    print(f"\npsi cyclic for a GENERIC asymmetric top (I1 != I2, should be False): {psi_check['cyclic']}")

    print("\n=== Symmetric top (I1 == I2) ===\n")
    I_perp, I3_sym = sp.symbols("I I3", positive=True)
    kin_sym = free_rigid_body_lagrangian(I_perp, I_perp, I3_sym)
    psi_check_sym = noether_cyclic_check(kin_sym["L"], kin_sym["psi"], kin_sym["t"])
    print(f"psi cyclic once I1=I2=I (material symmetry restored): {psi_check_sym['cyclic']}")
    print(f"  Noether conservation confirmed via Euler-Lagrange: {psi_check_sym['noether_conservation_confirmed']}")
    print(f"  conserved p_psi = {psi_check_sym['conserved_momentum']}  (= I3*omega3, the body-axis spin)")
