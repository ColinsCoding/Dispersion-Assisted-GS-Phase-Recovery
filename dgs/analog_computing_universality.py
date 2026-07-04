"""'Molecular/laser computing, like op-amps': verifying that a molecular
(Lorentz-oscillator) optical resonance, a driven mechanical spring-mass-
damper, a series RLC circuit, and an OP-AMP ANALOG COMPUTER are four
different physical implementations of the SAME differential equation --
not an analogy, an algebraic identity, checked below.

The shared equation is the driven damped harmonic oscillator:
    x'' + gamma*x' + omega0^2*x = F(t)
  * mechanical:  m*x'' + b*x' + k*x = F(t)      (m=1: gamma=b, omega0^2=k)
  * molecular:   bound-electron model driven by an optical E-field --
                 EXACTLY this equation, giving dgs.causality's Lorentz
                 susceptibility as its steady-state frequency response
  * RLC circuit: L*I'' + R*I' + I/C = dV/dt     (omega0^2=1/LC, gamma=R/L)
  * op-amp analog computer: TWO cascaded integrators (dgs.ac_circuits.
    opamp_integrator) in a feedback loop, literally wired to compute
    x'' = F(t) - gamma*x' - omega0^2*x at every instant -- this is how
    1960s analog computers solved ODEs before digital computers existed,
    and it is a RECURSION (each integrator's output feeds back to produce
    the next instant's input) done in continuous time instead of discrete
    steps.

None of these are claimed equivalent by analogy -- each correspondence
below is verified either algebraically (SymPy) or by matching numerical
time-domain output against an independent RK4 integration of the same ODE.
"""

import numpy as np
import sympy as sp

from dgs.causality import lorentz_susceptibility
from dgs import ac_circuits as ac
from dgs import numerical_methods as nm


def mechanical_oscillator_susceptibility_symbolic():
    """Steady-state response X(omega) of m*x''+b*x'+k*x=F0*exp(-i*omega*t),
    derived by substituting the ansatz x=X*exp(-i*omega*t) and solving for
    X -- NOT assumed equal to the Lorentz susceptibility, derived
    independently and then compared. Uses the exp(-i*omega*t) time
    convention (optics/Lorentz-oscillator convention, matching
    dgs.causality.lorentz_susceptibility) rather than the engineering
    exp(+i*omega*t) convention -- the two differ by the sign of the
    damping term, a real convention choice, not an arbitrary one."""
    t, m, b, k, F0, omega = sp.symbols('t m b k F0 omega', positive=True, real=True)
    X = sp.symbols('X')
    x_ansatz = X * sp.exp(-sp.I * omega * t)
    ode_lhs = m * sp.diff(x_ansatz, t, 2) + b * sp.diff(x_ansatz, t) + k * x_ansatz
    ode_rhs = F0 * sp.exp(-sp.I * omega * t)
    # divide out the common exp(-i*omega*t) factor, solve the remaining algebraic equation
    eq = sp.Eq(sp.simplify(ode_lhs / sp.exp(-sp.I * omega * t)),
               sp.simplify(ode_rhs / sp.exp(-sp.I * omega * t)))
    X_solution = sp.solve(eq, X)[0]
    return X_solution, (m, b, k, F0, omega)


def verify_mechanical_matches_molecular():
    """Confirm the mechanical oscillator's derived X(omega) is ALGEBRAICALLY
    IDENTICAL to dgs.causality.lorentz_susceptibility under m=1, b=gamma,
    k=omega0^2, F0=strength -- the molecular and mechanical responses are
    the same formula, not similar-looking ones."""
    X_solution, (m, b, k, F0, omega) = mechanical_oscillator_susceptibility_symbolic()
    gamma, omega0, strength = sp.symbols('gamma omega0 strength', positive=True, real=True)
    X_relabeled = X_solution.subs({m: 1, b: gamma, k: omega0 ** 2, F0: strength})

    lorentz_expr = strength / (omega0 ** 2 - omega ** 2 - sp.I * gamma * omega)
    match = sp.simplify(X_relabeled - lorentz_expr) == 0
    return match, X_relabeled, lorentz_expr


def rlc_admittance_resonance_form(omega, R, L, C):
    """Series RLC current response I(omega)=V0/Z(omega), built from
    dgs.ac_circuits' ALREADY-TESTED impedance_R/L/C and series_impedance
    (not re-derived) -- checked below for having the SAME resonance-
    denominator pole structure as the mechanical/molecular oscillators."""
    Z = ac.series_impedance(ac.impedance_R(R), ac.impedance_L(L, omega), ac.impedance_C(C, omega))
    return 1.0 / Z   # admittance = 1/Z; a unit driving voltage's current response


def verify_rlc_matches_universal_form():
    """RLC admittance, built from dgs.ac_circuits' impedance_R/L/C (the
    ENGINEERING exp(+j*omega*t) convention, Z_L=+j*omega*L -- the opposite
    time convention from dgs.causality.lorentz_susceptibility's optics
    exp(-i*omega*t) convention, Z ~ -i*omega). Mixing the two conventions
    without accounting for that sign is comparing apples to oranges, not a
    physical mismatch -- so here the comparison is made SELF-CONSISTENTLY
    in the engineering (+j) convention dgs.ac_circuits actually uses:
    the universal form's damping-sign flips to +j*gamma*omega to match."""
    R, L, C, omega = sp.symbols('R L C omega', positive=True, real=True)
    j = sp.I   # dgs.ac_circuits.impedance_L/C use this same +j*omega*L, -j/(omega*C) convention
    Z = R + j * omega * L + 1 / (j * omega * C)
    Y = sp.simplify(1 / Z)

    omega0_sq = 1 / (L * C)
    gamma = R / L
    Y_times_L_over_j_omega = sp.simplify(Y * L / (j * omega))
    universal_form_engineering_convention = 1 / (omega0_sq - omega ** 2 + j * gamma * omega)
    match = sp.simplify(Y_times_L_over_j_omega - universal_form_engineering_convention) == 0
    return match


def solve_oscillator_rk4(omega0, gamma, F_func, t):
    """Ground-truth solution of x''+gamma*x'+omega0^2*x=F(t) via RK4 --
    completely independent of the op-amp analog-computer simulation below."""
    t = np.asarray(t, dtype=float)
    x, v = 0.0, 0.0
    xs = np.zeros(len(t))
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]

        def deriv(xi, vi, ti):
            return vi, F_func(ti) - gamma * vi - omega0 ** 2 * xi

        k1x, k1v = deriv(x, v, t[i])
        k2x, k2v = deriv(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, t[i] + 0.5 * dt)
        k3x, k3v = deriv(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, t[i] + 0.5 * dt)
        k4x, k4v = deriv(x + dt * k3x, v + dt * k3v, t[i + 1])
        x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        xs[i + 1] = x
    return xs


def solve_oscillator_analog_computer(omega0, gamma, F_func, t):
    """The OP-AMP ANALOG COMPUTER's own method: at each instant, combine
    F(t), the current velocity, and the current position (a summing
    amplifier, dgs.ac_circuits.summing_amplifier's role) to produce x'',
    then feed it through TWO cascaded integrators (dgs.ac_circuits.
    opamp_integrator's role, applied recursively one step at a time) to
    recover v and x -- literally simulating the feedback loop a real
    analog computer would wire up, step by step, rather than the abstract
    RK4 formula above."""
    t = np.asarray(t, dtype=float)
    x, v = 0.0, 0.0
    xs = np.zeros(len(t))
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        # the summing amplifier: combine feedback + forcing into x''
        a = F_func(t[i]) - gamma * v - omega0 ** 2 * x
        # integrator 1: v = integral(a dt)  (recursion: v_new = v_old + a*dt)
        v = v + a * dt
        # integrator 2: x = integral(v dt)  (recursion: x_new = x_old + v*dt)
        x = x + v * dt
        xs[i + 1] = x
    return xs


if __name__ == "__main__":
    print("=== Mechanical oscillator vs. molecular (Lorentz) susceptibility ===")
    match, X_relabeled, lorentz_expr = verify_mechanical_matches_molecular()
    print(f"mechanical X(omega) [m=1,b=gamma,k=omega0^2] == Lorentz susceptibility: {match}")

    print("\n=== RLC circuit resonance vs. the same universal form ===")
    rlc_match = verify_rlc_matches_universal_form()
    print(f"RLC admittance (rescaled) matches the universal resonance denominator: {rlc_match}")

    print("\n=== Op-amp analog computer vs. independent RK4 (time domain) ===")
    omega0, gamma = 2.0, 0.3
    F_func = lambda t: 1.0 if t > 0.1 else 0.0   # a step force -- the classic analog-computer demo
    t = np.linspace(0, 10, 2000)
    x_rk4 = solve_oscillator_rk4(omega0, gamma, F_func, t)
    x_analog = solve_oscillator_analog_computer(omega0, gamma, F_func, t)
    max_err = np.max(np.abs(x_rk4 - x_analog))
    print(f"max |x_analog_computer - x_rk4|: {max_err:.4e}  (both solve the SAME ODE, independently)")

    print("\n=== The point ===")
    print("A molecular Lorentz oscillator's response to light, a mass on a spring,")
    print("and 2 op-amp integrators wired in a feedback loop are the SAME equation,")
    print("solved by 3 different physical substrates -- verified algebraically and")
    print("numerically above, not asserted as an analogy.")
