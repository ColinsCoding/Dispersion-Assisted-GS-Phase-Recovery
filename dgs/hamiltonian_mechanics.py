"""Hamiltonian mechanics: the same physics as Lagrange, told in phase space.

Lagrangian mechanics lives in (q, q_dot) and gives one second-order equation per
coordinate. A Legendre transform trades the velocity q_dot for the MOMENTUM
        p = dL/dq_dot,        H(q, p) = p * q_dot - L,
and the single second-order equation splits into TWO first-order ones -- Hamilton's
equations, the phase-space flow:
        q_dot =  dH/dp,        p_dot = -dH/dq.
The second is Newton in disguise: for H = p^2/2m + V(q), p_dot = -dV/dq = force, and
q_dot = p/m. Nothing new physically -- but the geometry is better.

Why phase space is better: the flow is SYMPLECTIC. It conserves the Hamiltonian H
(energy, when H has no explicit time) AND phase-space volume (Liouville's theorem --
a blob of initial conditions can deform but keeps its area). A SYMPLECTIC integrator
(leapfrog) respects both: its energy error stays bounded and oscillatory forever,
while a generic solver like RK4 slowly bleeds or pumps energy. This module builds the
leapfrog flow, checks energy stays bounded where RK4 drifts, and verifies Liouville
by evolving a phase-space square and measuring its (preserved) area.

The harmonic oscillator carries it: H = p^2/2m + 1/2 k q^2, q(t) = A cos(omega t + phi)
with omega = sqrt(k/m). This is the classical sibling of dgs.quantum_oscillator's
E_n = (n+1/2) hbar*omega, and the same energy-conservation idea as
dgs.symmetry_physics and dgs.ptz_camera. NumPy + SymPy (for the Legendre transform);
py-3.13.
"""

import numpy as np
import sympy as sp


# ----------------------------------------------------------------------
# The Legendre transform L(q, q_dot) -> H(q, p), symbolically
# ----------------------------------------------------------------------

def legendre_transform_oscillator():
    """Carry out p = dL/dq_dot and H = p*q_dot - L for the harmonic oscillator
    L = 1/2 m q_dot^2 - 1/2 k q^2 in SymPy. Returns the momentum and the
    Hamiltonian, which must simplify to p = m q_dot and H = p^2/(2m) + 1/2 k q^2."""
    q, qd, m, k, p = sp.symbols("q q_dot m k p", positive=True)
    L = sp.Rational(1, 2) * m * qd ** 2 - sp.Rational(1, 2) * k * q ** 2
    p_expr = sp.diff(L, qd)                       # p = m q_dot
    qd_of_p = sp.solve(sp.Eq(p, p_expr), qd)[0]   # q_dot = p/m
    H = sp.simplify((p * qd - L).subs(qd, qd_of_p))
    return {"p": p_expr, "H": H}


# ----------------------------------------------------------------------
# The harmonic oscillator and Hamilton's equations
# ----------------------------------------------------------------------

def harmonic_hamiltonian(q, p, m=1.0, k=1.0):
    """H(q, p) = p^2/(2m) + 1/2 k q^2 -- the conserved total energy."""
    return p ** 2 / (2 * m) + 0.5 * k * q ** 2


def hamiltons_equations_ho(q, p, m=1.0, k=1.0):
    """(q_dot, p_dot) = (dH/dp, -dH/dq) = (p/m, -k q). The second is Newton's
    law F = -k q; together they are two first-order equations."""
    return p / m, -k * q


# ----------------------------------------------------------------------
# Symplectic (leapfrog) vs generic (RK4) integration
# ----------------------------------------------------------------------

def simulate_symplectic(dHdq, dHdp, q0, p0, dt, n_steps):
    """Leapfrog / Stormer-Verlet flow for a separable H = T(p) + V(q): a
    kick-drift-kick that is exactly symplectic. dHdq(q) = -force, dHdp(p) = q_dot.
    Returns (t, q, p) arrays -- energy stays bounded for all time."""
    if dt <= 0 or n_steps < 1:
        raise ValueError("dt > 0 and n_steps >= 1 required")
    q = np.empty(n_steps + 1); p = np.empty(n_steps + 1)
    q[0], p[0] = q0, p0
    for i in range(n_steps):
        p_half = p[i] - 0.5 * dt * dHdq(q[i])
        q[i + 1] = q[i] + dt * dHdp(p_half)
        p[i + 1] = p_half - 0.5 * dt * dHdq(q[i + 1])
    return np.arange(n_steps + 1) * dt, q, p


def simulate_rk4(dHdq, dHdp, q0, p0, dt, n_steps):
    """RK4 on the same Hamilton's equations -- accurate per step but NOT
    symplectic, so its energy slowly drifts. Returns (t, q, p)."""
    if dt <= 0 or n_steps < 1:
        raise ValueError("dt > 0 and n_steps >= 1 required")
    q = np.empty(n_steps + 1); p = np.empty(n_steps + 1)
    q[0], p[0] = q0, p0

    def deriv(qq, pp):
        return dHdp(pp), -dHdq(qq)                # (q_dot, p_dot)

    for i in range(n_steps):
        k1 = deriv(q[i], p[i])
        k2 = deriv(q[i] + 0.5 * dt * k1[0], p[i] + 0.5 * dt * k1[1])
        k3 = deriv(q[i] + 0.5 * dt * k2[0], p[i] + 0.5 * dt * k2[1])
        k4 = deriv(q[i] + dt * k3[0], p[i] + dt * k3[1])
        q[i + 1] = q[i] + dt / 6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        p[i + 1] = p[i] + dt / 6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    return np.arange(n_steps + 1) * dt, q, p


def energy_drift(q, p, m=1.0, k=1.0):
    """Peak-to-peak fractional variation of H along a trajectory. For a
    SYMPLECTIC flow this is BOUNDED -- the same value whether you run 10 periods
    or 10,000 -- because the energy oscillates within a fixed band and never
    accumulates. For a non-symplectic flow it GROWS with the run length."""
    E = harmonic_hamiltonian(q, p, m, k)
    return float(np.ptp(E) / abs(E[0]))


def energy_net_drift(q, p, m=1.0, k=1.0):
    """Net fractional energy change from start to end, |E_final - E_0|/E_0.
    Near zero for a symplectic flow (energy returns); accumulates for RK4."""
    E = harmonic_hamiltonian(q, p, m, k)
    return float(abs(E[-1] - E[0]) / abs(E[0]))


# ----------------------------------------------------------------------
# Liouville's theorem: phase-space area is preserved
# ----------------------------------------------------------------------

def _polygon_area(qs, ps):
    """Shoelace area of a polygon given its vertices in phase space."""
    return 0.5 * abs(np.dot(qs, np.roll(ps, -1)) - np.dot(ps, np.roll(qs, -1)))


def liouville_area_ratio(dHdq, dHdp, corners, dt, n_steps):
    """Evolve a small square of initial conditions (4 phase-space corners) under
    the symplectic flow and return final_area/initial_area -- 1.0 by Liouville's
    theorem, however the blob distorts. corners is a (4,2) array of (q, p)."""
    corners = np.asarray(corners, float)
    if corners.shape != (4, 2):
        raise ValueError("corners must be 4 (q, p) points")
    a0 = _polygon_area(corners[:, 0], corners[:, 1])
    final = []
    for q0, p0 in corners:
        _, q, p = simulate_symplectic(dHdq, dHdp, q0, p0, dt, n_steps)
        final.append((q[-1], p[-1]))
    final = np.array(final)
    a1 = _polygon_area(final[:, 0], final[:, 1])
    return a1 / a0


if __name__ == "__main__":
    lt = legendre_transform_oscillator()
    print("Legendre transform of L = 1/2 m q_dot^2 - 1/2 k q^2:")
    print("  p =", lt["p"], "   H =", lt["H"])

    m, k = 1.0, 1.0
    dHdq = lambda q: k * q          # -force
    dHdp = lambda p: p / m          # q_dot
    omega = np.sqrt(k / m)
    dt = 0.1

    # short term: leapfrog tracks the analytic cos(omega t)
    ts, qs, ps = simulate_symplectic(dHdq, dHdp, 1.0, 0.0, 0.01, int(5*2*np.pi/0.01))
    print(f"\nHamilton's eqs give SHM: 5-period max error vs cos(omega t) = "
          f"{np.max(np.abs(qs - np.cos(ts))):.2e}")

    # the symplectic signature: energy band is BOUNDED as the run gets longer,
    # while RK4's accumulates
    print("\nenergy peak-to-peak drift vs run length:")
    print("  periods   symplectic (bounded)   RK4 (grows)")
    for periods in (50, 500, 5000):
        n = int(periods * 2 * np.pi / dt)
        _, qs, ps = simulate_symplectic(dHdq, dHdp, 1.0, 0.0, dt, n)
        _, qr, pr = simulate_rk4(dHdq, dHdp, 1.0, 0.0, dt, n)
        print(f"  {periods:6d}   {energy_drift(qs, ps):.2e}             "
              f"{energy_drift(qr, pr):.2e}")
    print("  -> symplectic band is constant; RK4 grows ~10x per 10x time and "
          "eventually exceeds it")

    sq = [[1.0, 0.0], [1.1, 0.0], [1.1, 0.1], [1.0, 0.1]]
    ratio = liouville_area_ratio(dHdq, dHdp, sq, dt, int(500*2*np.pi/dt))
    print(f"\nLiouville: phase-space area ratio after 500 periods = {ratio:.6f} "
          f"(preserved exactly)")
