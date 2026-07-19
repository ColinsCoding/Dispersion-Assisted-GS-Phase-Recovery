"""Differential equations in biochemistry: enzyme kinetics, derived rather
than assumed.

The Michaelis-Menten rate law v = Vmax*S/(Km+S) is usually just handed to
students. Here it's DERIVED from the underlying elementary mass-action
mechanism

    E + S <=(k1,k-1)=> ES  --k2--> E + P

by integrating the full 3-species nonlinear ODE system with RK4 and showing
it collapses onto the MM rate law under the quasi-steady-state approximation
(QSSA: d[ES]/dt ~ 0) -- the same "verify the closed form against the
ODE it's supposed to approximate" move used for GS/path-integral checks
elsewhere in this repo, just applied to enzyme kinetics.

Also: Hill cooperativity (a nonlinear algebraic rate law), competitive
inhibition, and a Lineweaver-Burk linear-regression fit -- the classic
biochemistry computational toolkit, in one module.
"""

import numpy as np


# -- RK4: the workhorse integrator for nonlinear biochemical ODEs ---------------

def rk4_step(f, t, y, h):
    """One classical 4th-order Runge-Kutta step for dy/dt = f(t, y)."""
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_rk4(f, y0, t):
    """Integrate dy/dt = f(t, y) over the time grid `t` via fixed-step RK4.
    y0 may be scalar or a vector (any shape numpy broadcasts cleanly)."""
    t = np.asarray(t, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    out = np.zeros((len(t),) + y0.shape)
    out[0] = y0
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        out[i + 1] = rk4_step(f, t[i], out[i], h)
    return out


# -- The elementary mass-action mechanism: E + S <-> ES -> E + P ----------------

def mass_action_rhs(state, k1, k_minus1, k2):
    """dE/dt, dS/dt, dES/dt, dP/dt for E+S<->ES->E+P (elementary mass action).
    state = [E, S, ES, P]."""
    E, S, ES, P = state
    dES_dt = k1 * E * S - (k_minus1 + k2) * ES
    dE_dt = -k1 * E * S + (k_minus1 + k2) * ES
    dS_dt = -k1 * E * S + k_minus1 * ES
    dP_dt = k2 * ES
    return np.array([dE_dt, dS_dt, dES_dt, dP_dt])


def simulate_mass_action(E0, S0, k1, k_minus1, k2, t):
    """Integrate the full 4-species elementary mechanism via RK4. Returns the
    (T, 4) trajectory [E, S, ES, P] -- the "ground truth" the MM rate law is
    supposed to approximate."""
    state0 = np.array([E0, S0, 0.0, 0.0])
    f = lambda time, state: mass_action_rhs(state, k1, k_minus1, k2)
    return integrate_rk4(f, state0, t)


# -- The Michaelis-Menten rate law, and the QSSA reduction that produces it ----

def michaelis_menten_rate(S, Vmax, Km):
    """v = Vmax * S / (Km + S) -- the textbook saturating rate law."""
    return Vmax * S / (Km + S)


def mm_constants_from_elementary(E0, k1, k_minus1, k2):
    """Map elementary rate constants to the MM constants Km = (k-1+k2)/k1 and
    Vmax = k2*E0 -- the algebra QSSA (d[ES]/dt ~ 0) produces."""
    Km = (k_minus1 + k2) / k1
    Vmax = k2 * E0
    return Vmax, Km


def integrate_mm_ode(S0, Vmax, Km, t):
    """Integrate the reduced 1-D MM ODE dS/dt = -Vmax*S/(Km+S) via RK4 --
    the cheap approximation to simulate_mass_action's full 4-species system."""
    f = lambda time, S: -michaelis_menten_rate(np.clip(S, 0, None), Vmax, Km)
    return integrate_rk4(f, np.array([S0]), t)[:, 0]


def qssa_validity(E0, S0, k1, k_minus1, k2, t):
    """Compare the full mass-action substrate trajectory S(t) against the
    QSSA-reduced MM ODE's S(t) using the SAME derived Vmax, Km. Returns both
    trajectories and their RMS relative difference -- QSSA is only valid when
    E0 << S0 (enzyme is catalytic, not stoichiometric); this lets you see
    exactly when the approximation breaks down instead of assuming it holds."""
    full = simulate_mass_action(E0, S0, k1, k_minus1, k2, t)
    S_full = full[:, 1]

    Vmax, Km = mm_constants_from_elementary(E0, k1, k_minus1, k2)
    S_mm = integrate_mm_ode(S0, Vmax, Km, t)

    rel_err = np.sqrt(np.mean((S_full - S_mm) ** 2)) / max(S0, 1e-12)
    return {"S_full": S_full, "S_mm": S_mm, "Vmax": Vmax, "Km": Km, "rms_relative_error": rel_err}


# -- Cooperative binding (Hill equation) and competitive inhibition -------------

def hill_equation(S, Vmax, K, n):
    """v = Vmax * S^n / (K^n + S^n) -- cooperative binding (n>1: positive
    cooperativity, e.g. hemoglobin-O2; n=1 reduces exactly to Michaelis-Menten)."""
    return Vmax * S ** n / (K ** n + S ** n)


def competitive_inhibition_rate(S, I, Vmax, Km, Ki):
    """v = Vmax*S / (Km*(1 + I/Ki) + S) -- a competitive inhibitor raises the
    *apparent* Km (Vmax unchanged), the classic enzyme-inhibition signature."""
    Km_apparent = Km * (1.0 + I / Ki)
    return Vmax * S / (Km_apparent + S)


# -- Lineweaver-Burk: recovering Vmax, Km from rate data by linear regression ---

def lineweaver_burk_fit(S, v):
    """Linearize v = Vmax*S/(Km+S) as 1/v = (Km/Vmax)*(1/S) + 1/Vmax, then fit
    by ordinary least squares (linear algebra: solve the 2-parameter normal
    equations) -- the classic graphical method for extracting Vmax, Km from
    a Michaelis-Menten dataset, done here as an explicit linear-algebra solve
    rather than read off a hand-drawn plot."""
    S = np.asarray(S, dtype=float)
    v = np.asarray(v, dtype=float)
    x = 1.0 / S
    y = 1.0 / v
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    Vmax = 1.0 / intercept
    Km = slope * Vmax
    return {"Vmax": Vmax, "Km": Km, "slope": slope, "intercept": intercept}
