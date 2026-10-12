"""Impedance and reactance -- from Griffiths' field equations to circuits and
back to waves. SymPy for the formalism, NumPy/SciPy to check every claim.

The one idea: impedance is the ratio of an "effort" phasor to a "flow" phasor
once d/dt has been replaced by a number s.

    circuits  Z = V / I       s = +j w  (EE,  e^{+j w t})
    waves     Z = E / H       s = -i w  (Griffiths / physics,  e^{-i w t})

The two conventions are complex conjugates of each other. Physics writes
Z_L = -i w L, EE writes Z_L = +j w L; both describe the same real current.
REACTANCE X is defined here so that X_L = +wL and X_C = -1/(wC) in BOTH
conventions:  EE  Z = R + jX,   physics  Z = R - iX.

Sections
  1. element impedances derived by substituting a complex exponential
  2. series RLC: |Z|, phase, resonance, Q -- checked against a time-domain ODE
  3. reactive power = 2 w (<W_m> - <W_e>) -- reactance is stored energy
  4. wave impedance Z = sqrt(mu/eps) = mu v; Griffiths' normal-incidence
     reflection (1 - beta)/(1 + beta) is (Z2 - Z1)/(Z2 + Z1)
  5. conductors: complex wave impedance, skin depth, surface reactance
  6. impedance transformation along a line / through a thin film; the
     quarter-wave anti-reflection coating, checked by a transfer matrix
"""
import numpy as np
import sympy as sp

w = sp.Symbol("omega", positive=True)
t = sp.Symbol("t", real=True)
R, L, C = sp.symbols("R L C", positive=True)
I0 = sp.Symbol("I_0")

CONVENTIONS = ("ee", "physics")
MU0 = 4e-7 * np.pi
EPS0 = 8.8541878128e-12
Z0_VACUUM = np.sqrt(MU0 / EPS0)


def _s(convention):
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}")
    return sp.I * w if convention == "ee" else -sp.I * w


# ── 1. element impedances ────────────────────────────────────────────

def derive_element_impedances(convention="ee"):
    """Substitute I(t) = I0 e^{s t} into each element law, return V/I.

    resistor   V = R I                (Ohm: J = sigma E integrated over the wire)
    inductor   V = L dI/dt            (Faraday: emf = -dPhi/dt, Phi = L I)
    capacitor  I = C dV/dt            (Gauss: Q = C V, I = dQ/dt)
    """
    s = _s(convention)
    I_t = I0 * sp.exp(s * t)
    V_R = R * I_t
    V_L = L * sp.diff(I_t, t)
    V_C = sp.integrate(I_t, t) / C          # V = Q/C, Q = integral of I
    return {k: sp.simplify(v / I_t) for k, v in
            (("R", V_R), ("L", V_L), ("C", V_C))}


def reactance(Z, convention="ee"):
    """X such that Z = R + jX (EE) or Z = R - iX (physics)."""
    _s(convention)
    Z = sp.sympify(Z)
    return sp.simplify(sp.im(Z) if convention == "ee" else -sp.im(Z))


# ── 2. series RLC ────────────────────────────────────────────────────

def series_rlc(convention="ee"):
    """Symbolic series RLC: Z, |Z|, phase of Z (EE sense), w0, Q."""
    Ze = derive_element_impedances(convention)
    Z = Ze["R"] + Ze["L"] + Ze["C"]
    X = reactance(Z, convention)
    w0 = sp.solve(sp.Eq(X, 0), w)[0]
    return {
        "Z": sp.simplify(Z),
        "X": X,
        "abs_Z": sp.sqrt(R ** 2 + X ** 2),
        "phase": sp.atan2(X, R),             # voltage leads current by this
        "w0": w0,
        "Q": sp.simplify(w0 * L / R),
    }


def series_rlc_numeric(R_, L_, C_, omega):
    """Complex EE impedance R + j(wL - 1/(wC)) as a NumPy array."""
    if min(R_, L_, C_) <= 0:
        raise ValueError("R, L, C must be positive")
    om = np.asarray(omega, dtype=float)
    if np.any(om <= 0):
        raise ValueError("omega must be positive")
    return R_ + 1j * (om * L_ - 1 / (om * C_))


def _steady_state(R_, L_, C_, V0, omega, n_periods=200, fit_periods=10):
    """Integrate L q'' + R q' + q/C = V0 cos(wt) to steady state.

    Returns (t, q, i) over the last fit_periods periods.
    """
    from scipy.integrate import solve_ivp
    if V0 <= 0 or omega <= 0:
        raise ValueError("V0 and omega must be positive")
    tau = 2 * L_ / R_                                    # envelope decay time
    T = 2 * np.pi / omega
    n_periods = max(n_periods, int(np.ceil(12 * tau / T)) + fit_periods)
    t_end = n_periods * T
    rhs = lambda tt, s: [s[1], (V0 * np.cos(omega * tt) - R_ * s[1] - s[0] / C_) / L_]
    tt = np.linspace(t_end - fit_periods * T, t_end, 400 * fit_periods,
                     endpoint=False)               # whole periods, no repeat
    sol = solve_ivp(rhs, (0, t_end), [0.0, 0.0], t_eval=tt, rtol=1e-10,
                    atol=1e-14, method="DOP853")
    return sol.t, sol.y[0], sol.y[1]


def time_domain_check(R_, L_, C_, V0, omega):
    """Steady-state current amplitude and phase from the ODE vs. the phasor.

    Fits i(t) = a cos wt + b sin wt by least squares. Returns dict with
    ode_amp, ode_phase (current phase relative to the drive, radians) and the
    phasor predictions V0/|Z| and -arg(Z).
    """
    tt, _, i = _steady_state(R_, L_, C_, V0, omega)
    M = np.column_stack([np.cos(omega * tt), np.sin(omega * tt)])
    a, b = np.linalg.lstsq(M, i, rcond=None)[0]
    Z = series_rlc_numeric(R_, L_, C_, omega)
    return {
        "ode_amp": float(np.hypot(a, b)),
        "ode_phase": float(np.arctan2(-b, a)),   # i = A cos(wt + phi)
        "phasor_amp": float(V0 / abs(Z)),
        "phasor_phase": float(-np.angle(Z)),
    }


# ── 3. reactive power is stored energy ───────────────────────────────

def complex_power(V, I):
    """S = (1/2) V I*  for peak phasors: P = Re S (watts), Q = Im S (vars)."""
    return 0.5 * V * np.conj(I)


def reactive_power_energy_check(R_, L_, C_, V0, omega):
    """Check Q = 2 w (<W_m> - <W_e>) using the time-domain steady state.

    <W_m> = <L i^2 / 2>,  <W_e> = <q^2 / (2C)>  averaged over whole periods.
    Returns (Q_phasor, Q_from_energies, P_phasor, P_from_resistor).
    """
    tt, q, i = _steady_state(R_, L_, C_, V0, omega)
    Wm = np.mean(0.5 * L_ * i ** 2)
    We = np.mean(0.5 * q ** 2 / C_)
    Z = series_rlc_numeric(R_, L_, C_, omega)
    I = V0 / Z
    S = complex_power(V0, I)
    return (float(S.imag), float(2 * omega * (Wm - We)),
            float(S.real), float(np.mean(R_ * i ** 2)))


# ── 4. wave impedance and reflection ─────────────────────────────────

def wave_impedance_symbolic():
    """Z = E/H for a plane wave = sqrt(mu/eps) = mu v,  v = 1/sqrt(mu eps)."""
    mu_, eps_ = sp.symbols("mu epsilon", positive=True)
    Z = sp.sqrt(mu_ / eps_)
    v = 1 / sp.sqrt(mu_ * eps_)
    assert sp.simplify(Z - mu_ * v) == 0
    return Z, mu_ * v


def griffiths_reflection():
    """Griffiths Sec. 9.3.2: E_R/E_I = (1 - beta)/(1 + beta), beta = mu1 v1/(mu2 v2),
    E_T/E_I = 2/(1 + beta).  Rewrite with Z = mu v and check energy conservation.

    Returns dict with r, t in impedance form and the simplified R + T.
    """
    Z1, Z2 = sp.symbols("Z_1 Z_2", positive=True)
    beta = Z1 / Z2
    r = sp.simplify((1 - beta) / (1 + beta))
    tr = sp.simplify(2 / (1 + beta))
    # intensity ~ E^2 / (2 Z): T = (Z1/Z2) t^2
    Rr, Tt = r ** 2, Z1 / Z2 * tr ** 2
    return {"r": r, "t": tr, "R": sp.simplify(Rr), "T": sp.simplify(Tt),
            "R+T": sp.simplify(Rr + Tt),
            "matches_line_formula": sp.simplify(r - (Z2 - Z1) / (Z2 + Z1)) == 0}


# ── 5. conductors ────────────────────────────────────────────────────

def conductor_impedance(sigma, omega, eps_r=1.0, mu_r=1.0):
    """Complex wave impedance of a conducting medium, EE convention.

    Griffiths 9.4 uses eps~ = eps + i sigma/w with e^{-iwt}; in EE (e^{jwt})
    that is eps - j sigma/w. Z = sqrt(mu / (eps - j sigma/w)).
    """
    if sigma < 0 or omega <= 0:
        raise ValueError("sigma >= 0 and omega > 0 required")
    eps_c = eps_r * EPS0 - 1j * sigma / omega
    return np.sqrt(mu_r * MU0 / eps_c)


def skin_depth(sigma, omega, mu_r=1.0):
    """delta = sqrt(2 / (mu sigma w)), the good-conductor limit (Griffiths 9.4.1)."""
    omega = np.asarray(omega, dtype=float)
    if sigma <= 0 or np.any(omega <= 0):
        raise ValueError("sigma and omega must be positive")
    return np.sqrt(2 / (mu_r * MU0 * sigma * omega))


def surface_impedance_good_conductor(sigma, omega, mu_r=1.0):
    """Z_s = (1 + j) / (sigma delta): equal resistance and INDUCTIVE reactance."""
    return (1 + 1j) / (sigma * skin_depth(sigma, omega, mu_r))


# ── 6. impedance transformation, quarter-wave coating ────────────────

def input_impedance_symbolic():
    """Z_in of a section (impedance Z0, electrical length theta) ended in Z_L.

    Built from forward + backward waves V = V+ (e^{-j b z} + Gamma e^{j b z}),
    I = (V+/Z0)(e^{-j b z} - Gamma e^{j b z}),  Gamma = (Z_L - Z0)/(Z_L + Z0),
    evaluated a distance l from the load. Returns (Z_in, Z_in at theta = pi/2).
    """
    Z0, ZL, th = sp.symbols("Z_0 Z_L theta", positive=True)
    G = (ZL - Z0) / (ZL + Z0)
    V = sp.exp(sp.I * th) + G * sp.exp(-sp.I * th)
    I = (sp.exp(sp.I * th) - G * sp.exp(-sp.I * th)) / Z0
    Zin = sp.simplify((V / I).rewrite(sp.cos))
    closed = Z0 * (ZL + sp.I * Z0 * sp.tan(th)) / (Z0 + sp.I * ZL * sp.tan(th))
    for v in (0.3, 1.1, 2.0):                      # wave picture == closed form
        assert abs(complex((Zin - closed).subs({Z0: 50, ZL: 75, th: v}).evalf())) < 1e-12
    quarter = sp.limit(closed, th, sp.pi / 2, "-")
    return closed, sp.simplify(quarter)


def film_reflectance_impedance(lam, n_film, d_film, n_in=1.0, n_sub=1.5):
    """Normal-incidence reflectance of one film via impedance transformation.

    Optical impedance of a nonmagnetic medium: Z = Z0_vacuum / n.
    """
    lam = np.asarray(lam, dtype=float)
    Zi, Zf, Zs = Z0_VACUUM / n_in, Z0_VACUUM / n_film, Z0_VACUUM / n_sub
    th = 2 * np.pi * n_film * d_film / lam
    Zin = Zf * (Zs + 1j * Zf * np.tan(th)) / (Zf + 1j * Zs * np.tan(th))
    r = (Zin - Zi) / (Zin + Zi)
    return np.abs(r) ** 2


def film_reflectance_matrix(lam, n_film, d_film, n_in=1.0, n_sub=1.5):
    """Same reflectance from the characteristic (transfer) matrix, as a check.

    [B, C]^T = [[cos d, j sin d / n], [j n sin d, cos d]] [1, n_sub]^T,
    r = (n_in B - C) / (n_in B + C).   (EE sign convention.)
    """
    lam = np.asarray(lam, dtype=float)
    d = 2 * np.pi * n_film * d_film / lam
    B = np.cos(d) + 1j * np.sin(d) * n_sub / n_film
    Cc = 1j * n_film * np.sin(d) + np.cos(d) * n_sub
    r = (n_in * B - Cc) / (n_in * B + Cc)
    return np.abs(r) ** 2
