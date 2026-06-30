"""Special relativity + group/phase velocity + GVD connection to TS-DFT.

Key insight: group velocity v_g = d(omega)/dk is the same derivative that gives
GVD beta_2 = d^2k/d(omega)^2. In TS-DFT the pulse peak position IS the group
delay tau(omega) = d(phi)/d(omega) -- measuring it directly avoids branch cuts
entirely because you never need to unwrap phi(omega).
"""
import numpy as np
import sympy as sp

C_SI = 2.99792458e8   # m/s


def lorentz_factor(v, c=C_SI):
    """gamma and beta for speed v."""
    beta = v / c
    if abs(beta) >= 1.0:
        raise ValueError(f"|v| must be < c; got beta={beta:.4f}")
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    return {"gamma": gamma, "beta": beta, "v": v, "c": c}


def lorentz_transform(x, t, v, c=C_SI):
    """Boost frame S -> S' moving at v along x-axis.

    x' = gamma*(x - v*t)
    t' = gamma*(t - v*x/c^2)
    """
    lf = lorentz_factor(v, c)
    g = lf["gamma"]
    x_prime = g * (x - v * t)
    t_prime = g * (t - v * x / c**2)
    return {
        "x_prime": x_prime,
        "t_prime": t_prime,
        "gamma": g,
        "beta": lf["beta"],
    }


def time_dilation(tau0, v, c=C_SI):
    """Proper time tau0 in moving frame -> coordinate time t in lab.

    t = gamma * tau0  (moving clocks run slow)
    """
    lf = lorentz_factor(v, c)
    g = lf["gamma"]
    t_lab = g * tau0
    return {
        "t_lab": t_lab,
        "tau0_proper": tau0,
        "gamma": g,
        "time_ratio": g,          # t_lab / tau0
    }


def length_contraction(L0, v, c=C_SI):
    """Rest length L0 -> contracted length in lab frame.

    L = L0 / gamma  (moving rulers are shorter)
    """
    lf = lorentz_factor(v, c)
    g = lf["gamma"]
    L_lab = L0 / g
    return {
        "L_lab": L_lab,
        "L0_rest": L0,
        "gamma": g,
        "contraction_ratio": 1.0 / g,
    }


def four_vector_boost(event, v, c=C_SI):
    """Boost a spacetime 4-vector [ct, x, y, z] to frame moving at v along x.

    Returns boosted [ct', x', y', z'].
    """
    ct, x, y, z = event
    lf = lorentz_factor(v, c)
    g, b = lf["gamma"], lf["beta"]
    ct_p = g * (ct - b * x)
    x_p  = g * (x - b * ct)
    return {
        "event_prime": np.array([ct_p, x_p, y, z]),
        "event_orig":  np.array([ct, x, y, z]),
        "gamma": g,
        "beta": b,
        "invariant_orig":  ct**2 - x**2 - y**2 - z**2,
        "invariant_prime": ct_p**2 - x_p**2 - y**2 - z**2,
    }


def relativistic_energy(m_kg, v, c=C_SI):
    """Total energy, kinetic energy, rest energy.

    E_total = gamma * m * c^2
    E_rest  = m * c^2
    KE      = (gamma - 1) * m * c^2
    """
    lf = lorentz_factor(v, c)
    g = lf["gamma"]
    E_rest  = m_kg * c**2
    E_total = g * E_rest
    KE      = (g - 1) * E_rest
    p       = g * m_kg * v
    # Verify E^2 = (pc)^2 + (mc^2)^2
    E_from_p = np.sqrt((p * c)**2 + E_rest**2)
    return {
        "E_total_J": E_total,
        "E_rest_J":  E_rest,
        "KE_J":      KE,
        "p_kgms":    p,
        "gamma":     g,
        "E_check_J": E_from_p,
        "energy_momentum_error": abs(E_total - E_from_p),
    }


def energy_momentum_relation(m_kg, p_kgms, c=C_SI):
    """E from E^2 = (pc)^2 + (mc^2)^2 (works for photons: m=0)."""
    E = np.sqrt((p_kgms * c)**2 + (m_kg * c**2)**2)
    v = p_kgms * c**2 / E if E > 0 else c
    return {"E_J": E, "v_ms": v, "p_kgms": p_kgms}


def relativistic_doppler(f0, v, c=C_SI, approaching=True):
    """Relativistic Doppler shift.

    approaching: f_obs = f0 * sqrt((1+beta)/(1-beta))
    receding:    f_obs = f0 * sqrt((1-beta)/(1+beta))
    """
    lf = lorentz_factor(v, c)
    b = lf["beta"]
    if approaching:
        f_obs = f0 * np.sqrt((1 + b) / (1 - b))
    else:
        f_obs = f0 * np.sqrt((1 - b) / (1 + b))
    z = (f0 / f_obs) - 1 if approaching else (f_obs / f0) - 1
    return {
        "f_obs": f_obs,
        "f0": f0,
        "delta_f": f_obs - f0,
        "redshift_z": z,
        "approaching": approaching,
    }


def velocity_addition(v1, v2, c=C_SI):
    """Relativistic velocity addition u = (v1+v2)/(1 + v1*v2/c^2)."""
    u = (v1 + v2) / (1.0 + v1 * v2 / c**2)
    if abs(u) > c:
        u = np.sign(u) * c * (1 - 1e-15)   # numerical clamp
    lf = lorentz_factor(abs(u), c)
    return {"u_ms": u, "gamma": lf["gamma"], "beta": u / c}


# ── Phase and group velocity ───────────────────────────────────────────────────

def phase_velocity(omega, k):
    """v_p = omega / k (element-wise)."""
    k = np.asarray(k, float)
    omega = np.asarray(omega, float)
    v_p = omega / np.where(k == 0, np.inf, k)
    return {"v_p": v_p, "omega": omega, "k": k}


def group_velocity(omega_arr, k_arr):
    """v_g = d(omega)/dk via central differences.

    Connects to TS-DFT: the pulse peak time IS tau(omega) = d(phi)/d(omega),
    measuring group delay directly without phase unwrapping.
    """
    omega = np.asarray(omega_arr, float)
    k     = np.asarray(k_arr, float)
    if len(omega) < 2:
        raise ValueError("Need >= 2 points to compute group velocity")
    # d(omega)/dk -- note: dispersion relation gives k(omega) or omega(k)
    # If k_arr is k as function of omega: v_g = 1 / (dk/d(omega))
    dk_domega = np.gradient(k, omega)
    v_g = 1.0 / np.where(dk_domega == 0, np.inf, dk_domega)
    v_p = phase_velocity(omega, k)["v_p"]
    return {
        "v_g": v_g,
        "v_p": v_p,
        "v_g_mean": float(np.mean(v_g)),
        "v_p_mean": float(np.mean(v_p)),
        "omega": omega,
        "k": k,
        "dk_domega": dk_domega,
    }


def gvd_from_dispersion(omega_arr, k_arr):
    """GVD beta_2 = d^2k/d(omega)^2 from a dispersion relation k(omega).

    Also returns group delay tau(omega) = dk/d(omega) = 1/v_g.
    This is what TS-DFT reads from the time-stretched pulse:
    the peak of the waveform at each wavelength gives tau directly,
    so you never need to unwrap phi(omega).
    """
    omega = np.asarray(omega_arr, float)
    k     = np.asarray(k_arr, float)
    dk      = np.gradient(k, omega)              # 1/v_g = group delay per meter
    d2k     = np.gradient(dk, omega)             # beta_2 (s^2/m)
    v_g     = 1.0 / np.where(dk == 0, np.inf, dk)
    return {
        "beta2": d2k,
        "beta2_mean": float(np.mean(d2k)),
        "group_delay_per_m": dk,
        "v_g": v_g,
        "omega": omega,
        "k": k,
        "tsdft_note": (
            "TS-DFT avoids branch cuts: peak time of stretched pulse = "
            "tau(omega) = dk/domega directly. No unwrapping needed."
        ),
    }


def smf28_dispersion(omega_arr):
    """SMF-28 fiber dispersion relation k(omega) near 1550 nm.

    Uses 2nd-order Taylor: k(omega) ~ k0 + (omega-omega0)/v_g0 + beta2*(omega-omega0)^2/2
    beta2 = -22e-27 s^2/m at 1550 nm (anomalous dispersion).
    """
    omega0 = 2 * np.pi * 3e8 / 1550e-9   # rad/s at 1550 nm
    n0     = 1.4682
    k0     = n0 * omega0 / 3e8
    beta1  = n0 / 3e8                    # 1/v_g ~ n/c
    beta2  = -22e-27                     # s^2/m (anomalous)
    dw     = np.asarray(omega_arr, float) - omega0
    k      = k0 + beta1 * dw + 0.5 * beta2 * dw**2
    return {"k": k, "omega": np.asarray(omega_arr, float),
            "beta2_theory": beta2, "omega0": omega0, "n0": n0}


def group_delay_from_intensity(t_arr, I_arr, omega_arr, E_complex_arr):
    """Extract group delay tau(omega) from complex field E(t,omega) -- TS-DFT method.

    For each frequency slice, finds the time-domain peak (group delay).
    This IS the branch-cut-free phase derivative method:
      tau(omega) = d(phi)/d(omega) = peak time of |E(t)| at that omega slice

    In TS-DFT, the time axis IS the wavelength axis after stretching, so
    reading peak positions = reading group delays without any phase unwrapping.

    Parameters
    ----------
    t_arr          : time axis (s)
    I_arr          : 2D array [n_omega, n_t] -- intensity at each freq slice
    omega_arr      : frequency axis (rad/s)
    E_complex_arr  : complex field (same shape as I_arr, or None)
    """
    t = np.asarray(t_arr, float)
    I = np.asarray(I_arr, float)
    tau = np.array([t[np.argmax(I[i])] for i in range(I.shape[0])])
    # GVD = d(tau)/d(omega)
    beta2_L = np.gradient(tau, omega_arr)    # beta2 * L (s^2)
    return {
        "tau_omega": tau,
        "beta2_L": beta2_L,
        "beta2_L_mean": float(np.mean(beta2_L)),
        "method": "peak_position (branch-cut free)",
    }


# ── SymPy ────────────────────────────────────────────────────────────────────

def sr_sympy_5():
    """Five symbolic SR equations."""
    v, c, m, p, gamma_s, tau, L0 = sp.symbols(
        "v c m p gamma tau L_0", positive=True)
    beta  = v / c
    gamma_expr = 1 / sp.sqrt(1 - beta**2)
    return {
        "Lorentz_factor":    sp.Eq(sp.Symbol("gamma"), gamma_expr),
        "Time_dilation":     sp.Eq(sp.Symbol("t"), gamma_s * tau),
        "Energy_momentum":   sp.Eq(sp.Symbol("E")**2,
                                   (p * c)**2 + (m * c**2)**2),
        "Group_velocity":    sp.Eq(sp.Symbol("v_g"),
                                   1 / sp.Symbol("dk_domega")),
        "GVD_beta2":         sp.Eq(sp.Symbol("beta_2"),
                                   sp.Symbol("d2k_domega2")),
    }


if __name__ == "__main__":
    print("=== Lorentz factor at 0.9c ===")
    lf = lorentz_factor(0.9 * C_SI)
    print(f"  gamma = {lf['gamma']:.4f}")

    print("\n=== Time dilation: 1 s proper at v=0.9c ===")
    td = time_dilation(1.0, 0.9 * C_SI)
    print(f"  t_lab = {td['t_lab']:.4f} s")

    print("\n=== Relativistic energy: proton at 0.5c ===")
    mp = 1.6726e-27
    re = relativistic_energy(mp, 0.5 * C_SI)
    print(f"  E_total = {re['E_total_J']:.4e} J  KE = {re['KE_J']:.4e} J")

    print("\n=== SMF-28 GVD from dispersion ===")
    om0   = 2 * np.pi * 3e8 / 1550e-9
    omega = np.linspace(om0 - 2*np.pi*5e12, om0 + 2*np.pi*5e12, 2048)
    smf   = smf28_dispersion(omega)
    gvd   = gvd_from_dispersion(omega, smf["k"])
    print(f"  beta2_recovered = {gvd['beta2_mean']:.3e} s^2/m  "
          f"(theory: {smf['beta2_theory']:.3e})")
    print(f"  {gvd['tsdft_note']}")

    print("\n=== SymPy 5 ===")
    for k, eq in sr_sympy_5().items():
        print(f"  {k}: {eq}")
