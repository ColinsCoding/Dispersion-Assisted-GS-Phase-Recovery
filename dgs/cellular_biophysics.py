"""Cellular biophysics -- membrane potential, ion channels, action potentials.

WHY CELLULAR BIOPHYSICS FOR EE:
  The neuron IS a nonlinear electrical circuit:
    Membrane = capacitor (C_m ~ 1 uF/cm^2)
    Ion channels = voltage-gated conductances (g_Na, g_K, g_L)
    Ion pumps = current sources (maintain concentration gradients)
    Resting potential = DC operating point (-70 mV)
    Action potential = a SPIKE -- a self-resetting nonlinear oscillation

  The Hodgkin-Huxley model (1952, Nobel Prize 1963) is the FIRST digital
  biological computation model. It is a system of 4 ODEs -- the same
  Euler/RK4 integration used in pid_controller.py.

  PHOTONICS CONNECTION:
    Optogenetics uses light to control ion channels (ChR2, NpHR, Arch).
    See biophotonics.py for the optical delivery side.
    Here we model the ELECTRICAL response to the photocurrent.

THE NERNST EQUATION (equilibrium potential for one ion species):
  E_ion = (RT/zF) * ln([ion]_out / [ion]_in)
  where:
    R = 8.314 J/(mol*K)   gas constant
    T = 310 K             body temperature
    z = ion valence       (+1 for Na+, K+; +2 for Ca2+; -1 for Cl-)
    F = 96485 C/mol       Faraday constant

  For K+ at 37 C with [K]_in=140 mM, [K]_out=5 mM:
    E_K = (26 mV) * ln(5/140) = -89 mV  (resting K+ is near E_K)
  For Na+ with [Na]_in=15 mM, [Na]_out=145 mM:
    E_Na = (26 mV) * ln(145/15) = +61 mV (Na+ drives depolarization)

THE GOLDMAN-HODGKIN-KATZ EQUATION (resting potential with multiple ions):
  V_rest = (RT/F) * ln[(P_K*[K]_out + P_Na*[Na]_out + P_Cl*[Cl]_in) /
                        (P_K*[K]_in  + P_Na*[Na]_in  + P_Cl*[Cl]_out)]
  Permeabilities: P_K : P_Na : P_Cl ~ 1 : 0.04 : 0.45 (at rest)

HODGKIN-HUXLEY MODEL (the ODE system):
  C_m * dV/dt = I_ext - g_Na*m^3*h*(V-E_Na) - g_K*n^4*(V-E_K) - g_L*(V-E_L)
  dm/dt = alpha_m(V)*(1-m) - beta_m(V)*m
  dh/dt = alpha_h(V)*(1-h) - beta_h(V)*h
  dn/dt = alpha_n(V)*(1-n) - beta_n(V)*n

  m, h, n are gating variables in [0,1]:
    m: Na+ activation (fast, opens channel)
    h: Na+ inactivation (slow, closes channel)
    n: K+ activation (slow)

  m^3*h controls Na+ channel open probability.
  n^4 controls K+ channel open probability.
  At rest: m~0.05, h~0.6, n~0.3
"""
import numpy as np
import sympy as sp


# ── physical/biological constants ──────────────────────────────────────

R_gas = 8.314       # J/(mol*K)
F_faraday = 96485.0 # C/mol
T_body_K = 310.0    # 37 C in Kelvin
V_T = R_gas * T_body_K / F_faraday   # thermal voltage ~26.7 mV

# canonical ion concentrations (mM) for mammalian neuron
ION_CONCENTRATIONS = {
    "K":  {"in": 140.0, "out": 5.0,   "z": +1},
    "Na": {"in": 15.0,  "out": 145.0, "z": +1},
    "Ca": {"in": 0.0001,"out": 2.0,   "z": +2},
    "Cl": {"in": 10.0,  "out": 110.0, "z": -1},
}

# Hodgkin-Huxley parameters (squid giant axon, Hodgkin & Huxley 1952)
HH_PARAMS = {
    "C_m":  1.0,    # uF/cm^2
    "g_Na": 120.0,  # mS/cm^2 maximal Na+ conductance
    "g_K":  36.0,   # mS/cm^2 maximal K+ conductance
    "g_L":  0.3,    # mS/cm^2 leak conductance
    "E_Na": 50.0,   # mV Na+ reversal potential
    "E_K":  -77.0,  # mV K+ reversal potential
    "E_L":  -54.4,  # mV leak reversal potential
    "V_rest": -65.0, # mV resting potential
}


# ── Nernst equation ────────────────────────────────────────────────────

def nernst_potential(ion_name=None, conc_in_mM=None, conc_out_mM=None,
                     z=None, T_K=T_body_K):
    """Equilibrium (Nernst) potential for a single ion species.

    E = (RT / zF) * ln([ion]_out / [ion]_in)

    If ion_name is given, uses canonical mammalian neuron concentrations.
    Otherwise uses conc_in_mM, conc_out_mM, and z.

    Returns potential in mV.
    """
    if ion_name is not None:
        if ion_name not in ION_CONCENTRATIONS:
            raise ValueError(f"Unknown ion '{ion_name}'. Options: {list(ION_CONCENTRATIONS.keys())}")
        conc_in_mM  = ION_CONCENTRATIONS[ion_name]["in"]
        conc_out_mM = ION_CONCENTRATIONS[ion_name]["out"]
        z           = ION_CONCENTRATIONS[ion_name]["z"]

    if conc_in_mM is None or conc_out_mM is None or z is None:
        raise ValueError("Provide ion_name OR (conc_in_mM, conc_out_mM, z)")
    if conc_in_mM <= 0 or conc_out_mM <= 0:
        raise ValueError("concentrations must be positive")
    if z == 0:
        raise ValueError("ion valence z must be non-zero")

    E_mV = (R_gas * T_K / (z * F_faraday)) * np.log(conc_out_mM / conc_in_mM) * 1000
    return {"E_mV": E_mV, "ion": ion_name, "z": z,
            "conc_in_mM": conc_in_mM, "conc_out_mM": conc_out_mM, "T_K": T_K}


# ── Goldman-Hodgkin-Katz equation ──────────────────────────────────────

def goldman_potential(permeabilities=None, T_K=T_body_K):
    """Resting membrane potential via the Goldman-Hodgkin-Katz (GHK) equation.

    V_m = (RT/F) * ln[ (P_K*[K]_o + P_Na*[Na]_o + P_Cl*[Cl]_i) /
                        (P_K*[K]_i + P_Na*[Na]_i + P_Cl*[Cl]_o) ]

    Default permeabilities: P_K=1, P_Na=0.04, P_Cl=0.45 (at rest)
    """
    if permeabilities is None:
        permeabilities = {"K": 1.0, "Na": 0.04, "Cl": 0.45}

    P_K  = permeabilities.get("K", 1.0)
    P_Na = permeabilities.get("Na", 0.04)
    P_Cl = permeabilities.get("Cl", 0.45)

    K  = ION_CONCENTRATIONS["K"]
    Na = ION_CONCENTRATIONS["Na"]
    Cl = ION_CONCENTRATIONS["Cl"]

    # numerator: outward-permeable cations + inward-permeable anion
    num = P_K * K["out"] + P_Na * Na["out"] + P_Cl * Cl["in"]
    den = P_K * K["in"]  + P_Na * Na["in"]  + P_Cl * Cl["out"]
    V_mV = (R_gas * T_K / F_faraday) * np.log(num / den) * 1000
    return {"V_rest_mV": V_mV, "P_K": P_K, "P_Na": P_Na, "P_Cl": P_Cl}


# ── Hodgkin-Huxley gating variable rate functions ─────────────────────

def _alpha_m(V):
    """Na+ activation: alpha_m(V)."""
    V_shifted = V + 40.0
    if abs(V_shifted) < 1e-6:
        return 1.0   # L'Hopital limit
    return 0.1 * V_shifted / (1 - np.exp(-V_shifted / 10.0))

def _beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def _alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def _beta_h(V):
    return 1.0 / (1 + np.exp(-(V + 35.0) / 10.0))

def _alpha_n(V):
    V_shifted = V + 55.0
    if abs(V_shifted) < 1e-6:
        return 0.1
    return 0.01 * V_shifted / (1 - np.exp(-V_shifted / 10.0))

def _beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


def hh_steady_state(V_mV):
    """Hodgkin-Huxley gating variables at steady state for voltage V (mV).

    x_inf(V) = alpha_x(V) / (alpha_x(V) + beta_x(V))
    tau_x(V) = 1 / (alpha_x(V) + beta_x(V))
    """
    am, bm = _alpha_m(V_mV), _beta_m(V_mV)
    ah, bh = _alpha_h(V_mV), _beta_h(V_mV)
    an, bn = _alpha_n(V_mV), _beta_n(V_mV)
    m_inf = am / (am + bm)
    h_inf = ah / (ah + bh)
    n_inf = an / (an + bn)
    return {
        "m_inf": m_inf, "h_inf": h_inf, "n_inf": n_inf,
        "tau_m_ms": 1000 / (am + bm), "tau_h_ms": 1000 / (ah + bh),
        "tau_n_ms": 1000 / (an + bn),
        "g_Na_frac": m_inf**3 * h_inf,
        "g_K_frac": n_inf**4,
    }


# ── Hodgkin-Huxley ODE integrator ─────────────────────────────────────

def hodgkin_huxley(I_ext_uA_cm2, t_end_ms=50.0, dt_ms=0.01,
                   V0_mV=-65.0):
    """Simulate the Hodgkin-Huxley neuron model using RK4 integration.

    Parameters
    ----------
    I_ext_uA_cm2 : float or callable(t) -> float
        External current density. Scalar for constant current; callable for
        time-varying (e.g., current pulse).
    t_end_ms : float -- simulation duration in ms
    dt_ms : float    -- time step (0.01 ms recommended)
    V0_mV : float    -- initial membrane potential (mV)

    Returns
    -------
    dict with arrays: t_ms, V_mV, m, h, n, I_Na, I_K, I_L, I_ext
    """
    if dt_ms <= 0 or t_end_ms <= 0:
        raise ValueError("dt and t_end must be positive")

    p = HH_PARAMS.copy()
    if V0_mV != -65.0:
        p["V_rest"] = V0_mV

    # initial conditions: steady state at V_rest
    ss = hh_steady_state(V0_mV)
    V, m, h, n = V0_mV, ss["m_inf"], ss["h_inf"], ss["n_inf"]

    t_arr = np.arange(0, t_end_ms, dt_ms)
    N = len(t_arr)
    V_arr = np.zeros(N); m_arr = np.zeros(N)
    h_arr = np.zeros(N); n_arr = np.zeros(N)
    INa_arr = np.zeros(N); IK_arr = np.zeros(N)
    IL_arr = np.zeros(N); Iext_arr = np.zeros(N)

    def I_ext_fn(t):
        if callable(I_ext_uA_cm2):
            return I_ext_uA_cm2(t)
        return float(I_ext_uA_cm2)

    def deriv(V_, m_, h_, n_, t_):
        I_ext_t = I_ext_fn(t_)
        I_Na = p["g_Na"] * m_**3 * h_ * (V_ - p["E_Na"])
        I_K  = p["g_K"]  * n_**4      * (V_ - p["E_K"])
        I_L  = p["g_L"]               * (V_ - p["E_L"])
        dV = (I_ext_t - I_Na - I_K - I_L) / p["C_m"]
        dm = _alpha_m(V_) * (1 - m_) - _beta_m(V_) * m_
        dh = _alpha_h(V_) * (1 - h_) - _beta_h(V_) * h_
        dn = _alpha_n(V_) * (1 - n_) - _beta_n(V_) * n_
        return dV, dm, dh, dn, I_Na, I_K, I_L, I_ext_t

    for i, t in enumerate(t_arr):
        dV1, dm1, dh1, dn1, INa, IK, IL, Iext = deriv(V, m, h, n, t)
        # RK4
        dV2, dm2, dh2, dn2, _, _, _, _ = deriv(V+dV1*dt_ms/2, m+dm1*dt_ms/2,
                                                 h+dh1*dt_ms/2, n+dn1*dt_ms/2,
                                                 t+dt_ms/2)
        dV3, dm3, dh3, dn3, _, _, _, _ = deriv(V+dV2*dt_ms/2, m+dm2*dt_ms/2,
                                                 h+dh2*dt_ms/2, n+dn2*dt_ms/2,
                                                 t+dt_ms/2)
        dV4, dm4, dh4, dn4, _, _, _, _ = deriv(V+dV3*dt_ms, m+dm3*dt_ms,
                                                 h+dh3*dt_ms, n+dn3*dt_ms,
                                                 t+dt_ms)
        V += dt_ms/6 * (dV1 + 2*dV2 + 2*dV3 + dV4)
        m += dt_ms/6 * (dm1 + 2*dm2 + 2*dm3 + dm4)
        h += dt_ms/6 * (dh1 + 2*dh2 + 2*dh3 + dh4)
        n += dt_ms/6 * (dn1 + 2*dn2 + 2*dn3 + dn4)
        m = np.clip(m, 0, 1); h = np.clip(h, 0, 1); n = np.clip(n, 0, 1)
        V_arr[i] = V; m_arr[i] = m; h_arr[i] = h; n_arr[i] = n
        INa_arr[i] = INa; IK_arr[i] = IK; IL_arr[i] = IL; Iext_arr[i] = Iext

    # detect action potentials (threshold crossings at +20 mV)
    spike_mask = np.where((V_arr[:-1] < 20) & (V_arr[1:] >= 20))[0]
    n_spikes = len(spike_mask)
    firing_rate_Hz = n_spikes / (t_end_ms / 1000) if t_end_ms > 0 else 0

    return {
        "t_ms": t_arr, "V_mV": V_arr, "m": m_arr, "h": h_arr, "n": n_arr,
        "I_Na": INa_arr, "I_K": IK_arr, "I_L": IL_arr, "I_ext": Iext_arr,
        "n_spikes": n_spikes, "firing_rate_Hz": firing_rate_Hz,
        "spike_times_ms": t_arr[spike_mask],
    }


# ── optogenetics interface (connects to biophotonics.py) ──────────────

def optogenetic_current(light_power_mW_mm2, wavelength_nm=470.0,
                        channel="ChR2", area_cm2=1e-4):
    """Photo-induced current from optogenetic activation.

    ChR2 (channelrhodopsin-2): cation channel activated at ~470 nm.
    Peak photocurrent: I_ChR2 ~ 1-10 pA per channel, up to ~100 pA/cell.

    Simple model: I_photo (uA/cm^2) = g_ChR2 * sigmoid(power) * (V - E_ChR2)
    where g_ChR2 peaks ~0.5 nS per cell, E_ChR2 ~ 0 mV (reversal for cations).

    For HH simulation: use I_ext_uA_cm2 = optogenetic_current(...)["I_uA_cm2"]
    """
    CHANNELS = {
        "ChR2":  {"lambda_peak_nm": 470, "I_sat_uA_cm2": 5.0, "E_rev_mV": 0.0},
        "NpHR":  {"lambda_peak_nm": 580, "I_sat_uA_cm2": -3.0, "E_rev_mV": -90.0},
        "Arch":  {"lambda_peak_nm": 566, "I_sat_uA_cm2": -4.0, "E_rev_mV": -80.0},
    }
    if channel not in CHANNELS:
        raise ValueError(f"Unknown channel '{channel}'. Options: {list(CHANNELS.keys())}")
    ch = CHANNELS[channel]

    # wavelength penalty (off-peak activation is attenuated)
    spectral_penalty = np.exp(-0.5 * ((wavelength_nm - ch["lambda_peak_nm"]) / 30)**2)

    # saturation sigmoid
    I_sat = ch["I_sat_uA_cm2"]
    P50 = 0.5   # mW/mm^2 half-saturation power
    activation = light_power_mW_mm2 / (light_power_mW_mm2 + P50)
    I_photo = I_sat * activation * spectral_penalty

    return {
        "channel": channel,
        "I_uA_cm2": I_photo,
        "activation": activation,
        "spectral_penalty": spectral_penalty,
        "wavelength_nm": wavelength_nm,
    }


# ── membrane capacitance and RC model ─────────────────────────────────

def membrane_rc(C_m_uF_cm2=1.0, g_total_mS_cm2=None, I_ext_uA_cm2=1.0,
                V_rest_mV=-65.0):
    """Passive (linear) membrane RC model: tau = C_m / g_total.

    V_inf = V_rest + I_ext / g_total
    V(t) = V_inf - (V_inf - V_rest) * exp(-t/tau)

    This is the SUBTHRESHOLD response (no action potential).
    The HH model reduces to this for small I_ext.
    """
    if g_total_mS_cm2 is None:
        g_total_mS_cm2 = HH_PARAMS["g_L"]  # leak only
    tau_ms = C_m_uF_cm2 / g_total_mS_cm2
    V_inf = V_rest_mV + I_ext_uA_cm2 / g_total_mS_cm2
    t = np.linspace(0, 5 * tau_ms, 500)
    V = V_inf - (V_inf - V_rest_mV) * np.exp(-t / tau_ms)
    return {
        "tau_ms": tau_ms, "V_inf_mV": V_inf,
        "t_ms": t, "V_mV": V,
        "C_m": C_m_uF_cm2, "g_total": g_total_mS_cm2,
    }


# ── SymPy formalism ───────────────────────────────────────────────────

def cellular_biophysics_sympy_5():
    """Five key cellular biophysics equations in SymPy."""
    R_s, T_s, z_s, F_s = sp.symbols('R T z F', positive=True)
    co, ci = sp.symbols('c_out c_in', positive=True)
    C_m, V_s, I_ext = sp.symbols('C_m V I_ext', real=True)
    g_Na, g_K, g_L = sp.symbols('g_Na g_K g_L', positive=True)
    E_Na, E_K, E_L = sp.symbols('E_Na E_K E_L', real=True)
    m_s, h_s, n_s = sp.symbols('m h n', positive=True)

    return {
        "Nernst_potential":
            sp.Eq(sp.Symbol('E_ion'),
                  (R_s * T_s / (z_s * F_s)) * sp.log(co / ci)),
        "HH_dVdt":
            sp.Eq(C_m * sp.Symbol('dV/dt'),
                  I_ext - g_Na*m_s**3*h_s*(V_s - E_Na)
                        - g_K*n_s**4*(V_s - E_K)
                        - g_L*(V_s - E_L)),
        "Gating_steady_state":
            sp.Eq(sp.Symbol('x_inf'),
                  sp.Symbol('alpha_x') / (sp.Symbol('alpha_x') + sp.Symbol('beta_x'))),
        "Membrane_time_constant":
            sp.Eq(sp.Symbol('tau_m'),
                  C_m / (g_Na + g_K + g_L)),
        "Na_channel_open_probability":
            sp.Eq(sp.Symbol('P_open_Na'),
                  m_s**3 * h_s),
    }


if __name__ == "__main__":
    print("=== Nernst potentials (37 C) ===")
    for ion in ["K", "Na", "Ca", "Cl"]:
        e = nernst_potential(ion)
        print(f"  E_{ion:2s} = {e['E_mV']:+.1f} mV")

    print("\n=== Goldman resting potential ===")
    g = goldman_potential()
    print(f"  V_rest = {g['V_rest_mV']:.1f} mV  "
          f"(canonical ~-70 mV; P_K={g['P_K']}, P_Na={g['P_Na']}, P_Cl={g['P_Cl']})")

    print("\n=== HH steady state at V=-65 mV (rest) ===")
    ss = hh_steady_state(-65.0)
    print(f"  m={ss['m_inf']:.3f}, h={ss['h_inf']:.3f}, n={ss['n_inf']:.3f}")
    print(f"  tau_m={ss['tau_m_ms']:.3f} ms, tau_h={ss['tau_h_ms']:.2f} ms, tau_n={ss['tau_n_ms']:.2f} ms")

    print("\n=== Hodgkin-Huxley: action potential (I_ext=10 uA/cm^2) ===")
    result = hodgkin_huxley(10.0, t_end_ms=50.0, dt_ms=0.01)
    peak_V = np.max(result["V_mV"])
    print(f"  Peak voltage: {peak_V:.1f} mV  (expected ~+40 mV)")
    print(f"  Number of spikes: {result['n_spikes']}")
    print(f"  Firing rate: {result['firing_rate_Hz']:.1f} Hz")
    if len(result["spike_times_ms"]) > 0:
        print(f"  First spike: {result['spike_times_ms'][0]:.1f} ms")

    print("\n=== Optogenetic ChR2 current ===")
    opto = optogenetic_current(1.0, 470)
    print(f"  I_ChR2 = {opto['I_uA_cm2']:.2f} uA/cm^2  (activation={opto['activation']:.2f})")
    opto_off = optogenetic_current(1.0, 530)   # off-peak
    print(f"  At 530 nm (off-peak): I = {opto_off['I_uA_cm2']:.2f} uA/cm^2")

    print("\n=== Passive RC membrane model ===")
    rc = membrane_rc(I_ext_uA_cm2=1.0)
    print(f"  tau = {rc['tau_ms']:.0f} ms,  V_inf = {rc['V_inf_mV']:.1f} mV")

    print("\n=== SymPy 5 ===")
    for k, eq in cellular_biophysics_sympy_5().items():
        print(f"  {k}: {eq}")
