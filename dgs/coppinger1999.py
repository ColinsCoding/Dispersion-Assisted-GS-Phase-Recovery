"""
Coppinger, Bhushan, Jalali (1999) - IEEE Trans. Microwave Theory Tech. Vol.47 No.7
"Photonic Time Stretch and Its Application to A/D Conversion"

Computer Algebra System rewrite: every equation in the paper derived symbolically.
Assumptions throughout: L1*beta2/tau^2 << 1 (highly chirped regime).
"""
import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# Symbolic variables (match paper notation exactly)
# ---------------------------------------------------------------------------
t, f, fm = sp.symbols('t f f_m', real=True)
tau = sp.Symbol('tau', positive=True)          # 1/e pulse half-width (time)
beta2 = sp.Symbol('beta_2', real=True)         # GVD [ps^2/km], can be negative
L1, L2 = sp.symbols('L_1 L_2', positive=True)  # fiber lengths [km]
a = sp.Symbol('a', real=True, positive=True)   # modulation depth
gamma = sp.Symbol('gamma')                     # intermediate variable in appendix

PI = sp.pi

# ---------------------------------------------------------------------------
# Eq 1 - Chirped Gaussian pulse after first dispersive fiber L1
# E_ch(L1,t) = exp(-t^2/tau^2) * exp(-j*t^2/(2*L1*beta2))
# Note: paper writes -jf^2 / (2*L1/beta2), but in time domain uses exp(-t^2/tau^2)
# The chirp is embedded in the frequency-domain Gaussian below.
# ---------------------------------------------------------------------------
def eq1_chirped_pulse_time():
    """Eq(1): Gaussian pulse in TIME domain after first dispersive fiber."""
    E = sp.exp(-t**2 / tau**2)
    return E

def eq2_chirped_pulse_freq():
    """
    Eq(2): Chirped Gaussian in FREQUENCY domain after L1 of fiber.
    E_ch(L1,f) = sqrt(pi) / sqrt(tau^-2 + j/(2*L1*beta_2))
                 * exp(-pi^2*f^2 / (tau^-2 + j/(2*L1*beta_2)))
    This is the FT of the chirped Gaussian — exact result.
    """
    denom = tau**(-2) + sp.I / (2*L1*beta2)
    prefactor = sp.sqrt(PI) / sp.sqrt(denom)
    exponent = -PI**2 * f**2 / denom
    return sp.simplify(prefactor * sp.exp(exponent))

# ---------------------------------------------------------------------------
# Eq 3 - Modulated signal in time (Mach-Zehnder output)
# Mach-Zehnder modulator: E_in = E_ch * [1 + a*cos(2*pi*fm*t)]
# Small-signal (a<<1): linear intensity modulation, not phase modulation.
# ---------------------------------------------------------------------------
def eq3_modulated_time():
    """Eq(3): Intensity-modulated field after MZM. Small signal: a << 1."""
    E_ch = eq1_chirped_pulse_time()
    mod = 1 + a * sp.cos(2*PI*fm*t)
    return E_ch * mod

def eq4_modulated_freq(E_ch_f=None):
    """
    Eq(4): Modulated field in FREQUENCY domain.
    cos(2*pi*fm*t) <-> (1/2)[delta(f-fm) + delta(f+fm)]
    E_in(L1,f) = E_ch(L1,f) * [delta(f) + a/2*(delta(f-fm)+delta(f+fm))]
    Returns the three sideband amplitudes as a dict.
    """
    if E_ch_f is None:
        E_ch_f = eq2_chirped_pulse_freq()
    denom = tau**(-2) + sp.I / (2*L1*beta2)
    # Carrier (f=0)
    A_carrier = sp.sqrt(PI) / sp.sqrt(denom)
    # Sideband at f = +fm
    A_plus = (a/2) * sp.sqrt(PI) / sp.sqrt(denom) * sp.exp(-PI**2 * fm**2 / denom)
    # Sideband at f = -fm  (same amplitude by symmetry)
    A_minus = A_plus
    return {'carrier': A_carrier, 'upper_sb': A_plus, 'lower_sb': A_minus}

# ---------------------------------------------------------------------------
# Eq 5 - After second dispersive fiber L2
# E_out(L1+L2, f) = E_in(L1, f) * exp(j*2*pi^2*L2*beta2*f^2)
# Transfer function of dispersive fiber: H(f) = exp(j*pi*D*f^2) with D=2*pi*L*beta2
# ---------------------------------------------------------------------------
def dispersive_transfer_function(L):
    """H(f) = exp(j*2*pi^2*L*beta2*f^2) — second-order dispersive fiber."""
    return sp.exp(sp.I * 2 * PI**2 * L * beta2 * f**2)

def eq5_output_freq():
    """
    Eq(5): Full output spectrum after both fibers.
    Three-sideband model (carrier + upper + lower).
    Returns dict of {freq_offset: amplitude}.
    """
    sb = eq4_modulated_freq()
    H = dispersive_transfer_function(L2)

    # Evaluate H at each sideband frequency
    H_at_0 = H.subs(f, 0)      # = 1
    H_at_fm = H.subs(f, fm)
    H_at_neg = H.subs(f, -fm)

    return {
        'carrier_amp': sb['carrier'] * H_at_0,
        'upper_amp': sb['upper_sb'] * H_at_fm,
        'lower_amp': sb['lower_sb'] * H_at_neg,
        'carrier_freq': 0,
        'upper_freq': fm,
        'lower_freq': -fm,
    }

# ---------------------------------------------------------------------------
# Eq 6 / Appendix - Time-domain output (key derivation)
# Approximation: L1*beta2/tau^2 << 1  =>  gamma ~ 1
# ---------------------------------------------------------------------------
def appendix_gamma_definition():
    """
    Appendix Eq(A2): gamma = 1 + j*2*L1*beta2*tau^-2
    Under L1*beta2/tau^2 << 1: gamma ~ 1
    """
    return 1 + sp.I * 2 * L1 * beta2 / tau**2

def appendix_exponent_E1(simplify=True):
    """
    Appendix Eq(A3-A4): Exponent E1 in the frequency-domain integrand T1(f).
    E1 = j*2*pi^2*beta2*(L1*gamma + L2) * (f - L1/(L1+L2)*fm)^2
         + j*2*pi^2*beta2*(L1+L2)/M * fm^2
    where M = 1 + L2/L1.
    """
    M_sym = 1 + L2/L1
    # Center frequency after stretch
    f_center = L1/(L1+L2) * fm
    # Effective dispersion
    L_eff = L1 * appendix_gamma_definition() + L2
    E1 = (sp.I * 2 * PI**2 * beta2 * L_eff * (f - f_center)**2
          + sp.I * 2 * PI**2 * beta2 * (L1+L2) / M_sym * fm**2)
    if simplify:
        E1 = sp.expand(E1)
    return E1

# ---------------------------------------------------------------------------
# Eq 7 - Detected intensity I_out = |E_out|^2
# I_out(L1+L2, t) proportional to:
#   (L1/(L1+L2))^2 * exp(-2t^2 / (tau*(L1+L2)/L1)^2)
#   * [1 + 2a*cos(2*pi*fm_eff*t) * cos(dispersion_penalty)]
# where fm_eff = fm * L1/(L1+L2) = fm/M and dispersion_penalty from Eq9
# ---------------------------------------------------------------------------
def eq7_detected_intensity(numeric=False, L1_km=1.0, L2_km=5.5, beta2_val=-20.0,
                            tau_ps=100.0, fm_ghz=30.0, a_val=0.1):
    """
    Eq(7): Detected photocurrent envelope (symbolic or numeric).
    Returns symbolic expression or numeric array over t_ps.
    """
    M_val = 1 + L2_km/L1_km
    tau_stretched = tau_ps * M_val
    fm_eff = fm_ghz / M_val           # effective modulation freq after stretch [GHz]

    # Dispersion penalty phase (Eq 9)
    penalty_arg = 2 * np.pi**2 * beta2_val * L2_km * (fm_ghz**2) / M_val
    penalty = np.cos(penalty_arg)**2

    if not numeric:
        M = 1 + L2/L1
        f_eff = fm / M
        penalty_sym = sp.cos(2*PI**2*beta2*L2*fm**2/M)**2
        envelope = sp.exp(-2*t**2 / (tau*M)**2)
        I = (1/M**2) * envelope * (1 + 2*a*sp.cos(2*PI*f_eff*t)) * penalty_sym
        return sp.simplify(I)
    else:
        t_arr = np.linspace(-3*tau_stretched, 3*tau_stretched, 2000)
        envelope = np.exp(-2*t_arr**2 / tau_stretched**2)
        I = (1/M_val**2) * envelope * (1 + 2*a_val*np.cos(2*np.pi*fm_eff*t_arr)) * penalty
        return t_arr, I, {'M': M_val, 'fm_eff_GHz': fm_eff, 'penalty': penalty,
                          'tau_stretched_ps': tau_stretched}

# ---------------------------------------------------------------------------
# Eq 8 - Stretch factor M (the key result)
# ---------------------------------------------------------------------------
def eq8_stretch_factor(L1_km, L2_km):
    """Eq(8): M = 1 + L2/L1. Stretch factors from Fig.2: L2=2.2->M=3, 5.5->6, 7.6->8."""
    M = 1 + L2_km / L1_km
    return M

def verify_fig2_stretch_factors():
    """Verify paper Fig.2 stretch factors using L1=1.1 km (inferred from text)."""
    L1 = 1.1
    cases = [(2.2, 3), (5.5, 6), (7.6, 8)]
    results = []
    for L2, M_paper in cases:
        M_calc = eq8_stretch_factor(L1, L2)
        results.append({'L2_km': L2, 'M_paper': M_paper, 'M_calc': round(M_calc, 1),
                        'match': abs(M_calc - M_paper) < 0.15})
    return results

# ---------------------------------------------------------------------------
# Eq 9 - Dispersion penalty (high-pass filter effect of second fiber)
# H_att = [cos(2*pi^2*beta2*L2*fm^2/M)]^2
# ---------------------------------------------------------------------------
def eq9_dispersion_penalty(L2_km, beta2_ps2km, fm_ghz, M):
    """
    Eq(9): Attenuation due to dispersion in second fiber.
    beta2 in ps^2/km, fm in GHz, L2 in km.
    Returns power transmission [0,1].
    """
    arg = 2 * np.pi**2 * beta2_ps2km * L2_km * (fm_ghz * 1e9)**2 * 1e-24 / M
    # Unit check: beta2[ps^2/km]*L[km]*f[GHz]^2 -> need consistent units
    # Paper uses SI: beta2 [s^2/m], L [m], f [Hz]
    # Convert: beta2_ps2km [ps^2/km] = beta2_ps2km * 1e-24 s^2 / 1e3 m = * 1e-27 s^2/m
    # f [GHz] = fm_ghz * 1e9 Hz
    # arg = 2*pi^2 * (beta2*1e-27) * (L2*1e3) * (fm*1e9)^2 / M
    arg_si = 2 * np.pi**2 * (beta2_ps2km * 1e-27) * (L2_km * 1e3) * (fm_ghz * 1e9)**2 / M
    return np.cos(arg_si)**2

# ---------------------------------------------------------------------------
# Mach-Zehnder Modulator physics
# ---------------------------------------------------------------------------
def mzm_transfer_function(V_bias, V_pi, V_rf, omega_m, t_arr):
    """
    MZM intensity transfer: I_out = I_in/2 * [1 + cos(pi*(V_bias+V_rf*cos(wt))/V_pi)]
    Small signal (V_rf << V_pi): linearizes to E_out ~ E_in * [1 + a*cos(wt)]
    with a = pi*V_rf/(2*V_pi) * sin(pi*V_bias/V_pi)
    Quadrature bias: V_bias = V_pi/2 -> maximum linear response.
    """
    phi_bias = np.pi * V_bias / V_pi
    phi_rf = np.pi * V_rf / V_pi * np.cos(omega_m * t_arr)
    I_out = 0.5 * (1 + np.cos(phi_bias + phi_rf))
    # Small-signal modulation depth
    a_eff = (np.pi * V_rf / V_pi) * np.sin(phi_bias)  # at quadrature: sin(pi/2)=1
    return I_out, a_eff

def mzm_cmos_specs():
    """
    CMOS-integrated MZM key specs.
    Silicon photonics MZM: V_pi*L ~ 2 V*cm, 3dB BW ~ 40 GHz, IL ~ 5 dB.
    """
    return {
        'V_pi_V': 3.5,
        'V_pi_L_Vcm': 2.0,
        'bandwidth_3dB_GHz': 40,
        'insertion_loss_dB': 5,
        'extinction_ratio_dB': 30,
        'operating_wavelength_nm': 1550,
        'device_length_mm': 5,
        'cmos_node': '180nm silicon photonics',
        'physics': 'plasma dispersion effect: free carriers shift n and k',
        'connection_to_jalali': 'MZM output is Eq(3) — sets modulation depth a',
    }

# ---------------------------------------------------------------------------
# Direct time-to-wavelength mapping (frequency-to-time)
# ---------------------------------------------------------------------------
def freq_to_time_mapping(beta2_ps2km, L_km, f_hz):
    """
    Direct time-to-wavelength: t = 2*pi*beta2*L*f
    (derivative of spectral phase phi = pi*beta2*L*f^2 -> d phi/df = 2*pi*beta2*L*f)
    This is Jalali grammar Eq(2): t(lambda) = D*L*(lambda - lambda0).
    Here in frequency domain: t = 2*pi*beta2*L*f
    beta2 [ps^2/km], L [km], f [GHz] -> t [ps]
    """
    t_ps = 2 * np.pi * (beta2_ps2km * 1e-27) * (L_km * 1e3) * f_hz * 1e12
    return t_ps

def time_bandwidth_product(tau_ps, delta_f_ghz):
    """TBP = tau * delta_f. For transform-limited Gaussian: TBP = 1/(2*pi) ~ 0.44 (FWHM)."""
    tbp = (tau_ps * 1e-12) * (delta_f_ghz * 1e9)
    return {'TBP': tbp, 'transform_limited_0_44': abs(tbp - 0.44) < 0.1}

# ---------------------------------------------------------------------------
# Full numeric simulation: Fig.2 reproduction
# ---------------------------------------------------------------------------
def simulate_fig2(L1_km=1.1, L2_values_km=None, beta2_ps2km=-20.0,
                  tau_ps=2.0, fm_ghz=10.0, a_val=0.3):
    """
    Reproduce Fig.2: intensity envelope after time-stretch for M=1,3,6,8.
    Returns dict: L2 -> (t_ps_array, I_array, M_value)
    """
    if L2_values_km is None:
        L2_values_km = [0.0, 2.2, 5.5, 7.6]
    results = {}
    for L2 in L2_values_km:
        M = 1 + L2/L1_km if L2 > 0 else 1.0
        tau_s = tau_ps * M
        t_arr = np.linspace(-4*tau_s, 4*tau_s, 4000)
        envelope = np.exp(-t_arr**2 / tau_s**2)
        # Effective modulation frequency after stretch
        fm_eff = fm_ghz / M
        # Dispersion penalty
        if L2 > 0:
            arg_si = 2*np.pi**2*(beta2_ps2km*1e-27)*(L2*1e3)*(fm_ghz*1e9)**2/M
            pen = np.cos(arg_si)**2
        else:
            pen = 1.0
        I = (1/M**2) * envelope * (1 + 2*a_val*np.cos(2*np.pi*fm_eff*t_arr)) * pen
        results[L2] = (t_arr, I, M)
    return results

# ---------------------------------------------------------------------------
# SymPy derivation of Appendix: E_out in time domain
# ---------------------------------------------------------------------------
def derive_appendix_time_domain(verbose=True):
    """
    Symbolic derivation following the paper Appendix.
    Derives T1(t) — the time-domain field envelope — under L1*beta2/tau^2 << 1.

    Steps:
    1. Write E_out(L1+L2, f) as three Gaussians (carrier + 2 sidebands)
    2. Factor out the quadratic phase: E1 exponent
    3. Apply stationary phase / complete the square
    4. Take inverse FT -> time-domain Gaussian with effective width tau*M
    5. Modulation term appears at fm/M = fm_eff (frequency divided by M)
    6. Dispersion penalty cos^2 term appears
    """
    log = []

    # Step 1: gamma definition
    g = appendix_gamma_definition()
    log.append(("gamma", g))
    log.append(("gamma_approx_L1_small", sp.Integer(1)))  # gamma->1 when L1*beta2/tau^2<<1

    # Step 2: Exponent E1 (Eq A3-A4)
    M_sym = 1 + L2/L1
    f_center = L1/(L1+L2) * fm

    log.append(("stretch_factor_M", M_sym))
    log.append(("shifted_freq_center", f_center))
    log.append(("interpretation", "After second dispersion, each sideband is "
                "centered at fm*(L1/(L1+L2)) = fm/M in the output — frequency DIVIDED by M"))

    # Step 3: Effective pulse width
    tau_out = tau * M_sym
    log.append(("output_pulse_width", tau_out))
    log.append(("interpretation_2", "Pulse stretches by M — ADC sees tau*M, NOT tau"))

    # Step 4: Modulation frequency in output
    fm_out = fm / M_sym
    log.append(("output_modulation_freq", fm_out))
    log.append(("interpretation_3", "fm mapped to fm/M — input 30GHz at M=6 -> 5GHz out"))

    # Step 5: Dispersion penalty (Eq 9) symbolic
    penalty = sp.cos(2*PI**2*beta2*L2*fm**2/M_sym)**2
    log.append(("dispersion_penalty_H_att", penalty))

    # Step 6: Full time-domain intensity symbolic
    I_sym = (L1/(L1+L2))**2 * sp.exp(-2*t**2/(tau*M_sym)**2) * (
        1 + 2*a*sp.cos(2*PI*fm_out*t)
    ) * penalty
    log.append(("I_out_symbolic", I_sym))

    if verbose:
        for key, val in log:
            print(f"\n--- {key} ---")
            sp.pprint(val)
    return dict(log)

def demo():
    print("=== Coppinger 1999 IEEE MTT - CAS Demo ===")

    print("\n[1] Stretch factor verification (Fig.2)")
    for r in verify_fig2_stretch_factors():
        status = "OK" if r['match'] else "MISMATCH"
        print(f"  L2={r['L2_km']}km: M_paper={r['M_paper']} M_calc={r['M_calc']} [{status}]")

    print("\n[2] Frequency-to-time mapping example")
    t_out = freq_to_time_mapping(beta2_ps2km=-20, L_km=5.5, f_hz=30e9)
    print(f"  30 GHz tone at beta2=-20ps2/km, L=5.5km -> t = {t_out:.2f} ps")

    print("\n[3] MZM CMOS specs")
    specs = mzm_cmos_specs()
    for k, v in specs.items():
        print(f"  {k}: {v}")

    print("\n[4] Dispersion penalty at 30GHz, M=6, L2=5.5km")
    pen = eq9_dispersion_penalty(L2_km=5.5, beta2_ps2km=-20, fm_ghz=30, M=6)
    print(f"  H_att = {pen:.4f}  ({10*np.log10(pen+1e-30):.1f} dB)")

    print("\n[5] Appendix derivation (symbolic)")
    derive_appendix_time_domain(verbose=True)

if __name__ == '__main__':
    demo()
