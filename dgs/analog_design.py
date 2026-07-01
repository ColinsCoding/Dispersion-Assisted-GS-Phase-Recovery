"""Analog circuit design -- op-amps, filters, ADC/DAC, transmission lines.

WHY ANALOG DESIGN STILL MATTERS (between wars):
  Every digital signal starts and ends as analog.
  A/D converters: analog voltage -> digital bits
  D/A converters: digital bits -> analog voltage -> speaker, laser driver, motor
  RF amplifiers: antenna (analog) -> LNA (analog) -> ADC (mixed-signal)
  Laser drivers: DAC + TIA + PIN photodiode = the GS receiver hardware front-end

  The GS receiver in this repo is a MIXED-SIGNAL system:
    Optical signal -> photodetector -> TIA -> ADC -> digital GS algorithm
  The noise floor is set by the ANALOG front-end, not the digital algorithm.

OP-AMP IDEAL ASSUMPTIONS (the basis of all op-amp analysis):
  1. Infinite open-loop gain A_OL -> infinity
  2. Infinite input impedance (no current into +/- terminals)
  3. Zero output impedance (can drive any load)
  4. Virtual short: v+ = v- when negative feedback is applied
  Implication: the op-amp forces its output to whatever makes v+ = v-.

FUNDAMENTAL OP-AMP TOPOLOGIES:
  Inverting amplifier:     Av = -Rf / Rin
  Non-inverting amplifier: Av = 1 + Rf / Rin
  Voltage follower:        Av = 1  (buffer; high input Z, low output Z)
  Summing amplifier:       Vout = -Rf*(V1/R1 + V2/R2 + ...)
  Difference amplifier:    Vout = Rf/Rin * (V2 - V1)
  Integrator:              Vout = -1/(RC) * integral(Vin dt)
  Differentiator:          Vout = -RC * dVin/dt

ACTIVE FILTERS (s-domain):
  First-order LPF (RC):  H(s) = 1 / (1 + s*RC)        f_c = 1/(2*pi*RC)
  Second-order Butterworth LPF: H(s) = 1 / (s^2/w0^2 + s*sqrt(2)/w0 + 1)
  Active filters use op-amps to avoid loading (unlike passive RLC)
  Sallen-Key topology: non-inverting 2nd order LP/HP/BP using 1 op-amp

NOISE IN ANALOG CIRCUITS:
  Johnson-Nyquist (thermal): v_n^2 = 4*k_B*T*R*Delta_f
  Shot noise: i_n^2 = 2*q*I*Delta_f
  1/f (flicker) noise: PSD ~ 1/f, dominant below ~10 kHz
  Input-referred noise: v_n_in = sqrt(4kTR_in + V_n_opamp^2 + (I_n_opamp*R_in)^2)
  Noise figure: NF = 10*log10(SNR_in / SNR_out)

TRANSMISSION LINES (analog at high frequency):
  At f > c/(10*L) the wire is electrically long: use transmission line theory.
  Characteristic impedance: Z0 = sqrt(L'/C') where L', C' are per-unit-length.
  Coax: Z0 = (60/sqrt(eps_r)) * ln(D/d) typically 50 or 75 Ohm.
  Termination: if Z_load != Z0, there is a reflection: Gamma = (ZL-Z0)/(ZL+Z0).
  VSWR = (1 + |Gamma|) / (1 - |Gamma|)

ADC/DAC (the mixed-signal bridge):
  Resolution: N bits -> 2^N quantization levels, LSB = V_FS / 2^N
  SNR_ideal = 6.02*N + 1.76 dB  (for sinusoidal input, ideal quantizer)
  ENOB: effective number of bits accounting for real noise: SNR = 6.02*ENOB + 1.76
  Nyquist: sample at fs >= 2*f_max (avoid aliasing)
  For the GS receiver: we need fs >= 20 GHz for 10 GHz signal bandwidth.
"""
import numpy as np
import sympy as sp


# ── physical constants ─────────────────────────────────────────────────

k_B = 1.380649e-23  # J/K Boltzmann
q_e = 1.602176634e-19  # C electron charge
T0_K = 290.0        # standard noise temperature (Friis)


# ── op-amp topologies ─────────────────────────────────────────────────

def inverting_amplifier(R_in_ohm, R_f_ohm, V_in_V):
    """Ideal inverting op-amp: Av = -Rf/Rin, Vout = -Rf/Rin * Vin.

    Virtual ground: v- = v+ = 0 (v+ tied to GND).
    Input impedance = R_in (not infinite -- set by Rin).
    """
    if R_in_ohm <= 0 or R_f_ohm <= 0:
        raise ValueError("resistances must be positive")
    Av = -R_f_ohm / R_in_ohm
    V_out = Av * V_in_V
    return {"Av": Av, "V_out_V": V_out, "R_in_ohm": R_in_ohm, "R_f_ohm": R_f_ohm}


def noninverting_amplifier(R_in_ohm, R_f_ohm, V_in_V):
    """Ideal non-inverting op-amp: Av = 1 + Rf/Rin.

    Input impedance: effectively infinite (+ terminal).
    Used for buffers (Rf=0, Rin=inf -> Av=1) and voltage followers.
    """
    if R_in_ohm <= 0 or R_f_ohm < 0:
        raise ValueError("R_in must be positive; R_f must be non-negative")
    Av = 1 + R_f_ohm / R_in_ohm
    V_out = Av * V_in_V
    return {"Av": Av, "V_out_V": V_out}


def summing_amplifier(R_f_ohm, inputs):
    """Inverting summing amplifier: Vout = -Rf * sum(Vi/Ri).

    inputs: list of (V_i, R_i) tuples.
    Virtual ground applies to v- terminal.
    Used in: DAC (weighted sum), audio mixer.
    """
    if R_f_ohm <= 0:
        raise ValueError("R_f must be positive")
    if not inputs:
        raise ValueError("inputs must be a non-empty list of (V, R) tuples")
    V_out = -R_f_ohm * sum(V / R for V, R in inputs)
    return {"V_out_V": V_out, "R_f_ohm": R_f_ohm, "n_inputs": len(inputs)}


def op_amp_integrator(R_ohm, C_F, V_in_V, t_s):
    """Op-amp integrator: Vout(t) = -1/(RC) * integral(Vin dt) for constant Vin.

    This is the HARDWARE IMPLEMENTATION of integration:
    Vout = -Vin * t / (R * C)

    Used in: ADC (charge integration), PID controller (I term), waveform generation.
    WARNING: DC offset causes integrator saturation (add reset switch or R_f in parallel).
    """
    if R_ohm <= 0 or C_F <= 0 or t_s < 0:
        raise ValueError("R, C must be positive; t must be non-negative")
    tau = R_ohm * C_F
    V_out = -V_in_V * t_s / tau
    return {"V_out_V": V_out, "tau_s": tau, "t_s": t_s, "RC_product": tau}


def op_amp_differentiator(R_ohm, C_F, dVin_dt_Vs):
    """Op-amp differentiator: Vout = -RC * dVin/dt.

    High-frequency noise is amplified by differentiation -- use a series
    resistor with C to limit gain at high frequency (practical differentiator).
    """
    if R_ohm <= 0 or C_F <= 0:
        raise ValueError("R and C must be positive")
    V_out = -R_ohm * C_F * dVin_dt_Vs
    return {"V_out_V": V_out, "RC_product": R_ohm * C_F}


# ── active filters ─────────────────────────────────────────────────────

def first_order_lpf(R_ohm, C_F, f_Hz):
    """First-order RC low-pass filter transfer function |H(f)|.

    H(f) = 1 / (1 + j * f / f_c)   where f_c = 1/(2*pi*RC)
    |H| = 1 / sqrt(1 + (f/f_c)^2)
    Phase = -atan(f/f_c)
    At f = f_c: |H| = 1/sqrt(2) = -3 dB, phase = -45 deg.
    """
    if R_ohm <= 0 or C_F <= 0:
        raise ValueError("R and C must be positive")
    f_c = 1 / (2 * np.pi * R_ohm * C_F)
    H_mag = 1 / np.sqrt(1 + (f_Hz / f_c)**2)
    H_dB = 20 * np.log10(H_mag) if H_mag > 0 else -np.inf
    phase_deg = -np.degrees(np.arctan(f_Hz / f_c))
    return {
        "f_c_Hz": f_c, "H_magnitude": H_mag,
        "H_dB": H_dB, "phase_deg": phase_deg,
        "at_3dB": np.abs(H_dB + 3) < 0.1 if np.isfinite(H_dB) else False,
    }


def butterworth_lpf_2nd_order(f_c_Hz, f_Hz):
    """Second-order Butterworth LPF: maximally flat in passband.

    H(s) = 1 / (s^2/w0^2 + s*sqrt(2)/w0 + 1)
    |H(jw)| = 1 / sqrt(1 + (f/f_c)^4)  (2nd order: rolls off at -40 dB/decade)
    Phase = -atan2(sqrt(2)*w/w0, 1 - (w/w0)^2)

    Sallen-Key topology: 2 equal resistors R, 2 equal capacitors C, one op-amp.
    Set f_c = 1/(2*pi*RC), Q = 1/sqrt(2) ~ 0.707 for Butterworth.
    """
    if f_c_Hz <= 0:
        raise ValueError("f_c_Hz must be positive")
    omega = 2 * np.pi * f_Hz
    omega_c = 2 * np.pi * f_c_Hz
    u = omega / omega_c
    H_mag = 1 / np.sqrt(1 + u**4)
    H_dB = 20 * np.log10(H_mag) if H_mag > 0 else -np.inf
    phase_deg = -np.degrees(np.arctan2(np.sqrt(2) * u, 1 - u**2))
    return {
        "f_c_Hz": f_c_Hz, "f_Hz": f_Hz,
        "H_magnitude": H_mag, "H_dB": H_dB,
        "phase_deg": phase_deg,
        "Q_butterworth": 1 / np.sqrt(2),
        "rolloff_dB_per_decade": -40,
    }


def butterworth_order_for_spec(f_pass_Hz, f_stop_Hz, A_pass_dB, A_stop_dB):
    """Minimum Butterworth order to meet passband/stopband spec.

    n >= log(10^(A_stop/10) - 1) / (10^(A_pass/10) - 1)) / (2 * log(f_stop/f_pass))
    """
    if f_pass_Hz <= 0 or f_stop_Hz <= f_pass_Hz:
        raise ValueError("f_stop must be > f_pass > 0")
    if A_pass_dB <= 0 or A_stop_dB <= 0:
        raise ValueError("attenuations must be positive dB values")
    num = np.log10((10**(A_stop_dB / 10) - 1) / (10**(A_pass_dB / 10) - 1))
    den = 2 * np.log10(f_stop_Hz / f_pass_Hz)
    n = np.ceil(num / den)
    return {"n_order": int(n), "f_pass_Hz": f_pass_Hz, "f_stop_Hz": f_stop_Hz,
            "A_pass_dB": A_pass_dB, "A_stop_dB": A_stop_dB}


# ── noise ─────────────────────────────────────────────────────────────

def johnson_nyquist_noise(R_ohm, BW_Hz, T_K=290.0):
    """Johnson-Nyquist thermal noise voltage: v_n = sqrt(4*k_B*T*R*BW).

    This is the fundamental noise floor from thermal agitation of electrons.
    Cannot be reduced by circuit design -- only by lowering temperature or bandwidth.
    """
    if R_ohm < 0 or BW_Hz < 0 or T_K < 0:
        raise ValueError("R, BW, and T must be non-negative")
    v_n_sq = 4 * k_B * T_K * R_ohm * BW_Hz
    v_n = np.sqrt(v_n_sq)
    v_n_uV = v_n * 1e6
    return {
        "R_ohm": R_ohm, "BW_Hz": BW_Hz, "T_K": T_K,
        "v_n_rms_V": v_n, "v_n_rms_uV": v_n_uV,
        "v_n_density_nV_rtHz": v_n / np.sqrt(BW_Hz) * 1e9 if BW_Hz > 0 else 0,
    }


def noise_figure_cascaded(NF_stages_dB, gain_stages_dB):
    """Friis formula for cascaded noise figure.

    NF_total = NF1 + (NF2-1)/G1 + (NF3-1)/(G1*G2) + ...

    Key insight: the FIRST stage dominates! Put the LNA (Low Noise Amplifier)
    FIRST to minimize noise figure. This is why the GS receiver front-end
    must have a low-noise TIA as the first analog stage.
    """
    if len(NF_stages_dB) != len(gain_stages_dB):
        raise ValueError("NF_stages and gain_stages must have same length")
    if not NF_stages_dB:
        raise ValueError("must provide at least one stage")
    F = [10**(nf/10) for nf in NF_stages_dB]   # convert to linear
    G = [10**(g/10) for g in gain_stages_dB]

    F_total = F[0]
    cumulative_gain = 1.0
    for i in range(1, len(F)):
        cumulative_gain *= G[i-1]
        F_total += (F[i] - 1) / cumulative_gain

    NF_total_dB = 10 * np.log10(F_total)
    return {
        "NF_total_dB": NF_total_dB,
        "F_total_linear": F_total,
        "dominant_stage": 0,
        "note": "First stage dominates -- put LNA first",
    }


# ── ADC/DAC ──────────────────────────────────────────────────────────

def adc_specs(n_bits, V_full_scale_V, f_sample_Hz):
    """Key ADC specifications.

    SNR_ideal = 6.02*N + 1.76 dB (for sinusoidal input, ideal quantizer).
    LSB = V_FS / 2^N.
    SFDR (spurious-free dynamic range) and ENOB are hardware-dependent.
    """
    if n_bits < 1 or V_full_scale_V <= 0 or f_sample_Hz <= 0:
        raise ValueError("n_bits >= 1; V_FS and f_sample must be positive")
    n_levels = 2**n_bits
    LSB_V = V_full_scale_V / n_levels
    SNR_ideal_dB = 6.02 * n_bits + 1.76
    f_nyquist_Hz = f_sample_Hz / 2
    return {
        "n_bits": n_bits,
        "n_levels": n_levels,
        "LSB_V": LSB_V,
        "LSB_uV": LSB_V * 1e6,
        "V_full_scale_V": V_full_scale_V,
        "f_sample_Hz": f_sample_Hz,
        "f_nyquist_Hz": f_nyquist_Hz,
        "SNR_ideal_dB": SNR_ideal_dB,
    }


def enob_from_snr(measured_snr_dB):
    """ENOB = (SNR_dB - 1.76) / 6.02."""
    return {"ENOB": (measured_snr_dB - 1.76) / 6.02,
            "measured_snr_dB": measured_snr_dB}


# ── transmission line ─────────────────────────────────────────────────

def coax_impedance(D_outer_m, d_inner_m, eps_r=2.25):
    """Characteristic impedance of a coaxial transmission line.

    Z0 = (60/sqrt(eps_r)) * ln(D/d)
    Standard values: RG-58 = 50 Ohm, RG-59 = 75 Ohm
    """
    if D_outer_m <= d_inner_m or d_inner_m <= 0:
        raise ValueError("D_outer > d_inner > 0 required")
    Z0 = (60 / np.sqrt(eps_r)) * np.log(D_outer_m / d_inner_m)
    return {"Z0_ohm": Z0, "D_outer_m": D_outer_m,
            "d_inner_m": d_inner_m, "eps_r": eps_r}


def transmission_line_reflection(Z_load_ohm, Z0_ohm=50.0, f_Hz=1e9, length_m=0.1):
    """Reflection coefficient, VSWR, and input impedance of a terminated line.

    Gamma = (ZL - Z0) / (ZL + Z0)
    VSWR = (1 + |Gamma|) / (1 - |Gamma|)
    Z_in = Z0 * (ZL + j*Z0*tan(beta*l)) / (Z0 + j*ZL*tan(beta*l))
    """
    if Z0_ohm <= 0:
        raise ValueError("Z0 must be positive")
    Gamma = (Z_load_ohm - Z0_ohm) / (Z_load_ohm + Z0_ohm)
    mag_Gamma = abs(Gamma)
    if mag_Gamma < 1 - 1e-9:
        VSWR = (1 + mag_Gamma) / (1 - mag_Gamma)
    else:
        VSWR = float("inf")
    c = 2.998e8
    beta = 2 * np.pi * f_Hz / c
    beta_l = beta * length_m
    Z_in = Z0_ohm * (Z_load_ohm + 1j*Z0_ohm*np.tan(beta_l)) / \
           (Z0_ohm + 1j*Z_load_ohm*np.tan(beta_l))
    return {
        "Gamma": Gamma, "Gamma_mag": mag_Gamma,
        "VSWR": VSWR, "Z_in_ohm": Z_in,
        "return_loss_dB": -20 * np.log10(mag_Gamma) if mag_Gamma > 0 else float("inf"),
    }


# ── GS receiver front-end sizing ─────────────────────────────────────

def gs_receiver_analog_frontend(
        I_photo_uA=100.0, R_TIA_kohm=10.0,
        BW_GHz=5.0, f_sample_GHz=20.0, n_bits_ADC=8):
    """Size the analog front-end for the TD-GS optical receiver.

    Signal chain: PIN photodetector -> TIA -> Anti-alias LPF -> ADC

    Transimpedance amplifier (TIA): converts photocurrent I to voltage V = I*Rf.
    The TIA bandwidth-gain tradeoff: f_3dB ~ 1/(2*pi*Rf*C_in) where C_in is
    the detector capacitance. For 5 GHz bandwidth with C_in=0.5pF: Rf < 64 Ohm.

    Rule: Rf*BW < 1/(2*pi*C_in). Higher transimpedance -> narrower bandwidth.
    """
    if I_photo_uA <= 0 or R_TIA_kohm <= 0 or BW_GHz <= 0:
        raise ValueError("I_photo, R_TIA, BW must be positive")
    V_signal_mV = I_photo_uA * 1e-6 * R_TIA_kohm * 1e3 * 1e3   # mV
    noise = johnson_nyquist_noise(R_TIA_kohm * 1e3, BW_GHz * 1e9)
    adc = adc_specs(n_bits_ADC, V_signal_mV / 1000 * 4, f_sample_GHz * 1e9)
    SNR_signal_dB = 20 * np.log10(V_signal_mV / (noise["v_n_rms_uV"] / 1000))
    return {
        "V_signal_mV": V_signal_mV,
        "thermal_noise_uV_rms": noise["v_n_rms_uV"],
        "SNR_analog_dB": SNR_signal_dB,
        "ADC_LSB_uV": adc["LSB_uV"],
        "ADC_SNR_ideal_dB": adc["SNR_ideal_dB"],
        "ADC_f_nyquist_GHz": adc["f_nyquist_Hz"] / 1e9,
        "bandwidth_GHz": BW_GHz,
        "oversampled_by": f_sample_GHz / (2 * BW_GHz),
        "note": "Thermal noise sets floor; ADC bits set ceiling; TIA Rf sets BW-gain tradeoff",
    }


# ── SymPy formalism ───────────────────────────────────────────────────

def analog_design_sympy_5():
    """Five key analog design equations in SymPy (init_printing ready)."""
    R_f, R_in, V_in = sp.symbols('R_f R_in V_in', positive=True)
    R_s, C_s, f_s = sp.symbols('R C f', positive=True)
    n_s = sp.Symbol('n', positive=True, integer=True)
    Z_L, Z_0, Gamma = sp.symbols('Z_L Z_0 Gamma')
    k_B_s, T_s, Delta_f = sp.symbols('k_B T Delta_f', positive=True)

    return {
        "Inverting_amplifier":
            sp.Eq(sp.Symbol('A_v'), -R_f / R_in),
        "RC_cutoff_frequency":
            sp.Eq(sp.Symbol('f_c'), 1 / (2 * sp.pi * R_s * C_s)),
        "Reflection_coefficient":
            sp.Eq(Gamma, (Z_L - Z_0) / (Z_L + Z_0)),
        "Ideal_ADC_SNR":
            sp.Eq(sp.Symbol('SNR_dB'), 6.02 * n_s + sp.Rational(176, 100)),
        "Johnson_noise":
            sp.Eq(sp.Symbol('v_n'),
                  sp.sqrt(4 * k_B_s * T_s * R_s * Delta_f)),
    }


if __name__ == "__main__":
    print("=== Op-amp: inverting amplifier ===")
    amp = inverting_amplifier(1e3, 10e3, 0.1)
    print(f"  Av = {amp['Av']:.1f},  Vout = {amp['V_out_V']:.2f} V")

    print("\n=== Second-order Butterworth LPF (f_c=1MHz) ===")
    for f in [0.1e6, 1e6, 10e6]:
        h = butterworth_lpf_2nd_order(1e6, f)
        print(f"  f={f/1e6:.1f} MHz: |H|={h['H_magnitude']:.4f} ({h['H_dB']:.1f} dB)")

    print("\n=== Thermal noise floor ===")
    n = johnson_nyquist_noise(50, 5e9, 290)
    print(f"  50 Ohm, 5 GHz BW, 290K: {n['v_n_rms_uV']:.1f} uV rms")
    print(f"  Noise density: {n['v_n_density_nV_rtHz']:.1f} nV/rtHz")

    print("\n=== Cascaded noise figure: LNA + Mixer ===")
    nf = noise_figure_cascaded([1.5, 8.0], [15.0, 0.0])
    print(f"  NF_total = {nf['NF_total_dB']:.2f} dB (dominated by LNA)")

    print("\n=== ADC specs: 8-bit, 2V, 20 GSa/s ===")
    a = adc_specs(8, 2.0, 20e9)
    print(f"  LSB = {a['LSB_uV']:.0f} uV,  SNR_ideal = {a['SNR_ideal_dB']:.1f} dB")

    print("\n=== GS receiver front-end sizing ===")
    fe = gs_receiver_analog_frontend()
    for k, v in fe.items():
        if k != "note":
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  note: {fe['note']}")

    print("\n=== SymPy 5 (init_printing ready) ===")
    sp.init_printing(use_unicode=False)
    for k, eq in analog_design_sympy_5().items():
        print(f"  {k}: {eq}")
