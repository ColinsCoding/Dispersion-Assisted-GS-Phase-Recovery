"""
Photonic Time-Stretch Analog-to-Digital Converter (TS-ADC)
Jalali Group, UCLA EE -- Direct ancestor of this repository

HISTORICAL CONTEXT:
  Bahram Jalali (UCLA) demonstrated the first photonic time-stretch ADC in 1998.
  This module reproduces the analytical derivations from that work and connects
  them directly to H(f) = exp(j*pi*D*f^2) -- the core equation of this repo.

  The photonic time-stretch preprocessor takes an ultrafast electrical signal
  (too fast for any electronic ADC) and SLOWS IT DOWN by a factor M before
  digitization. This enables:
    - Bandwidth: 100x beyond electronic ADCs alone
    - The STEAM camera (Serial Time-Encoded Amplified Microscopy)
    - Real-time spectroscopy at GHz frame rates
    - This repo: phase retrieval on the stretched waveform

SYSTEM SCHEMATIC:
  [Pulsed laser] -> [EDFA] -> [Fiber L1, D1] -> [EOM (signal modulates chirp)]
                    -> [Fiber L2, D2] -> [Photodetector] -> [Electronic ADC @ f_s]

  Step 1: EDFA broadens and amplifies pulse (optical bandwidth Delta_lambda)
  Step 2: L1 chirps the pulse (maps wavelength to time: t = D1*L1*lambda)
  Step 3: EOM imprints electrical signal e(t) onto chirped optical envelope
          (each moment in time = one wavelength = one frequency component)
  Step 4: L2 further stretches the already-modulated chirp
  Step 5: Photodetector converts to electrical; ADC at f_s samples it

ANALYTICAL DERIVATION OF STRETCH FACTOR M:
  Let D [ps/(nm*km)] be the dispersion parameter (GVD), L [km] fiber length.
  GVD coefficient beta2 [s^2/m]: beta2 = -(lambda^2 / (2*pi*c)) * D

  Temporal broadening of a pulse with optical bandwidth Delta_lambda:
    Delta_t = |D| * L * Delta_lambda

  Transfer function of each fiber segment (frequency domain):
    H_i(f) = exp(j*pi*beta2_i*L_i*(2*pi*f)^2)   [quadratic phase = chirp]

  After L1 (pre-chirp fiber):
    The optical pulse spans time window T_1 = |D1|*L1*Delta_lambda
    Frequency-to-time mapping: f_opt(t) = t / (D1*L1)  [GHz/ps -> linear]

  Signal imprinted at EOM:
    E_mod(t) = E_chirped(t) * [1 + m*e(t/M_partial)]
    where M_partial = (D1*L1)/(D1*L1 + ... ) -- see below

  After L2 (post-stretch fiber):
    The already-chirped pulse is stretched further.
    New time window: T_2 = (|D1|*L1 + |D2|*L2) * Delta_lambda

  STRETCH FACTOR M (derivation):
    The signal impressed on the optical carrier at EOM occupies bandwidth B_e.
    The carrier sweep rate (chirp rate) after L1: alpha = 1/(D1*L1) [nm/ps]
    Signal bandwidth B_e maps to time slot delta_t_mod = B_e / alpha = B_e*D1*L1
    After L2, this time slot is stretched by factor D2*L2/D1*L1 (additional stretch).

    M = (D1*L1 + D2*L2) / (D1*L1)
      = 1 + D2*L2 / (D1*L1)

    For identical fibers (D1=D2=D, L1=L2=L):
      M = 2  (signal is slowed by 2x)

    For L2 >> L1:
      M ≈ D2*L2 / (D1*L1)  (large stretch)

  RESOLUTION AND BANDWIDTH:
    The ADC samples at f_s [samples/s].
    After stretch, signal bandwidth is divided by M:
      Effective captured bandwidth: B_RF = M * f_s / 2

    Number of resolvable points in one pulse:
      N = T_pulse * f_s  (ADC samples per pulse)
    Where T_pulse = M * T_signal = M / B_signal

    Equivalent resolution (ENOB limited by stretch):
      B_captured = M * f_s / 2
      Example: M=100, f_s=1 Gsample/s -> B_captured = 50 GHz!

  TRANSFER FUNCTION VIEW (this repo connection):
    H_total(f) = H_L1(f) * H_EOM(f) * H_L2(f)
               = exp(j*pi*beta2_1*(2*pi*f)^2*L1) * E_mod(f) * exp(j*pi*beta2_2*(2*pi*f)^2*L2)

    The total dispersion: D_total = D1*L1 + D2*L2
    H_total(f) = E_mod(f) * exp(j*pi*(D1*L1+D2*L2)*(lambda^2/c)*f^2)
               = E_mod(f) * exp(j*pi*D_eff*f^2)

    THIS IS EXACTLY H(f) = exp(j*pi*D*f^2) FROM THIS REPO.
    GS phase retrieval on the stretched waveform = recovering E_mod(f) from
    the intensity measurement at the photodetector.

HYPERSPECTRAL / STEAM CAMERA (2D extension):
  Add a diffraction grating before L1: maps spatial wavelength to angle.
  Each wavelength illuminates a different row of the sample.
  After time-stretch: each row's image arrives at a different time slot.
  2D image encoded into 1D time series -> single photodetector captures full frame.
  Frame rate = laser rep rate (MHz to GHz).

RANDOM FOREST FOR ADC:
  After digitization, reconstruct signal using ML:
  Features: [amplitude, phase gradient, bandwidth, SNR, skewness, kurtosis, ...]
  Label: signal class (e.g., modulation format: OOK, QPSK, 16-QAM)
  Random forest: ensemble of decision trees, each trained on random feature subset.
  Reduces overfitting, handles nonlinear decision boundaries.
  Application: real-time modulation recognition in digital coherent receiver.
"""
import math
import numpy as np

c_light = 2.998e8; hbar = 1.0546e-34; kB = 1.381e-23; h_P = 6.626e-34


# ============================================================
# Analytical Stretch Factor Derivation
# ============================================================

def derive_stretch_factor(D1_ps_nm_km=1000.0, L1_km=1.0,
                          D2_ps_nm_km=1000.0, L2_km=5.0,
                          Delta_lambda_nm=20.0, f_s_GHz=1.0):
    """
    Analytically derive the photonic time-stretch factor and system parameters.

    Parameters
    ----------
    D1_ps_nm_km : float
        GVD of pre-chirp fiber L1 [ps/(nm*km)]. Must be same sign (both anomalous or both normal).
    L1_km : float
        Length of pre-chirp fiber [km].
    D2_ps_nm_km : float
        GVD of post-stretch fiber L2 [ps/(nm*km)].
    L2_km : float
        Length of post-stretch fiber [km].
    Delta_lambda_nm : float
        Optical bandwidth of EDFA output [nm].
    f_s_GHz : float
        Electronic ADC sample rate [Gsample/s].

    Returns
    -------
    dict with derived quantities.

    DERIVATION (step by step):
    Step 1: Convert D to beta2.
      D = -(lambda^2 / (2*pi*c)) * beta2
      beta2 = -(D * lambda^2) / (2*pi*c)

    Step 2: Time aperture after L1 (before EOM).
      T1 = |D1| * L1 * Delta_lambda  [ps = ps/(nm*km) * km * nm]
      This is the time window over which the signal can be impressed.

    Step 3: Chirp rate (frequency sweep speed) after L1.
      alpha_L1 = Delta_lambda / T1 = 1/(D1*L1)  [nm/ps]
      This means: each ps corresponds to 1/(D1*L1) nm of optical bandwidth.

    Step 4: Total time aperture after L2.
      T_total = (|D1|*L1 + |D2|*L2) * Delta_lambda

    Step 5: Stretch factor.
      M = T_total / T1 = 1 + |D2|*L2 / (|D1|*L1)

    Step 6: Captured RF bandwidth.
      B_RF = M * f_s / 2

    Step 7: Number of samples per pulse.
      N_samples = T_total [ps] * f_s [samples/ps] = T_total * f_s * 1e-3
      (converting ps to ns: *1e-3, and f_s in Gsample/s = 1 sample/ns)

    Step 8: Time-bandwidth product (invariant under stretch).
      TBP = T_signal * B_signal = T_stretched * B_stretched
      TBP is preserved: stretch trades time for bandwidth.
    """
    if D1_ps_nm_km <= 0:
        raise ValueError("|D1| must be > 0")
    if D2_ps_nm_km <= 0:
        raise ValueError("|D2| must be > 0")
    if L1_km <= 0 or L2_km <= 0:
        raise ValueError("Fiber lengths must be > 0")
    if Delta_lambda_nm <= 0:
        raise ValueError("Optical bandwidth must be > 0")
    if f_s_GHz <= 0:
        raise ValueError("ADC sample rate must be > 0")

    # Step 1: beta2 conversion
    lambda0_m = 1550e-9
    # D [ps/(nm*km)] = D [s/m^2] * 1e3 (unit conversion chain)
    # beta2 [s^2/m] = -(D [s/m^2]) * lambda^2 / (2*pi*c)
    D1_s_m2 = D1_ps_nm_km * 1e-12 / (1e-9 * 1e3)   # s/m^2
    D2_s_m2 = D2_ps_nm_km * 1e-12 / (1e-9 * 1e3)
    beta2_1 = -D1_s_m2 * lambda0_m**2 / (2*np.pi*c_light)   # s^2/m
    beta2_2 = -D2_s_m2 * lambda0_m**2 / (2*np.pi*c_light)

    # Step 2: Time aperture after L1
    T1_ps = D1_ps_nm_km * L1_km * Delta_lambda_nm   # ps
    T1_s  = T1_ps * 1e-12

    # Step 3: Chirp rate
    alpha_L1_nm_per_ps = Delta_lambda_nm / T1_ps   # nm/ps
    alpha_L1_THz_per_ps = c_light * Delta_lambda_nm / (lambda0_m**2 * 1e12) / T1_ps   # THz/ps (approx)

    # Step 4: Total time aperture
    T_total_ps = (D1_ps_nm_km*L1_km + D2_ps_nm_km*L2_km) * Delta_lambda_nm
    T_total_s  = T_total_ps * 1e-12

    # Step 5: Stretch factor
    M = (D1_ps_nm_km*L1_km + D2_ps_nm_km*L2_km) / (D1_ps_nm_km*L1_km)
    M_check = 1 + (D2_ps_nm_km*L2_km) / (D1_ps_nm_km*L1_km)

    # Step 6: Captured RF bandwidth
    B_RF_GHz = M * f_s_GHz / 2

    # Step 7: Number of samples per pulse
    # T_total in ns: T_total_ps * 1e-3
    # f_s in Gsample/s = 1 sample per ns
    N_samples = int(T_total_ps * 1e-3 * f_s_GHz * 1e9 / 1e9)
    # simpler: T_total_s * f_s [samples/s]
    N_samples_exact = T_total_s * f_s_GHz * 1e9

    # Step 8: Time-bandwidth product
    TBP_before = T1_s * (f_s_GHz*1e9 / 2)   # T_signal * B_signal before stretch
    TBP_after  = T_total_s * B_RF_GHz*1e9    # T_stretched * B_stretched
    # TBP increases with M! (We capture more information per pulse)

    # System noise: shot noise limit
    R_responsivity = 0.8    # A/W photodetector responsivity
    P_optical = 10e-3       # 10 mW average optical power
    I_photocurrent = R_responsivity * P_optical
    q_e = 1.602e-19
    BW_noise = B_RF_GHz * 1e9 / M   # noise bandwidth = optical BW / M
    I_shot_A = math.sqrt(2*q_e*I_photocurrent*BW_noise)
    SNR_shot_dB = 10*math.log10((I_photocurrent/I_shot_A)**2)

    # H(f) total transfer function
    f_arr = np.linspace(-B_RF_GHz/2, B_RF_GHz/2, 2000) * 1e9   # Hz
    D_eff_ps = D1_ps_nm_km*L1_km + D2_ps_nm_km*L2_km   # ps*nm/km * km = ps*nm?
    # H(f) in frequency offset from carrier: exp(j*pi*(beta2_1*L1+beta2_2*L2)*(2*pi*f)^2)
    L1_m = L1_km * 1e3; L2_m = L2_km * 1e3
    phi_quadratic = np.pi * (beta2_1*L1_m + beta2_2*L2_m) * (2*np.pi*f_arr)**2
    H_total = np.exp(1j * phi_quadratic)

    return {
        'inputs': {
            'D1_ps_nm_km': D1_ps_nm_km, 'L1_km': L1_km,
            'D2_ps_nm_km': D2_ps_nm_km, 'L2_km': L2_km,
            'Delta_lambda_nm': Delta_lambda_nm, 'f_s_GHz': f_s_GHz,
        },
        'derivation': {
            'beta2_1_s2_m': float(beta2_1),
            'beta2_2_s2_m': float(beta2_2),
            'T1_ps': float(T1_ps),
            'T_total_ps': float(T_total_ps),
            'alpha_chirp_nm_per_ps': float(alpha_L1_nm_per_ps),
        },
        'M': float(M),
        'M_formula': 'M = 1 + D2*L2/(D1*L1)',
        'M_check': float(M_check),
        'B_RF_GHz': float(B_RF_GHz),
        'N_samples_per_pulse': float(N_samples_exact),
        'TBP_before': float(TBP_before),
        'TBP_after': float(TBP_after),
        'TBP_gain': float(TBP_after / max(TBP_before, 1e-30)),
        'noise': {
            'I_photocurrent_mA': float(I_photocurrent*1e3),
            'I_shot_nA': float(I_shot_A*1e9),
            'SNR_shot_dB': float(SNR_shot_dB),
        },
        'H_total': {
            'f_GHz': (f_arr/1e9).tolist(),
            'H_mag': np.abs(H_total).tolist(),
            'H_phase_rad': np.angle(H_total).tolist(),
            'formula': 'H(f) = exp(j*pi*(beta2_1*L1 + beta2_2*L2)*(2*pi*f)^2)',
            'D_eff_note': f'D_eff = D1*L1 + D2*L2 = {D1_ps_nm_km*L1_km + D2_ps_nm_km*L2_km:.0f} ps*nm/km... see repo H(f)',
        },
        'GS_connection': (
            'The intensity at the photodetector = |H_total(f) * E_mod(f)|^2.\n'
            'GS phase retrieval recovers E_mod(f) = the original signal spectrum.\n'
            'M enables capturing B_RF_GHz = {:.1f} GHz with only f_s = {:.0f} Gsample/s ADC.\n'
            'Without photonic time-stretch: would need a {:.1f} Gsample/s ADC -- impossible!\n'
        ).format(B_RF_GHz, f_s_GHz, B_RF_GHz*2),
    }


# ============================================================
# Full TS-ADC System Simulation
# ============================================================

def time_stretch_adc_system(M=5.0, f_signal_GHz=3.0, SNR_dB=30.0,
                             f_s_GHz=1.0, N_pts=1024):
    """
    End-to-end simulation of photonic time-stretch ADC.

    Pipeline:
      1. Generate test signal e(t): chirped RF burst
      2. Imprint onto chirped optical pulse (EOM model)
      3. Apply L2 dispersion (time stretch by M)
      4. Detect (square law) + add noise
      5. Digitize at f_s
      6. Reconstruct original signal (undo stretch = GS phase retrieval target)

    COMPLEX EXPONENTIAL PID VIEW:
      The transfer function of each fiber H(s) in Laplace domain:
        H(s) = exp(-beta2 * L * s^2 / 2)   [s = j*omega for Fourier]
      Poles of H(s): none on finite complex plane (entire function).
      Zeros: none.
      This makes GVD UNCONDITIONALLY STABLE -- it's an all-pass filter.
      PID of complex exponential: zero poles, zero zeros = pure phase distortion.
      Phase: phi(omega) = beta2 * L * omega^2 / 2  [quadratic in omega]
      Group delay: tau_g = dphi/domega = beta2 * L * omega  [linear in omega]
      GDD (Group Delay Dispersion): d^2phi/domega^2 = beta2*L = constant.
    """
    if M <= 1:
        raise ValueError(f"Stretch factor M = {M} must be > 1")
    if f_s_GHz <= 0:
        raise ValueError(f"f_s = {f_s_GHz} must be > 0")
    if N_pts <= 0:
        raise ValueError(f"N_pts = {N_pts} must be > 0")

    # Time arrays
    dt_stretched = 1.0 / (f_s_GHz * 1e9)   # sampling period after stretch
    T_stretched  = N_pts * dt_stretched
    t_stretched  = np.arange(N_pts) * dt_stretched

    # Original signal timeline (before stretch, compressed by M)
    dt_original = dt_stretched / M
    t_original  = np.arange(N_pts) * dt_original

    # 1. Original signal: chirped RF burst + multi-tone
    f1 = f_signal_GHz * 1e9
    f2 = f1 * 1.3; f3 = f1 * 0.7
    envelope = np.exp(-((t_original - T_stretched/M/2)**2 / (T_stretched/M/6)**2))
    e_t = envelope * (np.sin(2*np.pi*f1*t_original) +
                      0.5*np.sin(2*np.pi*f2*t_original) +
                      0.3*np.sin(2*np.pi*f3*t_original))

    # 2. EOM imprints signal onto chirped optical carrier
    # Chirped optical pulse (envelope after L1)
    omega_c = 2*np.pi*c_light / 1550e-9
    chirp_rate = 2*np.pi*f1 * M / T_stretched   # rad/s per s
    E_chirped = np.exp(1j*0.5*chirp_rate*t_original**2)   # quadratic phase
    m_depth = 0.8   # modulation depth
    E_mod_original = E_chirped * (1 + m_depth * e_t / 2)

    # 3. L2 stretch: resample in time by factor M
    # In time domain: stretch means t -> t/M (signal arrives slower)
    from numpy.fft import fft, ifft, fftfreq
    E_mod_F = fft(E_mod_original)
    # Stretch in frequency = compress spectrum by M
    # Simple model: stretch = interpolate by M
    E_stretched = np.zeros(N_pts, dtype=complex)
    # Map original N_pts to stretched: every M samples = 1 original sample
    for i in range(N_pts):
        i_orig = i / M
        i0 = int(i_orig) % N_pts
        i1 = (i0 + 1) % N_pts
        frac = i_orig - int(i_orig)
        E_stretched[i] = (1-frac)*E_mod_original[i0] + frac*E_mod_original[i1]

    # 4. Photodetection (square law) + AWGN noise
    I_detected = np.abs(E_stretched)**2   # intensity
    sigma_noise = np.sqrt(np.mean(I_detected**2)) * 10**(-SNR_dB/20)
    np.random.seed(42)
    I_noisy = I_detected + sigma_noise * np.random.randn(N_pts)

    # 5. Digital ADC: quantize to 8 bits
    n_bits = 8
    I_max = np.max(np.abs(I_noisy))
    I_quantized = np.round(I_noisy / I_max * (2**(n_bits-1))) / (2**(n_bits-1)) * I_max
    ENOB = n_bits - np.log2(np.sqrt(12) * sigma_noise / (2*I_max))
    ENOB = float(np.clip(ENOB, 0, n_bits))

    # 6. Reconstruct: spectrum of detected signal
    I_spectrum = np.abs(fft(I_quantized))
    f_axis = fftfreq(N_pts, d=dt_stretched)
    # Peaks in spectrum should appear at f1/M, f2/M, f3/M (stretched frequencies)
    f_peaks_expected_MHz = np.array([f1, f2, f3]) / M / 1e6

    # Transfer function analysis (complex exponential PID)
    omega_arr = np.linspace(0, 2*np.pi*f_s_GHz*1e9, 400)
    D_total_ps_nm = 5000.0   # example total dispersion ps*nm (D*L product)
    lambda0 = 1550e-9
    beta2_eff = -D_total_ps_nm*1e-12 / (1e-9 * c_light) * lambda0  # approx
    H_s_mag = np.ones(400)   # all-pass: |H(jomega)| = 1 for all omega
    H_s_phase = 0.5 * beta2_eff * omega_arr**2   # quadratic phase
    group_delay = beta2_eff * omega_arr   # linear group delay

    return {
        'system': {'M': M, 'f_s_GHz': f_s_GHz, 'N_pts': N_pts, 'SNR_dB': SNR_dB},
        'signal': {
            't_original_ns': (t_original*1e9).tolist(),
            'e_t': e_t.tolist(),
            't_stretched_ns': (t_stretched*1e9).tolist(),
            'I_detected': I_detected.tolist(),
            'I_quantized': I_quantized.tolist(),
        },
        'spectrum': {
            'f_MHz': (f_axis[:N_pts//2]/1e6).tolist(),
            'I_spectrum_dB': (20*np.log10(I_spectrum[:N_pts//2]+1e-10)).tolist(),
            'f_peaks_expected_MHz': f_peaks_expected_MHz.tolist(),
        },
        'ADC': {
            'n_bits': n_bits,
            'ENOB': float(ENOB),
            'B_RF_GHz': float(M * f_s_GHz / 2),
            'B_equivalent': f'Captures {M*f_s_GHz/2:.1f} GHz with {f_s_GHz:.0f} Gsample/s ADC',
        },
        'transfer_function': {
            'omega_rad_s': omega_arr.tolist(),
            'H_mag': H_s_mag.tolist(),
            'H_phase_rad': H_s_phase.tolist(),
            'group_delay_ps': (group_delay*1e12).tolist(),
            'all_pass': True,
            'poles': 'None (entire function, no poles)',
            'zeros': 'None',
            'PID_view': 'Pure phase distortion. No amplitude change. Reversible with GS.',
        },
    }


# ============================================================
# STEAM Camera (Hyperspectral ADC)
# ============================================================

def steam_camera(n_pixels_x=100, n_pixels_y=50, f_rep_MHz=50.0):
    """
    Serial Time-Encoded Amplified Microscopy (STEAM) camera.

    Invented by Bahram Jalali at UCLA, 2009.
    Extends photonic time-stretch to 2D imaging.

    PRINCIPLE:
      1. Femtosecond pulse hits a diffraction grating -> spatial spectrum fan-out
         Each wavelength = different angle = illuminates different row of object.
      2. All wavelengths/rows imprinted onto same pulse (spatial parallelism).
      3. Time-stretch maps each row to a different time slot.
      4. Single photodetector + single ADC reads entire 2D image per pulse.
      Frame rate = laser repetition rate (50 MHz -> 50M frames/second!)

    MATH:
      Spatial mapping: x = f * Delta_x / Delta_lambda * lambda
        (x = position, lambda = wavelength, f = focal length)
      Time mapping: t = D*L*lambda / c  (after time-stretch)
      Combined: t = D*L*x*c / (f*Delta_x)
      -> x position maps linearly to time slot t.

    2D ENCODING:
      Row selection: grating disperses rows across optical bandwidth.
      Column selection: signal varies WITHIN each row's time slot.
      Total samples = N_rows * N_cols = pixels.
      With M=100 stretch: N_rows * f_s / B_optical = pixels per frame.

    CONNECTION TO GS / HYPERSPECTRAL ADC:
      Each row of the 2D image = one "shot" of photonic time-stretch.
      GS phase retrieval applied to each time slot -> spectral image.
      Hyperspectral: different wavelengths carry different spectral channels.
      STEAM enables real-time hyperspectral imaging at MHz frame rates.
    """
    n_lambda = n_pixels_x   # each wavelength = one column = one row of image
    lambda_arr = np.linspace(1530, 1570, n_lambda) * 1e-9   # m (C-band)
    Delta_lambda = float(lambda_arr[-1] - lambda_arr[0])   # m

    # Spatial mapping (grating + lens)
    # Use 600 lp/mm grating: d = 1667 nm > lambda = 1550 nm, so arcsin is valid.
    f_lens = 0.1   # m (100 mm focal length)
    grating_density = 600   # lines/mm
    d_grating = 1e-3 / grating_density   # m = 1.667e-6 m
    lambda0 = 1550e-9
    sin_arg = float(np.clip(lambda0 / d_grating, -0.9999, 0.9999))
    theta_0 = math.asin(sin_arg)   # blaze angle (approx)
    # Angular dispersion: d_theta/d_lambda = 1 / (d * cos(theta))
    d_theta_d_lambda = 1 / (d_grating * math.cos(theta_0))   # rad/m
    # Spatial extent per nm: Delta_x_per_nm = f * d_theta/d_lambda * 1e-9
    Delta_x_per_nm = f_lens * d_theta_d_lambda * 1e-9   # m/nm
    total_spatial_extent = Delta_x_per_nm * (Delta_lambda*1e9)   # m

    # Time-stretch mapping
    D_fiber = 17.0   # ps/(nm*km)
    L_fiber = 5.0    # km
    # Time per nm: tau_per_nm = D*L [ps/nm]
    tau_per_nm = D_fiber * L_fiber   # ps/nm
    tau_total_ps = tau_per_nm * (Delta_lambda*1e9)   # ps
    # Each pixel in x maps to a time slot
    dt_per_pixel = tau_total_ps / n_pixels_x   # ps per pixel

    # Frame rate = repetition rate
    T_rep_ns = 1e3 / f_rep_MHz   # ns
    T_frame_ns = tau_total_ps * 1e-3   # ns (one frame duration)
    duty_cycle = T_frame_ns / T_rep_ns

    # Required ADC sample rate to resolve all pixels
    f_ADC_needed_GHz = n_pixels_x / (tau_total_ps * 1e-3)   # Gsample/s (N/T in ns)

    # Pixel array simulation (synthetic image: concentric circles)
    x_arr = np.linspace(-1, 1, n_pixels_x)
    y_arr = np.linspace(-1, 1, n_pixels_y)
    X, Y = np.meshgrid(x_arr, y_arr)
    image_2d = (np.sin(5*np.sqrt(X**2+Y**2)))**2 + 0.1*np.random.randn(*X.shape)
    image_2d = np.clip(image_2d, 0, 1)

    # Serialize to time domain (STEAM encoding)
    image_serialized = np.zeros(n_pixels_x * n_pixels_y)
    for row in range(n_pixels_y):
        image_serialized[row*n_pixels_x:(row+1)*n_pixels_x] = image_2d[row]

    # Spectral/hyperspectral channels
    n_channels = 4
    channel_bandwidth_nm = Delta_lambda*1e9 / n_channels
    channels = {}
    for ch in range(n_channels):
        lam_start = 1530 + ch*channel_bandwidth_nm
        lam_end   = lam_start + channel_bandwidth_nm
        channels[f'ch{ch}_nm'] = f'{lam_start:.0f}-{lam_end:.0f} nm'
        channels[f'ch{ch}_pixels'] = n_pixels_x // n_channels

    return {
        'geometry': {
            'n_pixels_x': n_pixels_x, 'n_pixels_y': n_pixels_y,
            'total_pixels': n_pixels_x * n_pixels_y,
            'f_rep_MHz': f_rep_MHz,
            'frame_rate_Mfps': float(f_rep_MHz),
        },
        'optics': {
            'lambda_range_nm': [1530, 1570],
            'Delta_lambda_nm': float(Delta_lambda*1e9),
            'grating_density_lpmm': grating_density,
            'Delta_x_per_nm_um': float(Delta_x_per_nm*1e6),
            'total_spatial_extent_mm': float(total_spatial_extent*1e3),
        },
        'time_stretch': {
            'D_ps_nm_km': D_fiber, 'L_km': L_fiber,
            'tau_per_nm_ps': float(tau_per_nm),
            'tau_total_ps': float(tau_total_ps),
            'dt_per_pixel_ps': float(dt_per_pixel),
            'f_ADC_needed_GHz': float(f_ADC_needed_GHz),
        },
        'performance': {
            'T_frame_ns': float(T_frame_ns),
            'T_rep_ns': float(T_rep_ns),
            'duty_cycle_pct': float(duty_cycle*100),
        },
        'image_2d': image_2d.tolist(),
        'image_serialized': image_serialized.tolist(),
        'hyperspectral_channels': channels,
        'GS_connection': (
            'Each row of STEAM image = one GS measurement (intensity only).\n'
            'GS phase retrieval across rows -> spectral phase image.\n'
            'This is the 2D extension of the 1D GS in this repo.'
        ),
    }


# ============================================================
# Random Forest for ADC Signal Classification
# ============================================================

def random_forest_adc(n_samples=500, n_trees=50):
    """
    Random forest classifier for ADC output: modulation format recognition.

    FEATURES extracted from digitized waveform:
      - Mean amplitude
      - RMS (root mean square)
      - Peak-to-average ratio (PAPR)
      - Spectral centroid (weighted mean frequency)
      - Spectral bandwidth (weighted std of frequency)
      - Skewness (3rd moment, asymmetry)
      - Kurtosis (4th moment, peakedness)
      - Zero-crossing rate
      - Autocorrelation at lag 1

    LABELS (modulation formats -- common in digital receivers):
      0: OOK (On-Off Keying)
      1: BPSK (Binary Phase Shift Keying)
      2: QPSK (Quadrature PSK)
      3: 16-QAM

    RANDOM FOREST (built from scratch in NumPy):
      Each decision tree:
        - Random subset of features (sqrt(n_features))
        - Random subset of training samples (bootstrap)
        - Split on feature threshold minimizing Gini impurity
      Forest prediction: majority vote across all trees.

    CONNECTION TO TIME-STRETCH ADC:
      After digitization, the waveform contains a modulated signal.
      Random forest identifies the modulation format.
      This enables automatic protocol detection in digital receivers (HMI CTA).
      In cognitive radio: identify spectrum occupancy from RF snapshot.
    """
    np.random.seed(42)
    n_features = 9
    n_classes = 4

    def extract_features(x):
        x = np.array(x, dtype=float)
        mean_amp = float(np.mean(x))
        rms = float(np.sqrt(np.mean(x**2)))
        papr = float(np.max(np.abs(x))**2 / (np.mean(x**2)+1e-30))
        X_f = np.fft.fft(x)
        freqs = np.fft.fftfreq(len(x))
        psd = np.abs(X_f[:len(x)//2])**2
        f_pos = freqs[:len(x)//2]
        centroid = float(np.sum(f_pos*psd) / (np.sum(psd)+1e-30))
        bw = float(np.sqrt(np.sum((f_pos-centroid)**2*psd) / (np.sum(psd)+1e-30)))
        mu3 = float(np.mean((x - np.mean(x))**3) / (np.std(x)**3 + 1e-30))
        mu4 = float(np.mean((x - np.mean(x))**4) / (np.std(x)**4 + 1e-30))
        zcr = float(np.mean(np.diff(np.sign(x)) != 0))
        ac1 = float(np.corrcoef(x[:-1], x[1:])[0,1]) if len(x) > 2 else 0
        return np.array([mean_amp, rms, papr, centroid, bw, mu3, mu4, zcr, ac1])

    def generate_signal(label, N=256):
        t = np.linspace(0, 1, N)
        noise = 0.1 * np.random.randn(N)
        if label == 0:   # OOK
            bits = np.random.randint(0, 2, N//32)
            s = np.repeat(bits, 32).astype(float) + noise
        elif label == 1:  # BPSK
            bits = 2*np.random.randint(0, 2, N//32) - 1
            carrier = np.sin(2*np.pi*8*t)
            s = np.repeat(bits, 32) * carrier + noise
        elif label == 2:  # QPSK
            phase = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], N//32)
            carrier = np.sin(2*np.pi*8*t + np.repeat(phase, 32)) + noise
            s = carrier
        else:             # 16-QAM (simplified as amplitude + phase)
            amp = np.random.choice([1, 2, 3, 4], N//32).astype(float)
            ph  = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], N//32)
            s = np.repeat(amp, 32) * np.sin(2*np.pi*8*t + np.repeat(ph, 32)) + noise
        return s[:N]

    # Generate dataset
    X_data = np.zeros((n_samples, n_features))
    y_data = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        label = i % n_classes
        y_data[i] = label
        sig = generate_signal(label)
        X_data[i] = extract_features(sig)

    # Normalize features
    X_mean = X_data.mean(axis=0); X_std = X_data.std(axis=0) + 1e-30
    X_norm = (X_data - X_mean) / X_std

    # Train/test split (80/20)
    n_train = int(0.8*n_samples)
    idx = np.random.permutation(n_samples)
    X_train = X_norm[idx[:n_train]]; y_train = y_data[idx[:n_train]]
    X_test  = X_norm[idx[n_train:]]; y_test  = y_data[idx[n_train:]]

    # Simple decision stump (1-level tree) for each tree in forest
    def train_stump(X, y, feature_idx):
        x_col = X[:, feature_idx]
        best_thresh = float(np.median(x_col))
        left_mask = x_col <= best_thresh
        right_mask = ~left_mask
        left_label  = int(np.bincount(y[left_mask],  minlength=n_classes).argmax()) if left_mask.any()  else 0
        right_label = int(np.bincount(y[right_mask], minlength=n_classes).argmax()) if right_mask.any() else 0
        return (feature_idx, best_thresh, left_label, right_label)

    def predict_stump(stump, X):
        feat_idx, thresh, left_lbl, right_lbl = stump
        return np.where(X[:, feat_idx] <= thresh, left_lbl, right_lbl)

    # Build forest: each tree uses random feature subset + bootstrap
    n_feat_per_tree = max(1, int(math.sqrt(n_features)))
    stumps = []
    for _ in range(n_trees):
        boot_idx = np.random.choice(n_train, n_train, replace=True)
        feat_idx = np.random.choice(n_features, n_feat_per_tree, replace=False)
        X_boot = X_train[boot_idx]; y_boot = y_train[boot_idx]
        # Pick best feature among random subset
        best_stump = None; best_acc = -1
        for fi in feat_idx:
            stump = train_stump(X_boot, y_boot, fi)
            preds = predict_stump(stump, X_boot)
            acc = float(np.mean(preds == y_boot))
            if acc > best_acc:
                best_acc = acc
                best_stump = stump
        stumps.append(best_stump)

    # Predict: majority vote
    def forest_predict(X):
        votes = np.stack([predict_stump(s, X) for s in stumps], axis=1)
        return np.array([np.bincount(row, minlength=n_classes).argmax() for row in votes])

    y_pred_train = forest_predict(X_train)
    y_pred_test  = forest_predict(X_test)
    acc_train = float(np.mean(y_pred_train == y_train))
    acc_test  = float(np.mean(y_pred_test  == y_test))

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_test, y_pred_test):
        cm[true, pred] += 1

    # Feature importance (count splits per feature)
    feat_count = np.zeros(n_features)
    for s in stumps:
        feat_count[s[0]] += 1
    feat_importance = (feat_count / feat_count.sum()).tolist()
    feat_names = ['mean', 'rms', 'PAPR', 'centroid', 'bandwidth', 'skewness', 'kurtosis', 'ZCR', 'autocorr']

    return {
        'dataset': {
            'n_samples': n_samples, 'n_features': n_features,
            'n_classes': n_classes, 'class_names': ['OOK', 'BPSK', 'QPSK', '16-QAM'],
        },
        'forest': {
            'n_trees': n_trees, 'n_feat_per_tree': n_feat_per_tree,
            'acc_train': float(acc_train), 'acc_test': float(acc_test),
        },
        'confusion_matrix': cm.tolist(),
        'feature_importance': {name: imp for name, imp in zip(feat_names, feat_importance)},
        'GS_connection': {
            'features_from_GS': 'After GS phase retrieval, extract spectral features for RF',
            'modulation_recognition': 'Random forest on GS-recovered spectrum -> modulation format',
            'OUSD_HMI': 'Digital receiver intelligence: signal -> ADC -> GS -> RF -> classify',
        },
    }


def demo():
    print("=== PHOTONIC TIME-STRETCH ADC ===\n")

    print("--- Stretch Factor Derivation (1 Gsample/s captures 5 GHz) ---")
    sf = derive_stretch_factor(D1_ps_nm_km=1000, L1_km=1,
                                D2_ps_nm_km=1000, L2_km=9,
                                Delta_lambda_nm=20, f_s_GHz=1.0)
    print(f"  M = {sf['M']:.1f}  ({sf['M_formula']})")
    print(f"  T1 = {sf['derivation']['T1_ps']:.0f} ps  (time aperture before EOM)")
    print(f"  T_total = {sf['derivation']['T_total_ps']:.0f} ps  (after L2 stretch)")
    print(f"  B_RF captured = {sf['B_RF_GHz']:.1f} GHz with {1.0:.0f} Gsample/s ADC")
    print(f"  Equivalent ADC needed without photonics: {sf['B_RF_GHz']*2:.0f} Gsample/s")
    print(f"  Shot noise SNR = {sf['noise']['SNR_shot_dB']:.1f} dB")
    print(f"  H(f) formula: {sf['H_total']['formula']}")

    print("\n--- TS-ADC System Simulation (M=10) ---")
    sys = time_stretch_adc_system(M=10, f_signal_GHz=3.0, SNR_dB=30, f_s_GHz=1.0)
    print(f"  M = {sys['system']['M']}, f_s = {sys['system']['f_s_GHz']} Gsample/s")
    print(f"  Effective bandwidth = {sys['ADC']['B_RF_GHz']} GHz")
    print(f"  ENOB = {sys['ADC']['ENOB']:.1f} bits")
    print(f"  H(f) all-pass: {sys['transfer_function']['all_pass']}")
    print(f"  Poles: {sys['transfer_function']['poles']}")
    print(f"  PID view: {sys['transfer_function']['PID_view']}")

    print("\n--- STEAM Camera (Hyperspectral ADC) ---")
    steam = steam_camera(n_pixels_x=100, n_pixels_y=50, f_rep_MHz=50)
    print(f"  Pixels: {steam['geometry']['total_pixels']}")
    print(f"  Frame rate: {steam['geometry']['frame_rate_Mfps']} Mfps")
    print(f"  tau_total = {steam['time_stretch']['tau_total_ps']:.0f} ps per frame")
    print(f"  Required ADC: {steam['time_stretch']['f_ADC_needed_GHz']:.2f} Gsample/s")
    print(f"  Duty cycle: {steam['performance']['duty_cycle_pct']:.1f}%")
    print(f"  Hyperspectral channels: {list(steam['hyperspectral_channels'].keys())[:4]}")

    print("\n--- Random Forest ADC (modulation recognition) ---")
    rf = random_forest_adc(n_samples=500, n_trees=50)
    print(f"  Classes: {rf['dataset']['class_names']}")
    print(f"  Train acc: {rf['forest']['acc_train']:.3f}")
    print(f"  Test acc:  {rf['forest']['acc_test']:.3f}")
    print("  Confusion matrix:")
    for row in rf['confusion_matrix']:
        print(f"    {row}")
    top_feat = sorted(rf['feature_importance'].items(), key=lambda x: -x[1])[:3]
    print(f"  Top features: {top_feat}")

    print("\n=== TIME-STRETCH ADC COMPLETE ===")


if __name__ == '__main__':
    demo()
