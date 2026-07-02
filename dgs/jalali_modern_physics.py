"""
Jalali Modern Physics -- Photonic Time-Stretch and Its Applications
===================================================================
Reproduces and extends the Jalali group (UCLA) body of work:

1. Dispersive Fourier Transform (DFT / real-time spectroscopy)
   Coppinger, Bhushan & Jalali, 1999 (time-stretch ADC)
   Jalali & Mahjoubfar, Science 2015 (real-time spectroscopy)

2. STEAM Camera (Serial Time-Encoded Amplified Microscopy)
   Goda, Tsia & Jalali, Nature 2009

3. Optical Rogue Waves
   Solli, Ropers, Koonath & Jalali, Nature 2007

4. Compressed Sensing in Time-Stretch
   Asghari & Jalali, Optica 2014

5. Coherent Time-Stretch Detection
   Mahjoubfar et al., Nature Photonics 2017

6. Machine Learning on Photonic Time-Stretch Data
   Tong et al., 2017

KEY PHYSICS THREAD:
  All Jalali techniques exploit H(f) = exp(j*pi*D*f^2) -- the
  dispersive fiber transfer function.  This is the same quadratic
  phase that appears in:
    - Fresnel diffraction (optics)
    - chirp-z transform (DSP)
    - Wigner-Ville distribution (time-frequency)
    - Schrodinger eq. free particle (QM: phi(t) = exp(j*hbar*k^2*t/2m))

  The chain rule d/df[H(f)] = j*2*pi*D*f * H(f) connects to backprop:
  all of these are "just" differentiation through a quadratic exponent.
"""
import math
import numpy as np

c_light  = 2.998e8          # m/s
h_P      = 6.626e-34        # J*s
hbar     = 1.055e-34        # J*s
lambda0  = 1550e-9          # m (telecom C-band)
omega0   = 2*math.pi*c_light/lambda0
q_e      = 1.602e-19        # C


# ============================================================
# 1. Dispersive Fourier Transform (DFT)
# ============================================================

def dispersive_fourier_transform(
    pulse_spectrum=None,        # array-like complex amplitude spectrum, or None
    D_ps_nm_km=1000.0,         # GVD [ps/(nm*km)]
    L_km=5.0,                   # fiber length [km]
    n_pts=1024,                 # number of points
    f_bandwidth_GHz=100.0,     # optical bandwidth [GHz]
    lambda0_nm=1550.0,         # center wavelength [nm]
):
    """
    Dispersive Fourier Transform: maps optical spectrum to time.

    PRINCIPLE (Jalali 1999, 2015):
      After propagating through dispersive fiber with D*L:
        E_out(t) = FFT[E_in(f)] evaluated at f = t/(D*L)
        I_out(t) = |E_in(f=t/(D_eff))|^2

      where D_eff = D*L [ps/nm] is total accumulated dispersion.
      This gives a REAL-TIME spectrum without a spectrometer.

    TIME-TO-FREQUENCY MAPPING:
      t = D_eff * delta_lambda [ps]
      delta_lambda [nm] = t [ps] / D_eff [ps/nm]
      OR equivalently:
      delta_f [GHz] = -c/lambda^2 * delta_lambda [THz*nm]
      t [ps] = (lambda^2/c) * D_eff [ps/nm] * delta_f [GHz]

    GVD CONDITION (far-field / stationary phase):
      |D*L| >> T_pulse^2 / (2*pi) [in SI units]
      For T_pulse=1ps, D*L >> 0.16 ps/nm -> easily satisfied for D*L~100 ps/nm

    Returns dict with time-domain intensity and spectral mapping.
    """
    if D_ps_nm_km <= 0 and L_km <= 0:
        raise ValueError("D_ps_nm_km and L_km must be positive for DFT")

    D_eff = D_ps_nm_km * L_km      # ps/nm  (total accumulated dispersion)
    lambda0_m = lambda0_nm * 1e-9

    # Frequency axis (optical deviation from center)
    df_Hz = f_bandwidth_GHz * 1e9 / n_pts
    f_arr = np.linspace(-f_bandwidth_GHz/2, f_bandwidth_GHz/2, n_pts) * 1e9  # Hz

    # Default: Gaussian spectrum
    if pulse_spectrum is None:
        sigma_f = f_bandwidth_GHz/6 * 1e9   # 1/6 of bandwidth = ~3-sigma coverage
        E_in = np.exp(-f_arr**2/(2*sigma_f**2))
    else:
        E_in = np.asarray(pulse_spectrum, dtype=complex)
        if len(E_in) != n_pts:
            E_in = np.interp(
                np.linspace(0, 1, n_pts),
                np.linspace(0, 1, len(E_in)),
                np.abs(E_in)
            ).astype(complex)

    # H(f) = exp(j*pi*beta2*L*(2*pi*f)^2)
    # beta2*L [s^2] = -(lambda^2/(2*pi*c)) * D_eff [ps/nm * 1e-12/1e-9]
    D_eff_SI = D_eff * 1e-12 / 1e-9   # s/m -> but D_eff is ps/nm = 1e-12/1e-9 s/m = 1e-3 s/m
    beta2L = -(lambda0_m**2 / (2*math.pi*c_light)) * D_eff_SI   # s^2
    phi = math.pi * beta2L * (2*math.pi*f_arr)**2
    H_disp = np.exp(1j*phi)

    E_out = E_in * H_disp

    # Time-domain: IFFT -> time waveform
    E_time = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(E_out)))
    I_time = np.abs(E_time)**2

    # Time axis
    dt_s = 1.0 / (f_bandwidth_GHz * 1e9)
    t_arr_ps = np.linspace(-n_pts/2, n_pts/2, n_pts) * dt_s * 1e12

    # Time-to-wavelength mapping: t [ps] = D_eff [ps/nm] * delta_lambda [nm]
    delta_lambda_nm = t_arr_ps / D_eff if abs(D_eff) > 0 else t_arr_ps*0

    # DFT condition check (far-field: |beta2*L| >> T_pulse^2)
    T_pulse_ps = 1.0 / (f_bandwidth_GHz)  # ~1/(BW) in ps
    DFT_condition_OK = abs(beta2L) > (T_pulse_ps*1e-12)**2 / (2*math.pi)

    return {
        'principle': 'I_out(t) = |E_in(f=t/D_eff)|^2  [real-time spectroscopy]',
        'fiber': {
            'D_ps_nm_km': float(D_ps_nm_km),
            'L_km': float(L_km),
            'D_eff_ps_nm': float(D_eff),
        },
        'time_to_wavelength': {
            'formula': 'delta_lambda [nm] = t [ps] / D_eff [ps/nm]',
            'D_eff_ps_nm': float(D_eff),
            'time_window_ps': float(t_arr_ps[-1] - t_arr_ps[0]),
            'wavelength_window_nm': float(delta_lambda_nm[-1] - delta_lambda_nm[0]),
        },
        'H_f': {
            'formula': 'H(f) = exp(j*pi*beta2*L*(2*pi*f)^2)',
            'all_pass': True,
            'beta2L_s2': float(beta2L),
            'max_phase_rad': float(np.max(np.abs(phi))),
        },
        'DFT_condition': {
            'satisfied': bool(DFT_condition_OK),
            'beta2L_s2': float(beta2L),
            'T_pulse_s': float(T_pulse_ps*1e-12),
            'note': '|beta2*L| >> T_pulse^2 / (2*pi) required for time-to-frequency mapping',
        },
        'output': {
            'I_time': I_time.tolist(),
            't_ps': t_arr_ps.tolist(),
            'delta_lambda_nm': delta_lambda_nm.tolist(),
            'n_pts': n_pts,
        },
        'jalali_reference': 'Jalali & Mahjoubfar, Science 348, 1440 (2015)',
    }


# ============================================================
# 2. STEAM Camera
# ============================================================

def steam_camera(
    n_rows=100,           # spatial rows (image height)
    n_cols=200,           # spatial columns (image width)
    D_ps_nm_km=1000.0,   # fiber GVD
    L_km=5.0,             # fiber length
    f_rep_MHz=10.0,       # laser repetition rate [MHz]
    lambda0_nm=1550.0,
    BW_nm=20.0,           # optical bandwidth [nm]
    lpmm=600.0,           # grating lines/mm
):
    """
    STEAM Camera: Serial Time-Encoded Amplified Microscopy
    Goda, Tsia & Jalali, Nature 2009

    CONCEPT:
      1. Femtosecond laser pulse -> diffraction grating -> 1D spatial fan
         (each wavelength illuminates different x-position)
      2. Pulse reflects off 2D object -> collected wavelength-encodes ROWS
         by rotating the grating fan (one frame per pulse)
      3. Dispersive fiber -> DFT maps lambda -> t
      4. Result: 2D image encoded as 1D time waveform -> single photodetector

    SERIAL SERIALIZATION:
      For n_rows rows and f_rep laser:
        T_frame = 1/f_rep [s] per pulse
        T_row = T_frame / n_rows [s per row]
        lambda(row) = lambda0 + BW*(row/n_rows - 0.5) [nm]
        t(row) = D_eff * (lambda(row) - lambda0) [ps]

    GRATING EQUATIONS:
      Littrow angle: sin(theta_L) = lambda / (2*d) where d = 1/lpmm [mm]
      Angular dispersion: d(theta)/d(lambda) = 1/(d*cos(theta_L)) [rad/nm]
      Spatial dispersion: dx/dlambda = f_lens * d(theta)/d(lambda) [mm/nm]
    """
    D_eff = D_ps_nm_km * L_km      # ps/nm

    # Grating
    d_mm = 1.0/lpmm               # grating period [mm]
    d_nm = d_mm * 1e6             # [nm]
    lambda0_nm_f = float(lambda0_nm)

    sin_arg = float(np.clip(lambda0_nm_f / (2*d_nm), -0.9999, 0.9999))
    theta_L_rad = math.asin(sin_arg)   # Littrow angle
    theta_L_deg = math.degrees(theta_L_rad)

    cos_theta = math.cos(theta_L_rad)
    dtheta_dlambda_rad_nm = 1.0 / (d_nm * cos_theta)  # rad/nm

    f_lens_mm = 50.0   # typical collimating lens focal length [mm]
    dx_dlambda_mm_nm = f_lens_mm * dtheta_dlambda_rad_nm   # mm/nm

    # Wavelength allocation to rows
    row_idx = np.arange(n_rows)
    lambda_row_nm = lambda0_nm_f + BW_nm * (row_idx/n_rows - 0.5)

    # Time encoding: t(lambda) = D_eff * delta_lambda
    delta_lambda_nm = lambda_row_nm - lambda0_nm_f
    t_row_ps = D_eff * delta_lambda_nm   # [ps]

    # Frame rate
    T_frame_us = 1.0/f_rep_MHz          # [microseconds]
    T_row_ps = T_frame_us * 1e6 / n_rows   # [ps per row]

    # Generate synthetic image (Gaussian blob)
    rng = np.random.default_rng(42)
    Y, X = np.mgrid[:n_rows, :n_cols]
    image = (np.exp(-((X - n_cols//2)**2/(n_cols/5)**2 +
                      (Y - n_rows//2)**2/(n_rows/5)**2)) +
             0.05*rng.random((n_rows, n_cols)))

    # STEAM serialization: each row -> time slot
    serialized = np.zeros(n_rows * n_cols)
    for r in range(n_rows):
        start = r * n_cols
        serialized[start:start+n_cols] = image[r, :]

    # STEAM reconstruction: deserialize
    recon = serialized.reshape(n_rows, n_cols)
    recon_error = float(np.max(np.abs(recon - image)))

    # Frame rate = f_rep for 2D; row rate = n_rows * f_rep
    pixel_rate_MHz = n_rows * n_cols * f_rep_MHz

    return {
        'concept': 'Serial Time-Encoded Amplified Microscopy (STEAM)',
        'paper': 'Goda, Tsia & Jalali, Nature 458, 1145 (2009)',
        'grating': {
            'lpmm': float(lpmm),
            'd_nm': float(d_nm),
            'theta_Littrow_deg': float(theta_L_deg),
            'angular_dispersion_rad_per_nm': float(dtheta_dlambda_rad_nm),
            'spatial_dispersion_mm_per_nm': float(dx_dlambda_mm_nm),
            'validity': 'd > lambda required for Littrow',
            'valid': bool(d_nm > lambda0_nm_f),
        },
        'time_encoding': {
            'D_eff_ps_nm': float(D_eff),
            'BW_nm': float(BW_nm),
            'total_time_window_ps': float(t_row_ps[-1] - t_row_ps[0]),
            't_row_min_ps': float(np.min(t_row_ps)),
            't_row_max_ps': float(np.max(t_row_ps)),
            'T_row_ps': float(T_row_ps),
        },
        'system': {
            'n_rows': n_rows,
            'n_cols': n_cols,
            'frame_rate_MHz': float(f_rep_MHz),
            'pixel_rate_MHz': float(pixel_rate_MHz),
            'row_rate_MHz': float(n_rows * f_rep_MHz),
        },
        'reconstruction': {
            'serialized_length': n_rows * n_cols,
            'recon_shape': [n_rows, n_cols],
            'max_error': float(recon_error),
            'lossless': recon_error < 1e-12,
        },
        'image_2d': image.tolist(),
        'serialized_1d': serialized.tolist(),
    }


# ============================================================
# 3. Optical Rogue Waves
# ============================================================

def optical_rogue_waves(
    n_pts=2048,
    t_span_ps=100.0,
    gamma=1.3,              # nonlinearity [1/(W*km)]
    beta2=-20e-27,          # GVD [s^2/m] (anomalous)
    P0=1.0,                 # peak power [W]
    n_ensemble=20,          # number of random realizations
    rng_seed=7,
):
    """
    Optical Rogue Waves: Solli, Ropers, Koonath & Jalali, Nature 2007

    CONTEXT:
      Rogue waves = rare, extreme events. In optics: rare supercontinuum
      spikes that dwarf the average by >8x (analogous to oceanic rogue waves
      defined as H > 2*H_s, significant wave height).

    PHYSICS:
      Governed by Nonlinear Schrodinger Equation (NLSE):
        j*dA/dz = -(beta2/2)*d^2A/dt^2 + gamma*|A|^2*A

      Modulation Instability (MI) is the seed mechanism:
        MI gain: g(Omega) = |beta2|*|Omega|*sqrt(Omega_c^2 - Omega^2)
        where Omega_c^2 = 2*gamma*P0/|beta2|
        Peak MI gain at Omega = Omega_c/sqrt(2), g_max = gamma*P0

      Peregrine soliton (rational soliton): exact rogue wave solution
        A_P(z,t) = P0^0.5 * [1 - 4*(1+2j*gamma*P0*z) /
                              (1 + 4*gamma^2*P0^2*z^2 + 2*(t/t0)^2)]
                  * exp(j*gamma*P0*z)
        where t0 = sqrt(2/gamma/P0/|beta2|) * ... (characteristic time)
        Peak: |A_P|_max = 3*P0^0.5  ->  I_peak = 9*P0  (9x amplification)

    JALALI CONNECTION:
      TD-GS phase retrieval extracts the phase of rare events captured
      by the time-stretch ADC (sampling the optical waveform in real time).
      This enables characterization of rogue waves at THz bandwidth.

    Returns ensemble statistics and rogue wave probability analysis.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive [1/(W*km)]")
    if n_ensemble < 2:
        raise ValueError("n_ensemble must be >= 2")

    dt_ps = t_span_ps / n_pts
    t_arr = np.linspace(-t_span_ps/2, t_span_ps/2, n_pts)

    # Modulation instability gain spectrum
    # Omega_c = sqrt(2*gamma*P0/|beta2|) -- convert units
    # gamma [1/(W*km)] = gamma*1e-3 [1/(W*m)]
    # beta2 [s^2/m]
    gamma_SI = gamma * 1e-3    # 1/(W*m)
    Omega_c = math.sqrt(2*gamma_SI*P0/abs(beta2))   # rad/s

    n_omega = 512
    Omega_arr = np.linspace(0, 2*Omega_c, n_omega)
    arg = np.clip(Omega_c**2 - Omega_arr**2, 0, None)
    g_MI = abs(beta2)*Omega_arr*np.sqrt(arg)   # MI gain [1/m]
    g_max = gamma_SI * P0   # [1/m]
    Omega_peak = Omega_c / math.sqrt(2)   # [rad/s]
    f_MI_peak_GHz = Omega_peak / (2*math.pi) * 1e-9

    # Peregrine soliton: exact rational solution to NLSE
    # A_P(z,t) = sqrt(P0) * [1 - 4(1+2j*g*P0*z)/(1+4g^2P0^2z^2+2(t/t0)^2)] * exp(jgP0z)
    # Peak at t=0, z=0: A_P = sqrt(P0)*(1-4) = -3*sqrt(P0)  =>  I_peak = 9*P0
    L_NL = 1.0 / (gamma_SI * P0)   # nonlinear length [m]
    # t0: normalize time by dispersion/nonlinearity balance
    # t0 = 1/sqrt(gamma_SI*P0) * sqrt(|beta2|/(gamma_SI*P0)) [s] -> in ps
    t0 = float(math.sqrt(abs(beta2) / (gamma_SI*P0))) * 1e12   # [ps]
    t0 = max(t0, 0.01)
    # Evaluate the spatial profile at t=0 (vary z) to find emergence
    z_arr = np.linspace(0, 3*L_NL, n_pts)
    numer_z = 4*(1 + 2j*gamma_SI*P0*z_arr)
    denom_z = 1 + 4*(gamma_SI*P0*z_arr)**2   # at t=0
    A_P_t0 = math.sqrt(P0) * (1 - numer_z/denom_z) * np.exp(1j*gamma_SI*P0*z_arr)
    # Temporal profile at z=0 (peak slice)
    numer_t0 = 4*(1 + 0j)   # z=0
    denom_t_arr = 1 + 0 + 2*(t_arr/t0)**2
    A_P = math.sqrt(P0) * (1 - numer_t0/denom_t_arr)   # no phase shift at z=0
    I_Peregrine = np.abs(A_P)**2
    I_peak_P = float(np.max(I_Peregrine))   # should be ~9*P0
    amplification = I_peak_P / P0

    # Ensemble: random noise seeds -> intensity statistics
    rng = np.random.default_rng(rng_seed)
    peak_intensities = []
    for _ in range(n_ensemble):
        noise_amp = 0.05*math.sqrt(P0)
        A_noise = (math.sqrt(P0) +
                   noise_amp*(rng.random(n_pts) - 0.5 +
                              1j*(rng.random(n_pts) - 0.5)))
        # Simple MI amplification (first-order): amplify at Omega_peak
        omega_arr = np.fft.fftfreq(n_pts, d=dt_ps*1e-12) * 2*math.pi
        A_k = np.fft.fft(A_noise)
        # Amplify near Omega_peak
        mask = np.abs(np.abs(omega_arr) - Omega_peak) < (Omega_c * 0.2)
        gain_factor = 1.0 + 5*rng.random()*mask.astype(float)
        A_k_amp = A_k * gain_factor
        A_out = np.fft.ifft(A_k_amp)
        peak_intensities.append(float(np.max(np.abs(A_out)**2)))

    peak_arr = np.array(peak_intensities)
    I_mean = float(np.mean(peak_arr))
    I_max  = float(np.max(peak_arr))
    # Rogue wave criterion: I_peak > 2 * mean (analogous to oceanic H > 2*H_s)
    rogue_threshold = 2.0 * I_mean
    n_rogue = int(np.sum(peak_arr > rogue_threshold))
    rogue_probability = n_rogue / n_ensemble

    # L-shaped tail: fit a simple exponential to the upper tail
    tail_mask = peak_arr > I_mean
    if np.sum(tail_mask) >= 2:
        x_tail = peak_arr[tail_mask]
        # Exponential fit: log(rank/N) ~ -rate * x
        sorted_x = np.sort(x_tail)[::-1]
        log_rank = np.log(np.arange(1, len(sorted_x)+1) / n_ensemble)
        tail_rate = float(-np.polyfit(sorted_x, log_rank, 1)[0])
    else:
        tail_rate = 0.0

    return {
        'paper': 'Solli, Ropers, Koonath & Jalali, Nature 450, 1054 (2007)',
        'physics': {
            'gamma_SI': float(gamma_SI),
            'beta2': float(beta2),
            'P0_W': float(P0),
            'L_NL_m': float(L_NL),
        },
        'modulation_instability': {
            'Omega_c_rad_s': float(Omega_c),
            'f_MI_peak_GHz': float(f_MI_peak_GHz),
            'g_max_per_m': float(g_max),
            'g_MI': g_MI.tolist(),
            'Omega_arr': Omega_arr.tolist(),
        },
        'Peregrine_soliton': {
            'formula': 'A_P = sqrt(P0) * [1 - 4(1+2j*g*P0*z)/(1+...)] * exp(j*g*P0*z)',
            'I_peak_W': float(I_peak_P),
            'amplification_factor': float(amplification),
            'expected_9x': True,   # theory: I_peak = 9*P0
            'I_Peregrine': I_Peregrine.tolist(),
            't_ps': t_arr.tolist(),
        },
        'ensemble': {
            'n_ensemble': n_ensemble,
            'I_mean_W': float(I_mean),
            'I_max_W': float(I_max),
            'rogue_threshold_W': float(rogue_threshold),
            'n_rogue_events': n_rogue,
            'rogue_probability': float(rogue_probability),
            'tail_decay_rate': float(tail_rate),
            'peak_intensities': peak_arr.tolist(),
            'L_shaped_tail': tail_rate > 0,
        },
        'jalali_td_gs': (
            'TD-GS extracts complex A(t) from |A(t)|^2 (measured) + '
            'H(f)=exp(j*pi*D*f^2) (known dispersion) -> phase retrieval'
        ),
    }


# ============================================================
# 4. Compressed Sensing in Time-Stretch
# ============================================================

def compressed_sensing_ts(
    N=256,                  # signal length (Nyquist)
    K=16,                   # sparsity (# nonzero frequencies)
    M=None,                 # measurements (None -> auto: 4*K*log(N/K))
    stretch_factor=5.0,     # photonic time-stretch factor M
    rng_seed=99,
):
    """
    Compressed Sensing in Time-Stretch: Asghari & Jalali, Optica 2014.

    CONCEPT:
      Photonic time-stretch provides the analog sampling at M*f_Nyquist.
      Compressive sensing reconstructs K-sparse signals from M << N measurements:
        y = Phi * x
        x_hat = argmin ||x||_1  s.t.  ||y - Phi*x||_2 <= epsilon

      Recovery condition: Restricted Isometry Property (RIP)
        M >= C * K * log(N/K)  [number of measurements needed]

    PHOTONIC ADVANTAGE:
      The photonic time-stretch maps bandwidth by factor M_stretch:
        B_captured = M_stretch * B_ADC / 2
      Then CS reduces the number of ADC samples needed by factor N/M.
      Combined advantage: can capture M_stretch * N/M_cs * B_ADC bandwidth.

    L1 MINIMIZATION (basis pursuit):
      Implemented here as iterative soft thresholding (ISTA):
        x_hat[n+1] = S_lambda(x_hat[n] + Phi^T(y - Phi*x_hat[n]))
        where S_lambda is soft-thresholding at level lambda.

    MUTUAL COHERENCE:
      mu = max_{i!=j} |phi_i^T phi_j| / (||phi_i|| * ||phi_j||)
      Requirement: K < (1/mu + 1)/2  (sufficient for exact recovery)

    Returns recovery quality metrics and CS analysis.
    """
    if K <= 0 or K > N//2:
        raise ValueError(f"K must be in [1, {N//2}]")
    if stretch_factor < 1.0:
        raise ValueError("stretch_factor must be >= 1.0")

    rng = np.random.default_rng(rng_seed)

    # Auto-set M from RIP bound
    if M is None:
        M = int(4 * K * math.log(N/K))
    M = min(M, N)   # cannot exceed N

    # Sparse signal in frequency domain
    support = rng.choice(N, K, replace=False)
    x_sparse = np.zeros(N)
    x_sparse[support] = rng.random(K) + 0.5   # positive coefficients

    # Measurement matrix (random Gaussian, RIP holds with high probability)
    Phi = rng.standard_normal((M, N)) / math.sqrt(M)

    # Measurements
    y = Phi @ x_sparse

    # ISTA (Iterative Soft Thresholding Algorithm)
    step = 1.0 / (np.linalg.norm(Phi, ord=2)**2 + 1e-10)
    lam = 0.01
    x_hat = np.zeros(N)
    n_iter_ista = 200
    residuals = []
    for i in range(n_iter_ista):
        grad = Phi.T @ (Phi @ x_hat - y)
        x_hat = x_hat - step*grad
        # Soft thresholding
        x_hat = np.sign(x_hat) * np.maximum(np.abs(x_hat) - step*lam, 0)
        residuals.append(float(np.linalg.norm(Phi @ x_hat - y)))

    # Recovery quality
    nmse = float(np.linalg.norm(x_hat - x_sparse)**2 /
                 (np.linalg.norm(x_sparse)**2 + 1e-30))
    recovery_ok = nmse < 0.1   # within 10% NMSE

    # Mutual coherence (random Gaussian phi -> low coherence)
    norms = np.linalg.norm(Phi, axis=0) + 1e-30
    Phi_norm = Phi / norms
    gram = Phi_norm.T @ Phi_norm
    np.fill_diagonal(gram, 0)
    mu = float(np.max(np.abs(gram)))

    # RIP condition check
    rip_K_max = (1/mu + 1)/2 if mu > 0 else float('inf')
    rip_satisfied = K < rip_K_max

    # Compression ratio
    compression_ratio = float(N) / M
    # Bandwidth captured with photonic TS + CS
    B_ADC_GHz = 1.0   # 1 Gsample/s ADC
    B_captured_GHz = stretch_factor * B_ADC_GHz / 2 * compression_ratio

    return {
        'paper': 'Asghari & Jalali, Optica 1, 23 (2014)',
        'signal': {
            'N_Nyquist': N,
            'K_sparse': K,
            'M_measurements': M,
            'compression_ratio': float(compression_ratio),
        },
        'RIP': {
            'satisfied': bool(rip_satisfied),
            'mu_coherence': float(mu),
            'K_max_for_recovery': float(rip_K_max),
            'RIP_bound': f'M >= 4*K*log(N/K) = {int(4*K*math.log(N/K))}',
        },
        'ISTA': {
            'n_iterations': n_iter_ista,
            'final_residual': float(residuals[-1]),
            'residuals': residuals,
            'NMSE': float(nmse),
            'recovery_ok': bool(recovery_ok),
        },
        'photonic_advantage': {
            'stretch_factor': float(stretch_factor),
            'B_ADC_GHz': float(B_ADC_GHz),
            'B_captured_GHz': float(B_captured_GHz),
            'note': f'CS*TS captures {B_captured_GHz:.1f}x more bandwidth than ADC alone',
        },
        'x_original': x_sparse.tolist(),
        'x_recovered': x_hat.tolist(),
        'support': support.tolist(),
    }


# ============================================================
# 5. Coherent Time-Stretch Detection
# ============================================================

def coherent_time_stretch(
    D_ps_nm_km=1000.0,
    L_km=5.0,
    f_signal_GHz=10.0,      # signal frequency [GHz]
    SNR_in_dB=20.0,         # input optical SNR [dB]
    n_pts=1024,
    modulation_format='QPSK',
):
    """
    Coherent Time-Stretch Detection: Mahjoubfar et al., Nature Photonics 2017

    CONCEPT:
      Standard time-stretch uses direct detection: I(t) = |E(t)|^2
      This DISCARDS the phase -> half the Shannon capacity.

      Coherent detection adds a local oscillator (LO):
        I_I(t) = Re[E_sig(t) * E_LO*(t)]
        I_Q(t) = Im[E_sig(t) * E_LO*(t)]
      This recovers both amplitude AND phase -> full complex field.

    PHASE DIVERSITY RECEIVER:
      90-degree hybrid splits signal and LO into 4 outputs:
        I+(t) = |E_sig + E_LO|^2
        I-(t) = |E_sig - E_LO|^2
        Q+(t) = |E_sig + j*E_LO|^2
        Q-(t) = |E_sig - j*E_LO|^2
      Then: I(t) = (I+ - I-)/2 = Re[E_sig * E_LO*]
            Q(t) = (Q+ - Q-)/2 = Im[E_sig * E_LO*]

    CAPACITY:
      Shannon: C = B * log2(1 + SNR) [bits/s]
      Direct detection: uses |E|^2 -> real-valued -> SNR halved
      Coherent: recovers E -> complex -> full capacity

    SUPPORTED MODULATION: OOK, BPSK, QPSK, 16-QAM
    """
    valid_mods = {'OOK', 'BPSK', 'QPSK', '16-QAM'}
    if modulation_format not in valid_mods:
        raise ValueError(f"modulation_format must be one of {valid_mods}")

    D_eff = D_ps_nm_km * L_km
    rng = np.random.default_rng(42)

    t_arr = np.linspace(0, n_pts-1, n_pts) / (f_signal_GHz * 1e9) * 1e12  # ps

    # Generate complex signal
    n_symbols = n_pts // 16
    if modulation_format == 'OOK':
        symbols = rng.integers(0, 2, n_symbols).astype(complex)
    elif modulation_format == 'BPSK':
        bits = rng.integers(0, 2, n_symbols)
        symbols = 2*bits - 1 + 0j
    elif modulation_format == 'QPSK':
        bits = rng.integers(0, 4, n_symbols)
        symbols = np.exp(1j*math.pi/4 * (2*bits + 1))
    else:  # 16-QAM
        bits_I = rng.integers(0, 4, n_symbols)*2 - 3
        bits_Q = rng.integers(0, 4, n_symbols)*2 - 3
        symbols = (bits_I + 1j*bits_Q) / math.sqrt(10)

    # Upsample to n_pts
    E_sig = np.repeat(symbols, 16)[:n_pts]

    # Add noise
    SNR_lin = 10**(SNR_in_dB/10)
    noise_amp = 1.0 / math.sqrt(2*SNR_lin)
    noise = noise_amp * (rng.standard_normal(n_pts) + 1j*rng.standard_normal(n_pts))
    E_noisy = E_sig + noise

    # Direct detection: I = |E|^2
    I_direct = np.abs(E_noisy)**2

    # Coherent: LO = 1 (normalized), 90-deg hybrid
    E_LO = np.ones(n_pts, dtype=complex)
    I_coh = np.real(E_noisy * np.conj(E_LO))
    Q_coh = np.imag(E_noisy * np.conj(E_LO))

    # SNR comparison
    signal_power = float(np.mean(np.abs(E_sig)**2))
    noise_direct = float(np.var(I_direct - np.abs(E_sig)**2))
    noise_coh_I  = float(np.var(I_coh - np.real(E_sig)))
    SNR_direct_dB = 10*math.log10(max(signal_power**2/(noise_direct+1e-30), 1e-30))
    SNR_coh_dB    = 10*math.log10(max(signal_power/(noise_coh_I+1e-30), 1e-30))

    # Shannon capacity
    B_GHz = f_signal_GHz
    C_direct_Gbps = B_GHz * math.log2(max(1 + 10**(SNR_direct_dB/10), 1))
    C_coh_Gbps    = 2*B_GHz * math.log2(max(1 + 10**(SNR_coh_dB/10), 1))  # *2 for IQ

    # Time-stretch enhancement
    M_stretch = (D_ps_nm_km * L_km) / max(D_ps_nm_km, 1e-10)   # simplified
    B_captured = M_stretch * B_GHz / 2

    return {
        'paper': 'Mahjoubfar et al., Nature Photonics 11, 48 (2017)',
        'modulation': modulation_format,
        'fiber': {
            'D_ps_nm_km': float(D_ps_nm_km),
            'L_km': float(L_km),
            'D_eff_ps_nm': float(D_eff),
        },
        'SNR': {
            'SNR_in_dB': float(SNR_in_dB),
            'SNR_direct_dB': float(SNR_direct_dB),
            'SNR_coherent_dB': float(SNR_coh_dB),
            'improvement_dB': float(SNR_coh_dB - SNR_direct_dB),
        },
        'capacity': {
            'B_GHz': float(B_GHz),
            'C_direct_Gbps': float(C_direct_Gbps),
            'C_coherent_Gbps': float(C_coh_Gbps),
            'ratio': float(C_coh_Gbps / max(C_direct_Gbps, 1e-10)),
        },
        'n_symbols': n_symbols,
        'I_direct': I_direct.tolist(),
        'I_coherent': I_coh.tolist(),
        'Q_coherent': Q_coh.tolist(),
        't_ps': t_arr.tolist(),
    }


# ============================================================
# 6. ML on Photonic Time-Stretch Data
# ============================================================

def ml_on_pts_data(
    n_train=400,
    n_test=100,
    stretch_factor=5.0,
    n_pts_per_sample=128,
    n_classes=4,
    rng_seed=17,
):
    """
    Machine Learning on Photonic Time-Stretch Data (Jalali group, 2017).

    CONCEPT:
      PTS-ADC captures waveform snapshots at >> Nyquist rate.
      ML classifies waveform type / detects anomalies.

    FEATURE EXTRACTION from I(t) = |E(t)|^2:
      1. Mean, RMS, peak
      2. PAPR = peak^2 / mean^2 (crest factor squared)
      3. Spectral centroid, bandwidth
      4. Skewness, kurtosis (for rogue wave detection)
      5. Zero-crossing rate, autocorrelation peak

    CLASSIFIER: Random forest (bootstrapped stumps, majority vote)
      Same as in time_stretch_adc.py but adapted for PTS anomaly detection.

    CLASSES:
      0: Normal pulse (Gaussian)
      1: Rogue wave spike (Peregrine-like)
      2: Double pulse (collision)
      3: Dark soliton (intensity dip)

    Returns classification accuracy and feature importances.
    """
    if n_classes not in (2, 4):
        raise ValueError("n_classes must be 2 or 4")
    if n_train < 10:
        raise ValueError("n_train must be >= 10")

    rng = np.random.default_rng(rng_seed)
    t = np.linspace(-1, 1, n_pts_per_sample)

    def gen_sample(cls, idx):
        noise = 0.05*rng.standard_normal(n_pts_per_sample)
        if cls == 0:   # Normal Gaussian pulse
            A = rng.uniform(0.5, 1.5)
            sigma = rng.uniform(0.1, 0.3)
            return A * np.exp(-t**2/(2*sigma**2)) + noise
        elif cls == 1:   # Rogue wave (Peregrine-like: 3x amplitude)
            A = rng.uniform(0.5, 1.5)
            return 3*A * np.exp(-t**2/0.02) + noise   # narrow, tall
        elif cls == 2:   # Double pulse
            t1 = rng.uniform(-0.6, -0.2)
            t2 = rng.uniform(0.2, 0.6)
            return (np.exp(-(t-t1)**2/0.05) +
                    np.exp(-(t-t2)**2/0.05) + noise)
        else:    # Dark soliton (intensity dip)
            return np.ones(n_pts_per_sample) * rng.uniform(0.5,1.0) * (
                1 - 0.8*np.exp(-t**2/0.05)) + noise

    def extract_features(x):
        x = np.asarray(x)
        x_abs = np.abs(x)
        mean = float(np.mean(x_abs))
        rms  = float(np.sqrt(np.mean(x_abs**2)))
        peak = float(np.max(x_abs))
        papr = (peak**2 / (mean**2 + 1e-30))
        # Spectral
        Xf = np.abs(np.fft.rfft(x))
        freqs = np.arange(len(Xf), dtype=float)
        centroid = float(np.sum(freqs*Xf)/(np.sum(Xf)+1e-30))
        bw = float(np.sqrt(np.sum(Xf*(freqs-centroid)**2)/(np.sum(Xf)+1e-30)))
        # Statistical moments
        x_centered = x_abs - mean
        std = rms + 1e-30
        skew = float(np.mean(x_centered**3) / std**3)
        kurt = float(np.mean(x_centered**4) / std**4)
        # Temporal
        zcr = float(np.sum(np.abs(np.diff(np.sign(x)))) / (2*len(x)))
        ac = float(np.correlate(x-mean, x-mean, mode='full')[len(x)-1] /
                   (len(x) * (std**2 + 1e-30)))
        return [mean, rms, peak, papr, centroid, bw, skew, kurt, zcr, ac]

    # Generate dataset
    n_total = n_train + n_test
    X_all = []
    y_all = []
    for i in range(n_total):
        cls = i % n_classes
        X_all.append(extract_features(gen_sample(cls, i)))
        y_all.append(cls)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test,  y_test  = X_all[n_train:], y_all[n_train:]

    # Normalize
    mu_f = X_train.mean(axis=0); std_f = X_train.std(axis=0) + 1e-30
    X_tr_n = (X_train - mu_f) / std_f
    X_te_n = (X_test  - mu_f) / std_f

    n_features = X_all.shape[1]

    # Random forest (30 bootstrapped decision stumps)
    n_trees = 30
    trees = []
    for _ in range(n_trees):
        idx = rng.integers(0, n_train, n_train)
        Xb = X_tr_n[idx]; yb = y_train[idx]
        # Random feature
        fi = rng.integers(0, n_features)
        vals = Xb[:, fi]
        # Find best threshold among random thresholds
        best_acc = -1; best_thresh = 0.0; best_cls_L = 0; best_cls_R = 0
        for thresh in rng.uniform(float(vals.min()), float(vals.max()), 10):
            L = yb[vals <= thresh]; R = yb[vals > thresh]
            if len(L) == 0 or len(R) == 0:
                continue
            cls_L = int(np.bincount(L, minlength=n_classes).argmax())
            cls_R = int(np.bincount(R, minlength=n_classes).argmax())
            acc = (np.sum(yb[vals <= thresh] == cls_L) +
                   np.sum(yb[vals > thresh]  == cls_R)) / max(len(yb), 1)
            if acc > best_acc:
                best_acc = acc; best_thresh = thresh
                best_cls_L = cls_L; best_cls_R = cls_R
        trees.append((fi, best_thresh, best_cls_L, best_cls_R))

    def predict(X_n):
        preds = np.zeros((len(X_n), n_classes))
        for fi, thresh, cL, cR in trees:
            for i, x in enumerate(X_n):
                preds[i, cL if x[fi] <= thresh else cR] += 1
        return preds.argmax(axis=1)

    y_pred = predict(X_te_n)
    accuracy = float(np.mean(y_pred == y_test))

    # Feature importance (variance of each feature across classes)
    importance = []
    for fi in range(n_features):
        class_means = [float(np.mean(X_tr_n[y_train==c, fi]))
                       for c in range(n_classes)]
        importance.append(float(np.var(class_means)))
    importance = np.array(importance)
    importance /= importance.sum() + 1e-30

    feature_names = ['mean','rms','peak','PAPR','centroid','BW',
                     'skewness','kurtosis','ZCR','autocorr']

    return {
        'paper': 'Tong et al., arXiv:1708.02854 (2017)',
        'dataset': {
            'n_train': n_train,
            'n_test': n_test,
            'n_classes': n_classes,
            'n_features': n_features,
            'classes': ['Gaussian pulse','Rogue wave','Double pulse','Dark soliton'][:n_classes],
        },
        'accuracy': float(accuracy),
        'feature_importances': dict(zip(feature_names, importance.tolist())),
        'top_feature': feature_names[int(np.argmax(importance))],
        'rogue_detection': {
            'method': 'PAPR > threshold (kurtosis elevated for rare spikes)',
            'key_feature': 'PAPR (peak-to-average power ratio)',
        },
        'photonic_advantage': {
            'stretch_factor': float(stretch_factor),
            'frames_per_second': 'MHz-scale (laser rep rate)',
            'note': 'PTS enables real-time ML at THz bandwidth',
        },
        'confusion_hint': 'See confusion matrix in notebook for class-wise breakdown',
    }


# ============================================================
# System-level summary
# ============================================================

def jalali_system_summary():
    """
    High-level summary of Jalali group contributions and their unifying theme.

    UNIFYING THEME: H(f) = exp(j*pi*D*f^2)
      Every Jalali technique manipulates this one transfer function:
        - DFT:       propagate through H(f) -> I(t) ~ |E_in(f=t/D)|^2
        - TS-ADC:    two stages of H(f) -> M-fold time stretch
        - STEAM:     H(f) in spatial domain -> 2D to 1D serialization
        - Rogue waves: H(f) + Kerr -> MI -> Peregrine soliton
        - CS-TS:     random sampling after H(f) -> L1 recovery
        - Coherent:  LO beats after H(f) -> complex field recovery
        - ML/PTS:    features extracted from |E_out(t)|^2 -> classifier
        - TD-GS:     phase from |H(f)|=1 + diversity -> GS iterations

    TIMELINE:
      1999: Photonic time-stretch ADC (Coppinger, Bhushan, Jalali - IEEE)
      2007: Optical rogue waves (Solli, Ropers, Koonath, Jalali - Nature)
      2009: STEAM camera (Goda, Tsia, Jalali - Nature)
      2014: Compressed sensing in TS (Asghari, Jalali - Optica)
      2015: Real-time spectroscopy review (Jalali, Mahjoubfar - Science)
      2017: Coherent TS detection (Mahjoubfar et al. - Nature Photonics)
      2017: ML on PTS data (Tong et al.)
    """
    return {
        'H_f': {
            'formula': 'H(f) = exp(j*pi*beta2*L*(2*pi*f)^2)',
            'all_pass': True,
            'physical_meaning': 'Quadratic spectral phase from group velocity dispersion',
            'connects_to': [
                'Fresnel diffraction (paraxial optics)',
                'Free-particle Schrodinger equation (QM)',
                'Chirp-z transform (DSP)',
                'LFM radar (pulse compression)',
                'TD-GS phase retrieval (this repo)',
            ],
        },
        'techniques': {
            'DFT': 'I(t) = |E_in(f=t/D_eff)|^2 : real-time spectroscopy',
            'TS-ADC': 'M = 1+D2*L2/(D1*L1) : analog bandwidth multiplier',
            'STEAM': '2D image -> grating -> fiber -> 1D time series',
            'Rogue': 'MI -> Peregrine soliton -> I_peak = 9*P0',
            'CS-TS': 'y = Phi*x, M<<N, L1 recovery, RIP: M>=C*K*log(N/K)',
            'Coherent': 'IQ recovery: C = 2*B*log2(1+SNR) vs B*log2(1+SNR)',
            'ML-PTS': '9 features: PAPR,kurtosis,.. -> random forest classifier',
            'TD-GS': 'H(f)=const phase -> GS with |D|>=5000 -> phase recovery',
        },
        'timeline': [
            (1999, 'Photonic TS-ADC', 'IEEE Photon. Technol. Lett.'),
            (2007, 'Optical rogue waves', 'Nature 450'),
            (2009, 'STEAM camera', 'Nature 458'),
            (2014, 'CS in time-stretch', 'Optica 1'),
            (2015, 'Real-time spectroscopy review', 'Science 348'),
            (2017, 'Coherent TS detection', 'Nat. Photon. 11'),
            (2017, 'ML on PTS', 'arXiv'),
        ],
        'sbir_connection': (
            'RogueGuard (Phase I $275K, Phase II $1.75M): applies TD-GS + CNN '
            'for real-time rogue wave characterization at sea. '
            'H(f) from FIBER -> GS phase retrieval -> anomaly classifier.'
        ),
    }


def demo():
    print("=== JALALI MODERN PHYSICS: PHOTONIC TIME-STRETCH ===\n")

    # 1. DFT
    print("--- 1. Dispersive Fourier Transform ---")
    dft = dispersive_fourier_transform(D_ps_nm_km=1000.0, L_km=5.0,
                                       f_bandwidth_GHz=100.0)
    print(f"  D_eff = {dft['fiber']['D_eff_ps_nm']:.0f} ps/nm")
    print(f"  DFT condition: {dft['DFT_condition']['satisfied']}")
    print(f"  Time window: {dft['time_to_wavelength']['time_window_ps']:.1f} ps")
    print(f"  Lambda window: {dft['time_to_wavelength']['wavelength_window_nm']:.3f} nm")
    print(f"  {dft['principle']}")

    # 2. STEAM
    print("\n--- 2. STEAM Camera ---")
    st = steam_camera(n_rows=50, n_cols=100, D_ps_nm_km=1000.0, L_km=5.0,
                      f_rep_MHz=10.0, BW_nm=20.0)
    print(f"  Grating: {st['grating']['theta_Littrow_deg']:.1f} deg Littrow (valid={st['grating']['valid']})")
    print(f"  Frame rate: {st['system']['frame_rate_MHz']} MHz")
    print(f"  Pixel rate: {st['system']['pixel_rate_MHz']:.0f} MHz")
    print(f"  Reconstruction lossless: {st['reconstruction']['lossless']}")

    # 3. Rogue waves
    print("\n--- 3. Optical Rogue Waves ---")
    rw = optical_rogue_waves(n_pts=512, n_ensemble=30, rng_seed=42)
    print(f"  MI peak frequency: {rw['modulation_instability']['f_MI_peak_GHz']:.2f} GHz")
    print(f"  Peregrine amplification: {rw['Peregrine_soliton']['amplification_factor']:.1f}x (theory: ~9x)")
    print(f"  Rogue events: {rw['ensemble']['n_rogue_events']}/{rw['ensemble']['n_ensemble']} "
          f"(p={rw['ensemble']['rogue_probability']:.2f})")

    # 4. Compressed sensing
    print("\n--- 4. Compressed Sensing ---")
    cs = compressed_sensing_ts(N=128, K=8, stretch_factor=5.0)
    print(f"  M={cs['signal']['M_measurements']} of N={cs['signal']['N_Nyquist']} "
          f"(ratio {cs['signal']['compression_ratio']:.1f}x)")
    print(f"  RIP satisfied: {cs['RIP']['satisfied']} (mu={cs['RIP']['mu_coherence']:.3f})")
    print(f"  ISTA NMSE: {cs['ISTA']['NMSE']:.4f}  recovery_ok={cs['ISTA']['recovery_ok']}")
    print(f"  B_captured = {cs['photonic_advantage']['B_captured_GHz']:.1f} GHz")

    # 5. Coherent detection
    print("\n--- 5. Coherent Time-Stretch ---")
    cd = coherent_time_stretch(modulation_format='QPSK')
    print(f"  SNR improvement: {cd['SNR']['improvement_dB']:.1f} dB")
    print(f"  Capacity: {cd['capacity']['C_direct_Gbps']:.2f} -> "
          f"{cd['capacity']['C_coherent_Gbps']:.2f} Gbps "
          f"({cd['capacity']['ratio']:.1f}x)")

    # 6. ML
    print("\n--- 6. ML on PTS Data ---")
    ml = ml_on_pts_data(n_train=200, n_test=50)
    print(f"  Accuracy: {ml['accuracy']*100:.1f}%")
    print(f"  Top feature: {ml['top_feature']}")

    # Summary
    print("\n--- System Summary ---")
    ss = jalali_system_summary()
    print(f"  H(f): {ss['H_f']['formula']}")
    print(f"  Techniques: {len(ss['techniques'])}")
    for year, tech, ref in ss['timeline']:
        print(f"    {year}: {tech}")

    print("\n=== JALALI MODERN PHYSICS COMPLETE ===")


if __name__ == '__main__':
    demo()
