"""
Uncertainty, Vector Spaces, and Photonic QM
============================================
Senior project module: Physical design of photonic computing
through the lens of quantum mechanics and information theory.

THREAD: Heisenberg Uncertainty -> Time-Bandwidth -> Dispersion -> GS Phase Retrieval

SECTION 1: Heisenberg Uncertainty Principle
  [x, p] = j*hbar  =>  delta_x * delta_p >= hbar/2
  Fourier dual:          delta_t * delta_f >= 1/(4*pi)
  Photonic dual:         delta_t * delta_nu >= 0.4413 (Gaussian pulse)
  Engineering form:      TBP = T_pulse * BW >= 0.44 (time-bandwidth product)
  Time-stretch:          T_out = M * T_in  =>  BW_out = BW_in / M
                         TBP is CONSERVED by dispersion (unitary transform)

SECTION 2: Hilbert Space & Vector Spaces in QM/Photonics
  State |psi> = sum_n c_n |n>     (superposition)
  Operators: A|psi> = a|psi>      (eigenvalue equation)
  x-representation: psi(x) = <x|psi>
  p-representation: phi(k) = <k|psi> = FT{psi(x)}
  Wigner quasi-probability: W(x,p) = 1/pi * integral psi*(x+y)*psi(x-y)*e^{2jpy/hbar} dy

SECTION 3: Photonic Vector Space (Signal Processing)
  Optical field E(t) in L^2(R):
    Inner product: <E1|E2> = integral E1*(t)*E2(t) dt
    Norm: ||E||^2 = integral |E(t)|^2 dt  (energy)
    Basis: {e^{j*2*pi*f*t}} (Fourier / frequency modes)
    Completeness: Parseval's theorem
  Time-stretch maps this space: E_out(t) = IFT{E_in(f) * H(f)}
  GS phase retrieval: find E given {|E_in(t)|, |E_out(t)|}

SECTION 4: Statistical Resolution & Orders of Magnitude
  Shot noise limit:   SNR = sqrt(N_photons)
  Standard quantum limit: delta_phi = 1/sqrt(N)  (phase measurement)
  Heisenberg limit:   delta_phi = 1/N  (entangled states)
  Classical (RMS):    delta_x = lambda/(2*NA) (Rayleigh)
  Time-stretch ADC:   delta_t_resolved = 1/(M * BW_ADC)

SECTION 5: New Variables / Functions for Senior Project
  T(omega) = transmission function (ring resonator, MZI)
  G(t,f)   = Wigner-Ville distribution (joint time-freq)
  B(f)     = Bayesian posterior on optical phase
  A(q,k,v) = Attention(Q,K,V) = softmax(QK^T/sqrt(d))*V
"""
import math
import numpy as np


hbar     = 1.0546e-34   # J*s
h_P      = 6.626e-34    # J*s
c_light  = 2.998e8      # m/s
q_e      = 1.602e-19    # C
lambda0  = 1550e-9      # m (telecom C-band)
m_e      = 9.109e-31    # kg
kB       = 1.381e-23    # J/K


# ============================================================
# 1. Uncertainty Principle & Time-Bandwidth Product
# ============================================================

def uncertainty_principle(
    delta_x_nm=0.1,         # position uncertainty [nm]
    delta_t_ps=1.0,         # temporal pulse width [ps]
    BW_GHz=100.0,           # optical bandwidth [GHz]
    M_stretch=10.0,         # time-stretch factor
    n_pts=1024,
):
    """
    Heisenberg Uncertainty Principle and its photonic engineering duals.

    POSITION-MOMENTUM:
      [x, p] = j*hbar  (canonical commutation relation)
      delta_x * delta_p >= hbar/2
      For Gaussian state: delta_x * delta_p = hbar/2  (minimum uncertainty)

    TIME-FREQUENCY (ENGINEERING):
      delta_t [s] * delta_f [Hz] >= 1/(4*pi)
      Gaussian pulse: delta_t * delta_f = 1/(4*pi)  (minimum)
      TBP = T_FWHM * BW_FWHM >= 0.4413  (Gaussian: equality)

    PHOTONIC TIME-STRETCH CONSEQUENCE:
      Dispersive fiber stretches time by M:
        T_out = M * T_in  =>  f_out(t) = f_in(t/M)
        BW_out = BW_in / M   (bandwidth reduced)
        TBP_out = TBP_in     (CONSERVED -- unitary operation)
      This is WHY time-stretch works: the TBP is conserved
      while the temporal extent grows -> ADC can resolve it.

    ORDERS OF MAGNITUDE:
      hbar = 1.055e-34 J*s  (quantum of action)
      kT   = 25.85 meV at 300K  (thermal energy)
      h*f  = 0.80 eV at 1550 nm (photon energy)
      h*f >> kT: photons are quantum objects at telecom wavelengths

    RESOLUTION HIERARCHY:
      Heisenberg limit:    delta_phi = 1/N   [1 photon/measurement]
      Shot noise limit:    delta_phi = 1/sqrt(N)
      Classical Rayleigh:  delta_x = 0.61*lambda/NA
      Time-stretch ADC:    delta_t = 1/(M * f_ADC)
    """
    if M_stretch < 1.0:
        raise ValueError("M_stretch must be >= 1.0")
    if delta_t_ps <= 0 or BW_GHz <= 0:
        raise ValueError("delta_t_ps and BW_GHz must be positive")

    # Position-momentum
    delta_x_m = delta_x_nm * 1e-9
    delta_p_min = hbar / (2 * delta_x_m)   # minimum uncertainty [kg*m/s]
    delta_p_electron = delta_p_min   # For electron
    delta_v_electron = delta_p_min / m_e   # velocity uncertainty [m/s]

    # Time-frequency
    delta_t_s = delta_t_ps * 1e-12
    delta_f_min_Hz = 1.0 / (4*math.pi * delta_t_s)   # Heisenberg minimum [Hz]
    delta_f_min_GHz = delta_f_min_Hz * 1e-9
    TBP = delta_t_ps * BW_GHz   # [ps * GHz = dimensionless * 1e3... actually ps*GHz=1e-12*1e9=1e-3]
    # Engineering TBP: T[ps]*BW[GHz] = 1e-3 * TBP_SI; Gaussian min TBP_SI = 1/(4pi) = 0.0796
    # FWHM form: TBP_FWHM = 0.4413 for Gaussian
    TBP_SI = delta_t_s * BW_GHz * 1e9   # dimensionless
    TBP_FWHM_Gaussian = 0.4413
    TBP_satisfied = TBP_SI >= 1.0/(4*math.pi) - 1e-6

    # Time-stretch consequences
    T_out_ps = M_stretch * delta_t_ps
    BW_out_GHz = BW_GHz / M_stretch
    TBP_out = T_out_ps * 1e-12 * BW_out_GHz * 1e9   # should equal TBP_in
    TBP_conserved = abs(TBP_out - TBP_SI) < 1e-12

    # Resolution comparison
    NA = 0.5   # numerical aperture example
    rayleigh_nm = 0.61 * lambda0 * 1e9 / NA   # [nm]
    f_ADC_GHz = 1.0
    delta_t_ADC_ps = 1.0 / (M_stretch * f_ADC_GHz * 1e9) * 1e12   # [ps]
    delta_t_ADC_no_stretch_ps = 1.0 / (f_ADC_GHz * 1e9) * 1e12

    # Photon energy vs kT
    E_photon_eV = h_P * c_light / lambda0 / q_e
    kT_eV = kB * 300 / q_e

    # Gaussian wavefunction (minimum uncertainty state)
    x_arr = np.linspace(-5*delta_x_m, 5*delta_x_m, n_pts)
    psi_x = ((2*math.pi*delta_x_m**2)**(-0.25) *
              np.exp(-x_arr**2 / (4*delta_x_m**2)))
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi_x)))
    norm_check = float(np.trapezoid(np.abs(psi_x)**2, x_arr))

    # Wigner function (simplified 2D slice)
    N_w = 64
    x_w = np.linspace(-3*delta_x_m, 3*delta_x_m, N_w)
    p_arr = np.linspace(-3*delta_p_min, 3*delta_p_min, N_w)
    W = np.zeros((N_w, N_w))
    for i, x0 in enumerate(x_w):
        for j, p0 in enumerate(p_arr):
            y_int = np.linspace(-2*delta_x_m, 2*delta_x_m, 64)
            psi_plus  = ((2*math.pi*delta_x_m**2)**(-0.25) *
                         np.exp(-(x0+y_int)**2/(4*delta_x_m**2)))
            psi_minus = ((2*math.pi*delta_x_m**2)**(-0.25) *
                         np.exp(-(x0-y_int)**2/(4*delta_x_m**2)))
            integrand = np.conj(psi_plus)*psi_minus*np.exp(2j*p0*y_int/hbar)
            W[i,j] = float(np.real(np.trapezoid(integrand, y_int))) / math.pi

    return {
        'position_momentum': {
            'delta_x_nm': float(delta_x_nm),
            'delta_p_min_kg_m_s': float(delta_p_min),
            'delta_v_electron_m_s': float(delta_v_electron),
            'HUP_formula': 'delta_x * delta_p >= hbar/2',
            'commutator': '[x, p] = j*hbar',
        },
        'time_frequency': {
            'delta_t_ps': float(delta_t_ps),
            'BW_GHz': float(BW_GHz),
            'delta_f_min_GHz': float(delta_f_min_GHz),
            'TBP_SI': float(TBP_SI),
            'TBP_min': float(1.0/(4*math.pi)),
            'TBP_satisfied': bool(TBP_satisfied),
            'TBP_FWHM_Gaussian': float(TBP_FWHM_Gaussian),
        },
        'time_stretch': {
            'M': float(M_stretch),
            'T_in_ps': float(delta_t_ps),
            'T_out_ps': float(T_out_ps),
            'BW_in_GHz': float(BW_GHz),
            'BW_out_GHz': float(BW_out_GHz),
            'TBP_conserved': bool(TBP_conserved),
            'TBP_in': float(TBP_SI),
            'TBP_out': float(TBP_out),
            'insight': 'Dispersion is unitary: TBP is conserved. Stretch enables ADC resolution.',
        },
        'resolution': {
            'Rayleigh_nm': float(rayleigh_nm),
            'delta_t_no_stretch_ps': float(delta_t_ADC_no_stretch_ps),
            'delta_t_with_stretch_ps': float(delta_t_ADC_ps),
            'stretch_improvement': float(M_stretch),
            'Heisenberg_limit': '1/N',
            'shot_noise_limit': '1/sqrt(N)',
        },
        'energy_scales': {
            'E_photon_1550nm_eV': float(E_photon_eV),
            'kT_300K_eV': float(kT_eV),
            'ratio_E_photon_kT': float(E_photon_eV/kT_eV),
            'quantum_regime': bool(E_photon_eV > kT_eV),
        },
        'wavefunction': {
            'psi_x_norm': float(norm_check),
            'psi_x': psi_x.tolist(),
            'x_nm': (x_arr*1e9).tolist(),
        },
        'wigner': {
            'W': W.tolist(),
            'x_nm': (x_w*1e9).tolist(),
            'p_kg_m_s': p_arr.tolist(),
            'note': 'Gaussian minimum uncertainty: W >= 0 everywhere',
        },
    }


# ============================================================
# 2. Vector Spaces & Operators in Photonic QM
# ============================================================

def photonic_vector_space(
    n_modes=8,             # number of spatial/frequency modes
    n_pts=256,             # signal length
    BW_GHz=100.0,
    lambda0_nm=1550.0,
):
    """
    Hilbert Space formalism for photonic fields.

    QUANTUM MECHANICS -> PHOTONICS MAPPING:
      |psi>      -> E(t)           (state -> electric field)
      <psi|psi>  -> integral|E|^2  (norm -> total energy)
      H|psi>=E|psi> -> [D^2/dt^2]E = -(omega/c)^2*E (wave eq. eigenvalue)
      Measurement -> photodetection (|<n|psi>|^2 = probability)
      Unitary U  -> dispersive fiber H(f)=exp(j*phi(f))  [|H|=1]
      Hermitian A -> observable (real eigenvalues)

    MODES OF THE FIELD:
      Spatial modes:    E_n(x) = sqrt(2/w) * sin(n*pi*x/w)  [waveguide]
      Frequency modes:  E_n(t) = exp(j*2*pi*n*f0*t)  [frequency comb]
      Both: complete orthonormal bases in L^2

    OPERATORS:
      Time delay:   T_tau: E(t) -> E(t-tau)  (diagonal in freq. domain)
      Dispersion:   D_phi: E(f) -> E(f)*exp(j*phi(f))  (diagonal in freq)
      Modulation:   M_m:   E(t) -> E(t)*m(t)  (convolution in freq)
      Amplification:A_G:   E(t) -> sqrt(G)*E(t)  (NOT unitary, G>1)

    INNER PRODUCT SPACE:
      <E1, E2> = integral_{-inf}^{inf} E1*(t) * E2(t) dt
      By Parseval: = integral E1*(f) * E2(f) df
      Orthogonality: <E_m, E_n> = delta_{mn}

    SINGULAR VALUE DECOMPOSITION (SVD) OF FIBER CHANNEL:
      H = U * Sigma * V^H
      For dispersive fiber: |H(f)|=1 -> singular values all = 1 -> H is unitary
      U, V are rotation matrices in mode space.
    """
    t_arr = np.linspace(0, 1, n_pts) * (n_pts / (BW_GHz*1e9)) * 1e12   # [ps]
    dt = t_arr[1] - t_arr[0]   # [ps]

    # Waveguide spatial modes: sin basis on [0, w]
    w = 1.0   # normalized waveguide width
    x_arr = np.linspace(0, w, n_pts)
    modes = np.zeros((n_modes, n_pts))
    for n in range(1, n_modes+1):
        modes[n-1] = math.sqrt(2/w) * np.sin(n*math.pi*x_arr/w)

    # Orthonormality check: <E_m, E_n> = delta_mn
    gram = np.zeros((n_modes, n_modes))
    dx = w/n_pts
    for i in range(n_modes):
        for j in range(n_modes):
            gram[i,j] = float(np.trapezoid(modes[i]*modes[j], x_arr))
    orthonormal_error = float(np.max(np.abs(gram - np.eye(n_modes))))

    # Frequency comb modes
    f0_GHz = BW_GHz / n_modes
    comb_modes = np.array([
        np.exp(1j*2*math.pi*n*f0_GHz*1e9*t_arr*1e-12)
        for n in range(n_modes)
    ])
    # Gram matrix for comb (sampling -> approximate)
    gram_comb = np.abs(comb_modes @ np.conj(comb_modes).T) * dt*1e-12
    gram_comb_diag = float(np.mean(np.abs(np.diag(gram_comb))))

    # Dispersion operator (diagonal in frequency domain)
    f_arr = np.fft.fftfreq(n_pts, d=dt*1e-12) * 1e-9   # [GHz]
    D_ps_nm_km = 1000.0; L_km = 5.0
    D_eff = D_ps_nm_km * L_km
    lambda0_m = lambda0_nm * 1e-9
    D_eff_SI = D_eff * 1e-12/1e-9
    beta2L = -(lambda0_m**2/(2*math.pi*c_light)) * D_eff_SI
    H_f = np.exp(1j*math.pi*beta2L*(2*math.pi*f_arr*1e9)**2)

    # Verify unitarity: |H(f)| = 1
    H_unitary_err = float(np.max(np.abs(np.abs(H_f) - 1.0)))

    # SVD of H (as circulant matrix via diagonal in frequency)
    # For a diagonal unitary: U=I, Sigma=I, V=H -> singular values all 1
    singular_values = np.abs(H_f[:n_modes])

    # Momentum operator: p = -j*hbar * d/dx  (finite difference)
    dx_m = (lambda0_nm*1e-9) * w / n_pts   # scale dx to physical units
    D_mat = (np.diag(np.ones(n_pts-1), 1) -
             np.diag(np.ones(n_pts-1), -1)) / (2*dx_m)
    p_op = -1j*hbar * D_mat[:8, :8]   # 8x8 slice
    # Hermitian check: A = A^dag
    p_hermitian_err = float(np.max(np.abs(p_op - p_op.conj().T)))

    # Position operator: x (diagonal)
    x_op = np.diag(x_arr[:8] * lambda0_m * w)
    # Commutator [x, p] = j*hbar * I
    xp_px = x_op @ p_op - p_op @ x_op
    expected = 1j*hbar*np.eye(8)
    commutator_err = float(np.max(np.abs(xp_px - expected)))

    # Transfer matrix of MZI (photonic beamsplitter)
    theta = math.pi/4   # 50:50
    T_MZI = np.array([
        [math.cos(theta), 1j*math.sin(theta)],
        [1j*math.sin(theta), math.cos(theta)]
    ])
    T_unitary_err = float(np.max(np.abs(T_MZI @ np.conj(T_MZI).T - np.eye(2))))

    return {
        'hilbert_space_map': {
            '|psi>':           'E(t) -- optical field',
            '<psi|psi>':       'integral|E|^2 -- energy',
            'H|psi>=E|psi>':   'd^2E/dt^2 = -(omega/c)^2 E -- wave equation eigenvalue',
            'unitary U':       'H(f)=exp(j*phi) -- dispersive fiber (|H|=1)',
            'Hermitian A':     'observable -- real eigenvalues',
            'measurement':     'photodetection -- |<n|psi>|^2',
        },
        'spatial_modes': {
            'n_modes': n_modes,
            'orthonormality_error': float(orthonormal_error),
            'orthonormal': bool(orthonormal_error < 1e-10),
            'modes_shape': [n_modes, n_pts],
            'basis': 'sin(n*pi*x/w)  -- Dirichlet waveguide modes',
        },
        'operators': {
            'dispersion_H_unitary_error': float(H_unitary_err),
            'H_is_unitary': bool(H_unitary_err < 1e-10),
            'singular_values_of_H': singular_values.tolist(),
            'all_singular_values_1': bool(np.all(np.abs(singular_values - 1.0) < 1e-10)),
            'p_hermitian_error': float(p_hermitian_err),
            'p_is_hermitian': bool(p_hermitian_err < 1e-8),
            'commutator_[x,p]_error': float(commutator_err),
            'MZI_T_unitary_error': float(T_unitary_err),
        },
        'MZI': {
            'matrix': T_MZI.tolist(),
            'unitary': bool(T_unitary_err < 1e-14),
            'theta_deg': float(math.degrees(theta)),
            'splitting_ratio': '50:50',
        },
        'senior_project_variables': {
            'T(omega)': 'transmission: MZI, ring resonator -> eigenvalues of transfer matrix',
            'G(t,f)':   'Wigner-Ville distribution -> joint time-freq uncertainty',
            'B(f)':     'Bayesian posterior on phase(f) given |E(f)|^2',
            'A(Q,K,V)': 'Attention = softmax(QK^T/sqrt(d))*V -> phase pattern recognition',
            'H(f)':     'exp(j*pi*beta2*L*(2*pi*f)^2) -> connects ALL topics here',
            'S(omega)': 'power spectral density: Wiener-Khinchin theorem',
        },
    }


# ============================================================
# 3. Multiplexed Pulsed Laser + Dispersion Physics
# ============================================================

def multiplexed_pulsed_laser(
    n_channels=4,            # WDM channels
    f_rep_MHz=100.0,         # laser repetition rate [MHz]
    T_pulse_ps=1.0,          # pulse duration [ps]
    BW_per_ch_nm=5.0,        # bandwidth per channel [nm]
    D_ps_nm_km=1000.0,       # fiber GVD
    L_km=5.0,                # fiber length
    lambda_center_nm=1550.0,
    ch_spacing_nm=10.0,      # channel spacing [nm]
    n_pts=1024,
):
    """
    Wavelength-Division Multiplexed (WDM) pulsed laser for photonic time-stretch.

    SYSTEM:
      n_channels mode-locked lasers at lambda_i = lambda_c + i*d_lambda
      Each channel independently stretched by M_i = 1 + D2*L2/(D1*L1)
      Multiplexed onto single fiber -> parallel ADC streams

    TIME-BANDWIDTH PER CHANNEL:
      TBP_ch = T_pulse * BW_ch >= 0.44 (Gaussian minimum)
      After stretch M: T_out = M*T_pulse, BW_out = BW_ch/M -> TBP conserved

    DISPERSION-BASED CHANNEL DEMUX:
      Different lambdas arrive at different times:
        delta_t(lambda) = D * L * delta_lambda  [ps]
      This IS the DFT time-to-frequency mapping.
      Channel i arrives at t_i = D*L*(lambda_i - lambda_c)  [ps]

    TOTAL THROUGHPUT:
      B_total = n_channels * M * f_rep / 2  (Nyquist)
      Spectral efficiency = B_total / (n_channels * BW_ch)

    COLLISION / CROSS-TALK:
      Pulses must not overlap: delta_t_{i,i+1} > T_out
      => D*L*d_lambda > M*T_pulse
      => d_lambda > M*T_pulse / (D*L)  [nm]
    """
    D_eff = D_ps_nm_km * L_km   # ps/nm

    # Channel wavelengths
    lambda_ch = np.array([
        lambda_center_nm + (i - n_channels//2) * ch_spacing_nm
        for i in range(n_channels)
    ])

    # Stretch factor (assuming single-stage stretch: D*L for both)
    M = 5.0   # representative; could be computed from two-stage fiber

    # Arrival times (DFT mapping)
    t_arrival_ps = D_eff * (lambda_ch - lambda_center_nm)   # [ps]

    # Pulse duration after stretch
    T_out_ps = M * T_pulse_ps
    BW_out_GHz = BW_per_ch_nm * c_light * 1e-9 / lambda_center_nm**2 / M  # approx BW in GHz

    # Crosstalk check
    dt_adjacent_ps = float(D_eff * ch_spacing_nm)
    no_overlap = dt_adjacent_ps > T_out_ps

    # Total bandwidth
    total_BW_GHz = BW_per_ch_nm * c_light * 1e-9 / lambda_center_nm**2 * n_channels
    B_total_Gbps = n_channels * M * (BW_per_ch_nm * c_light * 1e-9 / lambda_center_nm**2) / 2

    # TBP per channel
    TBP_in = T_pulse_ps * 1e-12 * BW_per_ch_nm * c_light/lambda_center_nm**2
    TBP_out = T_out_ps * 1e-12 * BW_out_GHz * 1e9
    TBP_conserved = abs(TBP_out/max(TBP_in,1e-30) - 1.0) < 0.05

    # Simulate WDM pulse train (time domain)
    t_arr = np.linspace(-500, 500, n_pts)   # ps
    E_total = np.zeros(n_pts, dtype=complex)
    for i, lam in enumerate(lambda_ch):
        t0 = float(t_arrival_ps[i])
        sigma = T_out_ps / (2*math.sqrt(2*math.log(2)))   # FWHM -> sigma
        f_ch = c_light / (lam*1e-9) - c_light/lambda_center_nm   # detuning [Hz]
        E_ch = np.exp(-(t_arr-t0)**2/(2*sigma**2)) * np.exp(1j*2*math.pi*f_ch*t_arr*1e-12)
        E_total += E_ch

    I_total = np.abs(E_total)**2

    # Spectral density
    E_f = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_total)))
    f_GHz = np.fft.fftshift(np.fft.fftfreq(n_pts, d=(t_arr[1]-t_arr[0])*1e-12)) * 1e-9
    S_f = np.abs(E_f)**2

    return {
        'system': {
            'n_channels': n_channels,
            'f_rep_MHz': float(f_rep_MHz),
            'T_pulse_ps': float(T_pulse_ps),
            'lambda_channels_nm': lambda_ch.tolist(),
            'ch_spacing_nm': float(ch_spacing_nm),
            'D_eff_ps_nm': float(D_eff),
            'M': float(M),
        },
        'time_encoding': {
            't_arrival_ps': t_arrival_ps.tolist(),
            'T_out_ps': float(T_out_ps),
            'dt_adjacent_ps': float(dt_adjacent_ps),
            'no_overlap': bool(no_overlap),
            'crosstalk_condition': f'dt_adj ({dt_adjacent_ps:.0f} ps) > T_out ({T_out_ps:.1f} ps)',
        },
        'TBP': {
            'TBP_in': float(TBP_in),
            'TBP_out': float(TBP_out),
            'conserved': bool(TBP_conserved),
            'insight': 'Dispersion is symplectic: conserves phase space volume',
        },
        'throughput': {
            'B_total_Gbps': float(B_total_Gbps),
            'total_BW_GHz': float(total_BW_GHz),
            'spectral_efficiency': float(B_total_Gbps / max(total_BW_GHz, 1e-10)),
        },
        'time_domain': {
            't_ps': t_arr.tolist(),
            'I_total': I_total.tolist(),
            'f_GHz': f_GHz.tolist(),
            'S_f': S_f.tolist(),
        },
    }


# ============================================================
# 4. Math Operations Grammar (Symbolic + Numeric)
# ============================================================

def math_operations_grammar(x=2.5, n=4):
    """
    Complete arithmetic and transcendental operations grammar.
    Maps to photonic/QM expressions for computer engineering context.

    Every operation has a photonic interpretation:
      +  addition:       superposition of fields
      -  subtraction:    interference (destructive)
      *  multiplication: modulation / mixing
      /  division:       normalization / beam splitter ratio
      ** exponent:       H(f) = exp(j*phi)
      %  modulo:         phase wrapping: phi mod 2*pi
      log:               Shannon entropy: -p*log(p)
      sqrt:              amplitude from intensity: A=sqrt(I)
      integral:          energy = integral |E|^2 dt
      derivative:        group delay: tau_g = -d(phi)/d(omega)
      convolution:       output = input * h(t)  [impulse response]
    """
    if n < 1 or not isinstance(n, int):
        raise ValueError("n must be a positive integer")

    # Basic arithmetic
    ops = {
        'x': float(x),
        'n': n,
        'addition':        x + n,          # superposition
        'subtraction':     x - n,          # interference
        'multiplication':  x * n,          # modulation
        'division':        x / n,          # splitting
        'exponent':        x ** n,         # nonlinear optics: chi^(n)
        'modulo':          x % n,          # phase wrapping
        'floor_div':       x // n,         # pulse counting
        'absolute':        abs(x - n),     # intensity from field
    }

    # Transcendentals
    ops['sqrt']  = math.sqrt(abs(x))              # amplitude from power
    ops['log_e'] = math.log(max(abs(x), 1e-30))  # Shannon entropy
    ops['log_2'] = math.log2(max(abs(x), 1e-30)) # bits / capacity
    ops['log_10']= math.log10(max(abs(x), 1e-30))# dB = 10*log10
    ops['exp']   = math.exp(x)                    # H(f)=exp(j*phi)
    ops['sin']   = math.sin(x)                    # I-channel
    ops['cos']   = math.cos(x)                    # Q-channel
    ops['atan2_x_1'] = math.atan2(x, 1.0)        # phase recovery

    # Photonic expressions
    phi = x                      # phase [rad]
    ops['phase_wrapped_rad'] = phi % (2*math.pi)
    ops['phase_wrapped_deg'] = math.degrees(phi % (2*math.pi))

    # dB conversions
    ops['to_dB']   = 10*math.log10(max(abs(x), 1e-30))
    ops['from_dB'] = 10**(x/10)

    # Group delay: tau_g = -d(phi)/d(omega) -- finite difference
    omega = 2*math.pi*100e9   # 100 GHz angular frequency
    domega = 2*math.pi*1e9    # 1 GHz perturbation
    D_val = 1000.0*5.0        # D_eff [ps/nm]
    lambda_m = 1550e-9
    D_SI = D_val*1e-12/1e-9
    beta2L = -(lambda_m**2/(2*math.pi*c_light)) * D_SI
    phi_omega  = math.pi * beta2L * omega**2
    phi_plus   = math.pi * beta2L * (omega+domega)**2
    tau_g_s = -(phi_plus - phi_omega) / domega   # [s]
    tau_g_ps = tau_g_s * 1e12
    ops['group_delay_ps'] = float(tau_g_ps)
    ops['derivative_note'] = 'tau_g = -d(phi)/d(omega) = -2*pi*beta2*L*f [finite diff]'

    # Integral: energy normalization
    t_arr = np.linspace(-10*x, 10*x, 1000) if abs(x) > 1e-10 else np.linspace(-1, 1, 1000)
    sigma = max(abs(x), 0.01)
    gaussian = np.exp(-t_arr**2/(2*sigma**2))
    energy = float(np.trapezoid(gaussian**2, t_arr))
    ops['integral_gaussian_energy'] = energy
    ops['expected_energy'] = math.sqrt(math.pi/2) * sigma   # analytic: sqrt(pi/2)*sigma

    # Convolution (time-domain dispersive spreading)
    n_c = 64
    t_c = np.linspace(-5, 5, n_c)
    pulse = np.exp(-t_c**2)
    h = np.exp(-t_c**2/4)   # broader kernel
    conv = np.convolve(pulse, h, mode='same') * (t_c[1]-t_c[0])
    ops['convolution_peak'] = float(np.max(np.abs(conv)))

    # Shannon entropy / information
    p = np.array([0.5, 0.25, 0.125, 0.125])   # probability distribution
    H_shannon = float(-np.sum(p * np.log2(np.maximum(p, 1e-30))))
    ops['Shannon_entropy_bits'] = H_shannon
    ops['Shannon_formula'] = 'H = -sum(p*log2(p))'
    ops['capacity_formula'] = 'C = B*log2(1+SNR) [bits/s]'

    # Fourier duality (HUP from FT)
    ops['FT_HUP'] = 'delta_t * delta_f >= 1/(4*pi)  [from FT uncertainty]'
    ops['chain_rule'] = 'd/dt[exp(j*phi(t))] = j*dphi/dt * exp(j*phi(t))'

    return ops


# ============================================================
# 5. Statistical Analysis & Resolution
# ============================================================

def statistical_resolution(
    N_photons=1000,
    n_measurements=500,
    phi_true=0.75,          # true phase [rad]
    SNR_dB=20.0,
    rng_seed=5,
):
    """
    Statistical estimation of phase -- from shot noise to Heisenberg limit.

    ORDERS OF MAGNITUDE IN PHOTONICS:
      N_photons = 1:      delta_phi = 1 rad (shot noise limit for 1 photon)
      N_photons = 100:    delta_phi = 0.1 rad
      N_photons = 1e6:    delta_phi = 1e-3 rad
      Heisenberg limit:   delta_phi = 1/N  [requires entangled N-photon state]
      GI interferometer:  delta_phi = 1/N (NOON state)
      Classical:          delta_phi = 1/sqrt(N) (coherent state)

    BAYESIAN INFERENCE:
      Prior:    p(phi) ~ Uniform[0, 2*pi]
      Likelihood: L(x|phi) = exp(-N*(cos(x-phi)-1))  [Poisson photon counting]
      Posterior: p(phi|x) = L(x|phi) * p(phi) / Z
      MAP estimate: phi_MAP = argmax p(phi|x)
      MMSE estimate: phi_MMSE = integral phi * p(phi|x) dphi

    MAXIMUM LIKELIHOOD ESTIMATION (MLE):
      For homodyne: x = A*cos(phi) + noise, y = A*sin(phi) + noise
      phi_MLE = atan2(mean(y), mean(x))
      delta_phi_MLE = 1/sqrt(N) * 1/(signal power)   [Cramer-Rao bound]

    CRAMER-RAO BOUND:
      Var(phi_hat) >= 1 / I_Fisher(phi)
      I_Fisher = N * <(d/dphi log L)^2>  [Fisher information]
      For homodyne: I_Fisher = N * SNR_linear
      CRB: delta_phi_CRB = 1/sqrt(N * SNR_linear)
    """
    rng = np.random.default_rng(rng_seed)
    SNR_lin = 10**(SNR_dB/10)
    A = math.sqrt(SNR_lin)   # signal amplitude

    # Homodyne measurement: x = A*cos(phi) + n, y = A*sin(phi) + n
    noise_std = 1.0
    x_meas = A*math.cos(phi_true) + rng.standard_normal(n_measurements)*noise_std
    y_meas = A*math.sin(phi_true) + rng.standard_normal(n_measurements)*noise_std

    # MLE: batch and single-shot
    phi_MLE_batch = float(math.atan2(float(np.mean(y_meas)), float(np.mean(x_meas))))
    phi_hat_single = np.arctan2(y_meas, x_meas)
    phi_hat_single_unwrapped = phi_hat_single - 2*math.pi*np.round((phi_hat_single - phi_true)/(2*math.pi))

    # Errors
    error_MLE = float(abs(phi_MLE_batch - phi_true))
    std_single = float(np.std(phi_hat_single_unwrapped))

    # Cramer-Rao bound
    I_Fisher = n_measurements * SNR_lin
    CRB_std = 1.0 / math.sqrt(max(I_Fisher, 1e-30))

    # Shot noise and Heisenberg limits
    delta_phi_shot = 1.0 / math.sqrt(max(N_photons, 1))
    delta_phi_heisenberg = 1.0 / N_photons

    # Bayesian posterior (grid approximation)
    phi_grid = np.linspace(0, 2*math.pi, 512)
    log_prior = np.zeros(512)   # uniform prior
    # Likelihood from batch sufficient statistics
    x_bar = float(np.mean(x_meas))
    y_bar = float(np.mean(y_meas))
    log_L = n_measurements * A * (x_bar*np.cos(phi_grid) + y_bar*np.sin(phi_grid)) / (noise_std**2)
    log_posterior = log_L + log_prior
    log_posterior -= float(np.max(log_posterior))   # normalize
    posterior = np.exp(log_posterior)
    posterior /= float(np.trapezoid(posterior, phi_grid)) + 1e-30

    # MAP and MMSE from posterior
    phi_MAP = float(phi_grid[np.argmax(posterior)])
    phi_MMSE = float(np.trapezoid(phi_grid * posterior, phi_grid))
    posterior_std = float(math.sqrt(np.trapezoid((phi_grid - phi_MMSE)**2 * posterior, phi_grid)))

    # Summary table: resolution vs N_photons
    N_range = [1, 10, 100, 1000, 1e4, 1e5, 1e6]
    resolution_table = {
        f'N={int(N)}': {
            'shot_noise': float(1/math.sqrt(N)),
            'Heisenberg': float(1/N),
            'ratio': float(math.sqrt(N)),
        }
        for N in N_range
    }

    return {
        'true_phase_rad': float(phi_true),
        'N_photons': N_photons,
        'n_measurements': n_measurements,
        'MLE': {
            'phi_MLE_rad': float(phi_MLE_batch),
            'error_rad': float(error_MLE),
            'std_single_shot_rad': float(std_single),
            'CRB_std_rad': float(CRB_std),
            'CRB_satisfied': bool(std_single >= CRB_std * 0.5),
        },
        'Bayesian': {
            'phi_MAP_rad': float(phi_MAP),
            'phi_MMSE_rad': float(phi_MMSE),
            'posterior_std_rad': float(posterior_std),
            'posterior': posterior.tolist(),
            'phi_grid_rad': phi_grid.tolist(),
        },
        'limits': {
            'shot_noise_limit': float(delta_phi_shot),
            'Heisenberg_limit': float(delta_phi_heisenberg),
            'speedup': float(delta_phi_shot / delta_phi_heisenberg),
            'CRB': float(CRB_std),
        },
        'resolution_table': resolution_table,
        'GS_connection': (
            'TD-GS solves the same problem: recover phi from |E(t)| and |E_out(t)|. '
            'Each GS iteration ~ one Bayesian update on the phase posterior. '
            'Convergence rate ~ Fisher information of the diversity measurement.'
        ),
    }


def demo():
    print("=== UNCERTAINTY, VECTOR SPACES, PHOTONIC QM ===\n")

    # 1. HUP
    print("--- 1. Heisenberg Uncertainty ---")
    up = uncertainty_principle(delta_x_nm=0.1, delta_t_ps=1.0, BW_GHz=100.0, M_stretch=10.0)
    print(f"  delta_x = 0.1 nm  =>  delta_p >= {up['position_momentum']['delta_p_min_kg_m_s']:.2e} kg*m/s")
    print(f"  delta_v_electron = {up['position_momentum']['delta_v_electron_m_s']:.2e} m/s")
    print(f"  TBP (1ps, 100GHz) = {up['time_frequency']['TBP_SI']:.4f}  (min = {up['time_frequency']['TBP_min']:.4f})")
    print(f"  After stretch M=10: TBP conserved = {up['time_stretch']['TBP_conserved']}")
    print(f"  Photon energy 1550nm = {up['energy_scales']['E_photon_1550nm_eV']:.2f} eV >> kT = {up['energy_scales']['kT_300K_eV']:.3f} eV")

    # 2. Vector spaces
    print("\n--- 2. Vector Spaces ---")
    vs = photonic_vector_space(n_modes=8)
    print(f"  Mode orthonormality error: {vs['spatial_modes']['orthonormality_error']:.2e}")
    print(f"  H(f) unitarity error: {vs['operators']['H_is_unitary']} ({vs['operators']['dispersion_H_unitary_error']:.2e})")
    print(f"  p operator Hermitian: {vs['operators']['p_is_hermitian']}")
    print(f"  [x,p] commutator error: {vs['operators']['commutator_[x,p]_error']:.2e}")

    # 3. WDM
    print("\n--- 3. WDM Pulsed Laser ---")
    wdm = multiplexed_pulsed_laser(n_channels=4, T_pulse_ps=1.0, ch_spacing_nm=10.0)
    print(f"  4 channels: {[f'{l:.0f}' for l in wdm['system']['lambda_channels_nm']]} nm")
    print(f"  Arrival times: {[f'{t:.0f}' for t in wdm['time_encoding']['t_arrival_ps']]} ps")
    print(f"  No overlap: {wdm['time_encoding']['no_overlap']}")
    print(f"  TBP conserved: {wdm['TBP']['conserved']}")

    # 4. Math grammar
    print("\n--- 4. Math Operations ---")
    mg = math_operations_grammar(x=2.5, n=4)
    for k in ['addition','multiplication','exponent','modulo','log_2','exp','to_dB']:
        print(f"  {k:20s} = {mg[k]:.4f}")
    print(f"  group_delay_ps       = {mg['group_delay_ps']:.2f} ps")
    print(f"  Shannon_entropy_bits = {mg['Shannon_entropy_bits']:.4f}")

    # 5. Statistics
    print("\n--- 5. Statistical Resolution ---")
    sr = statistical_resolution(N_photons=1000, n_measurements=200, phi_true=0.75)
    print(f"  phi_true = {sr['true_phase_rad']:.3f} rad")
    print(f"  phi_MLE  = {sr['MLE']['phi_MLE_rad']:.4f} rad  (error={sr['MLE']['error_rad']:.4f})")
    print(f"  phi_MAP  = {sr['Bayesian']['phi_MAP_rad']:.4f} rad")
    print(f"  Shot noise limit: {sr['limits']['shot_noise_limit']:.4f} rad")
    print(f"  Heisenberg limit: {sr['limits']['Heisenberg_limit']:.6f} rad")
    print(f"  CRB:              {sr['limits']['CRB']:.4f} rad")
    print()
    print("  N_photons  |  shot noise  |  Heisenberg  |  speedup")
    for N_key, vals in list(sr['resolution_table'].items())[:5]:
        print(f"  {N_key:10s}   {vals['shot_noise']:.4f}         {vals['Heisenberg']:.6f}     {vals['ratio']:.0f}x")

    print("\n=== DONE ===")


if __name__ == '__main__':
    demo()
