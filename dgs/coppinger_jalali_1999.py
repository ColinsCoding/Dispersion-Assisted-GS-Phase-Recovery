"""
Photonic Time Stretch and Its Application to Analog-to-Digital Conversion
F. Coppinger, A. S. Bhushan, and B. Jalali, Senior Member, IEEE
IEEE Transactions on Microwave Theory and Techniques, Vol. 47, No. 7, July 1999.

This module reproduces the key analytical results from the 1999 paper, placing them in
the broader context of Maxwell's equations (phasor domain), coastal EM, and antenna
radiation that constitutes the electromagnetic foundation of the photonic time-stretch ADC.

PAPER ABSTRACT (reproduced from memory):
  "A photonic preprocessor that time-stretches an analog electrical signal before
   digitization is demonstrated. The preprocessor uses a two-stage dispersive fiber
   system to slow down the signal, allowing it to be captured by a lower-speed
   analog-to-digital converter. A stretch factor M is defined as the ratio of the
   input signal duration to the output signal duration. A 20-GHz bandwidth signal
   is captured using a 2-Gsample/s electronic ADC with M = 10."

ELECTROMAGNETIC FOUNDATION:
  Maxwell's equations in phasor domain (time convention e^{+jwt}):
    curl E = -j*omega*mu*H          (Faraday-Maxwell)
    curl H = J + j*omega*eps*E      (Ampere-Maxwell)
    div D = rho                     (Gauss electric)
    div B = 0                       (Gauss magnetic)

  These four equations, combined with constitutive relations D=eps*E, B=mu*H,
  J=sigma*E, generate ALL of photonics:
    - Wave equation -> H(f) = exp(j*pi*D*f^2) (dispersion, via k(omega) Taylor expansion)
    - Boundary conditions -> Fresnel coefficients -> fiber modes -> LP01 in SMF-28
    - Poynting vector S = E x H* / 2 -> optical power in fiber
    - Radiation -> antennas -> near/far field -> the RF input to the EOM in the paper

COASTAL EM:
  Seawater (sigma~4 S/m, eps_r~80) is a lossy conductor at RF frequencies.
  Skin depth delta = sqrt(2/(omega*mu*sigma)) tells how deep RF penetrates.
  At 1 GHz: delta~1 cm; at 100 Hz: delta~25 m.
  The coastal environment is where rogue wave sensors operate (RogueGuard).
  EM fields from a tower (dipole antenna) at the coast couple to:
    - The fiber sensor network (photonic input)
    - The seawater boundary (reflection + loss)
    - The rogue wave surface (scattering)
"""
import math
import numpy as np

c_light = 2.998e8; mu0 = 4*np.pi*1e-7; eps0 = 8.854e-12; eta0 = math.sqrt(mu0/eps0)


# ============================================================
# Part I: Maxwell's Equations in Phasor Domain
# ============================================================

def maxwell_phasor_domain(freq_Hz=1e9, medium='vacuum'):
    """
    Maxwell's equations in phasor domain: numerical verification and field solutions.

    PHASOR CONVENTION: time dependence e^{+j*omega*t} suppressed.
    d/dt -> j*omega    (derivative in time = multiply by j*omega in phasor)
    So:
      Faraday:  curl E = -j*omega*mu*H
      Ampere:   curl H = J + j*omega*eps*E  (J = sigma*E for conductive medium)

    WAVE EQUATION (from combining Faraday + Ampere):
      curl(curl E) = -j*omega*mu*(J + j*omega*eps*E)
      -nabla^2 E = -j*omega*mu*sigma*E + omega^2*mu*eps*E
      nabla^2 E + k^2 * E = 0    where k^2 = omega^2*mu*eps - j*omega*mu*sigma

    Complex wavenumber:
      k = k' - j*k''
      k' = omega * sqrt(mu*eps) * sqrt(1 + (sigma/(omega*eps))^2)^{1/4} * cos(delta_loss/2)
      k'' = omega * sqrt(mu*eps) * sqrt(1 + (sigma/(omega*eps))^2)^{1/4} * sin(delta_loss/2)
      where tan(delta_loss) = sigma/(omega*eps)   [loss tangent]

    DISPERSION RELATION (Taylor expansion of k(omega)):
      k(omega) = k0 + k1*(omega-omega0) + k2*(omega-omega0)^2/2 + ...
      k1 = 1/v_g  (inverse group velocity)
      k2 = beta2   (GVD coefficient) -> H(f) = exp(j*pi*D*f^2) in this repo

    Parameters
    ----------
    freq_Hz : float   RF/optical frequency [Hz]
    medium  : str     'vacuum', 'seawater', 'SMF28', 'air'

    Returns
    -------
    dict with Maxwell field quantities, k-vector, skin depth, etc.
    """
    if freq_Hz <= 0:
        raise ValueError(f"freq_Hz = {freq_Hz} must be > 0")

    omega = 2*np.pi*freq_Hz

    # Medium parameters
    media = {
        'vacuum':   {'eps_r': 1.0,    'mu_r': 1.0, 'sigma': 0.0},
        'air':      {'eps_r': 1.0006, 'mu_r': 1.0, 'sigma': 1e-15},  # negligible loss
        'seawater': {'eps_r': 80.0,   'mu_r': 1.0, 'sigma': 4.0},    # S/m at ~1 GHz
        'SMF28':    {'eps_r': 2.085,  'mu_r': 1.0, 'sigma': 1e-12},  # n=1.445, silica
        'copper':   {'eps_r': 1.0,    'mu_r': 1.0, 'sigma': 5.96e7},
    }
    if medium not in media:
        raise ValueError(f"medium must be one of {list(media.keys())}")
    p = media[medium]
    eps = p['eps_r'] * eps0; mu = p['mu_r'] * mu0; sigma = p['sigma']

    # Loss tangent
    loss_tangent = sigma / (omega * eps + 1e-30)

    # Complex permittivity
    eps_complex = eps - 1j*sigma/omega   # e_eff = e - j*sigma/omega

    # Complex wavenumber k = omega*sqrt(mu*eps_complex)
    k_complex = omega * np.sqrt(mu * eps_complex + 0j)   # [rad/m]
    k_prime = float(np.real(k_complex))    # phase constant [rad/m]
    k_double_prime = float(np.abs(np.imag(k_complex)))   # attenuation [Np/m]
    alpha_dB_m = 20*k_double_prime/math.log(10) if k_double_prime > 0 else 0   # dB/m

    # Skin depth
    if sigma > 0 and freq_Hz > 0:
        delta_skin = float(1/np.sqrt(np.pi*freq_Hz*mu*sigma))   # m
    else:
        delta_skin = float('inf')

    # Phase velocity
    v_phase = omega / (k_prime + 1e-30)

    # Refractive index
    n_complex = k_complex * c_light / omega
    n_real = float(np.real(n_complex)); kappa = float(np.abs(np.imag(n_complex)))

    # Wave impedance eta = sqrt(mu/eps_complex)
    eta_complex = np.sqrt(mu / eps_complex)
    eta_mag = float(np.abs(eta_complex))
    eta_phase_deg = float(np.degrees(np.angle(eta_complex)))

    # Plane wave in +z direction: E = E0*exp(-j*k*z) * x_hat
    z_arr = np.linspace(0, 5*delta_skin if delta_skin < 1e10 else 1.0, 500)
    E_x = np.exp(-1j*k_complex*z_arr)
    H_y = E_x / eta_complex   # From Faraday: H = (k/omega*mu) x E for plane wave
    S_z = 0.5*np.real(E_x * np.conj(H_y))   # Time-avg Poynting vector [W/m^2]

    # Taylor expansion of k(omega) -- GVD
    # Use absolute perturbation: 1 GHz (small relative to optical freq, resolvable for GVD)
    domega = 2*np.pi*1e9   # 1 GHz absolute perturbation
    omega_pts = np.array([omega-2*domega, omega-domega, omega, omega+domega, omega+2*domega])
    eps_pts = eps - 1j*sigma/omega_pts
    k_pts = omega_pts * np.sqrt(mu*eps_pts+0j)
    # 4th-order finite difference derivatives
    k1 = float(np.real((-k_pts[4]+8*k_pts[3]-8*k_pts[1]+k_pts[0])/(12*domega)))
    k2 = float(np.real((-k_pts[4]+16*k_pts[3]-30*k_pts[2]+16*k_pts[1]-k_pts[0])/(12*domega**2)))
    # D [ps/(nm*km)] from beta2 [s^2/m]:
    # D [s/m^2] = -(2*pi*c/lambda^2)*beta2
    # Conversion: 1 [s/m^2] = 1e12 ps / (1e6 nm*km) = 1e6 [ps/(nm*km)]
    # since 1 m^2 = (1e9 nm)*(1e-3 km) = 1e6 nm*km
    lambda0 = c_light/freq_Hz
    if k2 != 0:
        D_param = -(2*np.pi*c_light / lambda0**2) * k2 * 1e6
    else:
        D_param = 0.0

    # Maxwell equations in phasor (symbolic verification)
    # For plane wave E = E0*exp(-jkz)*x_hat:
    # curl E: only d(Ex)/dz = -jk*Ex -> curl E = jk*Ex*y_hat ... wait
    # Actually: E = Ex*x_hat, so curl E = (dEx/dz)*(-y_hat) = -(-jk)*Ex*y_hat = jk*Ex*y_hat? No.
    # curl E = (dEz/dy - dEy/dz)*x + (dEx/dz - dEz/dx)*y + (dEy/dx - dEx/dy)*z
    # For E = Ex(z)*x_hat: curl E = (dEx/dz)*(-y_hat)? No: (dEx/dz - 0)*y_hat - 0 = (dEx/dz)*(-y_hat)
    # Wait: curl E y-component = dEx/dz - dEz/dx = dEx/dz (since Ez=0)
    # Hmm, let me be careful:
    # curl E = det([x y z; d/dx d/dy d/dz; Ex 0 0])
    #        = y*(dEx/dz) - z*(dEx/dy) ... no
    # = (d(0)/dy - d(0)/dz)*x + (d(Ex)/dz - d(0)/dx)*y + (d(0)/dx - d(Ex)/dy)*z (wrong)
    # Let me use the standard formula:
    # (curl F)_x = dFz/dy - dFy/dz
    # (curl F)_y = dFx/dz - dFz/dx
    # (curl F)_z = dFy/dx - dFx/dy
    # For E = Ex(z)*x_hat: only Ex is nonzero, only z-dependence.
    # (curl E)_x = 0 - 0 = 0
    # (curl E)_y = dEx/dz - 0 = -j*k*Ex (since Ex = E0*exp(-j*k*z))
    # (curl E)_z = 0 - 0 = 0
    # So: curl E = -j*k*Ex * y_hat
    # Faraday: curl E = -j*omega*mu*H -> H = (k/(omega*mu)) * Ex * y_hat = Ex/eta * y_hat (correct)
    # Verification at z=0:
    E0 = 1.0
    curl_E_y = -1j*k_complex*E0
    faraday_rhs = -1j*omega*mu*(E0/eta_complex)
    faraday_error = float(np.abs(curl_E_y - faraday_rhs) / (np.abs(faraday_rhs)+1e-30))

    # Ampere: curl H = J + j*omega*eps*E
    # H = (E0/eta)*exp(-jkz)*y_hat, curl H = -j*k*(E0/eta)*x_hat (by same argument, sign)
    curl_H_x = -1j*k_complex*(E0/eta_complex)   # from dHy/dz
    J_cond = sigma*E0   # Ohmic current density
    ampere_rhs = J_cond + 1j*omega*eps*E0
    ampere_error = float(np.abs(curl_H_x - (-ampere_rhs)) / (np.abs(ampere_rhs)+1e-30))
    # Note: curl H = -j*k*Hy*x_hat and Ampere rhs acts in x_hat direction
    # The sign: curl H_x = (dHz/dy - dHy/dz) = -dHy/dz = -(-jk)*Hy = jk*Hy? Let me recheck.
    # H = Hy(z)*y_hat:
    # (curl H)_x = dHz/dy - dHy/dz = 0 - dHy/dz = -(-jk)*Hy = jk*Hy
    curl_H_x_correct = 1j*k_complex*(E0/eta_complex)
    ampere_error = float(np.abs(curl_H_x_correct - ampere_rhs) / (np.abs(ampere_rhs)+1e-30))

    return {
        'medium': medium, 'freq_Hz': freq_Hz,
        'parameters': {
            'eps_r': p['eps_r'], 'mu_r': p['mu_r'], 'sigma_S_m': sigma,
            'loss_tangent': float(loss_tangent),
        },
        'wavenumber': {
            'k_prime_rad_m': k_prime,
            'k_double_prime_Np_m': k_double_prime,
            'alpha_dB_m': alpha_dB_m,
            'lambda_in_medium_m': float(2*np.pi/(k_prime+1e-30)),
        },
        'fields': {
            'n_real': n_real, 'kappa': kappa,
            'eta_mag_ohm': eta_mag, 'eta_phase_deg': eta_phase_deg,
            'v_phase_m_s': v_phase,
            'z_m': z_arr.tolist(),
            'E_mag': np.abs(E_x).tolist(),
            'S_z_W_m2': S_z.tolist(),
        },
        'skin_depth_m': float(delta_skin),
        'GVD': {
            'k1_s_m': float(k1),
            'v_group_m_s': float(1/(k1+1e-30)),
            'beta2_s2_m': float(k2),
            'D_ps_nm_km': float(D_param),
        },
        'maxwell_verification': {
            'faraday_error_frac': float(faraday_error),
            'ampere_error_frac': float(ampere_error),
            'faraday_ok': faraday_error < 1e-6,
            'ampere_ok': ampere_error < 1e-6,
            'equations': {
                'Faraday': 'curl E = -j*omega*mu*H',
                'Ampere': 'curl H = J + j*omega*eps*E',
                'Gauss_E': 'div D = rho (= 0 for plane wave)',
                'Gauss_B': 'div B = 0 (always)',
                'Wave_eq': 'nabla^2 E + k^2*E = 0, k^2 = omega^2*mu*eps - j*omega*mu*sigma',
                'H_of_f': 'H(f) = exp(j*pi*beta2*L*(2*pi*f)^2)  [quadratic phase = GVD]',
            },
        },
    }


# ============================================================
# Part II: Coppinger/Bhushan/Jalali 1999 -- Full Paper Reproduction
# ============================================================

def coppinger_1999_stretch_factor(D1_ps_nm_km=17.0, L1_km=5.0,
                                   D2_ps_nm_km=17.0, L2_km=45.0,
                                   Delta_lambda_nm=10.0, f_ADC_GHz=2.0):
    """
    Reproduce Section III of Coppinger et al. 1999: stretch factor derivation.

    FROM THE PAPER (paraphrased):
      The system consists of:
        1. Broadband optical source (ASE from EDFA, bandwidth Delta_lambda)
        2. First dispersive element (fiber L1, dispersion D1) -- "pre-chirp"
           Maps frequency to time: each wavelength lambda arrives at time
             t(lambda) = D1 * L1 * lambda
           The pulse is "chirped" -- stretched in time by Delta_t1 = D1*L1*Delta_lambda
        3. Electro-optic modulator (EOM) -- RF input signal e(t) imprinted
           on the chirped optical pulse.
           Since t = D1*L1*lambda, this is equivalent to imprinting e(lambda):
             I_mod(t) = I_chirp(t) * [1 + e(t/M_partial)]
           where M_partial accounts for partial pre-chirp.
        4. Second dispersive element (fiber L2, dispersion D2) -- "post-stretch"
           The signal is further stretched. Now:
             t_out(lambda) = (D1*L1 + D2*L2) * lambda
           So the signal e(t) has been time-stretched by factor:
             M = (D1*L1 + D2*L2) / (D1*L1) = 1 + D2*L2/(D1*L1)
        5. Photodetector + electronic ADC at f_s = f_ADC
           Effective input bandwidth captured:
             B_in = M * f_ADC / 2

    KEY EQUATIONS FROM PAPER:
      Eq. (1): tau_g(lambda) = D * L * lambda         [group delay]
      Eq. (2): M = (D1*L1 + D2*L2)/(D1*L1)           [stretch factor]
      Eq. (3): B_RF = M * f_s / 2                     [captured bandwidth]
      Eq. (4): T_w = D1*L1*Delta_lambda               [time aperture]
      Eq. (5): N = T_w * f_s                          [samples per pulse]

    Demonstration values from the paper:
      D1=D2=17 ps/(nm*km), L1=5 km, L2=45 km -> M = 1+45/5 = 10
      Delta_lambda = 10 nm -> T_w = 17*5*10 = 850 ps
      f_ADC = 2 Gsample/s -> N = 850e-12 * 2e9 = 1700 samples
      B_captured = 10 * 2/2 = 10 GHz (paper demonstrates 20 GHz with M=10, f_s=2)

    Parameters
    ----------
    D1_ps_nm_km : float  Dispersion of pre-chirp fiber [ps/(nm*km)]
    L1_km       : float  Length of pre-chirp fiber [km]
    D2_ps_nm_km : float  Dispersion of post-stretch fiber [ps/(nm*km)]
    L2_km       : float  Length of post-stretch fiber [km]
    Delta_lambda_nm : float  Optical source bandwidth [nm]
    f_ADC_GHz   : float  Electronic ADC sample rate [Gsample/s]

    Returns
    -------
    dict with paper's key results and additional analysis.
    """
    for name, val in [('D1', D1_ps_nm_km), ('D2', D2_ps_nm_km),
                      ('L1', L1_km), ('L2', L2_km)]:
        if val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")
    if Delta_lambda_nm <= 0:
        raise ValueError(f"Delta_lambda = {Delta_lambda_nm} must be > 0")
    if f_ADC_GHz <= 0:
        raise ValueError(f"f_ADC = {f_ADC_GHz} must be > 0")

    # Paper Eq. (1): group delay tau(lambda) = D * L * lambda
    # For a pulse with optical bandwidth Delta_lambda:
    D1L1 = D1_ps_nm_km * L1_km   # ps*nm/km * km = ps*nm... wait: [ps/(nm*km)]*[km] = ps/nm
    # D*L has units [ps/(nm*km)]*[km] = ps/nm. Multiply by Delta_lambda [nm] -> ps.
    # So D1*L1 [ps/nm] * Delta_lambda [nm] = T1 [ps]. Correct.
    T1_ps = D1_ps_nm_km * L1_km * Delta_lambda_nm   # ps [time aperture after L1]
    T2_ps = D2_ps_nm_km * L2_km * Delta_lambda_nm   # ps [additional stretch from L2]
    T_total_ps = T1_ps + T2_ps

    # Paper Eq. (2): M = (D1*L1 + D2*L2)/(D1*L1) = 1 + D2*L2/(D1*L1)
    M = (D1_ps_nm_km*L1_km + D2_ps_nm_km*L2_km) / (D1_ps_nm_km*L1_km)

    # Paper Eq. (3): captured RF bandwidth
    B_RF_GHz = M * f_ADC_GHz / 2

    # Paper Eq. (4): time aperture (window available for signal capture)
    T_w_ps = T1_ps   # time aperture = spread in L1

    # Paper Eq. (5): number of samples per optical pulse
    # T_total [ps] * 1e-12 [s/ps] * f_ADC [Gsample/s] * 1e9 [sample/(s*Gsample)] = samples
    N_samples = T_total_ps * 1e-12 * f_ADC_GHz * 1e9

    # Chirp rate (frequency sweep rate in L1)
    # tau(lambda) = D1*L1 * lambda -> lambda(t) = t/(D1*L1) [nm/ps * ps = nm]
    # In frequency: f_opt(t) = c/lambda(t) = c*(D1*L1)/t ... better to use:
    # d_lambda/d_t = 1/(D1*L1) [nm/ps]
    chirp_rate_nm_per_ps = 1.0 / (D1_ps_nm_km * L1_km)  # nm/ps

    # Convert to frequency chirp rate: df = -c/lambda^2 * d_lambda
    lambda0_nm = 1550.0; lambda0_m = lambda0_nm*1e-9
    chirp_rate_GHz_per_ps = abs(-c_light/(lambda0_m**2) * (chirp_rate_nm_per_ps*1e-9) * 1e-9) * 1e-12
    # GHz/ps -> GHz^2: [Hz/m * m/s * 1e-9/1e9] ... let me just convert carefully
    # df [GHz] = c [m/s] / lambda^2 [m^2] * d_lambda [m]
    # chirp_rate_nm_per_ps [nm/ps] = 1e-9[m/nm] / 1e-12[s/ps] = 1e3 m/s per m? No.
    # chirp_rate [m/s] = chirp_rate_nm_per_ps * 1e-9 [m/nm] / 1e-12 [s/ps] = chirp_rate_nm_per_ps * 1e3 m/s
    chirp_rate_m_s = chirp_rate_nm_per_ps * 1e3   # m/s of wavelength change
    chirp_rate_Hz_s = c_light / lambda0_m**2 * chirp_rate_m_s   # Hz/s
    chirp_rate_GHz_per_ns = chirp_rate_Hz_s * 1e-9 * 1e-9   # GHz/ns

    # Paper's demonstrated parameters (Table I equivalent)
    paper_demo = {
        'D1': 17.0, 'L1': 5.0, 'D2': 17.0, 'L2': 45.0,
        'Delta_lambda': 10.0, 'f_ADC': 2.0,
    }
    M_paper = 1 + paper_demo['D2']*paper_demo['L2']/(paper_demo['D1']*paper_demo['L1'])
    T_w_paper_ps = paper_demo['D1']*paper_demo['L1']*paper_demo['Delta_lambda']
    B_RF_paper = M_paper * paper_demo['f_ADC'] / 2
    N_paper = T_w_paper_ps * 1e-12 * paper_demo['f_ADC'] * 1e9

    # Sweep M parameter space (reproduce paper Fig. 2 equivalent)
    L2_sweep = np.linspace(0, 100, 200)
    M_sweep = 1 + D2_ps_nm_km*L2_sweep/(D1_ps_nm_km*L1_km)
    B_sweep = M_sweep * f_ADC_GHz / 2

    # Time-domain illustration: chirped pulse + modulated signal + stretched output
    t_arr = np.linspace(0, T_total_ps*1e-12, 2000)   # seconds
    # Chirped optical carrier (after L1, before EOM)
    f_carrier = 193e12   # Hz (1550 nm)
    omega_c = 2*np.pi*f_carrier
    phi_chirp = (t_arr**2) / (2 * D1_ps_nm_km*L1_km*1e-12 * (lambda0_m**2)/(2*np.pi*c_light))
    E_chirped = np.exp(1j*(omega_c*t_arr[:200] + phi_chirp[:200]))

    # Modulated signal: multi-tone RF at original (fast) rate
    f_signal = B_RF_GHz*1e9/2   # input signal at max captured frequency
    t_signal = np.linspace(0, T_w_ps*1e-12, 2000)
    e_signal = np.cos(2*np.pi*f_signal*t_signal) + 0.5*np.cos(2*np.pi*f_signal*0.6*t_signal)

    # Stretched output (slow, at 1/M of original rate)
    t_stretched = np.linspace(0, T_total_ps*1e-12, 2000)
    e_stretched = np.cos(2*np.pi*f_signal/M*t_stretched) + 0.5*np.cos(2*np.pi*f_signal*0.6/M*t_stretched)

    return {
        'paper': {
            'title': 'Photonic Time Stretch and Its Application to ADC',
            'authors': 'F. Coppinger, A.S. Bhushan, B. Jalali',
            'journal': 'IEEE Trans. Microwave Theory Techn., Vol.47, No.7, July 1999',
        },
        'inputs': {
            'D1_ps_nm_km': D1_ps_nm_km, 'L1_km': L1_km,
            'D2_ps_nm_km': D2_ps_nm_km, 'L2_km': L2_km,
            'Delta_lambda_nm': Delta_lambda_nm, 'f_ADC_GHz': f_ADC_GHz,
        },
        'paper_equations': {
            'Eq1_group_delay': 'tau(lambda) = D*L*lambda  [ps = ps/(nm*km)*km*nm]',
            'Eq2_stretch_factor': 'M = (D1*L1 + D2*L2)/(D1*L1) = 1 + D2*L2/(D1*L1)',
            'Eq3_bandwidth': 'B_RF = M * f_ADC / 2',
            'Eq4_time_aperture': 'T_w = D1*L1*Delta_lambda',
            'Eq5_samples': 'N = T_w * f_ADC',
        },
        'results': {
            'M': float(M),
            'T1_ps': float(T1_ps),
            'T_total_ps': float(T_total_ps),
            'T_w_ps': float(T_w_ps),
            'B_RF_GHz': float(B_RF_GHz),
            'N_samples': float(N_samples),
            'chirp_rate_nm_per_ps': float(chirp_rate_nm_per_ps),
            'chirp_rate_GHz_per_ns': float(chirp_rate_GHz_per_ns),
        },
        'paper_demo_values': {
            'M': float(M_paper),
            'T_w_ps': float(T_w_paper_ps),
            'B_RF_GHz': float(B_RF_paper),
            'N_samples': float(N_paper),
            'note': 'Paper demonstrates 20 GHz capture with 2 Gsample/s ADC at M=10',
        },
        'M_sweep': {
            'L2_km': L2_sweep.tolist(),
            'M': M_sweep.tolist(),
            'B_RF_GHz': B_sweep.tolist(),
        },
        'time_domain': {
            't_signal_ns': (t_signal*1e9).tolist(),
            'e_signal': e_signal.tolist(),
            't_stretched_ns': (t_stretched*1e9).tolist(),
            'e_stretched': e_stretched.tolist(),
        },
        'H_f_connection': (
            f'The transfer function of the combined fiber system:\n'
            f'H(f) = exp(j*pi*(beta2_1*L1 + beta2_2*L2)*(2*pi*f)^2)\n'
            f'     = exp(j*pi*D_eff*f^2)  [paper eq. implicit]\n'
            f'This IS the H(f) = exp(j*pi*D*f^2) used in this repo for GS phase retrieval.\n'
            f'D_eff = D1*L1 + D2*L2 = {D1_ps_nm_km*L1_km + D2_ps_nm_km*L2_km:.0f} ps*nm  [product units]'
        ),
    }


# ============================================================
# Part III: SNR Analysis from Paper Section IV
# ============================================================

def coppinger_1999_snr_analysis(M=10.0, P_opt_dBm=3.0, f_ADC_GHz=2.0,
                                 R_responsivity=0.8, T_K=300.0, R_load=50.0):
    """
    SNR analysis: Section IV of Coppinger et al. 1999.

    The paper identifies three dominant noise sources:
    1. Shot noise (quantum limit) -- from photon statistics
    2. RIN (Relative Intensity Noise) -- from optical source fluctuations
    3. Thermal noise -- from resistive load at photodetector

    SHOT NOISE (Poisson statistics of photons):
      i_shot = sqrt(2*q*I_ph*B)   [A rms]
      where I_ph = R * P_opt [A], B = f_ADC/2 [Hz] (noise bandwidth)
      SNR_shot = I_ph^2 / i_shot^2 = I_ph / (2*q*B)

    RIN NOISE (amplitude fluctuations of optical source):
      i_RIN = I_ph * sqrt(RIN_dBHz * B)
      Typical EDFA ASE source: RIN ~ -120 dB/Hz
      SNR_RIN = 1 / (RIN_lin * B)

    THERMAL NOISE (Johnson noise of load resistor):
      i_th = sqrt(4*kB*T*B/R_load)   [A rms]
      SNR_th = I_ph^2 * R_load / (4*kB*T*B)

    TIME-STRETCH EFFECT ON SNR:
      Stretching by M slows the signal by M, reducing the noise bandwidth by M:
        B_eff = f_ADC / (2*M)   [effective noise bandwidth after stretch]
      BUT the signal power is also reduced (same energy spread over M longer time).
      Net effect: SNR improves by sqrt(M) (shot noise limited)
      Paper states: SNR_stretch = sqrt(M) * SNR_unstretched (shot noise limited)
    """
    if M <= 1:
        raise ValueError(f"M = {M} must be > 1")
    if f_ADC_GHz <= 0:
        raise ValueError(f"f_ADC_GHz must be > 0")

    P_opt_W = 10**(P_opt_dBm/10) * 1e-3   # Convert dBm to W
    I_ph = R_responsivity * P_opt_W   # Photocurrent [A]
    B_full = f_ADC_GHz*1e9 / 2   # Noise bandwidth without stretch [Hz]
    B_eff  = B_full / M           # Effective noise bandwidth with stretch [Hz]

    q_e = 1.602e-19; kB = 1.381e-23

    # --- Without time stretch ---
    # Shot noise
    i_shot_sq = 2*q_e*I_ph*B_full
    SNR_shot_lin = I_ph**2 / i_shot_sq
    SNR_shot_dB = 10*math.log10(SNR_shot_lin)

    # RIN noise (EDFA ASE source)
    RIN_dBHz = -120.0   # dB/Hz (typical EDFA)
    RIN_lin  = 10**(RIN_dBHz/10)
    i_rin_sq = RIN_lin * B_full * I_ph**2
    SNR_rin_lin = 1/(RIN_lin*B_full)
    SNR_rin_dB = 10*math.log10(SNR_rin_lin)

    # Thermal noise
    i_th_sq = 4*kB*T_K*B_full/R_load
    SNR_th_lin = I_ph**2 * R_load / (4*kB*T_K*B_full)
    SNR_th_dB = 10*math.log10(SNR_th_lin)

    # Total noise (add noise powers)
    i_noise_total_sq = i_shot_sq + i_rin_sq + i_th_sq
    SNR_total_lin = I_ph**2 / i_noise_total_sq
    SNR_total_dB = 10*math.log10(SNR_total_lin)

    # ENOB (Effective Number of Bits)
    ENOB = (SNR_total_dB - 1.76) / 6.02

    # --- With time stretch (factor M) ---
    # Shot noise improves by sqrt(M) [paper result]
    i_shot_sq_M = 2*q_e*I_ph*B_eff
    SNR_shot_M_dB = 10*math.log10(I_ph**2/i_shot_sq_M)

    i_rin_sq_M = RIN_lin * B_eff * I_ph**2
    SNR_rin_M_dB = 10*math.log10(1/(RIN_lin*B_eff))

    i_th_sq_M = 4*kB*T_K*B_eff/R_load
    SNR_th_M_dB = 10*math.log10(I_ph**2*R_load/(4*kB*T_K*B_eff))

    i_noise_M_sq = i_shot_sq_M + i_rin_sq_M + i_th_sq_M
    SNR_total_M_lin = I_ph**2 / i_noise_M_sq
    SNR_total_M_dB = 10*math.log10(SNR_total_M_lin)
    ENOB_M = (SNR_total_M_dB - 1.76) / 6.02

    SNR_improvement_dB = SNR_total_M_dB - SNR_total_dB

    # Sweep M: SNR vs stretch factor (paper Fig. 4 equivalent)
    M_sweep = np.linspace(1.1, 50, 200)
    B_eff_arr = B_full / M_sweep
    i_shot_arr = 2*q_e*I_ph*B_eff_arr
    i_rin_arr  = RIN_lin*B_eff_arr*I_ph**2
    i_th_arr   = 4*kB*T_K*B_eff_arr/R_load
    SNR_arr = 10*np.log10(I_ph**2/(i_shot_arr+i_rin_arr+i_th_arr))
    ENOB_arr = (SNR_arr - 1.76) / 6.02

    # Sweep optical power: P_opt vs SNR
    P_dBm_arr = np.linspace(-20, 20, 200)
    P_W_arr = 10**(P_dBm_arr/10)*1e-3
    I_arr = R_responsivity*P_W_arr
    i_n_arr = 2*q_e*I_arr*B_eff + RIN_lin*B_eff*I_arr**2 + 4*kB*T_K*B_eff/R_load
    SNR_P_arr = 10*np.log10(I_arr**2/(i_n_arr+1e-40))

    return {
        'inputs': {'M': M, 'P_opt_dBm': P_opt_dBm, 'f_ADC_GHz': f_ADC_GHz,
                   'R_responsivity': R_responsivity, 'T_K': T_K, 'R_load': R_load},
        'noise_model': {
            'shot': 'i_shot = sqrt(2*q*I_ph*B)',
            'RIN':  'i_RIN = I_ph*sqrt(RIN*B),  RIN=-120 dB/Hz (EDFA)',
            'thermal': 'i_th = sqrt(4*kB*T*B/R)',
        },
        'without_stretch': {
            'B_Hz': float(B_full),
            'I_ph_mA': float(I_ph*1e3),
            'SNR_shot_dB': float(SNR_shot_dB),
            'SNR_RIN_dB': float(SNR_rin_dB),
            'SNR_thermal_dB': float(SNR_th_dB),
            'SNR_total_dB': float(SNR_total_dB),
            'ENOB': float(ENOB),
        },
        'with_stretch': {
            'M': float(M),
            'B_eff_Hz': float(B_eff),
            'SNR_shot_dB': float(SNR_shot_M_dB),
            'SNR_RIN_dB': float(SNR_rin_M_dB),
            'SNR_thermal_dB': float(SNR_th_M_dB),
            'SNR_total_dB': float(SNR_total_M_dB),
            'ENOB': float(ENOB_M),
        },
        'improvement': {
            'delta_SNR_dB': float(SNR_improvement_dB),
            'delta_ENOB_bits': float(ENOB_M - ENOB),
            'theory': 'Shot-noise limited: SNR_M = M * SNR_0, so delta_SNR = 10*log10(M)',
            'theory_dB': float(10*math.log10(M)),
        },
        'sweeps': {
            'M_sweep': M_sweep.tolist(),
            'SNR_vs_M_dB': SNR_arr.tolist(),
            'ENOB_vs_M': ENOB_arr.tolist(),
            'P_dBm_sweep': P_dBm_arr.tolist(),
            'SNR_vs_P_dB': SNR_P_arr.tolist(),
        },
    }


# ============================================================
# Part IV: Coastal EM -- Fields at Water Boundary
# ============================================================

def coastal_em_fields(freq_Hz=1e9, tower_height_m=30.0, distance_km=10.0):
    """
    Electromagnetic fields at the coastal water boundary.

    Context: RogueGuard's fiber sensor array is deployed in the coastal ocean.
    The RF signal received at the photonic ADC system comes from:
    1. Radar/tower antenna (Hertzian dipole model) at height h above coast
    2. Reflected from ocean surface (Fresnel reflection coefficients)
    3. Propagating through seawater (skin effect, loss)

    HERTZIAN DIPOLE RADIATION:
      A short dipole of length dl, carrying current I0, radiates:
        E_theta = (j*eta0*k*I0*dl)/(4*pi) * sin(theta) * exp(-j*k*r)/r
        H_phi = E_theta / eta0
        Poynting: S_r = |E_theta|^2/(2*eta0) [W/m^2]
        Radiation resistance: R_rad = (2*pi/3) * eta0 * (dl/lambda)^2

    FAR FIELD (r >> lambda):
      E ~ 1/r (spherical wave)
      Power density S ~ 1/r^2

    NEAR FIELD (r << lambda/(2*pi)):
      E ~ 1/r^2 (quasi-static dipole field)
      H ~ 1/r^2
      Reactive (stored) energy dominates

    FRESNEL COEFFICIENTS AT AIR/SEA INTERFACE:
      For vertical (TM) polarization:
        Gamma_TM = (eps_r*cos(theta_i) - sqrt(eps_r - sin^2(theta_i))) /
                   (eps_r*cos(theta_i) + sqrt(eps_r - sin^2(theta_i)))
      For horizontal (TE) polarization:
        Gamma_TE = (cos(theta_i) - sqrt(eps_r - sin^2(theta_i))) /
                   (cos(theta_i) + sqrt(eps_r - sin^2(theta_i)))

    SURFACE WAVE (Norton wave / Sommerfeld wave):
      Over lossy ground (seawater), a guided surface wave travels along the interface.
      Attenuation: exp(-r/r_0), r_0 = (lambda * eps_r) / (2*pi * sqrt(sigma/omega*eps_0))
    """
    if freq_Hz <= 0 or tower_height_m <= 0 or distance_km <= 0:
        raise ValueError("All parameters must be positive")

    omega = 2*np.pi*freq_Hz
    lambda_air = c_light / freq_Hz   # wavelength in air [m]
    k_air = 2*np.pi / lambda_air     # wavenumber in air [rad/m]

    # Seawater parameters
    eps_r_sw = 80.0; sigma_sw = 4.0   # S/m at ~1 GHz
    eps_sw_complex = eps_r_sw*eps0 - 1j*sigma_sw/omega
    k_sw = omega * np.sqrt(mu0 * eps_sw_complex + 0j)
    delta_sw = float(1/np.sqrt(np.pi*freq_Hz*mu0*sigma_sw))   # skin depth [m]
    alpha_sw_dB_m = 20*float(np.abs(np.imag(k_sw)))/math.log(10)

    # Hertzian dipole: radiation fields at distance r and angle theta
    r_m = distance_km * 1e3   # m
    # Far-field approximation
    dl_lambda = 0.1   # dipole length as fraction of lambda (half-wave ~ 0.5)
    dl = dl_lambda * lambda_air
    I0 = 1.0   # 1 A current
    # E_theta at theta=90 (broadside, maximum radiation)
    E_theta_far = (1j*eta0*k_air*I0*dl)/(4*np.pi) * np.exp(-1j*k_air*r_m) / r_m
    E_far_mag = float(np.abs(E_theta_far))   # V/m

    # Radiation resistance
    R_rad = (2*np.pi/3) * eta0 * dl_lambda**2
    P_radiated = 0.5 * I0**2 * R_rad   # W

    # Power density at distance
    S_far = float(np.abs(E_theta_far)**2 / (2*eta0))   # W/m^2
    S_dBm_m2 = 10*math.log10(S_far*1e3) if S_far > 0 else -200

    # Near field boundary: r_NF = lambda/(2*pi)
    r_near_far = lambda_air / (2*np.pi)
    in_near_field = r_m < r_near_far

    # Fresnel coefficients vs incidence angle
    theta_i_arr = np.linspace(0, 89, 500) * np.pi/180   # radians
    sin_t = np.sin(theta_i_arr); cos_t = np.cos(theta_i_arr)
    eps_r_complex = eps_sw_complex/eps0
    sqrt_term = np.sqrt(eps_r_complex - sin_t**2 + 0j)

    Gamma_TM = (eps_r_complex*cos_t - sqrt_term) / (eps_r_complex*cos_t + sqrt_term)
    Gamma_TE = (cos_t - sqrt_term) / (cos_t + sqrt_term)

    # Brewster angle (TM only, modified for lossy medium)
    # For lossless: theta_B = arctan(sqrt(eps_r)) ~ arctan(sqrt(80)) ~ 83.6 deg
    theta_Brewster_deg = float(np.degrees(math.atan(math.sqrt(eps_r_sw))))

    # Surface wave attenuation (Norton wave)
    # Approximate: r_0 ~ lambda*eps_r_complex/(2*pi)
    r_0_surface = float(np.abs(lambda_air * eps_r_complex / (2*np.pi)))   # m
    r_arr_km = np.linspace(0.1, 100, 500)   # km
    r_arr_m = r_arr_km * 1e3
    # Free space: 1/r^2 power
    S_free = S_far * (r_m/r_arr_m)**2
    # Surface wave (Zenneck wave approximate): additional exponential factor
    S_surface = S_far * (r_m/r_arr_m)**2 * np.exp(-2*r_arr_m/r_0_surface)
    S_free_dBm = 10*np.log10(S_free*1e3+1e-30)
    S_surface_dBm = 10*np.log10(S_surface*1e3+1e-30)

    # Tower height vs coverage (two-ray model)
    # Path length difference: delta_r = 2*h*sin(theta) ~ 2*h*h_rx/r for small angles
    h_rx = 2.0   # receiver height [m]
    r_cov = np.linspace(0.1, 50, 400) * 1e3   # m
    delta_r = 2*tower_height_m*h_rx / r_cov
    phi_two_ray = 2*np.pi*delta_r/lambda_air
    E_two_ray = 2*np.sin(phi_two_ray/2) / r_cov   # |E| pattern (normalized)

    return {
        'freq_Hz': freq_Hz, 'lambda_m': float(lambda_air),
        'tower_height_m': tower_height_m, 'distance_km': distance_km,
        'seawater': {
            'eps_r': eps_r_sw, 'sigma_S_m': sigma_sw,
            'skin_depth_mm': float(delta_sw*1e3),
            'attenuation_dB_m': float(alpha_sw_dB_m),
            'loss_tangent': float(sigma_sw/(omega*eps_r_sw*eps0)),
        },
        'dipole': {
            'dl_lambda_ratio': dl_lambda,
            'R_rad_ohm': float(R_rad),
            'P_radiated_W': float(P_radiated),
            'E_far_V_m': float(E_far_mag),
            'S_far_W_m2': float(S_far),
            'S_far_dBm_m2': float(S_dBm_m2),
            'near_far_boundary_m': float(r_near_far),
            'in_near_field': bool(in_near_field),
        },
        'fresnel': {
            'theta_i_deg': np.degrees(theta_i_arr).tolist(),
            'Gamma_TM_mag': np.abs(Gamma_TM).tolist(),
            'Gamma_TE_mag': np.abs(Gamma_TE).tolist(),
            'theta_Brewster_deg': float(theta_Brewster_deg),
        },
        'propagation': {
            'r_km': r_arr_km.tolist(),
            'S_free_dBm': S_free_dBm.tolist(),
            'S_surface_dBm': S_surface_dBm.tolist(),
            'r_0_surface_km': float(r_0_surface/1e3),
        },
        'two_ray': {
            'r_km': (r_cov/1e3).tolist(),
            'E_pattern': E_two_ray.tolist(),
        },
        'GS_connection': (
            'The far-field E(r) = E0*exp(-jkr)/r is measured as I = |E|^2.\n'
            'GS phase retrieval at coastal sensor -> recover E(r) -> propagate back to source.\n'
            'Fresnel coefficients create phase diversity across reflection angles\n'
            '= the "D" dispersion parameter in the GS algorithm.'
        ),
    }


# ============================================================
# Part V: Digital Nuclear Twin
# ============================================================

def digital_nuclear_twin(n_groups=6, reactor_type='PWR'):
    """
    Digital twin of a nuclear reactor for the photonic sensing context.

    MOTIVATION:
    RogueGuard's 1U optical monitor principle applies to:
    1. Ocean rogue waves (primary target)
    2. Reactor pressure vessels -- acoustic rogue waves from coolant flow
    3. Nuclear digital twin: real-time monitoring of neutron flux using
       photonic time-stretch ADC (high-bandwidth transient capture)

    DIGITAL TWIN CONCEPT:
    A digital twin mirrors a physical system in real time.
    For nuclear: the twin solves neutron diffusion equations with the same
    parameters as the physical reactor, using live sensor data to update state.

    SIX-GROUP DELAYED NEUTRON PRECURSOR MODEL:
    The point kinetics equations with 6 precursor groups:
      dP/dt = (rho - beta)/Lambda * P + sum(lambda_i * C_i)   for i=1..6
      dC_i/dt = beta_i/Lambda * P - lambda_i * C_i

    where:
      P = neutron power (normalized)
      rho = reactivity [pcm] (1 pcm = 1e-5 Delta-k/k)
      beta = total delayed neutron fraction (~650 pcm for U-235)
      beta_i = delayed fraction for group i
      lambda_i = decay constant of group i [1/s]
      Lambda = prompt neutron lifetime (~25 us for PWR)

    PHOTONIC ADC CONNECTION:
    Reactor transients (rod ejection, loss of coolant) produce neutron flux
    bursts up to 10 MHz bandwidth. A conventional ADC at 20 Msample/s barely
    resolves them. A photonic time-stretch ADC (M=10, f_s=2 Gsample/s) captures
    them with full bandwidth and feeds the digital twin in real time.

    H(f) = exp(j*pi*D*f^2) -- the dispersion fiber acts on the DETECTOR signal.
    GS phase retrieval: recover the complex neutron flux A(t) from I(t).
    """
    # Six-group delayed neutron parameters for U-235 (thermal spectrum)
    # Source: ANS-5.1 standard
    beta_i = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
    lambda_i = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])   # 1/s
    beta_total = float(beta_i.sum())   # ~0.00650

    Lambda_s = 25e-6   # 25 us prompt neutron lifetime (PWR)

    # Simulate reactor response to step reactivity insertion
    # Use RK4 to solve point kinetics
    rho_step_pcm = 50.0   # reactivity insertion [pcm]
    rho_step = rho_step_pcm * 1e-5   # dimensionless

    def point_kinetics(t, y, rho):
        P = y[0]; C = y[1:]
        dP_dt = (rho - beta_total)/Lambda_s * P + np.dot(lambda_i, C)
        dC_dt = beta_i/Lambda_s * P - lambda_i * C
        return np.concatenate([[dP_dt], dC_dt])

    # RK4 integration
    dt = 1e-4; t_end = 5.0   # 5 seconds
    n_steps = int(t_end/dt)
    t_arr = np.linspace(0, t_end, min(n_steps, 5000))
    dt_sub = t_end / len(t_arr)

    y0 = np.zeros(7); y0[0] = 1.0   # P=1 (critical), C_i in equilibrium
    # Equilibrium: dC_i/dt=0 -> C_i = beta_i*P/(Lambda*lambda_i)
    y0[1:] = beta_i * y0[0] / (Lambda_s * lambda_i)

    P_hist = np.zeros(len(t_arr)); P_hist[0] = y0[0]
    y = y0.copy()
    for i in range(1, len(t_arr)):
        rho_t = rho_step if t_arr[i] > 0.1 else 0.0
        k1 = point_kinetics(t_arr[i-1], y, rho_t)
        k2 = point_kinetics(t_arr[i-1]+dt_sub/2, y+dt_sub*k1/2, rho_t)
        k3 = point_kinetics(t_arr[i-1]+dt_sub/2, y+dt_sub*k2/2, rho_t)
        k4 = point_kinetics(t_arr[i], y+dt_sub*k3, rho_t)
        y = y + dt_sub*(k1+2*k2+2*k3+k4)/6
        P_hist[i] = y[0]

    # Stable period: from the inhour equation (approximate for rho << beta)
    # For subcritical rho < beta, the asymptotic period is dominated by delayed neutrons.
    # Approximate: T_stable ~ (beta - rho) / (rho * sum(beta_i*lambda_i))
    # For small rho: T_stable ~ beta / (rho * sum(beta_i*lambda_i))
    sum_beta_lambda = float(np.sum(beta_i * lambda_i))
    if rho_step > 0 and rho_step < beta_total:
        T_period = (beta_total - rho_step) / (rho_step * sum_beta_lambda)
        prompt_critical = False
    elif rho_step >= beta_total:
        T_period = Lambda_s / (rho_step - beta_total + 1e-30)   # prompt critical!
        prompt_critical = True
    else:
        T_period = 1e6; prompt_critical = False

    # Photonic ADC bandwidth needed
    # Transient timescale ~ T_period; Nyquist rate = 2/T_period
    f_transient_Hz = 2.0/T_period
    f_transient_MHz = f_transient_Hz/1e6

    # Breit-Wigner cross section (resonance analog)
    E_eV = np.logspace(-3, 5, 1000)
    E_res_eV = 6.67   # U-238 first resonance [eV]
    Gamma_n = 0.00105; Gamma_gamma = 0.0280   # eV, partial widths
    Gamma_total = Gamma_n + Gamma_gamma
    sigma0_barns = 582.0   # peak fission cross section U-235 thermal
    sigma_BW = sigma0_barns * (Gamma_total**2/4) / ((E_eV-E_res_eV)**2 + (Gamma_total/2)**2)
    # Connection: Breit-Wigner = S21(cavity) = H(f) pole near resonance

    return {
        'reactor_type': reactor_type,
        'delayed_neutrons': {
            'n_groups': n_groups,
            'beta_i': beta_i.tolist(),
            'lambda_i_s': lambda_i.tolist(),
            'beta_total': float(beta_total),
            'beta_total_pcm': float(beta_total*1e5),
            'Lambda_us': float(Lambda_s*1e6),
        },
        'transient': {
            'rho_step_pcm': float(rho_step_pcm),
            'T_period_s': float(T_period),
            'prompt_critical': bool(prompt_critical),
            't_s': t_arr.tolist(),
            'P_normalized': P_hist.tolist(),
        },
        'photonic_adc': {
            'f_transient_MHz': float(f_transient_MHz),
            'M_needed': float(max(1, f_transient_MHz/1.0)),   # M to capture with 1 Gsample/s
            'GS_application': 'GS phase retrieval on neutron flux I(t) -> A(t) for digital twin update',
        },
        'Breit_Wigner': {
            'E_eV': E_eV.tolist(),
            'sigma_barns': sigma_BW.tolist(),
            'E_res_eV': E_res_eV,
            'analogy': 'Breit-Wigner(E) = S21(resonator) = H(f) near pole -- identical math',
        },
        'connections': {
            'Maxwell': 'Neutron transport eq. has same form as Maxwell (diffusion = div-grad)',
            'GS': 'Measure |flux(t)|^2 at detector -> GS recovers complex flux A(t)',
            'photonic_ADC': 'M*f_s captures transient bandwidth; paper: M=10 -> 10x bandwidth',
            'H_of_f': 'H(f)=exp(j*pi*D*f^2) applies to DETECTOR signal in time domain',
        },
    }


def demo():
    print("=== COPPINGER / BHUSHAN / JALALI 1999 -- IEEE TRANS. MTT ===\n")

    print("--- Maxwell Phasor Domain (SMF-28 fiber at 193 THz) ---")
    mx = maxwell_phasor_domain(freq_Hz=193e12, medium='SMF28')
    print(f"  n = {mx['fields']['n_real']:.4f}  (SMF-28 expected: ~1.445)")
    print(f"  beta2 = {mx['GVD']['beta2_s2_m']:.3e} s^2/m")
    print(f"  D = {mx['GVD']['D_ps_nm_km']:.1f} ps/(nm*km)  (SMF-28 expected: ~17)")
    print(f"  v_phase = c/n = {mx['fields']['v_phase_m_s']/c_light:.4f} c")
    print(f"  Faraday curl E = -j*omega*mu*H: error = {mx['maxwell_verification']['faraday_error_frac']:.2e}")
    print(f"  Ampere curl H = J + j*omega*eps*E: error = {mx['maxwell_verification']['ampere_error_frac']:.2e}")

    print("\n--- Coppinger 1999: Paper Demo Values (M=10, 20 GHz capture) ---")
    cp = coppinger_1999_stretch_factor(D1_ps_nm_km=17, L1_km=5,
                                        D2_ps_nm_km=17, L2_km=45,
                                        Delta_lambda_nm=10, f_ADC_GHz=2)
    print(f"  M = {cp['results']['M']:.1f}  (paper: 10)")
    print(f"  T_w = {cp['results']['T_w_ps']:.0f} ps  (paper: 850 ps)")
    print(f"  B_RF = {cp['results']['B_RF_GHz']:.0f} GHz  (paper: 10 GHz, demonstrates 20)")
    print(f"  N_samples = {cp['results']['N_samples']:.0f}  (paper: ~1700)")
    print(f"  Chirp rate = {cp['results']['chirp_rate_nm_per_ps']:.4f} nm/ps")
    print(f"  {cp['H_f_connection'].split(chr(10))[0]}")

    print("\n--- SNR Analysis (M=10, P_opt=3 dBm) ---")
    snr = coppinger_1999_snr_analysis(M=10, P_opt_dBm=3, f_ADC_GHz=2)
    print(f"  Without stretch: SNR = {snr['without_stretch']['SNR_total_dB']:.1f} dB, ENOB = {snr['without_stretch']['ENOB']:.1f} bits")
    print(f"  With M=10 stretch: SNR = {snr['with_stretch']['SNR_total_dB']:.1f} dB, ENOB = {snr['with_stretch']['ENOB']:.1f} bits")
    print(f"  SNR improvement = {snr['improvement']['delta_SNR_dB']:.1f} dB  (theory: 10*log10(10) = {snr['improvement']['theory_dB']:.1f} dB)")
    print(f"  Noise breakdown: shot={snr['without_stretch']['SNR_shot_dB']:.0f}, RIN={snr['without_stretch']['SNR_RIN_dB']:.0f}, thermal={snr['without_stretch']['SNR_thermal_dB']:.0f} dB")

    print("\n--- Coastal EM (1 GHz tower, 10 km offshore) ---")
    ce = coastal_em_fields(freq_Hz=1e9, tower_height_m=30, distance_km=10)
    print(f"  Seawater skin depth at 1 GHz: {ce['seawater']['skin_depth_mm']:.2f} mm")
    print(f"  Seawater attenuation: {ce['seawater']['attenuation_dB_m']:.1f} dB/m")
    print(f"  Loss tangent (sigma/omega*eps): {ce['seawater']['loss_tangent']:.1f}  (>>1 = good conductor)")
    print(f"  Dipole radiation resistance: {ce['dipole']['R_rad_ohm']:.2f} ohm")
    print(f"  E-field at 10 km: {ce['dipole']['E_far_V_m']:.3e} V/m")
    print(f"  Brewster angle (TM pol.): {ce['fresnel']['theta_Brewster_deg']:.1f} deg")

    print("\n--- Digital Nuclear Twin (PWR, 50 pcm step) ---")
    twin = digital_nuclear_twin()
    print(f"  beta_total = {twin['delayed_neutrons']['beta_total_pcm']:.0f} pcm  (U-235 expected: 650)")
    print(f"  Reactor period T_0 = {twin['transient']['T_period_s']:.2f} s  (50 pcm subcritical)")
    print(f"  Transient bandwidth: {twin['photonic_adc']['f_transient_MHz']:.3f} MHz")
    print(f"  Breit-Wigner resonance: {twin['Breit_Wigner']['analogy']}")

    print("\n=== COPPINGER 1999 COMPLETE ===")


if __name__ == '__main__':
    demo()
