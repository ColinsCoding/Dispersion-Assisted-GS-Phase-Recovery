"""
Modern Physics: Serway/Moses/Moyer outline (16 chapters)

Causality thread that runs through the whole course:
  Ch01: Special relativity PREVENTS causality violations (nothing > c)
  Ch04: Photon -- causality at the quantum level (which-path, delayed choice)
  Ch07: QM tunneling -- particle THROUGH a barrier (still causal, just nonclassical)
  Ch09: Atomic structure -- Pauli exclusion makes the periodic table causal/ordered
  Ch11: Statistical physics -- second law = arrow of time = macroscopic causality
  Ch15: Nuclear applications -- radiation causes cancer AND cures it (causal intervention)

Connection to this repo throughout:
  Ch03: de Broglie matter waves = SAME dispersion physics as H(f)=exp(j*pi*D*f^2)
  Ch05: Wave packets, group velocity = pulse broadening in fiber = GVD
  Ch07: Tunneling evanescent wave = same math as TIR in vector_calculus.py
  Ch10: Fermi-Dirac = semiconductor = transistor = every compute device running GS
  Ch15: Nuclear medicine imaging (PET, SPECT) = phase retrieval problem in disguise

Physical constants (SI):
  hbar = 1.0546e-34 J*s
  h    = 6.626e-34 J*s
  c    = 2.998e8 m/s
  e    = 1.602e-19 C
  me   = 9.109e-31 kg
  mp   = 1.673e-27 kg
  kB   = 1.381e-23 J/K
  eV   = 1.602e-19 J
  a0   = 5.292e-11 m  (Bohr radius)
"""
import numpy as np
import sympy as sp

# ------ constants ------
hbar = 1.0546e-34; h = 6.626e-34; c = 2.998e8; e = 1.602e-19
me = 9.109e-31; mp = 1.673e-27; mn = 1.675e-27
kB = 1.381e-23; eV = 1.602e-19; a0 = 5.292e-11
mu0 = 4*np.pi*1e-7; eps0 = 8.854e-12


# ============================================================
# Ch 1-2: Special Relativity I and II
# ============================================================

def special_relativity_I():
    """
    SR I: postulates, Lorentz transforms, time dilation, length contraction.

    Postulates (Einstein 1905):
      1. Laws of physics are identical in all inertial frames.
      2. Speed of light c is the same in all inertial frames.

    Lorentz factor: gamma = 1/sqrt(1 - v^2/c^2)
    Time dilation:   Delta_t' = gamma * Delta_t0  (moving clock runs slow)
    Length contraction: L = L0/gamma               (moving rod is shorter)
    Simultaneity:    events simultaneous in S need not be in S'

    CAUSALITY: To violate causality, an effect must precede its cause in some frame.
      For spacelike separation: Delta_x > c*Delta_t.
      Lorentz boost can REVERSE time order -> causality violation.
      BUT: to send a signal between spacelike events requires v > c.
      Since nothing travels > c, no causal signal can connect spacelike events.
      Causality is PROTECTED by the speed of light limit.
      This is the deepest reason c is a universal limit -- not just a speed limit,
      but a CAUSALITY PROTECTION THEOREM.
    """
    betas = np.linspace(0, 0.999, 1000)
    gammas = 1.0 / np.sqrt(1 - betas**2)

    def lorentz(v, x, t):
        beta = v/c; gamma_v = 1/np.sqrt(1-beta**2)
        x_prime = gamma_v*(x - v*t)
        t_prime = gamma_v*(t - v*x/c**2)
        return x_prime, t_prime

    # Time dilation: muon example
    tau_muon = 2.2e-6     # muon lifetime at rest [s]
    v_muon = 0.998*c
    beta_m = v_muon/c
    gamma_muon = 1/np.sqrt(1 - beta_m**2)
    tau_lab = gamma_muon * tau_muon

    # Length contraction: LHC proton
    L0 = 27e3             # LHC circumference [m]
    beta_LHC = 0.9999999
    gamma_LHC = 1/np.sqrt(1 - beta_LHC**2)
    L_proton_frame = L0 / gamma_LHC

    # Causality: spacelike interval
    Delta_x = 3e8; Delta_t = 0.5   # s
    interval = Delta_x**2 - (c*Delta_t)**2
    is_spacelike = interval > 0

    return {
        'beta': betas, 'gamma': gammas,
        'muon': {
            'beta': beta_m, 'gamma': gamma_muon,
            'lifetime_rest_us': tau_muon*1e6,
            'lifetime_lab_us': tau_lab*1e6,
            'ratio': gamma_muon,
        },
        'LHC': {
            'gamma': gamma_LHC,
            'L_rest_m': L0,
            'L_proton_frame_mm': L_proton_frame*1e3,
        },
        'causality': {
            'spacelike': is_spacelike,
            'interval': interval,
            'lesson': (
                'Spacelike interval: Delta_x^2 > (c*Delta_t)^2. '
                'Time order CAN flip under Lorentz boost. '
                'But no causal signal can travel spacelike -> causality safe. '
                'NOTHING travels faster than c is equivalent to: causality holds.'
            ),
        },
        'lorentz_transforms': {
            'x_prime': 'gamma*(x - v*t)',
            't_prime': 'gamma*(t - v*x/c^2)',
            'velocity_addition': 'u_prime = (u - v)/(1 - u*v/c^2)',
        },
    }


def special_relativity_II():
    """
    SR II: relativistic dynamics, 4-vectors, E=mc^2.

    Relativistic momentum: p = gamma*m*v
    Relativistic energy:   E = gamma*m*c^2 = sqrt((pc)^2 + (m*c^2)^2)
    Rest energy:           E0 = m*c^2
    Kinetic energy:        K = (gamma-1)*m*c^2

    4-momentum: P^mu = (E/c, p_x, p_y, p_z)
    Invariant mass: P^mu*P_mu = -(mc)^2  [in -+++ metric]

    Photon: m=0 -> E = pc = hf = hc/lambda
      Momentum: p = h/lambda = hbar*k
      THIS IS THE DE BROGLIE RELATION for massless particles.

    Pair production: gamma -> e+ + e-  (requires E_gamma > 2*m_e*c^2 = 1.022 MeV)
    Annihilation:    e+ + e- -> 2 gamma  (each gamma E = 511 keV)
      This is how PET scanners work: detect the two back-to-back 511 keV photons.
      Coincidence detection = phase retrieval in time domain.
    """
    def relativistic_energy(m_kg, v):
        beta = v/c; gamma_v = 1/np.sqrt(1 - min(beta**2, 1-1e-15))
        E_total = gamma_v * m_kg * c**2
        K = (gamma_v-1) * m_kg * c**2
        p = gamma_v * m_kg * v
        return {'gamma': gamma_v, 'E_J': E_total, 'K_J': K, 'p_kgms': p,
                'E_eV': E_total/eV, 'K_eV': K/eV}

    electron_half_c = relativistic_energy(me, 0.5*c)
    proton_LHC = relativistic_energy(mp, 0.9999999*c)

    # E=mc^2 examples
    m_1g = 1e-3   # kg
    E_1g = m_1g * c**2   # Joules
    E_1g_MT = E_1g / 4.184e15   # megaton TNT equivalent

    # Photon momentum
    lam_visible = 550e-9
    p_photon = h / lam_visible
    E_photon_eV = h*c/lam_visible / eV

    # Pair production threshold
    E_threshold_eV = 2 * me * c**2 / eV

    return {
        'electron_at_0p5c': electron_half_c,
        'proton_LHC': proton_LHC,
        'E_mc2': {
            '1g_in_joules': E_1g,
            '1g_in_megatons_TNT': E_1g_MT,
            'sun_power_W': 3.8e26,
            'sun_mass_loss_kg_per_s': 3.8e26/c**2,
        },
        'photon': {
            'lambda_nm': 550,
            'momentum_kgms': p_photon,
            'energy_eV': E_photon_eV,
            'pressure': 'radiation pressure = power/c = force per area',
        },
        'pair_production_threshold_MeV': E_threshold_eV/1e6,
        'PET_scanner': {
            'mechanism': 'beta+ emitter -> e+ annihilates nearby e- -> 2x 511 keV photons',
            'coincidence': '180 deg back-to-back detection within 1ns window',
            'reconstruction': 'filtered backprojection (inverse Radon transform)',
            'phase_retrieval': 'TOF-PET: time-of-flight difference = position phase -> inverse problem',
        },
    }


# ============================================================
# Ch 3-4: Quantum Theory of Light + Particle Nature of Matter
# ============================================================

def quantum_theory_of_light():
    """
    Blackbody radiation, photoelectric effect, Compton scattering.

    Planck (1900): E = n*h*f  [quantized oscillator energy]
    Einstein (1905): light itself is quantized -- photon E = hf
    Compton (1923): photon has momentum p = h/lambda

    Blackbody: Planck distribution
      u(f,T) = (8*pi*h*f^3/c^3) / (exp(h*f/kB*T) - 1)
      Wien displacement: lambda_max * T = 2.898e-3 m*K
      Stefan-Boltzmann: P = sigma*T^4, sigma=5.67e-8 W/m^2/K^4

    Photoelectric effect: K_max = hf - phi  (phi = work function)
      Stopping potential: V_stop = K_max/e = (hf - phi)/e
      Threshold frequency: f_0 = phi/h
      KEY: K_max depends on frequency, NOT intensity -> light is quantized.

    Compton scattering: lambda' - lambda = (h/m_e*c)*(1 - cos(theta))
      Compton wavelength: lambda_C = h/(m_e*c) = 2.426 pm
    """
    # Planck distribution
    f = np.logspace(11, 15, 1000)   # 100 GHz to 1 PHz
    def planck(f_arr, T):
        return (8*np.pi*h*f_arr**3/c**3) / (np.exp(h*f_arr/(kB*T)) - 1)

    u_3000K = planck(f, 3000)
    u_6000K = planck(f, 6000)
    u_300K  = planck(f, 300)

    # Wien displacement
    lambda_max_sun = 2.898e-3 / 6000   # m
    lambda_max_body = 2.898e-3 / 310   # m (37C body)

    # Photoelectric: Cs (cesium phi=2.1 eV)
    phi_Cs = 2.1 * eV
    f_0_Cs = phi_Cs / h
    lambda_0_Cs = c / f_0_Cs
    f_test = 8e14   # Hz (UV)
    K_max_eV = (h*f_test - phi_Cs) / eV
    V_stop = K_max_eV   # volts

    # Compton
    lambda_C = h/(me*c)
    theta_90 = np.pi/2
    delta_lambda_90 = lambda_C * (1 - np.cos(theta_90))
    lam_incident = 0.071e-9   # 0.071 nm X-ray
    lam_scattered_90 = lam_incident + delta_lambda_90

    return {
        'planck': {
            'f': f, 'u_3000K': u_3000K, 'u_6000K': u_6000K, 'u_300K': u_300K,
            'wien_sun_nm': lambda_max_sun*1e9,
            'wien_body_um': lambda_max_body*1e6,
        },
        'photoelectric': {
            'phi_Cs_eV': 2.1,
            'threshold_freq_Hz': f_0_Cs,
            'threshold_lambda_nm': lambda_0_Cs*1e9,
            'K_max_eV': K_max_eV,
            'V_stop': V_stop,
            'key_insight': 'K_max = hf - phi: depends on FREQUENCY not intensity',
        },
        'compton': {
            'lambda_C_pm': lambda_C*1e12,
            'delta_lambda_90_pm': delta_lambda_90*1e12,
            'lambda_scattered_90_nm': lam_scattered_90*1e9,
            'formula': 'delta_lambda = (h/m_e*c)*(1-cos(theta))',
        },
    }


def particle_nature_of_matter():
    """
    de Broglie matter waves, electron diffraction, Heisenberg uncertainty.

    de Broglie (1924): lambda = h/p  (matter has wave properties)
    For electron at voltage V: K = eV, p = sqrt(2*m_e*eV)
      lambda = h/sqrt(2*m_e*eV)

    Davisson-Germer (1927): electron diffraction from Ni crystal confirmed de Broglie.
    Same Bragg law: 2*d*sin(theta) = n*lambda

    Heisenberg uncertainty:
      Delta_x * Delta_p >= hbar/2
      Delta_E * Delta_t >= hbar/2
      These are NOT measurement errors -- they are fundamental limits on SIMULTANEOUS
      definition of position and momentum for a quantum state.

    Connection to repo:
      de Broglie: lambda = h/p -> p = hbar*k -> k is the quantum wavenumber.
      GS retrieves the phase of k (the spatial frequency of the wave).
      Uncertainty principle: Delta_x*Delta_k >= 1/2 -> time-bandwidth product.
      Short pulse (small Delta_t) -> wide spectrum (large Delta_f).
      This IS the bandwidth-time tradeoff in dispersive Fourier transform / STEAM.
    """
    # de Broglie wavelengths at various energies
    energies_eV = np.array([1, 10, 100, 1000, 1e4, 1e5])
    p_electron = np.sqrt(2*me*energies_eV*eV)
    lambda_electron_nm = h / p_electron * 1e9

    # Bragg diffraction: d_Ni = 0.215 nm, theta for 54V electrons
    V = 54; K = V*eV; p54 = np.sqrt(2*me*K); lam54 = h/p54
    d_Ni = 0.215e-9
    sin_theta = lam54/(2*d_Ni)
    theta_deg = np.degrees(np.arcsin(min(sin_theta, 1)))

    # Uncertainty principle examples
    # Electron in atom: Delta_x ~ a0 -> Delta_p >= hbar/(2*a0) -> K_min
    Delta_x_atom = a0
    Delta_p_min = hbar/(2*Delta_x_atom)
    K_min_eV = Delta_p_min**2/(2*me) / eV

    # Natural linewidth: Delta_t = tau -> Delta_E >= hbar/(2*tau)
    tau_excited = 1e-8   # 10 ns typical atomic lifetime
    Delta_E_eV = hbar/(2*tau_excited) / eV
    Delta_nu_Hz = Delta_E_eV*eV/h

    return {
        'de_broglie': {
            'energies_eV': energies_eV.tolist(),
            'lambda_nm': lambda_electron_nm.tolist(),
            'note': 'At 100 eV: lambda~0.12 nm (X-ray scale -> electron microscope)',
        },
        'davisson_germer': {
            'V_volts': V,
            'lambda_pm': lam54*1e12,
            'd_Ni_nm': d_Ni*1e9,
            'theta_deg': theta_deg,
            'Bragg': '2*d*sin(theta) = lambda verified experimentally 1927',
        },
        'uncertainty': {
            'position_momentum': 'Delta_x * Delta_p >= hbar/2',
            'energy_time': 'Delta_E * Delta_t >= hbar/2',
            'K_min_atom_eV': K_min_eV,
            'natural_linewidth_eV': Delta_E_eV,
            'natural_linewidth_Hz': Delta_nu_Hz,
            'repo_connection': (
                'Time-bandwidth product: Delta_t * Delta_f >= 1/(4*pi). '
                'STEAM camera: 36 Mfps requires Delta_t ~ 27 ns per frame -> '
                'Delta_f >= 12 MHz per pixel. '
                'Dispersion maps Delta_f to time -> must have enough bandwidth. '
                'This IS the quantum uncertainty principle applied to optical pulses.'
            ),
        },
    }


# ============================================================
# Ch 5-6: Matter Waves + QM in 1D
# ============================================================

def matter_waves_wave_packets():
    """
    Wave packets, group velocity, phase velocity, dispersion.

    Single wave: psi(x,t) = A*exp(j*(k*x - omega*t))
    Wave packet: psi(x,t) = int A(k)*exp(j*(k*x - omega(k)*t)) dk

    Phase velocity:   v_ph = omega/k  (speed of phase fronts)
    Group velocity:   v_gr = d(omega)/dk  (speed of wave packet envelope)
    Dispersion:       d^2(omega)/dk^2 != 0 -> packet spreads (GVD!)

    For free particle: omega = hbar*k^2/(2*m) [quadratic dispersion]
      v_ph = hbar*k/(2m) = v/2
      v_gr = hbar*k/m = v  (group velocity = particle velocity, as expected)
      d^2(omega)/dk^2 = hbar/m -> packet SPREADS in time

    For photon in fiber: omega = c*k/n + (beta_2/2)*(k-k0)^2
      This IS H(f) = exp(j*pi*D*f^2) in time domain.
      GVD = d^2(beta)/d(omega)^2 = beta_2.
      Matter wave spreading ~ photon pulse broadening. SAME EQUATION.

    THIS IS THE KEY CONNECTION:
      Free particle QM: psi(x,t) = (1/sqrt(sigma*sqrt(2*pi))) * exp(-x^2/(4*sigma^2))
                                  * exp(j*(k0*x - omega0*t)) * exp(-t^2/(2*tau^2))
      Dispersive photon: E(t) = E0*exp(-t^2/(2*tau0^2)) * exp(j*omega0*t)
        -> after fiber: E(t,L) = E0*exp(-t^2/(2*tau(L)^2)) * exp(j*chirp*t^2)
      SAME Gaussian spreading. QM wave packet = dispersive optical pulse.
    """
    # Gaussian wave packet spreading (QM)
    k0 = 1e10    # central wavevector (1/m) ~ electron
    sigma_x0 = 1e-10  # initial width 1 Angstrom
    m_particle = me

    t_arr = np.linspace(0, 1e-14, 400)   # 10 femtoseconds
    sigma_x_t = sigma_x0 * np.sqrt(1 + (hbar*t_arr/(2*m_particle*sigma_x0**2))**2)

    # Compare to optical pulse broadening
    beta2 = -20e-27   # typical SMF at 1550nm [s^2/m]
    tau0 = 1e-12      # 1 ps pulse
    L = np.linspace(0, 100e3, 400)   # 0-100 km fiber
    tau_L = tau0 * np.sqrt(1 + (beta2*L/tau0**2)**2)

    v_phase_free = hbar*k0 / (2*m_particle)
    v_group_free = hbar*k0 / m_particle

    return {
        't_fs': t_arr*1e15,
        'sigma_x_pm': sigma_x_t*1e12,
        'v_phase_m_per_s': v_phase_free,
        'v_group_m_per_s': v_group_free,
        'doubling_time_fs': float(2*m_particle*sigma_x0**2/hbar * 1e15),
        'optical_analog': {
            'L_km': L/1e3, 'tau_ps': tau_L*1e12,
            'beta2': beta2, 'tau0_ps': tau0*1e12,
        },
        'identical_math': (
            'QM packet: sigma(t) = sigma0*sqrt(1 + (hbar*t/(2*m*sigma0^2))^2)\n'
            'Optical:   tau(L)   = tau0  *sqrt(1 + (beta2*L/tau0^2)^2)\n'
            'Identical structure: hbar/(2*m) <-> beta2, t <-> L.\n'
            'GS phase retrieval recovers the chirp in BOTH cases.'
        ),
    }


def qm_1d():
    """
    Particle in a box (infinite square well), finite square well, harmonic oscillator.

    Infinite square well [0, L]:
      psi_n(x) = sqrt(2/L)*sin(n*pi*x/L)
      E_n = n^2 * pi^2 * hbar^2 / (2*m*L^2) = n^2 * E_1
      E_1 = pi^2*hbar^2/(2*m*L^2)

    Harmonic oscillator V=kx^2/2:
      E_n = (n + 1/2)*hbar*omega_0
      Zero-point energy: E_0 = hbar*omega_0/2 (CANNOT be zero -- uncertainty principle)
      psi_0 = (m*omega/pi*hbar)^(1/4)*exp(-m*omega*x^2/(2*hbar))  [Gaussian!]

    Orthonormality: int psi_n * psi_m dx = delta_{nm}
    Completeness: any wave function = sum c_n * psi_n  (Fourier series analog)
    """
    # Particle in a box: hydrogen atom sized, L=1 Angstrom
    L = 1e-10   # 1 Angstrom
    n_arr = np.arange(1, 11)
    E_n = n_arr**2 * np.pi**2 * hbar**2 / (2*me*L**2) / eV   # eV

    x = np.linspace(0, L, 500)
    wavefunctions = [np.sqrt(2/L)*np.sin(n*np.pi*x/L) for n in [1,2,3,4]]

    # Harmonic oscillator
    omega0 = 1e14   # rad/s (molecular vibration scale)
    E_ho_n = (np.arange(0, 6) + 0.5) * hbar * omega0 / eV
    x_ho = np.linspace(-5e-10, 5e-10, 500)
    # Ground state wavefunction
    alpha = me*omega0/hbar
    psi0_ho = (alpha/np.pi)**0.25 * np.exp(-alpha*x_ho**2/2)

    return {
        'box': {
            'L_ang': L*1e10,
            'E_n_eV': E_n.tolist(),
            'E_1_eV': E_n[0],
            'ratio_E2_E1': 4.0,
            'note': 'E_n = n^2 * E_1. Ground state NOT zero (zero-point energy).',
        },
        'wavefunctions': {
            'x_nm': x*1e9,
            'psi_1': wavefunctions[0].tolist(),
            'psi_2': wavefunctions[1].tolist(),
            'psi_3': wavefunctions[2].tolist(),
        },
        'harmonic_osc': {
            'omega0': omega0,
            'E_n_eV': E_ho_n.tolist(),
            'zero_point_eV': E_ho_n[0],
            'psi0_Gaussian': True,
            'note': 'Ground state is Gaussian: same shape as our optical pulse E(t).',
        },
        'orthonormality': 'int psi_n*psi_m dx = delta_nm  [Fourier series analog]',
        'laser_connection': (
            'Harmonic oscillator psi_n = photon number states |n>. '
            'psi_0 = vacuum state (coherent field). '
            'Coherent laser = displaced harmonic oscillator = Poisson photon statistics. '
            'Squeezed state = asymmetric uncertainty: small Delta_x, large Delta_p.'
        ),
    }


def tunneling_phenomena():
    """
    Quantum tunneling: particle penetrates classically forbidden region.

    Classically: K = E - V < 0 -> forbidden.
    Quantum: psi decays exponentially in forbidden region: psi ~ exp(-kappa*x)
      kappa = sqrt(2*m*(V-E))/hbar

    Transmission coefficient (thin barrier width d):
      T ~ exp(-2*kappa*d)  [WKB approximation]

    Applications:
      Alpha decay: alpha particle tunnels out of nucleus (Gamow 1928)
        half-life T_{1/2} ~ exp(2*pi*Z1*Z2*e^2/(hbar*v))  [Gamow factor]
      STM: scanning tunneling microscope -- tunneling current ~ exp(-2*kappa*d)
        Angstrom-scale resolution (1 Angstrom -> 10x change in current)
      Tunnel diode: negative resistance (Esaki 1957)
      Josephson junction: Cooper pairs tunnel between superconductors
        -> qubit hardware (from quantum_ce.py)

    Evanescent wave = same math:
      In optics: field beyond TIR interface: E ~ exp(-kappa*y)
      SAME as quantum tunneling: kappa = sqrt(2*m*(V-E))/hbar = k_i (imaginary k)
      Optical fiber = light tunnels between cores (directional coupler).
      (Covered in vector_calculus.py ray_tracing_bvh() and photonics_calculus.py)
    """
    # Transmission coefficient vs barrier width
    V0 = 5.0*eV; E = 1.0*eV   # particle at 1 eV, barrier 5 eV
    kappa = np.sqrt(2*me*(V0-E)) / hbar
    d_arr = np.linspace(0, 2e-9, 400)   # 0 to 2 nm
    T_arr = np.exp(-2*kappa*d_arr)

    # STM: 1 Angstrom height change
    T_at_0p5nm = np.exp(-2*kappa*0.5e-9)
    T_at_0p6nm = np.exp(-2*kappa*0.6e-9)
    current_ratio = T_at_0p6nm / T_at_0p5nm

    # Alpha decay: Gamow factor for Uranium-238
    Z1 = 2; Z2 = 90; m_alpha = 4*mp   # alpha + Th-234
    E_alpha = 4.27*eV*1e6   # 4.27 MeV
    r_nuclear = 1.2e-15 * 238**(1/3)
    v_alpha = np.sqrt(2*E_alpha/m_alpha)
    G_gamow = np.pi*Z1*Z2*e**2 / (hbar*v_alpha * 4*np.pi*eps0)
    T_gamow = np.exp(-2*G_gamow)
    # Very rough half-life estimate
    f_assault = v_alpha / (2*r_nuclear)   # frequency hitting barrier
    t_half = 0.693 / (f_assault * T_gamow)

    return {
        'kappa_per_nm': kappa*1e-9,
        'd_nm': d_arr*1e9,
        'T_arr': T_arr,
        'T_at_0p5nm': T_at_0p5nm,
        'stm': {
            'T_ratio_1A_increase': current_ratio,
            'lesson': f'1 Angstrom extra barrier -> current x{current_ratio:.3f} (order of magnitude)',
        },
        'alpha_decay_U238': {
            'E_alpha_MeV': 4.27,
            'Gamow_factor': G_gamow,
            'T_gamow': T_gamow,
            't_half_years_rough': t_half/3.15e7,
        },
        'evanescent_connection': (
            'Quantum tunneling: psi ~ exp(-kappa*x), kappa = sqrt(2m(V-E))/hbar\n'
            'TIR evanescent:    E  ~ exp(-kappa*y), kappa = sqrt(k_x^2 - k0^2*n^2)\n'
            'Same exponential decay. kappa = Im[k] = imaginary part of wavevector.\n'
            'The dark side of the wavevector.'
        ),
    }


# ============================================================
# Ch 7-8: QM in 3D + Atomic Structure
# ============================================================

def qm_3d_hydrogen():
    """
    Hydrogen atom: 3D Schrodinger equation in spherical coordinates.

    H*psi = E*psi:  psi(r,theta,phi) = R_{nl}(r) * Y_l^m(theta,phi)
    Quantum numbers: n=1,2,3... (principal), l=0..n-1 (orbital), m=-l..l (magnetic)

    Energy levels: E_n = -13.6 eV / n^2  (Bohr formula, exact for H)
    Bohr radius:   a0 = hbar^2/(m_e*e^2/(4*pi*eps0)) = 0.0529 nm

    Radial wavefunctions R_{nl}:
      n=1,l=0 (1s): R_10 = 2*(1/a0)^(3/2)*exp(-r/a0)
      n=2,l=0 (2s): R_20 = (1/sqrt(8))*(1/a0)^(3/2)*(2-r/a0)*exp(-r/(2*a0))
      n=2,l=1 (2p): R_21 = (1/sqrt(24))*(1/a0)^(3/2)*(r/a0)*exp(-r/(2*a0))

    Spectral lines: Delta_E = E_ni - E_nf = hf
      Lyman series: n -> 1 (UV)
      Balmer series: n -> 2 (visible, Hα=656nm, Hβ=486nm)
      Paschen series: n -> 3 (IR)
    """
    n_levels = np.arange(1, 8)
    E_n = -13.6 / n_levels**2   # eV

    # Hydrogen radial wavefunctions
    r_arr = np.linspace(0, 20*a0, 1000)
    rho = r_arr/a0
    R_1s = 2*(1/a0)**1.5 * np.exp(-rho)
    R_2s = (1/np.sqrt(8))*(1/a0)**1.5 * (2 - rho)*np.exp(-rho/2)
    R_2p = (1/np.sqrt(24))*(1/a0)**1.5 * rho*np.exp(-rho/2)

    # Radial probability density P(r) = r^2 * |R|^2
    P_1s = r_arr**2 * R_1s**2
    P_2s = r_arr**2 * R_2s**2
    P_2p = r_arr**2 * R_2p**2

    # Peak of 1s: at r = a0
    r_peak_1s = r_arr[np.argmax(P_1s)]

    # Spectral transitions
    Balmer = {}
    for n in [3,4,5,6]:
        dE = abs(E_n[n-1] - E_n[1])   # n->2
        lam = h*c/(dE*eV)*1e9   # nm
        Balmer[f'H{chr(ord("a")+n-3)}_n{n}_to_2'] = {'dE_eV': dE, 'lambda_nm': lam}

    return {
        'E_n_eV': dict(zip(n_levels.tolist(), E_n.tolist())),
        'ionization_eV': 13.6,
        'r_ang': r_arr*1e10,
        'R_1s': R_1s.tolist(), 'R_2s': R_2s.tolist(), 'R_2p': R_2p.tolist(),
        'P_1s': P_1s.tolist(), 'P_2s': P_2s.tolist(), 'P_2p': P_2p.tolist(),
        'r_peak_1s_ang': r_peak_1s*1e10,
        'Balmer_series': Balmer,
        'quantum_numbers': {
            'n': 'principal (1,2,3...): energy',
            'l': 'orbital (0..n-1): angular momentum  [s,p,d,f = 0,1,2,3]',
            'm': 'magnetic (-l..l): z-component of L',
            's': 'spin (+/-1/2): intrinsic angular momentum',
        },
    }


def atomic_structure():
    """
    Multi-electron atoms: Pauli exclusion, shell filling, periodic table.

    Pauli exclusion principle: no two electrons can have identical quantum numbers (n,l,m,s).
    Spin-orbit coupling: L.S interaction splits levels (fine structure).
    Hund's rules (for ground state):
      1. Maximize S (total spin) -- Hund's first rule
      2. Maximize L given S -- second rule
      3. J = |L-S| if shell < half full, J = L+S if > half full

    Shell structure:
      n=1: 1s (2 electrons)
      n=2: 2s (2) + 2p (6) = 8
      n=3: 3s (2) + 3p (6) + 3d (10) = 18  [but 3d fills after 4s!]

    Laser connection:
      Population inversion: upper state must have longer lifetime than lower.
      Selection rules: Delta_l = +-1, Delta_m = 0, +-1  (from angular momentum conservation)
      Forbidden transitions: metastable states -> population inversion possible -> laser.

    OUSD: Directed Energy lasers require atomic structure understanding for:
      gain media (Nd:YAG, Er:fiber, Ti:sapphire), pumping transitions, saturation.
    """
    # Aufbau principle: fill in energy order
    # 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p ...
    subshells = [
        ('1s',2), ('2s',2), ('2p',6), ('3s',2), ('3p',6), ('4s',2), ('3d',10),
        ('4p',6), ('5s',2), ('4d',10), ('5p',6), ('6s',2), ('4f',14), ('5d',10),
        ('6p',6), ('7s',2), ('5f',14), ('6d',10), ('7p',6),
    ]
    electron_config = {}
    total = 0
    for sub, cap in subshells:
        if total >= 118:
            break
        electron_config[sub] = min(cap, 118-total)
        total += electron_config[sub]

    # Key elements
    elements = {
        'H':  {'Z':1,  'config':'1s^1', 'laser_relevance':'simplest spectrum, Lyman alpha 121nm'},
        'He': {'Z':2,  'config':'1s^2', 'laser_relevance':'HeNe laser: He pumps Ne excited state'},
        'Na': {'Z':11, 'config':'[Ne]3s^1', 'laser_relevance':'589nm D-line, used as freq reference'},
        'Ne': {'Z':10, 'config':'[He]2s^2 2p^6', 'laser_relevance':'HeNe laser gain medium'},
        'Nd': {'Z':60, 'config':'[Xe]4f^4 6s^2', 'laser_relevance':'Nd:YAG 1064nm, HELLADS directed energy'},
        'Er': {'Z':68, 'config':'[Xe]4f^12 6s^2', 'laser_relevance':'Er:fiber 1550nm, telecom window, this repo'},
        'Ti': {'Z':22, 'config':'[Ar]3d^2 4s^2', 'laser_relevance':'Ti:sapphire 700-1000nm tunable, ultrafast'},
        'U':  {'Z':92, 'config':'[Rn]5f^3 6d 7s^2', 'laser_relevance':'nuclear fuel, uranium spectroscopy'},
    }

    return {
        'aufbau_order': [s[0] for s in subshells[:12]],
        'pauli': 'No two electrons: same (n, l, m, s). Max 2 per orbital (spin up/down).',
        'selection_rules': 'Delta_l = +/-1, Delta_m = 0, +/-1  (photon has angular momentum 1)',
        'elements': elements,
        'periodic_table_causality': (
            'Pauli exclusion + shell filling = periodic table structure.\n'
            'The entire chemical world follows from these quantum rules.\n'
            'Causality: electron configuration of atom DETERMINES its bonding.\n'
            'Bonding -> molecular structure -> protein folding -> life.\n'
            'From quantum numbers to biology: a causal chain.'
        ),
    }


# ============================================================
# Ch 9-10: Statistical Physics + Molecular Structure
# ============================================================

def statistical_physics():
    """
    Maxwell-Boltzmann, Fermi-Dirac, Bose-Einstein distributions.

    All three answer: what is the average number of particles in state with energy E?

    Maxwell-Boltzmann (classical): n(E) = A*exp(-E/(kB*T))
      Valid when quantum effects negligible (dilute gas, high T)

    Fermi-Dirac (fermions, half-integer spin: electrons, protons, neutrons):
      f(E) = 1 / (exp((E - mu)/(kB*T)) + 1)
      mu = Fermi energy (= E_F at T=0)
      At T=0: step function -- filled below E_F, empty above.
      Controls: electrical conductivity, semiconductors, white dwarf stars.

    Bose-Einstein (bosons, integer spin: photons, phonons, He-4):
      n(E) = 1 / (exp((E - mu)/(kB*T)) - 1)
      mu = 0 for photons (no conservation of photon number)
      -> Planck distribution! (blackbody radiation is Bose-Einstein)
      Bose-Einstein condensate (BEC): macroscopic occupation of ground state (T < T_c)

    Connection to repo:
      Photon number statistics: coherent laser = Poisson (neither FD nor BE exactly)
      Shot noise = sqrt(N) = sqrt(n_photons): Poisson fluctuations = fundamental noise floor.
      Thermal light = super-Poissonian (BE statistics) = more noise.
      Squeezed light = sub-Poissonian (below shot noise) -> quantum sensing.
    """
    T = np.array([100, 300, 1000, 5000, 30000])
    kBT = kB*T/eV   # in eV

    # Fermi-Dirac for copper (E_F = 7.04 eV)
    E_F_Cu = 7.04   # eV
    E_arr = np.linspace(0, 12, 1000)
    def fermi_dirac(E, E_F, T_K):
        return 1/(np.exp((E - E_F)/(kB*T_K/eV)) + 1)

    f_300K = fermi_dirac(E_arr, E_F_Cu, 300)
    f_1000K = fermi_dirac(E_arr, E_F_Cu, 1000)
    f_0K = (E_arr < E_F_Cu).astype(float)

    # Bose-Einstein for photons: n(E) = 1/(exp(E/kBT) - 1) [mu=0]
    def bose_einstein_photon(E_eV, T_K):
        x = E_eV / (kB*T_K/eV)
        return 1.0/(np.exp(x) - 1)

    photon_occupancy_visible = bose_einstein_photon(2.0, 300)  # 620nm photon at room T

    # Fermi temperature for copper
    T_Fermi_Cu = E_F_Cu*eV/kB

    return {
        'fermi_dirac': {
            'E_eV': E_arr.tolist(),
            'f_300K': f_300K.tolist(),
            'f_1000K': f_1000K.tolist(),
            'f_0K': f_0K.tolist(),
            'E_F_Cu_eV': E_F_Cu,
            'T_Fermi_Cu_K': T_Fermi_Cu,
        },
        'BE_photon_occupancy_room_T': photon_occupancy_visible,
        'distributions': {
            'Maxwell-Boltzmann': 'A*exp(-E/kBT): classical distinguishable particles',
            'Fermi-Dirac': '1/(exp((E-mu)/kBT)+1): fermions, half-int spin, 0<=f<=1',
            'Bose-Einstein': '1/(exp((E-mu)/kBT)-1): bosons, int spin, n>=0',
        },
        'photon_stats': {
            'thermal': 'Bose-Einstein: super-Poissonian, noisy',
            'laser': 'Poisson: shot-noise limited, sqrt(N)',
            'squeezed': 'sub-Poissonian: below shot noise, quantum sensing',
        },
        'semiconductor': (
            'Semiconductor: E_g between valence and conduction bands.\n'
            'Fermi level in gap -> exponentially few carriers.\n'
            'Doping shifts Fermi level -> n-type (E_F near conduction) / p-type (near valence).\n'
            'p-n junction = transistor = every compute device running this repo.'
        ),
    }


def molecular_structure():
    """
    LCAO (Linear Combination of Atomic Orbitals), bonding types.

    H2+ molecular ion: one electron, two protons.
    Bonding orbital:    psi_+ = (psi_A + psi_B)/sqrt(2)  E_+ < E_atom  [lower energy]
    Antibonding:        psi_- = (psi_A - psi_B)/sqrt(2)  E_- > E_atom  [higher energy]

    Bond types in solids (from the image: ionic, covalent, metallic, molecular):
      IONIC:    NaCl: Na+ and Cl- electrostatic attraction. Hard, brittle, high T_melt.
                E_bond = -e^2/(4*pi*eps0*r0) * Madelung_constant
      COVALENT: Diamond, Si: shared electron pairs. Very hard. Directional bonds.
      METALLIC: Cu, Al: delocalized electrons (free electron gas). Conducts.
      MOLECULAR: van der Waals: dipole-dipole, London dispersion. Soft, low T_melt.
      AMORPHOUS: No long-range order. Glass (SiO2), polymers, proteins, biological tissue.
        Life science: proteins are amorphous polypeptides. DNA = quasi-1D crystal.

    LCAO-MO -> Huckel theory -> band theory -> solid state.
    Same LCAO math used in: tight-binding band structure, BCS theory, quantum chemistry.
    """
    # H2+ overlap integral (approximate)
    R = 2.0*a0   # equilibrium bond length (atomic units)
    S_12 = np.exp(-R/a0) * (1 + R/a0 + (R/a0)**2/3)  # overlap integral
    E_bonding = -13.6/(1 + S_12) * eV   # rough estimate
    E_antibonding = -13.6/(1 - S_12) * eV

    # Madelung constant for NaCl
    M_NaCl = 1.7476
    r0_NaCl = 2.82e-10   # m
    E_pair_NaCl = -M_NaCl * e**2/(4*np.pi*eps0*r0_NaCl) / eV

    bond_types = {
        'ionic': {
            'example': 'NaCl, MgO, CaCO3 (calcite)',
            'E_pair_eV': E_pair_NaCl,
            'Madelung': M_NaCl,
            'property': 'hard, brittle, high Tm, transparent to visible',
        },
        'covalent': {
            'example': 'Diamond, Si, Ge, GaAs',
            'E_bond_eV': 7.4,
            'property': 'very hard (diamond), semiconductor (Si/Ge), directional bonds',
            'photonics': 'GaAs direct-gap semiconductor -> LED, laser diode. Same material as III-V photonics.',
        },
        'metallic': {
            'example': 'Cu, Al, Au, Fe',
            'free_electron_model': 'E_F = hbar^2/(2m)*(3*pi^2*n)^(2/3)',
            'property': 'ductile, conducts electricity and heat, opaque',
            'photonics': 'Surface plasmons: EM wave at metal-dielectric interface. Same dispersion math.',
        },
        'molecular': {
            'example': 'Ice, solid CO2, wax, aspirin',
            'interaction': 'van der Waals: London (induced dipole) + Keesom + Debye',
            'property': 'soft, low Tm, soluble in organic solvents',
        },
        'amorphous': {
            'example': 'Glass (SiO2), PDMS, proteins, DNA, amorphous silicon',
            'property': 'no long-range order, isotropic, can have local short-range order',
            'life_science': (
                'Proteins: 20 amino acids, amorphous 3D fold. '
                'Folding determined by sequence (primary) -> secondary -> tertiary. '
                'Misfolding = Alzheimer, Parkinson, prion disease. '
                'Drug discovery = find molecules that bind specific protein fold = inverse problem.'
            ),
            'photonics': 'Amorphous Si photovoltaics. Chalcogenide glasses for IR photonics (PCM memory).',
        },
    }

    return {
        'H2_plus': {
            'R_bohr': R/a0,
            'overlap_S12': S_12,
            'E_bonding_eV': E_bonding/eV,
            'LCAO': 'psi_+/- = (psi_A +/- psi_B)/sqrt(2)  [bonding/antibonding]',
        },
        'bond_types': bond_types,
        'band_theory_preview': (
            'LCAO for N atoms -> N molecular orbitals -> band (N very large).\n'
            'Filled band = insulator/semiconductor. Partially filled = metal.\n'
            'Band gap E_g determines optical properties:\n'
            '  E_g > 3.1 eV (photon energy at 400nm): transparent to visible\n'
            '  E_g ~ 1.1 eV (Si): absorbs visible, solar cell\n'
            '  E_g ~ 0.8 eV (Ge): absorbs NIR, photodetector'
        ),
    }


# ============================================================
# Ch 11: The Solid State
# ============================================================

def solid_state():
    """
    Bonding, band structure, conductors/semiconductors/insulators.
    Includes amorphous and life-science materials.
    """
    # Free electron model: Fermi energy
    def fermi_energy_metal(n_per_m3):
        return hbar**2/(2*me) * (3*np.pi**2 * n_per_m3)**(2/3) / eV

    # Metal densities
    metals = {
        'Cu': {'n': 8.49e28, 'work_phi_eV': 4.65},
        'Al': {'n': 18.1e28, 'work_phi_eV': 4.28},
        'Au': {'n': 5.90e28, 'work_phi_eV': 5.10},
    }
    for m, data in metals.items():
        data['E_F_eV'] = fermi_energy_metal(data['n'])
        data['v_F_m_per_s'] = np.sqrt(2*data['E_F_eV']*eV/me)

    # Semiconductor band gaps
    semiconductors = {
        'Si':    {'E_g_eV': 1.12, 'type': 'indirect', 'lambda_onset_nm': h*c/(1.12*eV)*1e9},
        'Ge':    {'E_g_eV': 0.67, 'type': 'indirect', 'lambda_onset_nm': h*c/(0.67*eV)*1e9},
        'GaAs':  {'E_g_eV': 1.42, 'type': 'direct',   'lambda_onset_nm': h*c/(1.42*eV)*1e9},
        'InP':   {'E_g_eV': 1.35, 'type': 'direct',   'lambda_onset_nm': h*c/(1.35*eV)*1e9},
        'GaN':   {'E_g_eV': 3.40, 'type': 'direct',   'lambda_onset_nm': h*c/(3.40*eV)*1e9},
        'InGaAsP': {'E_g_eV': 0.80, 'type': 'direct', 'lambda_onset_nm': 1550.0},
    }

    # Amorphous / life science
    amorphous = {
        'glass_SiO2': {'T_g_C': 1200, 'n_1550nm': 1.4468, 'use': 'optical fiber, this repo'},
        'PDMS':       {'T_g_C': -125, 'n_vis': 1.41, 'use': 'microfluidics (microfluidics.py)'},
        'amorphous_Si': {'E_g_eV': 1.7, 'use': 'solar cell, thin film'},
        'protein_fold':  {'order': 'short-range only', 'use': 'drug target, biomarker'},
    }

    return {
        'metals': metals,
        'semiconductors': semiconductors,
        'amorphous': amorphous,
        'InGaAsP_note': (
            'InGaAsP quaternary: bandgap tunable by composition. '
            'At E_g=0.80 eV: lambda=1550nm = telecom C-band = Er fiber amplifier window. '
            'This IS the wavelength of Jalali STEAM camera and this repo.'
        ),
        'causality_in_life_science': (
            'Amorphous life science causality:\n'
            '  DNA sequence -> mRNA (transcription) -> protein (translation)\n'
            '  Protein fold -> function -> cell signaling -> organism behavior\n'
            '  This IS a causal chain. Central dogma of molecular biology.\n'
            '  Breaking causality: prion (misfolded protein CAUSES misfolding of others)\n'
            '  = causal violation at molecular scale. Same mathematics as rumor spreading.'
        ),
    }


# ============================================================
# Ch 12-13: Nuclear Structure + Applications
# ============================================================

def nuclear_structure():
    """
    Liquid drop model, shell model, binding energy, magic numbers.

    Binding energy: B = (Z*m_H + N*m_n - M_atom)*c^2
    Semi-empirical mass formula (Bethe-Weizsacker):
      B = a_v*A - a_s*A^(2/3) - a_c*Z^2*A^(-1/3) - a_sym*(A-2Z)^2/A + delta
      a_v=15.8, a_s=18.3, a_c=0.714, a_sym=23.2 MeV

    Magic numbers: 2, 8, 20, 28, 50, 82, 126
      Extra stable nuclei at magic numbers (like noble gas electron configs).
      Evidence for nuclear shell model (same QM as atomic shells).

    Nuclear forces: strong (attractive, short-range ~1 fm), EM (repulsive, long-range).
    Competition: light nuclei -> strong wins (E/A increases). Heavy -> EM wins (E/A falls).
    Iron-56: maximum binding energy per nucleon (~8.8 MeV/A) = most stable nucleus.
      Elements lighter than Fe: energy released by FUSION.
      Elements heavier than Fe: energy released by FISSION.
    """
    # Binding energy per nucleon via SEMF
    a_v=15.8; a_s=18.3; a_c=0.714; a_sym=23.2

    def binding_energy_SEMF(Z, A):
        N = A - Z
        if A == 0: return 0
        delta = 0
        if A%2 == 0:
            delta = 12.0/np.sqrt(A) * (1 if Z%2==0 else -1)
        B = a_v*A - a_s*A**(2/3) - a_c*Z**2*A**(-1/3) - a_sym*(N-Z)**2/A + delta
        return max(B, 0)

    A_arr = np.arange(1, 250)
    # Approximate Z for stability: Z ~ A/(2 + A^(2/3)*a_c/(2*a_sym)) approx A/2 for light
    Z_stable = np.array([max(1, round(A/(2 + A**(2/3)*a_c/(2*a_sym)))) for A in A_arr])
    B_arr = np.array([binding_energy_SEMF(Z, A) for Z, A in zip(Z_stable, A_arr)])
    B_per_A = B_arr / A_arr

    # Key nuclei
    nuclei = {
        'H2_deuterium':  {'Z':1, 'A':2,  'B_MeV': binding_energy_SEMF(1,2)},
        'He4_alpha':     {'Z':2, 'A':4,  'B_MeV': binding_energy_SEMF(2,4)},
        'Fe56':          {'Z':26,'A':56, 'B_MeV': binding_energy_SEMF(26,56)},
        'U235':          {'Z':92,'A':235,'B_MeV': binding_energy_SEMF(92,235)},
        'U238':          {'Z':92,'A':238,'B_MeV': binding_energy_SEMF(92,238)},
        'Pu239':         {'Z':94,'A':239,'B_MeV': binding_energy_SEMF(94,239)},
    }
    for name, d in nuclei.items():
        d['B_per_A_MeV'] = d['B_MeV']/d['A'] if d['A']>0 else 0

    return {
        'A_arr': A_arr.tolist(),
        'B_per_A_MeV': B_per_A.tolist(),
        'peak_A': int(A_arr[np.argmax(B_per_A)]),
        'peak_B_per_A': float(np.max(B_per_A)),
        'magic_numbers': [2, 8, 20, 28, 50, 82, 126],
        'nuclei': nuclei,
        'SEMF': 'B = a_v*A - a_s*A^(2/3) - a_c*Z^2*A^(-1/3) - a_sym*(N-Z)^2/A + delta',
        'Fe56_is_maximum': 'Elements < Fe: fuse for energy. Elements > Fe: fission for energy.',
    }


def nuclear_applications():
    """
    Radioactive decay, fission, fusion, medical physics, nuclear vision.

    Radioactive decay: N(t) = N0*exp(-lambda*t), T_{1/2} = ln(2)/lambda
    Activity: A = lambda*N  [Becquerel = 1 decay/s, Curie = 3.7e10 Bq]

    Types:
      Alpha: A -> (A-4) + He-4. Short range (cm in air). Stopped by paper.
      Beta-: n -> p + e- + nu_e. Medium range. Blocked by plastic/Al.
      Beta+: p -> n + e+ + nu_e. Positron -> annihilation -> 2x 511 keV gamma.
      Gamma: nucleus de-excites -> high-energy photon. Long range. Needs lead shielding.

    FISSION: U-235 + n -> Ba-141 + Kr-92 + 3n + 200 MeV
      Chain reaction: k_eff > 1 -> exponential growth (bomb), k_eff=1 -> controlled (reactor)
      Criticality: enough fissile material in right geometry -> self-sustaining chain.

    FUSION: D + T -> He-4 + n + 17.6 MeV
      Requires T ~ 10^8 K (overcomes Coulomb barrier) OR tunneling at lower T.
      ITER (France): 500 MW fusion, 50 MW input (Q=10). First net gain ~2035.

    MEDICAL PHYSICS (nuclear vision):
      PET scan: F-18 beta+ emitter. e+ annihilates e- -> 2x 511 keV gammas back-to-back.
        Coincidence detection -> 3D activity distribution. Phase retrieval in disguise.
      SPECT: Tc-99m, gamma emitter. Single photon detection + collimator.
      Radiation therapy: photon/proton/carbon ion beams to tumor.
        Bragg peak: proton dose deposited at end of range (spares tissue before tumor).
      Radiation sterilization: gamma irradiation kills microorganisms in surgical instruments.

    CAUSALITY:
      Radioactive decay: completely random (cannot predict WHEN atom decays).
        But statistically: N(t) is perfectly causal and deterministic.
        Individual: indeterminate. Ensemble: causal.
        Same as quantum measurement: individual = random, distribution = causal.
      Causal intervention in medicine: radiation CAUSES cell death (mechanism) ->
        if tumor cells die faster than normal cells -> net therapeutic benefit.
        Same causal logic as all pharmacology.
    """
    # Radioactive decay
    isotopes = {
        'F18_PET':   {'T_half': 109.8*60,  'decay': 'beta+', 'use': 'PET scan'},
        'Tc99m':     {'T_half': 6.02*3600, 'decay': 'gamma (141 keV)', 'use': 'SPECT'},
        'I131':      {'T_half': 8.02*86400,'decay': 'beta- + gamma', 'use': 'thyroid therapy'},
        'C14':       {'T_half': 5730*365.25*86400,'decay':'beta-', 'use': 'carbon dating'},
        'U235':      {'T_half': 7.04e8*365.25*86400,'decay':'alpha','use':'nuclear reactor fuel'},
        'Ra226':     {'T_half': 1600*365.25*86400,'decay':'alpha', 'use': 'historical: Curie'},
    }
    for name, d in isotopes.items():
        d['lambda_per_s'] = np.log(2) / d['T_half']
        d['T_half_hours'] = d['T_half'] / 3600

    # Decay curves
    t_F18 = np.linspace(0, 12*3600, 400)   # 12 hours
    lambda_F18 = np.log(2)/(109.8*60)
    N_F18 = np.exp(-lambda_F18 * t_F18)

    # Fission energy
    E_fission_MeV = 200
    E_per_kg_U235 = E_fission_MeV*1e6*eV / (235*mp) * 235*mp  # J/kg actually
    # Correct: E per kg = (E per fission / mass per atom) = E_fission*eV / (A*mp)
    E_per_kg = E_fission_MeV*1e6*eV / (235*1.66e-27)   # J/kg
    E_per_kg_coal = 3.3e7   # J/kg

    return {
        'isotopes': isotopes,
        't_F18_hours': t_F18/3600,
        'N_F18_fraction': N_F18,
        'fission': {
            'reaction': 'U235 + n -> Ba141 + Kr92 + 3n + 200 MeV',
            'E_per_kg_U235_TJ': E_per_kg/1e12,
            'E_per_kg_coal_MJ': E_per_kg_coal/1e6,
            'ratio': E_per_kg/E_per_kg_coal,
        },
        'fusion': {
            'reaction': 'D + T -> He4 + n + 17.6 MeV',
            'ITER_Q': 10,
            'temperature_K': 1e8,
            'status': 'ITER 2035 first plasma. Net gain milestone 2025 (NIF laser achieved 2022)',
        },
        'PET_scanner': {
            'isotope': 'F-18 (FDG, 18-fluorodeoxyglucose)',
            'mechanism': 'e+ + e- -> 2x 511 keV gammas, 180 deg apart',
            'resolution_mm': 4,
            'phase_retrieval': (
                'PET reconstruction = inverse Radon transform (same as CT).\n'
                'TOF-PET: time difference of arrival of two gammas -> 1D position.\n'
                'Full 3D reconstruction = phase retrieval from projections.\n'
                'Same iterative algorithms as GS: MLEM, OSEM (ordered subsets).'
            ),
        },
        'Bragg_peak': {
            'lesson': 'Proton stops at depth R = alpha*E^(1.77). Dose deposited at R.',
            'advantage': 'Spares normal tissue before tumor. Treats tumor precisely.',
        },
        'causality': {
            'individual_atom': 'Cannot predict when individual atom decays (quantum randomness).',
            'ensemble': 'N(t)=N0*exp(-lambda*t): ensemble perfectly causal and predictable.',
            'medical_causality': 'Radiation causes DNA double-strand breaks -> cell death -> tumor regression.',
            'lesson': (
                'Individual quantum events: acausal (random).\n'
                'Statistical ensemble: causal (deterministic).\n'
                'This tension between quantum indeterminacy and classical causality\n'
                'is the deepest open question in foundations of physics.\n'
                'Decoherence bridges the gap: quantum randomness -> classical statistics.'
            ),
        },
    }


def demo():
    print("=== MODERN PHYSICS: 16 CHAPTERS ===\n")

    print("--- SR I: Causality Protection ---")
    sr1 = special_relativity_I()
    m = sr1['muon']
    print(f"  Muon: gamma={m['gamma']:.1f}, lifetime_lab={m['lifetime_lab_us']:.1f} us")
    print(f"  Causality: {sr1['causality']['lesson'][:80]}...")

    print("\n--- SR II: E=mc^2 ---")
    sr2 = special_relativity_II()
    print(f"  1g -> {sr2['E_mc2']['1g_in_megatons_TNT']:.2f} megaton TNT")
    print(f"  Pair production threshold: {sr2['pair_production_threshold_MeV']:.3f} MeV")

    print("\n--- Quantum Theory of Light ---")
    qtl = quantum_theory_of_light()
    pe = qtl['photoelectric']
    print(f"  Cs photoelectric: f_0={pe['threshold_freq_Hz']:.3e} Hz ({pe['threshold_lambda_nm']:.0f}nm)")
    print(f"  Compton lambda_C={qtl['compton']['lambda_C_pm']:.3f} pm")

    print("\n--- Matter Waves ---")
    pm = particle_nature_of_matter()
    for E, lam in zip(pm['de_broglie']['energies_eV'], pm['de_broglie']['lambda_nm']):
        if E <= 100:
            print(f"  E={E:6.0f}eV: lambda={lam:.4f} nm")

    print("\n--- Wave Packets (same as GVD) ---")
    wp = matter_waves_wave_packets()
    print(f"  {wp['identical_math'][:100]}...")

    print("\n--- QM 1D ---")
    qm = qm_1d()
    print(f"  Box 1A: E_n(eV) = {[round(e,2) for e in qm['box']['E_n_eV'][:5]]}")
    print(f"  HO ground state: {qm['harmonic_osc']['zero_point_eV']:.4f} eV (zero-point)")

    print("\n--- Tunneling ---")
    tun = tunneling_phenomena()
    print(f"  kappa = {tun['kappa_per_nm']:.2f} /nm")
    print(f"  STM 1A extra: current x{tun['stm']['T_ratio_1A_increase']:.4f}")
    print(f"  U238 half-life (rough): {tun['alpha_decay_U238']['t_half_years_rough']:.2e} yr")

    print("\n--- Hydrogen Atom ---")
    hyd = qm_3d_hydrogen()
    print(f"  E_n = {[round(v,2) for k,v in list(hyd['E_n_eV'].items())[:5]]} eV")
    print(f"  r_peak(1s) = {hyd['r_peak_1s_ang']:.2f} Angstrom (= a0 = 0.529 A)")
    for line, data in list(hyd['Balmer_series'].items())[:3]:
        print(f"  {line}: {data['lambda_nm']:.1f} nm")

    print("\n--- Statistical Physics ---")
    sp_r = statistical_physics()
    print(f"  Cu Fermi energy: {sp_r['fermi_dirac']['E_F_Cu_eV']} eV")
    print(f"  T_Fermi Cu: {sp_r['fermi_dirac']['T_F_Cu_K']:.0f} K" if 'T_F_Cu_K' in sp_r['fermi_dirac'] else f"  T_Fermi: {sp_r['fermi_dirac']['T_Fermi_Cu_K']:.0f} K")

    print("\n--- Molecular Structure + Solid State ---")
    ms = molecular_structure()
    ss = solid_state()
    for sem, data in list(ss['semiconductors'].items())[:4]:
        print(f"  {sem:10s}: E_g={data['E_g_eV']:.2f} eV  lambda_onset={data['lambda_onset_nm']:.0f} nm")
    print(f"  InGaAsP 1550nm: {ss['InGaAsP_note'][:60]}...")

    print("\n--- Nuclear Structure ---")
    ns = nuclear_structure()
    print(f"  SEMF peak at A={ns['peak_A']}: B/A={ns['peak_B_per_A']:.2f} MeV")
    print(f"  Magic numbers: {ns['magic_numbers']}")

    print("\n--- Nuclear Applications ---")
    na = nuclear_applications()
    for iso, d in list(na['isotopes'].items())[:4]:
        print(f"  {iso:12s}: T_1/2={d['T_half_hours']:.2f} h  ({d['use']})")
    print(f"  U235 vs coal: {na['fission']['ratio']:.0f}x more energy per kg")
    print(f"  Causality: {na['causality']['lesson'][:80]}...")

    print("\n=== MODERN PHYSICS COMPLETE ===")


if __name__ == '__main__':
    demo()
