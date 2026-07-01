"""
Photonics Course Reader: Calculus Solutions
CSUS Physics Minor / ECE self-study

Course reader chapters covered:
  Ch1  Maxwell equations -> wave equation (vector calculus)
  Ch2  Plane waves, dispersion relation, phase/group velocity
  Ch3  Gaussian beam propagation: w(z), R(z), Gouy phase (ODE + complex q-param)
  Ch4  Slab waveguide: TE/TM modes, eigenvalue transcendental equation
  Ch5  Group velocity dispersion: beta2 = d^2 beta/d omega^2 (Taylor expansion)
  Ch6  Coupled mode theory: directional coupler, power transfer (matrix ODE)
  Ch7  Fourier optics: diffraction as FT, 4f system, spatial filtering
  Ch8  SLM as photonics Photoshop -- phase mask computation for beam steering

Connection to this repo:
  - Ch5 GVD = the D in H(f)=exp(j*pi*D*f^2): gs_core.py, coppinger1999.py
  - Ch7 Fourier plane = GS algorithm constraint plane: gs_core.py
  - Ch8 SLM phase mask = computed hologram = same math as InSAR unwrapping
"""
import numpy as np
import sympy as sp

# Constants
c   = 2.99792458e8    # m/s
eps0 = 8.854e-12      # F/m
mu0  = 1.2566e-6      # H/m
h    = 6.626e-34      # J*s
hbar = 1.0546e-34     # J*s
e    = 1.602e-19      # C

# ===========================================================================
# Ch1: Maxwell -> Wave Equation
# ===========================================================================

def maxwell_to_wave_equation():
    """
    Derive the electromagnetic wave equation from Maxwell's equations.

    Maxwell in free space (no charges, no currents):
      (1) curl(E) = -dB/dt = -mu0 * dH/dt    (Faraday)
      (2) curl(H) =  dD/dt =  eps0 * dE/dt   (Ampere-Maxwell)
      (3) div(E)  = 0                          (Gauss electric)
      (4) div(B)  = 0                          (Gauss magnetic)

    Derivation (take curl of eq 1):
      curl(curl(E)) = -mu0 * d/dt(curl(H))
      grad(div(E)) - laplacian(E) = -mu0 * eps0 * d^2E/dt^2
      Since div(E) = 0:
      laplacian(E) = mu0 * eps0 * d^2E/dt^2   <- WAVE EQUATION
      laplacian(E) = (1/c^2) * d^2E/dt^2

    Speed of light falls out of Maxwell's equations:
      c = 1/sqrt(mu0 * eps0) = 2.998e8 m/s   (Maxwell 1865 -- same year as Civil War)

    In a medium with refractive index n:
      laplacian(E) = (n^2/c^2) * d^2E/dt^2
      n = sqrt(eps_r * mu_r)    (Sellmeier equation gives n(omega))
    """
    c_derived = 1.0 / np.sqrt(mu0 * eps0)
    return {
        'c_derived_m_per_s': c_derived,
        'c_exact_m_per_s': c,
        'error_ppm': abs(c_derived - c)/c * 1e6,
        'wave_eq': 'nabla^2 E = (1/c^2) * d^2E/dt^2',
        'plane_wave_solution': 'E(r,t) = E0 * exp(j*(k.r - omega*t))',
        'dispersion_relation': 'k^2 = omega^2 / c^2  ->  k = omega/c = n*omega/c_0',
        'calculus_used': ['curl of curl', 'vector identity: curl(curl)=grad(div)-laplacian', 'div=0 simplification'],
    }


def sympy_wave_equation():
    """Symbolic derivation: verify laplacian of exp(j*k*x)*exp(-j*omega*t) satisfies wave eq."""
    x, t_sym, k_sym, omega_sym, c_sym = sp.symbols('x t k omega c', real=True, positive=True)

    E = sp.exp(sp.I*(k_sym*x - omega_sym*t_sym))

    laplacian_E  = sp.diff(E, x, 2)
    d2E_dt2      = sp.diff(E, t_sym, 2)
    wave_residual = sp.simplify(laplacian_E - (1/c_sym**2)*d2E_dt2)

    # For wave eq to hold: laplacian = (1/c^2)*d^2/dt^2
    # => -k^2 * E = -(omega^2/c^2) * E
    # => k = omega/c

    return {
        'E_plane_wave': E,
        'laplacian_E': laplacian_E,
        'd2E_dt2': d2E_dt2,
        'residual_factored': sp.factor(wave_residual),
        'dispersion_condition': 'k^2 = omega^2/c^2  ->  k = omega/c',
    }


# ===========================================================================
# Ch2: Phase and Group Velocity
# ===========================================================================

def phase_group_velocity(n_func, omega_0, domega=1e9):
    """
    Phase velocity:  v_p = omega/k = c/n(omega)
    Group velocity:  v_g = d(omega)/d(k) = c / (n + omega * dn/domega)

    n_func: callable n(omega), refractive index
    omega_0: center frequency [rad/s]
    domega: step for numerical derivative [rad/s]

    Group index: n_g = c/v_g = n + omega * dn/domega
    GVD: D_gvd = -lambda/c * d^2n/dlambda^2  [ps/(nm km)]

    In a dispersive medium (fiber), v_g depends on omega -> pulse spreading.
    This is the beta2 that drives everything in coppinger1999.py.
    """
    n0 = n_func(omega_0)
    n1 = n_func(omega_0 + domega)
    n_1= n_func(omega_0 - domega)

    dn_domega = (n1 - n_1) / (2*domega)
    d2n_domega2 = (n1 - 2*n0 + n_1) / domega**2

    v_p = c / n0
    n_g = n0 + omega_0 * dn_domega
    v_g = c / n_g

    # beta2 = d^2 beta / d omega^2 = (1/c)(2*dn/domega + omega*d^2n/domega^2)
    beta2 = (1/c) * (2*dn_domega + omega_0*d2n_domega2)

    return {
        'n_0': n0,
        'v_phase_m_per_s': v_p,
        'n_group': n_g,
        'v_group_m_per_s': v_g,
        'dn_domega': dn_domega,
        'beta2_s2_per_m': beta2,
        'beta2_ps2_per_km': beta2 * 1e27,
        'slowdown': v_p / v_g,
        'lesson': 'Phase velocity = wavefront speed. Group velocity = energy/information speed.',
    }


def dispersion_relation_fiber(beta2_ps2km=-20.0, omega_0=None, N=512):
    """
    Taylor expansion of propagation constant around center frequency:
      beta(omega) = beta0 + beta1*(omega-omega0) + beta2/2*(omega-omega0)^2 + ...

    beta1 = 1/v_g  [s/m]  -- group delay
    beta2          [s^2/m] -- group velocity dispersion (GVD)
      beta2 < 0: anomalous dispersion (SMF-28 at 1550nm, beta2 ~ -20 ps^2/km)
      beta2 > 0: normal dispersion   (< 1300nm or LiNbO3 bulk)

    Transfer function H(f) = exp(j*pi*D*f^2) where D = 2*pi^2*beta2*L:
      This IS the Coppinger STEAM operator. GVD is the physics. H(f) is the math.
    """
    if omega_0 is None:
        omega_0 = 2*np.pi*c / 1550e-9   # telecom

    beta2_SI = beta2_ps2km * 1e-27       # s^2/m
    delta_omega = np.linspace(-2*np.pi*100e9, 2*np.pi*100e9, N)  # +/- 100 GHz

    # beta(omega) = beta0 + beta1*domega + beta2/2*domega^2
    beta0 = 2*np.pi / 1550e-9 * 1.4682   # approximate SMF-28
    beta1 = 1 / (c/1.4682)               # approximate
    beta = beta0 + beta1*delta_omega + 0.5*beta2_SI*delta_omega**2

    # Phase accumulated over 1km
    L = 1000  # m
    phase = beta * L

    return {
        'delta_omega_rad_per_s': delta_omega,
        'beta_rad_per_m': beta,
        'phase_accumulated_rad': phase,
        'beta2_ps2_per_km': beta2_ps2km,
        'H_f': np.exp(1j * 0.5 * beta2_SI * L * delta_omega**2),
        'connection': 'H(f) = exp(j*pi*D*f^2) -- the STEAM transfer function',
    }


# ===========================================================================
# Ch3: Gaussian Beam Propagation
# ===========================================================================

def gaussian_beam(w0_um=10.0, wavelength_nm=1550.0, z_mm=None, N=300):
    """
    Gaussian beam: TEM00 mode, the fundamental mode of any laser.

    Complex beam parameter q(z):
      1/q(z) = 1/R(z) - j*lambda/(pi*w(z)^2)

    Propagation: q(z) = q(0) + z  (paraxial approximation)

    Beam parameters:
      w(z)  = w0 * sqrt(1 + (z/z_R)^2)   beam radius [m]
      R(z)  = z * (1 + (z_R/z)^2)         radius of curvature [m]
      phi(z)= arctan(z/z_R)               Gouy phase [rad]
      z_R   = pi*w0^2/lambda              Rayleigh range [m]

    Calculus: w(z) is the solution to the paraxial wave equation
      d^2E/dx^2 + d^2E/dy^2 + 2jk*dE/dz = 0
      (drop d^2E/dz^2 << k^2*E, the paraxial approximation)

    The Gouy phase is a topological phase -- beam picks up pi phase
    through focus. Same math as Berry phase in QM.
    """
    w0   = w0_um * 1e-6        # m
    lam  = wavelength_nm * 1e-9
    z_R  = np.pi * w0**2 / lam  # Rayleigh range [m]
    k    = 2*np.pi / lam

    if z_mm is None:
        z = np.linspace(-5*z_R, 5*z_R, N)
    else:
        z = np.array(z_mm) * 1e-3

    w    = w0 * np.sqrt(1 + (z/z_R)**2)
    R    = np.where(z == 0, np.inf, z * (1 + (z_R/z)**2))
    gouy = np.arctan(z/z_R)

    # On-axis intensity I(0,z) = (w0/w)^2 * I0
    I_norm = (w0/w)**2

    # Electric field amplitude on axis
    E_field = (w0/w) * np.exp(-1j*k*z) * np.exp(1j*gouy)

    return {
        'z_m': z,
        'w_z_um': w*1e6,
        'R_z_m': R,
        'gouy_phase_rad': gouy,
        'I_norm': I_norm,
        'w0_um': w0_um,
        'z_R_mm': z_R*1e3,
        'divergence_angle_mrad': (lam/(np.pi*w0))*1e3,
        'E_field': E_field,
        'lesson': 'Beam waist w0 and Rayleigh range z_R are Fourier conjugates: w0*theta = lambda/pi',
    }


def abcd_matrix_propagation(w0_um=10.0, wavelength_nm=1550.0):
    """
    ABCD ray matrix method for Gaussian beam propagation.
    Complex beam parameter transforms as:
      q_out = (A*q_in + B) / (C*q_in + D)

    System matrices (2x2):
      Free space (L):    [[1, L], [0, 1]]
      Thin lens (f):     [[1, 0], [-1/f, 1]]
      Curved mirror (R): [[1, 0], [-2/R, 1]]
      Flat interface:    [[1, 0], [0, n1/n2]]

    Application: beam expander, fiber coupling, resonator stability.
    det(M) = AD-BC = 1  for any lossless system (symplectic constraint).
    """
    lam = wavelength_nm * 1e-9
    w0  = w0_um * 1e-6
    z_R = np.pi * w0**2 / lam
    q0  = 1j * z_R       # at waist: 1/q = -j*lambda/(pi*w0^2) = 1/(j*z_R)

    def transform_q(M, q):
        A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
        return (A*q + B) / (C*q + D)

    def q_to_beam(q, lam):
        inv_q = 1.0/q
        w = np.sqrt(-lam / (np.pi * inv_q.imag))
        R = 1.0/inv_q.real if inv_q.real != 0 else np.inf
        return w*1e6, R

    # Free space propagation 100mm
    L = 0.1
    M_free = np.array([[1, L], [0, 1]], dtype=complex)
    q_after_free = transform_q(M_free, q0)
    w_free, R_free = q_to_beam(q_after_free, lam)

    # Thin lens f=50mm then free space 50mm (focus at back focal plane)
    f = 0.05
    M_lens = np.array([[1, 0], [-1/f, 1]], dtype=complex)
    M_prop = np.array([[1, f], [0, 1]], dtype=complex)
    M_system = M_prop @ M_lens
    q_focused = transform_q(M_system, q0)
    w_focused, _ = q_to_beam(q_focused, lam)

    return {
        'q0': q0,
        'z_R_mm': z_R*1e3,
        'after_100mm_free_space': {'w_um': w_free, 'R_m': R_free},
        'after_lens_at_focus': {'w_um': w_focused},
        'det_M_system': float(np.real(np.linalg.det(M_system))),
        'lesson': 'ABCD matrix is Mobius transform on complex q. det=1 for any lossless system.',
    }


# ===========================================================================
# Ch4: Slab Waveguide
# ===========================================================================

def slab_waveguide_modes(n_core=1.5, n_clad=1.45, d_um=5.0, wavelength_nm=1310.0):
    """
    Symmetric slab waveguide: TE modes.
    Core refractive index n1, cladding n2 < n1, half-width d.

    Transcendental eigenvalue equation (TE even modes):
      kappa * tan(kappa * d) = gamma

    where:
      kappa = sqrt(n1^2 * k0^2 - beta^2)   transverse wavevector in core
      gamma = sqrt(beta^2 - n2^2 * k0^2)   evanescent decay in cladding
      beta = propagation constant
      k0 = 2*pi/lambda

    Numerical aperture: NA = sqrt(n1^2 - n2^2)
    V-number: V = k0 * d * NA  (single-mode if V < pi/2)

    Calculus: modes are solutions to Helmholtz equation with boundary conditions.
    Same math as particle-in-a-box (Schrodinger) -- both are eigenvalue problems.
    """
    k0   = 2*np.pi / (wavelength_nm*1e-9)
    d    = d_um * 1e-6
    NA   = np.sqrt(n_core**2 - n_clad**2)
    V    = k0 * d * NA

    # Sweep beta from n_clad*k0 to n_core*k0
    beta_min = n_clad * k0
    beta_max = n_core * k0
    beta_arr = np.linspace(beta_min*1.0001, beta_max*0.9999, 5000)

    kappa_arr = np.sqrt(np.maximum(0, n_core**2 * k0**2 - beta_arr**2))
    gamma_arr = np.sqrt(np.maximum(0, beta_arr**2 - n_clad**2 * k0**2))

    # TE even: f(beta) = kappa*tan(kappa*d) - gamma = 0
    f_even = kappa_arr * np.tan(kappa_arr * d) - gamma_arr
    # TE odd: f(beta) = kappa*(-cot(kappa*d)) - gamma = 0
    f_odd  = -kappa_arr / np.tan(kappa_arr * d) - gamma_arr

    # Find zero crossings (mode solutions)
    def find_zeros(f, beta):
        zeros = []
        for i in range(len(f)-1):
            if f[i]*f[i+1] < 0 and abs(f[i]-f[i+1]) < abs(f[i])*2:
                b = beta[i] - f[i]*(beta[i+1]-beta[i])/(f[i+1]-f[i])
                zeros.append(b)
        return zeros

    beta_even = find_zeros(f_even, beta_arr)
    beta_odd  = find_zeros(f_odd,  beta_arr)

    n_eff_even = [b/k0 for b in beta_even]
    n_eff_odd  = [b/k0 for b in beta_odd]

    return {
        'V_number': V,
        'single_mode': V < np.pi/2,
        'NA': NA,
        'n_modes_even': len(n_eff_even),
        'n_modes_odd': len(n_eff_odd),
        'n_eff_even': n_eff_even,
        'n_eff_odd': n_eff_odd,
        'beta_axis': beta_arr,
        'f_even': f_even,
        'f_odd': f_odd,
        'lesson': (
            'Same eigenvalue equation as Schrodinger particle in finite well. '
            'Modes are quantized -- only discrete beta values are allowed. '
            'n_eff = beta/k0 is the effective refractive index of the mode.'
        ),
    }


def evanescent_field(n_core=1.5, n_clad=1.45, beta_norm=None, wavelength_nm=1310.0, N=200):
    """
    Evanescent field: exponentially decaying field in cladding.
    E_clad(y) = E0 * exp(-gamma * (|y| - d))   for |y| > d

    gamma = sqrt(beta^2 - n_clad^2 * k0^2)
    Penetration depth: 1/gamma [m]

    Applications: evanescent coupling (directional coupler), TIRF microscopy,
    frustrated total internal reflection (prism coupler for integrated optics).
    """
    k0 = 2*np.pi / (wavelength_nm*1e-9)
    if beta_norm is None:
        beta_norm = (n_core + n_clad) / 2  # midway
    beta = beta_norm * k0
    gamma = np.sqrt(max(0, beta**2 - n_clad**2 * k0**2))
    pen_depth_nm = 1.0/gamma * 1e9 if gamma > 0 else np.inf

    y = np.linspace(0, 5/gamma if gamma > 0 else 1e-6, N) * 1e9  # nm
    E = np.exp(-gamma * y * 1e-9)

    return {
        'gamma_per_m': gamma,
        'penetration_depth_nm': pen_depth_nm,
        'y_nm': y,
        'E_norm': E,
        'lesson': 'Evanescent field decays as exp(-gamma*y). Basis of fiber sensor, coupler, TIRF.',
    }


# ===========================================================================
# Ch5: Group Velocity Dispersion -- the coppinger1999 connection
# ===========================================================================

def gvd_pulse_broadening(beta2_ps2km=-20.0, L_km=5.0, tau0_ps=1.0):
    """
    Gaussian pulse broadening due to GVD:
      sigma_out = sigma_in * sqrt(1 + (L/L_D)^2)

    where L_D = tau0^2 / |beta2|  is the dispersion length [km]

    Frequency-to-time mapping (STEAM/Coppinger):
      If L >> L_D:  sigma_out ~ L*|beta2| / tau0 = beta2 * L / tau0
      Time spread = beta2*L*Omega  where Omega is spectral bandwidth

    Instantaneous frequency (chirp):
      After dispersive propagation: f_inst(t) = -t / (2*pi*beta2*L)
      Linear chirp -> frequency IS time -> maps spectrum to time (STEAM)

    This is Eq(1) of Coppinger 1999: the heart of photonic time-stretch.
    """
    beta2_SI = beta2_ps2km * 1e-27   # s^2/m
    L_SI = L_km * 1e3                 # m
    tau0_SI = tau0_ps * 1e-12         # s

    L_D = tau0_SI**2 / abs(beta2_SI)   # dispersion length [m]
    tau_out = tau0_SI * np.sqrt(1 + (L_SI/L_D)**2)
    M_stretch = tau_out / tau0_SI      # temporal stretch factor

    # Time array
    t = np.linspace(-5*tau_out, 5*tau_out, 1000)

    # Input pulse (Gaussian)
    E_in  = np.exp(-t**2 / (2*tau0_SI**2))

    # Output pulse (broadened Gaussian with chirp)
    W = tau0_SI**2 - 1j*2*beta2_SI*L_SI
    E_out = (tau0_SI / np.sqrt(W)) * np.exp(-t**2 / W)
    I_out = np.abs(E_out)**2

    # Instantaneous frequency (linear chirp after dispersive propagation)
    f_inst = -t / (2*np.pi * beta2_SI * L_SI)

    return {
        't_ps': t*1e12,
        'E_in': np.abs(E_in)**2,
        'I_out': I_out / I_out.max(),
        'f_inst_GHz': f_inst * 1e-9,
        'tau0_ps': tau0_ps,
        'tau_out_ps': tau_out*1e12,
        'M_stretch': M_stretch,
        'L_D_km': L_D*1e-3,
        'beta2_ps2_per_km': beta2_ps2km,
        'chirp_slope_GHz_per_ps': -1/(2*np.pi*beta2_SI*L_SI) * 1e-9 * 1e-12,
        'connection': (
            'f_inst(t) = -t/(2*pi*beta2*L). Frequency is LINEAR in time. '
            'This is the STEAM mapping: spectrum -> time. '
            'M = 1 + L2/L1 in coppinger1999.'
        ),
    }


def beta2_from_sellmeier(coeffs, wavelength_nm=1550.0):
    """
    Compute GVD beta2 from Sellmeier equation using automatic differentiation (sympy).

    Sellmeier: n^2(lambda) = 1 + sum_i A_i*lambda^2 / (lambda^2 - B_i^2)
    beta(omega) = n(omega)*omega/c
    beta2 = d^2beta/domega^2 |_{omega0}

    coeffs: list of (A_i, B_i_um) Sellmeier terms
    """
    lam = sp.Symbol('lambda', positive=True)
    n2 = 1
    for A, B in coeffs:
        n2 += A * lam**2 / (lam**2 - B**2)
    n_expr = sp.sqrt(n2)

    # Convert to omega: lambda = 2*pi*c/omega -> lam_um = 2*pi*c_um_ps/omega_ps
    # For GVD calculation use numeric differentiation at lambda0
    lam0 = wavelength_nm * 1e-3   # um

    # Numeric evaluation
    n_func = sp.lambdify(lam, n_expr, 'numpy')

    dlam = 1e-5   # um
    n0 = float(n_func(lam0))
    n1 = float(n_func(lam0 + dlam))
    n_1= float(n_func(lam0 - dlam))

    # dn/dlambda, d^2n/dlambda^2
    dn_dlam = (n1 - n_1) / (2*dlam)
    d2n_dlam2 = (n1 - 2*n0 + n_1) / dlam**2

    # beta2 = (lambda^3 / (2*pi*c^2)) * d^2n/dlambda^2
    # [ps^2/km] when lambda in um, c in um/ps
    c_um_ps = 2.998e2
    lam0_um = lam0
    beta2_ps2km = (lam0_um**3 / (2*np.pi * c_um_ps**2)) * d2n_dlam2 * 1e3

    return {
        'n_at_lambda': n0,
        'dn_dlambda_per_um': dn_dlam,
        'd2n_dlambda2_per_um2': d2n_dlam2,
        'beta2_ps2_per_km': beta2_ps2km,
        'ZDW_note': 'ZDW where d^2n/dlambda^2 = 0. SMF-28 ZDW ~ 1310nm.',
    }


# ===========================================================================
# Ch6: Coupled Mode Theory
# ===========================================================================

def directional_coupler(kappa_per_m=100.0, L_m=0.01, delta_beta=0.0):
    """
    Directional coupler: two parallel waveguides exchange power via evanescent coupling.
    Coupled mode equations:
      da/dz = -j*kappa*b
      db/dz = -j*kappa*a   (phase-matched: delta_beta=0)

    Solution (power started in waveguide a):
      P_a(z) = cos^2(kappa*z)
      P_b(z) = sin^2(kappa*z)

    With phase mismatch delta_beta:
      kappa_eff = sqrt(kappa^2 + (delta_beta/2)^2)
      P_b_max = (kappa/kappa_eff)^2

    Transfer length: L_pi = pi/(2*kappa)  [half-coupling]

    Applications: MZI (modulator, beamsplitter), WDM filter, fiber coupler.
    Connection to repo: MZM in coppinger1999 is an integrated directional coupler.
    """
    z = np.linspace(0, L_m, 500)
    kappa_eff = np.sqrt(kappa_per_m**2 + (delta_beta/2)**2)

    P_a = 1 - (kappa_per_m/kappa_eff)**2 * np.sin(kappa_eff*z)**2
    P_b = (kappa_per_m/kappa_eff)**2 * np.sin(kappa_eff*z)**2

    L_pi = np.pi / (2*kappa_per_m)
    max_transfer = (kappa_per_m/kappa_eff)**2

    return {
        'z_mm': z*1e3,
        'P_a': P_a,
        'P_b': P_b,
        'L_pi_mm': L_pi*1e3,
        'max_power_transfer': max_transfer,
        'kappa_per_m': kappa_per_m,
        'delta_beta': delta_beta,
        'lesson': (
            'Coupled mode = matrix ODE: d[a,b]/dz = j*[[0,kappa],[kappa,0]]*[a,b]. '
            'Eigenvalues +/-kappa. Solution: sinusoidal power exchange. '
            'Phase mismatch reduces coupling: why WDM channels dont cross-couple.'
        ),
    }


def mzi_transfer_function(phi_arm, splitting_ratio=0.5):
    """
    Mach-Zehnder Interferometer transfer function.
    Input E_in splits 50/50, travels two arms, recombines.

    T = cos^2(phi/2)   (power, phase-matched 50/50 coupler)

    phi = (2*pi/lambda) * delta_n * L   (phase difference between arms)
       = (2*pi*n_e^3*r33*V*L) / (lambda*d)   (electro-optic: LiNbO3 MZM)

    V_pi = lambda*d / (n_e^3 * r33 * L)  [volts]  <- from lnbo3.py

    Bias at quadrature (phi = pi/2): maximum linear response, dT/dphi maximum.
    """
    phi = np.array(phi_arm, dtype=float)
    T_bar  = splitting_ratio * (1 - np.cos(phi)) + (1-splitting_ratio)*(1+np.cos(phi))
    T_cross= splitting_ratio * (1 + np.cos(phi)) + (1-splitting_ratio)*(1-np.cos(phi))
    # Simpler 50/50:
    T_out = np.cos(phi/2)**2
    return {
        'phi_rad': phi,
        'T_transmission': T_out,
        'quadrature_phi': np.pi/2,
        'V_pi_note': 'V_pi is the voltage that shifts phi by pi -> T goes 1->0',
        'lesson': 'MZI = analog cosine function. Bias matters: quadrature=linear, null=switch.',
    }


# ===========================================================================
# Ch7: Fourier Optics
# ===========================================================================

def fraunhofer_diffraction(aperture_func, lam_nm=633.0, z_m=1.0, N=512):
    """
    Fraunhofer (far-field) diffraction = Fourier transform of aperture.
    U(x, z) = FT[U_aperture(x0)] evaluated at f_x = x/(lambda*z)

    This is why a lens performs a Fourier transform:
    - Object at front focal plane -> FT at back focal plane
    - 4f system: FT -> filter -> inverse FT
    - Spatial light modulator (SLM) in Fourier plane = frequency domain filter

    aperture_func: array of complex aperture transmission (N points)
    Returns: far-field intensity pattern
    """
    U_ap = np.array(aperture_func, dtype=complex)
    U_ff = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(U_ap)))
    I_ff = np.abs(U_ff)**2

    lam = lam_nm * 1e-9
    dx_ap = 10e-6   # 10um pixel pitch (SLM-like)
    x_ff = np.fft.fftshift(np.fft.fftfreq(N, d=dx_ap)) * lam * z_m

    return {
        'x_mm': x_ff*1e3,
        'I_farfield': I_ff / I_ff.max(),
        'U_farfield': U_ff,
        'lesson': 'Far-field = FT of aperture. Lens does FT optically. SLM programs aperture.',
    }


def single_slit_diffraction(a_um=100.0, lam_nm=633.0, N=1024):
    """
    Single slit: U(x) = rect(x/a)
    FT[rect] = a * sinc(a*f_x) = a * sin(pi*a*f_x)/(pi*a*f_x)

    Minima at: a*sin(theta) = m*lambda, m=1,2,3...
    Central max width: 2*lambda/a

    Calculus: integrate E = E0 * exp(j*k*x*sin(theta)) over slit [-a/2, a/2]:
      E_ff = E0 * integral_{-a/2}^{a/2} exp(j*k*x*sin(theta)) dx
           = E0 * a * sinc(a*sin(theta)/lambda)
    """
    a = a_um * 1e-6
    lam = lam_nm * 1e-9
    dx = a / N

    x_ap = np.linspace(-5*a, 5*a, N)
    U_ap = np.where(np.abs(x_ap) < a/2, 1.0+0j, 0.0+0j)

    result = fraunhofer_diffraction(U_ap, lam_nm=lam_nm, z_m=1.0, N=N)

    # Analytic sinc
    f_x = np.linspace(-3/(a), 3/(a), N)
    sinc_arg = a * f_x
    I_analytic = np.sinc(sinc_arg)**2

    return {
        **result,
        'I_analytic_sinc': I_analytic,
        'first_min_angle_deg': np.arcsin(lam/a)*180/np.pi,
        'central_max_width_um': 2*lam/a * 1e6,
    }


# ===========================================================================
# Ch8: SLM as Photonics Photoshop -- Defensive Hardware
# ===========================================================================

def slm_phase_mask(target_intensity, wavelength_nm=1064.0, iterations=50):
    """
    Spatial Light Modulator (SLM) = photonics Photoshop.

    An SLM is a 2D array of pixels that each add an independently controlled
    phase 0 to 2*pi to the reflected/transmitted light.
    Programming an SLM = writing a 2D phase image.

    This IS Photoshop for light:
      - Photoshop: writes RGB values to pixels
      - SLM: writes phase values (0-2*pi) to pixels
      - Effect: shapes the laser beam in amplitude AND phase

    Defensive / directed energy hardware uses:
      1. Beam steering: phase ramp across SLM -> steer beam without moving parts
         phi(x) = 2*pi*x*sin(theta)/lambda  -> blazed grating -> deflects beam
      2. Beam shaping: focus to arbitrary pattern (DARPA HELLADS, LOCUST anti-drone)
      3. Wavefront correction: measure aberration, apply conjugate phase -> clean beam
      4. Optical jamming: shape beam to fill target with specific spatial pattern

    Algorithm: Gerchberg-Saxton (this repo's gs_core.py!) computes the SLM phase.
    GS recovers phase from two intensity measurements:
      - Near field (SLM plane) = uniform amplitude, unknown phase -> OPTIMIZE
      - Far field (target plane) = desired intensity pattern
    Same math as STEAM phase retrieval.

    target_intensity: 1D array, desired far-field beam shape
    Returns: SLM phase mask + achieved intensity
    """
    N = len(target_intensity)
    target_amp = np.sqrt(np.maximum(0, target_intensity / np.max(target_intensity)))

    # Gerchberg-Saxton algorithm
    phase = np.random.default_rng(42).uniform(0, 2*np.pi, N)
    A = np.ones(N)  # uniform amplitude in SLM plane

    history = []
    for i in range(iterations):
        # Forward FT (SLM -> far field)
        U_ff = np.fft.fftshift(np.fft.fft(A * np.exp(1j*phase)))
        # Replace far-field amplitude with target, keep phase
        phase_ff = np.angle(U_ff)
        U_ff_constrained = target_amp * np.exp(1j*phase_ff)
        # Inverse FT (far field -> SLM)
        U_slm = np.fft.ifft(np.fft.ifftshift(U_ff_constrained))
        # Keep SLM amplitude = 1, update phase
        phase = np.angle(U_slm)
        # Track convergence
        I_achieved = np.abs(U_ff_constrained)**2
        corr = np.corrcoef(I_achieved, target_intensity)[0,1]
        history.append(corr)

    # Final forward pass
    U_ff_final = np.fft.fftshift(np.fft.fft(np.exp(1j*phase)))
    I_final = np.abs(U_ff_final)**2

    return {
        'slm_phase_rad': phase,
        'I_achieved': I_final / I_final.max(),
        'target': target_amp**2,
        'convergence': history,
        'final_correlation': history[-1],
        'beam_steering': 'phi(x) = 2*pi*x*sin(theta)/lambda  (phase ramp = grating)',
        'algorithm': 'Gerchberg-Saxton (same as gs_core.py)',
        'defense_apps': [
            'HELLADS (High Energy Liquid Laser): 150kW class, SLM beam steering',
            'LOCUST: Navy drone swarm defeat, uses phased array + SLM principles',
            'LADAR: laser radar, GS phase retrieval for 3D imaging through turbulence',
            'Adaptive optics: Keck telescope, USAF atmospheric compensation',
        ],
    }


def beam_steering_phase_ramp(theta_deg, wavelength_nm=1064.0, N=256, pixel_pitch_um=8.0):
    """
    Steer a beam by angle theta using a linear phase ramp on SLM.
    phi(x) = 2*pi * x * sin(theta) / lambda

    Equivalent to a blazed diffraction grating with period:
      Lambda = lambda / sin(theta)

    This is how LIDAR systems steer without mechanical moving parts.
    Same principle as phased array radar (AESA), just optical.

    Defense connection:
    - F-35 EOTS uses this for laser designation without moving mirror
    - Starlink: optical inter-satellite link uses beam steering
    - Anti-drone: CRAM (Counter-Rocket Artillery Mortar) systems
    """
    lam = wavelength_nm * 1e-9
    p   = pixel_pitch_um * 1e-6
    theta = np.radians(theta_deg)

    x = np.arange(N) * p
    phi_ideal = 2*np.pi * x * np.sin(theta) / lam
    phi_slm   = np.mod(phi_ideal, 2*np.pi)  # wrap to [0, 2*pi]

    # Diffraction efficiency (for wrapped phase, ~100% for ideal blaze)
    efficiency = (np.sinc(1 - theta*p/lam))**2 if theta > 0 else 1.0

    # Achievable steering range: +/- lambda/(2*p)  [Nyquist limit]
    theta_max_deg = np.degrees(np.arcsin(lam/(2*p)))

    return {
        'x_mm': x*1e3,
        'phi_ideal_rad': phi_ideal,
        'phi_slm_rad': phi_slm,
        'theta_deg': theta_deg,
        'grating_period_um': lam/np.sin(theta)*1e6 if theta > 0 else np.inf,
        'max_steering_angle_deg': theta_max_deg,
        'lesson': (
            'SLM phase ramp = blazed grating. No moving parts. '
            'kHz steering rates (vs Hz for galvo mirror). '
            'This is how adaptive optics in DARPA/AFRL systems work.'
        ),
    }


def demo():
    print("=== PHOTONICS COURSE READER: CALCULUS SOLUTIONS ===\n")

    print("Ch1: Maxwell -> Wave Equation")
    mxw = maxwell_to_wave_equation()
    print(f"  c derived from mu0,eps0: {mxw['c_derived_m_per_s']:.6e} m/s")
    print(f"  Error: {mxw['error_ppm']:.3f} ppm")

    print("\nCh2: Phase/Group Velocity (SMF-28 at 1550nm)")
    # SMF-28 approximate n(omega) around 1550nm
    omega0 = 2*np.pi*c/1550e-9
    n_smf28 = lambda omega: 1.4682 - 0.004*(omega/omega0 - 1)
    pg = phase_group_velocity(n_smf28, omega0)
    print(f"  v_phase = {pg['v_phase_m_per_s']/c:.6f} c")
    print(f"  v_group = {pg['v_group_m_per_s']/c:.6f} c")

    print("\nCh3: Gaussian Beam (w0=10um, 1550nm)")
    gb = gaussian_beam(w0_um=10.0, wavelength_nm=1550.0)
    print(f"  Rayleigh range z_R = {gb['z_R_mm']:.2f} mm")
    print(f"  Divergence = {gb['divergence_angle_mrad']:.1f} mrad")

    print("\nCh4: Slab Waveguide (n_core=1.5, d=5um, 1310nm)")
    swg = slab_waveguide_modes()
    print(f"  V = {swg['V_number']:.3f}, single-mode: {swg['single_mode']}")
    print(f"  n_eff (even modes): {[round(n,5) for n in swg['n_eff_even']]}")

    print("\nCh5: GVD Pulse Broadening (beta2=-20 ps^2/km, L=5km, tau0=1ps)")
    gvd = gvd_pulse_broadening()
    print(f"  L_D = {gvd['L_D_km']:.3f} km")
    print(f"  tau_out = {gvd['tau_out_ps']:.2f} ps (stretch M={gvd['M_stretch']:.1f}x)")
    print(f"  Chirp slope = {gvd['chirp_slope_GHz_per_ps']:.3f} GHz/ps")

    print("\nCh6: Directional Coupler (kappa=100/m, L=1cm)")
    dc = directional_coupler(kappa_per_m=100.0, L_m=0.01)
    print(f"  Transfer length L_pi = {dc['L_pi_mm']:.1f} mm")
    print(f"  Max power transfer = {dc['max_power_transfer']*100:.0f}%")

    print("\nCh7: Single Slit Diffraction (a=100um, 633nm)")
    ss = single_slit_diffraction()
    print(f"  First minimum at {ss['first_min_angle_deg']:.3f} deg")

    print("\nCh8: SLM Beam Steering (10 degrees, 1064nm)")
    bs = beam_steering_phase_ramp(theta_deg=10.0)
    print(f"  Grating period = {bs['grating_period_um']:.2f} um")
    print(f"  Max steering angle = +/- {bs['max_steering_angle_deg']:.1f} deg")

    print("\nCh8: GS Phase Mask (flat-top beam)")
    N = 64
    target = np.ones(N)
    target[0:10] = 0; target[54:] = 0
    slm = slm_phase_mask(target, iterations=30)
    print(f"  GS converged: final correlation = {slm['final_correlation']:.4f}")
    print(f"  Defense apps: {slm['defense_apps'][0]}")

    print("\n=== DONE ===")


if __name__ == '__main__':
    demo()
