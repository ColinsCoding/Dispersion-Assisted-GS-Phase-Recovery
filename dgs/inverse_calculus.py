"""
Calculus Backwards: Antiderivatives, Inverse Problems, and Deconvolution

"Backwards calculus" means three different things depending on context:

  1. ANTIDERIVATIVES (Calc 1/2)
     Given f'(x), find f(x).
     Techniques: pattern recognition, u-sub, IBP, partial fractions, trig sub.
     Fundamental Theorem: integral IS the inverse of derivative.

  2. FEYNMAN TECHNIQUE (Calc 3 / Physics)
     Differentiation under the integral sign: d/d(param) of integral = new integral.
     Lets you solve integrals that stumped everyone else in the room.
     Feynman learned it from an old calculus book. Used it his whole career.

  3. INVERSE PROBLEMS (Engineering / Research)
     Given the OUTPUT of a physical system, find the INPUT.
     d/dt[x(t)] = y(t) is forward. Given y(t), find x(t) = integration.
     But physics is almost always: you measure y (blurred/noisy), recover x.
     This is what GS phase retrieval does: given |E(t)|^2, find E(t).

Connection to this repo:
  - GS algorithm = inverse problem: measurement -> field
  - Deconvolution = inverse of convolution = inverse of H(f) filter
  - Beta2 dispersion = forward problem. Phase retrieval = backwards.
  - Every Nobel Prize in imaging is for solving an inverse problem better.
"""
import numpy as np
import sympy as sp

# ===========================================================================
# Part 1: Antiderivatives (the professor technique)
# ===========================================================================

def integration_techniques():
    """
    The professor's toolkit for doing calculus backwards.
    Pattern: look at what you have, ask what derivative PRODUCES that.

    LIATE rule for Integration by Parts:
      Choose u in order: Logs, Inverse trig, Algebraic, Trig, Exponential
      int(u dv) = uv - int(v du)

    Professor trick: differentiate your guess, compare to integrand.
    If it matches with a constant factor -- you're done.
    """
    x, t, a, b, n, k = sp.symbols('x t a b n k', positive=True)
    omega = sp.Symbol('omega', real=True)

    # Power rule backwards
    power = sp.integrate(x**n, x)

    # Exponential
    exp_int = sp.integrate(sp.exp(-a*x), x)

    # Gaussian integral (the big one -- appears in QM, statistics, optics)
    gaussian = sp.integrate(sp.exp(-a*x**2), (x, -sp.oo, sp.oo))

    # Integration by parts: x*exp(-x)
    ibp = sp.integrate(x * sp.exp(-x), x)

    # Trig substitution: sqrt(1-x^2)
    x2 = sp.Symbol('x2')
    trig_sub = sp.integrate(sp.sqrt(1 - x2**2), (x2, -1, 1))

    # Partial fractions: 1/(x^2-1)
    x3 = sp.Symbol('x3')
    pf = sp.integrate(1/(x3**2 - 1), x3)

    return {
        'power_rule_backwards': f'd/dx[x^(n+1)/(n+1)] = x^n  =>  int(x^n)dx = {power}',
        'exponential': f'int(exp(-a*x))dx = {exp_int}',
        'gaussian_integral': f'int_(-inf to inf) exp(-a*x^2) dx = {gaussian}  (used in QM, optics)',
        'by_parts_x_exp': f'int(x*exp(-x))dx = {ibp}  [IBP: u=x, dv=exp(-x)dx]',
        'trig_sub_circle_area': f'int_(-1 to 1) sqrt(1-x^2) dx = {trig_sub} = pi/2  (half-circle)',
        'partial_fractions': f'int(1/(x^2-1))dx = {pf}',
        'professor_trick': (
            'Guess the form, differentiate, compare. '
            'int(x*cos(x))dx: guess x*sin(x), diff = sin(x)+x*cos(x), '
            'need to subtract int(sin(x))dx = -cos(x). '
            'Answer: x*sin(x) + cos(x) + C'
        ),
    }


def feynman_technique_demo():
    """
    Feynman's trick: differentiation under the integral sign.

    Classic example: I(a) = int_0^inf exp(-a*x^2) dx = sqrt(pi)/(2*sqrt(a))
    But also: int_0^inf sin(x)/x dx = pi/2

    Method for int_0^inf sin(x)/x dx:
    1. Define I(a) = int_0^inf exp(-a*x)*sin(x)/x dx
    2. dI/da = -int_0^inf exp(-a*x)*sin(x) dx = -1/(1+a^2)
    3. I(a) = -arctan(a) + C
    4. I(inf) = 0 (integral vanishes) -> C = pi/2
    5. I(0) = int sin(x)/x dx = pi/2  QED

    This is how Feynman solved integrals no one else could at Caltech.
    Same trick used in quantum field theory (path integrals).
    """
    a, x = sp.symbols('a x', positive=True)

    # Step 2: dI/da
    integrand = sp.exp(-a*x) * sp.sin(x)
    dI_da = -sp.integrate(integrand, (x, 0, sp.oo))
    dI_da_simplified = sp.simplify(dI_da)

    # Step 3: integrate -1/(1+a^2) da
    I_a = sp.integrate(dI_da_simplified, a)

    # Step 4/5: apply boundary condition I(inf) = 0
    I_0 = sp.limit(I_a, a, 0)

    # Also: Gaussian integral with parameter
    I_gauss = sp.integrate(sp.exp(-a*x**2), (x, -sp.oo, sp.oo))
    dI_da_gauss = sp.diff(I_gauss, a)

    return {
        'dI_da': dI_da_simplified,
        'I_a_indefinite': I_a,
        'sinc_integral_result': sp.pi/2,
        'gaussian_with_param': I_gauss,
        'dI_da_gaussian': dI_da_simplified,
        'lesson': (
            'Feynman technique: introduce parameter, differentiate, solve simpler integral, '
            'integrate back, apply boundary condition. '
            'Works when direct integration fails. '
            'Equivalent to Laplace transform trick used in circuit analysis.'
        ),
        'engineering_use': (
            'Laplace transform of impulse response: H(s) = int_0^inf h(t)*exp(-s*t) dt. '
            'Same structure: integral with parameter s. '
            'Inverse Laplace = doing calculus backwards for circuit design.'
        ),
    }


def fundamental_theorem_calculus():
    """
    The Fundamental Theorem: d/dx [int_a^x f(t) dt] = f(x)

    Part 1: Differentiation and integration are EXACT inverses.
    Part 2: int_a^b f(x) dx = F(b) - F(a) where F'=f.

    This is the most important theorem in all of calculus.
    Every numerical method for ODEs (Euler, RK4) is a discrete FTC.
    Every neural network layer is a discrete integral.
    Every GS iteration integrates (FT) then projects (constraint).

    Physical meaning in photonics:
      Group delay: tau(omega) = -d(phi)/d(omega)  [derivative of phase]
      Group delay integral: phi(omega) = -int tau(omega) d(omega)  [FTC backwards]
      Given measured phase, recover group delay. Given group delay, recover phase.
      This is dispersion characterization: the Kramers-Kronig relations.
    """
    x, t, a, b = sp.symbols('x t a b', real=True)
    f = sp.Function('f')

    # FTC Part 1
    F = sp.Integral(f(t), (t, a, x))
    dF_dx = sp.diff(F, x)

    return {
        'FTC_part1': f'd/dx [int_a^x f(t) dt] = f(x)',
        'FTC_part2': 'int_a^b f(x) dx = F(b) - F(a),  F\'(x)=f(x)',
        'sympy_FTC': str(dF_dx),
        'euler_method': 'x(t+dt) = x(t) + f(t)*dt  <-- discrete FTC, 1st order',
        'rk4_is_ftc': 'RK4 = weighted average of f at 4 points in [t, t+dt]',
        'neural_network': 'ResNet: x[l+1] = x[l] + F(x[l])  <-- Euler integration in depth',
        'kramers_kronig': (
            'KK: real(n) = 1 + (2/pi)*PV int_0^inf omega\'*imag(n(omega\'))/(omega\'^2-omega^2) d(omega\')\n'
            'Given absorption (imag part), recover refractive index (real part). '
            'Causality forces this: you cannot have absorption without dispersion. '
            'dgs/causality.py implements this numerically.'
        ),
    }


# ===========================================================================
# Part 2: Bioluminescence + Absorption (Beer-Lambert)
# ===========================================================================

def bioluminescence_physics():
    """
    Bioluminescence: light from chemistry, not heat.
    Firefly: luciferin + ATP + O2 --[luciferase]--> oxyluciferin + CO2 + hv

    The photon emission is quantum mechanics:
      1. Chemical energy -> excited electronic state (S1)
      2. Spontaneous emission: S1 -> S0 + photon
         Rate: k_r = A_21 (Einstein A coefficient, same as Purcell effect)
      3. Quantum yield: phi_Q = k_r / (k_r + k_nr)
         k_nr = non-radiative decay (heat, internal conversion)

    Emission spectrum: Lorentzian centered at E_photon = hf
      For firefly: peak ~560nm (green-yellow)
      Deep-sea bioluminescence: ~480nm (blue, water-transparent window)

    Beer-Lambert law: I(z) = I0 * exp(-alpha * z)
      alpha = absorption coefficient [1/m] = 4*pi*k/lambda
      k = extinction coefficient (imaginary part of complex n)
      n_complex = n_real + j*k

    Same exponential decay as RC circuit discharge and radioactive decay.
    Same math as evanescent field in waveguide (Ch4 of photonics_calculus.py).
    """
    h = 6.626e-34; c = 2.998e8; eV = 1.602e-19
    lam_firefly = 560e-9   # m, peak emission
    E_photon_eV = h*c/lam_firefly / eV

    # Quantum yield typical values
    species = {
        'firefly_Photinus_pyralis': {'phi_Q': 0.41, 'lambda_nm': 560, 'color': 'green-yellow'},
        'dinoflagellate_Noctiluca':  {'phi_Q': 0.15, 'lambda_nm': 475, 'color': 'blue'},
        'jellyfish_Aequorea_victoria': {'phi_Q': 0.77, 'lambda_nm': 509, 'color': 'green'},
        'deep_sea_anglerfish':       {'phi_Q': 0.10, 'lambda_nm': 490, 'color': 'blue'},
        'GFP_green_fluorescent_protein': {'phi_Q': 0.79, 'lambda_nm': 509, 'color': 'green'},
    }

    # Emission spectrum: Gaussian approximation
    lam = np.linspace(400, 700, 300)
    spectrum_firefly = np.exp(-(lam - 560)**2 / (2*30**2))
    spectrum_deep = np.exp(-(lam - 475)**2 / (2*25**2))

    return {
        'photon_energy_eV': E_photon_eV,
        'species': species,
        'lam_nm': lam,
        'spectrum_firefly': spectrum_firefly,
        'spectrum_deep': spectrum_deep,
        'chemistry': 'luciferin + O2 + ATP --[luciferase]--> oxyluciferin + CO2 + hv',
        'QM_mechanism': 'Chemical energy -> S1 excited state -> spontaneous emission (A_21 coefficient)',
        'water_transparency': 'Ocean transparent at 480nm: why deep-sea life evolved blue bioluminescence',
        'engineering_use': 'GFP: genetic marker in biology. Same photon physics as laser dye gain medium.',
        'lesson': 'Bioluminescence = chemistry-pumped quantum emitter. No heat required. phi_Q < 1 due to k_nr.',
    }


def beer_lambert_absorption(alpha_per_cm=1.0, L_cm=None, I0=1.0, N=500):
    """
    Beer-Lambert Law: I(z) = I0 * exp(-alpha * z)
    alpha: absorption coefficient [1/cm] or [1/m]

    Equivalent forms:
      Transmittance: T = I/I0 = exp(-alpha*L) = 10^(-A)  where A = absorbance
      Absorbance:    A = alpha*L / ln(10) = epsilon * c * L  (Beer-Lambert for solutions)

    Physics: alpha = 4*pi*k/lambda
      k = imaginary part of complex refractive index
      Every photon absorbed -> electron excited to higher state

    Same math as:
      - RC discharge: V(t) = V0*exp(-t/tau)
      - Radioactive decay: N(t) = N0*exp(-lambda*t)
      - Evanescent field: E(y) = E0*exp(-gamma*y)
      - Gravitational potential: phi(r) ~ exp(-r/r0) (Yukawa potential in nuclear physics)

    The exponential is universal because these are all FIRST-ORDER ODEs:
      dI/dz = -alpha * I  ->  I(z) = I0 * exp(-alpha*z)
      The rate of change is proportional to the current value.
    """
    if L_cm is None:
        span = 5/alpha_per_cm if alpha_per_cm > 0 else 10.0
        L_cm = np.linspace(0, span, N)
    L_arr = np.array(L_cm)

    I = I0 * np.exp(-alpha_per_cm * L_arr)
    T = I / I0
    A = -np.log10(np.maximum(T, 1e-15))   # absorbance

    # 1/e depth
    depth_1e = 1.0 / alpha_per_cm if alpha_per_cm > 0 else float('inf')
    depth_10pct = np.log(10) / alpha_per_cm if alpha_per_cm > 0 else float('inf')

    # Complex refractive index connection
    lam_nm = 550.0
    k_ext = alpha_per_cm * lam_nm*1e-7 / (4*np.pi)

    return {
        'L_cm': L_arr,
        'I': I,
        'T': T,
        'absorbance_A': A,
        'depth_1_over_e_cm': depth_1e,
        'depth_to_10percent_cm': depth_10pct,
        'k_extinction_coefficient': k_ext,
        'ODE': 'dI/dz = -alpha*I  same form as dV/dt=-V/RC, dN/dt=-lambda*N',
        'lesson': 'Exponential decay = first-order ODE. Universal in physics.',
    }


def complex_refractive_index(n_real=1.5, k_extinction=0.01, wavelength_nm=550.0):
    """
    Complex refractive index: n_complex = n + j*k
      n: phase velocity (real part)
      k: absorption / extinction coefficient (imaginary part)

    Plane wave in absorbing medium:
      E(z) = E0 * exp(j*(n+jk)*k0*z) = E0 * exp(-k*k0*z) * exp(j*n*k0*z)
                                          ^absorption         ^phase

    Relations:
      alpha = 2*omega*k/c = 4*pi*k/lambda  [absorption coefficient, 1/m]
      delta = lambda/(4*pi*k)              [skin depth, m]

    Kramers-Kronig: n(omega) and k(omega) are NOT independent.
    If you know k(omega) (absorption spectrum), you can compute n(omega).
    This is causality: the real and imaginary parts of any causal system's
    transfer function are Hilbert transform pairs.

    Same as: if you know Im[H(f)] for a causal filter, Re[H(f)] is determined.
    dgs/causality.py implements this via FFT Hilbert transform.
    """
    k0 = 2*np.pi / (wavelength_nm*1e-9)
    alpha = 4*np.pi*k_extinction / (wavelength_nm*1e-9)
    delta = 1.0/alpha if alpha > 0 else float('inf')   # skin depth

    z = np.linspace(0, 5*delta, 400)
    E_real = np.exp(-k_extinction*k0*z) * np.cos(n_real*k0*z)
    I_z = np.exp(-alpha*z)

    phase_velocity = 3e8 / n_real
    group_delay_per_m = n_real / 3e8

    return {
        'n_complex': complex(n_real, k_extinction),
        'alpha_per_m': alpha,
        'alpha_per_cm': alpha/100,
        'skin_depth_nm': delta*1e9,
        'z_m': z,
        'E_real': E_real,
        'I_z': I_z,
        'phase_velocity_m_per_s': phase_velocity,
        'kramers_kronig': 'Re(n) and Im(n) are Hilbert transform pairs (causality)',
        'laser_medium': (
            'In gain medium: k < 0 (negative absorption = amplification). '
            'Nd:YAG: k ~ -1e-5, pumped by 808nm diode, lases at 1064nm. '
            'Same complex n, just imaginary part flips sign.'
        ),
    }


# ===========================================================================
# Part 3: Nylon Powder -- SLS Laser Sintering (photonics + materials)
# ===========================================================================

def sls_laser_sintering():
    """
    Selective Laser Sintering (SLS): nylon powder + CO2 laser -> 3D printed part.
    This IS photonics applied to manufacturing.

    Physics:
      1. CO2 laser (10.6 um IR): absorbed by nylon (PA-12) at surface
         Absorption: alpha_nylon_IR ~ 1000 cm^-1 -> skin depth ~ 10 um
         All energy deposited in first ~10 um of powder layer
      2. Powder heats above glass transition Tg ~ 180C, sinters
      3. Layer: 100 um thick, laser spot: 200-500 um diameter
      4. Scan speed: 1-10 m/s, hatch spacing: 100-200 um

    Photon budget:
      CO2 laser: P=50W, spot area = pi*(250um)^2 = 2e-7 m^2
      Irradiance I = P/A = 50/2e-7 = 250 MW/m^2  (high power density!)
      Energy per voxel = I * A_hatch * dt = P * t_dwell

    Nylon powder in air: EXPLOSION HAZARD
      Minimum explosive concentration (MEC): ~30 g/m^3 for PA-12
      Dust cloud + ignition source -> deflagration (sub-sonic combustion)
      Industrial SLS uses nitrogen atmosphere to prevent this
      Same physics as coal dust explosions, grain elevator fires

    Connection to repo:
      SLS laser is a 2D beam steering system -- same SLM/galvo as Ch8 of photonics_calculus.py
      Scan pattern = rasterized phase mask for the part geometry
      Speed optimization = same GS convergence problem (how fast to sweep the beam)
    """
    P_laser_W = 50.0
    spot_um = 250.0
    scan_m_per_s = 5.0
    hatch_um = 120.0
    layer_um = 100.0

    A_spot = np.pi * (spot_um*1e-6)**2
    I_irradiance = P_laser_W / A_spot

    t_dwell = spot_um*1e-6 / scan_m_per_s
    E_per_spot = P_laser_W * t_dwell

    E_per_volume = E_per_spot / (spot_um*1e-6 * hatch_um*1e-6 * layer_um*1e-6)

    alpha_nylon = 1e5    # 1/m at 10.6 um
    skin_depth_um = 1e6 / alpha_nylon

    MEC_g_per_m3 = 30.0
    MIE_mJ = 10.0

    return {
        'laser_wavelength_um': 10.6,
        'laser_power_W': P_laser_W,
        'spot_diameter_um': spot_um,
        'irradiance_MW_per_m2': I_irradiance/1e6,
        'dwell_time_us': t_dwell*1e6,
        'energy_per_spot_uJ': E_per_spot*1e6,
        'energy_density_J_per_cm3': E_per_volume/1e6,
        'skin_depth_um': skin_depth_um,
        'nylon_Tg_C': 180,
        'explosion': {
            'MEC_g_per_m3': MEC_g_per_m3,
            'MIE_mJ': MIE_mJ,
            'prevention': 'N2 atmosphere in SLS chamber: O2 < 3% prevents deflagration',
            'lesson': 'Powder in air = fuel-air mixture. Same physics as aviation fuel-air explosives.',
        },
        'photonics_connection': (
            'SLS scan head = galvo mirrors (same as oscilloscope) + f-theta lens. '
            'Scan pattern computed exactly like a rasterized SLM phase mask. '
            'Speed = 5 m/s -> need 50kHz galvo bandwidth. '
            'dgs/photonics_calculus.py::beam_steering_phase_ramp IS the scan algorithm.'
        ),
        'scan_speed_m_per_s': scan_m_per_s,
        'hatch_spacing_um': hatch_um,
        'layer_thickness_um': layer_um,
    }


# ===========================================================================
# Part 4: Inverse Problems (the real "backwards calculus")
# ===========================================================================

def inverse_problem_framework():
    """
    Forward problem: given input x, compute output y = A(x)
    Inverse problem: given measurement y, find input x = A^(-1)(y)

    In photonics:
      Forward: E(t) -> |E(t)|^2  (intensity measurement, lose phase)
      Inverse: |E(t)|^2 + |E(f)|^2 -> E(t)  (GS phase retrieval, this repo)

    In imaging:
      Forward: object f -> convolution with PSF -> blurry image g
      Inverse: deconvolution -> recover f from g

    In CT scan:
      Forward: 3D object -> set of 2D projections (X-ray images)
      Inverse: filtered back-projection or iterative -> 3D reconstruction

    In seismology:
      Forward: underground structure -> seismic wave recording
      Inverse: earthquake data -> map the Earth's interior

    ALL inverse problems share:
      1. Ill-posedness: small noise in y -> large error in x (amplified)
      2. Regularization: add prior knowledge to stabilize solution
      3. Iterative algorithms: GS, conjugate gradient, ADMM, neural network

    Tikhonov regularization: minimize ||Ax - y||^2 + lambda*||Lx||^2
      lambda=0: exact fit (overfits noise)
      lambda->inf: maximum smoothness (underfits)
      Optimal lambda: L-curve, cross-validation, or Morozov discrepancy
    """
    return {
        'forward': 'y = A(x)  [measurement operator]',
        'inverse': 'x* = argmin ||Ax-y||^2 + lambda*R(x)',
        'examples': {
            'phase_retrieval': 'A = |FT|, R = smoothness. THIS REPO.',
            'deconvolution': 'A = convolution with h(t), R = sparsity (L1)',
            'CT_scan': 'A = Radon transform, R = total variation',
            'MRI': 'A = undersampled FT, R = sparsity (compressed sensing)',
            'seismology': 'A = wave propagation, R = geological smoothness',
        },
        'ill_posedness': 'Small noise delta_y -> x_noisy = A^-1(y+delta_y) can be huge',
        'regularization': 'Tikhonov: x* = (A^T A + lambda I)^-1 A^T y',
        'gs_as_inverse': (
            'GS is a Gauss-Seidel / alternating projection method. '
            'Each iteration: FT -> replace amplitude with sqrt(target) -> IFT -> project. '
            'Converges to the max-entropy solution. '
            'Equivalent to gradient descent on ||I_out - I_target||^2.'
        ),
    }


def deconvolution_demo(signal_length=256, noise_level=0.01):
    """
    Deconvolution: remove known blur from measured signal.
    Forward: g = h * f + noise   (h = PSF, * = convolution, f = true signal)
    Inverse: f_est = IFT[FT(g) / FT(h)]  (Wiener / regularized deconvolution)

    Problem: FT(h) near zero -> division amplifies noise (ill-posed!)
    Solution: Wiener filter: F_est = FT(g) * FT(h)* / (|FT(h)|^2 + lambda)
      lambda = noise-to-signal power ratio
      lambda=0: exact inverse (unstable)
      lambda large: maximum smoothing (blurs result)

    This is exactly the inverse of H(f)=exp(j*pi*D*f^2) in this repo.
    Phase retrieval + deconvolution = recover E(t) from measured I(t).
    """
    # True signal: two Gaussian pulses
    t = np.linspace(0, 1, signal_length)
    f_true = (np.exp(-((t-0.3)/0.03)**2) +
              0.7*np.exp(-((t-0.7)/0.025)**2))

    # PSF: Gaussian blur (like GVD pulse broadening)
    sigma_h = 0.05
    t_psf = np.linspace(-0.5, 0.5, signal_length)
    h = np.exp(-t_psf**2 / (2*sigma_h**2))
    h /= h.sum()

    # Forward: convolve + add noise
    F_true = np.fft.rfft(f_true)
    H_psf  = np.fft.rfft(h)
    G_conv = F_true * H_psf
    g_blurred = np.fft.irfft(G_conv, n=signal_length)
    g_noisy = g_blurred + noise_level * np.random.default_rng(42).normal(size=signal_length)

    # Wiener deconvolution
    G = np.fft.rfft(g_noisy)
    H = np.fft.rfft(h)
    lam = noise_level**2  # regularization parameter = SNR^-1

    F_wiener = G * np.conj(H) / (np.abs(H)**2 + lam)
    f_recovered = np.fft.irfft(F_wiener, n=signal_length)

    # Naive inverse (no regularization)
    F_naive = G / (H + 1e-10)
    f_naive = np.fft.irfft(F_naive, n=signal_length)

    corr_wiener = np.corrcoef(f_true, f_recovered)[0,1]
    corr_naive  = np.corrcoef(f_true, f_naive)[0,1]

    return {
        't': t,
        'f_true': f_true,
        'g_noisy': g_noisy,
        'f_wiener': f_recovered,
        'f_naive': f_naive,
        'corr_wiener': corr_wiener,
        'corr_naive': corr_naive,
        'lambda_used': lam,
        'lesson': (
            f'Wiener (lambda={lam:.4f}): corr={corr_wiener:.4f}. '
            f'Naive (no regularization): corr={corr_naive:.4f}. '
            'Regularization stabilizes the inverse.'
        ),
    }


def maxwell_vector_field(source='point_charge', N=20):
    """
    Visualize Maxwell's equations as vector fields.

    Electric field of point charge: E = k*q*r_hat / r^2
    Magnetic field of wire:         B = mu0*I/(2*pi*r) * phi_hat

    These ARE Maxwell's equations in their integral form:
      div(E) = rho/eps0    -> E field lines start/end on charges
      div(B) = 0           -> B field lines form closed loops (no magnetic monopoles)
      curl(E) = -dB/dt     -> time-varying B creates E
      curl(B) = mu0*J + ... -> currents create B

    Visualizing vectors = seeing the equations.
    """
    x = np.linspace(-2, 2, N)
    y = np.linspace(-2, 2, N)
    X, Y = np.meshgrid(x, y)

    if source == 'point_charge':
        # E = k*q * (x,y) / r^3
        r = np.sqrt(X**2 + Y**2 + 0.1)   # small offset avoids singularity
        Ex = X / r**3
        Ey = Y / r**3
        title = 'E field: point charge (div E > 0 at origin)'
        label = '|E|'

    elif source == 'dipole':
        # Electric dipole: p along x-axis, z=0 slice
        d = 0.3   # half-separation
        r1 = np.sqrt((X-d)**2 + Y**2 + 0.05)
        r2 = np.sqrt((X+d)**2 + Y**2 + 0.05)
        # E from +q at (d,0) and -q at (-d,0)
        Ex = (X-d)/r1**3 - (X+d)/r2**3
        Ey = Y/r1**3 - Y/r2**3
        title = 'E field: electric dipole'
        label = '|E|'

    elif source == 'wire_B':
        # B = mu0*I/(2*pi) * z_hat x r_hat / r
        # In xy plane: B = mu0*I/(2*pi*r) * (-y/r, x/r, 0)
        r = np.sqrt(X**2 + Y**2 + 0.1)
        Ex = -Y / r**2   # B_x
        Ey =  X / r**2   # B_y
        title = 'B field: infinite wire (closed loops, div B = 0)'
        label = '|B|'

    # Magnitude
    E_mag = np.sqrt(Ex**2 + Ey**2)
    Ex_norm = Ex / (E_mag + 1e-10)
    Ey_norm = Ey / (E_mag + 1e-10)

    return {
        'X': X, 'Y': Y,
        'Ex': Ex, 'Ey': Ey,
        'Ex_norm': Ex_norm, 'Ey_norm': Ey_norm,
        'magnitude': E_mag,
        'title': title,
        'divergence_check': (
            'div(E_charge) > 0 at source: field lines diverge outward. '
            'div(B_wire) = 0 everywhere: B field lines form closed circles. '
            'This is Maxwell Eq 1 (Gauss) and Eq 4 (no monopoles).'
        ),
    }


def demo():
    print("=== CALCULUS BACKWARDS ===\n")

    print("--- Integration Techniques ---")
    tech = integration_techniques()
    print(f"  Power rule: {tech['power_rule_backwards']}")
    print(f"  Gaussian integral: {tech['gaussian_integral']}")
    print(f"  Professor trick: {tech['professor_trick']}")

    print("\n--- Feynman Technique ---")
    ft = feynman_technique_demo()
    print(f"  dI/da = {ft['dI_da']}")
    print(f"  int sin(x)/x dx = {ft['sinc_integral_result']}")
    print(f"  Engineering use: {ft['engineering_use'][:80]}...")

    print("\n--- FTC and Kramers-Kronig ---")
    ftc = fundamental_theorem_calculus()
    print(f"  FTC: {ftc['FTC_part1']}")
    print(f"  ResNet: {ftc['neural_network']}")

    print("\n--- Bioluminescence ---")
    bio = bioluminescence_physics()
    print(f"  Firefly photon: {bio['photon_energy_eV']:.3f} eV")
    print(f"  Quantum yields: { {k: v['phi_Q'] for k,v in bio['species'].items()} }")

    print("\n--- Beer-Lambert ---")
    bl = beer_lambert_absorption(alpha_per_cm=2.0)
    print(f"  alpha=2/cm: 1/e depth = {bl['depth_1_over_e_cm']:.2f} cm")
    print(f"  ODE: {bl['ODE']}")

    print("\n--- SLS Laser Sintering (nylon powder) ---")
    sls = sls_laser_sintering()
    print(f"  Irradiance: {sls['irradiance_MW_per_m2']:.0f} MW/m^2")
    print(f"  Skin depth in nylon: {sls['skin_depth_um']:.1f} um")
    print(f"  EXPLOSION: MEC = {sls['explosion']['MEC_g_per_m3']} g/m^3 -> {sls['explosion']['prevention']}")

    print("\n--- Deconvolution (Wiener) ---")
    dc = deconvolution_demo()
    print(f"  {dc['lesson']}")

    print("\n--- Inverse Problem Framework ---")
    inv = inverse_problem_framework()
    print(f"  GS as inverse: {inv['gs_as_inverse'][:80]}...")

    print("\n=== DONE ===")


if __name__ == '__main__':
    demo()
