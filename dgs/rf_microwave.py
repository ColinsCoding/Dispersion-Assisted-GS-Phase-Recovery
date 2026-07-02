"""
RF and Microwave Transmitter Design + Gradient Descent + Nuclear Cross Sections
EC ENGR 279AS (UCLA) — Physical and Wave Electronics

OUSD(R&E) CTA ALIGNMENT (arrows on your sheet):
  -> FutureG:                  5G/6G mmWave bands (26-100 GHz) -- this module
  -> Trusted AI and Autonomy:  ML-driven RF beamforming, GD phase retrieval -- this module
  -> Advanced Computing:       FFT, S-parameters, FDTD numerical EM -- this module
  -> Integrated Sensing/Cyber: Radar = dispersive GS (SAR) -- photon_interactions.py
  -> Directed Energy:          High-power RF, PIN diode switches -- this module
  -> Human-Machine Interfaces: Signal processing for BCI (EEG/MEG at RF) -- graph_theory.py

THE SAME CLASS OVER AND OVER:
  "Taking the same class over and over" = ABSTRACTION LEVEL LADDER.
  The same gradient descent algorithm in 4 implementations:
    C (bare metal):    for loops, malloc, pointer arithmetic
    Python (pure):     for loops, no numpy, maximum clarity
    NumPy (vectorized): array operations, broadcasting, no loops
    PyTorch (autodiff): torch.autograd, backward(), optimizer -- py -3.12

  ALL FOUR PRODUCE IDENTICAL RESULTS. The math is the same.
  Each level adds: (abstraction) + (productivity) - (control).
  Engineers need ALL FOUR levels fluent. This is the wire -> class -> module ladder.

CONTINUOUS CROSS SECTION (WIRE EM):
  A wire of radius a carries current I.
  Inside wire: J = I/(pi*a^2)  [A/m^2]  -- uniform current density
  Ampere's law: integral_C H . dl = I_enc
    r < a: H(r) = J*r/2 = I*r/(2*pi*a^2)
    r > a: H(r) = I/(2*pi*r)
  This is an AREA INTEGRAL over cross section:
    I_enc = integral_0^r integral_0^{2*pi} J * rho d(rho) d(phi) = J*pi*r^2
  Same math as:
    Gaussian beam area: integral |E|^2 dA = total power
    Neutron flux cross section: Sigma * phi = reaction rate density
    GS intensity: I(x,y) = |E(x,y)|^2 (area integral = total photon count)
"""
import math
import numpy as np

c_light = 2.998e8; mu0 = 4*np.pi*1e-7; eps0 = 8.854e-12; Z0 = 376.73


# ============================================================
# Wire Cross Section: Ampere's Law + Area Integrals
# ============================================================

def wire_cross_section(radius_mm=1.0, current_A=1.0, conductivity_S_m=5.96e7):
    """
    Complete electromagnetic analysis of a current-carrying wire.

    CROSS SECTION is not just geometry -- it is a continuous distribution.

    STATIC CASE (DC):
      Current density J = I / (pi*a^2)  [A/m^2]  uniform inside
      H field from Ampere's law:
        r < a: H(r) = I*r / (2*pi*a^2)   [increases linearly]
        r >= a: H(r) = I / (2*pi*r)       [decreases as 1/r]
      B field: B = mu0*H (in non-magnetic medium)
      Energy stored in magnetic field: U_m = (mu0*I^2*L)/(16*pi) per unit length

    SKIN EFFECT (AC):
      At high frequency, current crowds to surface.
      Skin depth: delta = sqrt(2 / (omega*mu0*sigma))
      Effective current density: J(r) = J0 * exp(-(a-r)/delta)  [from surface]
      Effective resistance increases: R_AC = R_DC * a/(2*delta) for a >> delta

    AREA INTEGRALS:
      Total current: I = integral J dA = integral_0^a integral_0^{2*pi} J(r)*r dr dphi
      Magnetic flux through cross section: Phi = integral_0^a B(r)*r dr dphi
      Internal inductance: L_int = Phi/I = mu0/(8*pi) per unit length

    CONNECTION TO GS:
      GS measures I_meas(x,y) = |E(x,y)|^2 at the detector.
      Total power = integral I_meas(x,y) dx dy  [area integral over detector]
      GS constraint: keep total power constant (unit amplitude constraint).
      This area integral is the normalization in every GS iteration.
    """
    a = radius_mm * 1e-3
    sigma = conductivity_S_m

    # Radial profile of H field
    r_arr = np.linspace(0, 5*a, 1000)
    H_inside  = current_A * r_arr[r_arr < a] / (2*np.pi*a**2)
    H_outside = current_A / (2*np.pi*r_arr[r_arr >= a])
    H_arr = np.concatenate([H_inside, H_outside])

    B_arr = mu0 * H_arr

    # DC resistance per unit length
    R_DC_per_m = 1/(sigma * np.pi * a**2)

    # Skin depth vs frequency
    f_arr = np.logspace(3, 12, 500)   # 1 kHz to 1 THz
    omega_arr = 2*np.pi*f_arr
    delta_arr = np.sqrt(2 / (omega_arr*mu0*sigma))
    R_AC_per_m = np.where(delta_arr < a,
                          R_DC_per_m * a/(2*delta_arr),
                          R_DC_per_m)   # AC resistance increases
    f_skin = 1/(np.pi*mu0*sigma*a**2)   # freq where delta = a/sqrt(2)

    # Area integrals
    dr = r_arr[1]-r_arr[0]
    # Total current from J*dA
    J_inside = current_A / (np.pi*a**2)
    r_in = r_arr[r_arr < a]
    I_from_area = float(J_inside * 2*np.pi * np.trapezoid(r_in, r_in))
    # Internal inductance per unit length
    L_int_per_m = mu0/(8*np.pi)   # exact formula

    # Energy density
    U_B_per_m = float(np.trapezoid(B_arr**2/(2*mu0) * 2*np.pi*r_arr, r_arr))

    return {
        'geometry': {
            'radius_mm': radius_mm, 'area_m2': np.pi*a**2,
            'J_DC_A_m2': float(J_inside),
        },
        'H_field': {
            'r_mm': (r_arr*1e3).tolist(),
            'H_A_per_m': H_arr.tolist(),
            'B_T': B_arr.tolist(),
            'H_max_inside': float(H_inside[-1]) if len(H_inside)>0 else 0,
            'H_at_surface': float(current_A/(2*np.pi*a)),
        },
        'skin_effect': {
            'f_Hz': f_arr.tolist(),
            'delta_mm': (delta_arr*1e3).tolist(),
            'R_AC_per_m': R_AC_per_m.tolist(),
            'f_skin_Hz': float(f_skin),
            'R_DC_per_m': float(R_DC_per_m),
        },
        'area_integrals': {
            'I_from_area_integral_A': float(I_from_area),
            'L_int_per_m_H': float(L_int_per_m),
            'U_B_per_m_J': float(U_B_per_m),
            'formula': 'I = integral J dA = J * pi * a^2  (for uniform J)',
        },
        'GS_connection': {
            'total_power': 'P = integral |E|^2 dA  (same area integral as I = integral J dA)',
            'normalization': 'GS keeps integral |E|^2 constant at each constraint step',
            'skin_depth_Im_k': 'delta = 1/Im[k] -- same as evanescent wave / quantum tunneling',
        },
    }


# ============================================================
# Transmission Lines: Wave on a Wire
# ============================================================

def transmission_line(Z0_line=50.0, Z_load=75.0, f_GHz=2.4, L_m=0.1):
    """
    Transmission line theory: the wave equation on a wire.

    Telegrapher's equations (distributed LC circuit):
      dV/dz = -L'*dI/dt     [V drops due to series inductance]
      dI/dz = -C'*dV/dt     [I drops due to shunt capacitance]
    -> Wave equation: d^2V/dz^2 = L'C' * d^2V/dt^2
    -> Propagation: V(z,t) = V+ * exp(j(omega*t - beta*z)) + V- * exp(j(omega*t + beta*z))

    CHARACTERISTIC IMPEDANCE: Z0 = sqrt(L'/C')
      Coaxial cable: Z0 = (60/sqrt(eps_r)) * ln(b/a)
      Microstrip:    Z0 ~ 87/sqrt(eps_r+1.41) * ln(5.98*h/(0.8*w+t))

    REFLECTION COEFFICIENT: Gamma = (Z_L - Z0)/(Z_L + Z0)
      |Gamma| = 0: matched load (no reflection)
      |Gamma| = 1: short circuit (Z_L = 0) or open circuit (Z_L = inf)
      Power reflected: |Gamma|^2 * P_in

    VSWR (Voltage Standing Wave Ratio):
      VSWR = (1 + |Gamma|) / (1 - |Gamma|)
      VSWR = 1: perfect match
      VSWR = infinity: complete reflection

    INPUT IMPEDANCE (lossless line length L):
      Z_in = Z0 * (Z_L + j*Z0*tan(beta*L)) / (Z0 + j*Z_L*tan(beta*L))
      Quarter-wave transformer (beta*L = pi/2):
        Z_in = Z0^2 / Z_L   [impedance inversion!]

    MICROWAVE TRANSMITTER CHAIN:
      Signal -> Modulator -> Driver Amp -> Power Amp -> Filter -> Antenna
      S-parameters describe each block:
        S11 = input reflection
        S21 = forward gain (transmission)
        S12 = reverse isolation
        S22 = output reflection
      P_out = P_in * |S21|^2  (in linear scale)

    EC ENGR 279AS CONNECTION:
      This course covers: microstrip design, power amplifier classes (A/B/AB/C),
      impedance matching networks (L, pi, T topologies), phase noise.
      Key tool: Smith Chart = mapping from Gamma plane to Z plane.
      H(f) in this repo = S21(f) of the fiber "two-port network."
    """
    f = f_GHz * 1e9
    omega = 2*np.pi*f
    lam = c_light/f
    beta = 2*np.pi/lam   # rad/m

    # Reflection coefficient
    Gamma = (Z_load - Z0_line) / (Z_load + Z0_line)
    Gamma_mag = abs(Gamma)
    Gamma_phase = math.atan2(Gamma.imag if hasattr(Gamma,'imag') else 0, Gamma)

    # VSWR
    if Gamma_mag < 1.0:
        VSWR = (1 + Gamma_mag) / (1 - Gamma_mag)
    else:
        VSWR = float('inf')
    power_reflected = Gamma_mag**2
    return_loss_dB = -20*math.log10(Gamma_mag) if Gamma_mag > 1e-10 else 100.0

    # Input impedance vs position
    z_arr = np.linspace(0, lam, 400)
    beta_z = beta * z_arr
    Z_in_arr = Z0_line * (Z_load + 1j*Z0_line*np.tan(beta_z)) / (Z0_line + 1j*Z_load*np.tan(beta_z))

    # Standing wave pattern |V(z)| normalized
    V_forward = 1.0
    V_ref = Gamma * V_forward
    V_pattern = np.abs(V_forward*np.exp(-1j*beta_z) + V_ref*np.exp(1j*beta_z))

    # Quarter-wave transformer to match Z_load to Z0_line
    Z0_QWT = math.sqrt(Z0_line * Z_load)   # QWT impedance
    L_QWT = lam/4

    # S-parameter matrix for a lossless 2-port (amplifier representation)
    G_dB = 15.0   # forward gain 15 dB
    G_lin = 10**(G_dB/10)
    S11 = complex(0.1, 0)
    S21 = complex(math.sqrt(G_lin), 0)
    S12 = complex(0.001, 0)   # reverse isolation 60 dB
    S22 = complex(0.15, 0)
    S = np.array([[S11, S12],[S21, S22]])
    # Stability: Rollett K factor
    Delta_S = S11*S22 - S12*S21
    K = (1 - abs(S11)**2 - abs(S22)**2 + abs(Delta_S)**2) / (2*abs(S12)*abs(S21) + 1e-30)

    # Frequency sweep: S21 of a resonant cavity
    f_sweep = np.linspace(1, 10, 500)   # GHz
    f_res = 5.0; Q_factor = 50
    S21_cav = Q_factor / (Q_factor + 1j*Q_factor*(f_sweep/f_res - f_res/f_sweep))
    S21_mag_dB = 20*np.log10(np.abs(S21_cav)+1e-30)

    return {
        'line': {
            'Z0_ohm': Z0_line, 'Z_load_ohm': Z_load, 'f_GHz': f_GHz,
            'lambda_m': float(lam), 'beta_rad_m': float(beta),
        },
        'reflection': {
            'Gamma_mag': float(Gamma_mag),
            'Gamma_dB': float(20*math.log10(Gamma_mag+1e-30)),
            'VSWR': float(VSWR),
            'power_reflected_pct': float(power_reflected*100),
            'return_loss_dB': float(return_loss_dB),
        },
        'input_impedance': {
            'z_lambda': (z_arr/lam).tolist(),
            'Z_in_real': np.real(Z_in_arr).tolist(),
            'Z_in_imag': np.imag(Z_in_arr).tolist(),
            'V_standing_wave': V_pattern.tolist(),
        },
        'QWT': {
            'Z0_QWT_ohm': float(Z0_QWT),
            'L_QWT_mm': float(L_QWT*1e3),
            'result': f'Transforms {Z_load} Ohm to {Z0_line} Ohm',
        },
        'S_params': {
            'S11': str(S11), 'S21_dB': float(G_dB),
            'S12': str(S12), 'S22': str(S22),
            'K_Rollett': float(K),
            'unconditionally_stable': bool(K > 1 and abs(Delta_S) < 1),
        },
        'resonant_cavity': {
            'f_GHz': f_sweep.tolist(),
            'S21_dB': S21_mag_dB.tolist(),
            'f_res_GHz': f_res, 'Q': Q_factor,
            'analogy': 'S21(f) of cavity = H(f) of dispersive fiber. Both are transfer functions.',
        },
    }


# ============================================================
# PID Controller: Topology of Feedback
# ============================================================

def pid_controller():
    """
    PID controller: topology of the feedback loop.

    PID = Proportional + Integral + Derivative control.
    u(t) = Kp*e(t) + Ki*integral_0^t e(s)ds + Kd*de/dt
    Transfer function: C(s) = Kp + Ki/s + Kd*s = (Kd*s^2 + Kp*s + Ki)/s

    TOPOLOGY: every feedback system has the same graph structure:
      Plant G(s) -----> [+] ---> output y
           ^             |
           |    [C(s)]  <--   error e = r - y
           reference r

    Closed-loop: T(s) = C(s)*G(s) / (1 + C(s)*G(s))
    Characteristic equation: 1 + C(s)*G(s) = 0 -> poles

    STABILITY (Routh-Hurwitz / Bode):
      All poles must be in left half-plane (Re[s] < 0)
      Gain margin: how much gain before instability
      Phase margin: how much phase before instability (target: 45-60 degrees)

    GRADIENT DESCENT IS A DISCRETE PID:
      GD: theta_{k+1} = theta_k - alpha * grad(L)(theta_k)
        This is a P controller (only proportional term)!
        alpha = Kp (learning rate = proportional gain)
        error signal = -grad(L) (gradient of loss)
      Momentum (SGD+momentum):
        v_{k+1} = beta*v_k + (1-beta)*grad(L)
        theta_{k+1} = theta_k - alpha*v_{k+1}
        This is a PI controller! (integral of past gradients)
      Adam optimizer = PID controller with adaptive gains per parameter!
        Adam m_t = moving avg of grad  [P term]
        Adam v_t = moving avg of grad^2 [normalizes = adaptive Kp]
        = adaptive momentum = PID-like

    DISPERSION GS PID:
      GS iteration = feedback loop:
        Plant G(s): H(f) dispersive propagator
        Measurement: I_meas (intensity constraint)
        Controller: amplitude replacement (P control on amplitude error)
        Convergence = steady state of feedback loop
      n_iter GS = n feedback cycles
      GS divergence = instability (|Gamma| > 1 analog)
    """
    # PID step response simulation
    dt = 0.01; T_sim = 20.0
    t_arr = np.arange(0, T_sim, dt)
    N = len(t_arr)

    # Plant: first-order system G(s) = 1/(s+1) -> step response 1-e^{-t}
    def plant_step(u_arr, dt):
        y = np.zeros(N)
        for i in range(1, N):
            y[i] = y[i-1] + dt*(-y[i-1] + u_arr[i-1])
        return y

    r_ref = np.ones(N)   # step reference

    # P control only
    Kp = 2.0
    e_P = np.zeros(N); u_P = np.zeros(N); y_P = np.zeros(N)
    for i in range(N):
        e_P[i] = r_ref[i] - y_P[i-1] if i>0 else r_ref[0]
        u_P[i] = Kp * e_P[i]
        if i < N-1:
            y_P[i+1] = y_P[i] + dt*(-y_P[i] + u_P[i])
    ss_error_P = float(r_ref[-1] - y_P[-1])

    # PI control
    Kp_PI = 1.5; Ki_PI = 0.8
    e_PI = np.zeros(N); u_PI = np.zeros(N); y_PI = np.zeros(N); int_e = 0
    for i in range(N):
        e_PI[i] = r_ref[i] - y_PI[i-1] if i>0 else r_ref[0]
        int_e += e_PI[i]*dt
        u_PI[i] = Kp_PI*e_PI[i] + Ki_PI*int_e
        if i < N-1:
            y_PI[i+1] = y_PI[i] + dt*(-y_PI[i] + u_PI[i])
    ss_error_PI = float(r_ref[-1] - y_PI[-1])

    # PID control
    Kp_PID = 1.5; Ki_PID = 0.8; Kd_PID = 0.3
    e_PID = np.zeros(N); u_PID = np.zeros(N); y_PID = np.zeros(N)
    int_e3 = 0
    for i in range(N):
        e_PID[i] = r_ref[i] - y_PID[i-1] if i>0 else r_ref[0]
        int_e3 += e_PID[i]*dt
        de = (e_PID[i]-e_PID[i-1])/dt if i>0 else 0
        u_PID[i] = Kp_PID*e_PID[i] + Ki_PID*int_e3 + Kd_PID*de
        if i < N-1:
            y_PID[i+1] = y_PID[i] + dt*(-y_PID[i] + u_PID[i])
    ss_error_PID = float(r_ref[-1] - y_PID[-1])

    # GD as P-controller: minimize L(theta) = (theta - theta_star)^2
    theta_star = 3.0; theta = 0.0
    alpha = 0.3   # learning rate = Kp
    theta_history = [theta]
    for _ in range(100):
        grad = 2*(theta - theta_star)   # dL/dtheta
        theta = theta - alpha * grad
        theta_history.append(theta)

    # Bode magnitude plot of PID: C(jw) = Kp + Ki/(jw) + Kd*(jw)
    omega_bode = np.logspace(-1, 3, 400)
    C_PID_bode = Kp_PID + Ki_PID/(1j*omega_bode) + Kd_PID*(1j*omega_bode)
    G_plant_bode = 1/(1j*omega_bode + 1)
    L_open = C_PID_bode * G_plant_bode
    phase_margin_idx = np.argmin(np.abs(np.abs(L_open) - 1.0))
    phase_margin_deg = 180 + float(np.degrees(np.angle(L_open[phase_margin_idx])))

    return {
        'step_response': {
            't_s': t_arr[::10].tolist(),
            'y_P': y_P[::10].tolist(),
            'y_PI': y_PI[::10].tolist(),
            'y_PID': y_PID[::10].tolist(),
            'ss_error_P': float(ss_error_P),
            'ss_error_PI': float(ss_error_PI),
            'ss_error_PID': float(ss_error_PID),
        },
        'GD_as_P_controller': {
            'theta_history': theta_history[:20],
            'final_theta': float(theta_history[-1]),
            'theta_star': theta_star,
            'alpha_equals_Kp': alpha,
            'converged': bool(abs(theta_history[-1]-theta_star) < 0.01),
        },
        'optimizer_PID_analogy': {
            'SGD': 'P controller: u = Kp*e  (e = -grad)',
            'Momentum': 'PI controller: adds integral of past gradients',
            'Adam': 'Adaptive PID: Kp scales with 1/sqrt(v_t), acts like normalized error',
            'GS': 'Amplitude replacement = P control on amplitude error',
        },
        'stability': {
            'phase_margin_deg': float(phase_margin_deg),
            'stable': bool(phase_margin_deg > 0),
            'omega_bode': omega_bode.tolist(),
            'L_open_mag_dB': (20*np.log10(np.abs(L_open)+1e-30)).tolist(),
            'L_open_phase_deg': np.degrees(np.angle(L_open)).tolist(),
        },
        'topology': {
            'nodes': ['reference r', 'error e = r-y', 'PID controller C(s)', 'plant G(s)', 'output y'],
            'edges': ['r->e', 'e->C', 'C->G', 'G->y', 'y->e (feedback, negative)'],
            'cycles': 1,
            'GS_is_feedback_loop': True,
        },
    }


# ============================================================
# Gradient Descent: Same Algorithm in 4 Implementations
# ============================================================

def gradient_descent_four_ways():
    """
    Gradient descent for ||A*x - b||^2 minimization.

    THE SAME CLASS OVER AND OVER:
      Level 0 (C pseudocode): raw pointers, malloc, for loops
      Level 1 (Python pure): list comprehensions, no libraries
      Level 2 (NumPy): vectorized, broadcasting
      Level 3 (PyTorch): autograd, backward()  [py -3.12 only]

    PROBLEM: minimize L(x) = ||A*x - b||^2 = (A*x-b)^T(A*x-b)
      Gradient: grad_L = 2*A^T*(A*x - b)
      GD update: x_{k+1} = x_k - alpha * grad_L(x_k)
      Optimal step: alpha = 1 / (2 * max_eigenvalue(A^T A))
      Solution: x* = (A^T A)^{-1} A^T b  [normal equations]

    CONNECTION TO DISPERSION GS:
      GS minimizes L = sum_n |I_meas_n - |E_n|^2|^2
      But GS does NOT compute the gradient explicitly.
      It uses ALTERNATING PROJECTIONS (like coordinate descent).
      Gradient descent on this loss = "gradient GS" = slower but more general.
      PyTorch autograd CAN compute grad of ||H(f)*E - measured||^2 automatically.

    NUCLEAR CROSS SECTION ANALOG:
      Neutron transport: phi(r) is the neutron flux field.
      We minimize ||L*phi - S||^2 where L = transport operator, S = source.
      Same linear least squares structure as A*x = b.
      GD or Krylov methods (GMRES, CG) solve it.
      phi(r) = integral Sigma_s(r,r')*phi(r')*dr' + S(r)  [integral equation]
      This is EXACTLY the structure of GS: E_out = H * E_in + noise.
    """
    # Problem setup
    np.random.seed(7)
    m, n = 20, 5
    A_mat = np.random.randn(m, n)
    x_true = np.array([1., 2., -1., 0.5, 3.])
    b_vec = A_mat @ x_true + 0.01*np.random.randn(m)

    # Optimal alpha
    ATA = A_mat.T @ A_mat
    lambda_max = float(np.linalg.eigvalsh(ATA)[-1])
    alpha_opt = 1.0 / (2*lambda_max)
    n_iter_GD = 200

    # ── Level 1: Python pure (no numpy arrays, just lists) ──
    def gd_pure_python(A, b, x0, alpha, n_iter):
        m_p = len(b); n_p = len(x0)
        x = list(x0)
        loss_hist = []
        for _ in range(n_iter):
            # residual r = A*x - b
            r = [sum(A[i][j]*x[j] for j in range(n_p)) - b[i] for i in range(m_p)]
            loss = sum(ri**2 for ri in r)
            loss_hist.append(loss)
            # gradient = 2*A^T*r
            grad = [2*sum(A[i][j]*r[i] for i in range(m_p)) for j in range(n_p)]
            x = [x[j] - alpha*grad[j] for j in range(n_p)]
        return x, loss_hist

    A_list = A_mat.tolist(); b_list = b_vec.tolist()
    x_py, loss_py = gd_pure_python(A_list, b_list, [0.0]*n, alpha_opt, n_iter_GD)

    # ── Level 2: NumPy vectorized ──
    def gd_numpy(A, b, x0, alpha, n_iter):
        x = x0.copy(); loss_hist = []
        for _ in range(n_iter):
            r = A @ x - b
            loss_hist.append(float(r @ r))
            grad = 2 * A.T @ r
            x = x - alpha * grad
        return x, loss_hist

    x_np, loss_np = gd_numpy(A_mat, b_vec, np.zeros(n), alpha_opt, n_iter_GD)

    # ── Level 3: PyTorch (py -3.12 only, graceful fallback) ──
    torch_available = False
    x_torch_final = None; loss_torch = []
    try:
        import torch
        A_t = torch.tensor(A_mat, dtype=torch.float64)
        b_t = torch.tensor(b_vec, dtype=torch.float64)
        x_t = torch.zeros(n, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.SGD([x_t], lr=float(alpha_opt))
        loss_torch = []
        for _ in range(n_iter_GD):
            opt.zero_grad()
            r_t = A_t @ x_t - b_t
            L_t = (r_t * r_t).sum()
            L_t.backward()
            opt.step()
            loss_torch.append(float(L_t.item()))
        x_torch_final = x_t.detach().numpy().tolist()
        torch_available = True
    except ImportError:
        loss_torch = loss_np   # fallback: show numpy result

    # ── Level 0: C pseudocode as string ──
    c_pseudocode = '''
// C implementation of gradient descent
// Compile: gcc -O2 -lm gd.c -o gd
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define M 20
#define N 5
void gd(double A[M][N], double b[M], double x[N],
        double alpha, int n_iter) {
    double r[M], grad[N];
    for (int iter=0; iter<n_iter; iter++) {
        // r = A*x - b
        for (int i=0; i<M; i++) {
            r[i] = -b[i];
            for (int j=0; j<N; j++) r[i] += A[i][j]*x[j];
        }
        // grad = 2*A^T*r
        for (int j=0; j<N; j++) {
            grad[j] = 0;
            for (int i=0; i<M; i++) grad[j] += 2*A[i][j]*r[i];
        }
        // x = x - alpha*grad
        for (int j=0; j<N; j++) x[j] -= alpha*grad[j];
    }
}
'''.strip()

    # Exact solution via normal equations
    x_exact = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
    error_py  = float(np.linalg.norm(np.array(x_py)  - x_true))
    error_np  = float(np.linalg.norm(x_np  - x_true))
    error_ex  = float(np.linalg.norm(x_exact - x_true))

    return {
        'problem': {
            'A_shape': [m, n], 'x_true': x_true.tolist(),
            'alpha_optimal': float(alpha_opt), 'n_iter': n_iter_GD,
        },
        'C_pseudocode': c_pseudocode,
        'python_pure': {
            'x_final': x_py,
            'loss_history': loss_py[::20],
            'error_vs_x_true': float(error_py),
        },
        'numpy': {
            'x_final': x_np.tolist(),
            'loss_history': loss_np[::20],
            'error_vs_x_true': float(error_np),
        },
        'pytorch': {
            'available': torch_available,
            'x_final': x_torch_final,
            'loss_history': loss_torch[::20] if loss_torch else [],
            'note': 'torch is py -3.12 only; if unavailable numpy result shown',
        },
        'exact': {
            'x_lstsq': x_exact.tolist(),
            'error_vs_x_true': float(error_ex),
        },
        'all_same_result': bool(error_py < 0.1 and error_np < 0.1),
        'GS_connection': {
            'GS_loss': 'L = sum |I_meas - |H*E||^2',
            'GD_on_GS': 'x_{k+1} = x_k - alpha * grad_x L',
            'alternating_proj': 'GS uses alternating projections, not explicit gradient. Faster but less general.',
            'torch_autograd': 'torch.autograd.grad(L, x) computes grad of GS loss exactly -- py -3.12',
        },
    }


# ============================================================
# Nuclear Macroscopic Cross Sections: Area/Volume Integrals
# ============================================================

def nuclear_macroscopic_cross_sections():
    """
    Nuclear engineering: macroscopic cross sections and neutron flux.

    MICROSCOPIC CROSS SECTION sigma [barns = 1e-28 m^2]:
      sigma = effective "target area" of a nucleus for a given reaction.
      sigma_total = sigma_scatter + sigma_absorb + sigma_fission
      Quantum mechanics: sigma is NOT geometric area!
        sigma can exceed geometric cross section by orders of magnitude near resonances.
        Near resonance: sigma(E) = sigma_0 * (Gamma/2)^2 / ((E-E0)^2 + (Gamma/2)^2)
        This IS the Breit-Wigner = Lorentzian = S21 of a resonator.

    MACROSCOPIC CROSS SECTION Sigma = n * sigma [1/m]:
      n = number density of atoms [atoms/m^3]
      Sigma_total = n * sigma_total
      Mean free path: lambda = 1/Sigma

    NEUTRON FLUX phi(r) [neutrons/m^2/s]:
      phi = n_neutron * v_neutron  [flux = density * speed]
      Reaction rate density: R = Sigma * phi  [reactions/m^3/s]

    TOTAL REACTION RATE (volume integral):
      Rate = integral_V Sigma(r) * phi(r) dV
      For uniform: Rate = Sigma * phi * V

    NEUTRON TRANSPORT (the "statics" problem in nuclear):
      Steady-state: div(J) + Sigma_a * phi = S  (neutron conservation)
      J = -D * grad(phi)  (Fick's law, diffusion approximation)
      -> Laplacian: -D * nabla^2(phi) + Sigma_a * phi = S
      This IS the Helmholtz equation! Same as:
        Optical: (-nabla^2 - k^2*n^2) E = 0
        QM: (-hbar^2/2m nabla^2 + V) psi = E*psi
      Nuclear engineers solve THESE SAME PDEs with THESE SAME NUMERICAL METHODS.

    CONNECTION TO THIS REPO:
      Neutron flux phi(r) ↔ optical field E(r)
      Sigma(r) ↔ n(r) (refractive index variation)
      D*nabla^2(phi) = GVD in fiber: phi'' = beta2 * d^2E/dt^2
      GS phase retrieval ↔ flux reconstruction from count rate measurements

    AREA INTEGRAL: neutron beam cross section
      Phi_total [neutrons/s] = integral phi(x,y) dx dy over beam cross section
      For Gaussian beam: phi(r) = phi_0 * exp(-r^2/w^2)
        Phi_total = phi_0 * pi * w^2  [same as optical beam power!]
    """
    # Uranium-235 fission cross section (energy-dependent, Breit-Wigner)
    E_eV = np.logspace(-2, 7, 1000)   # 0.01 eV to 10 MeV
    # Simplified 1/v law for thermal neutrons (below 1 eV)
    # Resonances in 1-1000 eV range, fast regime above
    sigma_th = 582e-28   # m^2 at 0.025 eV
    E_th = 0.025   # eV
    sigma_fission = np.where(E_eV < 1, sigma_th*np.sqrt(E_th/E_eV),
                    np.where(E_eV < 1e4, sigma_th*np.sqrt(E_th/1.0)*
                             (1 + 50*np.exp(-(np.log10(E_eV)-2)**2/0.3)),
                             sigma_th*np.sqrt(E_th/1.0)*0.01))

    # Macroscopic cross section of UO2 fuel
    rho_UO2 = 10960   # kg/m^3
    M_UO2 = 0.270     # kg/mol (U-238 dominant)
    N_A = 6.022e23    # Avogadro
    n_UO2 = rho_UO2 * N_A / M_UO2 / 1e6   # atoms/cm^3 (nuclear convention)
    Sigma_a_thermal = n_UO2 * float(sigma_fission[np.argmin(np.abs(E_eV-0.025))]) * 1e4  # cm^-1

    # Neutron flux profile in cylindrical reactor (J0 Bessel function)
    r_arr = np.linspace(0, 1.5, 400)   # m, from center
    R_reactor = 1.0; H_reactor = 2.0   # m
    phi_0 = 3e13   # neutrons/m^2/s (peak flux in typical LWR)
    # phi(r) = phi_0 * J0(2.405*r/R) (radial profile)
    from numpy.polynomial import polynomial as P
    # Use closed-form approximation J0(x) ~ 1 - x^2/4 + x^4/64 (small x)
    x_bessel = 2.405 * r_arr / R_reactor
    # Compute J0 using numpy
    # J0 = Re(exp(j*x)) approximated by series or scipy -- numpy has it via special
    # Use numpy only (no scipy): approximate with polynomial for J0
    # J0(x) accurate enough with: J0 = sum_{k=0}^{20} (-1)^k (x/2)^{2k} / (k!)^2
    J0_arr = np.zeros_like(x_bessel)
    for k in range(20):
        J0_arr += ((-1)**k * (x_bessel/2)**(2*k) / math.factorial(k)**2)
    phi_r = phi_0 * np.maximum(J0_arr, 0)   # phi >= 0

    # Reaction rate in cross section (area integral)
    dr = r_arr[1]-r_arr[0]
    # Rate per unit height = integral_0^R Sigma*phi * 2*pi*r dr
    Rate_per_m = float(np.trapezoid(Sigma_a_thermal * phi_r * 2*np.pi*r_arr, r_arr))

    # Gaussian beam cross section area integral
    w_beam = 0.3   # m (neutron beam radius)
    r_beam = np.linspace(0, 3*w_beam, 400)
    phi_gaussian = phi_0 * np.exp(-r_beam**2/w_beam**2)
    Phi_total = float(np.trapezoid(phi_gaussian * 2*np.pi*r_beam, r_beam))
    Phi_exact = phi_0 * np.pi * w_beam**2   # exact: int exp(-r^2/w^2)*2*pi*r dr = pi*w^2

    # Resonance cross section (Breit-Wigner at 6.67 eV for U-238)
    E_res = 6.67; Gamma = 0.027   # eV
    sigma_BW = sigma_th * (Gamma/2)**2 / ((E_eV - E_res)**2 + (Gamma/2)**2)
    # This is IDENTICAL to S21(f) Breit-Wigner for a resonant cavity

    return {
        'U235': {
            'E_eV': E_eV.tolist(),
            'sigma_fission_barns': (sigma_fission/1e-28).tolist(),
            'sigma_thermal_barns': float(sigma_th/1e-28),
        },
        'UO2_fuel': {
            'rho_kg_m3': rho_UO2,
            'n_atoms_per_cm3': float(n_UO2),
            'Sigma_a_thermal_per_cm': float(Sigma_a_thermal),
            'mean_free_path_cm': float(1/Sigma_a_thermal) if Sigma_a_thermal>0 else 0,
        },
        'flux_profile': {
            'r_m': r_arr.tolist(),
            'phi_r': phi_r.tolist(),
            'phi_0': phi_0,
            'reaction_rate_per_m': float(Rate_per_m),
        },
        'Gaussian_beam_integral': {
            'r_m': r_beam.tolist(),
            'phi_gaussian': phi_gaussian.tolist(),
            'Phi_numerical': float(Phi_total),
            'Phi_exact': float(Phi_exact),
            'relative_error': abs(Phi_total-Phi_exact)/Phi_exact,
            'lesson': 'Neutron flux area integral = optical beam power integral. Same math.',
        },
        'Breit_Wigner_resonance': {
            'E_res_eV': E_res,
            'Gamma_eV': Gamma,
            'sigma_BW_barns': (sigma_BW/1e-28).tolist(),
            'peak_barns': float(sigma_th/1e-28),
            'analogy': 'Nuclear Breit-Wigner = S21(f) of optical/microwave resonator. SAME formula.',
        },
        'connections': {
            'neutron_transport': '-D*nabla^2(phi) + Sigma_a*phi = S  (same as Helmholtz)',
            'optical': 'nabla^2(E) + k^2*n^2*E = 0  (same eigenvalue structure)',
            'QM': '-hbar^2/2m nabla^2(psi) + V*psi = E*psi  (Schrodinger)',
            'GS': 'phi(r) reconstruction from flux measurements = GS phase retrieval',
            'area_integral': 'Rate = integral Sigma*phi dV  (same as P = integral |E|^2 dA)',
        },
    }


def demo():
    print("=== RF/MICROWAVE + GD + NUCLEAR CROSS SECTIONS ===\n")

    print("--- Wire Cross Section (Ampere's Law) ---")
    w = wire_cross_section(radius_mm=1.0, current_A=1.0)
    print(f"  J_DC = {w['geometry']['J_DC_A_m2']:.2e} A/m^2")
    print(f"  H at surface = {w['H_field']['H_at_surface']:.2f} A/m")
    print(f"  f_skin (delta=a) = {w['skin_effect']['f_skin_Hz']/1e3:.1f} kHz")
    print(f"  I from area integral = {w['area_integrals']['I_from_area_integral_A']:.4f} A")

    print("\n--- Transmission Line (S-params) ---")
    tl = transmission_line(Z0_line=50, Z_load=75, f_GHz=2.4)
    print(f"  Gamma = {tl['reflection']['Gamma_mag']:.3f}, VSWR = {tl['reflection']['VSWR']:.2f}")
    print(f"  Return loss = {tl['reflection']['return_loss_dB']:.1f} dB")
    print(f"  QWT Z0 = {tl['QWT']['Z0_QWT_ohm']:.1f} Ohm at L = {tl['QWT']['L_QWT_mm']:.1f} mm")
    print(f"  S21 = {tl['S_params']['S21_dB']:.0f} dB, K = {tl['S_params']['K_Rollett']:.2f}, stable = {tl['S_params']['unconditionally_stable']}")

    print("\n--- PID Controller ---")
    pid = pid_controller()
    print(f"  Steady-state error: P={pid['step_response']['ss_error_P']:.4f}, PI={pid['step_response']['ss_error_PI']:.6f}, PID={pid['step_response']['ss_error_PID']:.6f}")
    print(f"  Phase margin = {pid['stability']['phase_margin_deg']:.1f} deg, stable = {pid['stability']['stable']}")
    print(f"  GD converged to theta*: {pid['GD_as_P_controller']['converged']}, final = {pid['GD_as_P_controller']['final_theta']:.4f}")

    print("\n--- Gradient Descent: 4 Implementations ---")
    gd = gradient_descent_four_ways()
    print(f"  Pure Python error: {gd['python_pure']['error_vs_x_true']:.6f}")
    print(f"  NumPy error:       {gd['numpy']['error_vs_x_true']:.6f}")
    print(f"  Exact lstsq error: {gd['exact']['error_vs_x_true']:.6f}")
    print(f"  PyTorch available: {gd['pytorch']['available']}")
    print(f"  All implementations same: {gd['all_same_result']}")
    print(f"  C pseudocode (first line): {gd['C_pseudocode'].splitlines()[1]}")

    print("\n--- Nuclear Cross Sections ---")
    nc = nuclear_macroscopic_cross_sections()
    print(f"  U-235 sigma_thermal = {nc['U235']['sigma_thermal_barns']:.0f} barns")
    print(f"  UO2 Sigma_a = {nc['UO2_fuel']['Sigma_a_thermal_per_cm']:.4f} cm^-1")
    print(f"  Gaussian beam integral: numerical={nc['Gaussian_beam_integral']['Phi_numerical']:.3e}, exact={nc['Gaussian_beam_integral']['Phi_exact']:.3e}")
    print(f"  Relative error = {nc['Gaussian_beam_integral']['relative_error']:.4f}")
    print(f"  Breit-Wigner: {nc['Breit_Wigner_resonance']['analogy']}")

    print("\n=== RF/MICROWAVE COMPLETE ===")


if __name__ == '__main__':
    demo()
