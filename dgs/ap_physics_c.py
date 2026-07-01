"""
AP Physics C: Electricity & Magnetism
Self-tutoring module: Calculus + Probability/Stats for E&M

Curriculum coverage (College Board AP Physics C: E&M):
  Unit 1  Electrostatics (Coulomb, Gauss's law, E field, potential)
  Unit 2  Conductors, Capacitors, Dielectrics
  Unit 3  Electric Circuits (RC, Kirchhoff, power)
  Unit 4  Magnetic Fields (Biot-Savart, Ampere's law)
  Unit 5  Electromagnetism (Faraday, Lenz, inductance, RL, LC)
  STATS   Error propagation, Gaussian fits, chi-squared, residual plots

Calculus tools used throughout:
  - Gradient: E = -grad(V)
  - Divergence: div(E) = rho/eps0 (Gauss differential form)
  - Curl: curl(E) = -dB/dt (Faraday differential form)
  - Line integrals: V = -int(E.dl)
  - Surface integrals: Phi_E = int(E.dA)  (Gauss's law)
  - ODEs: RC / RL / LC circuits

Connection to this repo:
  - Gauss's law surface integral = same math as GS Fourier plane constraint
  - Faraday EMF = -dPhi/dt is d/dt of a flux -- same as instantaneous freq in coppinger1999
  - RL circuit V(t) = L*dI/dt has same structure as dispersive H(f) = exp(j*pi*D*f^2)
  - Stats here underpins SNR analysis in dgs/gs_core.py
"""
import numpy as np
import sympy as sp

# Physical constants
eps0 = 8.854187817e-12   # [F/m]  permittivity of free space
mu0  = 1.2566370614e-6   # [H/m]  permeability of free space
k_e  = 1/(4*np.pi*eps0) # [N m^2/C^2]  Coulomb constant ~8.99e9
e    = 1.602176634e-19   # [C]   elementary charge
kB   = 1.380649e-23      # [J/K] Boltzmann

# ===========================================================================
# UNIT 1: Electrostatics
# ===========================================================================

def coulombs_law(q1_C, q2_C, r_m):
    """
    F = k*q1*q2 / r^2  [N]  (scalar magnitude, positive = repulsion)

    AP Physics C: know this is inverse-square, derive from Gauss's law.
    """
    if r_m <= 0:
        raise ValueError("r_m must be > 0")
    F = k_e * q1_C * q2_C / r_m**2
    return {'F_N': F, 'repulsive': F > 0, 'k_e': k_e}


def electric_field_point_charge(q_C, r_m):
    """
    E = k*q / r^2  [N/C]  at distance r from point charge q.
    Direction: radially outward (q>0) or inward (q<0).

    AP trick: E from a uniform sphere = same as point charge outside.
    """
    if r_m <= 0:
        raise ValueError("r_m must be > 0")
    return k_e * q_C / r_m**2


def gauss_law_surface_integral(Q_enc_C):
    """
    Gauss's Law (integral form):
      closed_surface_integral(E . dA) = Q_enc / eps0

    AP Physics C KEY INSIGHT:
      This avoids integrating the E field directly -- pick a Gaussian surface
      where E is constant and parallel to dA, then:
        E * A_surface = Q_enc / eps0
        E = Q_enc / (eps0 * A_surface)

    Returns the flux for a given enclosed charge.
    """
    flux = Q_enc_C / eps0
    return {'flux_N_m2_per_C': flux, 'Q_enc_C': Q_enc_C}


def gauss_law_geometries(Q_total_C=1e-6, r_m=0.1):
    """
    Electric field from Gauss's law for three symmetric geometries.

    Sphere (solid, uniform rho):
      Outside (r>R):  E = Q/(4*pi*eps0*r^2)  -- same as point charge
      Inside  (r<R):  E = Q*r/(4*pi*eps0*R^3)  -- grows linearly

    Cylinder (length L >> r, linear charge lambda [C/m]):
      E = lambda / (2*pi*eps0*r)

    Infinite plane (surface charge sigma [C/m^2]):
      E = sigma / (2*eps0)    [on each side]
    """
    # Sphere
    E_sphere_outside = Q_total_C / (4*np.pi*eps0*r_m**2)
    # Assume R=0.05m, same charge
    R_sphere = 0.05
    if r_m < R_sphere:
        E_sphere_inside = Q_total_C * r_m / (4*np.pi*eps0*R_sphere**3)
    else:
        E_sphere_inside = None

    # Cylinder: lambda = Q/L, assume L=1m
    lam = Q_total_C / 1.0
    E_cylinder = lam / (2*np.pi*eps0*r_m)

    # Plane: sigma = Q/A, assume A=1m^2
    sigma = Q_total_C / 1.0
    E_plane = sigma / (2*eps0)

    return {
        'sphere_outside_N_per_C': E_sphere_outside,
        'sphere_inside_N_per_C': E_sphere_inside,
        'cylinder_N_per_C': E_cylinder,
        'plane_N_per_C': E_plane,
        'gauss_trick': 'Pick surface where E is CONSTANT and PARALLEL to dA -> E*A = Q_enc/eps0',
    }


def electric_potential_point_charge(q_C, r_m):
    """
    V = k*q / r  [volts]   (scalar -- easier than E for multiple charges)

    Key calculus: E = -grad(V)
      In 1D: E_x = -dV/dx

    Superposition of potentials: V_total = sum(k*qi/ri)  (scalars, no vectors!)
    """
    if r_m <= 0:
        raise ValueError("r_m must be > 0")
    return k_e * q_C / r_m


def potential_from_field_line_integral(E_func, a_m, b_m, n=1000):
    """
    V(b) - V(a) = -integral_a^b E(x) dx

    E_func: callable E(x), uniform field or arbitrary
    a_m, b_m: start and end positions [m]
    n: integration points

    AP Physics C: this is the DEFINITION of potential.
    Gradient theorem: V is path-independent (conservative field).
    """
    x = np.linspace(a_m, b_m, n)
    dx = (b_m - a_m) / (n - 1)
    E_vals = np.array([E_func(xi) for xi in x])
    delta_V = -np.trapezoid(E_vals, x)
    return {'delta_V_volts': delta_V, 'a_m': a_m, 'b_m': b_m}


def sympy_gradient_demo():
    """
    Symbolic gradient: E = -grad(V)
    Demo: V = k*q/r for point charge -> E = k*q/r^2 (radial)

    AP Physics C: the del operator
      grad(f) = (df/dx, df/dy, df/dz)
      div(F)  = dFx/dx + dFy/dy + dFz/dz
      curl(F) = (dFz/dy-dFy/dz, dFx/dz-dFz/dx, dFy/dx-dFx/dy)
    """
    x, y, z, q, r_sym, k = sp.symbols('x y z q r k', positive=True)
    r_expr = sp.sqrt(x**2 + y**2 + z**2)

    V = k * q / r_expr

    # Gradient (Cartesian)
    dVdx = sp.diff(V, x)
    dVdy = sp.diff(V, y)
    dVdz = sp.diff(V, z)

    # E = -grad(V), radial component
    Ex = sp.simplify(-dVdx)
    Ey = sp.simplify(-dVdy)
    Ez = sp.simplify(-dVdz)

    # Magnitude |E| = k*q/r^2 (verify)
    E_mag = sp.simplify(sp.sqrt(Ex**2 + Ey**2 + Ez**2))

    return {
        'V': V,
        'Ex': Ex, 'Ey': Ey, 'Ez': Ez,
        'E_magnitude': E_mag,
        'check': 'E_mag should simplify to k*q/(x^2+y^2+z^2)',
        'ap_lesson': 'V = kq/r is scalar; differentiate to get vector E. Easier than Coulomb integral.',
    }


# ===========================================================================
# UNIT 2: Capacitors
# ===========================================================================

def parallel_plate_capacitor(A_m2, d_m, kappa=1.0):
    """
    C = kappa * eps0 * A / d  [farads]
    kappa: dielectric constant (1 = vacuum, ~80 for water, ~3.9 for SiO2)

    Energy stored: U = Q^2/(2C) = C*V^2/2 = Q*V/2  [joules]
    E field inside: E = V/d = Q/(C*d) = sigma/eps0
    """
    if A_m2 <= 0 or d_m <= 0:
        raise ValueError("A and d must be positive")
    C = kappa * eps0 * A_m2 / d_m
    return {
        'C_farads': C,
        'C_pF': C * 1e12,
        'kappa': kappa,
        'energy_formula': 'U = Q^2/(2C) = C*V^2/2',
        'E_field_V_per_m': '= V/d (uniform between plates)',
    }


def capacitor_energy(C_F, V_volts=None, Q_coulombs=None):
    """
    U = C*V^2/2 = Q^2/(2C) = Q*V/2
    Provide either V or Q (or both for consistency check).

    AP Physics C: derive by integrating work to add dq:
      dU = V*dq = (q/C)*dq
      U = int_0^Q (q/C) dq = Q^2/(2C)   <-- calculus!
    """
    if V_volts is not None:
        U = 0.5 * C_F * V_volts**2
        Q = C_F * V_volts
    elif Q_coulombs is not None:
        U = Q_coulombs**2 / (2*C_F)
        Q = Q_coulombs
        V_volts = Q / C_F
    else:
        raise ValueError("Provide V_volts or Q_coulombs")
    return {'U_joules': U, 'Q_coulombs': Q, 'V_volts': V_volts, 'C_F': C_F}


def capacitors_series_parallel(C_list_F, config='parallel'):
    """
    Parallel: C_eq = sum(C_i)          -- same voltage, add capacitance
    Series:   1/C_eq = sum(1/C_i)      -- same charge, add inverse

    AP mnemonic: Capacitors parallel = add (like resistors series)
    """
    if config == 'parallel':
        C_eq = sum(C_list_F)
    elif config == 'series':
        C_eq = 1.0 / sum(1.0/c for c in C_list_F)
    else:
        raise ValueError("config must be 'parallel' or 'series'")
    return {'C_eq_F': C_eq, 'config': config, 'C_eq_uF': C_eq*1e6}


# ===========================================================================
# UNIT 3: Electric Circuits (RC)
# ===========================================================================

def rc_circuit(R_ohm, C_F, V0_volts=5.0, t_max_s=None, n=500):
    """
    RC charging circuit: V_C(t) = V0 * (1 - exp(-t/tau))
    RC discharging:      V_C(t) = V0 * exp(-t/tau)
    tau = R*C  [seconds]  -- time constant (63.2% of final value)

    ODE derivation (AP Physics C calculus):
      Kirchhoff voltage: V0 = I*R + V_C = R*C*dV_C/dt + V_C
      dV_C/dt + V_C/(RC) = V0/(RC)
      Solution: V_C(t) = V0*(1 - e^(-t/RC))   (charging from 0)

    At t=tau:  V_C = V0*(1-1/e) = 0.632*V0
    At t=5tau: V_C ~ 0.993*V0  (fully charged for practical purposes)
    """
    tau = R_ohm * C_F
    if t_max_s is None:
        t_max_s = 5 * tau
    t = np.linspace(0, t_max_s, n)

    V_charge    = V0_volts * (1 - np.exp(-t/tau))
    V_discharge = V0_volts * np.exp(-t/tau)
    I_charge    = (V0_volts / R_ohm) * np.exp(-t/tau)   # I=C*dV/dt

    return {
        't_s': t,
        'V_charge_V': V_charge,
        'V_discharge_V': V_discharge,
        'I_charge_A': I_charge,
        'tau_s': tau,
        'tau_ms': tau*1e3,
        'V_at_tau': V0_volts * (1 - 1/np.e),
        'energy_stored_J': 0.5 * C_F * V0_volts**2,
        'energy_dissipated_J': 0.5 * C_F * V0_volts**2,  # equal split R vs C
    }


def kirchhoff_two_loop():
    """
    Solve a two-loop circuit with Kirchhoff's laws (linear algebra).
    Circuit:
      V1=12V -- R1=4ohm -- node A -- R2=6ohm -- node B -- V2=6V -- back to V1
                                  |                    |
                                  R3=3ohm              |
                                  |____________________|

    Method: KVL gives 2 equations in 2 unknowns (I1, I2).
    Solve: A*I = b  using numpy.

    AP Physics C: Kirchhoff's laws are conservation laws:
      KCL: sum of currents INTO node = 0  (charge conservation)
      KVL: sum of voltages around loop = 0  (energy conservation)
    """
    # Loop 1: V1 - I1*R1 - (I1-I2)*R3 = 0  ->  7*I1 - 3*I2 = 12
    # Loop 2: V2 + (I1-I2)*R3 - I2*R2 = 0  -> -3*I1 + 9*I2 = 6
    A = np.array([[7.0, -3.0],
                  [-3.0, 9.0]])
    b = np.array([12.0, 6.0])
    I = np.linalg.solve(A, b)
    I1, I2 = I
    P_R1 = I1**2 * 4
    P_R2 = I2**2 * 6
    P_R3 = (I1-I2)**2 * 3

    return {
        'I1_A': round(I1, 4),
        'I2_A': round(I2, 4),
        'I3_A': round(I1-I2, 4),
        'P_R1_W': round(P_R1, 4),
        'P_R2_W': round(P_R2, 4),
        'P_R3_W': round(P_R3, 4),
        'method': 'KVL -> linear system A*I = b -> np.linalg.solve',
        'ap_lesson': 'Kirchhoff = conservation. Set up loop equations, solve linear algebra.',
    }


# ===========================================================================
# UNIT 4: Magnetic Fields
# ===========================================================================

def biot_savart_segment(I_A, dl_m, dl_vec, r_vec):
    """
    Biot-Savart Law (differential form):
      dB = (mu0/(4*pi)) * I * (dl x r_hat) / r^2

    I_A: current [A]
    dl_m: length of segment [m]
    dl_vec: direction of current (unit vector, 3-element array)
    r_vec: vector from segment to field point [m]

    AP Physics C: Biot-Savart is to magnetism what Coulomb is to electrostatics.
    Hard integral -- use Ampere's law when symmetry allows.
    """
    dl_vec = np.array(dl_vec, dtype=float)
    r_vec  = np.array(r_vec,  dtype=float)
    r_mag  = np.linalg.norm(r_vec)
    if r_mag == 0:
        raise ValueError("Field point cannot be at the current element")
    r_hat = r_vec / r_mag
    dB_vec = (mu0 / (4*np.pi)) * I_A * dl_m * np.cross(dl_vec, r_hat) / r_mag**2
    return {
        'dB_vec_T': dB_vec,
        'dB_mag_T': np.linalg.norm(dB_vec),
        'formula': 'dB = mu0*I*dl x r_hat / (4*pi*r^2)',
    }


def ampere_law_geometries(I_A=1.0, r_m=0.01):
    """
    Ampere's Law (integral form):
      closed_loop_integral(B . dl) = mu0 * I_enc

    Three symmetric cases where B is constant on Amperian loop:

    Infinite straight wire:
      B = mu0*I / (2*pi*r)   [T]  (circular field lines)

    Toroid (N turns, mean radius R):
      B_inside = mu0*N*I / (2*pi*R)
      B_outside = 0

    Solenoid (n turns/meter):
      B_inside = mu0*n*I   [T]  (uniform!)
      B_outside ~ 0
    """
    B_wire = mu0 * I_A / (2*np.pi*r_m)

    N_toroid = 100; R_toroid = 0.05
    B_toroid_inside = mu0 * N_toroid * I_A / (2*np.pi*R_toroid)

    n_sol = 1000  # turns/meter
    B_solenoid = mu0 * n_sol * I_A

    return {
        'B_wire_T': B_wire,
        'B_wire_mT': B_wire*1e3,
        'B_toroid_inside_T': B_toroid_inside,
        'B_solenoid_T': B_solenoid,
        'B_solenoid_mT': B_solenoid*1e3,
        'ampere_trick': 'Pick loop where B is CONSTANT and PARALLEL to dl -> B*2*pi*r = mu0*I',
        'symmetry_needed': 'Infinite wire, toroid, solenoid. NOT a finite wire.',
    }


def lorentz_force(q_C, v_vec, E_vec, B_vec):
    """
    F = q*(E + v x B)  [Newtons]

    AP Physics C: this is the DEFINITION of E and B.
    v x B produces force perpendicular to both v and B -> circular motion.
    """
    v = np.array(v_vec, dtype=float)
    E = np.array(E_vec, dtype=float)
    B = np.array(B_vec, dtype=float)
    F = q_C * (E + np.cross(v, B))
    return {
        'F_vec_N': F,
        'F_mag_N': np.linalg.norm(F),
        'electric_part_N': q_C * E,
        'magnetic_part_N': q_C * np.cross(v, B),
    }


def cyclotron_radius(m_kg, q_C, v_m_per_s, B_T):
    """
    Circular motion in magnetic field: F_B = q*v*B = m*v^2/r
    r = m*v / (q*B)   [cyclotron radius]
    omega_c = q*B/m   [cyclotron frequency, rad/s]
    f_c = omega_c / (2*pi)   [Hz]

    AP: magnetic force does NO WORK (always perpendicular to v).
    Speed unchanged, direction changes -> uniform circular motion.
    """
    r = m_kg * v_m_per_s / (abs(q_C) * B_T)
    omega_c = abs(q_C) * B_T / m_kg
    f_c = omega_c / (2*np.pi)
    return {
        'r_m': r,
        'omega_c_rad_per_s': omega_c,
        'f_c_Hz': f_c,
        'T_period_s': 1.0/f_c,
    }


# ===========================================================================
# UNIT 5: Faraday, Inductance, RL, LC
# ===========================================================================

def faraday_law(B_func, A_m2, dt=1e-6, t=0.0):
    """
    Faraday's Law: EMF = -d(Phi_B)/dt
    Phi_B = B * A * cos(theta)   [for uniform B, flat loop]

    Numerical derivative: EMF ~ -(Phi(t+dt) - Phi(t)) / dt

    B_func: callable B(t) [Tesla], returns scalar
    A_m2: loop area [m^2]
    t: time [s]

    AP Physics C: Lenz's law -- the MINUS sign means the induced current
    opposes the change (energy conservation). If flux increases, induced B opposes.
    """
    Phi_t  = B_func(t) * A_m2
    Phi_dt = B_func(t + dt) * A_m2
    EMF = -(Phi_dt - Phi_t) / dt
    return {
        'EMF_volts': EMF,
        'Phi_B_Wb': Phi_t,
        'dPhi_dt': (Phi_dt - Phi_t)/dt,
        'lenz_law': 'Induced current opposes change in flux (energy conservation)',
        'connection_to_repo': 'dPhi/dt = derivative = same structure as d/dt[phase] = instantaneous freq in coppinger1999',
    }


def self_inductance_solenoid(N_turns, A_m2, l_m):
    """
    L = mu0 * N^2 * A / l   [Henries]  for solenoid

    AP Physics C: self-inductance L defined by EMF = -L * dI/dt
    Energy stored: U = L*I^2/2   (magnetic analog of U = C*V^2/2)

    Units check: [H] = [V*s/A] = [Wb/A]
    """
    L = mu0 * N_turns**2 * A_m2 / l_m
    return {
        'L_H': L,
        'L_mH': L*1e3,
        'L_uH': L*1e6,
        'energy_formula': 'U = L*I^2/2  [J]',
        'emf_formula': 'EMF = -L * dI/dt',
    }


def rl_circuit(R_ohm, L_H, V0_volts=5.0, t_max_s=None, n=500):
    """
    RL circuit (switch closes at t=0):
      I(t) = (V0/R) * (1 - exp(-t*R/L))   [A]
      V_L(t) = V0 * exp(-t*R/L)            [V] (inductor)
      V_R(t) = V0 * (1 - exp(-t*R/L))     [V] (resistor)
      tau = L/R  [s]

    ODE derivation (AP Physics C calculus):
      KVL: V0 = I*R + L*dI/dt
      dI/dt + (R/L)*I = V0/L
      Solution: I(t) = (V0/R)*(1 - e^(-R*t/L))

    Inductor RESISTS change in current (dual of capacitor resisting change in voltage).
    """
    tau = L_H / R_ohm
    if t_max_s is None:
        t_max_s = 5 * tau
    t = np.linspace(0, t_max_s, n)

    I = (V0_volts/R_ohm) * (1 - np.exp(-t/tau))
    V_L = V0_volts * np.exp(-t/tau)
    V_R = V0_volts * (1 - np.exp(-t/tau))

    return {
        't_s': t,
        'I_A': I,
        'V_L_V': V_L,
        'V_R_V': V_R,
        'tau_s': tau,
        'I_final_A': V0_volts/R_ohm,
        'energy_final_J': 0.5 * L_H * (V0_volts/R_ohm)**2,
    }


def lc_oscillator(L_H, C_F, V0_volts=1.0, n=1000):
    """
    LC oscillator: undamped sinusoidal oscillation
      omega0 = 1/sqrt(L*C)   [rad/s]
      f0 = omega0 / (2*pi)   [Hz]
      V_C(t) = V0 * cos(omega0 * t)
      I(t) = (V0/omega0*L) * sin(omega0*t) = C*V0*omega0 * sin(omega0*t)

    ODE: L*C*d^2V_C/dt^2 + V_C = 0  -- same as SHM (F=-kx)!

    Analog: LC <-> mass-spring
      L = mass (inertia of current)
      C = 1/spring_constant (stores energy in E field)
      V_C <-> position x
      I  <-> velocity v

    AP Physics C: LC oscillation conserves energy:
      U_E = C*V_C^2/2  (electric, max when I=0)
      U_B = L*I^2/2    (magnetic, max when V_C=0)
      Total U = constant
    """
    omega0 = 1.0 / np.sqrt(L_H * C_F)
    f0 = omega0 / (2*np.pi)
    T = 1.0 / f0
    t = np.linspace(0, 3*T, n)

    V_C = V0_volts * np.cos(omega0 * t)
    I   = C_F * V0_volts * omega0 * np.sin(omega0 * t)
    U_E = 0.5 * C_F * V_C**2
    U_B = 0.5 * L_H * I**2
    U_total = U_E + U_B

    return {
        't_s': t,
        'V_C_V': V_C,
        'I_A': I,
        'U_E_J': U_E,
        'U_B_J': U_B,
        'U_total_J': U_total,
        'omega0_rad_per_s': omega0,
        'f0_Hz': f0,
        'T_period_s': T,
        'energy_conserved': np.allclose(U_total, U_total[0], atol=1e-12),
        'analogy': 'LC = mass-spring: L<->mass, C<->1/k, V_C<->x, I<->v',
    }


# ===========================================================================
# PROBABILITY & STATISTICS for E&M lab
# ===========================================================================

def gaussian_measurement(true_value, sigma, n_measurements=1000, seed=42):
    """
    Model repeated measurement with Gaussian (normal) distribution.
    Central Limit Theorem: any sum of independent random variables -> Gaussian.

    true_value: the actual physical quantity
    sigma: standard deviation (instrument precision)
    Returns: measurements, mean, std, standard error of mean (SEM)

    AP lab connection:
      - sigma = systematic + random uncertainty
      - SEM = sigma/sqrt(N) -- more measurements -> more precise MEAN
      - But SEM != sigma: the individual measurements still scatter by sigma
    """
    rng = np.random.default_rng(seed)
    measurements = rng.normal(true_value, sigma, n_measurements)
    mean = np.mean(measurements)
    std  = np.std(measurements, ddof=1)
    sem  = std / np.sqrt(n_measurements)
    return {
        'measurements': measurements,
        'true_value': true_value,
        'mean': mean,
        'std': std,
        'sem': sem,
        'n': n_measurements,
        'bias': mean - true_value,
        'within_1sigma': np.mean(np.abs(measurements - true_value) < sigma),
        'within_2sigma': np.mean(np.abs(measurements - true_value) < 2*sigma),
        'lesson': '68% within 1-sigma, 95% within 2-sigma (68-95-99.7 rule)',
    }


def error_propagation_em(R_ohm, dR, V_volts, dV):
    """
    Propagate measurement uncertainties through Ohm's law I = V/R.
    General formula: (dI/I)^2 = (dV/V)^2 + (dR/R)^2  (fractional errors add in quadrature)

    AP Physics C lab: when quantities multiply/divide, add RELATIVE errors in quadrature.
    When quantities add/subtract, add ABSOLUTE errors in quadrature.

    dR, dV: absolute uncertainties (1-sigma)
    Returns: I and dI
    """
    I = V_volts / R_ohm
    dI = I * np.sqrt((dV/V_volts)**2 + (dR/R_ohm)**2)

    # Power P = I*V = V^2/R
    P = V_volts**2 / R_ohm
    dP = P * np.sqrt((2*dV/V_volts)**2 + (dR/R_ohm)**2)

    # Energy U = P*t for t=1s
    return {
        'I_A': I,
        'dI_A': dI,
        'I_relative_error': dI/I,
        'P_W': P,
        'dP_W': dP,
        'formula': 'dI/I = sqrt((dV/V)^2 + (dR/R)^2)',
        'rule': 'Multiply/divide -> add fractional errors in quadrature',
    }


def chi_squared_test(observed, expected):
    """
    Chi-squared goodness-of-fit test:
      chi2 = sum((O_i - E_i)^2 / E_i)
      dof = N - 1 - n_params  (degrees of freedom)

    chi2/dof ~ 1.0 -> good fit
    chi2/dof >> 1 -> model doesn't fit data (underestimate errors or wrong model)
    chi2/dof << 1 -> overestimate errors

    AP Physics C: use this to judge if your RC circuit model fits your data.
    """
    observed = np.array(observed, dtype=float)
    expected = np.array(expected, dtype=float)
    if len(observed) != len(expected):
        raise ValueError("observed and expected must have same length")
    chi2 = np.sum((observed - expected)**2 / np.where(expected > 0, expected, 1))
    dof = len(observed) - 1
    return {
        'chi2': chi2,
        'dof': dof,
        'chi2_per_dof': chi2/dof if dof > 0 else float('inf'),
        'interpretation': (
            'GOOD FIT' if 0.5 < chi2/dof < 2.0 else
            'OVER-DISPERSED (model wrong or errors underestimated)' if chi2/dof > 2.0 else
            'UNDER-DISPERSED (errors overestimated)'
        ),
    }


def linear_regression_em(x, y, dy=None):
    """
    Weighted linear regression: y = m*x + b
    Used in AP Physics C labs (e.g., V vs I for resistance, V_C vs t for RC time constant).

    dy: measurement uncertainties on y (if None: unweighted)
    Returns: slope m, intercept b, uncertainties dm, db, R^2

    AP use case: plot ln(V_C/V0) vs t for RC decay -> slope = -1/tau
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    if dy is not None:
        w = 1.0 / np.array(dy, dtype=float)**2
    else:
        w = np.ones(n)
    S   = np.sum(w)
    Sx  = np.sum(w * x)
    Sy  = np.sum(w * y)
    Sxx = np.sum(w * x**2)
    Sxy = np.sum(w * x * y)
    det = S * Sxx - Sx**2
    m = (S * Sxy - Sx * Sy) / det
    b = (Sxx * Sy - Sx * Sxy) / det
    dm = np.sqrt(S / det)
    db = np.sqrt(Sxx / det)
    y_pred = m * x + b
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 1.0
    return {
        'slope': m, 'intercept': b,
        'dm': dm, 'db': db,
        'R_squared': R2,
        'y_pred': y_pred,
        'residuals': y - y_pred,
    }


def rc_fit_from_data(t_s, V_C_V, V0_V):
    """
    Extract tau from RC decay data using linear regression on ln(V_C/V0) vs t.
    ln(V_C/V0) = -t/tau  ->  slope = -1/tau

    AP Physics lab technique: linearize exponential by taking log.
    Same technique works for radioactive decay, population growth, etc.
    """
    t_s = np.array(t_s)
    V_C_V = np.array(V_C_V)
    mask = (V_C_V > 0) & (V_C_V < V0_V)
    y = np.log(V_C_V[mask] / V0_V)
    result = linear_regression_em(t_s[mask], y)
    tau_fit = -1.0 / result['slope']
    dtau = tau_fit**2 * result['dm']
    return {
        'tau_fit_s': tau_fit,
        'dtau_s': dtau,
        'slope': result['slope'],
        'R_squared': result['R_squared'],
        'method': 'ln(V/V0) vs t -> linear; slope = -1/tau',
    }


# ===========================================================================
# SELF-TUTOR: AP Problems with solutions
# ===========================================================================

AP_PROBLEMS = [
    {
        'unit': 1,
        'topic': 'Gauss Law - spherical shell',
        'problem': (
            'A spherical shell of radius R=0.1m carries total charge Q=4uC. '
            'Find E at r=0.05m (inside) and r=0.2m (outside).'
        ),
        'solution': (
            'Inside (r<R): NO enclosed charge -> E=0. '
            'Gauss surface (sphere r=0.05m): Q_enc=0 -> E*4*pi*r^2=0 -> E=0. '
            'Outside (r>R): Q_enc=Q -> E=kQ/r^2 = (8.99e9*4e-6)/0.04 = 899 kN/C. '
            'Key: conductor or shell -> E=0 inside.'
        ),
        'numeric': lambda: {
            'E_inside': 0.0,
            'E_outside': k_e * 4e-6 / 0.2**2
        },
    },
    {
        'unit': 3,
        'topic': 'RC circuit time constant',
        'problem': (
            'An RC circuit has R=10kohm, C=100uF. Charged to V0=9V. '
            'Find tau, V_C at t=1s, time to reach 1V.'
        ),
        'solution': (
            'tau = R*C = 10e3 * 100e-6 = 1.0 s. '
            'V_C(t) = 9*exp(-t/1.0). '
            'V_C(1s) = 9/e = 3.31V. '
            'Time to 1V: t = -tau*ln(1/9) = 1.0*ln(9) = 2.197s.'
        ),
        'numeric': lambda: {
            'tau_s': 10e3 * 100e-6,
            'V_at_1s': 9*np.exp(-1.0),
            't_to_1V': -1.0*np.log(1.0/9.0)
        },
    },
    {
        'unit': 5,
        'topic': 'LC oscillator frequency',
        'problem': (
            'An LC circuit has L=2mH, C=50pF. Find the resonant frequency f0. '
            'If the initial voltage is V0=5V, what is the maximum current?'
        ),
        'solution': (
            'omega0 = 1/sqrt(LC) = 1/sqrt(2e-3 * 50e-12) = 1e7 rad/s. '
            'f0 = omega0/(2*pi) = 1.59 MHz. '
            'Max current: energy conservation: L*I_max^2/2 = C*V0^2/2 '
            '-> I_max = V0*sqrt(C/L) = 5*sqrt(50e-12/2e-3) = 5*5e-3 = 25 mA.'
        ),
        'numeric': lambda: {
            'f0_MHz': 1/(2*np.pi*np.sqrt(2e-3*50e-12)) / 1e6,
            'I_max_mA': 5.0 * np.sqrt(50e-12/2e-3) * 1e3
        },
    },
    {
        'unit': 'stats',
        'topic': 'Error propagation in resistance measurement',
        'problem': (
            'You measure V=5.00+-0.05V and I=0.250+-0.003A. '
            'Find R and its uncertainty using R=V/I.'
        ),
        'solution': (
            'R = V/I = 5.00/0.250 = 20.0 ohm. '
            'dR/R = sqrt((dV/V)^2 + (dI/I)^2) = sqrt((0.01)^2 + (0.012)^2) = 0.01562. '
            'dR = 20.0 * 0.01562 = 0.312 ohm. '
            'Report: R = 20.0 +- 0.3 ohm (1 sig fig on uncertainty).'
        ),
        'numeric': lambda: error_propagation_em(20.0, 0.312, 5.0, 0.05),
    },
]


def run_problem(index=0):
    """Run an AP problem and verify the numeric answer."""
    p = AP_PROBLEMS[index]
    result = p['numeric']()
    return {
        'unit': p['unit'],
        'topic': p['topic'],
        'problem': p['problem'],
        'solution': p['solution'],
        'numeric_result': result,
    }


def demo():
    print("=== AP PHYSICS C: E&M SELF-TUTOR ===\n")

    print("--- Coulomb's Law ---")
    r = coulombs_law(1e-6, -2e-6, 0.1)
    print(f"  F = {r['F_N']:.4f} N  (negative = attractive)")

    print("\n--- Gauss Law: field outside sphere ---")
    g = gauss_law_geometries(Q_total_C=1e-6, r_m=0.1)
    print(f"  E_sphere_outside = {g['sphere_outside_N_per_C']:.1f} N/C")
    print(f"  E_cylinder = {g['cylinder_N_per_C']:.1f} N/C")

    print("\n--- Gradient demo (symbolic) ---")
    grad = sympy_gradient_demo()
    print(f"  V = {grad['V']}")
    print(f"  Ex = {grad['Ex']}")

    print("\n--- Capacitor energy ---")
    cap = capacitor_energy(100e-6, V_volts=9.0)
    print(f"  C=100uF, V=9V: U = {cap['U_joules']*1e3:.3f} mJ, Q = {cap['Q_coulombs']*1e3:.3f} mC")

    print("\n--- RC Circuit ---")
    rc = rc_circuit(10e3, 100e-6, 9.0)
    print(f"  tau = {rc['tau_ms']:.1f} ms")
    print(f"  V_C at tau = {rc['V_at_tau']:.3f} V (should be 63.2% of 9V = 5.687V)")

    print("\n--- RL Circuit ---")
    rl = rl_circuit(100, 0.1, 5.0)
    print(f"  tau = {rl['tau_s']*1e3:.1f} ms, I_final = {rl['I_final_A']:.3f} A")

    print("\n--- LC Oscillator ---")
    lc = lc_oscillator(2e-3, 50e-12, 5.0)
    print(f"  f0 = {lc['f0_Hz']/1e6:.3f} MHz")
    print(f"  Energy conserved: {lc['energy_conserved']}")

    print("\n--- Ampere's Law ---")
    amp = ampere_law_geometries(I_A=1.0, r_m=0.01)
    print(f"  B at 1cm from wire = {amp['B_wire_mT']:.3f} mT")
    print(f"  B in solenoid (1000 t/m) = {amp['B_solenoid_mT']:.3f} mT")

    print("\n--- AP Problem: LC frequency ---")
    p = run_problem(2)
    print(f"  {p['problem']}")
    print(f"  Answer: f0={p['numeric_result']['f0_MHz']:.3f} MHz, "
          f"I_max={p['numeric_result']['I_max_mA']:.1f} mA")

    print("\n--- Error Propagation ---")
    err = error_propagation_em(R_ohm=20.0, dR=0.1, V_volts=5.0, dV=0.05)
    print(f"  R=20ohm+-0.1, V=5V+-0.05V")
    print(f"  I = {err['I_A']:.4f} A +- {err['dI_A']*1e3:.2f} mA")

    print("\n--- Gaussian Measurement ---")
    meas = gaussian_measurement(true_value=9.0, sigma=0.1, n_measurements=1000)
    print(f"  True=9.0V, sigma=0.1V, N=1000")
    print(f"  Measured mean={meas['mean']:.4f}V, SEM={meas['sem']*1e3:.2f}mV")
    print(f"  Within 2-sigma: {meas['within_2sigma']*100:.1f}% (expect 95%)")

    print("\n=== DONE ===")


if __name__ == '__main__':
    demo()
