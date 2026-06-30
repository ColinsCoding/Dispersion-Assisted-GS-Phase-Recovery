"""Gyroscopes: center of mass, moment of inertia, torque-driven precession,
small-angle approximation, and the electrical-engineering analog circuit.

A spinning gyroscope precesses because torque = dL/dt, and L is nearly
horizontal (spin-aligned) for a fast top -- so dL/dt is also horizontal,
rotating L around the vertical axis instead of tipping it over. That's
precession: Omega_p = tau / (I * omega_spin) = m*g*r / (I*omega_spin).

Small-angle approximation (sin(theta) ~ theta for theta << 1 rad) turns the
nonlinear pendulum ODE d^2theta/dt^2 = -(g/L)*sin(theta) into the linear SHM
ODE d^2theta/dt^2 = -(g/L)*theta -- same trick used for nutation in a
gyroscope's wobble about the steady precession cone.

EE analogy: steady-state precession (constant Omega_p driven by constant
torque) is the *resistive* limit -- like a circuit's DC steady state.
Nutation (the wobble) is the *oscillatory* limit -- like an undriven RLC
circuit's resonant ringing. Same linear-ODE machinery (dgs.spice's RLC RK4
solver) applies to both.
"""
import numpy as np
import sympy as sp


# -- Center of mass and moment of inertia ------------------------------------

def center_of_mass(masses, positions):
    """R_cm = sum(m_i * r_i) / sum(m_i).  positions: (N, dim) array."""
    m = np.asarray(masses, float)
    r = np.asarray(positions, float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    total_m = m.sum()
    if total_m <= 0:
        raise ValueError("total mass must be positive")
    R_cm = (m[:, None] * r).sum(axis=0) / total_m
    return {"R_cm": R_cm, "total_mass": total_m}


def moment_of_inertia_point_masses(masses, radii):
    """I = sum(m_i * r_i^2) about a given axis -- radii are perpendicular
    distances from that axis, not full 3D positions."""
    m = np.asarray(masses, float)
    r = np.asarray(radii, float)
    return float(np.sum(m * r**2))


# Standard shapes about their symmetry axis (used for the spinning disk/rotor)
def moment_of_inertia_disk(mass, radius):
    """Solid disk about its central symmetry axis: I = (1/2) m R^2."""
    return 0.5 * mass * radius**2


def moment_of_inertia_hoop(mass, radius):
    """Thin hoop/ring about its central axis: I = m R^2."""
    return mass * radius**2


def moment_of_inertia_rod_center(mass, length):
    """Thin rod about its center, perpendicular to its length: I = (1/12) m L^2."""
    return mass * length**2 / 12.0


def parallel_axis_theorem(I_cm, mass, d):
    """I_parallel = I_cm + m*d^2 -- shift the axis a distance d from the COM."""
    return I_cm + mass * d**2


# -- Small-angle approximation -------------------------------------------------

def small_angle_error(theta_rad):
    """sin(theta) vs theta -- relative error of the small-angle approximation."""
    theta = np.asarray(theta_rad, float)
    exact = np.sin(theta)
    approx = theta
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(theta != 0, np.abs(exact - approx) / np.abs(exact), 0.0)
    return {"theta_rad": theta, "sin_theta": exact, "small_angle": approx,
            "relative_error": rel_err}


def pendulum_period_small_angle(L, g=9.80665):
    """T = 2*pi*sqrt(L/g) -- valid only for small theta (linearized ODE)."""
    return 2 * np.pi * np.sqrt(L / g)


def pendulum_rk4(theta0_rad, omega0, L, g=9.80665, t_max=10.0, dt=0.001, small_angle=False):
    """Integrate the pendulum ODE with RK4.
    small_angle=True linearizes sin(theta)->theta (SHM); False keeps sin(theta)
    (the real nonlinear pendulum) -- compare periods to see where the
    approximation breaks down (theta0 not << 1 rad)."""
    def deriv(state):
        theta, omega = state
        alpha = -(g / L) * (theta if small_angle else np.sin(theta))
        return np.array([omega, alpha])

    n = int(t_max / dt)
    t = np.zeros(n)
    theta = np.zeros(n)
    state = np.array([theta0_rad, omega0], float)
    for i in range(n):
        t[i] = i * dt
        theta[i] = state[0]
        k1 = deriv(state)
        k2 = deriv(state + dt / 2 * k1)
        k3 = deriv(state + dt / 2 * k2)
        k4 = deriv(state + dt * k3)
        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return {"t": t, "theta": theta}


# -- Gyroscopic precession -----------------------------------------------------

def precession_rate(mass, g, r, I_spin, omega_spin):
    """Omega_p = tau / (I*omega_spin) = m*g*r / (I*omega_spin) for a top with
    its pivot a horizontal distance r from the center of mass."""
    if omega_spin == 0:
        raise ValueError("omega_spin must be nonzero -- a non-spinning top just falls")
    tau = mass * g * r
    Omega_p = tau / (I_spin * omega_spin)
    return {"Omega_p_rad_s": Omega_p, "torque_Nm": tau,
            "precession_period_s": 2 * np.pi / Omega_p if Omega_p != 0 else np.inf}


def nutation_frequency(I_spin, I_transverse, omega_spin):
    """Fast-top approximation: nutation (wobble about the precession cone)
    angular frequency omega_n = I_spin * omega_spin / I_transverse."""
    return I_spin * omega_spin / I_transverse


def gyroscope_ee_analogy():
    """Map gyroscope precession/nutation onto an RLC circuit's steady-state /
    transient response -- same linear-ODE structure, different physics."""
    return {
        "Steady precession": {
            "mechanical": "Omega_p = m*g*r / (I*omega_spin), constant under constant torque",
            "electrical": "I_DC = V / R, constant current under constant voltage (resistive limit)",
        },
        "Nutation (wobble)": {
            "mechanical": "omega_n = I_spin*omega_spin / I_transverse, free oscillation about the cone",
            "electrical": "omega_0 = 1/sqrt(LC), free ringing of an undriven RLC circuit",
        },
        "common_math": "Both solve a 2nd-order linear ODE: a driven term gives a constant "
                        "(DC / steady precession) response, a homogeneous term gives an "
                        "oscillatory (resonant / nutating) response.",
    }


# -- SymPy equations ------------------------------------------------------------

def gyroscope_sympy_5():
    """Five symbolic equations: center of mass, moment of inertia (disk),
    small-angle pendulum ODE, precession rate, nutation frequency."""
    m, r, R, L, g_s, I_s, I_t, w_s = sp.symbols(
        'm r R L g I_s I_t omega_s', positive=True)
    theta = sp.Function('theta')
    t = sp.Symbol('t', positive=True)

    return {
        "Center_of_mass":
            sp.Eq(sp.Symbol('R_cm'),
                  sp.Sum(sp.Symbol('m_i') * sp.Symbol('r_i'), (sp.Symbol('i'), 1, sp.Symbol('N')))
                  / sp.Sum(sp.Symbol('m_i'), (sp.Symbol('i'), 1, sp.Symbol('N')))),
        "Moment_of_inertia_disk":
            sp.Eq(sp.Symbol('I'), sp.Rational(1, 2) * m * R**2),
        "Pendulum_small_angle_ODE":
            sp.Eq(sp.diff(theta(t), t, 2), -(g_s / L) * theta(t)),
        "Precession_rate":
            sp.Eq(sp.Symbol('Omega_p'), m * g_s * r / (I_s * w_s)),
        "Nutation_frequency":
            sp.Eq(sp.Symbol('omega_n'), I_s * w_s / I_t),
    }


if __name__ == "__main__":
    print("=== Center of mass: 3 masses on a line ===")
    com = center_of_mass([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    print(f"  R_cm = {com['R_cm']}, total mass = {com['total_mass']}")

    print("\n=== Moment of inertia: bicycle wheel rotor (hoop), m=0.5kg, R=0.3m ===")
    I = moment_of_inertia_hoop(0.5, 0.3)
    print(f"  I = {I:.5f} kg*m^2")

    print("\n=== Small-angle approximation error ===")
    sae = small_angle_error(np.array([0.01, 0.1, 0.5, 1.0]))
    for th, err in zip(sae["theta_rad"], sae["relative_error"]):
        print(f"  theta={th:.2f} rad -> relative error = {err*100:.3f}%")

    print("\n=== Pendulum period (small angle), L=1m ===")
    T = pendulum_period_small_angle(1.0)
    print(f"  T = {T:.4f} s")

    print("\n=== Gyroscope precession: spinning top ===")
    prec = precession_rate(mass=0.2, g=9.80665, r=0.05, I_spin=I, omega_spin=100.0)
    print(f"  Omega_p = {prec['Omega_p_rad_s']:.4f} rad/s, period = {prec['precession_period_s']:.2f} s")

    print("\n=== EE analogy ===")
    analogy = gyroscope_ee_analogy()
    for k, v in analogy.items():
        if isinstance(v, dict):
            print(f"  {k}: mech={v['mechanical']}")

    print("\n=== SymPy 5 ===")
    for k, eq in gyroscope_sympy_5().items():
        print(f"  {k}: {eq}")
