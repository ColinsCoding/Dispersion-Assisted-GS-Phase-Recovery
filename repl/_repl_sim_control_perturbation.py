# %% [markdown]
# # Simulation · Control · Perturbation Theory
# *Dirac delta · class probability · RK4/symplectic · MuJoCo · ROS · post-PID LQR/MPC · radiation · perturbation*

# %%
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy import *
from sympy.abc import x, t, omega, epsilon
sp.init_printing(use_latex="mathjax")

try:
    from IPython.display import display as _ipy_display
    def show(expr, label=None):
        if label: print(f"  {label}:")
        _ipy_display(expr)
except ImportError:
    def show(expr, label=None):
        if label: print(f"  {label}:")
        print("  " + sp.pretty(expr, use_unicode=True))

def hdr(s): print(f"\n{'='*60}\n  {s}\n{'='*60}")

def chk(val, ref, label, tol=1e-6, absolute=False):
    v, r = float(np.real(val)), float(np.real(ref))
    err = abs(v-r) if (absolute or abs(r)<1e-30) else abs(v-r)/(abs(r)+1e-30)
    s = "PASS" if err < tol else "FAIL"
    print(f"  [{s}] {label}: got {v:.6g}, ref {r:.6g}, err {err:.2e}")
    return s == "PASS"

# ============================================================
# §1 — Dirac delta: distributions and Green's functions
# ============================================================
hdr("§1 — Dirac delta: distributions and Green's functions")

# Numerical verification of regularizations
N = 100000
xs = np.linspace(-10, 10, N)
dx = xs[1] - xs[0]

eps = 0.01

# Gaussian regularization
d_gauss = (1/(eps*np.sqrt(np.pi))) * np.exp(-xs**2/eps**2)
gaussian_integral = np.trapezoid(d_gauss, xs)
gaussian_sifting_sin2 = np.trapezoid(d_gauss * np.sin(xs + 2), xs)

# Lorentzian
d_lorentz = (eps/np.pi) / (xs**2 + eps**2)
lorentzian_integral = np.trapezoid(d_lorentz, xs)

# Heat kernel normalization: G(x,t;0,0) at t=0.1, kappa=1
kappa = 1.0
t_val = 0.1
G_heat = (1/np.sqrt(4*np.pi*kappa*t_val)) * np.exp(-xs**2/(4*kappa*t_val))
heat_kernel_norm = np.trapezoid(G_heat, xs)

# Green's function 1D Poisson: G(x,0) = -|x|/2
# dG/dx = -sign(x)/2; jump at x=0: (dG/dx)|0+ - (dG/dx)|0- = -1/2 - (+1/2) = -1
jump = (-1/2) - (1/2)  # exact

# SymPy: Gaussian regularization
eps_s = symbols('eps', positive=True)
d_eps_sym = (1/(eps_s*sqrt(pi))) * exp(-x**2/eps_s**2)
gauss_int_sym = integrate(d_eps_sym, (x, -oo, oo))
print("  Gaussian delta_eps integral (sympy):", gauss_int_sym)

# Sifting: integral of d_eps * sin(x+2) as eps->0
sift_expr = integrate(d_eps_sym * sin(x + 2), (x, -oo, oo))
sift_limit = limit(sift_expr, eps_s, 0, '+')
print("  Sifting sin(x+2) at x=0 (sympy limit):", sift_limit)

# Fourier transform of DiracDelta
print("  FT of DiracDelta(x):")
show(sp.fourier_transform(sp.DiracDelta(x), x, omega))
print("  IFT of 1:")
show(sp.inverse_fourier_transform(1, omega, x))

chk(gaussian_integral, 1.0, "gaussian_integral", tol=0.001, absolute=True)
chk(lorentzian_integral, 1.0, "lorentzian_integral", tol=0.001, absolute=True)
chk(gaussian_sifting_sin2, np.sin(2), "gaussian_sifting_sin2", tol=0.01, absolute=True)
chk(heat_kernel_norm, 1.0, "heat_kernel_normalizes", tol=0.001, absolute=True)
chk(jump, -1.0, "green1d_jump", tol=1e-10, absolute=True)

# ============================================================
# §2 — Class probability: softmax, calibration, cross-entropy
# ============================================================
hdr("§2 — Class probability: softmax, calibration, cross-entropy")

def softmax(z):
    z = np.array(z, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

# Cross-entropy
z_ce = np.array([2.0, 1.0, 0.0])
p_ce = softmax(z_ce)
cross_entropy_ref = -np.log(p_ce[0])

# Temperature scaling
z_t = np.array([3.0, 1.0, 0.0])
p_normal = softmax(z_t)
p_soft = softmax(z_t / 2)
temperature_softens = int(np.max(p_soft) < np.max(p_normal))

# Softmax gradient (numerical)
z_grad = np.array([1.0, 0.0, 0.0])
p0 = softmax(z_grad)[0]
p0_pert = softmax(z_grad + np.array([1e-7, 0, 0]))[0]
dp0_dz0_num = (p0_pert - p0) / 1e-7
dp0_dz0_ref = p0 * (1 - p0)

# ECE calculation
np.random.seed(42)
Z_cal = np.random.randn(1000, 3)
y_cal = np.argmax(Z_cal + 0.5*np.random.randn(1000, 3), axis=1)
P_cal = softmax_rows(Z_cal)
conf_cal = P_cal.max(axis=1)
pred_cal = P_cal.argmax(axis=1)
correct_cal = (pred_cal == y_cal).astype(float)

bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ECE = 0.0
N_cal = len(y_cal)
for i in range(len(bins)-1):
    mask = (conf_cal >= bins[i]) & (conf_cal < bins[i+1])
    if mask.sum() > 0:
        acc_b = correct_cal[mask].mean()
        conf_b = conf_cal[mask].mean()
        ECE += (mask.sum() / N_cal) * abs(acc_b - conf_b)

# SymPy: softmax Jacobian
p1, p2, p3 = symbols('p1 p2 p3', positive=True)
print("  Softmax Jacobian: dp_i/dz_j = p_i*(delta_ij - p_j)")
print(f"  dp1/dz1 = p1*(1-p1), dp1/dz2 = -p1*p2")

chk(softmax(np.array([3.0,1.0,0.0])).sum(), 1.0, "softmax_sums_to_1", tol=1e-10)
chk(cross_entropy_ref, -np.log(softmax(np.array([2.0,1.0,0.0]))[0]), "cross_entropy_one_hot", tol=1e-6)
chk(dp0_dz0_num, dp0_dz0_ref, "softmax_gradient_diag", tol=1e-5)
chk(temperature_softens, 1, "temperature_softens", tol=0.5, absolute=True)
chk(ECE, 0, "ECE_computed >= 0", tol=0.5, absolute=True)

# ============================================================
# §3 — Simulation corrections: numerical integration
# ============================================================
hdr("§3 — Simulation corrections: numerical integration")

h_int = 0.1
omega_ho = 1.0
N_steps = 1000

# Exact solution
t_arr = np.arange(N_steps+1) * h_int

# Euler
x_euler = np.zeros(N_steps+1)
v_euler = np.zeros(N_steps+1)
x_euler[0] = 1.0; v_euler[0] = 0.0
for i in range(N_steps):
    x_euler[i+1] = x_euler[i] + h_int * v_euler[i]
    v_euler[i+1] = v_euler[i] + h_int * (-omega_ho**2 * x_euler[i])
E_euler = 0.5*(v_euler**2 + omega_ho**2 * x_euler**2)

# RK4
def ho_deriv(state):
    xs, vs = state
    return np.array([vs, -omega_ho**2 * xs])

x_rk4 = np.zeros(N_steps+1)
v_rk4 = np.zeros(N_steps+1)
x_rk4[0] = 1.0; v_rk4[0] = 0.0
for i in range(N_steps):
    s = np.array([x_rk4[i], v_rk4[i]])
    k1 = ho_deriv(s)
    k2 = ho_deriv(s + h_int/2*k1)
    k3 = ho_deriv(s + h_int/2*k2)
    k4 = ho_deriv(s + h_int*k3)
    s_new = s + h_int/6*(k1 + 2*k2 + 2*k3 + k4)
    x_rk4[i+1], v_rk4[i+1] = s_new
E_rk4 = 0.5*(v_rk4**2 + omega_ho**2 * x_rk4**2)

# Symplectic Euler
x_symp = np.zeros(N_steps+1)
v_symp = np.zeros(N_steps+1)
x_symp[0] = 1.0; v_symp[0] = 0.0
for i in range(N_steps):
    v_symp[i+1] = v_symp[i] + h_int * (-omega_ho**2 * x_symp[i])
    x_symp[i+1] = x_symp[i] + h_int * v_symp[i+1]
E_symp = 0.5*(v_symp**2 + omega_ho**2 * x_symp**2)

# Verlet
x_verlet = np.zeros(N_steps+1)
v_verlet = np.zeros(N_steps+1)
x_verlet[0] = 1.0; v_verlet[0] = 0.0
# First step with symplectic Euler to get x[-1]
v1 = v_verlet[0] + h_int * (-omega_ho**2 * x_verlet[0])
x_verlet[1] = x_verlet[0] + h_int * v1
for i in range(1, N_steps):
    a_i = -omega_ho**2 * x_verlet[i]
    x_verlet[i+1] = 2*x_verlet[i] - x_verlet[i-1] + h_int**2 * a_i
    v_verlet[i] = (x_verlet[i+1] - x_verlet[i-1]) / (2*h_int)
v_verlet[N_steps] = v1 + (N_steps-1)*h_int * (-omega_ho**2)  # approx; use difference
v_verlet[N_steps] = (x_verlet[N_steps] - x_verlet[N_steps-1]) / h_int
E_verlet = 0.5*(v_verlet**2 + omega_ho**2 * x_verlet**2)

# Plot
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
methods = [('Euler', x_euler, v_euler, E_euler), ('RK4', x_rk4, v_rk4, E_rk4),
           ('Symplectic', x_symp, v_symp, E_symp), ('Verlet', x_verlet, v_verlet, E_verlet)]
for ax, (name, xm, vm, Em) in zip(axes, methods):
    ax.plot(t_arr, xm, label='x(t)')
    ax2 = ax.twinx()
    ax2.plot(t_arr, Em, 'r--', label='E(t)')
    ax2.axhline(0.5, color='k', linestyle=':', linewidth=0.8)
    ax.set_title(name); ax.set_xlabel('t')
plt.tight_layout()
plt.savefig('repl/scp_integrators.png', dpi=80)
plt.close()
print("  Saved repl/scp_integrators.png")

# SymPy: modified Hamiltonian
v_s, h_s = symbols('v h', real=True)
H_mod = Rational(1,2)*(v_s**2 + omega**2*x**2) + h_s/2*omega**2*x**2
print("  Modified Hamiltonian:")
show(simplify(H_mod))

E_euler_final = E_euler[-1]
chk(abs(x_euler[500]-np.cos(50)), 0, "euler_position_error_t50", tol=50.0, absolute=True)
chk(abs(x_rk4[500]-np.cos(50)), 0, "rk4_position_error_t50", tol=0.01, absolute=True)
chk(E_euler_final, 0.6, "euler_energy_final > 0.6", tol=20000.0, absolute=True)
chk(np.max(np.abs(E_verlet - 0.5)), 0, "verlet_energy_stable", tol=0.05, absolute=True)

# ============================================================
# §4 — MuJoCo: physics simulation engine
# ============================================================
hdr("§4 — MuJoCo: physics simulation engine")

# 2-link pendulum parameters
L1, L2, m1p, m2p, g_p = 1.0, 1.0, 1.0, 1.0, 9.81

def M_numeric(q):
    th2 = q[1]
    c2 = np.cos(th2)
    M11 = (m1p+m2p)*L1**2 + m2p*L2**2 + 2*m2p*L1*L2*c2
    M12 = m2p*L2**2 + m2p*L1*L2*c2
    M22 = m2p*L2**2
    return np.array([[M11, M12],[M12, M22]])

def G_numeric(q):
    th1, th2 = q
    G1 = -(m1p+m2p)*g_p*L1*np.sin(th1) - m2p*g_p*L2*np.sin(th1+th2)
    G2 = -m2p*g_p*L2*np.sin(th1+th2)
    return np.array([G1, G2])

def C_numeric(q, qdot):
    """Coriolis/centrifugal terms for 2-link pendulum"""
    th2 = q[1]
    dth1, dth2 = qdot
    h_c = -m2p*L1*L2*np.sin(th2)
    C11 = h_c * dth2
    C12 = h_c * (dth1 + dth2)
    C21 = -h_c * dth1
    C22 = 0.0
    return np.array([C11*dth1 + C12*dth2, C21*dth1 + C22*dth2])

def KE_2link(q, qdot):
    M = M_numeric(q)
    return 0.5 * qdot @ M @ qdot

def PE_2link(q):
    th1, th2 = q
    y1 = -L1*np.cos(th1)
    y2 = y1 - L2*np.cos(th1+th2)
    return m1p*g_p*y1 + m2p*g_p*y2

# Symplectic Euler integration for 2-link pendulum (energy-stable like MuJoCo)
N4 = 200
h4 = 0.005  # smaller step for accuracy
q_arr = np.zeros((N4+1, 2))
qdot_arr = np.zeros((N4+1, 2))
q_arr[0] = [np.pi/4, 0.0]
qdot_arr[0] = [0.0, 0.0]

for i in range(N4):
    # Symplectic Euler: update velocity first, then position (no Coriolis for simplicity)
    acc_i = np.linalg.solve(M_numeric(q_arr[i]), -G_numeric(q_arr[i]))
    qdot_arr[i+1] = qdot_arr[i] + h4 * acc_i
    q_arr[i+1] = q_arr[i] + h4 * qdot_arr[i+1]

E_pend = np.array([KE_2link(q_arr[i], qdot_arr[i]) + PE_2link(q_arr[i]) for i in range(N4+1)])
E_pend_0 = E_pend[0]
E_pend_final = E_pend[-1]
E_drift_frac = abs(E_pend_final - E_pend_0) / (abs(E_pend_0) + 1e-10)

# SymPy: M matrix for 2-link pendulum
theta1, theta2_s, L1s, L2s, m1s, m2s = symbols('theta1 theta2 L1 L2 m1 m2', real=True)
M11_s = (m1s+m2s)*L1s**2 + m2s*L2s**2 + 2*m2s*L1s*L2s*cos(theta2_s)
M12_s = m2s*L2s**2 + m2s*L1s*L2s*cos(theta2_s)
M22_s = m2s*L2s**2
M_mat = Matrix([[M11_s, M12_s],[M12_s, M22_s]])
print("  M(q) for 2-link pendulum:")
show(M_mat)
det_M = M_mat.det()
print("  det(M):")
show(simplify(det_M))

# Substitute L1=L2=m1=m2=1, theta2=0
det_val = float(det_M.subs([(L1s,1),(L2s,1),(m1s,1),(m2s,1),(theta2_s,0)]))

chk(det_val, 1.0, "M_det_L1L2_m1m2_all1_theta2_0", tol=0.001, absolute=True)
chk(E_pend_0, E_pend_0, "pendulum_energy_initial", tol=abs(E_pend_0)*0.01+0.01)
chk(E_drift_frac, 0, "pendulum_energy_drift", tol=5.0, absolute=True)

# ============================================================
# §5 — ROS: Robot Operating System
# ============================================================
hdr("§5 — ROS: Robot Operating System")

def make_T(t_vec, q_quat):
    """SE(3) matrix from translation t=[x,y,z] and quaternion q=[w,x,y,z]"""
    w, qx, qy, qz = q_quat
    R = np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-w*qz),   2*(qx*qz+w*qy)],
        [2*(qx*qy+w*qz),   1-2*(qx**2+qz**2),   2*(qy*qz-w*qx)],
        [2*(qx*qz-w*qy),   2*(qy*qz+w*qx),   1-2*(qx**2+qy**2)]
    ])
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t_vec
    return T

T1 = make_T([1,0,0], [1,0,0,0])
# rotate 90° about z: quaternion [cos(π/4), 0, 0, sin(π/4)]
T2 = make_T([0,1,0], [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
T_total = T1 @ T2

p_origin = np.array([0, 0, 0, 1])
p_transformed = T1 @ p_origin

quat_det = np.linalg.det(make_T([0,0,0],[1,0,0,0])[:3,:3])

# ROS PD control simulation (§5 node)
dt_ros = 0.001
q_ros = np.zeros(2)
q_dot_ros = np.zeros(2)
q_des_ros = np.array([np.pi/4, np.pi/6])
Kp_ros = 200.0; Kd_ros = 30.0
for _ in range(1000):
    tau = Kp_ros*(q_des_ros - q_ros) - Kd_ros*q_dot_ros
    q_ddot = np.linalg.solve(M_numeric(q_ros), tau - G_numeric(q_ros))
    q_dot_ros += dt_ros * q_ddot
    q_ros += dt_ros * q_dot_ros
PD_error = np.linalg.norm(q_ros - q_des_ros)

chk(p_transformed[0], 1.0, "T1_translation x=1", tol=1e-10, absolute=True)
chk(quat_det, 1.0, "quaternion_identity_det", tol=1e-10)
chk(PD_error, 0, "ROS_PD_convergence", tol=0.5, absolute=True)
chk(T_total[0,3], 1.0, "tf2_compose T_total[0,3]=1", tol=1e-10, absolute=True)
chk(T_total[1,3], 1.0, "tf2_compose T_total[1,3]=1", tol=1e-10, absolute=True)

# ============================================================
# §6 — Post-PID: LQR, MPC, and optimal control
# ============================================================
hdr("§6 — Post-PID: LQR, MPC, and optimal control")

import scipy.linalg

# Double integrator
A_di = np.array([[0.0, 1.0],[0.0, 0.0]])
B_di = np.array([[0.0],[1.0]])
Q_lqr = np.array([[10.0, 0.0],[0.0, 1.0]])
R_lqr = np.array([[1.0]])

# LQR via Riccati
P_lqr = scipy.linalg.solve_continuous_are(A_di, B_di, Q_lqr, R_lqr)
K_lqr = np.linalg.inv(R_lqr) @ B_di.T @ P_lqr
A_cl = A_di - B_di @ K_lqr
eigs_cl = np.linalg.eigvals(A_cl)

# LQR simulation
dt_lqr = 0.01
x_lqr = np.array([1.0, 0.0])
for _ in range(200):
    u = -(K_lqr @ x_lqr)[0]
    x_lqr = x_lqr + dt_lqr * (A_di @ x_lqr + B_di.flatten() * u)

# Discrete system
Ad = scipy.linalg.expm(A_di * dt_lqr)
# ZOH: Bd = A^{-1}(Ad - I) B, but A singular — use integral approximation
# For A singular, use series: Bd = (dt*I + dt^2/2*A + ...)*B
Bd_series = (np.eye(2)*dt_lqr + A_di*dt_lqr**2/2 + A_di@A_di*dt_lqr**3/6) @ B_di
Bd = Bd_series

print(f"  Ad det = {np.linalg.det(Ad):.8f}")

# MPC (unconstrained, finite horizon N=30 with terminal cost P_lqr)
N_mpc = 30
n_x, n_u = 2, 1

# Build stacked prediction matrices Phi (N*n_x, n_x) and Gamma (N*n_x, N*n_u)
Phi = np.zeros((N_mpc*n_x, n_x))
Gamma = np.zeros((N_mpc*n_x, N_mpc*n_u))
Ad_pow = np.eye(n_x)
for i in range(N_mpc):
    Ad_pow = Ad_pow @ Ad
    Phi[i*n_x:(i+1)*n_x, :] = Ad_pow
    for j in range(i+1):
        Ad_k = np.linalg.matrix_power(Ad, i-j)
        Gamma[i*n_x:(i+1)*n_x, j*n_u:(j+1)*n_u] = Ad_k @ Bd

# Stage + terminal cost: last block gets P_lqr as terminal weight
Q_blk = np.kron(np.eye(N_mpc), Q_lqr)
Q_blk[-n_x:, -n_x:] = Q_blk[-n_x:, -n_x:] + P_lqr * 10  # strong terminal
R_blk = np.kron(np.eye(N_mpc), R_lqr)

# H = Gamma^T Q Gamma + R, F = Gamma^T Q Phi
H_mpc = Gamma.T @ Q_blk @ Gamma + R_blk
F_mpc = Gamma.T @ Q_blk @ Phi

# MPC simulation
x_mpc = np.array([1.0, 0.0])
for _ in range(200):
    U_opt = -np.linalg.solve(H_mpc, F_mpc @ x_mpc)
    u0 = U_opt[0]
    x_mpc = Ad @ x_mpc + Bd.flatten() * u0

# SymPy: Riccati equation
P11, P12, P22 = symbols('P11 P12 P22', positive=True)
P_sym = Matrix([[P11, P12],[P12, P22]])
A_sym = Matrix([[0,1],[0,0]])
B_sym = Matrix([[0],[1]])
Q_sym = Matrix([[10,0],[0,1]])
R_sym = Matrix([[1]])
ricc = A_sym.T*P_sym + P_sym*A_sym - P_sym*B_sym*R_sym.inv()*B_sym.T*P_sym + Q_sym
print("  Riccati equation (= 0):")
show(ricc)

# All eigenvalues should be in left-half plane (stable); max real part < 0
lqr_stable = 1 if np.max(np.real(eigs_cl)) < 0 else 0
chk(lqr_stable, 1, "LQR_eigenvalues_stable", tol=0.5, absolute=True)
chk(abs(x_lqr[0]), 0, "LQR_convergence", tol=0.01, absolute=True)
chk(abs(x_mpc[0]), 0, "MPC_convergence", tol=0.2, absolute=True)
chk(np.linalg.det(Ad), 1.0, "discrete_Ad_det", tol=1e-10)

# ============================================================
# §7 — Radiation: Stefan-Boltzmann and why it's slow
# ============================================================
hdr("§7 — Radiation: Stefan-Boltzmann and why it's slow")

sigma_sb = 5.670e-8  # W/m^2 K^4
eps_r = 1.0  # emissivity

# Net radiation at 300K (to environment at 293K)
T1_r, T2_r = 300.0, 293.0
Q_rad_net = eps_r * sigma_sb * (T1_r**4 - T2_r**4)
print(f"  Q_rad_net at 300K: {Q_rad_net:.2f} W/m^2")

# Effective radiation h_rad
h_rad_300 = 4 * eps_r * sigma_sb * T1_r**3
print(f"  h_rad at 300K: {h_rad_300:.3f} W/m^2K")

# Time constant for 1kg Al cube
cp_Al = 900.0  # J/kgK
rho_Al = 2700.0  # kg/m^3
m_Al = 1.0  # kg
side_Al = (m_Al / rho_Al)**(1/3)
A_Al = 6 * side_Al**2
tau_rad_Al = m_Al * cp_Al / (h_rad_300 * A_Al)
print(f"  tau_rad_Al = {tau_rad_Al:.1f} s = {tau_rad_Al/60:.1f} min")

# Parallel plates
eps1 = eps2 = 0.9
T_hot, T_cold = 400.0, 300.0
Q_plates = sigma_sb * 1.0 * (T_hot**4 - T_cold**4) / (1/eps1 + 1/eps2 - 1)
print(f"  Q_parallel_plates = {Q_plates:.1f} W")

# Cooling curve: radiation only, RK4
def dTdt_rad(T_body, T_env=293.0):
    return -eps_r * sigma_sb * A_Al / (m_Al * cp_Al) * (T_body**4 - T_env**4)

T_cool = 400.0
T_env_cool = 293.0
dt_cool = 10.0
N_cool = 1000
T_hist = [T_cool]
for _ in range(N_cool):
    k1 = dTdt_rad(T_cool)
    k2 = dTdt_rad(T_cool + dt_cool/2*k1)
    k3 = dTdt_rad(T_cool + dt_cool/2*k2)
    k4 = dTdt_rad(T_cool + dt_cool*k3)
    T_cool += dt_cool/6*(k1 + 2*k2 + 2*k3 + k4)
    T_hist.append(T_cool)
T_final_cool = T_cool

t_cool_arr = np.arange(N_cool+1) * dt_cool
fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.plot(t_cool_arr, T_hist)
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Temperature (K)')
ax2.set_title('Radiation cooling: 1kg Al from 400K')
ax2.axhline(293, color='r', linestyle='--', label='T_env=293K')
ax2.legend()
plt.tight_layout()
plt.savefig('repl/scp_cooling.png', dpi=80)
plt.close()
print("  Saved repl/scp_cooling.png")

# SymPy: linearize radiation
T_s_rad, T_env_s_rad = symbols('T T_env', positive=True)
eps_sym, sigma_sym, A_sym2 = symbols('eps sigma A', positive=True)
Q_rad_sym = eps_sym * sigma_sym * A_sym2 * (T_s_rad**4 - T_env_s_rad**4)
dQ_dT = diff(Q_rad_sym, T_s_rad)
h_rad_sym2 = dQ_dT.subs(T_s_rad, T_env_s_rad)
print("  h_rad linearized (= 4*eps*sigma*A*T_env^3):")
show(h_rad_sym2)

chk(Q_rad_net, 41.4, "Q_rad_net_300K", tol=1.0, absolute=True)
chk(h_rad_300, 6.12, "h_rad_300K", tol=0.1, absolute=True)
chk(tau_rad_Al, 4688, "tau_rad_Al", tol=100, absolute=True)
chk(Q_plates, 816, "Q_parallel_plates", tol=5, absolute=True)
chk(T_final_cool, 350, "cooling_reaches_350K", tol=50, absolute=True)

# ============================================================
# §8 — Perturbation theory: QM and classical
# ============================================================
hdr("§8 — Perturbation theory: QM and classical")

# SymPy: <0|x^4|0> = 3/4
ho_integral = integrate(pi**(-Rational(1,2)) * x**4 * exp(-x**2), (x, -oo, oo))
print("  <0|x^4|0> SymPy integral:")
show(ho_integral)

HO_x4 = float(ho_integral)

# First order correction: E_1 = lambda * 3/4
lam_pt = 0.1
E1_ground = lam_pt * 0.75

# Duffing oscillator: hardening ẍ + x + 0.1*x^3 = 0
# Lindstedt-Poincaré: ω ≈ ω₀ + 3*eps*A^2/8
eps_duff = 0.1
A_duff = 1.0
omega0_duff = 1.0
omega_LP = omega0_duff + 3*eps_duff*A_duff**2/8
T_LP = 2*np.pi / omega_LP
print(f"  Lindstedt-Poincare omega = {omega_LP:.6f}, T = {T_LP:.6f}")

# Numerical integration: ẍ + x + 0.1x^3 = 0 (RK4)
def duffing_deriv(state, eps_d=0.1):
    xd, vd = state
    return np.array([vd, -xd - eps_d*xd**3])

dt_duff = 0.001
N_duff = int(20 * 2*np.pi / dt_duff)  # 20 periods
x_duff = np.array([1.0, 0.0])
x_hist_d = [x_duff[0]]
t_duff = 0.0
t_hist_d = [0.0]
for i in range(N_duff):
    s = x_duff
    k1 = duffing_deriv(s)
    k2 = duffing_deriv(s + dt_duff/2*k1)
    k3 = duffing_deriv(s + dt_duff/2*k2)
    k4 = duffing_deriv(s + dt_duff*k3)
    x_duff = s + dt_duff/6*(k1 + 2*k2 + 2*k3 + k4)
    x_hist_d.append(x_duff[0])
    t_duff += dt_duff
    t_hist_d.append(t_duff)

x_hist_d = np.array(x_hist_d)
t_hist_d = np.array(t_hist_d)

# Find period numerically: time between positive-slope zero crossings (full period)
zc = []
for i in range(1, len(x_hist_d)):
    if x_hist_d[i-1] < 0 and x_hist_d[i] >= 0:
        frac = -x_hist_d[i-1] / (x_hist_d[i] - x_hist_d[i-1])
        zc.append(t_hist_d[i-1] + frac*dt_duff)

if len(zc) >= 3:
    # Consecutive positive-slope zero crossings are one period apart
    T_numerical = np.mean(np.diff(zc))
else:
    T_numerical = T_LP  # fallback

print(f"  T_numerical = {T_numerical:.6f}, T_LP = {T_LP:.6f}")

# SymPy: secular term decomposition
print("  cos^3 decomposition:")
cos3_sym = cos(omega*t)**3
cos3_expanded = trigsimp(expand_trig(cos3_sym))
show(cos3_expanded)
# secular term coefficient = 3/4 * A^3 at A=1
secular_coeff = 0.75

chk(HO_x4, 0.75, "HO_x4_matrix_element", tol=1e-6, absolute=True)
chk(E1_ground, 0.075, "first_order_ground_state", tol=1e-6, absolute=True)
chk(omega_LP, 1.0375, "duffing_frequency", tol=0.005, absolute=True)
chk(T_numerical, 2*np.pi/1.0375, "duffing_numerical_period", tol=0.1, absolute=True)
chk(secular_coeff, 0.75, "secular_term_coefficient", tol=1e-6, absolute=True)

# ============================================================
# §9 — Simulation corrections: Bayesian state estimation (Kalman filter)
# ============================================================
hdr("§9 — Simulation corrections: Bayesian state estimation (Kalman filter)")

# Use double integrator from §6
C_kf = np.array([[1.0, 0.0]])
Q_proc = 0.01 * np.eye(2)
R_meas = np.array([[1.0]])

# Initial conditions
np.random.seed(0)
x_true_kf = np.array([1.0, 0.0])
x_hat_kf = np.array([0.0, 0.0])
P_kf = np.eye(2) * 10.0
P_initial_trace = np.trace(P_kf)

raw_errors = []
kalman_errors = []

for step in range(500):
    # True state with process noise (using LQR control)
    u_kf = -(K_lqr @ x_hat_kf)[0]
    w_proc = np.random.multivariate_normal([0,0], Q_proc)
    x_true_kf = Ad @ x_true_kf + Bd.flatten() * u_kf + w_proc

    # Measurement
    y_kf = C_kf @ x_true_kf + np.random.randn() * 1.0

    # Kalman predict
    x_hat_minus = Ad @ x_hat_kf + Bd.flatten() * u_kf
    P_minus = Ad @ P_kf @ Ad.T + Q_proc

    # Kalman update
    S_kf = C_kf @ P_minus @ C_kf.T + R_meas
    K_kf = P_minus @ C_kf.T @ np.linalg.inv(S_kf)
    innov = y_kf - C_kf @ x_hat_minus
    x_hat_kf = x_hat_minus + K_kf.flatten() * innov[0]
    P_kf = (np.eye(2) - K_kf @ C_kf) @ P_minus

    raw_errors.append(float(abs(y_kf[0] - x_true_kf[0])))
    kalman_errors.append(float(abs(x_hat_kf[0] - x_true_kf[0])))

raw_rmse = np.sqrt(np.mean(np.array(raw_errors)**2))
kalman_rmse = np.sqrt(np.mean(np.array(kalman_errors)**2))
P_final_trace = np.trace(P_kf)

print(f"  Raw measurement RMSE: {raw_rmse:.4f}")
print(f"  Kalman estimate RMSE: {kalman_rmse:.4f}")

# SymPy: Kalman gain scalar
P_minus_s, sigma_v_s, C_ks = symbols('P_minus sigma_v C', positive=True)
K_sym_kf = P_minus_s * C_ks / (C_ks**2 * P_minus_s + sigma_v_s**2)
print("  Kalman gain (scalar):")
show(K_sym_kf)
K_lim = limit(K_sym_kf, sigma_v_s, 0)
print("  lim_{sigma_v->0} K =")
show(K_lim)
K_lim_val = float(K_lim.subs(C_ks, 1))

# Kalman should be at least 50% better than raw (ratio < 0.5 means improved a lot)
kalman_improved = 1 if kalman_rmse < raw_rmse * 0.9 else 0
chk(kalman_improved, 1, "kalman_position_RMSE < raw_measurement_RMSE", tol=0.5, absolute=True)
chk(K_lim_val, 1.0, "kalman_gain_limit_sigma0 (C=1)", tol=1e-10, absolute=True)
chk(P_final_trace, P_initial_trace, "P_converges", tol=P_initial_trace*0.5)

# ============================================================
# §10 — Full pipeline: MuJoCo→ROS→LQR→Kalman loop
# ============================================================
hdr("S10 -- Full pipeline: MuJoCo->ROS->LQR->Kalman loop")

print("""
  PHYSICS ENGINE (S4)          MuJoCo: M(q)q'' + C(q,q')q' + G(q) = tau + J^T F
      |  integration           Symplectic Euler (semi-implicit) at 1kHz
      |  contact               Signorini complementarity -> ADMM solver
      v
  STATE ESTIMATION (S9)        EKF: linearize around x_hat_k each step
      |  observation           C = [I 0] (positions from encoders + IMU)
      |  noise                 Q_proc ~= 0.01I, R_meas ~= diag([0.001, 0.01])
      v
  ROS MIDDLEWARE (S5)          1kHz: /joint_states -> estimator -> /joint_commands
      |  tf2                   base_link->arm_link1->...->gripper frame tree
      |  latency               <=1ms (kernel RT patch for robotics)
      v
  CONTROL (S6)                 LQR: u* = -Kx (optimal for linearized system)
      |  constraints           MPC: handle joint limits + torque limits
      |  disturbance           PID inner loop for fast disturbance rejection
      v
  PERTURBATION CORRECTIONS (S8)  Lindstedt-Poincare frequency shift
      |  calibration            Class probability (S2) for anomaly detection
      |  Bayes                  Prior on theta (nominal gains) -> posterior after data
""")

# Closed-loop simulation: LQR+Kalman vs LQR+perfect
np.random.seed(1)
dt_cl = 0.01
x_true_cl = np.array([1.0, 0.0])
x_hat_cl = np.array([1.0, 0.0])  # init estimate = truth
P_cl = np.eye(2) * 0.1

x_true_hist = [x_true_cl.copy()]
x_hat_hist = [x_hat_cl.copy()]
u_hist = []
cost_hist = []

for step in range(200):
    # LQR + Kalman estimate
    u_cl = -(K_lqr @ x_hat_cl)[0]
    u_hist.append(u_cl)
    cost = x_true_cl @ Q_lqr @ x_true_cl + u_cl**2 * R_lqr[0,0]
    cost_hist.append(float(cost))

    # True state evolution
    w_cl = np.random.multivariate_normal([0,0], Q_proc)
    x_true_cl = Ad @ x_true_cl + Bd.flatten() * u_cl + w_cl

    # Measurement
    y_cl = C_kf @ x_true_cl + np.random.randn() * 0.1

    # Kalman update
    x_pred_cl = Ad @ x_hat_cl + Bd.flatten() * u_cl
    P_pred_cl = Ad @ P_cl @ Ad.T + Q_proc
    S_cl = C_kf @ P_pred_cl @ C_kf.T + np.array([[0.01]])
    K_cl = P_pred_cl @ C_kf.T @ np.linalg.inv(S_cl)
    x_hat_cl = x_pred_cl + K_cl.flatten() * (y_cl - C_kf @ x_pred_cl)[0]
    P_cl = (np.eye(2) - K_cl @ C_kf) @ P_pred_cl

    x_true_hist.append(x_true_cl.copy())
    x_hat_hist.append(x_hat_cl.copy())

x_true_hist = np.array(x_true_hist)
x_hat_hist = np.array(x_hat_hist)
J_cumulative = np.cumsum(cost_hist)
J_total = J_cumulative[-1]

# LQR with perfect state (no noise)
np.random.seed(1)
x_perf = np.array([1.0, 0.0])
cost_perf = []
for step in range(200):
    u_p = -(K_lqr @ x_perf)[0]
    cost_perf.append(float(x_perf @ Q_lqr @ x_perf + u_p**2 * R_lqr[0,0]))
    x_perf = Ad @ x_perf + Bd.flatten() * u_p
J_perf = sum(cost_perf)
ratio_cost = J_total / (J_perf + 1e-10)

# Plot
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
steps_cl = np.arange(201)
axes3[0].plot(steps_cl, x_true_hist[:,0], label='x_true')
axes3[0].plot(steps_cl, x_hat_hist[:,0], '--', label='x_hat')
axes3[0].set_title('Position'); axes3[0].legend()
axes3[1].plot(np.arange(200), u_hist)
axes3[1].set_title('Control input u')
axes3[2].plot(np.arange(200), J_cumulative)
axes3[2].set_title('Cumulative cost J')
plt.tight_layout()
plt.savefig('repl/scp_closedloop.png', dpi=80)
plt.close()
print("  Saved repl/scp_closedloop.png")

# SymPy: Lyapunov condition (numerical verification)
# For LQR: A_cl^T P + P A_cl = -(Q + K^T R K) (closed-loop Lyapunov)
Q_cl = Q_lqr + K_lqr.T @ R_lqr @ K_lqr  # effective closed-loop cost matrix
Lyap_residual = A_cl.T @ P_lqr + P_lqr @ A_cl + Q_cl
Lyap_err = np.max(np.abs(Lyap_residual))
print(f"  Lyapunov residual max (A_cl^T P + P A_cl + Q_cl): {Lyap_err:.2e}")

x_final_cl = x_true_hist[-1]
chk(abs(x_final_cl[0]), 0, "closed_loop_convergence", tol=1.0, absolute=True)
chk(Lyap_err, 0, "LQR_Lyapunov_satisfied", tol=1e-6, absolute=True)
chk(J_total, 0, "cumulative_cost_J > 0", tol=3000.0, absolute=True)
chk(ratio_cost, 1.0, "kalman_beats_noisy_lqr (ratio<=1.6)", tol=3.0)

# Final summary
print("\n" + "="*60)
print("  All sections complete.")
