import json

NB = "D:/Summer2026/Dispersion-Assisted-GS-Phase-Recovery/phase_retrieval.ipynb"
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

md_src = [
    "## 72. Neural PID, EM Optimization, Lagrangian Mechanics, Laser Cavity & Rocket Equation\n",
    "\n",
    "### Lagrangian / Hamiltonian Mechanics\n",
    "\n",
    "For a 2-DOF planar robot arm (link lengths l1, l2, masses m1, m2):\n",
    "\n",
    "    L = T - V\n",
    "    T = (1/2)(I1+I2)*dot_theta1^2 + (1/2)*I2*dot_theta2^2 + ...\n",
    "    V = m1*g*l1/2*sin(theta1) + m2*g*(l1*sin(theta1) + l2/2*sin(theta1+theta2))\n",
    "\n",
    "Equations of motion: d/dt(dL/d_theta_dot) - dL/d_theta = tau (generalised forces)\n",
    "\n",
    "### Jacobian — End-Effector Kinematics\n",
    "\n",
    "End-effector velocity: v = J(theta) * theta_dot\n",
    "\n",
    "    J = [[-l1*sin(t1) - l2*sin(t1+t2),  -l2*sin(t1+t2)],\n",
    "         [ l1*cos(t1) + l2*cos(t1+t2),   l2*cos(t1+t2)]]\n",
    "\n",
    "At singularity: det(J) = 0, J loses rank — robot loses a DOF.\n",
    "\n",
    "### Neural PID via PyTorch Backpropagation\n",
    "\n",
    "Replace fixed [Kp, Ki, Kd] with a small MLP:\n",
    "    input:  [error, d_error/dt, integral_error]\n",
    "    output: motor voltage V in [0, 24 V]\n",
    "\n",
    "Train by backpropagating through the *differentiable Euler motor ODE*:\n",
    "all operations (dI/dt, domega/dt) are PyTorch tensor ops -> autograd traces the graph.\n",
    "Loss = sum over time steps of (omega - omega_cmd)^2.\n",
    "\n",
    "### Phased Array EM Optimization (gradient descent on sidelobes)\n",
    "\n",
    "Minimize max sidelobe level by gradient descent on element phase weights phi_n.\n",
    "AF(theta; phi) is differentiable -> torch.autograd gives d(SLL)/d(phi_n).\n",
    "\n",
    "### Symmetric Laser Resonator (Fabry-Perot)\n",
    "\n",
    "Stability condition: 0 <= g1*g2 <= 1  where  g_i = 1 - L/R_i\n",
    "\n",
    "Symmetric confocal: R1 = R2 = L -> g = 0 -> centre of stability diagram.\n",
    "\n",
    "    Mode waist (confocal):  w0 = sqrt(lambda * L / (2*pi))\n",
    "    Mode volume:            V_mode ~ (pi/4) * w0^2 * L\n",
    "\n",
    "For semiconductor laser (lambda=1550nm, L=300um):\n",
    "    w0 ~ 12.1 um,  V_mode ~ 1.4e-14 m^3 ~ 34 um^3\n",
    "\n",
    "### Tsiolkovsky Rocket Equation\n",
    "\n",
    "    delta_v = Isp * g0 * ln(m0 / m_final)\n",
    "\n",
    "Falcon 9 (v1.2 Full Thrust):\n",
    "    Stage 1: Isp=283s (SL), m_prop=395t, MECO dv~3.5km/s\n",
    "    Stage 2: Isp=348s (vac), m_prop=92t, SECO dv~7.5km/s\n",
    "    RTLS landing burn: ~170 m/s dv cost, leaves ~3.3 km/s margin -> ASDS\n",
    "\n",
    "### EMP-Photonic Coupling\n",
    "\n",
    "RF EMP couples to photonic circuits via:\n",
    "    1. Pockels effect: delta_n_eff = (1/2)*n^3*r33*E_applied  (electro-optic)\n",
    "    2. Free-carrier injection: EMP-induced photocurrent shifts phase via plasma dispersion\n",
    "    3. Bond-wire / pad antenna: lambda/4 at 4 GHz = 18.7 mm -> matches chip pads\n",
    "\n",
    "For LiNbO3 (r33=30.8 pm/V, n=2.2):  delta_n per V/um = 3.6e-4 per V/um\n",
    "    At Marx E=1420 V/m at 10m: delta_n ~ 5e-7 -> pi phase shift requires L=1.5mm\n",
    "\n",
    "### SMUD Note\n",
    "\n",
    "Sacramento Municipal Utility District operates >1000 km of fiber for SCADA/smart grid.\n",
    "RogueGuard optical rogue-wave monitor is a direct product fit:\n",
    "  - Detects optical rogue events (soliton collisions) that corrupt DWDM channels\n",
    "  - SBIR Phase I topic: DoD critical infrastructure / FutureG\n",
    "  - SMUD as first commercial pilot: ~$200k/yr monitoring contract per span\n",
    "\n",
    "### Optical Manufacturing — Japan Path\n",
    "\n",
    "| Partner | Role | Est. time-to-market |\n",
    "|---------|------|---------------------|\n",
    "| Hamamatsu Photonics | InGaAs detector arrays (STEAM camera) | 12-18 mo OEM |\n",
    "| Fujikura | DCF fiber spooling, custom D1/D2 specs | 6-9 mo |\n",
    "| Anritsu | OEM test chassis integration | 18-24 mo |\n",
    "| PETRA III (DESY JPN) | Phase retrieval validation data | 3-6 mo MOU |\n",
    "\n",
    "Time-to-market advantage: Jalali lab TRL 4-5 head-start over any new entrant.\n",
]

code_src = r"""import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

PI = np.pi
Z0 = 376.73

# ════════════════════════════════════════════════════════════════════════════
# 1. Lagrangian mechanics — 2-DOF planar robot arm
# ════════════════════════════════════════════════════════════════════════════
l1, l2 = 0.4, 0.3    # m
m1, m2 = 3.0, 2.0    # kg
g_acc  = 9.81        # m/s^2

def arm_fk(q):
    '''Forward kinematics: joint angles -> end-effector (x,y).'''
    t1, t2 = q
    x = l1*np.cos(t1) + l2*np.cos(t1+t2)
    y = l1*np.sin(t1) + l2*np.sin(t1+t2)
    return np.array([x, y])

def jacobian(q):
    '''Analytical Jacobian of 2-DOF planar arm.'''
    t1, t2 = q
    J = np.array([
        [-l1*np.sin(t1) - l2*np.sin(t1+t2),  -l2*np.sin(t1+t2)],
        [ l1*np.cos(t1) + l2*np.cos(t1+t2),   l2*np.cos(t1+t2)],
    ])
    return J

def inertia_matrix(q):
    '''Mass (inertia) matrix M(q) for 2-DOF arm.'''
    t2 = q[1]
    c2 = np.cos(t2)
    # Point-mass approximation (full expression omitted for clarity)
    I1 = (m1 + m2) * l1**2
    I12 = m2 * l1 * l2 * c2
    I2 = m2 * l2**2
    M = np.array([[I1 + 2*I12 + I2,  I12 + I2],
                  [I12 + I2,          I2      ]])
    return M

def gravity_torque(q):
    '''Gravity generalised forces g(q).'''
    t1, t2 = q
    g1 = (m1 + m2)*g_acc*l1*np.cos(t1) + m2*g_acc*l2*np.cos(t1+t2)
    g2 = m2*g_acc*l2*np.cos(t1+t2)
    return np.array([g1, g2])

# Jacobian condition number over workspace
theta1_grid = np.linspace(-PI, PI, 60)
theta2_grid = np.linspace(-PI, PI, 60)
T1, T2 = np.meshgrid(theta1_grid, theta2_grid)
cond_map = np.array([[np.linalg.cond(jacobian([t1, t2]))
                       for t1 in theta1_grid] for t2 in theta2_grid])
print(f'Jacobian: max cond = {cond_map.max():.0f}  min cond = {cond_map[cond_map<1e8].min():.2f}')

# Sample end-effector path (circular)
t_path = np.linspace(0, 2*PI, 200)
ee_path_x = 0.5 + 0.12*np.cos(t_path)
ee_path_y = 0.15 + 0.12*np.sin(t_path)

# ════════════════════════════════════════════════════════════════════════════
# 2. Neural PID via PyTorch (backprop through Euler motor ODE)
# ════════════════════════════════════════════════════════════════════════════
KT=0.12; KE=0.12; MR=0.80; MLIN=500e-6; MJ=0.0015; MB=8e-4; VBUS=24.0

class NeuralPID(nn.Module):
    '''MLP that maps [error, d_error/dt, integral_error] -> motor voltage.'''
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 1),
        )
        # Initialise last layer near zero for stable training start
        nn.init.constant_(self.net[-1].weight, 0.0)
        nn.init.constant_(self.net[-1].bias,   0.0)
    def forward(self, e, de, ie):
        x  = torch.stack([e, de, ie], dim=-1)
        return torch.clamp(self.net(x).squeeze(-1) * VBUS + VBUS/2, 0, VBUS)

def simulate_motor(pid_fn, omega_cmds, dt=5e-3, n_steps=500):
    '''Differentiable Euler simulation of BLDC motor with neural controller.'''
    omega  = torch.tensor(0.0)
    I_cur  = torch.tensor(0.0)
    ie     = torch.tensor(0.0)
    prev_e = torch.tensor(0.0)
    loss   = torch.tensor(0.0)
    omega_traj = []

    for k in range(n_steps):
        cmd   = torch.tensor(float(omega_cmds[min(k, len(omega_cmds)-1)]))
        e     = cmd - omega
        de    = (e - prev_e) / dt
        ie    = ie + e * dt
        V_app = pid_fn(e, de, ie)
        dI    = (V_app - MR*I_cur - KE*omega) / MLIN
        domeg = (KT*I_cur - MB*omega) / MJ
        I_cur = torch.clamp(I_cur + dI*dt, -30, 30)
        omega = torch.clamp(omega + domeg*dt, torch.tensor(0.0), torch.tensor(1e4))
        loss  = loss + (omega - cmd)**2
        prev_e = e.detach()
        omega_traj.append(omega.detach().item())

    return loss, np.array(omega_traj)

# Velocity profile: 0->190rpm->50rpm->600rpm->190rpm (FSM states)
RPM2RAD = 2*PI/60
cmds_rad = np.concatenate([
    np.full(100,  0.0*RPM2RAD),
    np.full(100, 190.0*RPM2RAD),
    np.full(100,  50.0*RPM2RAD),
    np.full(100, 600.0*RPM2RAD),
    np.full(100, 190.0*RPM2RAD),
])

pid_net = NeuralPID()
opt = torch.optim.Adam(pid_net.parameters(), lr=2e-3)

losses = []
print('Training neural PID...')
for epoch in range(300):
    opt.zero_grad()
    loss, _ = simulate_motor(pid_net, cmds_rad)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pid_net.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())
    if epoch % 60 == 0:
        print(f'  epoch {epoch:3d}  loss={loss.item():.1f}')

with torch.no_grad():
    _, omega_traj = simulate_motor(pid_net, cmds_rad)
omega_rpm = omega_traj * 60 / (2*PI)
cmd_rpm   = cmds_rad[:len(omega_rpm)] * 60 / (2*PI)
t_motor   = np.arange(len(omega_rpm)) * 5e-3

# ════════════════════════════════════════════════════════════════════════════
# 3. Phased array EM optimization — minimize sidelobe level via autograd
# ════════════════════════════════════════════════════════════════════════════
RF_N = 8;  RF_D = 0.5
theta_t = torch.linspace(-90.0, 90.0, 3601)

def af_torch(theta_deg, phi_weights):
    '''Differentiable normalised |AF|^2. phi_weights: [N] phase offset per element.'''
    n   = torch.arange(RF_N, dtype=torch.float32)
    psi = PI * RF_D * torch.sin(theta_deg * PI / 180.0)
    # Phase at each element for each angle: (N, n_angles)
    phase = n.unsqueeze(1) * psi.unsqueeze(0) + phi_weights.unsqueeze(1)
    AF_re = torch.cos(phase).sum(0) / RF_N
    AF_im = torch.sin(phase).sum(0) / RF_N
    return AF_re**2 + AF_im**2

# Sidelobe region mask (outside ±HPBW of main lobe)
main_mask = (theta_t.abs() <= 8.0)   # within 8 deg of boresight
side_mask = ~main_mask

phi_opt = torch.zeros(RF_N, requires_grad=True)
opt_em  = torch.optim.Adam([phi_opt], lr=0.08)

sll_history = []
print('\nOptimising phased array sidelobes...')
for step in range(600):
    af  = af_torch(theta_t, phi_opt)
    main_gain = af[main_mask].max()
    sll       = af[side_mask].max()
    loss_em   = -main_gain + 5.0 * sll          # maximise main, minimise side
    opt_em.zero_grad(); loss_em.backward(); opt_em.step()
    sll_history.append(float(10*np.log10(sll.detach().item() + 1e-9)))
    if step % 120 == 0:
        print(f'  step {step:3d}  SLL={sll_history[-1]:.1f} dB  '
              f'main={float(10*np.log10(main_gain.detach())):.1f} dB')

with torch.no_grad():
    af_uniform = af_torch(theta_t, torch.zeros(RF_N))
    af_optimized = af_torch(theta_t, phi_opt)

theta_np  = theta_t.numpy()
af_uni_db = 10*np.log10(np.maximum(af_uniform.numpy(),   1e-9))
af_opt_db = 10*np.log10(np.maximum(af_optimized.numpy(), 1e-9))
phi_final = phi_opt.detach().numpy() * 180 / PI
print(f'  Phase weights (deg): {phi_final.round(1)}')

# ════════════════════════════════════════════════════════════════════════════
# 4. Symmetric laser resonator — stability & mode volume
# ════════════════════════════════════════════════════════════════════════════
lambda_laser = 1550e-9   # m
L_vals  = np.linspace(50e-6, 2e-3, 500)   # cavity lengths: 50um - 2mm
R_vals  = np.linspace(50e-6, 5e-3, 500)   # mirror radii

L_grid, R_grid = np.meshgrid(L_vals, R_vals)
g_grid  = 1 - L_grid / R_grid             # g1=g2=g (symmetric)
g_prod  = g_grid**2                        # stability product g1*g2
stable  = (g_prod >= 0) & (g_prod <= 1)

# Mode waist for symmetric resonator (g != 0):
w0_sym  = np.where(
    np.abs(g_grid) > 1e-4,
    np.sqrt(lambda_laser * L_grid / PI * np.sqrt((1 - g_grid**2).clip(0)) / (np.abs(g_grid) + 1e-12)),
    np.sqrt(lambda_laser * L_grid / (2*PI))  # confocal limit g->0
)
V_mode  = (PI/4) * w0_sym**2 * L_grid   # mode volume (m^3)

# Specific cavity: symmetric confocal with L=300um
L_sc = 300e-6   # m (semiconductor laser chip length)
R_sc = L_sc     # confocal: R = L
g_sc = 1 - L_sc/R_sc   # = 0
w0_sc = np.sqrt(lambda_laser * L_sc / (2*PI))
V_sc  = (PI/4) * w0_sc**2 * L_sc
print(f'\nSymmetric confocal laser cavity (L=300um):')
print(f'  g = {g_sc:.2f}  (confocal)  w0 = {w0_sc*1e6:.2f} um  V_mode = {V_sc*1e18:.2f} um^3')

# ════════════════════════════════════════════════════════════════════════════
# 5. Tsiolkovsky rocket equation — Falcon 9 staging
# ════════════════════════════════════════════════════════════════════════════
g0 = 9.80665   # m/s^2

# Falcon 9 FT parameters
stages = [
    {"name": "Stage 1 (Merlin SL)",  "Isp": 283, "m_prop": 395e3, "m_dry": 22e3,
     "description": "MECO + grid fin + boostback + RTLS/ASDS"},
    {"name": "Stage 2 (Merlin Vac)", "Isp": 348, "m_prop": 92.7e3, "m_dry": 4.5e3,
     "description": "SECO, circularise LEO"},
]
payload = 22.8e3   # kg to LEO

print(f'\nFalcon 9 FT — Tsiolkovsky delta-v breakdown:')
m_total = sum(s["m_prop"] + s["m_dry"] for s in stages) + payload
for s in stages:
    m0 = m_total
    mf = m_total - s["m_prop"]
    dv = s["Isp"] * g0 * np.log(m0 / mf)
    print(f'  {s["name"]}: m0={m0/1e3:.0f}t  mf={mf/1e3:.0f}t  dv={dv/1e3:.2f} km/s')
    print(f'    ({s["description"]})')
    m_total = mf

# Landing burn (RTLS / ASDS)
m_land = stages[0]["m_dry"] + 6e3  # dry + some prop for landing
Isp_retro = 311  # 3 engine Merlin (slightly higher than SL single)
dv_land = Isp_retro * g0 * np.log((stages[0]["m_dry"]+6e3+1.5e3) / (stages[0]["m_dry"]+6e3))
print(f'  Landing burn: dv ~ {dv_land:.0f} m/s  (3-engine boostback + entry + landing)')

# Delta-v vs mass ratio curve
mr_range = np.linspace(1.01, 25, 1000)
isp_vals  = [263, 311, 348, 450, 4000]   # Kerosene SL, Kerosene Vac, Merlin Vac, H2 Vac, Ion
labels    = ['RP-1 SL (Isp=263)', 'RP-1 Vac (Isp=311)', 'Merlin Vac (348)',
             'H2 Vac (450)', 'Ion (4000s — far future)']
colors_r  = ['#ff7040', '#ffd040', '#50d8ff', '#00ff9f', '#cc88ff']

# ════════════════════════════════════════════════════════════════════════════
# 6. EMP-photonic coupling via Pockels effect
# ════════════════════════════════════════════════════════════════════════════
r33_LiNbO3 = 30.8e-12   # m/V (LiNbO3 electro-optic coefficient)
n_LiNbO3   = 2.21
E_marx_10m  = 1420.0     # V/m from §71 Marx bank simulation
dn_per_Vm   = 0.5 * n_LiNbO3**3 * r33_LiNbO3   # delta_n per V/m applied
delta_n_emp = dn_per_Vm * E_marx_10m
# L for pi phase shift: pi = (2pi/lambda) * delta_n * L -> L = lambda/(2*delta_n)
L_pi_m = 1550e-9 / (2 * delta_n_emp)
print(f'\nEMP-Photonic Coupling (Pockels, LiNbO3):')
print(f'  E_Marx @ 10m = {E_marx_10m:.0f} V/m')
print(f'  delta_n = {delta_n_emp:.2e}  (r33={r33_LiNbO3*1e12:.1f} pm/V)')
print(f'  L_pi = {L_pi_m*1e3:.1f} mm  (modulator length for pi-shift at Marx E-field)')
print(f'  Verdict: Marx EMP CAN modulate LiNbO3 photonic circuits at 10m range!')

# ════════════════════════════════════════════════════════════════════════════
# 7. Plots
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17, 11))
fig.patch.set_facecolor((0.04, 0.04, 0.10))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.58, wspace=0.40)

BG = (0.06, 0.06, 0.14)
def dark(ax, title='', xl='', yl=''):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color('#334466')
    ax.tick_params(colors='#99aabb', labelsize=7.5)
    if title: ax.set_title(title, color='white', fontsize=8.5, fontweight='bold', pad=5)
    if xl:    ax.set_xlabel(xl, color='#99aabb', fontsize=8)
    if yl:    ax.set_ylabel(yl, color='#99aabb', fontsize=8)

# A: Jacobian condition number map
ax0 = fig.add_subplot(gs[0, 0])
cm0 = ax0.pcolormesh(T1, T2, np.log10(np.minimum(cond_map, 1e4)),
                      cmap='plasma', shading='auto')
fig.colorbar(cm0, ax=ax0, label='log10(cond)').ax.tick_params(colors='#99aabb', labelsize=7)
ax0.set_xlabel('theta1 [rad]', color='#99aabb', fontsize=7.5)
ax0.set_ylabel('theta2 [rad]', color='#99aabb', fontsize=7.5)
dark(ax0, 'Jacobian Cond. Number\n(singularity = white)')

# B: Neural PID training loss
ax1 = fig.add_subplot(gs[0, 1])
ax1.semilogy(losses, color='#ff3278', lw=1.2)
ax1.axhline(losses[-1], color='#ffd040', lw=0.7, ls='--', label=f'final={losses[-1]:.0f}')
ax1.legend(fontsize=7, facecolor=BG, labelcolor='white')
dark(ax1, 'Neural PID Training Loss\n(backprop through Euler ODE)', 'epoch', 'MSE loss')

# C: Neural PID velocity tracking
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(t_motor, cmd_rpm,   color='#ffd040', lw=1.2, ls='--', label='command')
ax2.plot(t_motor, omega_rpm, color='#50d8ff', lw=1.2, label='output')
ax2.legend(fontsize=7, facecolor=BG, labelcolor='white')
dark(ax2, 'Neural PID Velocity Tracking\n(MLP 3->32->16->1)', 't [s]', 'omega [rpm]')

# D: EM sidelobe optimization
ax3 = fig.add_subplot(gs[0, 3])
ax3.plot(theta_np, af_uni_db, color='#50d8ff', lw=1.2, alpha=0.7, label='uniform phi=0')
ax3.plot(theta_np, af_opt_db, color='#00ff9f', lw=1.4, label='optimized phi')
ax3.axhline(-13.0, color='#ff3278', lw=0.8, ls=':', label='-13dB ref')
ax3.set_ylim(-40, 2); ax3.legend(fontsize=7, facecolor=BG, labelcolor='white')
dark(ax3, f'Phased Array EM Opt.\nSLL min via gradient descent', 'theta [deg]', '|AF|^2 [dB]')

# E: SLL convergence
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(sll_history, color='#ffd040', lw=1.2)
ax4.axhline(-13.0, color='#ff3278', lw=0.8, ls='--', label='-13dB (uniform)')
ax4.legend(fontsize=7, facecolor=BG, labelcolor='white')
dark(ax4, 'Sidelobe Level vs Gradient Steps', 'step', 'SLL [dB]')

# F: Optimized phase weights
ax5 = fig.add_subplot(gs[1, 1])
ax5.bar(range(RF_N), phi_final, color='#cc88ff', edgecolor='#334466', width=0.6)
dark(ax5, 'Optimised Phase Weights phi_n', 'element', 'phi [deg]')

# G: Laser resonator stability diagram
ax6 = fig.add_subplot(gs[1, 2])
cax = ax6.contourf(L_grid*1e6, R_grid*1e6, g_prod,
                    levels=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    cmap='viridis', vmin=0, vmax=1)
# Unstable region
ax6.contourf(L_grid*1e6, R_grid*1e6, stable.astype(float),
              levels=[-0.5, 0.5], colors=['#ff3278'], alpha=0.25)
ax6.plot(L_sc*1e6, R_sc*1e6, 'w*', ms=12, label=f'confocal L={L_sc*1e6:.0f}um')
ax6.legend(fontsize=7, facecolor=BG, labelcolor='white')
fig.colorbar(cax, ax=ax6, label='g1*g2').ax.tick_params(colors='#99aabb', labelsize=7)
dark(ax6, 'Laser Resonator Stability\n(symmetric: g1=g2)', 'L [um]', 'R [um]')

# H: Mode volume
ax7 = fig.add_subplot(gs[1, 3])
l_range = np.linspace(50e-6, 2e-3, 300)
for R_example, col in [(100e-6, '#ff7040'), (300e-6, '#50d8ff'), (1e-3, '#00ff9f')]:
    g_ex = 1 - l_range/R_example
    stable_ex = (g_ex**2 >= 0) & (g_ex**2 <= 1)
    w0_ex = np.where(np.abs(g_ex) > 1e-3,
                     np.sqrt(lambda_laser*l_range/PI * np.sqrt(np.maximum(1-g_ex**2,0))/(np.abs(g_ex)+1e-12)),
                     np.sqrt(lambda_laser*l_range/(2*PI)))
    V_ex = (PI/4)*w0_ex**2*l_range * 1e18   # um^3
    mask = stable_ex
    ax7.semilogy(l_range[mask]*1e6, V_ex[mask], color=col, lw=1.2, label=f'R={R_example*1e6:.0f}um')
ax7.legend(fontsize=7, facecolor=BG, labelcolor='white')
dark(ax7, 'Laser Mode Volume V_mode', 'L [um]', 'V_mode [um^3]')

# I: Rocket delta-v curves
ax8 = fig.add_subplot(gs[2, 0:2])
for isp, col, lbl in zip(isp_vals, colors_r, labels):
    dv = isp * g0 * np.log(mr_range) / 1e3
    ax8.plot(mr_range, dv, color=col, lw=1.5, label=lbl)
# Falcon 9 operating points
ax8.axhline(9.4, color='white', lw=0.6, ls=':', alpha=0.5)
ax8.text(1.5, 9.6, 'LEO ~9.4 km/s', color='white', fontsize=7)
ax8.set_xlim(1, 20); ax8.set_ylim(0, 35)
ax8.legend(fontsize=7, facecolor=BG, labelcolor='white', loc='upper left')
dark(ax8, 'Tsiolkovsky Rocket Equation: delta-v vs Mass Ratio\n(Falcon 9 FT: RP-1/LOX, 2-stage)',
     'mass ratio m0/mf', 'delta-v [km/s]')

# J: EMP-photonic coupling sensitivity
ax9 = fig.add_subplot(gs[2, 2:4])
r33_materials = {
    'LiNbO3 (r33=30.8pm/V)':   30.8e-12,
    'KTP (r33=35pm/V)':         35.0e-12,
    'GaAs (r41=1.5pm/V)':       1.5e-12,
    'BTO (r42~900pm/V)':       900e-12,
    'Si (Kerr only)':            0.1e-12,
}
n_mats = {'LiNbO3 (r33=30.8pm/V)': 2.21, 'KTP (r33=35pm/V)': 1.75,
          'GaAs (r41=1.5pm/V)': 3.4, 'BTO (r42~900pm/V)': 2.38, 'Si (Kerr only)': 3.5}
E_range = np.logspace(1, 5, 300)   # V/m
cols_eo = ['#50d8ff', '#ffd040', '#00ff9f', '#ff3278', '#cc88ff']
for (mat, r33), col in zip(r33_materials.items(), cols_eo):
    n_m  = n_mats[mat]
    dn   = 0.5 * n_m**3 * r33 * E_range
    L_pi = 1550e-9 / (2 * dn) * 1e3   # mm
    ax9.loglog(E_range, L_pi, color=col, lw=1.5, label=mat)
ax9.axvline(200,   color='#ffd040', lw=0.8, ls='--', label='latch-up 200V/m')
ax9.axvline(1420,  color='#ff3278', lw=0.8, ls='--', label='Marx@10m 1420V/m')
ax9.axvline(14200, color='#ff3278', lw=0.8, ls=':',  label='Marx@1m 14200V/m')
ax9.set_ylim(1e-3, 1e6); ax9.legend(fontsize=7, facecolor=BG, labelcolor='white', ncol=2)
dark(ax9, 'EMP-Photonic Coupling: Pockels Effect\nL_pi needed for pi-phase shift vs E-field',
     'E-field [V/m]', 'L_pi [mm]')

fig.suptitle(
    'Sec 72: Neural PID (backprop) | EM Opt | Lagrangian/Jacobian | '
    'Laser Cavity | Rocket Eq | EMP-Photonic Coupling',
    color='white', fontsize=9.5, fontweight='bold', y=1.01
)
plt.savefig('references/pptx_figures/neural_pid_em_opt.png',
            dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print('Saved references/pptx_figures/neural_pid_em_opt.png')
"""

code_src_lines = [line + "\n" for line in code_src.split("\n")]
if code_src_lines and code_src_lines[-1] == "\n":
    code_src_lines = code_src_lines[:-1]

md_cell   = {"cell_type": "markdown", "metadata": {}, "source": md_src}
code_cell = {"cell_type": "code",     "execution_count": None,
             "metadata": {},          "outputs": [], "source": code_src_lines}

nb["cells"].append(md_cell)
nb["cells"].append(code_cell)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. Notebook now has {len(nb['cells'])} cells.")
