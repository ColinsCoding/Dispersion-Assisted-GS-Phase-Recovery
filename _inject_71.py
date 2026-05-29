import json, pathlib

NB = "D:/Summer2026/Dispersion-Assisted-GS-Phase-Recovery/phase_retrieval.ipynb"
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── Markdown cell ──────────────────────────────────────────────────────────
md_src = [
    "## 71. Electronics vs Photonics — Combat Robot: Phased Array, Marx EMP, BLDC PID\n",
    "\n",
    "C firmware: `firmware/beamformer.c`\n",
    "```\n",
    "gcc -O2 -std=c11 firmware/beamformer.c -o beamformer -lm && ./beamformer\n",
    "```\n",
    "\n",
    "### Beamforming Battle Card\n",
    "\n",
    "| Property | RF Array | Optical Phased Array (OPA) |\n",
    "|----------|----------|----------------------------|\n",
    "| Frequency / λ | 2.4 GHz / 12.5 cm | 193 THz / 1550 nm |\n",
    "| Elements N | 8 | 512 |\n",
    "| Element pitch d | λ/2 = 62.5 mm | 2 μm = 1.29λ |\n",
    "| HPBW (steer=0) | **12.8°** | **0.077°** (166× narrower) |\n",
    "| Grating lobes | None (d=λ/2) | ±50.8° (d > λ) |\n",
    "| Phase shifter | VCO / DAC, ~10 ns | Thermal tuner, ~10 μs |\n",
    "| EM coupling | Yes — induces currents | No — photons |\n",
    "| EMP weapon | **RF Marx bank** | N/A |\n",
    "\n",
    "### Marx Bank Supercapacitor EMP\n",
    "\n",
    "8 stages × (1 mF @ 1 kV):  V_Marx = 8 kV,  E_stored = 4 kJ\n",
    "\n",
    "RLC discharge:  L = 10 μH,  R_total = 1.4 Ω,  ζ = 2.47 (overdamped)\n",
    "\n",
    "| Metric | Value |\n",
    "|--------|-------|\n",
    "| I_peak | 5189 A |\n",
    "| P_peak | 26.9 MW |\n",
    "| E-field @ 10 m | 1420 V/m (latch-up zone) |\n",
    "| E-field @ 1 m | 14 200 V/m (hard destroy) |\n",
    "| Latch-up dwell time | ~300 μs |\n",
    "\n",
    "**Semiconductor thresholds (empirical):**\n",
    "- Latch-up:   E > 200 V/m\n",
    "- Hard burn:  E > 2 kV/m\n",
    "- Junction destruction: dV/dt > 10¹⁰ V/s (Marx di/dt satifies this at 1 m)\n",
    "\n",
    "### BLDC Motor PID\n",
    "\n",
    "Motor: Kt=0.12 N·m/A, Ke=0.12 V·s/rad, R=0.8 Ω, J=1.5×10⁻³ kg·m², 24 V bus\n",
    "\n",
    "Velocity PID: Kp=8, Ki=2, Kd=0.05  (inner loop, output=voltage)\n",
    "\n",
    "FSM states:  PATROL(190 rpm) → ACQUIRE(slew turret 45°) → CHARGE(caps 1.2 s) → FIRE → EVADE(600 rpm)\n",
]

# ── Code cell ──────────────────────────────────────────────────────────────
code_src = r"""import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.integrate import solve_ivp

PI = np.pi
Z0 = 376.73   # Ohm, free-space impedance

# ── 1. Phased array beam pattern ──────────────────────────────────────────
def af_norm(theta_deg, theta0_deg, d_lam, N):
    '''Normalised |AF|^2 = [sin(N*psi) / (N*sin(psi))]^2'''
    psi = PI * d_lam * (np.sin(np.deg2rad(theta_deg))
                       - np.sin(np.deg2rad(theta0_deg)))
    eps = 1e-9
    psi = np.where(np.abs(psi) < eps, eps, psi)
    return (np.sin(N * psi) / (N * np.sin(psi)))**2

theta = np.linspace(-90, 90, 7200)          # 0.025 deg resolution

# RF: 2.4 GHz, N=8, d = lambda/2
RF_N = 8;  RF_D = 0.5
# OPA: 1550 nm, N=512, d = 2 um -> d/lambda = 1.290
OPA_N = 512; OPA_D = 2e-6 / 1550e-9

AF_RF_0   = af_norm(theta,  0, RF_D,  RF_N)
AF_RF_30  = af_norm(theta, 30, RF_D,  RF_N)
AF_RF_60  = af_norm(theta, 60, RF_D,  RF_N)
AF_OPA_0  = af_norm(theta,  0, OPA_D, OPA_N)

def hpbw(d_lam, N, t0=0):
    '''Full HPBW (deg) by numerical search in 0.01 deg steps.'''
    th  = np.linspace(t0, t0 + 89.9, 500000)
    a   = af_norm(th, t0, d_lam, N)
    idx = np.where(a < 0.5)[0]
    return 2 * abs(th[idx[0]] - t0) if len(idx) else 180.0

hpbw_rf  = hpbw(RF_D, RF_N)
hpbw_opa = hpbw(OPA_D, OPA_N)
print(f'RF  HPBW = {hpbw_rf:.2f} deg')
print(f'OPA HPBW = {hpbw_opa:.5f} deg  ({hpbw_rf/hpbw_opa:.0f}x narrower)')

# ── 2. Marx bank RLC discharge (Euler ODE, dt=0.5 us) ────────────────────
MARX_N = 8; MC = 1e-3; MV0 = 1000.0
ML = 10e-6; MR_INT = 0.05; MR_ANT = 1.0; EFF_RAD = 0.25

C_eff  = MC / MARX_N
V_marx = MARX_N * MV0
R_tot  = MARX_N * MR_INT + MR_ANT
Z_char = np.sqrt(ML / C_eff)
f0     = 1 / (2*PI*np.sqrt(ML*C_eff))
zeta   = R_tot / (2*Z_char)
E_stored = 0.5 * C_eff * V_marx**2
print(f'Marx: V0={V_marx:.0f}V  C_eff={C_eff*1e3:.2f}mF  '
      f'Z_char={Z_char:.3f}Ohm  zeta={zeta:.3f}  f0={f0/1e3:.1f}kHz')
print(f'E_stored = {E_stored/1e3:.3f} kJ')

dt = 0.5e-6
n_steps = 1000
t_arr = np.zeros(n_steps); V_arr = np.zeros(n_steps); I_arr = np.zeros(n_steps)
P_arr = np.zeros(n_steps); E10_arr = np.zeros(n_steps); E1_arr = np.zeros(n_steps)

V, I = V_marx, 0.0
for k in range(n_steps):
    dI = (V - R_tot*I) / ML
    dV = -I / C_eff
    I  = max(0, I + dI*dt)
    V  = max(0, V + dV*dt)
    P_ant = I**2 * MR_ANT
    t_arr[k] = (k+1)*dt
    V_arr[k] = V; I_arr[k] = I; P_arr[k] = P_ant
    E10_arr[k] = np.sqrt(Z0 * P_ant*EFF_RAD / (4*PI*10**2))
    E1_arr[k]  = np.sqrt(Z0 * P_ant*EFF_RAD / (4*PI*1**2))

E_ant = np.trapz(P_arr, t_arr)
print(f'I_peak = {I_arr.max():.0f} A  P_peak = {P_arr.max()/1e6:.3f} MW  '
      f'E_ant = {E_ant:.0f} J  ({100*E_ant/E_stored:.0f}% of stored)')
print(f'Peak E-field @ 10m = {E10_arr.max():.0f} V/m  @ 1m = {E1_arr.max():.0f} V/m')

# ── 3. BLDC motor PID step response ──────────────────────────────────────
KT=0.12; KE=0.12; MR=0.80; MLIN=500e-6; MJ=0.0015; MB=8e-4; VBUS=24.0

def pid_step(sp, meas, integ, prev_e, kp, ki, kd, dt,
             out_min=-VBUS, out_max=VBUS):
    e     = sp - meas
    integ = integ + e*dt
    integ = np.clip(integ, out_min/max(ki,1e-9), out_max/max(ki,1e-9))
    u     = kp*e + ki*integ + kd*(e-prev_e)/dt
    return np.clip(u, out_min, out_max), integ, e

def motor_run(omega_cmd_seq, t_seq, dt=0.001, kp=8.0, ki=2.0, kd=0.05):
    '''Simulate BLDC velocity PID. omega_cmd_seq: list of (t_start, omega_cmd).'''
    t_end   = t_seq[-1]
    steps   = int(t_end / dt)
    t_out   = np.zeros(steps); w_out = np.zeros(steps); I_out = np.zeros(steps)
    omega   = 0.0; I_cur  = 0.0; integ = 0.0; prev_e = 0.0
    for k in range(steps):
        t   = k * dt
        # find current command
        cmd = t_seq[0]
        for ti, wi in zip(t_seq, omega_cmd_seq):
            if t >= ti: cmd = wi
        V_app, integ, prev_e = pid_step(cmd, omega, integ, prev_e, kp, ki, kd, dt,
                                        out_min=0, out_max=VBUS)
        dI    = (V_app - MR*I_cur - KE*omega) / MLIN
        domeg = (KT*I_cur - MB*omega) / MJ
        I_cur = np.clip(I_cur + dI*dt, -30, 30)
        omega = max(0, omega + domeg*dt)
        t_out[k] = t; w_out[k] = omega * 60/(2*PI); I_out[k] = I_cur
    return t_out, w_out, I_out

t_cmds  = [0.0, 1.0, 2.0, 3.5, 5.0]
w_cmds  = [0.0, 20.0*60/(2*PI),  # patrol: ~190 rpm
               5.0*60/(2*PI),    # acquire: slow
               62.8*60/(2*PI),   # evade: 600 rpm
               20.0*60/(2*PI)]   # back to patrol
t_motor, w_motor, I_motor = motor_run(w_cmds, t_cmds)

# ── 4. Battle score radar ─────────────────────────────────────────────────
categories   = ['Beam\nPrecision', 'EMP\nPower', 'Steer\nSpeed',
                'Range', 'Stealth', 'Cost\n(low=good)']
scores_rf    = [0.25, 0.85, 0.90, 0.60, 0.30, 0.70]
scores_opa   = [0.99, 0.00, 0.75, 0.90, 0.90, 0.35]
scores_hybrid= [0.99, 0.85, 0.85, 0.85, 0.60, 0.45]

N_ax = len(categories)
angles = np.linspace(0, 2*PI, N_ax, endpoint=False).tolist()
angles += angles[:1]

def close(lst): return lst + [lst[0]]

# ── 5. Plot ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor((0.04, 0.04, 0.10))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.45)

BG  = (0.06, 0.06, 0.14)
def dark(ax, title='', xl='', yl=''):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color('#334466')
    ax.tick_params(colors='#99aabb', labelsize=8)
    if title: ax.set_title(title, color='white', fontsize=9, fontweight='bold', pad=6)
    if xl:    ax.set_xlabel(xl, color='#99aabb', fontsize=8)
    if yl:    ax.set_ylabel(yl, color='#99aabb', fontsize=8)

# ── A: RF beam pattern ───────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(theta, 10*np.log10(np.maximum(AF_RF_0,  1e-6)), color='#50d8ff', lw=1.5, label='steer 0°')
ax0.plot(theta, 10*np.log10(np.maximum(AF_RF_30, 1e-6)), color='#ffd040', lw=1.2, label='steer 30°', ls='--')
ax0.plot(theta, 10*np.log10(np.maximum(AF_RF_60, 1e-6)), color='#ff7040', lw=1.0, label='steer 60°', ls=':')
ax0.axhline(-3, color='white', lw=0.6, ls=':', alpha=0.5)
ax0.set_ylim(-40, 1); ax0.set_xlim(-90, 90)
ax0.legend(fontsize=7, facecolor=BG, labelcolor='white', loc='upper right')
dark(ax0, f'RF Array  N={RF_N}, d=λ/2\nHPBW={hpbw_rf:.1f}°', 'θ [deg]', '|AF|² [dB]')

# ── B: OPA beam pattern (zoomed) ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
# Show full ±90° to show grating lobes
ax1.plot(theta, 10*np.log10(np.maximum(AF_OPA_0, 1e-6)), color='#00ff9f', lw=1.2)
ax1.axvline( np.degrees(np.arcsin(1/OPA_D)), color='#ff4444', lw=1, ls='--', label='grating lobe')
ax1.axvline(-np.degrees(np.arcsin(1/OPA_D)), color='#ff4444', lw=1, ls='--')
ax1.axhline(-3, color='white', lw=0.6, ls=':', alpha=0.5)
ax1.set_ylim(-50, 1); ax1.set_xlim(-90, 90)
ax1.legend(fontsize=7, facecolor=BG, labelcolor='white')
dark(ax1, f'OPA  N={OPA_N}, d=1.29λ\nHPBW={hpbw_opa:.4f}° ({hpbw_rf/hpbw_opa:.0f}x narrower)',
     'θ [deg]', '|AF|² [dB]')
# inset: zoom mainlobe
axin = ax1.inset_axes([0.62, 0.45, 0.36, 0.50])
axin.plot(theta, 10*np.log10(np.maximum(AF_OPA_0, 1e-6)), color='#00ff9f', lw=1)
axin.set_xlim(-0.3, 0.3); axin.set_ylim(-10, 1)
axin.set_facecolor((0.03,0.03,0.10)); axin.tick_params(labelsize=6, colors='#99aabb')
for sp in axin.spines.values(): sp.set_color('#334466')
axin.set_title('zoom', color='white', fontsize=6)

# ── C: Marx discharge V(t) and I(t) ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
t_us = t_arr * 1e6
ax2.plot(t_us, V_arr, color='#ffd040', lw=1.5, label='V_cap [V]')
ax2c = ax2.twinx()
ax2c.plot(t_us, I_arr, color='#ff3278', lw=1.5, label='I [A]')
ax2c.tick_params(colors='#ff3278', labelsize=8)
ax2c.set_ylabel('Current [A]', color='#ff3278', fontsize=8)
ax2c.yaxis.label.set_color('#ff3278')
for sp in ax2c.spines.values(): sp.set_color('#334466')
ax2.legend(loc='upper right',   fontsize=7, facecolor=BG, labelcolor='white')
ax2c.legend(loc='center right', fontsize=7, facecolor=BG, labelcolor='white')
dark(ax2, f'Marx RLC Discharge\nζ={zeta:.2f} overdamped  f₀={f0/1e3:.1f} kHz', 't [μs]', 'Voltage [V]')

# ── D: E-field vs time with thresholds ───────────────────────────────────
ax3 = fig.add_subplot(gs[0, 3])
ax3.semilogy(t_us, np.maximum(E10_arr, 0.1), color='#50d8ff', lw=1.5, label='E @ 10 m')
ax3.semilogy(t_us, np.maximum(E1_arr,  0.1),  color='#ff7040', lw=1.2, label='E @ 1 m', ls='--')
ax3.axhline(200,  color='#ffd040', lw=1, ls=':', label='Latch-up 200 V/m')
ax3.axhline(2000, color='#ff3278', lw=1, ls=':', label='Destroy 2 kV/m')
ax3.fill_between(t_us, 200, 2000,
                 where=(E10_arr >= 200) & (E10_arr < 2000),
                 color='#ffd040', alpha=0.10)
ax3.fill_between(t_us, 2000, 2e5,
                 where=E1_arr >= 2000, color='#ff3278', alpha=0.08)
ax3.set_xlim(0, t_us[-1]); ax3.set_ylim(1, 1e5)
ax3.legend(fontsize=7, facecolor=BG, labelcolor='white')
dark(ax3, 'EMP E-field vs Time\n(25% radiation efficiency)', 't [μs]', 'E-field [V/m]')

# ── E: Motor PID step response ────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
cmd_plot = np.array(w_cmds)[np.searchsorted(np.array(t_cmds), t_motor, side='right')-1]
ax4.plot(t_motor, w_motor,   color='#50d8ff', lw=1.5, label='ω actual [rpm]')
ax4.plot(t_motor, cmd_plot,  color='#ffd040', lw=0.8, ls='--', label='ω command')
ax4c = ax4.twinx()
ax4c.plot(t_motor, I_motor, color='#ff3278', lw=0.8, alpha=0.6, label='I_phase [A]')
ax4c.set_ylabel('Phase Current [A]', color='#ff3278', fontsize=8)
ax4c.tick_params(colors='#ff3278', labelsize=8)
for sp in ax4c.spines.values(): sp.set_color('#334466')
# State annotations
state_times  = [0.0, 1.0, 2.0, 3.5, 5.0]
state_labels = ['PATROL', 'ACQUIRE', 'CHARGE', 'EVADE', 'PATROL']
state_colors = ['#00ff9f', '#50d8ff', '#ffd040', '#ff3278', '#00ff9f']
for ti, lbl, col in zip(state_times, state_labels, state_colors):
    ax4.axvline(ti, color=col, lw=0.6, ls=':')
    ax4.text(ti+0.05, 680, lbl, color=col, fontsize=7, rotation=90, va='top')
ax4.legend(loc='upper left',  fontsize=7, facecolor=BG, labelcolor='white')
ax4c.legend(loc='upper right', fontsize=7, facecolor=BG, labelcolor='white')
dark(ax4, 'BLDC Motor PID — Combat Robot Drive (Kp=8 Ki=2 Kd=0.05)', 't [s]', 'Speed [rpm]')

# ── F: Battle score radar ─────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2:4], polar=True)
ax5.set_facecolor(BG)
ax5.spines['polar'].set_color('#334466')
ax5.tick_params(colors='#99aabb', labelsize=8)

for vals, col, lbl in [
        (scores_rf,     '#50d8ff', f'RF array'),
        (scores_opa,    '#00ff9f', f'OPA'),
        (scores_hybrid, '#ffd040', f'Hybrid (winner)'),
]:
    v = close(vals)
    ax5.plot(angles, v, color=col, lw=1.8, label=lbl)
    ax5.fill(angles, v, color=col, alpha=0.08)

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories, color='white', fontsize=8)
ax5.set_yticks([0.25, 0.5, 0.75, 1.0])
ax5.set_yticklabels(['', '0.5', '', '1.0'], color='#445566', fontsize=7)
ax5.set_ylim(0, 1)
ax5.legend(loc='lower right', bbox_to_anchor=(1.35, -0.05),
           fontsize=8, facecolor=BG, labelcolor='white')
ax5.set_title('Battle Score Radar\n(higher = better, except Cost)',
              color='white', fontsize=9, fontweight='bold', pad=14)
ax5.set_facecolor(BG); ax5.grid(color='#223344', lw=0.5)

fig.suptitle(
    'Sec 71: Electronics vs Photonics Combat Robot  '
    '|  RF Beamformer + OPA + Marx EMP + BLDC PID',
    color='white', fontsize=10, fontweight='bold', y=1.01
)

out_path = 'references/pptx_figures/beamformer_battle.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print(f'Saved: {out_path}')

# ── Print battle summary ─────────────────────────────────────────────────
print('\n--- BATTLE SUMMARY ---')
print(f'RF  HPBW  = {hpbw_rf:.2f} deg     (2.4 GHz, N=8)')
print(f'OPA HPBW  = {hpbw_opa:.5f} deg  (1550 nm, N=512)  {hpbw_rf/hpbw_opa:.0f}x narrower')
print(f'Marx I_pk = {I_arr.max():.0f} A   P_pk = {P_arr.max()/1e6:.2f} MW')
print(f'E @ 10 m  = {E10_arr.max():.0f} V/m  [{"DESTROY" if E10_arr.max()>2000 else "LATCH-UP" if E10_arr.max()>200 else "safe"}]')
print(f'E @  1 m  = {E1_arr.max():.0f} V/m  [{"DESTROY" if E1_arr.max()>2000 else "LATCH-UP" if E1_arr.max()>200 else "safe"}]')
print('VERDICT: RF for EMP, OPA for targeting, Marx for kill => HYBRID WINS')
"""

code_src = [line + "\n" for line in code_src.split("\n")]
# remove trailing newline
if code_src and code_src[-1] == "\n":
    code_src = code_src[:-1]

md_cell   = {"cell_type": "markdown", "metadata": {}, "source": md_src}
code_cell = {"cell_type": "code",     "execution_count": None,
             "metadata": {},          "outputs": [], "source": code_src}

nb["cells"].append(md_cell)
nb["cells"].append(code_cell)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. Notebook now has {len(nb['cells'])} cells.")
