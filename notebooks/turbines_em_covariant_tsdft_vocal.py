# Builder script — writes notebooks/turbines_em_covariant_tsdft_vocal.ipynb
# Run:  py -3.12 -X utf8 notebooks/turbines_em_covariant_tsdft_vocal.py
import json, pathlib, textwrap

NB_PATH = pathlib.Path(__file__).with_suffix(".ipynb")

def md(*lines):
    return {"cell_type":"markdown","metadata":{},"source":list(lines)}

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
            "source":[src]}

cells = [

# ── Title ──────────────────────────────────────────────────────────────────
md(
"# Turbines · EM · TS-DFT · SR · Covariant · SAR · Vocal\n",
"**Jalali Lab Engineering Reference** — special topics in EM, applied math, modern physics\n",
"\n",
"| § | Topic |\n",
"|---|-------|\n",
"| 1 | Turbine blade aerodynamics — NACA profile, Euler turbomachinery |\n",
"| 2 | Friction clutch + neuromuscular junction |\n",
"| 3 | EM special topics — waveguide TE/TM, dipole, Friis |\n",
"| 4 | Applied math — residue theorem, Green's functions, Bessel |\n",
"| 5 | Time-stretch DFT (Jalali lab) |\n",
"| 6 | Secondary sexual / HPG axis |\n",
"| 7 | Modern physics — SR, E=mc², photoelectric |\n",
"| 8 | Covariant electrodynamics — F$^{\\mu\\nu}$ tensor |\n",
"| 9 | Multiphysics biotech — SAR, thermal dose, FEM |\n",
"| 10 | Vocal tract acoustics — formants, rolled-R trill |\n",
),

code(textwrap.dedent("""\
import numpy as np, math, cmath
from scipy.special import jv, jn_zeros
from scipy.integrate import quad
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
if not hasattr(np,"trapezoid"): np.trapezoid=np.trapz
c_SI=2.998e8; mu0=4*np.pi*1e-7; eps0=8.854e-12
hbar=1.0546e-34; h_p=6.626e-34; k_B=1.381e-23; e_ch=1.602e-19; m_e=9.109e-31
""")),

# ── §1 ────────────────────────────────────────────────────────────────────
md(
"## §1 Turbine Blade Aerodynamics\n",
"\n",
"### Euler Turbomachinery Equation\n",
"\n",
"$$w_{\\text{shaft}} = U_2 C_{\\theta 2} - U_1 C_{\\theta 1} \\quad [\\text{J/kg}]$$\n",
"\n",
"where $U = \\omega r$ (blade speed) and $C_\\theta$ is the tangential (swirl) velocity.\n",
"\n",
"**Stage parameters:**\n",
"- Stage loading: $\\psi = \\Delta C_\\theta / U$\n",
"- Flow coefficient: $\\phi = C_a / U$\n",
"- Degree of reaction: $\\Lambda = 1 - (C_{\\theta 1}+C_{\\theta 2})/(2U)$\n",
"\n",
"### NACA 4-Digit Airfoil\n",
"\n",
"Thickness distribution:\n",
"$$y_t = 5t \\left[0.2969\\sqrt{x} - 0.1260x - 0.3516x^2 + 0.2843x^3 - 0.1015x^4\\right]$$\n",
),

code(textwrap.dedent("""\
import numpy as np, math
c_SI=2.998e8; gamma_air=1.4; cp_air=1005.0

def naca4(M_pct,P_10,TT_pct,n_pts=200):
    M,P,t = M_pct/100, P_10/10, TT_pct/100
    x = np.linspace(0,1,n_pts)
    yt = 5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
    if P>0 and P<1:
        yc = np.where(x<P, M/P**2*(2*P*x-x**2), M/(1-P)**2*(1-2*P+2*P*x-x**2))
        dyc= np.where(x<P, 2*M/P**2*(P-x),      2*M/(1-P)**2*(P-x))
    else:
        yc=dyc=np.zeros_like(x)
    th=np.arctan(dyc)
    return x-yt*np.sin(th), yc+yt*np.cos(th), x+yt*np.sin(th), yc-yt*np.cos(th), yc

# Axial turbine: U=300m/s, Ca=200m/s, ψ=1, Λ=0.5
U,Ca,psi,Lambda = 300.0,200.0,1.0,0.5
Ct1 = psi*U; Ct2 = 0.0
w_euler = U*(Ct1-Ct2)
T0_in=1600; dT0=w_euler/cp_air; eta_tt=0.88
PR = (1 - dT0*eta_tt/T0_in)**(gamma_air/(gamma_air-1))

fig,axs=plt.subplots(1,2,figsize=(12,4))

# Velocity triangle
for ax,title,Ct,beta_sign in [(axs[0],'Rotor Inlet (α₁,β₁)',Ct1,1),
                               (axs[1],'Rotor Exit (α₂,β₂)',Ct2,-1)]:
    alpha=math.atan2(Ct,Ca); beta=math.atan2(U-Ct,Ca)
    ax.annotate('',xy=(Ca,Ct),xytext=(0,0),
                arrowprops=dict(arrowstyle='->',color='C0',lw=2))
    ax.annotate('',xy=(Ca,Ct),xytext=(Ca,0),
                arrowprops=dict(arrowstyle='->',color='C1',lw=2))
    ax.annotate('',xy=(Ca,0),xytext=(0,0),
                arrowprops=dict(arrowstyle='->',color='C2',lw=2))
    ax.text(Ca/2,-30,'$C_a$',ha='center',fontsize=11,color='C2')
    ax.text(Ca+15,Ct/2,'Cθ',fontsize=11,color='C1')
    ax.text(Ca/2+15,Ct/2+10,'$C$',fontsize=12,color='C0')
    ax.set_xlim(-20,350); ax.set_ylim(-50,350)
    ax.set_xlabel('Axial [m/s]'); ax.set_ylabel('Tangential [m/s]')
    ax.set_title(f'{title}\\nα={math.degrees(alpha):.1f}° β={math.degrees(beta):.1f}°')
    ax.grid(True,alpha=0.3)
plt.suptitle(f'Velocity Triangles  w_Euler={w_euler/1000:.1f}kJ/kg  PR={PR:.3f}',fontsize=13)
plt.tight_layout(); plt.savefig('fig_s1_triangles.png',dpi=100); plt.show()

# NACA profiles
fig2,ax2=plt.subplots(figsize=(10,4))
colors=['C0','C1','C2']
for (M,P,T),col,lbl in zip([(0,0,12),(4,4,12),(2,4,15)],colors,
                            ['NACA 0012','NACA 4412','NACA 2415']):
    xu,yu,xl,yl,_ = naca4(M,P,T)
    ax2.fill_between(np.concatenate([xu,xl[::-1]]),
                     np.concatenate([yu,yl[::-1]]),alpha=0.25,color=col)
    ax2.plot(xu,yu,color=col,label=lbl); ax2.plot(xl,yl,color=col)
ax2.set_xlabel('x/c'); ax2.set_ylabel('y/c')
ax2.set_title('NACA Profiles (0012, 4412, 2415)')
ax2.legend(); ax2.grid(alpha=0.3); ax2.set_aspect('equal')
plt.tight_layout(); plt.savefig('fig_s1_naca.png',dpi=100); plt.show()
print(f"Stage: w={w_euler/1000:.1f}kJ/kg  ΔT₀={dT0:.1f}K  PR={PR:.4f}")
""")),

# ── §2 ────────────────────────────────────────────────────────────────────
md(
"## §2 Friction Clutch + Neuromuscular Junction\n",
"\n",
"### Disc Clutch Torque Capacity\n",
"\n",
"**Uniform wear** (most common design criterion):\n",
"$$T = n\\,\\mu\\,F_N\\,\\frac{R_o+R_i}{2}$$\n",
"\n",
"**Uniform pressure:**\n",
"$$T = \\frac{2}{3}\\,n\\,\\mu\\,F_N\\,\\frac{R_o^3-R_i^3}{R_o^2-R_i^2}$$\n",
"\n",
"### Engagement Dynamics\n",
"\n",
"$$t_{\\text{engage}} = \\frac{I_{\\text{eff}}\\,\\Delta\\omega}{T_{\\text{clutch}}}, \\quad"
" I_{\\text{eff}} = \\frac{I_f I_d}{I_f+I_d}, \\quad"
" E_{\\text{slip}} = \\tfrac{1}{2}I_{\\text{eff}}\\,\\Delta\\omega^2$$\n",
"\n",
"### Hill Muscle Model\n",
"\n",
"$$(F+a)(v+b) = (F_0+a)\\,b \\quad \\Rightarrow \\quad"
" P_{\\max}\\text{ at }v^* = b(\\sqrt{5}-1)$$\n",
),

code(textwrap.dedent("""\
import numpy as np, math, matplotlib.pyplot as plt

Ro,Ri,mu_cl,FN,n_s = 0.12,0.075,0.35,8000.0,2
Rm=(Ro+Ri)/2; T_uw=n_s*mu_cl*FN*Rm; T_up=(2/3)*n_s*mu_cl*FN*(Ro**3-Ri**3)/(Ro**2-Ri**2)

omega_e=2*np.pi*3000/60; omega_s=2*np.pi*1000/60; dw=omega_e-omega_s
If,Id=0.15,0.08; Ieff=If*Id/(If+Id); Tload=50.0
t_eng=Ieff*dw/(T_uw-Tload); E_slip=0.5*Ieff*dw**2

# Hill model
F0,vmax=1000.0,0.5; a_h=F0/4; b_h=vmax/4
v_arr=np.linspace(0,vmax*0.99,300)
F_hill=(F0*b_h-a_h*v_arr)/(v_arr+b_h); P_hill=F_hill*v_arr
v_opt_an=b_h*(math.sqrt(5)-1); F_at_v=(F0*b_h-a_h*v_opt_an)/(v_opt_an+b_h)
Pmax=F_at_v*v_opt_an

fig,axs=plt.subplots(1,2,figsize=(12,4))
axs[0].plot(v_arr,F_hill,'C0',lw=2,label='Force F(v)')
ax0b=axs[0].twinx()
ax0b.plot(v_arr,P_hill,'C1',lw=2,label='Power P(v)')
ax0b.axvline(v_opt_an,color='C1',ls='--',alpha=0.7)
ax0b.axhline(Pmax,color='C1',ls=':',alpha=0.7)
ax0b.set_ylabel('Power [W]',color='C1')
axs[0].set_xlabel('Velocity [m/s]'); axs[0].set_ylabel('Force [N]',color='C0')
axs[0].set_title(f'Hill Muscle F–v\\nP_max={Pmax:.1f}W at v*={v_opt_an:.3f}m/s')
axs[0].grid(alpha=0.3)

# Clutch engagement: Δω(t)
t_sim=np.linspace(0,t_eng*1.5,300)
dw_t = np.maximum(dw - T_uw/Ieff*t_sim, 0.0)
engaged = t_sim >= t_eng
axs[1].plot(t_sim*1000, dw_t,'C0',lw=2,label='Slip speed Δω')
axs[1].axvline(t_eng*1000,color='C2',ls='--',label=f't_eng={t_eng*1000:.1f}ms')
axs[1].set_xlabel('Time [ms]'); axs[1].set_ylabel('Δω [rad/s]')
axs[1].set_title(f'Clutch Engagement  E_slip={E_slip:.0f}J')
axs[1].legend(); axs[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_s2_clutch.png',dpi=100); plt.show()
print(f"T_uw={T_uw:.1f}Nm  T_up={T_up:.1f}Nm  t_eng={t_eng*1000:.1f}ms  E_slip={E_slip:.0f}J")
""")),

# ── §3 ────────────────────────────────────────────────────────────────────
md(
"## §3 EM Special Topics\n",
"\n",
"### Rectangular Waveguide — TE/TM Cutoff\n",
"\n",
"$$f_{c,mn} = \\frac{c}{2}\\sqrt{\\left(\\frac{m}{a}\\right)^2+\\left(\\frac{n}{b}\\right)^2}$$\n",
"\n",
"Above cutoff: $\\beta = \\sqrt{k^2-k_c^2}$, phase/group velocities:\n",
"$$v_p = \\frac{\\omega}{\\beta} > c, \\qquad v_g = \\frac{c^2}{v_p} < c$$\n",
"\n",
"Wave impedance: $Z_{TE} = \\eta/\\sqrt{1-(f_c/f)^2}$\n",
"\n",
"### Half-Wave Dipole\n",
"\n",
"$$F(\\theta) = \\left[\\frac{\\cos\\!(\\frac{\\pi}{2}\\cos\\theta)}{\\sin\\theta}\\right]^2, "
"\\quad R_{\\rm rad}=73.1\\,\\Omega, \\quad D=1.64\\,(2.15\\,\\text{dBi})$$\n",
"\n",
"### Friis Transmission\n",
"\n",
"$$\\frac{P_r}{P_t} = G_t G_r \\left(\\frac{\\lambda}{4\\pi R}\\right)^2$$\n",
),

code(textwrap.dedent("""\
import numpy as np, math, matplotlib.pyplot as plt
c_SI=2.998e8; eta0=377.0

a_wg,b_wg = 22.86e-3,10.16e-3   # WR-90
f_op=10e9; k_op=2*np.pi*f_op/c_SI

fig,axs=plt.subplots(1,2,figsize=(12,4))

# TE10 propagation vs frequency
f_arr=np.linspace(7e9,18e9,400)
f_c10=c_SI/(2*a_wg); fc20=c_SI/a_wg
beta_arr=np.sqrt(np.maximum((2*np.pi*f_arr/c_SI)**2-(np.pi/a_wg)**2,0))
vp_c=np.where(beta_arr>0,(2*np.pi*f_arr/c_SI)/beta_arr,np.nan)
vg_c=np.where(beta_arr>0,beta_arr/(2*np.pi*f_arr/c_SI),np.nan)
axs[0].plot(f_arr/1e9,vp_c,'C0',lw=2,label='$v_p/c$')
axs[0].plot(f_arr/1e9,vg_c,'C1',lw=2,label='$v_g/c$')
axs[0].axhline(1,color='k',ls='--',alpha=0.5,label='c')
axs[0].axvline(f_c10/1e9,color='C3',ls=':',label=f'$f_{{c,10}}$={f_c10/1e9:.2f}GHz')
axs[0].set_xlabel('f [GHz]'); axs[0].set_ylabel('v/c')
axs[0].set_title('WR-90 TE₁₀: Phase & Group Velocity')
axs[0].legend(); axs[0].grid(alpha=0.3); axs[0].set_ylim(0,3)

# Dipole radiation pattern
theta=np.linspace(0.01,np.pi-0.01,500)
F_dip=(np.cos(np.pi/2*np.cos(theta))/np.sin(theta))**2
F_dip/=F_dip.max()
ax_polar=plt.subplot(122,polar=True)
ax_polar.plot(theta,F_dip,'C0',lw=2); ax_polar.plot(-theta,F_dip,'C0',lw=2)
ax_polar.set_title('Half-wave Dipole Radiation Pattern', pad=15)
ax_polar.set_theta_zero_location('N'); ax_polar.set_theta_direction(-1)

plt.tight_layout(); plt.savefig('fig_s3_wg_dipole.png',dpi=100); plt.show()

# Friis
f_5g=28e9; Pt=30; Gt=25; Gr=20; R=100
lam5g=c_SI/f_5g; FSPL=20*math.log10(4*math.pi*R/lam5g)
Pr=Pt+Gt+Gr-FSPL
print(f"TE10 cutoff={f_c10/1e9:.3f}GHz  v_p/c at 10GHz={float(c_SI/(c_SI*np.sqrt(1-(f_c10/f_op)**2))):.4f}")
print(f"Friis 28GHz: FSPL={FSPL:.1f}dB  P_r={Pr:.1f}dBm")
""")),

# ── §4 ────────────────────────────────────────────────────────────────────
md(
"## §4 Applied Math — Residue Theorem, Green's Functions, Bessel\n",
"\n",
"### Residue Theorem\n",
"\n",
"$$\\oint_C f(z)\\,dz = 2\\pi i \\sum_k \\mathrm{Res}[f,\\,z_k]$$\n",
"\n",
"**Example:** Close in upper half-plane, pole at $z=i$:\n",
"$$\\int_{-\\infty}^{\\infty}\\frac{dx}{1+x^2} = 2\\pi i\\cdot\\mathrm{Res}\\left[\\frac{1}{1+z^2},\\,i\\right]"
" = 2\\pi i\\cdot\\frac{1}{2i} = \\pi$$\n",
"\n",
"### 1-D Green's Function\n",
"\n",
"$$-\\frac{d^2}{dx^2}G(x,x') = \\delta(x-x'), \\quad G(0)=G(1)=0"
"\\quad\\Rightarrow\\quad G(x,x') = x_<(1-x_>)$$\n",
"\n",
"### Bessel Functions\n",
"\n",
"$$x^2 y'' + x y' + (x^2-n^2)y = 0, \\qquad"
" J_n(x)\\underset{x\\to\\infty}{\\longrightarrow}"
"\\sqrt{\\tfrac{2}{\\pi x}}\\cos\\!\\left(x-\\tfrac{n\\pi}{2}-\\tfrac{\\pi}{4}\\right)$$\n",
),

code(textwrap.dedent("""\
import numpy as np, math
from scipy.special import jv, jn_zeros
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Residue theorem
I_exact = quad(lambda x: 1/(1+x**2), -np.inf, np.inf)[0]
I_residue = 2*np.pi   # 2πi * 1/(2i) = π   BUT wait: 2πi*(1/2i)=π not 2π
# Correction: Res[1/(1+z²),i] = 1/(2i), so 2πi*(1/(2i)) = π
I_residue_correct = np.pi
print(f"∫_{{-∞}}^∞ 1/(1+x²)dx = {I_exact:.6f}  (= π = {np.pi:.6f})  err={abs(I_exact-np.pi):.2e}")

# 1D Green's function
def G1D(x,xp): return min(x,xp)*(1-max(x,xp))
xg=np.linspace(0,1,150)
u_GF=np.array([np.trapezoid([G1D(xi,xp) for xp in xg],xg) for xi in xg])
u_ex=xg*(1-xg)/2
print(f"Green's fn max error: {np.max(np.abs(u_GF-u_ex)):.2e}")

# Bessel J_0..J_3
x_b=np.linspace(0,20,400)
fig,axs=plt.subplots(1,2,figsize=(12,4))
for n,col in enumerate(['C0','C1','C2','C3']):
    axs[0].plot(x_b,jv(n,x_b),color=col,lw=1.8,label=f'$J_{n}(x)$')
axs[0].axhline(0,color='k',lw=0.7)
axs[0].set_xlabel('x'); axs[0].set_ylabel('$J_n(x)$')
axs[0].set_title('Bessel Functions $J_n(x)$')
axs[0].legend(); axs[0].grid(alpha=0.3); axs[0].set_ylim(-0.5,1.05)

# Green's function solution
axs[1].plot(xg,u_ex,'C0',lw=2,label='Exact $x(1-x)/2$')
axs[1].plot(xg,u_GF,'C1--',lw=1.5,label="Green's fn")
axs[1].set_xlabel('x'); axs[1].set_ylabel('u(x)')
axs[1].set_title(r"Green's Function: $-u''=1$, $u(0)=u(1)=0$")
axs[1].legend(); axs[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_s4_bessel_gf.png',dpi=100); plt.show()
print(f"Zeros of J_0: {jn_zeros(0,4).round(4)}")
""")),

# ── §5 ────────────────────────────────────────────────────────────────────
md(
"## §5 Time-Stretch Dispersive Fourier Transform\n",
"\n",
"### Stationary Phase Derivation\n",
"\n",
"After propagation through dispersive fiber with transfer function "
"$H(\\omega)=e^{i\\beta_2 L\\omega^2/2}$:\n",
"\n",
"$$E_{\\rm out}(t) = \\int \\tilde{E}(\\omega)\\,e^{i\\omega t}\\,e^{i\\beta_2 L\\omega^2/2}\\,d\\omega$$\n",
"\n",
"Stationary phase condition: $\\partial_\\omega(\\omega t + \\beta_2 L\\omega^2/2)=0 "
"\\Rightarrow \\omega^* = -t/(\\beta_2 L)$\n",
"\n",
"$$\\boxed{|E_{\\rm out}(t)|^2 \\propto \\left|\\tilde{E}\\!\\left(\\frac{t}{\\beta_2 L}\\right)\\right|^2}$$\n",
"\n",
"**The optical spectrum is mapped to time.** "
"Valid when $|\\beta_2 L| \\gg \\tau_{\\rm pulse}^2$ (temporal Fraunhofer regime).\n",
"\n",
"### TS-ADC Bandwidth Multiplication\n",
"\n",
"$$\\text{Effective BW} = M \\times \\text{ADC BW}, \\qquad M = 1 + D_2/D_1$$\n",
"\n",
"### Connection to GS Phase Retrieval\n",
"\n",
"$$H_{\\rm disp}(\\omega) = e^{i D\\omega^2/2}, \\quad |H|=1 \\text{ (all-pass)}, "
"\\quad \\phi(\\omega)=D\\omega^2/2 \\text{ (diversity)}$$\n",
"\n",
"Measure $I_1=|E|^2$, $I_2=|H_{\\rm disp}\\cdot E|^2$ → GS retrieves $\\phi(\\omega)$. "
"Our research requires $|D|\\geq 5000\\,\\text{ps}^2$.\n",
),

code(textwrap.dedent("""\
import numpy as np, matplotlib.pyplot as plt

N=2048; T_win=10e-9; dt=T_win/N
t_arr=np.linspace(-T_win/2,T_win/2,N)
f_arr=np.fft.rfftfreq(N,dt)

tau_p=10e-12; f_sig=3e9; A_mod=0.5
E_in=np.exp(-t_arr**2/(2*tau_p**2))*(1+A_mod*np.cos(2*np.pi*f_sig*t_arr))

# Three dispersion values
fig,axs=plt.subplots(2,3,figsize=(14,7))
for col_i,(b2L,label) in enumerate([(1e-21,'β₂L=1ps²'),(5e-21,'β₂L=5ps²'),(20e-21,'β₂L=20ps²')]):
    omega=2*np.pi*np.fft.fftfreq(N,dt)
    H=np.exp(1j*b2L/2*omega**2)
    E_out=np.fft.ifft(np.fft.fft(E_in)*H)
    I_out=np.abs(E_out)**2
    I_spec=np.abs(np.fft.rfft(E_in))**2
    axs[0,col_i].plot(t_arr*1e9,np.abs(E_in)**2,'C0',lw=1.2,label='Input intensity')
    axs[0,col_i].plot(t_arr*1e9,I_out/I_out.max(),'C1',lw=1.5,label='TS-DFT output')
    axs[0,col_i].set_xlabel('Time [ns]'); axs[0,col_i].set_title(label)
    axs[0,col_i].legend(fontsize=8); axs[0,col_i].grid(alpha=0.3)
    axs[1,col_i].semilogy(f_arr[:N//2]/1e9,I_spec[:N//2]+1e-20,'C0',lw=1.5)
    axs[1,col_i].set_xlabel('Frequency [GHz]'); axs[1,col_i].set_ylabel('Power spectrum')
    axs[1,col_i].set_xlim(0,10); axs[1,col_i].grid(alpha=0.3)
    axs[1,col_i].set_title(f'Spectrum ({label})')
axs[0,0].set_ylabel('Norm. intensity')
plt.suptitle('Time-Stretch DFT: larger β₂L → spectrum mapped to time', fontsize=13)
plt.tight_layout(); plt.savefig('fig_s5_tsdft.png',dpi=100); plt.show()

# GS diversity connection
D_ps2=5000; omega_test=2*np.pi*1e12
phi_D=D_ps2*1e-24/2*omega_test**2
print(f"D={D_ps2}ps²: φ(1THz)={phi_D/(2*np.pi):.0f} cycles (quadratic diversity phase)")
print(f"TS-ADC: M=100 × 10GHz ADC → effective BW=1000 GHz (Jalali group)")
""")),

# ── §6 ────────────────────────────────────────────────────────────────────
md(
"## §6 Secondary Sexual Characteristics — HPG Axis\n",
"\n",
"### Goodwin Oscillator (GnRH Pulse Generator)\n",
"\n",
"$$\\frac{dx_1}{dt} = \\frac{1}{1+x_3^n} - \\alpha_1 x_1, \\quad"
"\\frac{dx_2}{dt} = x_1 - \\alpha_2 x_2, \\quad"
"\\frac{dx_3}{dt} = x_2 - \\alpha_3 x_3$$\n",
"\n",
"where $x_1\\equiv$GnRH, $x_2\\equiv$LH, $x_3\\equiv$Testosterone. "
"Oscillates for $n>8$ (Hill cooperativity).\n",
"\n",
"### Androgen Receptor Binding (Langmuir Isotherm)\n",
"\n",
"$$B_{\\rm bound} = \\frac{B_{\\rm max}\\cdot[T]}{K_d + [T]}$$\n",
"\n",
"### HPG Feedback Loop\n",
"\n",
"$$\\text{Hypothalamus} \\xrightarrow{\\text{GnRH (60-90 min pulses)}} "
"\\text{Anterior Pituitary} \\xrightarrow{\\text{LH/FSH}} "
"\\text{Gonads} \\xrightarrow{\\text{T, E₂, Inhibin}} \\text{(−) feedback}$$\n",
),

code(textwrap.dedent("""\
import numpy as np, matplotlib.pyplot as plt
from scipy.signal import find_peaks

n_hw=8.0; alpha=np.array([0.5,0.5,0.5])
dt_h=0.1; t_h=np.arange(0,200,dt_h)
X=np.zeros((len(t_h),3)); X[0]=[0.5,0.5,0.5]
for i in range(len(t_h)-1):
    x1,x2,x3=X[i]
    X[i+1]=X[i]+dt_h*np.array([1/(1+x3**n_hw)-alpha[0]*x1,
                                  x1-alpha[1]*x2, x2-alpha[2]*x3])

pks,_=find_peaks(X[100:,0],height=0.3)
period=float(np.mean(np.diff(pks)))*dt_h if len(pks)>=2 else float('nan')

# AR binding
Bmax=1000; Kd=0.5; T_c=np.linspace(0,10,200)
B_b=Bmax*T_c/(Kd+T_c)

fig,axs=plt.subplots(1,2,figsize=(12,4))
for i,(lbl,col) in enumerate([('GnRH $x_1$','C0'),('LH $x_2$','C1'),('Testosterone $x_3$','C2')]):
    axs[0].plot(t_h,X[:,i],color=col,lw=1.8,label=lbl)
axs[0].set_xlabel('Time [a.u.]'); axs[0].set_ylabel('Concentration [a.u.]')
axs[0].set_title(f'Goodwin HPG Oscillator  T={period:.1f} units')
axs[0].legend(); axs[0].grid(alpha=0.3); axs[0].set_xlim(80,200)

axs[1].plot(T_c,B_b,'C0',lw=2)
axs[1].axvline(Kd,color='C1',ls='--',label=f'$K_d$={Kd}nM (50% occupancy)')
axs[1].axhline(Bmax/2,color='C1',ls=':')
axs[1].set_xlabel('[T] free (nM)'); axs[1].set_ylabel('Bound receptor (fmol/mg)')
axs[1].set_title('Androgen Receptor Langmuir Binding')
axs[1].legend(); axs[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_s6_hpg.png',dpi=100); plt.show()
print(f"GnRH pulse period ≈ {period:.1f} model units  (real ~60-90 min)")
""")),

# ── §7 ────────────────────────────────────────────────────────────────────
md(
"## §7 Modern Physics — Special Relativity, E=mc², Photoelectric\n",
"\n",
"### Special Relativity\n",
"\n",
"$$\\gamma = \\frac{1}{\\sqrt{1-\\beta^2}}, \\qquad \\beta=v/c$$\n",
"\n",
"$$E^2 = (pc)^2 + (m_0 c^2)^2 \\qquad (\\text{energy-momentum relation})$$\n",
"\n",
"### Photoelectric Effect (Einstein 1905)\n",
"\n",
"$$KE_{\\max} = h\\nu - \\phi \\qquad (\\phi = \\text{work function})$$\n",
"\n",
"$$\\nu_0 = \\phi/h \\quad \\text{(threshold frequency)}$$\n",
"\n",
"### Relativistic Velocity Addition\n",
"\n",
"$$u = \\frac{u_1+u_2}{1+u_1 u_2/c^2} \\qquad (u_1,u_2 \\to c \\Rightarrow u\\to c)$$\n",
"\n",
"### de Broglie Matter Wave\n",
"\n",
"$$\\lambda = \\frac{h}{p} = \\frac{h}{\\gamma m v}$$\n",
),

code(textwrap.dedent("""\
import numpy as np, math, matplotlib.pyplot as plt
c_SI=2.998e8; h_p=6.626e-34; e_ch=1.602e-19; m_e=9.109e-31

betas=np.linspace(0,0.9999,500)
gammas=1/np.sqrt(1-betas**2)

# Photoelectric (Na, φ=2.27eV)
phi_Na=2.27
nu_arr=np.linspace(4e14,1.2e15,300)
KE_arr=np.maximum(h_p*nu_arr/e_ch-phi_Na,0)
nu0=phi_Na*e_ch/h_p

fig,axs=plt.subplots(1,2,figsize=(12,4))
axs[0].semilogy(betas,gammas,'C0',lw=2,label='Lorentz factor γ')
axs[0].semilogy(betas,(gammas-1),'C1',lw=2,label='KE / m₀c² = γ−1')
axs[0].semilogy(betas,1/gammas,'C2',lw=2,label='Length L/L₀ = 1/γ')
axs[0].axvline(0.9,color='k',ls='--',alpha=0.4)
axs[0].set_xlabel('β = v/c'); axs[0].set_ylabel('Dimensionless factor')
axs[0].set_title('Special Relativity: γ, Length Contraction, KE')
axs[0].legend(); axs[0].grid(alpha=0.3,which='both')
axs[0].set_ylim(0.1,50); axs[0].set_xlim(0,1)

axs[1].plot(nu_arr/1e14,KE_arr,'C0',lw=2)
axs[1].axvline(nu0/1e14,color='C1',ls='--',label=f'nu0={nu0/1e14:.2f}e14 Hz ({c_SI/nu0*1e9:.0f}nm)')
axs[1].set_xlabel('Frequency [×10¹⁴ Hz]'); axs[1].set_ylabel('KE [eV]')
axs[1].set_title(f'Photoelectric Effect (Na, φ={phi_Na}eV)')
axs[1].legend(); axs[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_s7_sr.png',dpi=100); plt.show()

# Velocity addition
v1=v2=0.9*c_SI; u_rel=(v1+v2)/(1+v1*v2/c_SI**2)
lam_1keV=h_p/math.sqrt(2*m_e*1000*e_ch)
print(f"v_add(0.9c+0.9c)={u_rel/c_SI:.4f}c  de Broglie 1keV e⁻: λ={lam_1keV*1e12:.2f}pm")
""")),

# ── §8 ────────────────────────────────────────────────────────────────────
md(
"## §8 Covariant Electrodynamics — $F^{\\mu\\nu}$ Tensor\n",
"\n",
"### Electromagnetic Field Tensor\n",
"\n",
"$$F^{\\mu\\nu} = \\partial^\\mu A^\\nu - \\partial^\\nu A^\\mu, \\qquad "
"F^{\\mu\\nu} = \\begin{pmatrix} 0 & -E_x/c & -E_y/c & -E_z/c \\\\ "
"E_x/c & 0 & -B_z & B_y \\\\ "
"E_y/c & B_z & 0 & -B_x \\\\ "
"E_z/c & -B_y & B_x & 0 \\end{pmatrix}$$\n",
"\n",
"### Maxwell in Covariant Form\n",
"\n",
"$$\\underbrace{\\partial_\\mu F^{\\mu\\nu} = \\mu_0 J^\\nu}_{\\text{Gauss + Ampere-Maxwell}}, "
"\\qquad \\underbrace{\\partial_{[\\mu}F_{\\nu\\rho]}=0}_{\\text{Gauss-B + Faraday}}$$\n",
"\n",
"### EM Lagrangian Density\n",
"\n",
"$$\\mathcal{L} = -\\frac{F_{\\mu\\nu}F^{\\mu\\nu}}{4\\mu_0} - A_\\mu J^\\mu "
"\\xrightarrow{\\text{E-L}} \\partial_\\mu F^{\\mu\\nu} = \\mu_0 J^\\nu$$\n",
"\n",
"Gauge invariance: $A^\\mu \\to A^\\mu + \\partial^\\mu \\chi$ leaves $F_{\\mu\\nu}$ unchanged.\n",
),

code(textwrap.dedent("""\
import numpy as np, matplotlib.pyplot as plt
c_SI=2.998e8; mu0=4*np.pi*1e-7

def F_tensor(E,B):
    Ex,Ey,Ez=E; Bx,By,Bz=B
    F=np.zeros((4,4))
    F[0,1]=-Ex/c_SI; F[0,2]=-Ey/c_SI; F[0,3]=-Ez/c_SI
    F[1,0]= Ex/c_SI; F[2,0]= Ey/c_SI; F[3,0]= Ez/c_SI
    F[1,2]=-Bz; F[2,1]= Bz; F[1,3]= By; F[3,1]=-By
    F[2,3]=-Bx; F[3,2]= Bx
    return F

# Visualize F^μν for various field configurations
configs=[
    ([1e3,0,0],[0,0,1e-3],'$E_x$=1kV/m, $B_z$=1mT'),
    ([0,1e3,0],[1e-3,0,0],'$E_y$=1kV/m, $B_x$=1mT'),
    ([1e3,1e3,0],[0,0,0],'Pure E field'),
]
fig,axs=plt.subplots(1,3,figsize=(14,4))
labels=['t','x','y','z']
for ax,(E,B,title) in zip(axs,configs):
    F=F_tensor(E,B)
    eta=np.diag([1.0,-1.0,-1.0,-1.0])
    F_low=eta@F@eta
    im=ax.imshow(F_low/np.abs(F_low).max() if np.abs(F_low).max()>0 else F_low,
                 cmap='RdBu_r',vmin=-1,vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    clean=title.replace('$','').replace('_','').replace('{','').replace('}','')
    ax.set_title(f'F_mu_nu: {clean}',fontsize=9)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f'{F_low[i,j]:.1e}',ha='center',va='center',fontsize=7)
plt.colorbar(im,ax=axs[-1])
plt.suptitle('EM Field Tensor F_mu_nu (lowered indices, +--- metric)',fontsize=12)
plt.tight_layout(); plt.savefig('fig_s8_fmunu.png',dpi=100); plt.show()

# Lorentz invariants
E=[1e3,0,0]; B=[0,0,1e-3]
Fm=F_tensor(E,B); eta=np.diag([1.0,-1.0,-1.0,-1.0])
I1=np.sum(eta@Fm@eta * Fm)
print(f"F_{{μν}}F^{{μν}} = {I1:.4e}  (= 2(B²−E²/c²)/... Lorentz invariant)")
print(f"Antisymmetry check: F^01+F^10 = {Fm[0,1]+Fm[1,0]:.2e} (=0)")
""")),

# ── §9 ────────────────────────────────────────────────────────────────────
md(
"## §9 Multiphysics Biotech — SAR, Bio-Heat, CEM43\n",
"\n",
"### Specific Absorption Rate\n",
"\n",
"$$\\text{SAR} = \\frac{\\sigma |\\mathbf{E}|^2}{2\\rho} \\quad [\\text{W/kg}]$$\n",
"\n",
"FCC limit: 1.6 W/kg per 1g tissue. Implant safety: < 0.1 W/kg.\n",
"\n",
"### Pennes Bio-Heat Equation\n",
"\n",
"$$\\rho c\\,\\frac{\\partial T}{\\partial t} = \\nabla\\cdot(k\\nabla T) "
"+ Q_{\\rm SAR} - W_b c_b(T-T_a) + Q_{\\rm met}$$\n",
"\n",
"### CEM43 Thermal Dose\n",
"\n",
"$$\\text{CEM}_{43} = \\int_0^{t_{\\rm end}} R^{43-T(t)}\\,dt, \\qquad"
"R=\\begin{cases}0.25 & T<43°C \\\\ 0.5 & T\\geq 43°C\\end{cases}$$\n",
"\n",
"Threshold: CEM43 > 240 min → irreversible muscle damage.\n",
),

code(textwrap.dedent("""\
import numpy as np, matplotlib.pyplot as plt

N=100; xb=np.linspace(0,0.02,N); dx=xb[1]-xb[0]
k_t=0.5; rho_t=1060; Wb=0.5e-3; cb=3825; Ta=37.0; Qmet=2500

sigma_EM=0.5; E_amp=500.0; delta_pen=0.025
SAR=sigma_EM*E_amp**2*np.exp(-2*xb/delta_pen)/(2*rho_t)
Qsar=SAR*rho_t

A=np.zeros((N,N)); b=np.zeros(N)
cd=2*k_t/dx**2+Wb*cb; co=-k_t/dx**2
for i in range(1,N-1):
    A[i,i-1]=co; A[i,i]=cd; A[i,i+1]=co
    b[i]=Qsar[i]+Wb*cb*Ta+Qmet
A[0,0]=1; b[0]=Ta; A[-1,-1]=1; A[-1,-2]=-1; b[-1]=0
T_bio=np.linalg.solve(A,b)
T_max=float(T_bio.max()); x_max=float(xb[np.argmax(T_bio)])

# CEM43
t_heat=np.linspace(0,600,1000)
T_path=Ta+(T_max-Ta)*np.sin(np.pi/2*t_heat/600)  # gradual heating profile
dt_cem=t_heat[1]-t_heat[0]
R_arr=np.where(T_path>=43,0.5,0.25)
CEM43=np.cumsum(R_arr**(43-T_path)*dt_cem/60)

fig,axs=plt.subplots(1,3,figsize=(14,4))
axs[0].plot(xb*100,SAR,'C0',lw=2)
axs[0].axhline(1.6,color='C3',ls='--',label='FCC limit 1.6W/kg')
axs[0].set_xlabel('Depth [cm]'); axs[0].set_ylabel('SAR [W/kg]')
axs[0].set_title(f'SAR depth profile\\nSurface={SAR[0]:.1f}W/kg')
axs[0].legend(); axs[0].grid(alpha=0.3)

axs[1].plot(xb*100,T_bio,'C0',lw=2)
axs[1].axhline(37,color='k',ls=':',alpha=0.5,label='Body temp 37°C')
axs[1].axhline(43,color='C3',ls='--',label='43°C threshold')
axs[1].set_xlabel('Depth [cm]'); axs[1].set_ylabel('Temperature [°C]')
axs[1].set_title(f'Pennes Bio-Heat  T_max={T_max:.1f}°C at {x_max*100:.1f}cm')
axs[1].legend(); axs[1].grid(alpha=0.3)

axs[2].plot(t_heat/60,CEM43,'C0',lw=2,label='CEM43(t)')
axs[2].axhline(240,color='C3',ls='--',label='240 min (ablation threshold)')
axs[2].axhline(25,color='C1',ls='--',label='25 min (skin damage)')
axs[2].set_xlabel('Time [min]'); axs[2].set_ylabel('CEM43 [min]')
axs[2].set_title('Thermal Dose Accumulation'); axs[2].legend(); axs[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_s9_bioheat.png',dpi=100); plt.show()
print(f"T_max={T_max:.2f}°C  SAR_surface={SAR[0]:.1f}W/kg  CEM43(10min)={CEM43[-1]:.1f}min")
""")),

# ── §10 ───────────────────────────────────────────────────────────────────
md(
"## §10 Vocal Tract Acoustics — Formants, Rolled-R Trill\n",
"\n",
"### Uniform Tube Resonances (quarter-wave)\n",
"\n",
"$$F_n = \\frac{(2n-1)\\,c_s}{4L}, \\quad n=1,2,3,\\ldots \\qquad (L\\approx 17\\,\\text{cm})$$\n",
"\n",
"### Source-Filter Model (Fant 1960)\n",
"\n",
"$$S(f) = G(f)\\cdot H(f)\\cdot R(f)$$\n",
"\n",
"- $G(f)$: glottal source ($\\propto 1/f^2$, harmonic comb at $f_0$)\n",
"- $H(f)$: vocal tract filter (resonance peaks = formants)\n",
"- $R(f)$: lip radiation ($\\propto f$, +6 dB/oct)\n",
"\n",
"### Rolled-R Aeromechanics\n",
"\n",
"Self-sustaining tongue-tip flutter via Bernoulli suction:\n",
"\n",
"$$\\Delta P_{\\rm Bernoulli} = \\tfrac{1}{2}\\rho(v_2^2-v_1^2) \\approx 225\\,\\text{Pa}$$\n",
"\n",
"Oscillation frequency $f_{\\rm trill}\\approx 30\\,\\text{Hz}$, "
"requires supraglottal pressure 3–8 cmH₂O $\\approx$ 300–800 Pa.\n",
),

code(textwrap.dedent("""\
import numpy as np, matplotlib.pyplot as plt

cs=350; L=0.17
F_n=[(2*n-1)*cs/(4*L) for n in range(1,6)]
vowels={'i':(270,2290),'ɪ':(390,1990),'ɛ':(610,1900),'æ':(860,1550),
        'ɑ':(730,1090),'ɔ':(570,840),'u':(300,870),'ʊ':(440,1020),
        'ə':(500,1500),'ʌ':(640,1190)}

# Source-filter synthesis of /ɑ/
fs=16000; N_syn=2048
f_arr=np.fft.rfftfreq(N_syn,1/fs); f0=120
G=np.zeros(len(f_arr))
for fh in np.arange(f0,fs/2,f0):
    idx=np.argmin(np.abs(f_arr-fh)); G[idx]=1/(fh**2+1)
f1a,f2a,f3a=730,1090,2500; bws=[60,70,150]
H=np.ones(len(f_arr))
for fc,bw in zip([f1a,f2a,f3a],bws):
    H/=np.sqrt((1-(f_arr/fc)**2)**2+(bw*f_arr/(fc**2+1))**2+1e-15)
R=np.sqrt(f_arr+1)
S=G*H*R; S/=(S.max()+1e-30)
s_time=np.fft.irfft(S,N_syn)

fig,axs=plt.subplots(1,3,figsize=(14,4))
# Vowel chart
f1s=[v[0] for v in vowels.values()]; f2s=[v[1] for v in vowels.values()]
axs[0].scatter(f2s,f1s,s=80,zorder=5)
for (vow,(f1,f2)) in vowels.items():
    axs[0].annotate(f'/{vow}/',xy=(f2,f1),fontsize=10,ha='center',va='bottom',
                    xytext=(0,5),textcoords='offset points')
axs[0].invert_xaxis(); axs[0].invert_yaxis()
axs[0].set_xlabel('F₂ [Hz]'); axs[0].set_ylabel('F₁ [Hz]')
axs[0].set_title('Vowel Formant Chart (F₁ vs F₂)'); axs[0].grid(alpha=0.3)

# Spectrum of /ɑ/
f_plot=f_arr[:N_syn//4]; H_plot=H[:N_syn//4]
axs[1].semilogy(f_plot,H_plot/H_plot.max(),'C0',lw=2)
for fc in [f1a,f2a,f3a]:
    axs[1].axvline(fc,color='C1',ls='--',alpha=0.7)
axs[1].set_xlabel('Frequency [Hz]'); axs[1].set_ylabel('|H(f)|')
axs[1].set_title('/ɑ/ Vocal Tract Filter (730, 1090, 2500 Hz)')
axs[1].grid(alpha=0.3,which='both'); axs[1].set_xlim(0,4000)

# Trill simulation
t_trill=np.linspace(0,0.2,2000); f_trill=30
flutter=np.sin(2*np.pi*f_trill*t_trill)
# Modulate noise by trill gate function
gate=0.5*(1+flutter); gate=np.maximum(gate-0.4,0)
np.random.seed(42); noise=np.random.randn(len(t_trill))
trill_sig=noise*gate
axs[2].plot(t_trill*1000,trill_sig,'C0',lw=0.8,alpha=0.7)
axs[2].plot(t_trill*1000,gate,'C1',lw=2,label=f'Tongue gate ({f_trill}Hz)')
axs[2].set_xlabel('Time [ms]'); axs[2].set_ylabel('Amplitude')
axs[2].set_title('Rolled-R [r] Trill: 30Hz Tongue Flutter'); axs[2].legend()
axs[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_s10_vocal.png',dpi=100); plt.show()
print(f"Uniform tube formants: {[f'{f:.0f}' for f in F_n]} Hz")
print(f"Trill period = {1000/30:.1f}ms  Bernoulli ΔP≈225Pa≈2.3cmH₂O")
""")),

# ── Summary ───────────────────────────────────────────────────────────────
md(
"## Summary\n",
"\n",
"| § | Topic | Key Results |\n",
"|---|-------|-------------|\n",
"| 1 | Turbine blade | $w=U\\Delta C_\\theta=90$ kJ/kg, PR=0.838, NACA $y_t,y_c$ |\n",
"| 2 | Clutch + NMJ | $T=546$ Nm (UW), $t_{\\rm eng}=22$ ms, Hill $P_{\\max}=47.7$ W |\n",
"| 3 | Waveguide / Dipole | TE10 at 6.56 GHz, $v_p>c$, HPBW=78°, Friis −26dBm |\n",
"| 4 | Residue / Bessel | $\\oint=\\pi$, GF error $<10^{-16}$, J₀ zeros 2.405,5.52 |\n",
"| 5 | TS-DFT | $|E_{\\rm out}(t)|^2\\propto|\\tilde{E}(t/\\beta_2 L)|^2$, $M=100\\times$ BW |\n",
"| 6 | HPG axis | Goodwin $n=8$ oscillates, $K_d=0.5$ nM AR binding |\n",
"| 7 | SR / Photoelectric | $\\gamma(\\beta)$ table, Na $\\nu_0=5.49\\times10^{14}$ Hz |\n",
"| 8 | Covariant EM | $F^{\\mu\\nu}=\\partial^\\mu A^\\nu-\\partial^\\nu A^\\mu$, $\\partial_\\mu F^{\\mu\\nu}=\\mu_0 J^\\nu$ |\n",
"| 9 | SAR / CEM43 | $T_{\\max}=47.2°C$, CEM43=185 min, Pennes FD converged |\n",
"| 10 | Vocal tract | F₁=515,F₂=1544 Hz, trill $f=30$ Hz, Bernoulli 225 Pa |\n",
),

]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Saved {NB_PATH}")
