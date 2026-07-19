"""
Photon Interactions: Feynman Diagrams for Photonics

Every photonic device is a Feynman diagram at optical frequencies.
The electron is replaced by the atom; the photon is dressed by the medium.

SYNTAX GUIDE (0, 1, 2 vertex count):
  0-vertex:  Free propagation. Photon travels, phase accumulates.
             S-matrix element = 1.  |A_fi|^2 = 1.  No interaction.
  1-vertex:  Single absorption OR emission. Requires atom in excited/ground state.
             |A_fi|^2 ~ e^2 ~ alpha = 1/137.  Spontaneous emission.
  2-vertex:  Scattering (absorption then re-emission). Stimulated processes.
             |A_fi|^2 ~ e^4 ~ alpha^2 ~ 1/18000.  Raman, Compton, Mie.
  3-vertex:  Nonlinear optics. Three photons interact via chi^(2) or chi^(3).
             SHG, SPDC (parametric down-conversion), Kerr effect, THG.
  4-vertex:  Four-wave mixing (FWM). Cross-phase modulation (XPM).
             Dominant nonlinear impairment in WDM fiber systems.

TDGSA Feynman diagram:
  The Temporal Dispersion-assisted Gerchberg-Saxton Algorithm (this repo)
  can be written as a sequence of propagator legs:
    |E_in> --[free propagation in fiber, H(f)]--> |E_dispersed>
            --[intensity measurement, Pi]--> |E_measured>
            --[phase replacement, A]--> |E_corrected>
            --[inverse propagation, H(f)^{-1}]--> |E_recovered>
  Each arrow = a propagator. Each bracket = a vertex (interaction/projection).
  The GS LOOP is a Feynman path integral over all phase configurations.

Connection to circuit syntax:
  Free photon propagator = wire (no resistance, no interaction)
  Beamsplitter = 2-vertex interaction = 50:50 power split
  Phase modulator = e^{j phi} = complex gain = impedance in phasor domain
  Detector = projector Pi = measurement vertex = collapses state
  Amplifier (EDFA) = stimulated emission = gain > 1 vertex
  Fiber dispersion = momentum-dependent phase = H(k) = transfer function
"""
import math
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import io, base64

hbar = 1.0546e-34; c = 2.998e8; e_charge = 1.602e-19; eV = e_charge
alpha_fs = 1/137.036; kB = 1.381e-23; h = 6.626e-34


# ============================================================
# Einstein A and B Coefficients (1-vertex processes)
# ============================================================

def einstein_coefficients():
    """
    Einstein A and B coefficients: the 0,1,2 syntax of light-matter interaction.

    Two-level atom: ground |g>, excited |e>, energy gap hbar*omega_0.

    Three processes (all are Feynman diagrams):
      B_12 (absorption):         |g> + gamma -> |e>           [1 vertex, B coeff]
      B_21 (stimulated emission): |e> + gamma -> |g> + 2*gamma [2 vertex, gain]
      A_21 (spontaneous emission):|e>          -> |g> + gamma  [1 vertex, A coeff]

    Einstein relations (thermodynamic equilibrium -> Planck distribution):
      A_21 = (hbar*omega^3)/(pi^2*c^3) * B_21
      B_12 = B_21  (for non-degenerate levels; g1*B_12 = g2*B_21 in general)

    Laser condition: gain > loss
      G = N2*sigma*c > alpha_loss
      Population inversion: N2 > N1  (impossible in 2-level! Need 3 or 4 level)

    Rate equations:
      dN2/dt = R_pump - A21*N2 - B21*rho(omega)*N2 + B12*rho(omega)*N1
      dN1/dt = A21*N2 + B21*rho(omega)*N2 - B12*rho(omega)*N1 - R_decay

    EDFA (Erbium-doped fiber amplifier) uses Er^{3+} in glass:
      Lambda_signal = 1530-1565 nm (C-band), pump = 980 nm or 1480 nm.
      Gain: G = exp(sigma_e * (N2-N1) * L)
      THIS IS the amplifier in every long-haul fiber link carrying this repo's data.
    """
    # Spontaneous emission rate A_21 for hydrogen 2p -> 1s (Lyman alpha)
    omega_Ly_alpha = 2*np.pi*c / (121.6e-9)   # rad/s
    # A_21 = omega^3 * |d|^2 / (3*pi*eps0*hbar*c^3)
    # |d| ~ e*a0 for hydrogen
    a0 = 5.292e-11; eps0 = 8.854e-12
    d_matrix = e_charge * a0
    A_21_hydrogen = omega_Ly_alpha**3 * d_matrix**2 / (3*np.pi*eps0*hbar*c**3)
    tau_sp_hydrogen = 1/A_21_hydrogen   # spontaneous lifetime

    # EDFA parameters
    lambda_signal_nm = 1550; lambda_pump_nm = 980
    omega_signal = 2*np.pi*c / (1550e-9)
    omega_pump   = 2*np.pi*c / (980e-9)

    # Gain coefficient estimate
    sigma_emission = 4e-25   # m^2 (Er emission cross-section at 1550 nm)
    N_Er = 1e25              # m^-3 (Er doping density)
    N2_fraction = 0.6        # inversion fraction under pump
    N2 = N2_fraction * N_Er; N1 = (1-N2_fraction)*N_Er
    L_fiber = 10             # m
    G_dB = 10*np.log10(np.exp(sigma_emission*(N2-N1)*L_fiber))

    # Rate equation simulation: pump-on transient
    dt = 1e-9; t_max = 100e-6; N = int(t_max/dt)
    N2_arr = np.zeros(N); N1_arr = np.zeros(N)
    N2_arr[0] = 0; N1_arr[0] = N_Er
    A21 = 1/10e-3   # 10 ms Er lifetime
    R_pump = 0.8*A21*N_Er   # pump rate
    for i in range(1,N):
        dN2 = R_pump*N1_arr[i-1]/N_Er - A21*N2_arr[i-1]
        N2_arr[i] = N2_arr[i-1] + dN2*dt
        N1_arr[i] = N_Er - N2_arr[i]

    t_us = np.linspace(0, t_max*1e6, N)

    return {
        'hydrogen': {
            'omega_Ly_alpha': omega_Ly_alpha,
            'A21_per_s': A_21_hydrogen,
            'tau_sp_ns': tau_sp_hydrogen*1e9,
            'lambda_nm': 121.6,
        },
        'EDFA': {
            'lambda_signal_nm': lambda_signal_nm,
            'lambda_pump_nm': lambda_pump_nm,
            'sigma_emission_m2': sigma_emission,
            'N_Er_per_m3': N_Er,
            'gain_dB_10m': G_dB,
            'Ein_A_coeff_relation': 'A21 = (hbar*omega^3/(pi^2*c^3)) * B21',
        },
        'rate_eq': {
            't_us': t_us[::100].tolist(),
            'N2_fraction': (N2_arr[::100]/N_Er).tolist(),
            'N1_fraction': (N1_arr[::100]/N_Er).tolist(),
            'steady_state_inversion': float(N2_arr[-1]/N_Er),
        },
        'laser_condition': {
            'threshold': 'G = N2*sigma*c > alpha_loss*c',
            'two_level_impossible': 'In 2-level: max N2=N1=N/2 -> no net gain. Need 3 or 4 level.',
            'EDFA_level': '3-level at 1530nm, 4-level at 1550nm (better inversion)',
        },
        'vertex_count': {
            'absorption': '1 vertex, coupling ~ e, |A|^2 ~ alpha = 1/137',
            'stimulated': '2 vertices, coherent, SAME phase as input photon -> GAIN',
            'spontaneous': '1 vertex, random phase -> ASE noise in EDFA',
        },
    }


# ============================================================
# Nonlinear Optics: 2, 3, 4-vertex diagrams
# ============================================================

def nonlinear_photon_interactions():
    """
    Nonlinear optics as Feynman diagrams with vertex count.

    chi^(2) materials (non-centrosymmetric: GaAs, LiNbO3, BBO):
      SHG (3-vertex): 2*gamma(omega) -> gamma(2*omega)
        Two pump photons annihilated, one SH photon created.
        Phase matching: k(2*omega) = 2*k(omega)  [momentum conservation]
      SPDC (3-vertex): gamma(2*omega) -> gamma_s(omega_s) + gamma_i(omega_i)
        One pump photon -> signal + idler.  omega_p = omega_s + omega_i.
        This creates ENTANGLED photon pairs! (Hong-Ou-Mandel effect)
        Used in: quantum key distribution, Bell test experiments.

    chi^(3) materials (all materials, including glass fiber):
      SPM (4-vertex): gamma -> gamma, phase shift proportional to I
        Delta_phi = gamma*P*L  [rad], gamma = nonlinear coefficient [1/(W*m)]
        Causes spectral broadening (self-phase modulation)
      Kerr effect: n = n_0 + n_2*I  (intensity-dependent refractive index)
        n_2(silica) = 2.6e-20 m^2/W
      XPM (4-vertex): cross-phase modulation between channels in WDM
      FWM (4-vertex): omega_1+omega_2 -> omega_3+omega_4

    TDGSA connection:
      SPM causes nonlinear chirp on the optical pulse.
      If SPM is significant: H(f) = exp(j*pi*D*f^2) * exp(j*gamma*P*L)
      The GS algorithm must account for this; otherwise phase retrieval fails.
      Nonlinear phase = extra constraint = reduces degrees of freedom = helps GS converge.
    """
    # SPM: nonlinear phase shift in SMF-28
    gamma_fiber = 1.3e-3   # 1/(W*m), SMF-28 at 1550nm
    P_arr = np.logspace(-3, 1, 200)   # 1 mW to 10 W
    L_arr = np.array([1e3, 10e3, 100e3])   # 1km, 10km, 100km
    phi_NL = {}
    for L in L_arr:
        phi_NL[f'L={L/1e3:.0f}km'] = (gamma_fiber * P_arr * L).tolist()

    # Phase matching for SHG: BBO crystal
    n_e_omega = 1.6551; n_o_2omega = 1.6551   # at phase matching angle
    lambda_pump = 800e-9   # Ti:sapphire
    L_crystal = 1e-3       # 1mm BBO
    # Bandwidth of SHG (phase matching bandwidth)
    Delta_k_per_nm = 2*np.pi/lambda_pump * abs(1/c*(n_e_omega - n_o_2omega))
    L_coherence_mm = np.pi / (Delta_k_per_nm * 1e9) * 1e3 if Delta_k_per_nm > 0 else 1.0

    # SPDC: entangled photon pairs
    lambda_pump_SPDC = 405e-9   # violet pump
    lambda_signal = 810e-9; lambda_idler = 810e-9   # degenerate
    E_pump = h*c/lambda_pump_SPDC / eV
    E_signal = h*c/lambda_signal / eV
    E_conservation = abs(E_pump - E_signal - (h*c/lambda_idler/eV)) < 0.001

    # Kerr: n2 induced phase
    n2 = 2.6e-20   # m^2/W silica
    I_arr = np.logspace(9, 15, 200)   # W/m^2
    delta_n = n2 * I_arr
    lambda0 = 1550e-9
    phi_Kerr = 2*np.pi/lambda0 * delta_n * 1e-3   # 1mm medium

    return {
        'SPM': {
            'gamma_fiber_per_W_km': gamma_fiber*1e3,
            'P_W': P_arr.tolist(),
            'phi_NL_rad': phi_NL,
            'L_NL_formula': 'L_NL = 1/(gamma*P)',
            'EE_analogy': 'SPM = voltage-dependent capacitor (varactor). phi_NL = Q(V).',
        },
        'SHG': {
            'lambda_pump_nm': 800,
            'lambda_SH_nm': 400,
            'crystal': 'BBO (beta-barium borate)',
            'phase_matching': 'n_e(omega)*cos(theta) = n_o(2*omega)',
            'vertex_count': 3,
        },
        'SPDC': {
            'lambda_pump_nm': 405,
            'lambda_signal_idler_nm': 810,
            'energy_conserved': bool(E_conservation),
            'entanglement': 'Signal and idler are frequency-entangled (time-energy entanglement)',
            'HOM_effect': 'Hong-Ou-Mandel: two indistinguishable photons always exit same port of beamsplitter',
            'QKD_use': 'E91 protocol: Bell test on entangled pairs from SPDC',
        },
        'Kerr': {
            'n2_silica': n2,
            'I_W_per_m2': I_arr.tolist(),
            'phi_Kerr_rad': phi_Kerr.tolist(),
            'vertex_count': 4,
            'TDGSA': 'Kerr adds j*gamma*P*L to dispersion phase -> GS must include this in H(f)',
        },
    }


# ============================================================
# Photon Propagator: Dressed Photon in Medium
# ============================================================

def photon_propagator_in_medium():
    """
    The dressed photon propagator: free photon + interaction with medium.

    Free photon propagator (vacuum):
      D_F(k,omega) = -g_{mu nu} / (k^2 - omega^2/c^2 + j*eps)
      = phase velocity = c everywhere

    Dressed propagator (photon + medium):
      D*(k,omega) = D_F(k,omega) * [1 + Pi(k,omega)*D_F + Pi^2*D_F^2 + ...]
                  = D_F / (1 - Pi*D_F)
      Pi(k,omega) = self-energy = sum of all 1-particle-irreducible diagrams

    In photonics: this IS the Sellmeier equation!
      n^2(omega) = 1 + sum_j B_j*omega_j^2 / (omega_j^2 - omega^2)
      Each term = one resonance = one Lorentz oscillator = one loop diagram.

    Dispersion H(f) = exp(j*phi(f)):
      phi(f) = (2*pi/lambda_0)*n(f)*L
      Taylor expand: phi(f) = phi_0 + phi'*f + (1/2)*phi''*f^2 + ...
      phi_0 = carrier phase (GS ignores, global phase)
      phi'  = group delay tau_g = L/v_g
      phi'' = GVD = D*L = pi*D*(lambda_0)^2/c * L  [ps^2]
      -> H(f) = exp(j*pi*D*f^2)  [quadratic phase = this repo]

    0,1,2 SYNTAX FOR THE PROPAGATOR:
      0-order Taylor: phi_0. Phase offset. Shifts carrier. GS ignores it.
      1st-order:      phi'. Group delay. Shifts pulse in time. GS tracks it.
      2nd-order:      phi''. GVD. Chirps pulse. GS RECOVERS THIS.
      3rd-order:      phi'''. TOD. Asymmetric distortion.
      4th-order:      phi''''. FOD. Small correction.
    """
    # Sellmeier for fused silica (Malitson 1965)
    B = [0.6961663, 0.4079426, 0.8974794]
    C = [(0.0684043e-6)**2, (0.1162414e-6)**2, (9.896161e-6)**2]   # m^2

    lambda_nm = np.linspace(400, 2400, 1000)
    lambda_m = lambda_nm * 1e-9
    n2_sell = 1.0
    for Bj, Cj in zip(B,C):
        n2_sell = n2_sell + Bj * lambda_m**2 / (lambda_m**2 - Cj)
    n_sell = np.sqrt(n2_sell)

    # GVD: D = d(1/v_g)/d(lambda) = -(lambda/c)*d^2n/d(lambda)^2
    # Numerical second derivative of n
    dlam = lambda_m[1]-lambda_m[0]
    d2n_dlam2 = np.gradient(np.gradient(n_sell, dlam), dlam)
    D_ps_nm_km = -(lambda_m/c)*d2n_dlam2 * 1e3 * 1e12 * 1e9   # ps/(nm*km)

    # ZDW (zero-dispersion wavelength) of silica
    idx_zdw = np.argmin(np.abs(D_ps_nm_km))
    zdw_nm = lambda_nm[idx_zdw]

    # H(f) for a specific fiber link
    D_param = +17.0   # ps/(nm*km) at 1550 nm [SMF-28] anomalous -> D>0 -> beta2<0
    L_km = 100
    f_arr = np.linspace(-500e9, 500e9, 2000)   # Hz offset from carrier
    lambda0_m = 1550e-9
    beta2 = -(D_param*1e-12)*(lambda0_m**2)/(2*np.pi*c)*1e3   # s^2/m
    H_f = np.exp(1j * np.pi * beta2*(2*np.pi)**2 * f_arr**2 * L_km*1e3)

    # Taylor coefficients of phi(f)
    phi_arr = np.angle(H_f)
    # phi = (pi*beta2*(2*pi)^2*L)*f^2 -> coefficient
    beta2_eff = beta2*(2*np.pi)**2*L_km*1e3
    phi_2nd_coeff = beta2_eff   # s^2

    return {
        'Sellmeier': {
            'lambda_nm': lambda_nm.tolist(),
            'n': n_sell.tolist(),
            'D_ps_nm_km': D_ps_nm_km.tolist(),
            'ZDW_nm': float(zdw_nm),
            'n_at_1550nm': float(n_sell[np.argmin(np.abs(lambda_nm-1550))]),
        },
        'H_f': {
            'f_GHz': (f_arr/1e9).tolist(),
            'H_mag': np.abs(H_f).tolist(),
            'H_phase_rad': np.angle(H_f).tolist(),
            'D_ps_nm_km': D_param,
            'L_km': L_km,
            'beta2_s2_m': float(beta2),
            'formula': 'H(f) = exp(j*pi*beta2*(2*pi)^2*L*f^2)',
        },
        'Taylor_syntax': {
            '0th_order': 'phi_0: carrier phase (global), GS invariant',
            '1st_order': 'phi_1 = tau_g: group delay (arrival time)',
            '2nd_order': 'phi_2 = beta2*L: GVD = THIS REPO, chirps the pulse',
            '3rd_order': 'phi_3: TOD, asymmetric pulse distortion',
            'vertex_count': {
                '0': 'free propagation, phi=const per freq',
                '1': 'linear phase = time shift = 1 interaction with medium per photon',
                '2': 'quadratic phase = H(f)=exp(j*pi*D*f^2) = GVD = THIS REPO',
            },
        },
    }


# ============================================================
# Matplotlib Feynman Diagram for TDGSA
# ============================================================

def feynman_diagram_tdgsa(save_path=None):
    """
    Draw the TDGSA Feynman-style diagram using matplotlib.

    Represents the GS loop as a sequence of propagator legs and vertices.

    Diagram layout (horizontal = time/iteration):
      |E_in> ---[H(f)]---> |E_disp> ---[Pi_meas]---> |E_proj>
                                                          |
      |E_out> <--[H^{-1}(f)]--- |E_corr> <--[A_repl]---+

    Vertices (circles):
      H(f):    dispersion propagator, e^{j*pi*D*f^2}
      Pi_meas: intensity measurement, replaces amplitude
      A_repl:  amplitude replacement in Fourier domain
      H^{-1}:  inverse propagation
    """
    fig, ax = plt.subplots(figsize=(9, 4), facecolor='#0f0f18')
    ax.set_facecolor('#0f0f18')
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-1.5, 2.5)
    ax.axis('off')

    # Color scheme
    C_line   = '#5588ff'
    C_vertex = '#7c6fff'
    C_meas   = '#f0c060'
    C_out    = '#7cf0b0'
    C_text   = '#c0c0e8'
    C_label  = '#8888cc'

    def draw_line(x0,y0,x1,y1,color=C_line,lw=2,style='-'):
        ax.plot([x0,x1],[y0,y1],color=color,lw=lw,ls=style,solid_capstyle='round')

    def draw_arrow(x0,y0,x1,y1,color=C_line,lw=2):
        ax.annotate('',xy=(x1,y1),xytext=(x0,y0),
            arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                           mutation_scale=14))

    def draw_vertex(x,y,label,sublabel='',color=C_vertex,r=0.28):
        circle = plt.Circle((x,y),r,color=color,zorder=5,alpha=0.9)
        ax.add_patch(circle)
        ax.text(x,y+0.01,label,ha='center',va='center',color='white',
                fontsize=8,fontweight='bold',zorder=6,fontfamily='monospace')
        if sublabel:
            ax.text(x,y-r-0.22,sublabel,ha='center',va='top',color=C_label,
                    fontsize=7,fontfamily='monospace')

    def draw_state(x,y,label,color=C_text):
        ax.text(x,y,label,ha='center',va='center',color=color,
                fontsize=8.5,fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3',facecolor='#1a1a2a',
                          edgecolor='#33334a',linewidth=0.8))

    # ── Top row: forward pass ──
    y0 = 1.3
    draw_state(0, y0, '|E_in⟩', C_line)
    draw_arrow(0.55,y0,1.45,y0,C_line)
    draw_vertex(1.8,y0,'H(f)','dispersion\nH=e^{jπDf²}',C_vertex)
    draw_arrow(2.15,y0,3.05,y0,C_line)
    draw_state(3.5,y0,'|E_disp⟩',C_line)
    draw_arrow(4.05,y0,4.95,y0,C_line)
    draw_vertex(5.3,y0,'Π','intensity\nmeasure',C_meas)
    draw_arrow(5.65,y0,6.55,y0,C_line)
    draw_state(7.0,y0,'|E_proj⟩',C_meas)

    # ── Vertical connector (right side) ──
    draw_arrow(7.0,y0-0.18,7.0,-0.3,C_meas,1.5)

    # ── Bottom row: backward pass ──
    y1 = -0.7
    draw_state(7.0,y1,'|E_corr⟩',C_out)
    draw_vertex(5.3,y1,'A','amplitude\nreplace',C_out)
    draw_arrow(6.55,y1,5.65,y1,C_out)
    draw_arrow(4.95,y1,4.05,y1,C_out)
    draw_state(3.5,y1,'|E_amp⟩',C_out)
    draw_arrow(3.05,y1,2.15,y1,C_out)
    draw_vertex(1.8,y1,'H⁻¹','inverse\nH^{-1}=e^{-jπDf²}',C_vertex)
    draw_arrow(1.45,y1,0.55,y1,C_out)
    draw_state(0,y1,'|E_out⟩',C_out)

    # ── Convergence arrow (loop) ──
    ax.annotate('',xy=(0.3,y0-0.22),xytext=(0.3,y1+0.22),
        arrowprops=dict(arrowstyle='->',color='#6060a0',lw=1,
                       connectionstyle='arc3,rad=-0.5'))
    ax.text(-0.3,0.3,'iterate\nuntil\nconverge',ha='center',va='center',
            color='#6060a0',fontsize=7,fontfamily='monospace')

    # ── Vertex count legend ──
    legend_items = [
        (C_vertex,'propagator vertex (H(f), H⁻¹)'),
        (C_meas,  'measurement vertex (Π: amplitude from data)'),
        (C_out,   'update vertex (A: phase from estimate)'),
    ]
    for i,(color,label) in enumerate(legend_items):
        ax.plot([8.2],[2.0-i*0.42],'o',color=color,markersize=7,zorder=5)
        ax.text(8.45,2.0-i*0.42,label,va='center',color=C_label,
                fontsize=7,fontfamily='monospace')

    # ── Title + equation ──
    ax.text(5.0,2.3,'TDGSA  —  Temporal Dispersion-assisted Gerchberg-Saxton',
            ha='center',va='center',color='#a0a0d0',fontsize=9,
            fontfamily='monospace',fontweight='bold')
    ax.text(5.0,-1.35,
            'H(f) = exp(jπDf²)     Π: |Ê|→√I_meas·e^{j∠Ê}     A: F⁻¹[√I_in·e^{j∠F{E}}]',
            ha='center',va='center',color='#6060a0',fontsize=7.5,fontfamily='monospace')

    plt.tight_layout(pad=0.4)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight',
                    facecolor='#0f0f18')
        plt.close()
        return save_path
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                    facecolor='#0f0f18')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()


# ============================================================
# Capacitance-Voltage (C-V): circuit + photonics
# ============================================================

def capacitance_voltage():
    """
    C-V relationship: phasor/circuit view AND photonic modulator view.

    CIRCUIT SYNTAX (phasor domain):
      Z_C = 1/(j*omega*C)  [Ohms]  -- impedance is a phasor
      I_C = C * dV/dt              -- current leads voltage by 90°
      In phasor: I = j*omega*C*V   -- multiplication by j = 90° phase shift
      Energy: U = (1/2)*C*V^2      -- stored in electric field

    VOLTAGE-DEPENDENT CAPACITANCE (varactor diode):
      C(V) = C0 / (1 - V/V_bi)^m
      m = 1/2 (abrupt junction), m = 1/3 (linearly graded)
      V_bi = built-in voltage (~0.7 V for Si)
      VCO: C(V) tunes resonance omega_0 = 1/sqrt(L*C(V))
      Same math: frequency-dependent permittivity eps(omega) = C(omega)/C_0

    PHOTONIC MODULATOR (electro-optic effect = V -> phi):
      Pockels effect (chi^2): Delta_n = n^3*r*E/(2)
        r = electro-optic coefficient [m/V]
        E = V/d  (voltage V across gap d)
        Delta_phi = (2*pi/lambda) * Delta_n * L = (pi*V)/(V_pi)
        V_pi = lambda*d/(n^3*r*L)  [half-wave voltage]
      MZI modulator: E_out = E_in * cos(Delta_phi/2)
        At V=V_pi/2: Delta_phi=pi/2 -> quadrature point
        At V=V_pi:   Delta_phi=pi   -> NULL (180° phase diff = zero output)

    EE-photonics bridge:
      LC resonator:   omega_0 = 1/sqrt(LC)    -> f changes with C(V)
      MZI modulator:  phi_out = pi*V/V_pi     -> same linear V->phi law
      Phasor: V*exp(j*omega*t) * exp(j*Delta_phi) = capacitor + phase modulator
    """
    # Varactor C(V) -- abrupt junction
    V_bi = 0.75; C0 = 10e-12   # 10 pF at V=0
    V_arr = np.linspace(-5, 0.5, 400)
    C_V = C0 / np.sqrt(np.maximum(1 - V_arr/V_bi, 0.01))
    # VCO tuning range
    L_tank = 1e-9   # 1 nH
    f0_arr = 1/(2*np.pi*np.sqrt(L_tank * C_V)) / 1e9   # GHz

    # MZI modulator
    lambda0_nm = 1550; n = 3.47; r_LiNbO3 = 30e-12; d = 5e-6; L_mod = 1e-2
    V_pi = lambda0_nm*1e-9 * d / (n**3 * r_LiNbO3 * L_mod)
    V_mod = np.linspace(-V_pi, V_pi, 400)
    Delta_phi = np.pi * V_mod / V_pi
    P_out_norm = np.cos(Delta_phi/2)**2   # power transfer function

    # Impedance phasor
    omega_arr = np.logspace(6, 12, 400)   # 1 MHz to 1 THz
    Z_C_ohm = 1/(1j*omega_arr*C0)
    Z_C_mag = np.abs(Z_C_ohm)
    Z_C_phase = np.angle(Z_C_ohm, deg=True)

    return {
        'varactor': {
            'V_arr': V_arr.tolist(),
            'C_pF': (C_V*1e12).tolist(),
            'f0_GHz': f0_arr.tolist(),
            'tuning_range_GHz': float(max(f0_arr)-min(f0_arr)),
        },
        'MZI_modulator': {
            'V_pi_volts': float(V_pi),
            'V_mod': V_mod.tolist(),
            'P_out_normalized': P_out_norm.tolist(),
            'null_at_V_pi': True,
            'quadrature_at_Vpi_over_2': True,
            'photon_connection': 'At V=V_pi: Delta_phi=pi -> 180 deg -> NULL = destructive',
        },
        'impedance_phasor': {
            'omega': omega_arr.tolist(),
            'Z_mag_ohm': Z_C_mag.tolist(),
            'Z_phase_deg': Z_C_phase.tolist(),
            'angle_is_minus90': True,
        },
        'syntax': {
            '0': 'V=0: no modulation, phi=0, full transmission',
            '1': 'V=V_pi/2: phi=pi/2, quadrature, linear regime, IQ modulator',
            '2': 'V=V_pi: phi=pi, 180 deg, NULL output = carrier suppressed',
        },
    }


# ============================================================
# Complex Vector-Valued Line Integrals → Electrodynamics
# ============================================================

def complex_line_integrals_to_electrodynamics():
    """
    Integration of complex vector-valued functions → Maxwell's equations.

    LINE INTEGRAL of vector field F along curve C:
      integral_C F . dl  [work done by field along path]

    Stokes' theorem:  integral_C F . dl = integral_S (curl F) . dA
      Left: line integral around closed loop C
      Right: surface integral of curl over any surface S bounded by C

    This IS Faraday's law:
      EMF = integral_C E . dl = -d/dt integral_S B . dA
      Stokes: integral_C E . dl = integral_S (curl E) . dA
      -> curl E = -dB/dt  [Faraday, one of Maxwell's 4 equations]

    Divergence theorem: integral_V div(F) dV = integral_S F . dA
      -> div(E) = rho/eps0  [Gauss, another Maxwell equation]
      -> div(B) = 0         [no magnetic monopoles]

    COMPLEX VECTOR FIELD (photonics / phasor domain):
      E(r,t) = Re[E_phasor(r) * exp(j*omega*t)]
      All four Maxwell equations become algebraic in phasor domain:
        curl E_phasor = -j*omega*mu0*H_phasor     [Faraday]
        curl H_phasor = +j*omega*eps*E_phasor + J  [Ampere-Maxwell]
        div  E_phasor = rho/eps                    [Gauss E]
        div  H_phasor = 0                          [Gauss B]
      COMPLEX curl = j*omega*mu -> 90 degree rotation in phasor diagram.

    Poynting vector (complex):
      S = (1/2) E_phasor x H_phasor*   [complex power density]
      Re[S] = time-averaged power flow [W/m^2]
      Im[S] = reactive power (oscillates, no net transport)
      This IS the photon flux direction.

    NUMERICAL EXAMPLE: line integral of E field around rectangular loop
      If B is changing inside: EMF = -dPhi_B/dt  (Faraday, measurable!)
      If B = 0 inside:         EMF = 0            (curl E = 0 -> conservative)
    """
    # Numerical Stokes theorem check
    # E = [-y, x, 0] (circular field, curl E = 2*z_hat)
    # Integral around unit circle should = 2*pi (area * curl_z)
    N_path = 1000
    theta = np.linspace(0, 2*np.pi, N_path, endpoint=False)
    dtheta = 2*np.pi/N_path
    # Path: circle r=1
    x = np.cos(theta); y = np.sin(theta)
    dx = -np.sin(theta)*dtheta; dy = np.cos(theta)*dtheta
    # Field E = [-y, x]
    Ex = -y; Ey = x
    line_integral = float(np.sum(Ex*dx + Ey*dy))
    surface_integral = 2*np.pi   # curl_z = 2, area = pi, result = 2*pi

    # Phasor curl example: plane wave
    k0 = 2*np.pi/1550e-9   # wavenumber
    n_glass = 1.5; omega = k0*c
    # E = E0 * exp(j*k*z) x_hat, H = (E0/Z0) * exp(j*k*z) y_hat
    Z0 = np.sqrt(4*np.pi*1e-7 / 8.854e-12)   # impedance of free space [Ohm]
    Z_medium = Z0/n_glass
    E0 = 1.0
    z_arr = np.linspace(0, 5e-6, 400)
    E_z = E0 * np.exp(1j*k0*n_glass*z_arr)
    H_z = (E0/Z_medium) * np.exp(1j*k0*n_glass*z_arr)
    S_complex = 0.5 * E_z * np.conj(H_z)

    # Verify curl E = -j*omega*mu0*H in phasor domain
    # d(E_x)/dz = j*k0*n*E_x = E_z (numerically)
    dE_dz = np.gradient(E_z, z_arr)
    curl_E_y = dE_dz   # d(Ex)/dz = -curl_E_y component
    mu0 = 4*np.pi*1e-7
    expected_curl = -1j*omega*mu0*H_z
    # Magnitude check: sign depends on H field orientation convention
    faraday_check = np.allclose(np.abs(curl_E_y), np.abs(expected_curl), rtol=1e-3)

    return {
        'Stokes_check': {
            'line_integral': float(line_integral),
            'surface_integral': float(surface_integral),
            'match': abs(line_integral - surface_integral) < 0.01,
        },
        'Maxwell_phasor': {
            'Faraday':   'curl E = -j*omega*mu0*H',
            'Ampere':    'curl H = +j*omega*eps*E + J',
            'Gauss_E':   'div E  = rho/eps',
            'Gauss_B':   'div H  = 0',
            'key': 'All 4 Maxwell equations become ALGEBRAIC in phasor domain (j*omega replaces d/dt)',
        },
        'plane_wave': {
            'z_um': (z_arr*1e6).tolist(),
            'E_real': np.real(E_z).tolist(),
            'E_imag': np.imag(E_z).tolist(),
            'S_real_W_m2': np.real(S_complex).tolist(),
            'S_imag_W_m2': np.imag(S_complex).tolist(),
            'Faraday_satisfied': bool(faraday_check),
        },
        'integration_syntax': {
            'line_integral': 'integral F.dl: work done = EMF = -dPhi_B/dt',
            'surface_integral': 'integral (curl F).dA: flux of curl = enclosed EMF',
            'volume_integral': 'integral (div F) dV = surface flux (divergence theorem)',
            'complex_valued': 'Complex F: split into Re and Im, integrate each separately',
            'Stokes_to_Faraday': 'Stokes theorem applied to E field = Faraday law',
        },
        'mathematical_maturity': (
            'Mathematical maturity = seeing the same structure in different clothes:\n'
            '  Stokes theorem         = Faraday law (physics)\n'
            '  Cauchy integral formula = residue theorem = pole extraction = partial fractions\n'
            '  Green function          = impulse response = propagator = transfer function\n'
            '  Fourier transform       = eigenfunction expansion = diagonal operator\n'
            '  Gradient descent        = steepest descent = saddle-point approximation\n'
            '  GS algorithm            = Feynman path integral + projection onto constraints\n'
            'When you see the same math in a new context, you have maturity.'
        ),
    }


def demo():
    print("=== PHOTON INTERACTIONS: FEYNMAN DIAGRAMS FOR PHOTONICS ===\n")

    print("--- Einstein A/B Coefficients ---")
    ein = einstein_coefficients()
    print(f"  H Lyman alpha A21 = {ein['hydrogen']['A21_per_s']:.3e} /s")
    print(f"  tau_sp = {ein['hydrogen']['tau_sp_ns']:.2f} ns")
    print(f"  EDFA gain (10m): {ein['EDFA']['gain_dB_10m']:.1f} dB")
    print(f"  Vertex: {ein['vertex_count']['stimulated']}")

    print("\n--- Nonlinear Interactions ---")
    nl = nonlinear_photon_interactions()
    print(f"  gamma(SMF-28) = {nl['SPM']['gamma_fiber_per_W_km']:.3f} /(W*km)")
    print(f"  SPDC energy conserved: {nl['SPDC']['energy_conserved']}")
    print(f"  Kerr vertex count: {nl['Kerr']['vertex_count']}")

    print("\n--- Photon Propagator / H(f) ---")
    pp = photon_propagator_in_medium()
    print(f"  ZDW of silica: {pp['Sellmeier']['ZDW_nm']:.1f} nm")
    print(f"  n(1550nm) = {pp['Sellmeier']['n_at_1550nm']:.4f}")
    print(f"  H(f) formula: {pp['H_f']['formula']}")
    for k,v in pp['Taylor_syntax']['vertex_count'].items():
        print(f"    vertex {k}: {v}")

    print("\n--- TDGSA Feynman Diagram ---")
    from pathlib import Path

    outfile = Path(__file__).parent / "tdgsa_feynman.png"
    path = feynman_diagram_tdgsa(str(outfile))
    print(f"Diagram saved to: {outfile.resolve()}")
    print(f"  Diagram saved to: {path if isinstance(path,str) and path.endswith('.png') else 'base64 PNG'}")

    print("\n--- Capacitance-Voltage ---")
    cv = capacitance_voltage()
    print(f"  V_pi (LiNbO3 MZI, 1cm): {cv['MZI_modulator']['V_pi_volts']:.2f} V")
    print(f"  Varactor tuning: {cv['varactor']['tuning_range_GHz']:.2f} GHz")
    for k,v in cv['syntax'].items():
        print(f"  {k}: {v}")

    print("\n--- Complex Line Integrals -> Maxwell ---")
    li = complex_line_integrals_to_electrodynamics()
    print(f"  Stokes check: line={li['Stokes_check']['line_integral']:.4f}, "
          f"surface={li['Stokes_check']['surface_integral']:.4f}, "
          f"match={li['Stokes_check']['match']}")
    print(f"  Faraday satisfied: {li['plane_wave']['Faraday_satisfied']}")
    for k,v in li['Maxwell_phasor'].items():
        if k != 'key': print(f"  {k}: {v}")
    print(f"  Key: {li['Maxwell_phasor']['key']}")

    print("\n=== PHOTON INTERACTIONS COMPLETE ===")


if __name__ == '__main__':
    demo()
