"""STEAM: Serial Time-Encoded Amplified Microscopy -- Jalali lab flagship.

THE CORE IDEA (sub-second imaging of humans / biological phenomena):
  Normal camera: 10^3 fps.  STEAM: 36 * 10^6 fps (36 million frames/second).
  Fastest biological events: action potential (1 ms), rogue waves (< 1 us),
  cancer cell detection in blood flow (< 100 us at 10 m/s flow).

HOW IT WORKS (this IS dispersion-assisted phase recovery):
  1. Broadband pulse -> spatial encoding via diffraction grating (lambda -> x)
  2. Pulse hits sample -> each wavelength carries amplitude of one pixel column
  3. Dispersion (D >> 0) -> H(f) = exp(i*pi*D*f^2) -> time-stretches pulse
     NOW: time axis = spatial axis. Each time sample = one pixel.
  4. ADC digitizes the stretched pulse.  ADC clock = imaging clock.
  5. Phase is LOST (only |E|^2 measured). -> GS phase retrieval recovers it.

CLOCK SIGNAL:
  The ADC sampling rate f_s determines spatial resolution.
  f_s = B / (D * lambda_0)   where B = optical bandwidth [Hz]
  At 36 Mfps, f_s ~ 10 GHz ADC (one of fastest available).
  Clock jitter -> timing noise -> phase error in retrieved field.
  This is WHY clock quality matters for phase retrieval accuracy.

GRIFFITHS CONNECTION:
  Ch 9:  E(x,t) = integral A(f) * exp(i*2*pi*f*t - i*k*x) df  (wave packet)
  Dispersion: k(f) = k0 + k1*(f-f0) + (1/2)*k2*(f-f0)^2
  -> H(f) = exp(i*pi*k2*L*f^2) = exp(i*pi*D*f^2)   [Taylor to 2nd order]
  The GROUP DELAY tau(f) = k2*L*f  maps frequency to time.  This is the
  integral of the dispersion relation -- Griffiths Ch 9 eq 9.150.

NONLINEAR PHYSICS EXTENSION:
  At high pulse power: NLSE (dgs/nlse.py) adds Kerr term:
    i*dA/dz = (beta_2/2)*d^2A/dt^2 - gamma*|A|^2*A
  -> self-phase modulation (SPM) warps H(f) -> nonlinear phase
  -> phase retrieval must account for nonlinear term (Project 5 extension)

UCLA AI CONNECTION:
  - STEAM generates 10 GB/s data -> real-time AI classification required
  - NN replaces GS phase retrieval at 10^6 cells/second
  - This repo's dgs/nn_spectral_regression.py is a prototype of that NN

Run: py -3.13 -c "from dgs.steam_imaging import demo; demo()"
"""
import numpy as np

C_LIGHT = 2.998e8   # m/s
H_PLANCK = 6.626e-34


# ── 1. Dispersive time-stretch (the STEAM forward model) ──────────────────

def time_stretch_pulse(E_in, f_arr, D_ps2, lambda0_nm=1550.0):
    """Apply dispersion H(f) = exp(i*pi*D*f^2) to an input field E_in.

    This IS the same H(f) as dgs/gs_core.py -- STEAM and GS share the physics.

    D_ps2: group delay dispersion in ps^2 (negative = anomalous)
    f_arr: frequency array relative to carrier [Hz] or normalized
    Returns: E_out (complex), I_out = |E_out|^2 (what the ADC measures)

    The time-stretched intensity I_out(t) maps to space: t -> x via
      x = v_flow * t   (for flowing cell)
    or x = lambda_0 * D * f  (for spectral encoding)
    """
    f = np.asarray(f_arr, float)
    E = np.asarray(E_in, complex)
    if E.shape != f.shape:
        raise ValueError("E_in and f_arr must have same shape")
    # transfer function in frequency domain
    H = np.exp(1j * np.pi * D_ps2 * f**2)
    # dispersed field
    E_f = np.fft.fft(E)
    E_f_disp = E_f * H
    E_out = np.fft.ifft(E_f_disp)
    I_out = np.abs(E_out)**2
    return {"E_out": E_out, "I_out": I_out, "H": H, "D_ps2": D_ps2}


def adc_clock_requirements(optical_bandwidth_nm, D_ps_per_nm, lambda0_nm=1550.0,
                           fps_target=36e6):
    """Compute ADC clock rate needed for STEAM at a given frame rate.

    STEAM clock signal derivation:
      Group delay per unit frequency: tau(lambda) = D * lambda  [ps/nm * nm = ps]
      Total time window: T_window = D * B_lambda  [ps]
      Spatial resolution: delta_x = lambda0^2 / (n * B_lambda)  [m]
      To resolve N_pixels per frame: f_ADC = N_pixels / T_window

    At 36 Mfps with 1000 pixels/frame:
      f_ADC = 36e6 * 1000 = 36 GHz (ultra-high speed ADC)

    CLOCK JITTER effect:
      Phase error: delta_phi = 2*pi*f0*delta_t  (Griffiths: phase = omega*t)
      For f0=193 THz, delta_t=100 fs jitter: delta_phi = 0.12 rad -> bad
      Fix: optical clock derived from same laser (dgs/adc.py)
    """
    if D_ps_per_nm <= 0 or optical_bandwidth_nm <= 0:
        raise ValueError("D and bandwidth must be positive")
    T_window_ps = D_ps_per_nm * optical_bandwidth_nm
    f0_Hz = C_LIGHT / (lambda0_nm * 1e-9)
    N_pixels_per_frame = 1000   # typical STEAM
    f_ADC_GHz = (fps_target * N_pixels_per_frame) / (1e9 / T_window_ps * 1e12)
    # simpler: f_ADC = N_pix * fps
    f_ADC_direct_GHz = N_pixels_per_frame * fps_target / 1e9
    jitter_fs_limit = 1e15 / (2 * np.pi * f0_Hz * 0.1)   # for 0.1 rad phase error
    return {
        "T_window_ps": round(T_window_ps, 2),
        "f_ADC_GHz_needed": round(f_ADC_direct_GHz, 1),
        "f0_THz": round(f0_Hz / 1e12, 2),
        "jitter_limit_fs": round(jitter_fs_limit, 2),
        "fps": fps_target,
        "note": "f_ADC = N_pixels * fps  (Nyquist: sample each pixel once per frame)",
    }


# ── 2. Sub-second biological phenomena catalog ─────────────────────────────

ULTRAFAST_PHENOMENA = {
    "rogue_wave_fiber": {
        "timescale_s": 1e-12,  # picosecond
        "description": "Optical rogue wave in fiber (dgs/nlse.py MI instability)",
        "griffiths": "Ch 9: wave packet group velocity; NLSE Kerr nonlinearity",
        "jalali_use": "RogueGuard: detect rare events in real-time STEAM stream",
        "imaging_fps_needed": 1e12,
        "steam_feasible": False,
        "alternative": "Electronic scope at 100 GHz bandwidth",
    },
    "action_potential": {
        "timescale_s": 1e-3,   # millisecond
        "description": "Neural action potential, cardiac cell depolarization",
        "griffiths": "Ch 2: Poisson eq for membrane potential V; Ch 7: ion current",
        "jalali_use": "STEAM optically images beating heart cells at 36 Mfps",
        "imaging_fps_needed": 1e4,
        "steam_feasible": True,
    },
    "cancer_cell_in_blood": {
        "timescale_s": 1e-4,   # 100 microseconds at 10 m/s
        "description": "Circulating tumor cell transiting optical field of view",
        "griffiths": "Ch 9: scattering cross section; Ch 2: Gauss law for cell charge",
        "jalali_use": "STEAM detects 1 cancer cell in 10^6 blood cells at 36 Mfps",
        "imaging_fps_needed": 1e6,
        "steam_feasible": True,
    },
    "cardiac_arrhythmia": {
        "timescale_s": 1e-2,   # 10 milliseconds
        "description": "Fibrillation wavefront in heart tissue",
        "griffiths": "Ch 7: Faraday induction (EKG); Ch 3: diffusion eq for wave",
        "jalali_use": "STEAM optically maps spiral wave pattern in 2D heart slice",
        "imaging_fps_needed": 1e3,
        "steam_feasible": True,
    },
    "protein_folding": {
        "timescale_s": 1e-6,   # microsecond
        "description": "Fast-folding protein conformational change",
        "griffiths": "Statistical mechanics (Ch 5 Schrodinger analog: quantum tunneling)",
        "jalali_use": "Time-stretch Raman spectroscopy maps folding in real time",
        "imaging_fps_needed": 1e7,
        "steam_feasible": True,
    },
}


# ── 3. Nonlinear extension of H(f): SPM + GVD ─────────────────────────────

def nonlinear_phase_accumulation(peak_power_W, gamma_per_W_km,
                                 length_km, beta2_ps2_km, T0_ps=1.0, N_t=256):
    """Nonlinear phase from NLSE: self-phase modulation + GVD.

    Split-step Fourier method (SSFM) -- industry standard for fiber simulation.

    NLSE: i*dA/dz + (beta2/2)*d^2A/dt^2 - gamma*|A|^2*A = 0

    For STEAM extension: if peak power is high enough that gamma*P0*L > 0.1 rad,
    the GS algorithm MUST include the nonlinear phase or it diverges.

    Rule of thumb: nonlinear length L_NL = 1/(gamma*P0)
    If L > L_NL: nonlinear effects dominate -> need NLSE-corrected H(f).
    """
    if peak_power_W <= 0 or gamma_per_W_km <= 0 or length_km <= 0:
        raise ValueError("power, gamma, length must be positive")

    L_NL = 1.0 / (gamma_per_W_km * peak_power_W)
    L_D  = T0_ps**2 / abs(beta2_ps2_km) if beta2_ps2_km != 0 else float('inf')

    # Gaussian pulse
    T0 = T0_ps
    dt = 5.0 * T0 / N_t
    t = (np.arange(N_t) - N_t//2) * dt
    A = np.sqrt(peak_power_W) * np.exp(-t**2 / (2 * T0**2))

    # SSFM: alternate nonlinear and linear half-steps
    dz = length_km / 50.0
    z = 0.0
    f = np.fft.fftfreq(N_t, d=dt)
    omega = 2 * np.pi * f
    linear_phase = np.exp(1j * (beta2_ps2_km / 2.0) * omega**2 * dz)

    A_curr = A.copy().astype(complex)
    for _ in range(50):
        # half-step nonlinear
        A_curr *= np.exp(1j * gamma_per_W_km * np.abs(A_curr)**2 * dz / 2)
        # full linear step in frequency domain
        A_f = np.fft.fft(A_curr)
        A_f *= linear_phase
        A_curr = np.fft.ifft(A_f)
        # half-step nonlinear
        A_curr *= np.exp(1j * gamma_per_W_km * np.abs(A_curr)**2 * dz / 2)
        z += dz

    phi_NL = gamma_per_W_km * peak_power_W * length_km
    spectrum_in  = np.abs(np.fft.fft(A))**2
    spectrum_out = np.abs(np.fft.fft(A_curr))**2
    spectral_broadening = np.sqrt(np.average(f**2, weights=spectrum_out) /
                                  (np.average(f**2, weights=spectrum_in) + 1e-30))

    return {
        "L_NL_km": round(L_NL, 3),
        "L_D_km": round(L_D, 3),
        "phi_NL_rad": round(phi_NL, 4),
        "nonlinear_dominant": L_NL < L_D,
        "spectral_broadening_factor": round(float(spectral_broadening), 3),
        "GS_needs_NLSE_correction": phi_NL > 0.1,
        "warning": "Include NLSE term in GS cost function if phi_NL > 0.1 rad" if phi_NL > 0.1 else "Linear approximation valid",
    }


# ── 4. Griffiths problem outcomes mapped to this project ──────────────────

GRIFFITHS_OUTCOMES = {
    "Ch1_Vector_Calculus": {
        "key_result": "Stokes theorem: curl integral = line integral",
        "used_in": "Faraday's law derivation (Ch 7); Maxwell's equations",
        "repo": "griffiths/vectors.py + dgs/line_integrals.py",
    },
    "Ch2_Electrostatics": {
        "key_result": "Gauss's law: flux = Q_enc/eps0; Poisson eq: nabla^2 V = -rho/eps0",
        "used_in": "p-n junction built-in voltage (dgs/transistor_logic.py); Laplace for membrane",
        "repo": "griffiths/electrostatics.py + dgs/static_membrane.py (Laplace FD)",
    },
    "Ch3_Potentials": {
        "key_result": "Uniqueness theorem; method of images; Legendre polynomial expansion",
        "used_in": "Green's function for Laplace (Jackson bridge); boundary conditions",
        "repo": "griffiths/potentials.py",
    },
    "Ch4_Dielectrics": {
        "key_result": "P = eps0*chi_e*E; eps_r = 1 + chi_e; Lorentz oscillator",
        "used_in": "Sellmeier n(lambda) -> H(f) = exp(i*pi*D*f^2); STEAM dispersion",
        "repo": "dgs/classical_ed.py",
    },
    "Ch7_Electrodynamics": {
        "key_result": "Faraday's law (complete); Maxwell's equations; Poynting theorem",
        "used_in": "Ground bounce V=L*dI/dt (dgs/computer_engineering.py); PCB wave eq",
        "repo": "griffiths/electrodynamics.py",
    },
    "Ch9_EM_Waves": {
        "key_result": "k = omega*n/c; group velocity v_g = domega/dk; GVD beta_2",
        "used_in": "H(f) = exp(i*pi*D*f^2) IS the Taylor expansion of k(omega) to 2nd order",
        "repo": "dgs/classical_ed.py + dgs/em_dispersion.py -- CORE of this project",
    },
    "Ch10_Lienard_Wiechert": {
        "key_result": "Fields from moving charge: E,B depend on retarded position and velocity",
        "used_in": "Radiation from accelerating electrons (synchrotron); antenna patterns",
        "repo": "NOT YET -- griffiths/radiation.py only has dipole, not full LW",
    },
    "Ch11_Radiation": {
        "key_result": "Larmor formula: P = q^2*a^2/(6*pi*eps0*c^3)",
        "used_in": "Laser gain/loss balance; spontaneous emission rate",
        "repo": "dgs/ballistics_radiation.py + scripts/build_radiation.py",
    },
}


def griffiths_problem_map():
    """Print outcomes of all Griffiths chapters and how they connect to this repo."""
    print("  Ch    Key Result -> Repo Connection")
    print("  " + "-"*60)
    for ch, d in GRIFFITHS_OUTCOMES.items():
        flag = "[TODO]" if "NOT YET" in d["repo"] else "[DONE]"
        print(f"  {flag} {ch}")
        print(f"         Result: {d['key_result'][:65]}")
        print(f"         Repo:   {d['repo'][:65]}")


# ── 5. Integrals in computer engineering (UCLA AI science) ────────────────

CE_INTEGRALS = {
    "RC_charging": {
        "integral": "V(t) = (1/C) integral_0^t i dt",
        "result": "V = V_in*(1 - e^{-t/RC})",
        "CE_use": "Settling time in ADC sample-hold; STEAM ADC front-end bandwidth",
        "griffiths": "Ch 7: displacement current i_D = eps0 * dE/dt -> integral form",
        "repo": "dgs/circuits.py rc_step()",
    },
    "Fourier_transform": {
        "integral": "X(f) = integral_{-inf}^{inf} x(t) * e^{-i*2*pi*f*t} dt",
        "result": "Frequency spectrum; convolution theorem: FT[x*h] = X*H",
        "CE_use": "FFT in ADC digital filter; STEAM spectral encoding H(f)",
        "griffiths": "Ch 9: plane wave decomposition of arbitrary wave packet",
        "repo": "dgs/fourier_tools.py + gs_core.py",
    },
    "energy_stored": {
        "integral": "E = (1/2)*C*V^2 = integral_0^V Q dV = integral q/C dq",
        "result": "Energy in capacitor; switching energy per bit = C*V^2/2",
        "CE_use": "CMOS power: P = alpha*C*V_DD^2*f; STEAM ADC power budget",
        "griffiths": "Ch 2: energy of charge distribution W = (eps0/2)*integral E^2 dV",
        "repo": "dgs/computer_engineering.py (cmos_switching_energy)",
    },
    "transmission_line_energy": {
        "integral": "P = integral S*dA = (1/2)*Re[E x H*] (Poynting)",
        "result": "Power flow = Z0 * |I|^2 / 2  on 50-ohm line",
        "CE_use": "Signal power budget in PCB high-speed link (STEAM ADC board)",
        "griffiths": "Ch 8: Poynting vector S = (1/mu0)*E x B",
        "repo": "dgs/classical_ed.py poynting_vector_plane_wave()",
    },
    "charge_carrier_integral": {
        "integral": "J = q * integral_0^inf v * f(E) * g(E) dE  (current density)",
        "result": "Ohm's law J = sigma*E from microscopic carrier distribution",
        "CE_use": "Photodetector responsivity integral: R = q*lambda/(h*c)*QE",
        "griffiths": "Ch 5: current density J = rho*v -> Ampere source term",
        "repo": "dgs/computer_engineering.py + dgs/electrons.py",
    },
}


def ce_integrals_print():
    print("  Integral -> CE Application -> Griffiths")
    print("  " + "-"*60)
    for name, d in CE_INTEGRALS.items():
        print(f"\n  {name.upper().replace('_',' ')}")
        print(f"    Integral: {d['integral']}")
        print(f"    Result:   {d['result']}")
        print(f"    CE use:   {d['CE_use']}")
        print(f"    Griffiths:{d['griffiths']}")


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 65)
    print("  dgs/steam_imaging.py  --  Jalali STEAM + nonlinear extension")
    print("=" * 65)

    print("\n--- ADC clock requirements for STEAM ---")
    r = adc_clock_requirements(optical_bandwidth_nm=10, D_ps_per_nm=400,
                               fps_target=36e6)
    print(f"  Time window per frame: {r['T_window_ps']} ps")
    print(f"  Carrier frequency:     {r['f0_THz']} THz  (1550 nm)")
    print(f"  ADC clock needed:      {r['f_ADC_GHz_needed']:.0f} GHz")
    print(f"  Jitter limit:          {r['jitter_limit_fs']:.1f} fs  (for <0.1 rad phase error)")
    print(f"  Note: {r['note']}")

    print("\n--- Sub-second human imaging phenomena ---")
    print(f"  {'Phenomenon':28s} {'Timescale':12s} {'STEAM?':8s} {'fps needed'}")
    print("  " + "-"*60)
    for name, p in ULTRAFAST_PHENOMENA.items():
        ts = p["timescale_s"]
        fps = p["imaging_fps_needed"]
        feasible = "YES" if p["steam_feasible"] else "NO (scope)"
        ts_str = f"{ts:.0e} s"
        print(f"  {name:28s} {ts_str:12s} {feasible:8s} {fps:.0e} fps")

    print("\n--- Nonlinear phase: when GS needs NLSE correction ---")
    cases = [
        ("Low power (1 mW)",   0.001, 1.3, 50, -20),
        ("Medium (100 mW)",    0.1,   1.3, 50, -20),
        ("High power (1 W)",   1.0,   1.3, 50, -20),
        ("EDFA booster (5W)",  5.0,   1.3, 50, -20),
    ]
    print(f"  {'Case':22s} {'phi_NL':8s} {'L_NL':8s} {'NL dominant':12s} {'GS needs fix?'}")
    for label, P, gamma, L, beta2 in cases:
        r = nonlinear_phase_accumulation(P, gamma, L, beta2)
        print(f"  {label:22s} {r['phi_NL_rad']:6.3f}rad  {r['L_NL_km']:6.3f}km "
              f"  {'YES' if r['nonlinear_dominant'] else 'no':10s}  "
              f"{'FIX NEEDED' if r['GS_needs_NLSE_correction'] else 'linear OK'}")

    print("\n--- Griffiths outcomes -> this project ---")
    griffiths_problem_map()

    print("\n--- Integrals in computer engineering (UCLA AI science) ---")
    ce_integrals_print()

    print("\n" + "=" * 65)
    print("  PROJECT EXTENSION ROADMAP")
    print("=" * 65)
    print("  1. STEAM forward model:     dgs/steam_imaging.py (this file)")
    print("  2. Linear GS recovery:      dgs/gs_core.py  H(f)=exp(i*pi*D*f^2)")
    print("  3. Nonlinear correction:    dgs/nlse.py SSFM + phi_NL > 0.1 rad")
    print("  4. Neural net (Paper [3]):  dgs/nn_spectral_regression.py")
    print("  5. Real-time deployment:    dgs/gs_unsupervised.py + pygame")
    print("  6. Griffiths gap (Ch10):    Lienard-Wiechert -> add to griffiths/")
    print()
    print("  TITLE for UC Davis / Jalali cold email:")
    print("  'Dispersion-Assisted GS Phase Recovery for Real-Time STEAM'")


if __name__ == "__main__":
    demo()
