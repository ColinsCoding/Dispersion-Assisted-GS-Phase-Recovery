"""Jalali Lab mathematical grammar -- the equations behind every paper.

WHO IS JALALI:
  Prof. Bahram Jalali, UCLA EE (bjjalali@ucla.edu)
  NAE member 2022.  300+ papers, 25,000+ citations, 32 US patents.
  Inventor of: photonic time stretch, STEAM microscopy, optical rogue wave
  detection, anamorphic stretch transform, warped stretch transform.

THE MEASUREMENT PEOPLE (who runs experiments in the lab):
  Keisuke Goda     -- STEAM, dispersive FT, high-throughput microscopy (2008-13)
  Kevin Tsia       -- time-stretch, silicon photonics, confocal microscopy (2007-10)
  Daniel Solli     -- rogue waves, nonlinear optics, supercontinuum (2007-12)
  Ata Mahjoubfar   -- warped stretch, time-stretch imaging, AI (2011-17)
  Claire Chen      -- deep learning + cell classification (2014-16)
  Ali Fard         -- time-stretch ADC, digitizers (2010-13)
  Osman Boyraz     -- silicon Raman laser (2002-05)
  Eric Diebold     -- beat-frequency multiplexed fluorescence microscopy

  CURRENT LAB: check photonics.ucla.edu/people for active grad students.
  Best cold-email target: the most recent first-author on a paper closest
  to your topic -- they are the experimentalist who RUNS the setup.

JALALI'S MATHEMATICAL GRAMMAR (the 8 equations that appear in every paper):

  [1] TIME-STRETCH FACTOR M (the master equation)
      M = 1 + D2*L2 / (D1*L1)
      where D1,L1 = pre-disperser GVD and length
            D2,L2 = stretch disperser GVD and length
      Physical meaning: the pulse is stretched by factor M in time.
      After stretch: f_ADC_needed = f_signal / M  (ADC sees slowed signal)

  [2] TIME-WAVELENGTH MAPPING (the STEAM coordinate transform)
      t(lambda) = D_total * (lambda - lambda_0)
      where D_total = total accumulated GVD [ps/nm]
      Physical meaning: each wavelength arrives at a DIFFERENT TIME.
      This maps frequency -> time -> space (via grating before sample).
      Griffiths Ch 9: this is group delay tau(omega) = d(phi)/domega = D*omega

  [3] DISPERSIVE FOURIER TRANSFORM (DFT, the STEAM measurement)
      E_out(t) = FT[E_in(f)] |_{f = t / (D*lambda_0^2/c)}
      Physical meaning: the time-domain output IS the Fourier transform
      of the input optical spectrum. This is only true when:
        condition: D*L >> T0^2 / (2*pi)  (far-field / Fraunhofer condition)
      Same math as Fraunhofer diffraction (Griffiths Ch 9, far field).

  [4] TIME-BANDWIDTH PRODUCT (TBP, the conservation law)
      TBP = Delta_t * Delta_f = constant  (for linear dispersive systems)
      After stretch by M: Delta_t -> M*Delta_t, Delta_f -> Delta_f/M
      -> TBP is conserved.
      Physical meaning: you CANNOT get more time resolution AND more
      bandwidth simultaneously -- the stretch just trades one for the other.
      This is the photonic analog of Heisenberg uncertainty: Delta_t*Delta_omega >= 1/2

  [5] ROGUE WAVE STATISTICS (from Solli, Ropers, Jalali, Nature 2007)
      P(I > I_threshold) ~ exp(-I_threshold / <I>)  (exponential tail)
      Rogue waves: I_threshold >> <I>  (L-shaped probability distribution)
      Modulation instability gain: g(Omega) = |beta_2|/2 * Omega * sqrt(4*gamma^2*P0^2/beta_2^2 - Omega^2)
      Peak gain at: Omega_max = sqrt(2*gamma*P0 / |beta_2|)
      This is NLSE (dgs/nlse.py) linearized around CW solution.

  [6] WARPED STRETCH TRANSFORM (Asghari & Jalali, Applied Optics 2013)
      phi(omega) = phi_0 + phi_1*omega + (1/2)*phi_2*omega^2 + higher order
      Warped: phi(omega) = arbitrary nonlinear phase -> non-uniform time-freq map
      t(omega) = d(phi)/d(omega) = phi_1 + phi_2*omega + ...
      Purpose: focus time resolution on frequency bands of interest
      (like fovea in human eye -> "foveated time stretch")
      Repo: H(f) = exp(i*pi*D*f^2) is the SPECIAL CASE phi_2 only (linear chirp)

  [7] SIGNAL-TO-NOISE RATIO AFTER STRETCH (Mahjoubfar et al. 2017)
      SNR_out = SNR_in * sqrt(M)   (shot-noise limited, incoherent detection)
      SNR_out = SNR_in * M         (coherent detection, phase-sensitive)
      Physical meaning: stretching amplifies the signal relative to bandwidth-
      limited noise. This is WHY STEAM can image faster than a camera.
      In GS context: SNR_out determines how small a delta_phi we can recover.

  [8] DEEP LEARNING CELL CLASSIFIER (Chen, Mahjoubfar, Jalali, Sci Reports 2016)
      Features extracted from STEAM intensity I(t):
        x = [I_max, I_mean, std(I), skew(I), kurtosis(I), entropy(I)]
      Classification: y = softmax(W2 * relu(W1 * x + b1) + b2)
      AUC = 0.99 for cancer vs normal cells in blood (30,000 fps)
      THIS IS Project 5 in this repo (dgs/nn_spectral_regression.py).

KEY PAPER SUMMARY (read these in order):

  FOUNDATION (read first):
    Coppinger, Bhushan, Jalali (1999) IEEE MTT -- original time-stretch ADC paper
    Han & Jalali (2003) JLT -- full mathematical framework of photonic time-stretch

  STEAM:
    Goda, Tsia, Jalali (2009) Nature -- STEAM microscopy (world's fastest camera)
    Goda & Jalali (2013) Nature Photonics -- dispersive Fourier transform review

  ROGUE WAVES:
    Solli, Ropers, Koonath, Jalali (2007) Nature -- first optical rogue wave paper
    Solli, Ropers, Jalali (2008) PRL -- active control of rogue waves

  AI + STEAM:
    Chen, Mahjoubfar, Jalali et al. (2016) Sci Reports -- deep learning cell class.

  MATHEMATICAL REVIEW:
    Mahjoubfar et al. (2017) Nature Photonics -- "Time stretch and its applications"
    Jalali & Mahjoubfar (2015) Proc IEEE -- tailoring wideband signals

  RECENT (2022+):
    Zhou et al. (2022) Laser & Photonics Reviews -- unified framework for time-stretch

Run: py -3.13 -c "from dgs.jalali_grammar import demo; demo()"
"""
import numpy as np

C_LIGHT = 2.998e8
H_PLANCK = 6.626e-34


def time_stretch_factor(D1_ps_nm, L1_km, D2_ps_nm, L2_km):
    """Jalali equation [1]: temporal stretch factor M.

    M = 1 + (D2*L2) / (D1*L1)

    For STEAM:  D1 (pre-disperser, short) << D2 (stretch fiber, long)
    Typical:    D1=-17 ps/nm/km, L1=0.1 km, D2=-17 ps/nm/km, L2=10 km -> M=101
    """
    if D1_ps_nm == 0 or L1_km == 0:
        raise ValueError("pre-disperser D1, L1 must be nonzero")
    M = 1.0 + (D2_ps_nm * L2_km) / (D1_ps_nm * L1_km)
    f_adc_reduction = abs(M)
    return {"M": round(M, 3), "f_ADC_reduction_factor": round(f_adc_reduction, 2),
            "meaning": f"ADC needs {f_adc_reduction:.0f}x lower sample rate than signal BW"}


def time_wavelength_map(D_total_ps_nm, lambda0_nm, lambda_arr_nm):
    """Jalali equation [2]: t(lambda) = D_total * (lambda - lambda0).

    Maps each optical frequency to a unique arrival time.
    This IS the STEAM coordinate transform.
    """
    t_arr = D_total_ps_nm * (np.asarray(lambda_arr_nm) - lambda0_nm)
    return {"t_ps": t_arr, "D_total_ps_nm": D_total_ps_nm,
            "lambda0_nm": lambda0_nm,
            "dt_dlambda_ps_nm": D_total_ps_nm,
            "fraunhofer_condition": "D*L >> T0^2/(2*pi): must verify for each pulse"}


def tbp_conservation(Delta_t_in_ps, Delta_f_in_GHz, M):
    """Jalali equation [4]: TBP = Delta_t * Delta_f is conserved under stretch.

    After stretch by M:
      Delta_t_out = M * Delta_t_in
      Delta_f_out = Delta_f_in / M
      TBP_out = TBP_in  (conservation)

    Analogy to Heisenberg: Delta_t * Delta_omega >= 1/2
    """
    TBP_in = Delta_t_in_ps * Delta_f_in_GHz * 1e-3   # dimensionless
    Delta_t_out = M * Delta_t_in_ps
    Delta_f_out = Delta_f_in_GHz / abs(M)
    TBP_out = Delta_t_out * Delta_f_out * 1e-3
    return {"TBP_in": round(TBP_in, 4), "TBP_out": round(TBP_out, 4),
            "Delta_t_out_ps": round(Delta_t_out, 2),
            "Delta_f_out_GHz": round(Delta_f_out, 4),
            "conserved": abs(TBP_out - TBP_in) < 1e-8 * TBP_in}


def rogue_wave_mi_gain(beta2_ps2_km, gamma_per_W_km, P0_W,
                       Omega_arr_GHz=None):
    """Jalali equation [5]: modulation instability gain spectrum.

    g(Omega) = (|beta2|/2)*Omega * sqrt(4*gamma^2*P0^2/beta2^2 - Omega^2)

    Peak gain at Omega_max = sqrt(2*gamma*P0 / |beta2|)
    Frequency range of instability: 0 < Omega < 2*sqrt(gamma*P0/|beta2|)

    This is the mechanism that creates optical rogue waves in fiber.
    Same math as Benjamin-Feir instability in water waves.
    """
    if Omega_arr_GHz is None:
        Omega_max_sq = 2 * gamma_per_W_km * P0_W / abs(beta2_ps2_km) * 1e3
        Omega_max = np.sqrt(max(Omega_max_sq, 0))
        Omega_arr_GHz = np.linspace(0, 2.5 * Omega_max, 200)

    b2 = beta2_ps2_km * 1e-3 / 1e24   # convert to s^2/m
    g_per_W_m = gamma_per_W_km / 1e3
    Om = Omega_arr_GHz * 2 * np.pi * 1e9   # rad/s
    under_sqrt = 4 * g_per_W_m**2 * P0_W**2 / b2**2 - Om**2
    gain = np.where(under_sqrt > 0,
                    (abs(b2)/2) * Om * np.sqrt(under_sqrt), 0.0)

    idx_max = np.argmax(gain)
    Omega_peak = Omega_arr_GHz[idx_max]
    g_peak = gain[idx_max]
    return {"gain_m_inv": gain, "Omega_GHz": Omega_arr_GHz,
            "Omega_peak_GHz": round(Omega_peak, 3),
            "g_peak_m_inv": round(float(g_peak), 6),
            "g_peak_dB_m": round(float(g_peak) * 20/np.log(10), 3)}


def snr_after_stretch(SNR_in_dB, M, detection="coherent"):
    """Jalali equation [7]: SNR improvement from time-stretch.

    Coherent:   SNR_out = SNR_in * M      (20*log10(M) dB improvement)
    Incoherent: SNR_out = SNR_in * sqrt(M) (10*log10(M) dB improvement)

    This is WHY STEAM can image faster than conventional cameras:
    the stretch factor M > 1 IMPROVES SNR, not worsens it.
    """
    if detection == "coherent":
        SNR_gain_dB = 20 * np.log10(abs(M))
    else:
        SNR_gain_dB = 10 * np.log10(abs(M))
    SNR_out_dB = SNR_in_dB + SNR_gain_dB
    return {"SNR_in_dB": SNR_in_dB, "SNR_out_dB": round(SNR_out_dB, 2),
            "SNR_gain_dB": round(SNR_gain_dB, 2),
            "M": M, "detection": detection}


def warped_stretch_phase(omega_arr, phi2, phi3=0.0, phi4=0.0):
    """Jalali equation [6]: warped stretch phase phi(omega).

    phi(omega) = (1/2)*phi2*omega^2 + (1/6)*phi3*omega^3 + (1/24)*phi4*omega^4

    Group delay: tau(omega) = d(phi)/d(omega) = phi2*omega + (1/2)*phi3*omega^2 + ...

    Linear chirp (phi3=phi4=0): tau = phi2*omega  -> uniform time-freq map
    Warped (phi3 nonzero):       tau = phi2*omega + phi3/2*omega^2  -> nonlinear

    This repo uses linear chirp ONLY: H(f)=exp(i*pi*D*f^2) = exp(i/2*phi2*omega^2)
    phi2 = D [ps^2] = the D in our GS algorithm.
    """
    om = np.asarray(omega_arr)
    phi = 0.5 * phi2 * om**2 + (1/6) * phi3 * om**3 + (1/24) * phi4 * om**4
    tau = phi2 * om + 0.5 * phi3 * om**2 + (1/6) * phi4 * om**3
    H = np.exp(1j * phi)
    return {"phi": phi, "tau_group_delay": tau, "H": H,
            "phi2": phi2, "phi3": phi3, "phi4": phi4,
            "this_repo": "uses phi3=phi4=0 (linear H(f)=exp(i*pi*D*f^2))"}


def demo():
    print("=" * 65)
    print("  Jalali Lab Mathematical Grammar")
    print("  photonics.ucla.edu  |  bjjalali@ucla.edu")
    print("=" * 65)

    print("\n--- Eq [1]: Time-Stretch Factor ---")
    cases = [
        ("Lab demo",   -17, 0.1, -17, 10),
        ("High-M",     -17, 0.05, -17, 25),
        ("STEAM typ.", -100, 0.5, -17, 20),
    ]
    for label, D1, L1, D2, L2 in cases:
        r = time_stretch_factor(D1, L1, D2, L2)
        print(f"  {label:12s}: M={r['M']:7.1f}  {r['meaning']}")

    print("\n--- Eq [2]: Time-Wavelength Map (D=400 ps/nm, lambda0=1550 nm) ---")
    lambdas = np.array([1545, 1548, 1550, 1552, 1555])
    r = time_wavelength_map(400.0, 1550.0, lambdas)
    for lam, t in zip(lambdas, r["t_ps"]):
        print(f"  lambda={lam} nm -> t={t:+.0f} ps")

    print("\n--- Eq [4]: TBP Conservation (M=100) ---")
    r = tbp_conservation(1.0, 100.0, 100)
    print(f"  Input:  Delta_t={1} ps, Delta_f={100} GHz, TBP={r['TBP_in']:.2f}")
    print(f"  Output: Delta_t={r['Delta_t_out_ps']} ps, Delta_f={r['Delta_f_out_GHz']} GHz, TBP={r['TBP_out']:.2f}")
    print(f"  Conserved: {r['conserved']}  (TBP is photonic Heisenberg uncertainty)")

    print("\n--- Eq [5]: Rogue Wave MI Gain (SMF-28, P0=100 mW) ---")
    r = rogue_wave_mi_gain(beta2_ps2_km=-20, gamma_per_W_km=1.3, P0_W=0.1)
    print(f"  Peak gain at Omega={r['Omega_peak_GHz']:.2f} GHz")
    print(f"  Peak gain = {r['g_peak_dB_m']:.2f} dB/m")
    print(f"  (Rogue waves grow in this frequency band -> L-shaped statistics)")

    print("\n--- Eq [7]: SNR After Stretch (M=100, SNR_in=20 dB) ---")
    for det in ["coherent", "incoherent"]:
        r = snr_after_stretch(20.0, 100, det)
        print(f"  {det:12s}: SNR_out={r['SNR_out_dB']:.1f} dB  (+{r['SNR_gain_dB']:.1f} dB from stretch)")

    print("\n--- Eq [6]: Warped vs Linear Phase (this repo uses linear only) ---")
    om = np.linspace(-1, 1, 5)
    phi2 = -5000.0   # ps^2 = D in this repo
    r_lin = warped_stretch_phase(om, phi2, phi3=0)
    r_warp = warped_stretch_phase(om, phi2, phi3=500)
    print(f"  omega:         {om}")
    print(f"  tau linear:    {r_lin['tau_group_delay'].round(1)}")
    print(f"  tau warped:    {r_warp['tau_group_delay'].round(1)}")
    print(f"  This repo:     {r_lin['this_repo']}")

    print("\n--- KEY PAPERS (read in this order) ---")
    papers = [
        ("1999", "Coppinger, Jalali", "IEEE MTT", "Original photonic time-stretch ADC"),
        ("2007", "Solli, Jalali", "Nature", "First optical rogue wave paper"),
        ("2009", "Goda, Tsia, Jalali", "Nature", "STEAM: world's fastest camera"),
        ("2013", "Goda, Jalali", "Nature Photonics", "Dispersive FT review"),
        ("2016", "Chen, Mahjoubfar, Jalali", "Sci Reports", "Deep learning cell classification"),
        ("2017", "Mahjoubfar, Jalali", "Nature Photonics", "Time stretch and its applications"),
        ("2022", "Zhou et al.", "Laser & Phot. Rev.", "Unified framework for time-stretch"),
    ]
    for year, auth, journal, title in papers:
        print(f"  [{year}] {auth:28s} {journal:20s} {title}")

    print("\n--- WHO TO EMAIL ---")
    print("  Prof. Jalali:    bjjalali@ucla.edu")
    print("  Measurement PI:  Check photonics.ucla.edu/people for current postdoc/grad")
    print("  Best strategy:   Email FIRST AUTHOR of most recent STEAM paper")
    print("  Subject:         'GS Phase Recovery for STEAM -- Collaboration'")
    print("  Attach:          notebooks/crispr_steam_theory.ipynb (executed)")
    print("  GitHub:          github.com/ColinsCoding/Dispersion-Assisted-GS-Phase-Recovery")


if __name__ == "__main__":
    demo()
