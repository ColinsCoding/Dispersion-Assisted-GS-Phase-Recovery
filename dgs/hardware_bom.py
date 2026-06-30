"""Hardware bill of materials for a dispersion-assisted GS phase retrieval bench.

THE CENTRAL QUESTION: how do you actually measure I1(t) and I2(t)?

  I1(t) = |E_in(t)|^2   -- intensity BEFORE the dispersive fiber
  I2(t) = |E_out(t)|^2  -- intensity AFTER the dispersive fiber

Both are optical power vs. time. A photodetector converts photons -> electrons ->
voltage. An oscilloscope or digitizer samples that voltage -> I(t) as a numpy array.
That array is the INPUT to gs_tsdft() in this repo.

The dispersive fiber plays the role of H(f) = exp(i*pi*D*f^2). SMF-28 standard
single-mode fiber gives D_fiber = -17 ps/(nm*km) at 1550 nm. You need
|D_total| >> 5000 ps^2 -- that is about 300 m of SMF-28, or a compact
chirped fiber Bragg grating (CFBG) that fits in a matchbox.

MINIMUM VIABLE BENCH (under $3000 total, student lab):
  1 x Thorlabs DET01CFC -- 1 GHz InGaAs photodetector, FC/PC, $280
  1 x 50:50 fiber coupler (1550 nm, SMF-28) -- splits input to give I1, $120
  CFBG (chirped fiber Bragg grating) ~5000 ps^2, ~$800  (300 m SMF-28 only gives ~6 ps^2 -- need 230 km of fiber OR a CFBG)
  1 x oscilloscope >= 2 GSa/s, 1 GHz BW (Rigol DS1054Z is too slow; use
      Rigol DHO914S 1.5 GSa/s 200 MHz or Keysight DSOX1204G), ~$500-800
  1 x 1550 nm laser diode (DFB, 1 mW) or access to a lab laser, $200-600
  SMA cables, FC/APC patch cords, bias-T for photodetector, $100

FULL LAB (Jalali lab level):
  2 x Newport 818-BB-45F broadband photodetector, DC-45 GHz, $2800 each
  1 x Keysight DSAZ594A real-time oscilloscope, 59 GHz, 160 GSa/s, ~$500K
  1 x chirped FBG (CFBG) with D = -1000 ps/nm (=-10^6 ps^2/nm), compact
  1 x EDFA (Er-doped fiber amplifier) to boost signal, ~$3000
  1 x tunable laser (Santec TSL-710), ~$30K

This module encodes the engineering math so you can SIZE your hardware before buying.
"""

import numpy as np
import sympy as sp


# ── fiber dispersion calculator ───────────────────────────────────────

SMF28_DISPERSION_PS_PER_NM_PER_KM = -17.0   # at 1550 nm (anomalous dispersion)
C_LIGHT_NM_PS = 299792.458                   # nm/ps


def fiber_gdd(length_m, D_fiber_ps_per_nm_per_km=SMF28_DISPERSION_PS_PER_NM_PER_KM,
              wavelength_nm=1550.0):
    """Total group-delay dispersion (GDD) in ps^2 for a fiber of given length.

    GDD (ps^2) = D_fiber (ps/nm/km) * length (km) * lambda^2 (nm^2) / (2*pi*c (nm/ps))

    For SMF-28 at 1550 nm, D_fiber = -17 ps/(nm*km).
    The GS algorithm needs |GDD| >= 5000 ps^2 for clean phase retrieval.
    """
    if length_m <= 0:
        raise ValueError("length_m must be positive")
    length_km = length_m / 1000.0
    # GDD = D * L * lambda^2 / (2*pi*c)
    gdd_ps2 = (D_fiber_ps_per_nm_per_km * length_km
               * wavelength_nm**2 / (2 * np.pi * C_LIGHT_NM_PS))
    return {
        "gdd_ps2": gdd_ps2,
        "length_m": length_m,
        "D_fiber": D_fiber_ps_per_nm_per_km,
        "wavelength_nm": wavelength_nm,
        "absolute_gdd_ps2": abs(gdd_ps2),
        "meets_5000_ps2_threshold": abs(gdd_ps2) >= 5000.0,
    }


def fiber_length_for_gdd(target_gdd_ps2, D_fiber_ps_per_nm_per_km=SMF28_DISPERSION_PS_PER_NM_PER_KM,
                          wavelength_nm=1550.0):
    """How many metres of SMF-28 do you need to reach target |GDD|?"""
    if target_gdd_ps2 <= 0:
        raise ValueError("target_gdd_ps2 must be positive")
    # |L| = |GDD| * 2*pi*c / (|D_fiber| * lambda^2)
    length_km = (abs(target_gdd_ps2) * 2 * np.pi * C_LIGHT_NM_PS
                 / (abs(D_fiber_ps_per_nm_per_km) * wavelength_nm**2))
    return {
        "length_m": length_km * 1000.0,
        "length_km": length_km,
        "target_gdd_ps2": target_gdd_ps2,
    }


# ── photodetector SNR budget ─────────────────────────────────────────

def photodetector_snr(optical_power_mW, responsivity_A_per_W=0.9,
                       bandwidth_GHz=1.0, dark_current_nA=5.0, load_ohm=50.0,
                       temperature_K=300.0):
    """Estimate SNR (dB) at the photodetector output.

    Signal current: I_sig = R * P
    Shot noise:     i_shot^2 = 2*q*I_sig*BW
    Thermal noise:  i_th^2 = 4*k_B*T*BW / R_load
    Dark current:   i_dark^2 = 2*q*I_dark*BW

    SNR = I_sig^2 / (i_shot^2 + i_th^2 + i_dark^2)  [power SNR]

    Parameters
    ----------
    optical_power_mW : float  -- input optical power in milliwatts
    responsivity_A_per_W : float  -- detector responsivity (A/W); InGaAs ~0.9 at 1550 nm
    bandwidth_GHz : float  -- electrical bandwidth (GHz)
    dark_current_nA : float  -- dark current (nA)
    load_ohm : float  -- load resistance (Ohm)
    temperature_K : float  -- temperature (K)

    Returns dict with currents, noise terms, and SNR_dB.
    """
    if optical_power_mW <= 0:
        raise ValueError("optical_power_mW must be positive")
    q = 1.602e-19       # C
    k_B = 1.381e-23     # J/K
    P_W = optical_power_mW * 1e-3
    BW_Hz = bandwidth_GHz * 1e9
    I_dark_A = dark_current_nA * 1e-9

    I_sig = responsivity_A_per_W * P_W           # signal current (A)
    i_shot2 = 2 * q * (I_sig + I_dark_A) * BW_Hz
    i_th2 = 4 * k_B * temperature_K * BW_Hz / load_ohm
    i_dark2 = 2 * q * I_dark_A * BW_Hz
    i_noise2 = i_shot2 + i_th2 + i_dark2

    snr_linear = I_sig**2 / i_noise2
    snr_db = 10 * np.log10(snr_linear)

    return {
        "I_signal_uA": I_sig * 1e6,
        "i_shot_nA_rms": np.sqrt(i_shot2) * 1e9,
        "i_thermal_nA_rms": np.sqrt(i_th2) * 1e9,
        "SNR_dB": snr_db,
        "noise_limited_by": "shot" if i_shot2 > i_th2 else "thermal",
    }


# ── sampling rate requirement ─────────────────────────────────────────

def sampling_requirement(signal_bandwidth_GHz, oversampling_factor=2.5):
    """Minimum ADC sampling rate to capture a signal of given bandwidth.

    Nyquist: f_sample >= 2 * f_max.
    In practice use 2.5x to leave margin for anti-alias filter roll-off.
    """
    if signal_bandwidth_GHz <= 0:
        raise ValueError("signal_bandwidth_GHz must be positive")
    f_nyquist = 2.0 * signal_bandwidth_GHz
    f_practical = oversampling_factor * signal_bandwidth_GHz
    return {
        "nyquist_rate_GSa_s": f_nyquist,
        "practical_rate_GSa_s": f_practical,
        "oversampling_factor": oversampling_factor,
        "note": "Use f_sample >= practical_rate_GSa_s for clean capture",
    }


# ── bill of materials ─────────────────────────────────────────────────

BOM_MINIMUM_VIABLE = [
    {"item": "InGaAs photodetector 1 GHz",
     "example_part": "Thorlabs DET01CFC",
     "qty": 1,
     "approx_usd": 280,
     "spec": "800-1700 nm, 1 GHz BW, FC/PC, 50 Ohm",
     "role": "Measures I1(t) before fiber (and I2(t) after -- swap cable)"},
    {"item": "50:50 fiber coupler 1550 nm",
     "example_part": "Thorlabs TW1550R5A2",
     "qty": 1,
     "approx_usd": 120,
     "spec": "1550 nm, SMF-28, 50/50 split ratio, 2x2",
     "role": "Splits input beam: one arm -> I1 detector, other arm -> fiber -> I2 detector"},
    {"item": "Chirped fiber Bragg grating (CFBG) ~5000 ps^2",
     "example_part": "Proximion DCM or custom CFBG; alt: 230 km SMF-28 via recirculating loop",
     "qty": 1,
     "approx_usd": 800,
     "spec": "D = -5000 ps^2 at 1550 nm, compact. NOTE: 300 m SMF-28 gives only ~6 ps^2; "
             "need 230 km SMF-28 OR a CFBG for the 5000 ps^2 threshold.",
     "role": "Dispersive element H(f) = exp(i*pi*D*f^2); maps frequency -> time (TS-DFT)"},
    {"item": "Oscilloscope 1 GHz BW, >=2 GSa/s",
     "example_part": "Keysight DSOX1204G (200 MHz / 2 GSa/s) or Rigol DHO914S",
     "qty": 1,
     "approx_usd": 700,
     "spec": ">=1 GHz BW, >=2 GSa/s, USB export of waveform data",
     "role": "Digitizes I1(t) and I2(t) -> numpy array for gs_tsdft()"},
    {"item": "1550 nm DFB laser diode",
     "example_part": "Thorlabs S1FC1550 or lab fiber laser",
     "qty": 1,
     "approx_usd": 400,
     "spec": "CW, 1 mW, single-mode fiber pigtail",
     "role": "Optical source; modulate with external modulator for BPSK test signal"},
    {"item": "FC/APC patch cords + SMA cables",
     "example_part": "Thorlabs P1-SMF28E-FC-x or generic",
     "qty": 6,
     "approx_usd": 100,
     "spec": "SMF-28, FC/APC or FC/PC to match detector",
     "role": "Interconnects"},
]

BOM_FULL_LAB = [
    {"item": "Broadband photodetector 45 GHz",
     "example_part": "Newport 818-BB-45F",
     "qty": 2,
     "approx_usd": 2800,
     "spec": "DC-45 GHz, 1064/1310/1550 nm, simultaneous I1 and I2",
     "role": "Simultaneous dual-channel intensity capture"},
    {"item": "Real-time oscilloscope 33 GHz",
     "example_part": "Keysight DSAZ334A (33 GHz / 80 GSa/s)",
     "qty": 1,
     "approx_usd": 150000,
     "spec": "33 GHz BW, 80 GSa/s, 4 channels, deep memory",
     "role": "Full-bandwidth I1(t) and I2(t) capture"},
    {"item": "Chirped fiber Bragg grating (CFBG)",
     "example_part": "Proximion or TeraXion custom CFBG",
     "qty": 1,
     "approx_usd": 3000,
     "spec": "D = -1000 ps/nm at 1550 nm = 1e6 ps^2/nm, compact <10 cm",
     "role": "Compact dispersive element replacing km of fiber"},
    {"item": "EDFA erbium-doped fiber amplifier",
     "example_part": "Thorlabs EDFA100S",
     "qty": 1,
     "approx_usd": 3000,
     "spec": "+23 dBm output, 1530-1565 nm, low-noise",
     "role": "Boost optical power before detector"},
    {"item": "Tunable laser C-band",
     "example_part": "Santec TSL-710",
     "qty": 1,
     "approx_usd": 30000,
     "spec": "1480-1640 nm tunable, <100 kHz linewidth",
     "role": "Clean single-frequency source for coherent experiments"},
]


def bom_total_cost(bom_list):
    """Sum the approximate cost of a BOM list."""
    return sum(item["qty"] * item["approx_usd"] for item in bom_list)


def print_bom(bom_list, title="Bill of Materials"):
    """Print BOM in man-page style."""
    total = bom_total_cost(bom_list)
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for i, item in enumerate(bom_list, 1):
        print(f"\n[{i}] {item['item']}")
        print(f"    Part:   {item['example_part']}")
        print(f"    Qty:    {item['qty']}  (~${item['approx_usd']} each)")
        print(f"    Spec:   {item['spec']}")
        print(f"    Role:   {item['role']}")
    print(f"\nESTIMATED TOTAL: ${total:,}")
    print(f"{'='*60}\n")


# ── SymPy formalism ───────────────────────────────────────────────────

def hardware_sympy_5():
    """Five key hardware/measurement equations in SymPy."""
    R, P, q, BW = sp.symbols('R P q BW', positive=True)
    D_sym, L, lam, c_sym = sp.symbols('D L lambda c', real=True)
    k_B, T, R_L = sp.symbols('k_B T R_L', positive=True)
    f_s, f_max = sp.symbols('f_s f_max', positive=True)
    I_sig = sp.Symbol('I_sig')
    I_ph = sp.Symbol('I_photocurrent')

    return {
        "Photocurrent":
            sp.Eq(I_ph, R * P),
        "GDD_fiber":
            sp.Eq(sp.Symbol('GDD'), D_sym * L * lam**2 / (2 * sp.pi * c_sym)),
        "Shot_noise_power":
            sp.Eq(sp.Symbol('i_shot^2'), 2 * q * I_sig * BW),
        "Thermal_noise_power":
            sp.Eq(sp.Symbol('i_th^2'), 4 * k_B * T * BW / R_L),
        "Nyquist_sampling":
            sp.Eq(f_s, 2 * f_max),
    }


if __name__ == "__main__":
    print("=== DISPERSION-ASSISTED GS RECEIVER: HARDWARE SIZING ===")

    print("\n--- How much SMF-28 fiber for |D| >= 5000 ps^2? ---")
    r = fiber_length_for_gdd(5000.0)
    print(f"  Need {r['length_m']:.0f} m ({r['length_km']:.3f} km) of SMF-28 at 1550 nm")

    print("\n--- Check 300 m spool ---")
    g = fiber_gdd(300.0)
    print(f"  300 m SMF-28: GDD = {g['gdd_ps2']:.0f} ps^2, "
          f"meets threshold: {g['meets_5000_ps2_threshold']}")

    print("\n--- Photodetector SNR at 1 mW optical power, 1 GHz BW ---")
    snr = photodetector_snr(optical_power_mW=1.0, bandwidth_GHz=1.0)
    print(f"  Signal current: {snr['I_signal_uA']:.1f} uA")
    print(f"  SNR:            {snr['SNR_dB']:.1f} dB")
    print(f"  Noise limited:  {snr['noise_limited_by']}")

    print("\n--- Sampling rate for 1 GHz signal bandwidth ---")
    sr = sampling_requirement(1.0)
    print(f"  Nyquist:   {sr['nyquist_rate_GSa_s']:.0f} GSa/s")
    print(f"  Practical: {sr['practical_rate_GSa_s']:.1f} GSa/s (use 2.5x for margin)")

    print_bom(BOM_MINIMUM_VIABLE, "MINIMUM VIABLE BENCH (<$3000)")
    print_bom(BOM_FULL_LAB, "FULL JALALI-LAB LEVEL SETUP")

    print("=== SymPy 5 ===")
    for k, eq in hardware_sympy_5().items():
        print(f"  {k}: {eq}")
