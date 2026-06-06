"""
_repl_spectrum_allocation.py

Government vs. non-government frequency allocation (NTIA/FCC).
US spectrum 3 kHz - 300 GHz: who owns what and why.
"""

import numpy as np

SEP = "=" * 65

# ------------------------------------------------------------------ #
# S1: NTIA / FCC SPLIT
# ------------------------------------------------------------------ #
print(SEP)
print("GOVERNMENT SPECTRUM ALLOCATION (US)")
print("NTIA = federal/government  |  FCC = non-government/commercial")
print(SEP)

print("""
  LEGAL BASIS:
    Communications Act 1934 + Telecommunications Act 1996
    NTIA (Dept of Commerce): manages federal government use
    FCC (independent agency): manages all other use
    ITU Radio Regulations: international treaty framework (binding)

  SPECTRUM RUNS:  3 kHz --> 300 GHz  (10 decades)
    Total allocatable: ~300 GHz of bandwidth

  ROUGH SPLIT BY ALLOCATED BANDWIDTH:
    Exclusive government:      ~13%
    Exclusive non-government:  ~19%
    SHARED (gov + non-gov):    ~68%
    -> most spectrum is SHARED, with priority rules

  Why so much sharing?
    Time-domain sharing: radar sweeps, then commercial fills gaps
    Geographic sharing: military base exclusion zones
    Power-limited sharing: commercial transmits low power near gov band
""")

# S2: Band-by-band allocation table
print(SEP)
print("BAND-BY-BAND ALLOCATION")
print(SEP)

bands = [
    # (name, f_low, f_high, unit, primary_user, notes)
    ("VLF",  3,    30,   "kHz",  "GOV (Navy)",    "submarine comms, OMEGA nav; seawater penetrates to ~10m"),
    ("LF",   30,   300,  "kHz",  "SHARED",        "LORAN-C (GOV), AM broadcast 153-279kHz (non-gov, ITU R1/R2)"),
    ("MF",   300,  3000, "kHz",  "SHARED",        "AM broadcast 535-1705kHz (non-gov); distress 500kHz (GOV)"),
    ("HF",   3,    30,   "MHz",  "SHARED",        "shortwave broadcast, ham, MILCOM; skywave propagation"),
    ("VHF",  30,   300,  "MHz",  "SHARED",        "FM 88-108MHz (non-gov); VHF aviation 108-137MHz (GOV/non-gov); TV"),
    ("UHF",  300,  3000, "MHz",  "SHARED",        "cellular 700/850/AWS (non-gov); GPS L1 1575MHz (GOV); 1-2GHz radar"),
    ("L",    1,    2,    "GHz",  "SHARED",        "GPS 1.176/1.228/1.575GHz (GOV); ATC secondary radar 1.03/1.09GHz"),
    ("S",    2,    4,    "GHz",  "SHARED",        "WiFi 2.4GHz (non-gov); weather radar 2.7-3GHz (GOV); Bluetooth"),
    ("C",    4,    8,    "GHz",  "SHARED",        "satellite downlink 3.7-4.2GHz (non-gov); military radar (GOV)"),
    ("X",    8,    12,   "GHz",  "GOV heavy",     "military radar (X-band SAR, fire control); some satellite"),
    ("Ku",   12,   18,   "GHz",  "SHARED",        "FSS satellite TV 12-12.7GHz (non-gov); some military SATCOM"),
    ("K",    18,   27,   "GHz",  "SHARED",        "5G mmW 24.25GHz (non-gov); 22.3GHz water vapor absorption (GOV radar)"),
    ("Ka",   27,   40,   "GHz",  "SHARED",        "5G/broadband sat (non-gov); high-res radar; 35GHz GOV radar"),
    ("V/W",  40,   110,  "GHz",  "MOSTLY GOV",    "60GHz O2 absorption (non-gov unlicensed ISM); 77GHz auto radar"),
    ("mm",   110,  300,  "GHz",  "MOSTLY GOV",    "EESS passive (weather sat sensing, GOV); atmospheric windows"),
]

print(f"  {'Band':<6} {'Range':<16} {'Primary':<14} Notes")
print("  " + "-" * 62)
for name, f_lo, f_hi, unit, user, note in bands:
    rng = f"{f_lo}-{f_hi} {unit}"
    print(f"  {name:<6} {rng:<16} {user:<14} {note[:52]}")

# S3: Key government-exclusive bands
print()
print(SEP)
print("GOVERNMENT-EXCLUSIVE ALLOCATIONS (selected)")
print(SEP)
print("""
  Frequency         Service              Why government-exclusive
  ---------------------------------------------------------------
  14.0-14.35 MHz    FIXED / MOBILE       HF gov comms (some shared)
  225-400 MHz       Military UHF SATCOM  DOD MILSATCOM (AEHF, MUOS)
  960-1215 MHz      Aeronautical nav     TACAN, DME, IFF transponders
  1215-1350 MHz     Radionavigation      GPS L2 (1227.6 MHz), radars
  1350-1390 MHz     Radiolocation        Air surveillance radar
  1755-1850 MHz     Federal broadband    DOD operations (being auctioned)
  3.1-3.5 GHz       Radiolocation        Shipborne/airborne radar
  5.25-5.925 GHz    Radiolocation        Airborne weather/mapping radar
  8.5-10.55 GHz     Radiolocation        X-band military radar (SAR, FC)
  13.4-14.0 GHz     Radiolocation        GOV Earth exploration satellite
  15.4-17.3 GHz     Fixed satellite      Military SATCOM
  36-37 GHz         Earth exploration    Passive sensor (weather sat)
  50.2-50.4 GHz     Earth exploration    PASSIVE -- no transmissions allowed
  52.6-59.3 GHz     Earth exploration    PASSIVE (O2 absorption window)
""")

# S4: Quantitative ratio calculation
print(SEP)
print("QUANTITATIVE RATIO (3 kHz - 300 GHz log-decade analysis)")
print(SEP)

# approximate MHz allocations from NTIA chart
# (gov_only, shared, nongov_only) in MHz bandwidth
allocation_table = [
    # band_name, f_low_MHz, f_high_MHz, frac_gov_excl, frac_shared, frac_nongov_excl
    ("VLF-LF",    0.003,    0.3,        0.70, 0.25, 0.05),
    ("LF-MF",     0.3,      3.0,        0.15, 0.50, 0.35),
    ("MF-HF",     3.0,      30,         0.10, 0.60, 0.30),
    ("HF-VHF",    30,       300,        0.12, 0.55, 0.33),
    ("VHF-UHF",   300,      3000,       0.18, 0.52, 0.30),
    ("UHF-SHF",   3000,     30000,      0.22, 0.55, 0.23),
    ("SHF-EHF",   30000,    300000,     0.28, 0.60, 0.12),
]

total_bw = 0
total_gov = 0
total_shared = 0
total_nongov = 0

print(f"  {'Band':<10} {'BW (MHz)':>12} {'Gov excl':>10} {'Shared':>10} {'NonGov':>10}")
print("  " + "-" * 54)
for name, f_lo, f_hi, fg, fs, fn in allocation_table:
    bw = f_hi - f_lo
    g = bw * fg; s = bw * fs; ng = bw * fn
    total_bw += bw; total_gov += g; total_shared += s; total_nongov += ng
    print(f"  {name:<10} {bw:>12.1f} {g:>10.1f} {s:>10.1f} {ng:>10.1f}")

print("  " + "-" * 54)
print(f"  {'TOTAL':<10} {total_bw:>12.1f} {total_gov:>10.1f} {total_shared:>10.1f} {total_nongov:>10.1f}")
print(f"\n  Gov exclusive:     {100*total_gov/total_bw:.1f}% of total bandwidth")
print(f"  Shared:            {100*total_shared/total_bw:.1f}%")
print(f"  Non-gov exclusive: {100*total_nongov/total_bw:.1f}%")
print(f"\n  Gov access (excl + half shared): {100*(total_gov + 0.5*total_shared)/total_bw:.1f}%")
print(f"  Non-gov access    (excl + half): {100*(total_nongov + 0.5*total_shared)/total_bw:.1f}%")

# S5: Connection to RogueGuard / photonic sensing
print()
print(SEP)
print("RELEVANCE TO PHOTONIC / FIBER SENSING")
print(SEP)
print("""
  Our RogueGuard system operates at optical frequencies:
    Telecom C-band: 1530-1565 nm -> 191.5-196.1 THz
    This is 5.6 THz of bandwidth -- > the ENTIRE radio spectrum

  Optical spectrum is NOT under NTIA/FCC:
    ITU-T G.694 defines DWDM channel grid (50/100 GHz spacing)
    No government exclusivity on optical fiber wavelengths
    BUT: fiber physical infrastructure is regulated (pole access, etc.)

  Why this matters for SBIR/DoD alignment:
    GPS L1/L2 (GOV exclusive): 1.176 / 1.575 GHz
    -> Our fiber rogue wave monitor can provide GPS-independent
       timing reference using optical frequency combs
    -> DoD FutureG needs resilient timing NOT dependent on GPS
    -> Optical timing: <1 fs jitter vs GPS ~30 ns -> 30,000x better

  Dispersion D and group delay spread:
    D [ps/(nm*km)] = -(lambda^2 / 2*pi*c) * GVD
    At 1550nm, SMF-28: D = +17 ps/(nm*km)
    GVD = 2*pi*c / lambda^2 * D = 2.18e-26 s^2/m
    Phase shift in GS retrieval: phi = pi * D * nu^2
    -> same equation governs ALL dispersive media (RF + optical)

  RF dispersive media:
    Plasma: n(omega) = sqrt(1 - omega_p^2/omega^2)
            -> GVD is NEGATIVE (anomalous) below plasma frequency
    Waveguide: n_eff(omega) = sqrt(1 - (omega_c/omega)^2)
            -> also negative GVD near cutoff

  FREQUENCY RATIO (musical / harmonic context):
    Octave:    f2/f1 = 2     (12 semitones)
    Fifth:     f2/f1 = 3/2   (7 semitones, Pythagorean)
    Bandwidth: BW ratio = f_high / f_low
    Spectrum 3kHz-300GHz: ratio = 1e8  (8 decades, 160 octaves)
    C-band optical 5.6THz: ratio = 1.03 (30 nm, narrow slice)
""")

# frequency ratios table
print("  Key frequency ratios:")
refs = [
    ("AM broadcast top/bottom", 1705e3, 535e3),
    ("FM broadcast top/bottom", 108e6, 88e6),
    ("GPS L1/L2",               1575.42e6, 1227.6e6),
    ("5G FR1 high/low",         7.125e9, 0.410e9),
    ("Telecom C-band top/bot",  196.1e12, 191.5e12),
    ("Optical vs RF spectrum",  300e12, 300e9),
]
for name, fh, fl in refs:
    ratio = fh / fl
    octaves = np.log2(ratio)
    print(f"  {name:<30} ratio={ratio:.3f}  ({octaves:.2f} octaves)")

print()
print(SEP)
print("Done.")
print(SEP)
