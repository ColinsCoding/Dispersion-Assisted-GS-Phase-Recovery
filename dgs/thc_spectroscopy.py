"""NIR photonics for THC% measurement via dispersive Fourier transform.

PHYSICAL BASIS:
  THC (Δ9-tetrahydrocannabinol, C₂₁H₃₀O₂, MW=314.46 g/mol) has characteristic
  near-infrared absorption peaks from C-H stretch overtones:
    ~1680 nm   — 1st overtone C-H stretch (main peak)
    ~1720 nm   — aromatic/aliphatic C-H combination
    ~2300 nm   — C-H combination band (requires InGaAs detector)
    ~1200 nm   — 2nd overtone (weaker, needs less dispersive fiber)

MEASUREMENT PRINCIPLE (JALALI TS-DFT):
  1. Broadband ultrashort laser pulse (1500–1900 nm, femtosecond)
  2. Pulse dispersed through single-mode fiber (SMF-28, GVD=17 ps/nm/km)
  3. Time-stretched pulse: time mapped to wavelength via GVD
  4. Pulse passes through cannabis sample (transmission measurement)
  5. Photodetector measures I(t) ~ |E(omega)|² with omega = t/(beta2*L)
  6. Beer-Lambert: A(omega) = -log10(I_sample/I_ref) = epsilon*c*L_cell

BEER-LAMBERT LAW:
  A(lambda) = epsilon(lambda) * C * L_cell
  where:
    A = absorbance (dimensionless)
    epsilon = molar absorptivity (L/(mol·cm))
    C = concentration (mol/L)
    L_cell = path length (cm)

  THC at 1680 nm: epsilon ~ 0.8 L/(mol·cm) (estimated from literature)
  Typical sample: 0.1–0.3 mol/L THC solution (20–100 mg/mL in ethanol)

PRINT OUTPUT:
  GS phase recovery -> reconstruct E(omega) from I(t) measurements
  -> Beer-Lambert inversion -> THC concentration in mg/g (% by weight)
  -> Report card: strain, THC%, CBD%, measurement uncertainty

3D PRINTING CONNECTION:
  The measurement device can be 3D printed:
  - SLA (resin) print: optical cell body (1 cm path, quartz windows)
  - FDM (PLA): housing for fiber connectors, photodiode mount
  - Hash function for printing: spatial hash assigns material density to voxels
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict, List, Optional, Tuple


# Physical constants
C_LIGHT    = 2.998e8        # m/s
H_PLANCK   = 6.626e-34      # J·s
N_AVOGADRO = 6.022e23       # mol⁻¹
MW_THC     = 314.46         # g/mol
MW_CBD     = 314.46         # g/mol (same molecular formula, isomers)


# ════════════════════════════════════════════════════════════════════════════
# §1  THC NIR ABSORPTION MODEL
# ════════════════════════════════════════════════════════════════════════════

def thc_absorption_spectrum(wavelength_nm: np.ndarray,
                              concentration_mgmL: float = 10.0,
                              path_cm: float = 1.0) -> Dict:
    """Simulated NIR absorption spectrum of THC in ethanol solution.

    Lorentzian peaks at known C-H overtone positions.
    Based on: Sanchez et al. (2014), Rodriguez-Saona et al. (2010).

    Parameters
    ----------
    wavelength_nm       : wavelength grid (nm)
    concentration_mgmL  : THC concentration (mg/mL = g/L)
    path_cm             : cell path length (cm)
    """
    lam = wavelength_nm
    # THC absorption peaks (nm, width_nm, molar absorptivity L/(mol*cm))
    # Estimated from NIR spectroscopy of similar cannabinoids
    peaks = [
        (1680, 25, 0.80),   # 1st overtone C-H stretch (ring)
        (1724, 20, 0.55),   # aliphatic C-H combination
        (1760, 15, 0.30),   # aromatic overtone
        (1208, 18, 0.25),   # 2nd overtone (weaker)
        (1393, 22, 0.40),   # O-H combination (from ethanol solvent)
    ]
    # CBD has different peak ratios (isomers differ in ring structure)
    cbd_peaks = [
        (1670, 28, 0.72),
        (1718, 22, 0.50),
        (1755, 16, 0.28),
        (1205, 20, 0.22),
    ]

    # Molar concentration
    C_mol_L = (concentration_mgmL / 1000) / MW_THC * 1000   # mol/L

    # Beer-Lambert absorbance
    A_thc = np.zeros_like(lam, dtype=float)
    for lam0, dlam, eps in peaks:
        # Lorentzian: A = eps*C*L * (dlam/2)^2 / ((lam-lam0)^2 + (dlam/2)^2)
        A_thc += eps * C_mol_L * path_cm * (dlam/2)**2 / ((lam - lam0)**2 + (dlam/2)**2)

    # Ethanol background (broad absorption)
    A_ethanol = 0.05 * np.exp(-((lam - 1450)/200)**2)

    A_total = A_thc + A_ethanol
    T = 10**(-A_total)   # transmittance

    return {
        "wavelength_nm":      lam,
        "A_total":            A_total,
        "A_thc":              A_thc,
        "A_ethanol":          A_ethanol,
        "transmittance":      T,
        "concentration_mgmL": concentration_mgmL,
        "C_mol_L":            C_mol_L,
        "path_cm":            path_cm,
        "peak_wavelengths_nm": [p[0] for p in peaks],
        "A_at_1680nm":        float(np.interp(1680, lam, A_thc)),
        "LOD_mgmL":           0.1,   # limit of detection (estimated)
        "LOQ_mgmL":           0.5,   # limit of quantification
    }


def beer_lambert_inversion(A_measured: np.ndarray,
                            wavelength_nm: np.ndarray,
                            path_cm: float = 1.0,
                            peak_nm: float = 1680.0,
                            epsilon_L_molcm: float = 0.80) -> Dict:
    """Recover THC concentration from measured absorbance.

    A = epsilon * C * L  =>  C = A / (epsilon * L)

    For multi-wavelength: use least-squares fit to the full spectrum.
    Single-wavelength: just invert at peak.
    """
    # Single-wavelength inversion at peak
    A_peak = float(np.interp(peak_nm, wavelength_nm, A_measured))
    C_mol_L = A_peak / (epsilon_L_molcm * path_cm)
    C_mgmL  = C_mol_L * MW_THC * 1000 / 1000   # mg/mL

    # Multi-wavelength least squares (more robust)
    # Build reference spectrum at 1 mg/mL
    A_ref_1mgmL = thc_absorption_spectrum(
        wavelength_nm, concentration_mgmL=1.0, path_cm=path_cm
    )["A_thc"]
    # Normalize: A_measured = C * A_ref_1mgmL + background
    # Least squares: C = (A_ref^T A_measured) / (A_ref^T A_ref)
    A_ref = A_ref_1mgmL
    C_lstsq = float(np.dot(A_ref, A_measured) / (np.dot(A_ref, A_ref) + 1e-300))

    # Uncertainty estimate (photon noise, 1% noise level)
    noise_level = 0.01
    sigma_A = noise_level * np.sqrt(np.abs(A_measured) + 0.001)
    sigma_C_mgmL = float(np.sqrt(np.mean((sigma_A / (epsilon_L_molcm * path_cm))**2))) * MW_THC

    return {
        "C_mol_L_peak":    C_mol_L,
        "C_mgmL_peak":     C_mgmL,
        "C_mgmL_lstsq":    C_lstsq,
        "sigma_C_mgmL":    sigma_C_mgmL,
        "A_peak":          A_peak,
        "peak_nm":         peak_nm,
        "epsilon":         epsilon_L_molcm,
        "path_cm":         path_cm,
    }


# ════════════════════════════════════════════════════════════════════════════
# §2  TS-DFT MEASUREMENT SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def simulate_tsdft_measurement(concentration_mgmL: float = 15.0,
                                 rep_rate_GHz: float = 1.0,
                                 L_fiber_km: float = 10.0,
                                 n_avg: int = 1000,
                                 seed: int = 42) -> Dict:
    """Simulate TS-DFT NIR spectroscopy measurement for THC.

    Ultrafast laser -> dispersive fiber -> sample -> detector -> spectrum.
    Time-domain signal I(t) is Fourier-mapped to wavelength via GVD.

    GVD of SMF-28 at 1550nm: beta2 = -22 ps²/km = -22e-27 s²/m
    Group velocity dispersion: D = 17 ps/(nm·km)
    """
    rng = np.random.default_rng(seed)

    # Wavelength axis (1500–1900 nm)
    n_pts = 2048
    lam_nm = np.linspace(1500, 1900, n_pts)

    # Time axis via GVD mapping: t = beta2 * L * omega -> lambda mapping
    beta2 = -22e-27   # s²/m
    L_m   = L_fiber_km * 1e3
    # Dispersion D [ps/(nm·km)]: D = -2*pi*c/lam^2 * beta2
    lam0_m  = 1700e-9
    D_ps_nm_km = -2 * np.pi * C_LIGHT * beta2 * 1e27 / lam0_m**2   # ps/(nm·km)
    dt_dlam_ps_nm = D_ps_nm_km * L_fiber_km   # ps/nm for the full fiber

    # Gaussian input pulse (transform-limited, 100 fs FWHM)
    T0_fs = 100.0
    T0_s  = T0_fs * 1e-15
    E0    = np.exp(-((lam_nm - 1700)/50)**2)   # spectral envelope

    # THC absorption spectrum
    thc_spec = thc_absorption_spectrum(lam_nm, concentration_mgmL, path_cm=1.0)
    T_sample = thc_spec["transmittance"]

    # Reference and sample spectra
    I_ref    = E0**2
    I_sample = E0**2 * T_sample

    # Add shot noise: sigma² = I * (1/sqrt(n_avg))
    snr_shot = np.sqrt(n_avg * 1e6)   # 1M photons/pulse, n_avg averaged
    noise_ref    = rng.standard_normal(n_pts) * np.sqrt(I_ref)    / snr_shot
    noise_sample = rng.standard_normal(n_pts) * np.sqrt(I_sample) / snr_shot
    I_ref_noisy    = np.maximum(I_ref    + noise_ref,    1e-10)
    I_sample_noisy = np.maximum(I_sample + noise_sample, 1e-10)

    # Measured absorbance
    A_measured = -np.log10(I_sample_noisy / (I_ref_noisy + 1e-10))

    # Invert to get concentration
    inv = beer_lambert_inversion(A_measured, lam_nm, path_cm=1.0)

    return {
        "wavelength_nm":      lam_nm,
        "I_ref":              I_ref_noisy,
        "I_sample":           I_sample_noisy,
        "A_measured":         A_measured,
        "A_true":             thc_spec["A_thc"],
        "C_true_mgmL":        concentration_mgmL,
        "C_measured_mgmL":    inv["C_mgmL_lstsq"],
        "sigma_C_mgmL":       inv["sigma_C_mgmL"],
        "error_pct":          abs(inv["C_mgmL_lstsq"] - concentration_mgmL) / concentration_mgmL * 100,
        "beta2_s2m":          beta2,
        "L_fiber_km":         L_fiber_km,
        "dt_dlam_ps_nm":      dt_dlam_ps_nm,
        "rep_rate_GHz":       rep_rate_GHz,
        "spectra_per_sec":    rep_rate_GHz * 1e9,
        "n_avg":              n_avg,
    }


# ════════════════════════════════════════════════════════════════════════════
# §3  3D SPATIAL HASH (FOR PRINTING REPORT + VOXELIZATION)
# ════════════════════════════════════════════════════════════════════════════

def spatial_hash_3d(points: np.ndarray,
                     cell_size: float = 1.0) -> Dict:
    """3D spatial hash: maps (x,y,z) coordinates to voxel grid index.

    Used for:
    - 3D printing: slicing a point cloud into voxel layers
    - Spectroscopy: mapping (lambda, intensity, time) to a 3D histogram
    - Collision detection in robotics

    Hash function: h(i,j,k) = (i*p1 + j*p2 + k*p3) mod table_size
    where p1=73856093, p2=19349663, p3=83492791 (Teschner 2003).
    This gives O(1) lookup with low collision rate.

    Parameters
    ----------
    points    : (N, 3) array of (x,y,z) coordinates
    cell_size : voxel size
    """
    pts = np.asarray(points, dtype=float)
    N = len(pts)

    # Voxel indices
    ijk = np.floor(pts / cell_size).astype(int)

    # Hash primes (Teschner et al. 2003)
    P1, P2, P3 = 73856093, 19349663, 83492791
    table_size  = max(1024, 2 * N)
    h = ((ijk[:, 0] * P1) ^ (ijk[:, 1] * P2) ^ (ijk[:, 2] * P3)) % table_size

    # Build hash table: bucket -> list of point indices
    table: dict = {}
    for idx in range(N):
        bucket = int(h[idx])
        if bucket not in table:
            table[bucket] = []
        table[bucket].append(idx)

    # Occupied voxels
    voxel_set = {(ijk[i, 0], ijk[i, 1], ijk[i, 2]) for i in range(N)}

    # Stats
    bucket_sizes = [len(v) for v in table.values()]
    return {
        "voxel_indices":    ijk,
        "hashes":           h,
        "table":            table,
        "n_occupied_voxels": len(voxel_set),
        "n_buckets":         len(table),
        "max_bucket_size":   max(bucket_sizes) if bucket_sizes else 0,
        "mean_bucket_size":  float(np.mean(bucket_sizes)) if bucket_sizes else 0,
        "load_factor":       len(table) / table_size,
        "cell_size":         cell_size,
    }


def voxelize_for_printing(A_spectrum: np.ndarray,
                           wavelength_nm: np.ndarray,
                           concentration_mgmL: float,
                           thc_pct: float) -> Dict:
    """Convert spectral measurement to 3D printable report card.

    Maps the NIR spectrum to a 3D surface that can be printed:
    - X axis: wavelength (nm)
    - Y axis: absorbance
    - Z axis: time series / batch index
    The 3D surface extruded by 2mm gives a physical record.

    Also generates text report for labeling.
    """
    # 2D surface: wavelength × absorbance
    n = len(wavelength_nm)
    x = wavelength_nm
    y = A_spectrum
    z = np.zeros(n)  # single measurement, z=0

    points = np.column_stack([
        (x - x.min()) / (x.max() - x.min()) * 50,   # scale to 50mm
        y / (y.max() + 1e-6) * 20,                    # scale to 20mm
        z
    ])

    h = spatial_hash_3d(points, cell_size=0.5)

    report = (
        f"=== THC NIR SPECTROSCOPY REPORT ===\n"
        f"  Concentration: {concentration_mgmL:.1f} mg/mL\n"
        f"  THC%:          {thc_pct:.1f}%\n"
        f"  Peak A(1680nm): {float(np.interp(1680, wavelength_nm, A_spectrum)):.4f}\n"
        f"  Method: TS-DFT NIR + Beer-Lambert inversion\n"
        f"  Voxels for print: {h['n_occupied_voxels']}\n"
        f"  Print volume: 50mm × 20mm × 2mm\n"
        "==================================="
    )
    return {
        "points_mm":       points,
        "spatial_hash":    h,
        "report_text":     report,
        "thc_pct":         thc_pct,
        "concentration_mgmL": concentration_mgmL,
    }


# ════════════════════════════════════════════════════════════════════════════
# §4  FULL PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def thc_measurement_pipeline(sample_thc_pct: float = 20.0,
                               density_g_mL: float = 0.95,
                               extraction_efficiency: float = 0.90) -> Dict:
    """Full pipeline: THC% -> NIR measurement -> Beer-Lambert -> report.

    Cannabis flower typically 15–30% THC by dry weight.
    Extraction in ethanol (common solvent for NIR):
      concentration [mg/mL] = THC_pct/100 * density * 1000 * extraction_efficiency
    """
    # Convert THC% to solution concentration
    concentration_mgmL = (sample_thc_pct / 100) * density_g_mL * 1000 * extraction_efficiency

    # Run TS-DFT simulation
    meas = simulate_tsdft_measurement(concentration_mgmL, n_avg=1000)

    # Convert measured concentration back to %
    C_measured_mgmL = meas["C_measured_mgmL"]
    thc_pct_measured = C_measured_mgmL / (density_g_mL * 1000 * extraction_efficiency) * 100

    # Voxelize for printing
    viz = voxelize_for_printing(meas["A_measured"], meas["wavelength_nm"],
                                 C_measured_mgmL, thc_pct_measured)
    return {
        "sample_thc_pct":      sample_thc_pct,
        "concentration_mgmL":  concentration_mgmL,
        "measured_thc_pct":    thc_pct_measured,
        "sigma_thc_pct":       meas["sigma_C_mgmL"] / (density_g_mL * 1000 * extraction_efficiency) * 100,
        "error_pct_abs":       abs(thc_pct_measured - sample_thc_pct),
        "wavelength_nm":       meas["wavelength_nm"],
        "A_measured":          meas["A_measured"],
        "spectra_per_sec":     meas["spectra_per_sec"],
        "report":              viz["report_text"],
        "print_voxels":        viz["spatial_hash"]["n_occupied_voxels"],
        "measurement":         meas,
        "voxel_data":          viz,
    }


# ════════════════════════════════════════════════════════════════════════════
# §5  SYMPY: 5 KEY EQUATIONS
# ════════════════════════════════════════════════════════════════════════════

def thc_spectroscopy_sympy_5() -> Dict:
    """5 equations: Beer-Lambert, GVD mapping, SNR, hash function, THC%."""
    A, eps, C, L_cell = sp.symbols("A epsilon C L", positive=True)
    lam, beta2, omega, L_fib = sp.symbols("lambda beta2 omega L_fiber", real=True)
    N_ph = sp.Symbol("N_ph", positive=True)
    i, j, k = sp.symbols("i j k", integer=True)
    # 1. Beer-Lambert
    eq1 = sp.Eq(A, eps * C * L_cell)
    # 2. GVD time-frequency mapping
    eq2 = sp.Eq(sp.Symbol("t"), beta2 * L_fib * omega)
    # 3. Shot-noise SNR
    eq3 = sp.Eq(sp.Symbol("SNR"), sp.sqrt(N_ph))
    # 4. 3D spatial hash (Teschner 2003)
    P1, P2, P3, M = sp.symbols("P1 P2 P3 M", integer=True, positive=True)
    eq4 = sp.Eq(sp.Symbol("h(i,j,k)"),
                sp.Mod(i*P1 + j*P2 + k*P3, M))
    # 5. THC% from concentration
    rho, eta = sp.symbols("rho eta", positive=True)
    eq5 = sp.Eq(sp.Symbol("THC%"),
                C / (rho * 1000 * eta) * 100)
    return {
        "Beer_Lambert":    eq1,
        "GVD_mapping":     eq2,
        "Shot_noise_SNR":  eq3,
        "Spatial_hash":    eq4,
        "THC_percent":     eq5,
    }


if __name__ == "__main__":
    print("=== THC NIR Absorption Spectrum ===")
    lam = np.linspace(1500, 1900, 1000)
    spec = thc_absorption_spectrum(lam, concentration_mgmL=20.0, path_cm=1.0)
    print(f"  A(1680 nm) = {spec['A_at_1680nm']:.4f}")
    print(f"  Max absorbance at lambda = {lam[np.argmax(spec['A_thc'])]:.0f} nm")
    print(f"  LOD = {spec['LOD_mgmL']} mg/mL")

    print("\n=== TS-DFT Measurement Simulation ===")
    meas = simulate_tsdft_measurement(concentration_mgmL=15.0, n_avg=1000)
    print(f"  True:     15.0 mg/mL")
    print(f"  Measured: {meas['C_measured_mgmL']:.2f} mg/mL")
    print(f"  Error:    {meas['error_pct']:.2f}%")
    print(f"  dt/dlam:  {meas['dt_dlam_ps_nm']:.1f} ps/nm  (maps time -> wavelength)")
    print(f"  Spectra/sec: {meas['spectra_per_sec']:.2e}")

    print("\n=== Full Pipeline: 20% THC flower ===")
    pipe = thc_measurement_pipeline(sample_thc_pct=20.0)
    print(pipe["report"])
    print(f"  True THC%: {pipe['sample_thc_pct']:.1f}%")
    print(f"  Measured:  {pipe['measured_thc_pct']:.1f}% ± {pipe['sigma_thc_pct']:.1f}%")
    print(f"  Abs error: {pipe['error_pct_abs']:.2f}%")

    print("\n=== 3D Spatial Hash ===")
    pts = np.random.default_rng(0).standard_normal((500, 3))
    h = spatial_hash_3d(pts, cell_size=0.5)
    print(f"  N points: 500")
    print(f"  Occupied voxels: {h['n_occupied_voxels']}")
    print(f"  Load factor: {h['load_factor']:.3f}")
    print(f"  Max bucket: {h['max_bucket_size']}")
