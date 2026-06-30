"""CCD / CMOS photonic sensor physics.

Covers the full signal chain from photon to ADU:
  photon -> photoelectric effect -> electron -> charge well -> voltage -> ADC code

Key physics:
  QE(lambda): quantum efficiency, peaks ~550-700 nm for Si, zero above ~1100 nm
  Full-well capacity: maximum electrons before saturation
  Shot noise: sqrt(N_e) from Poisson statistics
  Read noise: floor set by amplifier (electrons RMS)
  Dark current: thermally generated electrons (halves every ~7 deg C drop)
  SNR = signal / sqrt(shot^2 + read^2 + dark^2)

Optics stage:
  f/# = f / D:  aperture number
  Airy disk radius: r_Airy = 1.22 * lambda * f/#
  PSF (Fraunhofer): J1 Bessel pattern computed via numpy recurrence
  Nyquist: pixel pitch p >= r_Airy (otherwise sampling the PSF too coarsely)

Product rule: photocurrent = integral[ QE(lambda) * B(lambda,T) * A_pixel ] dlambda
  Uses numerical_product_rule from dgs.blackbody for verification.

Embedded C interface at end: ADC register map + SPI readout pseudocode
  mirroring what firmware/ccd_driver.c would contain.
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict, Tuple, Optional

# ── Physical constants ────────────────────────────────────────────────────────
Q_E        = 1.602176634e-19   # C
H_PLANCK   = 6.62607015e-34   # J*s
C_LIGHT    = 2.99792458e8     # m/s
K_BOLTZ    = 1.380649e-23     # J/K
HC_EV_NM   = 1239.84193       # eV*nm


# ── Silicon QE model ─────────────────────────────────────────────────────────
def si_quantum_efficiency(wavelength_nm: float | np.ndarray,
                           peak_qe: float = 0.65) -> float | np.ndarray:
    """Empirical QE(lambda) for front-illuminated silicon CCD.

    Model: trapezoidal response
      < 350 nm: UV suppression (absorption in poly-Si gate)
      350-550:  linear rise to peak
      550-750:  flat top at peak_qe
      750-1100: linear fall (bandgap cutoff at ~1100 nm = 1.12 eV)
      > 1100:   zero (Si bandgap)

    Real CCDs have AR coatings; back-illuminated reach ~95% peak QE.
    """
    lam = np.asarray(wavelength_nm, dtype=float)
    qe  = np.zeros_like(lam)

    mask1 = (lam >= 350) & (lam < 550)
    qe[mask1] = peak_qe * (lam[mask1] - 350) / 200

    mask2 = (lam >= 550) & (lam <= 750)
    qe[mask2] = peak_qe

    mask3 = (lam > 750) & (lam <= 1100)
    qe[mask3] = peak_qe * (1100 - lam[mask3]) / 350

    return float(qe) if np.ndim(wavelength_nm) == 0 else qe


def si_bandgap_eV(T_K: float = 300.0) -> float:
    """Silicon bandgap vs temperature (Varshni formula).

    E_g(T) = E_g(0) - alpha*T^2 / (T + beta)
    E_g(0)=1.1702 eV, alpha=4.73e-4 eV/K, beta=636 K  (Sze & Ng)
    """
    E_g0  = 1.1702
    alpha = 4.73e-4
    beta  = 636.0
    return E_g0 - alpha * T_K**2 / (T_K + beta)


def si_cutoff_wavelength_nm(T_K: float = 300.0) -> float:
    """Wavelength at Si bandgap: lambda_cut = hc/Eg = 1240/Eg(eV) nm."""
    return HC_EV_NM / si_bandgap_eV(T_K)


# ── CCD pixel model ───────────────────────────────────────────────────────────
class CCDPixel:
    """Single CCD pixel noise and SNR model.

    Parameters
    ----------
    full_well_e : int
        Full-well capacity in electrons (typical 20 000 – 100 000 e⁻).
    read_noise_e : float
        RMS read noise in electrons (typical 3–15 e⁻).
    dark_current_e_per_s : float
        Dark current in e⁻/s at operating temperature.
    gain_e_per_adu : float
        Conversion gain: electrons per ADU (analog-to-digital unit).
    bit_depth : int
        ADC bit depth (8, 12, 14, or 16 bit).
    pixel_pitch_um : float
        Physical pixel size in micrometers.
    """

    def __init__(self, full_well_e: int = 50_000,
                 read_noise_e: float = 5.0,
                 dark_current_e_per_s: float = 0.1,
                 gain_e_per_adu: float = 1.5,
                 bit_depth: int = 14,
                 pixel_pitch_um: float = 5.6):
        if full_well_e <= 0:
            raise ValueError("full_well_e must be positive")
        if read_noise_e < 0:
            raise ValueError("read_noise_e must be non-negative")
        self.full_well_e    = full_well_e
        self.read_noise_e   = read_noise_e
        self.dark_current   = dark_current_e_per_s
        self.gain           = gain_e_per_adu
        self.bit_depth      = bit_depth
        self.pixel_pitch_um = pixel_pitch_um
        self.adu_max        = (1 << bit_depth) - 1

    def signal_electrons(self, photon_flux: float, qe: float,
                          exposure_s: float) -> float:
        """N_e = photon_flux * QE * t_exp (clipped to full-well)."""
        return min(photon_flux * qe * exposure_s, self.full_well_e)

    def dark_electrons(self, exposure_s: float) -> float:
        return self.dark_current * exposure_s

    def snr(self, photon_flux: float, qe: float,
            exposure_s: float) -> Dict[str, float]:
        """Full SNR calculation including shot, dark, and read noise.

        SNR = N_sig / sqrt(N_sig + N_dark + N_read^2)

        Returns dict with signal, noise breakdown, and SNR in dB.
        """
        N_sig  = self.signal_electrons(photon_flux, qe, exposure_s)
        N_dark = self.dark_electrons(exposure_s)
        N_read = self.read_noise_e

        shot_var = N_sig + N_dark   # Poisson variance
        total_noise = np.sqrt(shot_var + N_read**2)
        snr_linear = N_sig / total_noise if total_noise > 0 else 0.0
        snr_db     = 20 * np.log10(snr_linear) if snr_linear > 0 else -np.inf

        adu_value = int(N_sig / self.gain)
        saturated = N_sig >= self.full_well_e

        return {
            "N_signal_e":    N_sig,
            "N_dark_e":      N_dark,
            "shot_noise_e":  np.sqrt(shot_var),
            "read_noise_e":  N_read,
            "total_noise_e": total_noise,
            "SNR_linear":    snr_linear,
            "SNR_dB":        snr_db,
            "ADU_value":     min(adu_value, self.adu_max),
            "saturated":     saturated,
            "dynamic_range_dB": 20 * np.log10(self.full_well_e / N_read)
                                 if N_read > 0 else np.inf,
        }

    def dark_doubling_temp(self) -> float:
        """Dark current doubles every ~7 deg C (rule of thumb for Si)."""
        return 7.0

    def electrons_to_adu(self, n_electrons: float) -> int:
        """Convert electrons to ADU via gain."""
        return int(min(n_electrons / self.gain, self.adu_max))


# ── Optics stage ──────────────────────────────────────────────────────────────
def airy_disk_radius_um(wavelength_nm: float, f_number: float) -> float:
    """Radius of first Airy dark ring: r = 1.22 * lambda * f/#.

    This is the diffraction-limited resolution of a circular aperture.
    For Nyquist sampling: pixel pitch should be <= r_Airy.
    """
    if np.ndim(f_number) == 0 and f_number <= 0:
        raise ValueError("f_number must be positive")
    return 1.22 * wavelength_nm * 1e-3 * f_number   # um


def psf_airy_pattern(r_um: np.ndarray, wavelength_nm: float,
                      f_number: float) -> np.ndarray:
    """Normalised Airy PSF I(r) = [2*J1(x)/x]^2.

    J1 computed via numpy (no scipy): uses 5-term Taylor series for small x,
    asymptotic expansion for large x.
    """
    def j1_numpy(x):
        """J1(x) via Abramowitz & Stegun polynomial approx (error < 5e-7).
        Polynomial for |x|<=3, asymptotic expansion for |x|>3.
        No scipy required.
        """
        x  = np.asarray(x, dtype=float)
        ax = np.abs(x)
        result = np.zeros_like(x)
        sign   = np.sign(x); sign[sign == 0] = 1.0

        # -- small x: A&S 9.4.2 polynomial in (x/3)^2 --
        lo = ax <= 3.0
        if np.any(lo):
            t  = (ax[lo] / 3.0)**2
            p  = (0.50000000 + t*(-0.56249985 + t*(0.21093573
                 + t*(-0.03954289 + t*(0.00443319
                 + t*(-0.00031761 + t*0.00001109))))))
            result[lo] = sign[lo] * ax[lo] * p

        # -- large x: asymptotic form A&S 9.4.2 --
        hi = ~lo
        if np.any(hi):
            t  = 3.0 / ax[hi]
            f1 = (0.79788456 + t*(0.00000156 + t*(0.01659667
                 + t*(0.00017105 + t*(-0.00249511
                 + t*(0.00113653 + t*(-0.00020033)))))))
            th = (ax[hi] - 2.35619449
                  + t*(0.12499612 + t*(0.00005650
                  + t*(-0.00637879 + t*(0.00074348
                  + t*(0.00079824 + t*(-0.00029166)))))))
            result[hi] = sign[hi] * f1 * np.cos(th) / np.sqrt(ax[hi])
        return result

    lam_um = wavelength_nm * 1e-3
    k      = 2 * np.pi / lam_um
    NA     = 1 / (2 * f_number)    # numerical aperture (in air)
    x      = k * NA * np.abs(r_um)
    J1x    = j1_numpy(x)
    # avoid divide-by-zero at r=0
    with np.errstate(invalid="ignore"):
        airy = np.where(np.abs(x) < 1e-10, 1.0, (2 * J1x / x)**2)
    return airy / airy.max()


def check_nyquist_sampling(pixel_pitch_um: float, wavelength_nm: float,
                             f_number: float) -> Dict:
    """Check if pixel pitch satisfies Nyquist for the Airy PSF.

    Nyquist criterion: pixel_pitch <= r_Airy / 2
    (sample at twice the Nyquist frequency of the PSF cutoff).
    """
    r_airy = airy_disk_radius_um(wavelength_nm, f_number)
    nyquist_limit = r_airy / 2
    oversampled   = pixel_pitch_um <= nyquist_limit
    ratio         = pixel_pitch_um / r_airy
    return {
        "r_airy_um":       r_airy,
        "nyquist_limit_um": nyquist_limit,
        "pixel_pitch_um":  pixel_pitch_um,
        "pitch_to_airy":   ratio,
        "nyquist_ok":      oversampled,
        "recommendation":  "OK" if oversampled else
                           f"UNDERSAMPLED -- use f/# < {f_number*ratio:.1f}",
    }


# ── Photocurrent integral: product rule application ───────────────────────────
def photocurrent_from_blackbody(T_K: float,
                                  pixel_area_um2: float = 5.6**2,
                                  lam_min_nm: float = 300.0,
                                  lam_max_nm: float = 1100.0,
                                  n_pts: int = 2000) -> Dict:
    """Photocurrent I = integral[ QE(lam) * B(lam,T) * A_pixel ] dlam.

    This is the product rule in action:
      d/dlam[ QE(lam) * B(lam,T) ] = QE * dB/dlam + B * dQE/dlam

    The spectral photocurrent density is QE(lam) * B(lam,T),
    and the integral gives total photocurrent.

    Returns signal electrons/s and the integrand spectrum.
    """
    try:
        from dgs.blackbody import planck_radiance
    except ModuleNotFoundError:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from dgs.blackbody import planck_radiance

    lam  = np.linspace(lam_min_nm, lam_max_nm, n_pts)
    dlam = float(lam[1] - lam[0])
    B    = planck_radiance(lam, T_K)          # W/(m^2 sr nm)
    qe   = si_quantum_efficiency(lam)

    integrand = qe * B                         # spectral response W/(m^2 sr nm)
    A_m2      = pixel_area_um2 * 1e-12         # um^2 -> m^2
    # solid angle acceptance: assume pi sr (Lambertian source, f/2 lens ~ 0.2 sr, approx pi)
    omega_sr  = np.pi
    power_W   = np.trapezoid(integrand, lam) * A_m2 * omega_sr

    E_mean_J  = H_PLANCK * C_LIGHT / (np.mean(lam[qe > 0.01]) * 1e-9)
    e_per_s   = power_W / E_mean_J

    # Leibniz product derivative d/dlam[QE*B]
    d_integrand = np.gradient(integrand, dlam)
    leibniz_check = (qe * np.gradient(B, dlam) +
                     B * np.gradient(qe, dlam))
    residual = np.abs(d_integrand - leibniz_check)

    return {
        "lam_nm":         lam,
        "QE":             qe,
        "B_lam":          B,
        "integrand":      integrand,
        "d_integrand":    d_integrand,
        "leibniz_check":  leibniz_check,
        "product_rule_max_residual": float(residual.max()),
        "power_W":        power_W,
        "electrons_per_s": e_per_s,
        "T_K":            T_K,
    }


# ── SymPy: CCD signal chain equations ────────────────────────────────────────
def ccd_signal_chain_sympy() -> Dict[str, sp.Expr]:
    """5 key CCD signal-chain equations for sp.init_printing display."""
    N_ph, qe, t, I_dark, sigma_r = sp.symbols(
        "N_ph QE t I_dark sigma_r", positive=True
    )
    N_sig  = N_ph * qe * t
    N_dark = I_dark * t
    sigma_shot = sp.sqrt(N_sig + N_dark)
    SNR = N_sig / sp.sqrt(N_sig + N_dark + sigma_r**2)

    lam, h, c = sp.symbols("lambda h c", positive=True)
    E_g0, alpha, beta_, T_K = sp.symbols("E_{g0} alpha beta T", positive=True)
    E_g_T = E_g0 - alpha * T_K**2 / (T_K + beta_)

    QE_sym, B_sym = sp.Function("QE"), sp.Function("B")
    I_ph = sp.Integral(QE_sym(lam) * B_sym(lam), lam)

    return {
        "signal_electrons":  sp.Eq(sp.Symbol("N_sig"), N_sig),
        "SNR":               sp.Eq(sp.Symbol("SNR"), SNR),
        "shot_noise":        sp.Eq(sp.Symbol("sigma_shot"), sigma_shot),
        "Si_bandgap_Varshni": sp.Eq(sp.Symbol("E_g"), E_g_T),
        "photocurrent_integral": sp.Eq(sp.Symbol("I_ph"), I_ph),
    }


# ── Embedded C ADC / SPI interface description ────────────────────────────────
EMBEDDED_C_ADC_PSEUDOCODE = """\
/* CCD pixel readout -- embedded C register map (pseudocode)
 * Target: ARM Cortex-M4, SPI @ 10 MHz, 14-bit ADC
 * Physical layer: shift register drains charge packet -> ADC -> SPI
 */

#include <stdint.h>

#define CCD_PIXELS_H   3008u
#define CCD_PIXELS_V   2008u
#define ADC_BITS       14u
#define ADC_MAX        ((1u << ADC_BITS) - 1u)   /* 16383 */
#define DARK_COLS      24u    /* optical black columns for offset calibration */

/* SPI frame: 16-bit word, upper 2 bits = channel tag, lower 14 = ADU */
typedef uint16_t adu_t;

/* Shift a single charge packet from CCD register into ADC.
 * H_CLK: horizontal clock, OG: output gate, R: reset gate
 * Returns raw ADU value (0 - ADC_MAX).
 */
static inline adu_t ccd_read_pixel(void) {
    GPIO_PULSE(H_CLK);          /* transfer charge to output node */
    GPIO_PULSE(OG);             /* open output gate */
    uint16_t raw = SPI_READ16() & ADC_MAX;
    GPIO_PULSE(R);              /* reset output capacitor (correlated double sampling) */
    return (adu_t)raw;
}

/* Read one full row into buffer.  CDS (correlated double sampling) subtracts
 * reset level from signal level to cancel kTC noise.
 */
void ccd_read_row(adu_t *row_buf, uint16_t n_pixels) {
    for (uint16_t i = 0; i < n_pixels; i++) {
        adu_t reset_level = ccd_read_pixel();   /* sample reset */
        GPIO_PULSE(SIGNAL_GATE);
        adu_t signal_level = ccd_read_pixel();  /* sample signal */
        /* CDS: subtract reset to eliminate kTC noise */
        row_buf[i] = (signal_level > reset_level) ?
                     (signal_level - reset_level) : 0u;
    }
}

/* Dark current correction: subtract mean of optical black columns */
void apply_dark_correction(adu_t *row_buf, uint16_t n_pixels) {
    uint32_t dark_sum = 0;
    for (uint16_t i = 0; i < DARK_COLS; i++)
        dark_sum += row_buf[i];
    uint16_t dark_offset = (uint16_t)(dark_sum / DARK_COLS);
    for (uint16_t i = DARK_COLS; i < n_pixels; i++) {
        row_buf[i] = (row_buf[i] > dark_offset) ?
                     (row_buf[i] - dark_offset) : 0u;
    }
}
"""


def embedded_c_summary() -> str:
    return EMBEDDED_C_ADC_PSEUDOCODE


if __name__ == "__main__":
    print("=== Silicon QE model ===")
    for lam, label in [(350,"UV edge"), (550,"green peak"), (700,"red"),
                        (900,"NIR"), (1100,"Si cutoff"), (1200,"beyond Si")]:
        q = si_quantum_efficiency(lam)
        print(f"  {label:12s} {lam:5d} nm -> QE = {q:.3f}")

    print(f"\n=== Si bandgap (Varshni) ===")
    for T in [77, 200, 300, 400]:
        Eg = si_bandgap_eV(T)
        lam_cut = si_cutoff_wavelength_nm(T)
        print(f"  T={T:4d} K: Eg={Eg:.4f} eV, cutoff={lam_cut:.1f} nm")

    print("\n=== CCD pixel SNR ===")
    px = CCDPixel(full_well_e=50_000, read_noise_e=5.0,
                   dark_current_e_per_s=0.05, gain_e_per_adu=1.5, bit_depth=14)
    for flux in [100, 1000, 10000, 50000]:
        res = px.snr(photon_flux=flux, qe=0.65, exposure_s=0.1)
        print(f"  flux={flux:6d} ph/s: SNR={res['SNR_dB']:.1f} dB, "
              f"ADU={res['ADU_value']}, sat={res['saturated']}")

    print(f"\n=== Optics stage: Airy disk ===")
    for fnum in [1.4, 2.8, 5.6, 11]:
        r = airy_disk_radius_um(550, fnum)
        ny = check_nyquist_sampling(5.6, 550, fnum)
        print(f"  f/{fnum:4.1f}: Airy={r:.2f} um, "
              f"Nyquist OK={ny['nyquist_ok']}")

    print("\n=== Photocurrent from 5778 K blackbody ===")
    result = photocurrent_from_blackbody(5778)
    print(f"  Power on pixel: {result['power_W']:.3e} W")
    print(f"  Electrons/s:    {result['electrons_per_s']:.3e}")
    print(f"  Product rule residual: {result['product_rule_max_residual']:.3e}")

    print("\n=== SymPy signal chain equations ===")
    eqs = ccd_signal_chain_sympy()
    for name, eq in eqs.items():
        print(f"  {name}: {eq}")

    print("\n=== Embedded C (first 10 lines) ===")
    lines = EMBEDDED_C_ADC_PSEUDOCODE.strip().split("\n")
    for line in lines[:10]:
        print(line)
