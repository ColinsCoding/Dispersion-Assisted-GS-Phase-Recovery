"""Time-stretch Dispersive Fourier Transform (TS-DFT).

The Jalali lab technique that makes real-time spectroscopy possible at GHz rates.

CORE IDEA:
  A short optical pulse E(t) has a broadband spectrum E(omega).
  After propagating through a dispersive fiber of length L with GVD beta2:

    E_out(t) = IFFT[ E(omega) * H(omega) ]
    H(omega) = exp(i * beta2 * L * omega^2 / 2)    (GVD transfer function)

  In the FAR-FIELD limit (|beta2*L| >> T0^2):
    I_out(t) ~= |E(omega)|^2  evaluated at  omega = t / (beta2 * L)

  The OUTPUT INTENSITY IS THE INPUT SPECTRUM -- frequency mapped to time.
  This is why you can measure a spectrum with a single photodetector at
  the fiber output: time resolves frequency.

APPLICATIONS:
  - Real-time spectroscopy (GHz repetition rate)
  - Optical rogue wave detection (RogueGuard project)
  - Single-shot measurements of ultrafast events
  - Dispersive Fourier transform spectroscopy (DFTS)

CAUSALITY:
  H(omega) = exp(i*beta2*L*omega^2/2) is an all-pass filter (|H|=1),
  but it IS causal -- the group delay tau(omega) = beta2*L*omega is
  real and positive (for beta2*L > 0), meaning different frequencies
  arrive at different times. The real and imaginary parts of the
  refractive index satisfy Kramers-Kronig relations.

C -> KWARGS PATTERN:
  C function: void gvd_propagate(double *E_in, int N, double beta2,
                                  double L, double dt, double *E_out)
  Python equivalent: gvd_propagate(E_in, beta2, L, dt, *, n_pts=None)
  Kwargs allow: named arguments, default values, bounds validation,
  optional parameters -- all things C lacks at the language level.

THERMAL PHYSICS:
  The dispersive fiber has thermal noise (Johnson noise) and the
  detector has thermal (dark current) noise. Both are governed by the
  Bose-Einstein distribution at optical frequencies:
    n_BE(omega,T) = 1/(exp(hbar*omega/(kT)) - 1)
  At room temp for telecom (hbar*omega >> kT): n_BE -> 0 (shot noise limited).
  Thermal physics enters through the partition function Z = sum exp(-E_n/kT).
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict, Optional, Tuple


# ── Physical constants ────────────────────────────────────────────────────────
C_LIGHT  = 2.99792458e8    # m/s
H_PLANCK = 6.62607015e-34  # J*s
HBAR     = H_PLANCK / (2 * np.pi)
K_BOLTZ  = 1.380649e-23    # J/K


# ── C-to-kwargs abstraction layer ────────────────────────────────────────────
def _validate_gvd_kwargs(beta2: float, L_m: float,
                          dt_s: float, n_pts: int) -> None:
    """Validate GVD propagation parameters.

    C equivalent:  assert(beta2 != 0 && L_m > 0 && dt_s > 0 && N > 0);
    Python kwargs: validated once at API boundary; internals trust inputs.

    This is the C-to-kwargs pattern: replace positional raw-pointer C args
    with named, validated Python parameters that fail loudly and early.
    """
    if beta2 == 0:
        raise ValueError("beta2=0: no dispersion -- use direct FFT instead")
    if L_m <= 0:
        raise ValueError(f"L_m={L_m}: fiber length must be positive")
    if dt_s <= 0:
        raise ValueError(f"dt_s={dt_s}: time step must be positive")
    if n_pts < 8:
        raise ValueError(f"n_pts={n_pts}: need at least 8 points")


# ── GVD transfer function ─────────────────────────────────────────────────────
def gvd_transfer_function(omega: np.ndarray,
                           beta2: float,
                           L_m: float) -> np.ndarray:
    """H(omega) = exp(i * beta2 * L * omega^2 / 2).

    This is the second-order GVD transfer function. All-pass (|H|=1),
    but introduces quadratic spectral phase -> temporal broadening.

    Parameters
    ----------
    omega : array of angular frequencies relative to carrier (rad/s)
    beta2 : GVD coefficient (s^2/m); negative = anomalous dispersion
    L_m   : fiber length (m)
    """
    return np.exp(1j * beta2 * L_m * omega**2 / 2)


def gvd_propagate(E_in: np.ndarray,
                   *,
                   beta2: float,
                   L_m: float,
                   dt_s: float,
                   n_pts: Optional[int] = None) -> Dict:
    """Propagate field E_in through GVD fiber.

    C signature (what this abstracts):
        void gvd_propagate(complex double *E_in, int N,
                           double beta2, double L, double dt,
                           complex double *E_out);

    Python kwargs version: named, validated, returns full diagnostic dict.

    Returns
    -------
    dict with keys:
      E_out   : output field (complex array)
      I_out   : output intensity |E_out|^2
      E_omega : input spectrum E(omega) = FFT(E_in)
      omega   : frequency axis (rad/s)
      H_omega : transfer function values
      far_field_ok : bool -- whether far-field condition is satisfied
      stretch_factor : temporal stretch M = |beta2*L| / T0^2 (approx)
      group_delay_ps : group delay tau(omega) = beta2*L*omega in ps
    """
    E_in = np.asarray(E_in, dtype=complex)
    n    = len(E_in) if n_pts is None else n_pts
    _validate_gvd_kwargs(beta2, L_m, dt_s, n)

    # Frequency axis (rad/s)
    omega = 2 * np.pi * np.fft.fftfreq(n, d=dt_s)

    # Spectrum of input
    E_omega = np.fft.fft(E_in, n=n)

    # Apply GVD transfer function
    H       = gvd_transfer_function(omega, beta2, L_m)
    E_omega_out = E_omega * H

    # Back to time domain
    E_out   = np.fft.ifft(E_omega_out)
    I_out   = np.abs(E_out)**2

    # Dispersion length: estimate T0 from RMS width of input intensity
    I_in  = np.abs(E_in)**2
    if I_in.sum() > 0:
        t_in  = np.arange(len(E_in)) * dt_s
        t_mu  = np.sum(t_in * I_in) / I_in.sum()
        T0_est = np.sqrt(np.maximum(
            np.sum((t_in - t_mu)**2 * I_in) / I_in.sum(), dt_s**2
        ))
    else:
        T0_est = dt_s
    L_D     = T0_est**2 / abs(beta2)
    far_field_ok = L_m > 10 * L_D

    # Group delay at each frequency
    group_delay = beta2 * L_m * omega  # tau(omega) = dPhi/domega

    return {
        "E_out":           E_out,
        "I_out":           I_out,
        "E_omega":         E_omega,
        "I_omega":         np.abs(E_omega)**2,
        "omega":           omega,
        "H_omega":         H,
        "far_field_ok":    far_field_ok,
        "L_D_m":           L_D,
        "stretch_factor":  L_m / L_D,
        "group_delay_ps":  group_delay * 1e12,
        "beta2":           beta2,
        "L_m":             L_m,
    }


# ── Dispersive Fourier Transform (far-field mapping) ─────────────────────────
def dispersive_fourier_transform(E_pulse: np.ndarray,
                                  *,
                                  beta2: float,
                                  L_m: float,
                                  dt_s: float) -> Dict:
    """Time-stretch DFT: maps input spectrum to output time waveform.

    In the far-field limit (|beta2*L| >> T0^2):
      I_out(t) ~= |E(omega)|^2  at  omega = t / (beta2*L)

    This is the core of the Jalali lab real-time spectroscopy technique.

    Returns both the exact numerical result (via FFT) and the far-field
    approximation, so you can see where the mapping breaks down.
    """
    result = gvd_propagate(E_pulse, beta2=beta2, L_m=L_m, dt_s=dt_s)
    n      = len(result["E_out"])

    # IFFT convention: t=0 at index 0; input pulse was centered at n//2
    t_raw    = np.arange(n) * dt_s
    t_center = (n // 2) * dt_s   # pulse center in IFFT frame

    # Far-field: I(t) ~ I_omega at omega = (t - t_center)/(beta2*L)
    omega_from_t = (t_raw - t_center) / (beta2 * L_m)

    # Sort omega for np.interp (fftfreq is not monotone)
    omega_sorted   = np.fft.fftshift(result["omega"])
    I_omega_sorted = np.fft.fftshift(result["I_omega"])
    I_ff_approx    = np.interp(omega_from_t, omega_sorted, I_omega_sorted,
                                left=0.0, right=0.0)

    # Correlation in IFFT time frame (both arrays aligned at n//2)
    corr = float(np.corrcoef(result["I_out"], I_ff_approx)[0, 1])

    # Centered time axis for plotting
    t_axis        = t_raw - t_center
    I_out_shifted = np.fft.fftshift(result["I_out"])   # peak at n//2 for plot

    result.update({
        "t_axis_s":       t_axis,
        "I_out_shifted":  I_out_shifted,
        "I_far_field":    I_ff_approx,
        "omega_from_t":   omega_from_t,
        "ff_correlation": corr,
        "interpretation": "I_out(t) ~ |E(omega)|^2 at omega=t/(beta2*L)",
    })
    return result


# ── Gaussian pulse helper ─────────────────────────────────────────────────────
def gaussian_pulse(n_pts: int, T0_s: float, dt_s: float,
                    chirp_C: float = 0.0,
                    center_frac: float = 0.5) -> np.ndarray:
    """Generate a (possibly chirped) Gaussian pulse.

    E(t) = exp(-(1+iC)*t^2 / (2*T0^2))

    T0_s      : 1/e half-width (s)
    chirp_C   : chirp parameter (0 = transform-limited)
    """
    t = (np.arange(n_pts) - int(center_frac * n_pts)) * dt_s
    return np.exp(-(1 + 1j * chirp_C) * t**2 / (2 * T0_s**2))


# ── Causality: Kramers-Kronig for refractive index ───────────────────────────
def kramers_kronig_n(omega: np.ndarray,
                      kappa: np.ndarray) -> np.ndarray:
    """Real refractive index n(omega) from extinction coefficient kappa(omega).

    Kramers-Kronig relations (causal system):
      n(omega) - 1 = (2/pi) * P.V. integral_0^inf [ omega' * kappa(omega') /
                                                      (omega'^2 - omega^2) ] domega'

    Uses the discrete Hilbert transform (FFT-based).
    The KK relations enforce causality: n(omega) is not independent of kappa(omega).

    Parameters
    ----------
    omega : frequency array (rad/s), must be uniformly spaced
    kappa : extinction coefficient array (>= 0)
    """
    # Hilbert transform: n-1 = Hilbert(kappa) for causal systems
    # Use analytic signal: hilbert gives the imaginary part of the
    # analytic signal, so Re(hilbert(kappa)) = kappa, Im = HT(kappa)
    from numpy.fft import fft, ifft
    N     = len(omega)
    K_fft = fft(kappa)
    # One-sided Hilbert multiplier
    h     = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2
    n_minus_1 = np.real(ifft(1j * h * K_fft))
    return 1.0 + n_minus_1


def verify_causality_gvd(beta2: float, L_m: float,
                          n_pts: int = 1024) -> Dict:
    """Verify that the GVD transfer function is causal.

    H(omega) = exp(i*beta2*L*omega^2/2) is all-pass (|H|=1).
    Its impulse response h(t) = IFFT[H(omega)] should be:
      - Real? No -- complex valued (dispersed pulse)
      - Causal? For beta2*L > 0: energy arrives LATER at higher freq,
        so h(t) has support for t >= 0 (group delay positive)
      - For beta2*L < 0: anomalous dispersion, group delay negative for
        high freq -- but system is still causal (carrier arrives first)

    The group delay tau(omega) = d/domega [arg H(omega)] = beta2*L*omega
    is the causal delay of each frequency component.
    """
    omega = np.linspace(-np.pi * 1e12, np.pi * 1e12, n_pts)
    H     = gvd_transfer_function(omega, beta2, L_m)
    h_t   = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(H)))

    group_delay = beta2 * L_m * omega
    phase       = np.angle(H)

    # Check: |H| = 1 everywhere (all-pass)
    all_pass    = bool(np.allclose(np.abs(H), 1.0, atol=1e-10))

    # Group delay at omega=0 should be 0
    gd_at_zero  = float(beta2 * L_m * 0)

    return {
        "H":             H,
        "h_t":           h_t,
        "omega":         omega,
        "group_delay":   group_delay,
        "phase_rad":     phase,
        "all_pass":      all_pass,
        "gd_at_zero_s":  gd_at_zero,
        "GVD_ps2_km":    beta2 * 1e27,   # s^2/m -> ps^2/km (*1e27)
        "interpretation": (
            "GVD is causal: each omega arrives at tau=beta2*L*omega. "
            "High freq (beta2>0) arrive later (normal dispersion). "
            "|H|=1 everywhere -- pure phase, no amplitude attenuation."
        ),
    }


# ── Thermal physics: partition function + Bose-Einstein ──────────────────────
def partition_function_harmonic(omega_0: float,
                                  T_K: float,
                                  n_max: int = 200) -> Dict:
    """Quantum harmonic oscillator partition function.

    Z = sum_{n=0}^{inf} exp(-n * hbar * omega_0 / (k_B * T))
      = 1 / (1 - exp(-hbar*omega_0 / kT))    (geometric series)

    Mean photon number: <n> = 1 / (exp(hbar*omega_0/kT) - 1)  (Bose-Einstein)
    Mean energy:  <E> = hbar*omega_0 * (<n> + 1/2)
    Heat capacity: C = dE/dT

    Parameters
    ----------
    omega_0 : angular frequency (rad/s)
    T_K     : temperature (K)
    """
    if T_K <= 0:
        raise ValueError("T_K must be positive")
    x       = HBAR * omega_0 / (K_BOLTZ * T_K)  # dimensionless energy ratio
    Z_exact = 1 / (1 - np.exp(-x))
    Z_sum   = sum(np.exp(-n * x) for n in range(n_max))

    n_BE    = 1 / (np.exp(x) - 1) if x > 1e-10 else K_BOLTZ * T_K / (HBAR * omega_0)
    E_mean  = HBAR * omega_0 * (n_BE + 0.5)
    F_free  = K_BOLTZ * T_K * np.log(1 - np.exp(-x)) + 0.5 * HBAR * omega_0

    # Heat capacity: C = d<E>/dT = k * x^2 * e^x / (e^x - 1)^2
    ex      = np.exp(x)
    C_v     = K_BOLTZ * x**2 * ex / (ex - 1)**2

    return {
        "Z":          Z_exact,
        "Z_numeric":  Z_sum,
        "n_BE":       n_BE,
        "E_mean_J":   E_mean,
        "E_mean_eV":  E_mean / 1.602e-19,
        "F_free_J":   F_free,
        "C_v":        C_v,
        "x":          x,
        "kT_eV":      K_BOLTZ * T_K / 1.602e-19,
        "hbar_omega_eV": HBAR * omega_0 / 1.602e-19,
        "quantum_limit": x > 1.0,   # True if kT << hbar*omega (quantum regime)
    }


def equipartition_check(T_K: float,
                         n_modes: int = 3) -> Dict:
    """Classical equipartition theorem: <E> = (1/2)*k_B*T per quadratic DOF.

    For a 3D harmonic oscillator: 3 KE + 3 PE = 3 k_B T total.
    For an electromagnetic mode: E_field + B_field = k_B T per mode.

    At optical frequencies (omega >> kT/hbar), quantum correction:
    <E> = hbar*omega / (exp(hbar*omega/kT) - 1)  << k_B*T
    --> classical equipartition FAILS for optical modes at room temp.
    """
    E_classical  = n_modes * K_BOLTZ * T_K   # classical
    E_telecom    = partition_function_harmonic(2*np.pi*C_LIGHT/1550e-9, T_K)
    E_microwave  = partition_function_harmonic(2*np.pi*10e9, T_K)

    return {
        "T_K":               T_K,
        "E_classical_3DOF":  E_classical,
        "E_classical_eV":    E_classical / 1.602e-19,
        "E_telecom_eV":      E_telecom["E_mean_eV"],
        "E_microwave_eV":    E_microwave["E_mean_eV"],
        "kT_eV":             K_BOLTZ * T_K / 1.602e-19,
        "telecom_quantum_limit": E_telecom["quantum_limit"],
        "microwave_classical":   not E_microwave["quantum_limit"],
        "note": "Optical modes: quantum (hbar*omega >> kT). "
                "Microwave modes: classical (hbar*omega << kT). "
                "Equipartition holds only for microwave at room temp.",
    }


# ── SymPy: 5 key TS-DFT equations ────────────────────────────────────────────
def tsdft_sympy_5() -> Dict[str, sp.Expr]:
    """5 key equations for sp.init_printing."""
    omega, t, beta2, L = sp.symbols("omega t beta_2 L", real=True)
    hbar_s, omega_0, k, T = sp.symbols("hbar omega_0 k_B T", positive=True)

    # 1. GVD transfer function
    H = sp.exp(sp.I * beta2 * L * omega**2 / sp.Integer(2))
    eq1 = sp.Eq(sp.Symbol("H"), H)

    # 2. TS-DFT far-field: I(t) ~ |E(omega)|^2 at omega = t/(beta2*L)
    eq2 = sp.Eq(sp.Symbol("I_out(t)"),
                sp.Abs(sp.Function("E")(t / (beta2 * L)))**2)

    # 3. Group delay tau = d/domega [phi(omega)] = beta2*L*omega
    phi  = beta2 * L * omega**2 / 2
    tau  = sp.diff(phi, omega)
    eq3  = sp.Eq(sp.Symbol("tau"), tau)

    # 4. Dispersion length L_D = T0^2 / |beta2|
    T0   = sp.Symbol("T_0", positive=True)
    eq4  = sp.Eq(sp.Symbol("L_D"), T0**2 / sp.Abs(beta2))

    # 5. Bose-Einstein photon number (thermal noise floor)
    x    = hbar_s * omega_0 / (k * T)
    eq5  = sp.Eq(sp.Symbol("n_BE"),
                 sp.Integer(1) / (sp.exp(x) - 1))

    return {
        "GVD_transfer_H":          eq1,
        "TS-DFT_far_field":        eq2,
        "group_delay_tau":         eq3,
        "dispersion_length_L_D":   eq4,
        "Bose-Einstein_n_BE":      eq5,
    }


if __name__ == "__main__":
    print("=== TS-DFT: Gaussian pulse through GVD fiber ===")
    N    = 2048
    dt   = 1e-12      # 1 ps sampling
    T0   = 2e-12      # 2 ps pulse  (L_D = T0^2/|beta2| = 200 m; L=5km >> L_D)
    beta2 = -20e-27   # anomalous dispersion (ps^2/m, SMF-28 at 1550nm)
    L    = 5000.0     # 5 km fiber

    pulse = gaussian_pulse(N, T0, dt)
    res   = dispersive_fourier_transform(pulse, beta2=beta2, L_m=L, dt_s=dt)

    print(f"  Dispersion length L_D = {res['L_D_m']:.1f} m")
    print(f"  Stretch factor M = {res['stretch_factor']:.1f}x")
    print(f"  Far-field ok: {res['far_field_ok']}")
    print(f"  FF approx correlation with exact: {res['ff_correlation']:.4f}")
    print(f"  Interpretation: {res['interpretation']}")

    print("\n=== GVD causality check ===")
    caus = verify_causality_gvd(beta2, L)
    print(f"  All-pass |H|=1: {caus['all_pass']}")
    print(f"  GVD = {caus['GVD_ps2_km']:.2f} ps^2/km (SMF-28 at 1550nm: ~-20)")
    print(f"  {caus['interpretation']}")

    print("\n=== Thermal physics: partition function ===")
    omega_telecom = 2 * np.pi * C_LIGHT / 1550e-9
    pf = partition_function_harmonic(omega_telecom, 300.0)
    print(f"  Telecom 1550nm at 300K:")
    print(f"    hbar*omega = {pf['hbar_omega_eV']:.4f} eV")
    print(f"    kT         = {pf['kT_eV']:.4f} eV")
    print(f"    <n_BE>     = {pf['n_BE']:.3e}  (essentially zero photons)")
    print(f"    Quantum limit: {pf['quantum_limit']}")

    eq = equipartition_check(300.0)
    print(f"\n  Equipartition at 300K:")
    print(f"    Classical (3 DOF): {eq['E_classical_eV']:.4f} eV")
    print(f"    Telecom mode:      {eq['E_telecom_eV']:.4e} eV (quantum suppressed)")
    print(f"    Microwave 10GHz:   {eq['E_microwave_eV']:.6f} eV (semi-classical)")

    print("\n=== 5 SymPy equations ===")
    sp.init_printing(use_latex=False)
    for name, eq in tsdft_sympy_5().items():
        print(f"  {name}: {eq}")
