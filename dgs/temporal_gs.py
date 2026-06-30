"""Temporal Gerchberg-Saxton algorithm for phase recovery in the DFT.

Based on: Solli, Gupta, Jalali — "Optical phase recovery in the dispersive
Fourier transform", Appl. Phys. Lett. 95, 231108 (2009).

CORE IDEA (paper Fig. 2):
  Two time-domain intensity measurements at dispersions D1, D2:
    f1(t) = |E_out(t; D1)|   (near-field waveform at dispersion D1)
    f2(t) = |E_out(t; D2)|   (near-field waveform at dispersion D2)

  The GVD transfer function in CWEETS (dispersion parameter D, units s^2):
    H(D, omega) = exp(i * D * omega^2 / 2)

  Algorithm (each half-iteration):
    1. Field estimate:  u(t) = f1(t) * exp(i * phi(t))
    2. FFT:            U(omega) = FFT[u(t)]
    3. Remove D1:      U_base(omega) = U(omega) * exp(-i * D1 * omega^2/2)
    4. Apply D2:       U2(omega) = U_base(omega) * exp(i * D2 * omega^2/2)
    5. IFFT:           u2(t) = IFFT[U2]
    6. Replace mag:    u2_new(t) = f2(t) * exp(i * angle(u2(t)))
    7. Reverse back:   same steps swapping D1 <-> D2, f1 <-> f2
    8. Repeat

  DIVERSITY REQUIREMENT (paper Fig. 4):
    D2/D1 ratio should be >= 1.33 for good convergence.
    At D2/D1=1.05, ghost peaks appear (poor diversity).
    At D2/D1=3, near-perfect recovery in 20 iterations.

  NEAR-FIELD vs FAR-FIELD (from paper):
    Far-field:   |D|*z >> 1/(2c) * (lambda/Delta_lambda)^2
                 Temporal envelope = spectral profile (direct mapping)
    Near-field:  Insufficient dispersion -> chirped ringing artifacts
                 Both amplitude AND phase encoded in near-field envelope
                 Temporal GS extracts this information

  CONNECTION TO SPATIAL GS:
    Spatial GS:  two planes (image plane, Fourier plane) with intensity known
    Temporal GS: two dispersions (D1, D2) playing role of two planes
    Phase propagator: exp(i*D*omega^2/2) = GVD instead of exp(i*k*z) = free space
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Dispersion transfer function ──────────────────────────────────────────────
def gvd_phase(omega: np.ndarray, D: float) -> np.ndarray:
    """GVD quadratic phase: phi(omega) = D * omega^2 / 2.

    D is in s^2 (same sign convention as beta2*L in dispersive_fourier.py).
    Negative D = anomalous dispersion (SMF-28 at 1550 nm).
    """
    return D * omega**2 / 2


def apply_dispersion(E_omega: np.ndarray, omega: np.ndarray,
                     D: float) -> np.ndarray:
    """Apply GVD transfer function H(D, omega) = exp(i*D*omega^2/2) in freq domain."""
    return E_omega * np.exp(1j * gvd_phase(omega, D))


def remove_dispersion(E_omega: np.ndarray, omega: np.ndarray,
                      D: float) -> np.ndarray:
    """Remove GVD transfer function: multiply by exp(-i*D*omega^2/2)."""
    return E_omega * np.exp(-1j * gvd_phase(omega, D))


# ── Temporal GS algorithm ─────────────────────────────────────────────────────
def temporal_gs(f1_t: np.ndarray,
                f2_t: np.ndarray,
                *,
                D1: float,
                D2: float,
                dt_s: float,
                n_iter: int = 20,
                diversity_warn: bool = True) -> Dict:
    """Temporal GS phase recovery from two near-field DFT measurements.

    Parameters
    ----------
    f1_t  : measured intensity envelope sqrt(I1(t)) at dispersion D1
    f2_t  : measured intensity envelope sqrt(I2(t)) at dispersion D2
    D1    : dispersion parameter at measurement 1 (s^2, e.g. -695e-12*1e-9 for -695 ps/nm)
    D2    : dispersion parameter at measurement 2 (s^2)
    dt_s  : time step (s)
    n_iter: number of full iterations (each = forward + reverse half)
    diversity_warn: warn if D2/D1 < 1.33 (paper recommends >= 1.33)

    Returns
    -------
    dict with:
      E_recovered  : recovered complex field (time domain, D1 frame)
      spectrum     : recovered spectrum |FFT(E_D1 * exp(-iD1*omega^2/2))|^2
      omega        : frequency axis (rad/s)
      phase_history: phase error vs iteration (normalized)
      mag_history  : magnitude error vs iteration
      diversity    : D2/D1 ratio
      n_iter       : iterations run
    """
    f1_t = np.asarray(f1_t, dtype=float)
    f2_t = np.asarray(f2_t, dtype=float)
    n = len(f1_t)
    assert len(f2_t) == n, "f1 and f2 must have same length"
    if n < 8:
        raise ValueError(f"n={n}: need at least 8 samples")
    if D1 == 0 or D2 == 0:
        raise ValueError("D1 and D2 must be non-zero (need actual dispersion)")
    if D1 == D2:
        raise ValueError("D1 == D2: zero diversity, no basis for iteration")

    diversity = abs(D2 / D1)
    if diversity_warn and diversity < 1.33:
        import warnings
        warnings.warn(
            f"D2/D1={diversity:.2f} < 1.33: low diversity -> ghost peaks. "
            "Paper recommends D2/D1 >= 1.33 (Fig. 4).", UserWarning, stacklevel=2
        )

    omega = 2 * np.pi * np.fft.fftfreq(n, d=dt_s)

    # Initial phase guess: chirped pulse with bandwidth matching f1
    # A reasonable initial phi(t) = D1 * (t - t0)^2 / (2 * T0^2) chirp
    t = np.arange(n) * dt_s
    t_center = t[n // 2]
    I1 = f1_t**2
    if I1.sum() > 0:
        t_mu = np.sum(t * I1) / I1.sum()
        T0_est = np.sqrt(np.maximum(
            np.sum((t - t_mu)**2 * I1) / I1.sum(), dt_s**2))
    else:
        t_mu, T0_est = t_center, dt_s * n / 4
    # Chirped Gaussian initial phase: phi(t) ~ -(t-t0)^2/(2*T0^2) * D1/|D1|
    phi_init = -np.sign(D1) * (t - t_mu)**2 / (2 * T0_est**2)

    # Current field estimate in D1 frame
    phi = phi_init.copy()
    u = f1_t * np.exp(1j * phi)

    phase_errors: List[float] = []
    mag_errors: List[float] = []

    for it in range(n_iter):
        # ── Forward half: D1 -> D2 ──────────────────────────────────────────
        U1 = np.fft.fft(u)
        U_base = remove_dispersion(U1, omega, D1)   # remove D1
        U2 = apply_dispersion(U_base, omega, D2)    # apply D2
        u2 = np.fft.ifft(U2)

        # Replace magnitude with measured f2(t)
        phi2 = np.angle(u2)
        u2_new = f2_t * np.exp(1j * phi2)

        # ── Reverse half: D2 -> D1 ──────────────────────────────────────────
        U2_new = np.fft.fft(u2_new)
        U_base2 = remove_dispersion(U2_new, omega, D2)  # remove D2
        U1_new = apply_dispersion(U_base2, omega, D1)   # apply D1
        u_new = np.fft.ifft(U1_new)

        # Replace magnitude with measured f1(t)
        phi_new = np.angle(u_new)
        u = f1_t * np.exp(1j * phi_new)

        # Track errors (just prior to f1 replacement, per paper convention)
        # Phase error: RMS change in phase
        ph_err = float(np.sqrt(np.mean((phi_new - phi)**2)))
        # Magnitude error: RMS difference between |u2_est| and f2
        mag_err = float(np.sqrt(np.mean((np.abs(u2) - f2_t)**2)))
        phase_errors.append(ph_err)
        mag_errors.append(mag_err)
        phi = phi_new

    # Recover base spectrum: remove D1 from final estimate
    U_final = np.fft.fft(u)
    U_spectrum = remove_dispersion(U_final, omega, D1)
    spectrum_intensity = np.abs(U_spectrum)**2
    spectrum_intensity_shifted = np.fft.fftshift(spectrum_intensity)
    omega_shifted = np.fft.fftshift(omega)

    return {
        "E_recovered":       u,
        "I_recovered":       np.abs(u)**2,
        "spectrum":          spectrum_intensity,
        "spectrum_shifted":  spectrum_intensity_shifted,
        "omega":             omega,
        "omega_shifted":     omega_shifted,
        "phase_history":     np.array(phase_errors),
        "mag_history":       np.array(mag_errors),
        "diversity":         diversity,
        "D1":                D1,
        "D2":                D2,
        "n_iter":            n_iter,
        "converged":         (len(phase_errors) > 3 and
                              phase_errors[-1] < 0.1 * phase_errors[0]),
    }


# ── Simulate near-field measurements ─────────────────────────────────────────
def simulate_near_field(E_true_omega: np.ndarray,
                         omega: np.ndarray,
                         D: float,
                         dt_s: float) -> Dict:
    """Simulate a near-field DFT measurement.

    Given the true spectrum E(omega), propagate through GVD D and
    return the time-domain intensity envelope.

    This is the forward model:
      E_out(t) = IFFT[ E(omega) * exp(i*D*omega^2/2) ]
      f(t)     = |E_out(t)|
    """
    E_out_omega = apply_dispersion(E_true_omega, omega, D)
    E_out_t = np.fft.ifft(E_out_omega)
    t = np.arange(len(E_out_t)) * dt_s
    return {
        "E_out_t":     E_out_t,
        "I_out_t":     np.abs(E_out_t)**2,
        "f_t":         np.abs(E_out_t),     # sqrt(I), what the photodetector sees
        "t":           t,
    }


def simulate_absorption_lines(n_pts: int,
                               dt_s: float,
                               line_centers_GHz: List[float],
                               line_widths_GHz: List[float],
                               line_depths: Optional[List[float]] = None,
                               T0_s: float = 5e-11) -> Dict:
    """Simulate a pulsed source through gas absorption lines (like CO in paper).

    Replicates the scenario in Solli et al. Fig. 1:
    - Mode-locked laser pulse with broad spectrum
    - Lorentzian absorption lines at specified frequencies
    - Returns true spectrum and time-domain intensity at D1, D2

    Parameters
    ----------
    n_pts         : number of FFT points
    dt_s          : time step (s)
    line_centers_GHz: absorption line center frequencies (GHz, relative)
    line_widths_GHz : FWHM of each Lorentzian line (GHz)
    line_depths   : depth of each line (0=no absorption, 1=full absorption)
    T0_s          : input pulse half-width (s)
    """
    if line_depths is None:
        line_depths = [0.9] * len(line_centers_GHz)

    omega = 2 * np.pi * np.fft.fftfreq(n_pts, d=dt_s)
    freq_GHz = omega / (2 * np.pi * 1e9)

    # Input pulse: Gaussian in time centered at n//2
    t = (np.arange(n_pts) - n_pts // 2) * dt_s
    E_pulse_t = np.exp(-t**2 / (2 * T0_s**2))
    E_pulse_omega = np.fft.fft(E_pulse_t)

    # Apply absorption: H_abs(omega) = 1 - sum[ depth * Lorentzian ]
    H_abs = np.ones(n_pts, dtype=complex)
    for fc, fw, fd in zip(line_centers_GHz, line_widths_GHz, line_depths):
        gamma = fw / 2  # half-width
        lorentz = (gamma**2) / ((freq_GHz - fc)**2 + gamma**2)
        H_abs *= (1 - fd * lorentz)

    E_true_omega = E_pulse_omega * H_abs
    true_spectrum = np.abs(E_true_omega)**2

    return {
        "E_true_omega": E_true_omega,
        "true_spectrum": true_spectrum,
        "omega":         omega,
        "freq_GHz":      freq_GHz,
        "H_absorption":  H_abs,
        "E_pulse_omega": E_pulse_omega,
    }


# ── Convergence analysis ──────────────────────────────────────────────────────
def diversity_sweep(f1_t: np.ndarray,
                    f2_t: np.ndarray,
                    *,
                    D1: float,
                    D2_ratios: List[float],
                    dt_s: float,
                    n_iter: int = 20) -> Dict:
    """Run temporal GS at multiple D2/D1 ratios to study diversity effect.

    Replicates paper Fig. 4: recovered spectrum at D2/D1 = 1.05, 1.33, 3.

    Returns dict of results keyed by D2/D1 ratio.
    """
    results = {}
    for ratio in D2_ratios:
        D2 = D1 * ratio
        res = temporal_gs(f1_t, f2_t, D1=D1, D2=D2, dt_s=dt_s,
                          n_iter=n_iter, diversity_warn=False)
        results[ratio] = res
    return results


# ── SymPy: 5 key equations ────────────────────────────────────────────────────
def temporal_gs_sympy_5():
    """5 key equations for the temporal GS algorithm."""
    import sympy as sp
    omega, t, D1, D2 = sp.symbols("omega t D_1 D_2", real=True)
    phi, E = sp.Function("phi"), sp.Function("E")

    # 1. GVD transfer function (temporal phase propagator)
    H = sp.exp(sp.I * D1 * omega**2 / sp.Integer(2))
    eq1 = sp.Eq(sp.Symbol("H(D_1,omega)"), H)

    # 2. Near-field measurement: f1(t) = |IFFT[E(omega)*H(D1)]|
    E_out = sp.Function("E_out")
    eq2 = sp.Eq(sp.Symbol("f_1(t)"),
                sp.Abs(sp.Function("IFFT")(E(omega) * H)))

    # 3. Phase replacement step (GS constraint)
    f1 = sp.Function("f_1")
    phi_est = sp.Function("phi_est")
    eq3 = sp.Eq(sp.Symbol("u_new(t)"),
                f1(t) * sp.exp(sp.I * phi_est(t)))

    # 4. Diversity requirement: D2/D1 >= 1.33 for convergence
    eq4 = sp.Eq(sp.Symbol("diversity"), D2 / D1)

    # 5. Spectrum recovery (remove D1 phase from converged estimate)
    U = sp.Function("U")
    eq5 = sp.Eq(sp.Symbol("S_recovered(omega)"),
                sp.Abs(U(omega) * sp.exp(-sp.I * D1 * omega**2 / 2))**2)

    return {
        "GVD_propagator":     eq1,
        "near_field_meas":    eq2,
        "GS_constraint":      eq3,
        "diversity_ratio":    eq4,
        "spectrum_recovery":  eq5,
    }


# ── Uncertainty principle check ───────────────────────────────────────────────
def far_field_condition(D_s2: float,
                         lambda_nm: float,
                         delta_lambda_nm: float) -> Dict:
    """Check whether dispersion is sufficient for far-field (undistorted) mapping.

    From paper: |D|*z >= [1/(2c)] * (lambda/delta_lambda)^2
    In our notation: |D| >= (1/2) * (lambda^2 / (c * delta_lambda^2))

    Parameters
    ----------
    D_s2          : dispersion parameter (s^2)
    lambda_nm     : center wavelength (nm)
    delta_lambda_nm: spectral linewidth (nm)
    """
    C_LIGHT = 2.998e8   # m/s
    lam = lambda_nm * 1e-9
    dlam = delta_lambda_nm * 1e-9

    # Uncertainty principle condition: |D| >= (lam/dlam)^2 / (2c/lam)
    # = lam^3 / (2c * dlam^2)
    D_min = lam**3 / (2 * C_LIGHT * dlam**2)
    tau_linewidth = abs(D_s2) * dlam / lam**2   # temporal width of mapped line (s)
    nu_linewidth = C_LIGHT * dlam / lam**2       # frequency width (Hz)

    return {
        "D_required_s2":   D_min,
        "D_actual_s2":     abs(D_s2),
        "far_field":       abs(D_s2) >= D_min,
        "D_actual_psnm":   abs(D_s2) * 1e12 / (lambda_nm**2 / (C_LIGHT*1e-9) * 1e-3),
        "D_required_ratio": abs(D_s2) / D_min,
        "tau_mapped_ps":   tau_linewidth * 1e12,
        "nu_linewidth_GHz": nu_linewidth / 1e9,
        "interpretation": (
            f"For {delta_lambda_nm:.3f} nm linewidth at {lambda_nm:.0f} nm: "
            f"need |D| >= {D_min:.2e} s^2. "
            f"{'FAR-FIELD (direct mapping)' if abs(D_s2) >= D_min else 'NEAR-FIELD (use temporal GS)'}"
        ),
    }


if __name__ == "__main__":
    import sympy as sp

    print("=== Temporal GS Algorithm Demo ===")
    print("Replicating Solli/Jalali APL 2009 (Fig. 1 + Fig. 4)\n")

    # Setup: 1-GHz absorption line at 1550 nm, two dispersions
    N      = 4096
    dt     = 5e-12      # 5 ps sampling (200 GHz BW)
    D1     = -600e-27   # -600 ps/nm expressed in s^2
    D2_ratios = [1.05, 1.33, 3.0]

    # True spectrum: broadband pulse + one 5-GHz absorption line (Fig. 1b)
    sim = simulate_absorption_lines(
        N, dt,
        line_centers_GHz=[50.0],
        line_widths_GHz=[5.0],
        line_depths=[0.85],
        T0_s=5e-10,
    )
    omega = sim["omega"]

    # Near-field measurements at D1 and D2=1.33*D1
    D2 = D1 * 1.33
    meas1 = simulate_near_field(sim["E_true_omega"], omega, D1, dt)
    meas2 = simulate_near_field(sim["E_true_omega"], omega, D2, dt)

    print(f"Far-field condition at D1={D1*1e27:.0f} ps^2/km:")
    ff = far_field_condition(D1, 1550.0, 0.04)   # 5 GHz @ 1550nm ~ 0.04 nm
    print(f"  {ff['interpretation']}")
    print(f"  D_required/D_actual = {ff['D_required_ratio']:.2f}")
    print()

    # Run temporal GS
    res = temporal_gs(
        meas1["f_t"], meas2["f_t"],
        D1=D1, D2=D2,
        dt_s=dt, n_iter=20,
    )
    print(f"Diversity D2/D1 = {res['diversity']:.2f}")
    print(f"Converged: {res['converged']}")
    print(f"Phase error: {res['phase_history'][0]:.4f} -> {res['phase_history'][-1]:.4f}")
    print(f"Mag error:   {res['mag_history'][0]:.4f} -> {res['mag_history'][-1]:.4f}")
    print()

    # Diversity sweep
    meas1_05 = simulate_near_field(sim["E_true_omega"], omega, D1*1.05, dt)
    meas3_00 = simulate_near_field(sim["E_true_omega"], omega, D1*3.0,  dt)

    print("Diversity sweep (D2/D1 ratios from paper Fig. 4):")
    for ratio in [1.05, 1.33, 3.0]:
        D2_r = D1 * ratio
        m2 = simulate_near_field(sim["E_true_omega"], omega, D2_r, dt)
        r = temporal_gs(meas1["f_t"], m2["f_t"], D1=D1, D2=D2_r,
                        dt_s=dt, n_iter=20, diversity_warn=False)
        # Correlation of recovered spectrum with true spectrum
        true_sh = np.fft.fftshift(sim["true_spectrum"])
        rec_sh = r["spectrum_shifted"]
        # Normalize
        true_n = true_sh / (true_sh.max() + 1e-30)
        rec_n  = rec_sh  / (rec_sh.max()  + 1e-30)
        corr = float(np.corrcoef(true_n, rec_n)[0, 1])
        print(f"  D2/D1={ratio:.2f}: corr={corr:.4f}, converged={r['converged']}")

    print()
    print("=== 5 SymPy equations ===")
    sp.init_printing(use_latex=False)
    for name, eq in temporal_gs_sympy_5().items():
        print(f"  [{name}]  {eq}")
