"""
Z-Transform and Digital Filter Design for Computer Engineering

The Z-transform is the discrete-time version of the Laplace transform.
It is to digital signal processing what the Laplace transform is to analog circuits.

Z{x[n]} = sum_{n=-inf}^{inf} x[n] * z^{-n}    [bilateral Z-transform]

Connection to Fourier:
  On the unit circle: z = exp(j*omega)
  Z-transform evaluated on unit circle = DTFT (discrete-time Fourier transform)
  DTFT evaluated at omega = 2*pi*k/N -> DFT (what FFT computes)

Connection to Laplace:
  Continuous: H(s), poles in left half-plane -> stable
  Discrete:   H(z), poles inside unit circle -> stable
  Mapping:    z = exp(s*T_s)  [bilinear transform approximates this]

Connection to this repo's H(f) = exp(j*pi*D*f^2):
  This is a dispersive all-pass filter: |H|=1, phase = pi*D*f^2
  In discrete time: H(z) evaluated on unit circle z=exp(j*omega)
  Phase = pi*D*(omega/2*pi*fs)^2  -> digital GVD dispersion filter
  Digital dispersion compensation: H_comp(z) = exp(-j*pi*D*(omega/2*pi*fs)^2)
  Implemented as: multiply in frequency domain = convolve in time = FIR filter

Jane Street / CE context:
  Digital filters are in EVERYTHING: audio codecs, radio receivers, financial
  time series smoothing, GPS signal processing, radar pulse compression.
  Same math as IIR filter -> recursive algorithm -> fast execution in C.
  FIR filter -> matrix multiply -> implementable as SIMD in C.
"""
import numpy as np
import sympy as sp


def z_transform_common_pairs():
    """
    Z-transform pairs that every CE must memorize.

    These are the discrete-time analogs of Laplace transform pairs.
    Laplace: L{u(t)} = 1/s          Z: Z{u[n]} = z/(z-1)
    Laplace: L{e^-at} = 1/(s+a)    Z: Z{a^n u[n]} = z/(z-a)
    Laplace: L{delta(t)} = 1        Z: Z{delta[n]} = 1
    """
    n, z, a, k = sp.symbols('n z a k', complex=True)

    pairs = {
        'unit_impulse': {
            'x_n': 'delta[n]  (1 at n=0, 0 elsewhere)',
            'X_z': '1',
            'ROC': 'all z',
            'lesson': 'Impulse -> constant FT (all frequencies equally). Same as delta(t)->1.'
        },
        'unit_step': {
            'x_n': 'u[n]  (1 for n>=0, 0 for n<0)',
            'X_z': 'z/(z-1)',
            'ROC': '|z| > 1',
            'lesson': 'Pole at z=1 (unit circle) -> marginally stable (DC). DC input -> DC output.'
        },
        'exponential': {
            'x_n': 'a^n * u[n]',
            'X_z': 'z/(z-a)',
            'ROC': '|z| > |a|',
            'lesson': '|a|<1 -> pole inside unit circle -> stable decay. |a|>1 -> unstable growth.'
        },
        'cosine': {
            'x_n': 'cos(omega_0*n) * u[n]',
            'X_z': 'z*(z-cos(omega_0)) / (z^2 - 2*cos(omega_0)*z + 1)',
            'ROC': '|z| > 1',
            'lesson': 'Complex conjugate poles on unit circle -> pure oscillation (undamped).'
        },
        'ramp': {
            'x_n': 'n * u[n]',
            'X_z': 'z/(z-1)^2',
            'ROC': '|z| > 1',
            'lesson': 'Double pole at z=1. Laplace analogy: L{t} = 1/s^2.'
        },
        'finite_sequence': {
            'x_n': 'u[n] - u[n-N]  (N-point rectangular window)',
            'X_z': '(1 - z^-N)/(1 - z^-1)  = (z^N - 1)/(z^(N-1)*(z-1))',
            'ROC': 'z != 0',
            'lesson': 'N-1 zeros equally spaced on unit circle. FT = sinc-like sinc(N*omega/2)/sinc(omega/2).'
        },
    }

    # Initial Value Theorem: x[0] = lim_{z->inf} X(z)
    # Final Value Theorem: lim_{n->inf} x[n] = lim_{z->1} (z-1)*X(z)  [if stable]
    theorems = {
        'initial_value': 'x[0] = lim_{z->inf} X(z)',
        'final_value': 'x[inf] = lim_{z->1} (z-1)*X(z)  (only if system stable)',
        'convolution': 'x[n]*h[n] <-> X(z)*H(z)  [convolution in time = multiply in z]',
        'time_shift': 'x[n-k] <-> z^{-k} * X(z)  [delay by k samples = multiply by z^{-k}]',
        'scaling': 'a^n * x[n] <-> X(z/a)  [frequency scaling in z-domain]',
    }
    return {'pairs': pairs, 'theorems': theorems}


def transfer_function_poles_zeros(num_coeffs, den_coeffs, fs=1.0):
    """
    H(z) = B(z)/A(z) = (b_0 + b_1*z^-1 + ... + b_M*z^-M) /
                        (a_0 + a_1*z^-1 + ... + a_N*z^-N)

    num_coeffs: [b_0, b_1, ..., b_M]  (numerator polynomial in z^-1)
    den_coeffs: [a_0, a_1, ..., a_N]
    fs: sampling frequency [Hz]

    Poles: roots of A(z)  -- determine stability and resonance
    Zeros: roots of B(z)  -- determine frequency nulls
    Gain:  |H(z)| on unit circle = magnitude response

    Stability: ALL poles must be INSIDE the unit circle.
      |pole| < 1: stable (decaying exponential response)
      |pole| = 1: marginally stable (pure oscillation, grows with noise)
      |pole| > 1: unstable (exponentially growing)

    FIR filter: A(z) = 1 (all poles at z=0) -> always stable
    IIR filter: A(z) != 1 -> poles can be anywhere -> must check stability
    """
    b = np.array(num_coeffs, dtype=complex)
    a = np.array(den_coeffs, dtype=complex)

    zeros = np.roots(b)
    poles = np.roots(a)

    stable = all(abs(p) < 1.0 - 1e-10 for p in poles)
    marginally_stable = all(abs(p) <= 1.0 + 1e-10 for p in poles) and not stable

    # Frequency response on unit circle
    N_freq = 1024
    omega = np.linspace(0, np.pi, N_freq)   # 0 to Nyquist
    freq_hz = omega / (2*np.pi) * fs

    # Evaluate H(exp(j*omega))
    z_vec = np.exp(1j * omega)
    H_vec = np.polyval(b, z_vec) / np.polyval(a, z_vec)
    magnitude_dB = 20 * np.log10(np.abs(H_vec) + 1e-12)
    phase_deg = np.angle(H_vec) * 180 / np.pi

    return {
        'zeros': zeros,
        'poles': poles,
        'stable': stable,
        'marginally_stable': marginally_stable,
        'stability_check': 'STABLE' if stable else ('MARGINALLY STABLE' if marginally_stable else 'UNSTABLE'),
        'omega': omega,
        'freq_hz': freq_hz,
        'magnitude_dB': magnitude_dB,
        'phase_deg': phase_deg,
        'H_complex': H_vec,
        'pole_magnitudes': np.abs(poles).tolist(),
    }


def digital_filter_examples():
    """
    Common digital filters used in CE and photonics.

    MOVING AVERAGE (FIR):
      y[n] = (1/N) * sum_{k=0}^{N-1} x[n-k]
      H(z) = (1/N) * (1 + z^-1 + ... + z^{-(N-1)}) = (1/N) * (1 - z^-N)/(1 - z^-1)
      Zeros at z = exp(j*2*pi*k/N) for k=1..N-1 (nulls at harmonics of 1/N*fs)
      Low-pass filter. Simple. Always stable (FIR).
      Used in: stock price smoothing, sensor averaging, CIC decimation filter.

    FIRST-ORDER IIR LOW-PASS (leaky integrator):
      y[n] = alpha * x[n] + (1-alpha) * y[n-1]
      H(z) = alpha / (1 - (1-alpha)*z^-1)
      Pole at z = 1-alpha. For alpha=0.1: pole=0.9 (stable, slow rolloff).
      Time constant: tau = -T_s / ln(1-alpha) ~ T_s/alpha for small alpha.
      Used in: DC estimation, automatic gain control, IIR notch filters.

    NOTCH FILTER (removes one frequency):
      H(z) = (1 - 2*cos(omega_0)*z^-1 + z^-2) / (1 - 2*r*cos(omega_0)*z^-1 + r^2*z^-2)
      Zeros ON unit circle at omega_0: complete null at that frequency.
      Poles just inside at radius r~0.9: narrow bandwidth.
      Used in: remove 60Hz power line noise from ECG/EEG/audio.

    DISPERSION COMPENSATION (all-pass, for this repo):
      H_comp(z) on unit circle: phase = -pi*D*(omega/(2*pi))^2
      FIR approximation: truncate impulse response of ideal filter.
      Better: frequency-domain multiplication (overlap-add algorithm).
      This is what dgs/causality.py phase correction implements.
    """
    fs = 44100.0   # standard audio sample rate

    # Moving average N=8
    N = 8
    ma_num = np.ones(N) / N
    ma_den = np.array([1.0])
    ma = transfer_function_poles_zeros(ma_num, ma_den, fs)

    # First-order IIR LP, alpha=0.1
    alpha = 0.1
    iir_num = np.array([alpha])
    iir_den = np.array([1.0, -(1-alpha)])
    iir = transfer_function_poles_zeros(iir_num, iir_den, fs)

    # Notch at 60 Hz
    omega0 = 2*np.pi*60/fs
    r = 0.95
    notch_num = np.array([1.0, -2*np.cos(omega0), 1.0])
    notch_den = np.array([1.0, -2*r*np.cos(omega0), r**2])
    notch = transfer_function_poles_zeros(notch_num, notch_den, fs)

    # All-pass dispersion (phase-only):
    # H(omega) = exp(j*pi*D*(omega/(2*pi))^2) evaluated numerically
    D = 5000.0   # dispersion parameter (same as gs_core.py)
    omega = np.linspace(0, np.pi, 1024)
    f_norm = omega / (2*np.pi)   # normalized frequency 0..0.5
    H_disp = np.exp(1j * np.pi * D * f_norm**2)
    phase_disp = np.angle(H_disp) * 180/np.pi

    return {
        'moving_average_N8': {
            'stability': ma['stability_check'],
            'poles': ma['poles'].tolist(),
            'n_zeros': len(ma['zeros']),
            'freq_hz': ma['freq_hz'],
            'mag_dB': ma['magnitude_dB'],
        },
        'iir_lowpass_alpha01': {
            'stability': iir['stability_check'],
            'pole': complex(iir['poles'][0]),
            'pole_magnitude': float(abs(iir['poles'][0])),
            'freq_hz': iir['freq_hz'],
            'mag_dB': iir['magnitude_dB'],
        },
        'notch_60Hz': {
            'stability': notch['stability_check'],
            'notch_freq_hz': 60.0,
            'pole_magnitude': r,
            'freq_hz': notch['freq_hz'],
            'mag_dB': notch['magnitude_dB'],
        },
        'dispersion_allpass': {
            'D': D,
            'magnitude_flat': True,
            'phase_deg': phase_disp,
            'omega': omega,
            'note': (
                'H(f)=exp(j*pi*D*f^2): magnitude=1 everywhere (all-pass), '
                'phase = quadratic (GVD). '
                'Digital dispersion compensation = multiply by conjugate H*.'
            ),
        },
    }


def bilinear_transform(analog_poles, analog_zeros, analog_gain, fs):
    """
    Bilinear transform: convert analog H(s) -> digital H(z).

    Mapping: s = (2*fs) * (z-1)/(z+1)  [bilinear / Tustin method]
    Inverse: z = (1 + s/(2*fs)) / (1 - s/(2*fs))

    Properties:
      - Maps left half s-plane -> inside unit circle (stability preserved)
      - Maps j*omega axis -> unit circle (frequency axis mapped, warped)
      - Frequency warping: omega_digital = 2*arctan(omega_analog / (2*fs))
        -> high analog frequencies get compressed near Nyquist
      - To correct: pre-warp the desired cutoff before designing analog filter

    Example: convert Butterworth LP (cutoff fc) to digital:
      1. Pre-warp: omega_c_a = 2*fs * tan(pi*fc/fs)
      2. Design analog Butterworth at omega_c_a
      3. Apply bilinear transform
      4. Result: digital Butterworth with -3dB at fc (exact)

    Alternative: impulse invariant (no frequency warping, but aliasing issues).
    """
    Ts = 1.0 / fs
    # Bilinear: s -> z via s = (2/Ts)*(z-1)/(z+1)
    # Poles: z = (1 + s*Ts/2) / (1 - s*Ts/2)
    digital_poles = [(1 + p*Ts/2) / (1 - p*Ts/2) for p in analog_poles]
    digital_zeros = [(1 + z*Ts/2) / (1 - z*Ts/2) for z in analog_zeros]

    # Add zeros at z=-1 for each pole without a corresponding analog zero
    n_extra_zeros = len(analog_poles) - len(analog_zeros)
    digital_zeros += [-1.0] * n_extra_zeros

    # Gain correction: match DC gain (omega=0, z=1)
    # H_digital(z=1) should equal H_analog(s=0)
    s0_gain = analog_gain
    z1_num = np.prod([1 - z for z in digital_zeros]) if digital_zeros else 1.0
    z1_den = np.prod([1 - p for p in digital_poles]) if digital_poles else 1.0
    digital_gain = (s0_gain * z1_den / z1_num).real if z1_num != 0 else analog_gain

    stable = all(abs(p) < 1.0 for p in digital_poles)

    return {
        'digital_poles': digital_poles,
        'digital_zeros': digital_zeros,
        'digital_gain': float(digital_gain.real if hasattr(digital_gain, 'real') else digital_gain),
        'stable': stable,
        'method': 's = (2*fs)*(z-1)/(z+1)',
        'frequency_warping': 'omega_d = 2*arctan(omega_a / (2*fs)) -- pre-warp cutoff to correct',
    }


def first_order_iir_c_implementation():
    """
    Generate C code for a first-order IIR filter (direct form I).
    y[n] = alpha*x[n] + (1-alpha)*y[n-1]

    This runs in real time on a microcontroller.
    Used in: ADC averaging, servo control loops, RogueGuard signal conditioning.

    Fixed-point version for MCU without FPU:
      Use Q15 (1 sign bit + 15 fractional bits).
      alpha_q15 = (int)(alpha * 32768)
      y[n] = (alpha_q15 * x[n] + (32768 - alpha_q15) * y[n-1]) >> 15

    Same computation as:
      - Exponential moving average (finance: smoothing price data)
      - RC low-pass filter discretized via forward Euler: y[n] = y[n-1] + (T_s/RC)*(x[n]-y[n-1])
        where alpha = T_s/RC (small alpha = small T_s/RC = slow response)
    """
    c_code = """
/* First-order IIR low-pass filter (direct form I) */
/* y[n] = alpha*x[n] + (1-alpha)*y[n-1]            */
/* pole at z = 1-alpha, |pole| < 1 iff 0 < alpha < 2 (always stable for 0<alpha<1) */

#include <stdint.h>

/* Floating-point version (for MCUs with FPU: Cortex-M4F, M7) */
float iir_lp_state = 0.0f;

float iir_lowpass_f32(float x, float alpha) {
    iir_lp_state = alpha * x + (1.0f - alpha) * iir_lp_state;
    return iir_lp_state;
}

/* Fixed-point Q15 version (for MCUs without FPU: Cortex-M0, M3, AVR) */
/* alpha_q15 = round(alpha * 32768), e.g. alpha=0.1 -> alpha_q15=3277 */
static int16_t iir_lp_state_q15 = 0;

int16_t iir_lowpass_q15(int16_t x, int16_t alpha_q15) {
    int32_t acc;
    acc = (int32_t)alpha_q15 * x;
    acc += (int32_t)(32768 - alpha_q15) * iir_lp_state_q15;
    iir_lp_state_q15 = (int16_t)(acc >> 15);  /* Q15 -> Q0 normalize */
    return iir_lp_state_q15;
}

/* Usage:
 *   float y = iir_lowpass_f32(adc_reading, 0.05f);  // alpha=0.05, tau ~ 20 samples
 *   // For fs=1MHz, alpha=0.05 -> tau = 1/(0.05*1e6) = 20us
 */
"""
    return {
        'c_code': c_code,
        'transfer_function': 'H(z) = alpha / (1 - (1-alpha)*z^{-1})',
        'pole': '1 - alpha  (must be in (0,1) for stability)',
        'time_constant_samples': '1/alpha',
        'rc_analogy': 'alpha = T_s / (RC + T_s) ~ T_s/RC for RC >> T_s',
        'finance_analogy': 'EMA(n) = alpha*price[n] + (1-alpha)*EMA(n-1)',
        'rogue_guard_use': (
            'ADC samples at 1MHz on RPi CM4. '
            'IIR LP with alpha=0.001 -> tau=1000 samples=1ms -> removes RF interference. '
            'Then GS phase retrieval on the smoothed envelope.'
        ),
    }


def dtft_vs_dft_vs_fft():
    """
    The three related transforms every CE must know:

    DTFT (Discrete-Time FT):
      X(omega) = sum_{n=-inf}^{inf} x[n] * exp(-j*omega*n)
      omega in [0, 2*pi]. Continuous in omega. Defined for infinite sequences.
      This is Z-transform evaluated on unit circle: X(omega) = X(z)|_{z=exp(j*omega)}

    DFT (Discrete FT):
      X[k] = sum_{n=0}^{N-1} x[n] * exp(-j*2*pi*k*n/N),  k=0..N-1
      Samples DTFT at N equally spaced points: omega_k = 2*pi*k/N.
      Assumes x[n] is periodic with period N.
      Matrix form: X = W * x, where W_{kn} = exp(-j*2*pi*kn/N) [DFT matrix]

    FFT (Fast FT):
      Same computation as DFT, but O(N log N) instead of O(N^2).
      Cooley-Tukey algorithm (1965): recursively split N into N/2 even/odd.
      N=1024: DFT = 1,048,576 ops; FFT = 10,240 ops (100x faster).
      np.fft.fft() IS an FFT. Always use FFT, never implement DFT directly.

    Z-TRANSFORM vs LAPLACE:
      Continuous: H(s), s = sigma + j*omega. Frequency axis = j*omega axis.
      Discrete:   H(z), z = r*exp(j*omega). Frequency axis = unit circle.
      Mapping:    z = exp(s*T_s). s=j*omega -> z=exp(j*omega*T_s).

    CONNECTION TO THIS REPO:
      H(f) = exp(j*pi*D*f^2) = continuous dispersion (Laplace/FT domain).
      np.fft.fft(E) = DFT approximation of FT[E(t)] (sampled signal).
      H_discrete[k] = exp(j*pi*D*(k*delta_f)^2) = sampled dispersion filter.
      This IS the Z-transform of the dispersive channel, evaluated at z=exp(j*omega_k).
    """
    N = 64
    # DFT matrix
    n_vec = np.arange(N)
    k_vec = np.arange(N)
    W = np.exp(-1j * 2*np.pi * np.outer(k_vec, n_vec) / N)

    # Verify: DFT via matrix == np.fft.fft
    x = np.random.default_rng(0).standard_normal(N)
    X_matrix = W @ x
    X_fft    = np.fft.fft(x)
    match = np.allclose(X_matrix, X_fft)

    # Dispersion filter in discrete domain
    D = 5000.0
    delta_f = 1.0 / N
    k = np.arange(N)
    H_disp = np.exp(1j * np.pi * D * (k*delta_f)**2)

    return {
        'DTFT': 'X(omega) = sum x[n]*exp(-j*omega*n), continuous omega, Z|_{unit circle}',
        'DFT': f'X[k] = sum x[n]*exp(-j*2*pi*k*n/N), N samples in, N samples out',
        'FFT': f'Same as DFT, O(N log N) Cooley-Tukey, N={N}: {int(N*np.log2(N))} ops vs {N*N} ops',
        'DFT_matrix_N': N,
        'DFT_matches_FFT': bool(match),
        'Z_vs_Laplace': {
            'Laplace': 'H(s), poles in left half-plane for stability',
            'Z_transform': 'H(z), poles inside unit circle for stability',
            'mapping': 'z = exp(s*T_s)  [sampling relation]',
        },
        'H_disp_sample': H_disp[:8].tolist(),
        'repo_connection': (
            'gs_core.py: E_f = np.fft.fft(E_t); E_f *= H_disp; E_out = np.fft.ifft(E_f).\n'
            'H_disp[k] = exp(j*pi*D*(k/N)^2) = H(f)|_{f=k*df}.\n'
            'This is the DFT-domain implementation of H(f)=exp(j*pi*D*f^2).\n'
            'Same as digital filter implementation: multiply in frequency = convolve in time.'
        ),
    }


def demo():
    print("=== Z-TRANSFORM AND DIGITAL FILTER DESIGN ===\n")

    print("--- Z-Transform Pairs ---")
    pairs = z_transform_common_pairs()
    for name, p in pairs['pairs'].items():
        print(f"  {name:20s}: X(z)={p['X_z']}  ROC={p['ROC']}")
    print()
    for name, thm in pairs['theorems'].items():
        print(f"  {name}: {thm}")

    print("\n--- Filter Examples ---")
    filters = digital_filter_examples()
    print(f"  Moving average (N=8): {filters['moving_average_N8']['stability']}, "
          f"{filters['moving_average_N8']['n_zeros']} zeros")
    print(f"  IIR LP (alpha=0.1): {filters['iir_lowpass_alpha01']['stability']}, "
          f"pole={filters['iir_lowpass_alpha01']['pole_magnitude']:.3f}")
    print(f"  Notch 60Hz: {filters['notch_60Hz']['stability']}, "
          f"r={filters['notch_60Hz']['pole_magnitude']}")
    print(f"  Dispersion all-pass: {filters['dispersion_allpass']['note'][:70]}...")

    print("\n--- Bilinear Transform Example ---")
    # Analog 1st-order Butterworth LP at 1kHz (pole at s=-2*pi*1000)
    fc = 1000.0; fs = 44100.0
    analog_pole = -2*np.pi*fc
    bt = bilinear_transform([analog_pole], [], 2*np.pi*fc, fs)
    print(f"  Analog pole: s={analog_pole:.0f} -> Digital pole: z={bt['digital_poles'][0]:.5f}")
    print(f"  Stable: {bt['stable']}")

    print("\n--- IIR C Code ---")
    c = first_order_iir_c_implementation()
    print(f"  H(z) = {c['transfer_function']}")
    print(f"  RC analogy: {c['rc_analogy']}")
    print(f"  Finance: {c['finance_analogy']}")

    print("\n--- DTFT / DFT / FFT ---")
    d = dtft_vs_dft_vs_fft()
    print(f"  DFT matrix == FFT: {d['DFT_matches_FFT']}")
    print(f"  Complexity: {d['FFT']}")
    print(f"  Repo: {d['repo_connection'][:80]}...")

    print("\n=== DONE ===")


if __name__ == '__main__':
    demo()
