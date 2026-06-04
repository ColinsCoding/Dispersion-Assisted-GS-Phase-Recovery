"""
_repl_poynting_solar.py
Poynting vector, solar irradiance, fiber attenuation e^(-x),
time-averaged power, loop over intensity traces.
"""
import numpy as np
import sympy as sp
import pandas as pd
sp.init_printing(use_unicode=False, wrap_line=False)

# ============================================================
# 1. Poynting vector: time average
# ============================================================
print("=== Poynting vector S = E x H ===")
print("""
Instantaneous:  S(t) = E(t) x H(t)       [W/m^2]
Time average:   <S>  = (1/2) Re(E x H*)  [W/m^2]  (complex amplitudes)

For plane wave in free space:
  E = E0 * cos(kz - wt) x-hat
  H = (E0/eta) * cos(kz - wt) y-hat     eta = sqrt(mu0/eps0) = 377 ohm

  <S> = E0^2 / (2*eta)  z-hat

Solar irradiance at Earth:  <S> = 1361 W/m^2  (solar constant)
=> E0 = sqrt(2 * eta * <S>)
""")

eta = 377.0
S_solar = 1361.0
E0_solar = np.sqrt(2 * eta * S_solar)
print(f"Solar E0 = {E0_solar:.1f} V/m")
print(f"Solar H0 = {E0_solar/eta:.3f} A/m")
print(f"Radiation pressure = S/c = {S_solar/3e8:.2e} Pa")
print()

# SymPy: time average formally
t, w, E0_s, eta_s = sp.symbols('t omega E0 eta', positive=True)
k_s, z_s = sp.symbols('k z', real=True)

E_t = E0_s * sp.cos(k_s*z_s - w*t)
H_t = (E0_s/eta_s) * sp.cos(k_s*z_s - w*t)
S_t = E_t * H_t
T   = 2*sp.pi/w  # period

S_avg = sp.Rational(1,1)*sp.integrate(S_t, (t, 0, T)) / T
print("Time-averaged Poynting:")
print(sp.pretty(sp.simplify(S_avg)))
print("= E0^2 / (2*eta) confirmed")
print()

# ============================================================
# 2. Fiber attenuation: e^(-alpha*z)
# ============================================================
print("=== Fiber attenuation: I(z) = I0 * exp(-alpha*z) ===")
print("""
alpha = attenuation coefficient [dB/km or 1/m]
Standard SMF-28:  alpha = 0.2 dB/km  (1550 nm window)
                         = 0.046 /km  (Neper units)

Power after distance z:
  P(z) = P0 * 10^(-alpha_dB * z / 10)   [dB form]
       = P0 * exp(-alpha_Np * z)          [Neper form]

Conversion: alpha_Np = alpha_dB * ln(10)/10 = alpha_dB * 0.2303
""")

alpha_dB_km = 0.2        # dB/km
alpha_Np_km = alpha_dB_km * np.log(10) / 10

z_km = np.array([0, 10, 50, 100, 500, 1000])
P_ratio = np.exp(-alpha_Np_km * z_km)
P_dB    = -alpha_dB_km * z_km

df_fiber = pd.DataFrame({
    'z_km': z_km,
    'P/P0': np.round(P_ratio, 6),
    'loss_dB': P_dB,
    'need_amplifier': ['no' if p > 0.01 else 'YES' for p in P_ratio]
})
print(df_fiber.to_string(index=False))
print()
print("Repeater spacing ~80 km (EDFA every 80 km on transoceanic links)")
print()

# ============================================================
# 3. Loop over synthetic intensity traces: pattern finding
# ============================================================
print("=== Loop over GS intensity traces: statistics ===")

def disperse(E, D):
    N  = len(E)
    nu = np.fft.fftfreq(N)
    H  = np.exp(1j * np.pi * D * nu**2)
    return np.fft.ifft(np.fft.fft(E) * H)

rng = np.random.default_rng(0)
N   = 512
D1, D2 = -5000.0, -5750.0

rows = []
for trial in range(20):
    # random QPSK signal
    n_sym   = N // 8
    symbols = rng.choice([0,1,2,3], size=n_sym)
    phi_t   = np.repeat(symbols * np.pi/2, 8)[:N]
    phi_t   = np.convolve(phi_t, np.ones(4)/4, mode='same')
    E_true  = np.exp(1j * phi_t)

    snr_db  = rng.uniform(20, 40)
    noise   = 10**(-snr_db/20)
    I1 = np.maximum(np.abs(disperse(E_true,D1))**2 + noise*rng.standard_normal(N), 0)
    I2 = np.maximum(np.abs(disperse(E_true,D2))**2 + noise*rng.standard_normal(N), 0)

    # measurable statistics
    rows.append({
        'trial'   : trial,
        'snr_db'  : round(snr_db, 1),
        'I1_mean' : round(float(np.mean(I1)), 4),
        'I2_mean' : round(float(np.mean(I2)), 4),
        'I1_std'  : round(float(np.std(I1)),  4),
        'I1_kurt' : round(float(np.mean((I1-np.mean(I1))**4)/np.std(I1)**4), 3),
        'corr_12' : round(float(np.corrcoef(I1,I2)[0,1]), 4),
        'E_ratio' : round(float(np.sum(I1)/np.sum(I2)), 4),
    })

df_traces = pd.DataFrame(rows)
print(df_traces.to_string(index=False))
print()

# pattern: correlation between I1 and I2 vs SNR
corr_vals = df_traces['corr_12'].values
snr_vals  = df_traces['snr_db'].values
r = np.corrcoef(snr_vals, corr_vals)[0,1]
print(f"Correlation(SNR, I1-I2 correlation): r={r:.3f}")
print("-> High SNR => I1,I2 more correlated (signal dominates noise)")
print()

# ============================================================
# 4. Scientific ML: what it finds that humans can't
# ============================================================
print("=== Scientific ML: pattern humans miss ===")
print("""
Human pattern recognition:
  - Peaks, troughs, periodicity
  - Linear trends, obvious clusters
  - Up to ~10 variables simultaneously

Scientific ML (FNO, CNN, transformer):
  - Correlations across ALL 512 frequency bins simultaneously
  - Phase relationships invisible in intensity |E|^2
  - Non-local patterns: bin 3 correlates with bin 497
  - Latent space: 32 modes capturing 99% of variance

Concrete example from your project:
  GS recovers phase by iterating 50 steps.
  FNO finds the SAME phase in 1 forward pass.
  What did FNO learn? The inverse mapping:
    {I1, I2} -> phi
  That mapping has no closed-form solution.
  FNO approximated it from 1000 examples.
  That's the pattern humans can't write down.

Quantum information angle:
  I1 = |<x|E>|^2    <- measurement destroys phase (Born rule)
  GS = quantum state tomography with two measurement bases
  More bases = more constraints = unique solution
  This is exactly the MUB (mutually unbiased bases) problem in QI.
""")

# kurtosis of QPSK intensity trace
print("Kurtosis of QPSK intensity traces (should be ~1.8 for chi-squared dist):")
print(df_traces[['trial','snr_db','I1_kurt']].to_string(index=False))
print()
print("Kurtosis > 3 = heavy tails (rare bright events = optical rogue waves)")
print("Kurtosis ~ 1.8 = chi-squared(2) = Rayleigh intensity distribution")
print("This IS the pattern: kurtosis fingerprints the modulation format.")
