"""
repl/_repl_rogue_market.py
Rogue waves: optical vs financial markets vs epidemics.
Extreme value theory, Rician distribution (Bessel I_0), SIR network,
secant method, plane waves, Python kwargs -> C interface.
"""
import math
import numpy as np
import sympy as sp
from scipy.special import i0, i0e, ive
from scipy.stats import genextreme, norm
from scipy.optimize import brentq

sp.init_printing(use_unicode=False, wrap_line=False)

print("=" * 65)
print("ROGUE WAVES: OPTICAL + MARKET + EPIDEMIC  (same math)")
print("=" * 65)
print()

# ============================================================
# 0. THE UNIFIED PICTURE
# ============================================================
print("""=== 0. ROGUE WAVES APPEAR IN THREE DOMAINS ===

  OPTICAL (fiber):       Peregrine soliton, NLS equation
    dA/dz = -i(beta2/2)d^2A/dt^2 + i*gamma|A|^2*A
    Rogue wave: amplitude 3x background, appears/disappears suddenly
    Detection: GS phase recovery finds anomalous phase spikes

  FINANCIAL (markets):   Extreme price moves, fat-tailed returns
    S(t) = S0 * exp(sigma*W(t) + mu*t)   (geometric Brownian motion)
    Rogue return: > 4 sigma event (should be 1-in-63yr, but happens 1-in-5yr)
    Fat tails: Levy stable, Student-t, GEV distribution

  EPIDEMIC (network):    Super-spreader events
    R0 >> 1 outbreak in hub node of scale-free network
    Rogue node: one person infects 100+ (k >> mean degree)

  SAME UNDERLYING MATH:
    All three: rare large-amplitude events from nonlinear dynamics
    All three: heavy-tailed probability distributions
    All three: early warning from higher-order statistics (kurtosis)
    Detection: kurtosis > 3 (excess kurtosis > 0) -> fat tails -> rogue possible
""")

# ============================================================
# 1. FINANCIAL ROGUE WAVES: EXTREME VALUE THEORY
# ============================================================
print("=== 1. FINANCIAL ROGUE WAVES: EXTREME VALUE THEORY ===")

rng = np.random.default_rng(42)

# Simulate 30 years of daily returns
T_days  = 252 * 30   # ~7560 trading days
# Fat-tailed: Student-t with nu=4 (heavier than Gaussian)
from scipy.stats import t as student_t
nu = 4.0
sigma_daily = 0.012   # 1.2% daily vol (S&P500-like)
returns_fat  = student_t.rvs(df=nu, scale=sigma_daily, size=T_days, random_state=42)
returns_norm = rng.normal(0, sigma_daily, T_days)

# Kurtosis
kurt_fat  = ((returns_fat  - returns_fat.mean())**4).mean() / returns_fat.std()**4
kurt_norm = ((returns_norm - returns_norm.mean())**4).mean() / returns_norm.std()**4

print(f"  30-year daily return simulation (sigma={sigma_daily*100:.1f}%/day):")
print(f"  {'Metric':24s}  {'Gaussian':12s}  {'Student-t (nu={:.0f})'.format(nu):14s}  {'Rogue wave implication'}")
print(f"  {'Kurtosis':24s}  {kurt_norm:12.3f}  {kurt_fat:14.3f}  (>3 = fat tails = rogue possible)")
print(f"  {'Max |return| (%)':24s}  {np.abs(returns_norm).max()*100:12.2f}  {np.abs(returns_fat).max()*100:14.2f}")
print(f"  {'Events > 4-sigma':24s}  {(np.abs(returns_norm)>4*sigma_daily).sum():12d}  {(np.abs(returns_fat)>4*sigma_daily).sum():14d}  (expected Gaussian: ~1)")
print(f"  {'Events > 6-sigma':24s}  {(np.abs(returns_norm)>6*sigma_daily).sum():12d}  {(np.abs(returns_fat)>6*sigma_daily).sum():14d}")
print()

# GEV fit to annual maxima (block maxima method)
annual_max = np.array([np.abs(returns_fat[i*252:(i+1)*252]).max() for i in range(30)])
c_gev, loc_gev, scale_gev = genextreme.fit(annual_max)
print(f"  GEV fit to annual maxima (block maxima, 30 years):")
print(f"    Shape xi = {c_gev:.4f}  ({'Frechet (heavy tail)' if c_gev>0 else 'Weibull (thin tail)'})")
print(f"    Location = {loc_gev*100:.3f}%   Scale = {scale_gev*100:.3f}%")

# Return period estimates
for rp_yr in [10, 50, 100, 1000]:
    p = 1 - 1/rp_yr
    q = genextreme.ppf(p, c_gev, loc_gev, scale_gev)
    print(f"    {rp_yr:5d}-year return level: {q*100:.2f}%  daily move")
print()

# Ethereum context: crypto has higher kurtosis
print("  Crypto (Ethereum-like): nu~2.5 (heavier tails than equities)")
returns_eth = student_t.rvs(df=2.5, scale=0.035, size=252*5, random_state=7)
kurt_eth = ((returns_eth - returns_eth.mean())**4).mean() / returns_eth.std()**4
print(f"    Daily vol: {returns_eth.std()*100:.1f}%   Kurtosis: {kurt_eth:.2f}")
print(f"    Events > 4-sigma: {(np.abs(returns_eth)>4*0.035).sum()} in 5 years (ETH crashes)")
print(f"    Same math as optical rogue waves: NLS-type nonlinear amplification")
print()

# ============================================================
# 2. RICIAN DISTRIBUTION: BESSEL I_0 IN PROBABILITY
# ============================================================
print("=== 2. RICIAN DISTRIBUTION: BESSEL I_0 IN PROBABILITY ===")
print("""
  Rician PDF: p(x | nu, sigma) = (x/sigma^2) * exp(-(x^2+nu^2)/(2*sigma^2))
                                 * I_0(x*nu/sigma^2)

  where I_0(z) = J_0(iz) = sum_{k=0}^inf (z/2)^{2k} / (k!)^2  (modified Bessel)

  APPEARS IN:
    MRI magnitude images:    signal = sqrt(real^2 + imag^2), noise is Rician
    Radar detection:         target + Gaussian noise -> Rician amplitude
    Fiber noise:             |E + noise|  (amplitude of complex Gaussian)
    Optical rogue wave:      amplitude distribution of random field
    WiFi/5G:                 Nakagami-m fading (generalization of Rician)

  Special cases:
    nu=0: Rayleigh distribution  (no signal, pure noise)
    nu>>sigma: approaches Gaussian (strong signal)

  KEY: I_0(x) = Bessel function of 1st kind, order 0, IMAGINARY argument
    I_0(x) = J_0(ix)  -> exponentially growing (not oscillating)
    I_0(0) = 1
    I_0(x) ~ exp(x)/sqrt(2*pi*x)  for large x
""")

# Rician PDF
def rician_pdf(x, nu, sigma):
    return (x / sigma**2) * np.exp(-(x**2 + nu**2)/(2*sigma**2)) * i0(x*nu/sigma**2)

x_r = np.linspace(0, 5, 500)
print(f"  Rician PDF at nu=1.5, sigma=1.0:")
print(f"  {'x':8s}  {'p(x)':10s}  {'I_0(x*nu/sig^2)':18s}  {'cumulative'}")
nu_r, sig_r = 1.5, 1.0
cdf = 0
dx = x_r[1] - x_r[0]
show_at = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
for xv in show_at:
    pdf_v = rician_pdf(xv, nu_r, sig_r)
    i0_v  = i0(xv * nu_r / sig_r**2)
    cdf_v = np.trapezoid(rician_pdf(x_r[x_r<=xv], nu_r, sig_r), x_r[x_r<=xv])
    print(f"  {xv:8.2f}  {pdf_v:10.5f}  {i0_v:18.5f}  {cdf_v:.4f}")
print()

# Rician mean and variance (closed form involves I_0, I_1)
print("""  Rician mean = sigma * sqrt(pi/2) * L_{1/2}(-nu^2/(2*sigma^2))
  where L_{1/2} is Laguerre polynomial -- complex closed form
  Variance = 2*sigma^2 + nu^2 - mean^2

  Fiber sensor noise model:
    E_signal + E_noise -> |E_total| follows Rician(nu=|E_signal|, sigma=noise_rms)
    GS phase recovery works in regime where nu >> sigma (high SNR)
    Low SNR (nu/sigma < 3): phase estimates biased, need Rician correction
""")

snr_vals = [0.5, 1, 2, 5, 10, 20]
print(f"  SNR = nu/sigma  |  Phase bias (degrees)  |  Amplitude bias (%)")
for snr in snr_vals:
    # Phase bias from Rician distribution (approximate)
    phase_bias_deg = (180/math.pi) * math.atan(1/snr) * math.exp(-snr**2/2) / snr
    amp_bias_pct   = 100 * (math.sqrt(math.pi/2) * math.exp(-snr**2/4) *
                            ((1 + snr**2/2)*i0(snr**2/4) + snr**2/2*i0(snr**2/4+0.01))
                            / snr - 1) if snr > 0.1 else 100
    print(f"  {snr:8.1f}               {phase_bias_deg:8.3f}                    significant if >1%")
print()

# ============================================================
# 3. SECANT METHOD ON MANIFOLDS
# ============================================================
print("=== 3. SECANT METHOD: ROOT FINDING (GS fixed-point) ===")
print("""
  Secant method (no derivative needed):
    x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

  Convergence: superlinear (order ~1.618, golden ratio)
  vs Newton: quadratic (order 2) but needs f'(x)
  vs Bisection: linear (order 1) but guaranteed

  ON MANIFOLDS:
    Replace subtraction x_n - x_{n-1} with geodesic difference
    Replace x_{n+1} = ... with retraction onto manifold
    Used in: Riemannian optimization, GS on complex unit circle

  GS AS FIXED-POINT ON MANIFOLD:
    Constraint set C1: {E : |E|^2 = I1}     (amplitude manifold)
    Constraint set C2: {E : |H*E|^2 = I2}   (dispersed amplitude manifold)
    GS = alternating projections onto C1 and C2
    Each projection is a retraction (closest point on manifold)
    Convergence: geometric with rate rho = cos(angle between manifolds)
""")

# Secant method demo: find Bessel zero (same zeros used for fiber modes)
from scipy.special import jn

def find_jn_zero_secant(n, guess1, guess2, tol=1e-12, maxiter=50):
    x0, x1 = guess1, guess2
    history = [(x0, jn(n, x0))]
    for it in range(maxiter):
        f0, f1 = jn(n, x0), jn(n, x1)
        if abs(f1 - f0) < 1e-15:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append((x2, jn(n, x2)))
        if abs(f1) < tol:
            break
        x0, x1 = x1, x2
    return x1, history

print("  Secant method finding J_0 first zero (should be 2.4048):")
x_sol, hist = find_jn_zero_secant(0, 2.0, 2.5)
print(f"  {'Iter':6s}  {'x':12s}  {'J_0(x)':14s}  {'error'}")
for i, (xi, fi) in enumerate(hist[:8]):
    print(f"  {i:6d}  {xi:12.8f}  {fi:14.2e}  {abs(xi-2.40482556):8.2e}")
print(f"  Converged to: {x_sol:.10f}  (exact: 2.4048255577)")
print()

# ============================================================
# 4. PLANE WAVES: C INTERFACE + KWARGS PATTERN
# ============================================================
print("=== 4. PLANE WAVES + PYTHON KWARGS -> C INTERFACE ===")
print("""
  PLANE WAVE:  E(r,t) = E0 * exp(i*(k.r - omega*t + phi0))

  k vector:    k = n * omega/c * k_hat   (propagation direction)
  Dispersion:  omega^2 = c^2 * (kx^2 + ky^2 + kz^2) / n^2

  In 1D fiber: E(z,t) = E0 * exp(i*(beta*z - omega*t))
               beta = n * omega/c = 2*pi*n / lambda

  DIGITAL PLANE WAVE (sampled):
    E[k] = E0 * exp(i * 2*pi * k * f0 / fs)   k = 0, 1, ..., N-1
    Everything is digital: continuous -> DFT -> complex array
    Same math as GS: work in DFT domain, project, IDFT

  PYTHON KWARGS -> C FUNCTION SIGNATURE:
""")

# Show the kwargs pattern connecting Python to C
print("""  Python side (high-level, flexible):
    def disperse(E, D, **kwargs):
        N       = kwargs.get('N',       len(E))
        dt      = kwargs.get('dt',      1.0)
        n_iter  = kwargs.get('n_iter',  50)
        window  = kwargs.get('window',  'none')
        return _disperse_core(E, D, N, dt, n_iter, window)

  C side (fast, fixed signature):
    void disperse_c(
        float complex *E_in,   // input field
        float complex *E_out,  // output field
        int    N,              // samples
        float  D,              // dispersion
        float  dt,             // sample interval
        int    n_iter,         // GS iterations
        int    window_type     // 0=none, 1=hann, 2=hamming
    );

  CTYPES BRIDGE (Python calls C):
    import ctypes
    lib = ctypes.CDLL("./gs_core.so")
    lib.disperse_c.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # E_in  (real part)
        ctypes.POINTER(ctypes.c_float),  # E_in  (imag part)
        ctypes.POINTER(ctypes.c_float),  # E_out (real)
        ctypes.POINTER(ctypes.c_float),  # E_out (imag)
        ctypes.c_int,                    # N
        ctypes.c_float,                  # D
        ctypes.c_float,                  # dt
        ctypes.c_int,                    # n_iter
        ctypes.c_int,                    # window_type
    ]
    lib.disperse_c.restype = ctypes.c_void_p

  EVERYTHING IS DIGITAL:
    Continuous plane wave E(z,t) = E0*exp(i*beta*z)
    Digital DFT:          E[k]   = sum_n E[n]*exp(-2*pi*i*k*n/N)
    Nyquist:              f_max  = fs/2  (sample at 2x highest frequency)
    Bessel LP modes:      J_0(k_r * r)  sampled on r grid dr = a/Nr
    All the same:  sample -> DFT -> multiply by H(k) -> IDFT -> sample
""")

# verify plane wave sampling
N_pw   = 64
f0_pw  = 5.0   # cycles per N
n_pw   = np.arange(N_pw)
E_pw   = np.exp(1j * 2*np.pi * f0_pw * n_pw / N_pw)
F_pw   = np.fft.fft(E_pw)
peak_k = np.argmax(np.abs(F_pw))
print(f"  Digital plane wave: N={N_pw}, f0={f0_pw} bins")
print(f"    |E[n]| = {np.abs(E_pw).mean():.6f} (constant amplitude = 1)")
print(f"    DFT peak at bin k={peak_k}  (expected {int(f0_pw)})")
print(f"    All other bins: max = {np.abs(np.delete(F_pw,peak_k)).max():.2e} (numerical zero)")
print()

# ============================================================
# 5. SIR EPIDEMIC + NETWORK GRAPH: ROGUE SUPER-SPREADER
# ============================================================
print("=== 5. SIR EPIDEMIC: ROGUE NODE / SUPER-SPREADER ===")
print("""
  SIR model (Susceptible -> Infected -> Recovered):
    dS/dt = -beta * S * I / N
    dI/dt = +beta * S * I / N  - gamma * I
    dR/dt = +gamma * I

  R0 = beta/gamma  (basic reproduction number)
    R0 < 1: epidemic dies out
    R0 > 1: epidemic spreads
    R0 = 1: endemic equilibrium

  COVID-19:  R0 ~ 2.5-3.5 (original), ~6-8 (Omicron)
  HIV:       R0 ~ 2-5 in high-risk networks (sexual contact)
  Measles:   R0 ~ 12-18 (most contagious known)

  ROGUE NODE (super-spreader):
    Homogeneous SIR assumes uniform mixing
    Real networks: scale-free (power-law degree distribution)
    P(k) ~ k^(-alpha),  alpha ~ 2-3 for social/sexual networks
    Hub node with degree k >> <k> -> infects k people in one step
    -> rogue wave in epidemic space

  PREP BIOENGINEERING:
    PrEP = Pre-Exposure Prophylaxis (Truvada, Descovy)
    Reduces HIV transmission by >99% when taken daily
    Pharmacokinetics: C(t) = Dose/V * exp(-k_elim * t)
    Effective concentration: Cmin > IC90 = 40 ng/mL (tenofovir)
    Dosing interval: 24h  (half-life ~17h, need trough > IC90)
""")

# Numerical SIR simulation
from scipy.integrate import odeint

def sir(y, t, beta, gamma, N):
    S, I, R = y
    dS = -beta * S * I / N
    dI =  beta * S * I / N - gamma * I
    dR =  gamma * I
    return [dS, dI, dR]

N_pop = 10000
I0    = 10
S0    = N_pop - I0
R0_v  = 0.0
t_sim = np.linspace(0, 200, 2000)

scenarios = [
    ("COVID original", 0.25, 0.10),   # R0=2.5
    ("COVID Omicron",  0.48, 0.06),   # R0=8
    ("HIV (network)",  0.15, 0.05),   # R0=3
    ("With PrEP 99%",  0.0015, 0.05), # R0=0.03 -> dies immediately
]
print(f"  SIR simulation (N={N_pop}, I0={I0}):")
print(f"  {'Scenario':22s}  {'R0':6s}  {'Peak I':8s}  {'Peak day':10s}  {'Total infected'}")
for name, beta, gamma in scenarios:
    R0_calc = beta/gamma
    sol = odeint(sir, [S0, I0, R0_v], t_sim, args=(beta, gamma, N_pop))
    S_t, I_t, R_t = sol[:,0], sol[:,1], sol[:,2]
    peak_I   = I_t.max()
    peak_day = t_sim[I_t.argmax()]
    total_inf = R_t[-1] + I_t[-1]
    print(f"  {name:22s}  {R0_calc:6.2f}  {peak_I:8.1f}  {peak_day:10.1f}  {total_inf:8.1f}")
print()

# PrEP pharmacokinetics
print("  PrEP pharmacokinetics (tenofovir disoproxil fumarate 300mg):")
dose_mg   = 300.0
V_L       = 1300.0   # volume of distribution (L)
k_elim    = math.log(2) / 17.0   # elimination rate (1/h), t1/2=17h
C0_ng_mL  = dose_mg * 1e6 / (V_L * 1e3)  # ng/mL
t_hours   = np.array([0, 2, 6, 12, 24, 48])
IC90      = 40.0  # ng/mL effective concentration

print(f"  Dose={dose_mg}mg, Vd={V_L}L, t_half=17h, IC90={IC90} ng/mL")
print(f"  {'Time (h)':10s}  {'C (ng/mL)':12s}  {'> IC90?'}")
for t_h in t_hours:
    C = C0_ng_mL * math.exp(-k_elim * t_h)
    above = "YES -- protected" if C > IC90 else "NO -- missed dose risk"
    print(f"  {t_h:10.0f}  {C:12.2f}  {above}")
print()

# ============================================================
# 6. CUDA PROBABILITY: RICIAN PDF KERNEL
# ============================================================
print("=== 6. CUDA: RICIAN PDF + ROGUE WAVE DETECTION ===")
print("""
  CUDA KERNEL: compute Rician PDF for N samples (fiber amplitude data)

  __global__ void rician_pdf_kernel(
      const float *x,      // input amplitudes
      float       *pdf,    // output probabilities
      int          N,      // number of samples
      float        nu,     // signal amplitude
      float        sigma   // noise std
  ) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= N) return;

      float xi = x[i];
      float s2 = sigma * sigma;
      float arg = xi * nu / s2;

      // I_0(arg) via polynomial approximation (Abramowitz & Stegun 9.8.1)
      // For arg < 3.75: I_0(z) = 1 + 3.5156*(z/3.75)^2 + ... (7 terms)
      // For arg >= 3.75: I_0(z) = exp(z)/sqrt(z) * (0.39894 + ...)
      // Numerically stable: use scaled I_0e(z) = I_0(z)*exp(-z)

      float log_i0;
      if (arg < 700.0f) {
          // direct: no overflow
          log_i0 = logf(i0f_approx(arg));  // custom polynomial
      } else {
          // large arg: log I_0(z) ~ z - 0.5*log(2*pi*z)
          log_i0 = arg - 0.5f * logf(2.0f * 3.14159f * arg);
      }

      float log_pdf = logf(xi) - logf(s2)
                    - (xi*xi + nu*nu) / (2.0f*s2)
                    + log_i0;
      pdf[i] = expf(log_pdf);
  }

  ROGUE WAVE DETECTION KERNEL:
  __global__ void rogue_detect(
      const float *amplitude,  // time series |E(t)|
      float       *score,      // anomaly score
      int          N,
      float        bg_mean,    // background amplitude
      float        bg_std
  ) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= N) return;
      float z = (amplitude[i] - bg_mean) / bg_std;
      // Rogue if amplitude > 2x background (oceanic definition)
      // Optical: > 3x background (Solli et al. 2007 Nature)
      score[i] = (amplitude[i] > 2.0f * bg_mean) ? z * z : 0.0f;
  }
""")

# Python verification of rogue detection logic
print("  Rogue wave detection (Python verification):")
N_sig = 10000
bg_mean, bg_std = 1.0, 0.3
signal = rng.rayleigh(scale=bg_mean*0.8, size=N_sig).astype(np.float32)

# inject 3 rogue events
rogue_idx = [1000, 4500, 8200]
for ri in rogue_idx:
    signal[ri] = bg_mean * (2.5 + rng.uniform(0, 1))

rogue_threshold = 2.0 * bg_mean
detected_idx = np.where(signal > rogue_threshold)[0]
scores = np.where(signal > rogue_threshold,
                  ((signal - bg_mean)/bg_std)**2, 0)

print(f"  Signal: N={N_sig}, bg_mean={bg_mean}, bg_std={bg_std}")
print(f"  Injected rogues at: {rogue_idx}")
print(f"  Detected (A > 2*bg):  indices {detected_idx.tolist()[:10]}")
print(f"  All 3 injected rogues caught: {all(ri in detected_idx for ri in rogue_idx)}")
print(f"  False alarms: {len(detected_idx) - 3}")
print(f"  Max score: {scores.max():.2f} at index {scores.argmax()}")
print()

# ============================================================
# 7. FULL PICTURE
# ============================================================
print("=== 7. UNIFIED ROGUE WAVE DETECTION STACK ===")
print("""
  OPTICAL FIBER (RogueGuard):
    Signal: E(t) amplitude from photodetector
    Method: GS phase recovery -> anomalous phase spike detection
    Kernel: rogue_detect<<<grid,block>>>(amplitude, score, N, bg, std)
    Threshold: A > 2*bg_mean (oceanic definition) or > 3*bg (optical)

  FINANCIAL MARKET:
    Signal: daily return r(t)
    Method: GEV fit to annual maxima -> return period estimate
    Kurtosis > 3 -> fat tail -> model as Student-t, not Gaussian
    ETH/crypto: nu=2.5 degrees of freedom, daily vol 3.5%

  EPIDEMIC NETWORK:
    Signal: daily new cases I(t) in SIR model
    Method: estimate R0 from growth rate; detect hub nodes
    PrEP: reduces effective beta by 99% -> R0 from 3.0 to 0.03
    CUDA graph: BFS on scale-free contact network, find k >> <k> nodes

  ALL THREE connect to Bessel functions:
    Rician PDF    -> I_0(x) Bessel (amplitude of complex Gaussian)
    Cylindrical waveguide -> J_n(k_r*r) Bessel (fiber modes)
    SIR on cylinder tank  -> J_0 in concentration diffusion
    Secant method finds zeros of any nonlinear f -> fiber cutoffs

  SBIR ANGLE:
    RogueGuard detects optical rogue waves in undersea cables
    Same algorithm (threshold + kurtosis) applies to
    financial risk monitoring and epidemic early warning
    -> dual-use platform, three markets, one codebase
""")
