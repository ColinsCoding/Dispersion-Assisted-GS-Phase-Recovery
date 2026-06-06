# %% [markdown]
# # Dirac Delta — Mental Steps + Qualcomm DSP
# `init_printing(use_latex="mathjax")` throughout.
#
# **Structure:**
# §1  The 7 mental steps — what to think BEFORE writing math
# §2  Continuous δ(t): testing machine, not a function
# §3  Discrete δ[n]: the impulse — how Qualcomm Hexagon DSP thinks
# §4  Convolution = sum of shifted impulses (the DSP identity)
# §5  FIR filter design via δ[n]
# §6  Sampling theorem: multiplying by a pulse train
# §7  CDMA spreading codes: near-delta correlation
# §8  Figure (12 panels)

# %% [markdown]
# ## Setup

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import sympy as sp
from sympy import (symbols, DiracDelta, integrate, diff, exp, sqrt, pi,
                   sin, cos, oo, latex, Eq, simplify, fourier_transform,
                   Heaviside, Function, limit, ln, Abs, Rational)
from sympy import init_printing
from IPython.display import display, Math, Markdown
import scipy.signal as sig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

init_printing(use_latex="mathjax")

def show(expr, label=""):
    try:
        if label:
            display(Math(r"\textbf{" + label + r"}\quad" + latex(expr)))
        else:
            display(Math(latex(expr)))
    except Exception:
        print(f"{label}  {expr}")

def hdr(s):
    try: display(Markdown(f"### {s}"))
    except: print(f"\n=== {s} ===")

def chk(val, ref, label, tol=1e-6):
    err = abs(float(val)-float(ref))/(abs(float(ref))+1e-30)
    ok  = err < tol
    print(f"  [{'PASS' if ok else 'FAIL'}]  {label}  got={float(val):.8g}  ref={float(ref):.8g}")
    return ok

print("=== Dirac Delta: Mental Steps + Qualcomm DSP ===")

# %% [markdown]
# ---
# ## §1 · The 7 Mental Steps — What You Run In Your Head
#
# Before touching the math, run this checklist mentally every time you see δ.

# %%
hdr("§1 — The 7 Mental Steps")

print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │              DIRAC DELTA: 7 MENTAL STEPS                               │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  STEP 1 — It is NOT a function.                                         │
  │    A function assigns a number to each input.                           │
  │    delta(x) = "infinity at 0, zero elsewhere" is not a number.         │
  │    Think of it as a MACHINE that acts on other functions via integral.  │
  │    Mental image: a spike so thin and tall that area = 1 exactly.       │
  │                                                                         │
  │  STEP 2 — It SIFTS. That is its one job.                                │
  │    int f(x) * delta(x-a) dx = f(a)                                     │
  │    Mental check: "where does delta(x-a) fire?"  Answer: at x=a.        │
  │    Then: "what is f there?"  Answer: f(a).  Done.                      │
  │                                                                         │
  │  STEP 3 — Scaling compresses the spike, so the peak grows.             │
  │    delta(ax) = delta(x) / |a|                                           │
  │    Mental check: "is the argument scaled?" -> divide by |scale|.       │
  │    Why: area must stay = 1. Narrower base -> taller peak.              │
  │                                                                         │
  │  STEP 4 — Derivative flips the derivative onto the test function.      │
  │    int f(x) * delta'(x-a) dx = -f'(a)                                  │
  │    Mental check: IBP. The boundary term vanishes (delta=0 at +/-inf).  │
  │    Result: minus sign + evaluate derivative of f at a.                 │
  │                                                                         │
  │  STEP 5 — Composition: find zeros, divide by |slope|.                  │
  │    delta(g(x)) = sum_i delta(x - x_i) / |g'(x_i)|                     │
  │    Mental check: "where does g(x)=0?" -> those are the spike locations.│
  │    Then: "how fast is g crossing zero?" -> that sets the height.       │
  │                                                                         │
  │  STEP 6 — Fourier: delta is WHITE. All frequencies equally.            │
  │    F{delta(t)} = 1   (flat spectrum, all k present)                    │
  │    F{1}        = 2*pi*delta(k)   (pure DC = spike in frequency)       │
  │    Mental image: opposite of a pure tone (which is delta in freq).     │
  │                                                                         │
  │  STEP 7 — Impulse response: delta in -> system behavior out.           │
  │    h(t) = system response to delta(t) = "what does this system DO?"    │
  │    Then: y(t) = (x * h)(t) = integral x(tau)*h(t-tau) dtau            │
  │    Mental model: every signal = sum of shifted, scaled deltas.         │
  │    The system responds to each one -> add all responses up.            │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
""")

# %% [markdown]
# ---
# ## §2 · Continuous δ(t) — Every Step Verified
#
# Apply each mental step and verify with SymPy.

# %%
hdr("§2 — Continuous delta: every mental step verified")

x, t, k, a = symbols('x t k a', real=True)

# STEP 1 — Gaussian approximation (what it "looks like")
print("  STEP 1 — Approximating family:")
eps_pos = symbols('epsilon', positive=True)
gauss = 1/(eps_pos*sqrt(pi)) * exp(-x**2/eps_pos**2)
norm  = integrate(gauss, (x, -oo, oo))
show(norm, "norm =")
chk(float(norm), 1.0, "Gaussian norm = 1")
lim_peak = limit(gauss.subs(x,0), eps_pos, 0, '+')
show(lim_peak, "peak as eps->0 =")

# STEP 2 — Sifting
print("\n  STEP 2 — Sifting property:")
test_fns = [
    (x**3,          3,   27,   "x^3 at x=3"),
    (sp.exp(x),     0,   1,    "e^x at x=0"),
    (sp.cos(x),     pi,  -1,   "cos(x) at x=pi"),
    (x**2 - 2*x,   -1,   3,   "x^2-2x at x=-1"),
]
for f_expr, a_val, expected, label in test_fns:
    result = integrate(f_expr * DiracDelta(x - a_val), (x, -oo, oo))
    show(result, f"  int {label} * delta(x-{a_val}) =")
    chk(float(result.evalf()), expected, label)

# STEP 3 — Scaling
print("\n  STEP 3 — Scaling: delta(ax) = delta(x)/|a|")
a_sym = symbols('a', positive=True)
for a_val in [2, 3, 5, 0.5]:
    result = integrate(DiracDelta(a_val*x), (x, -oo, oo))
    show(result, f"  int delta({a_val}x) dx =")
    chk(float(result), 1/a_val, f"delta({a_val}x) -> 1/{a_val}")

# Combined: int f(x)*delta(ax-b)dx = f(b/a)/|a|
r_combined = integrate(x**2 * DiracDelta(3*x - 6), (x, -oo, oo))
show(r_combined, "int x^2 * delta(3x-6) dx = (x=2)/3 =")
chk(float(r_combined), 4/3, "x^2*delta(3x-6) = f(2)/3 = 4/3")

# STEP 4 — Derivative (mental: IBP, flip to -f'(a))
print("\n  STEP 4 — Derivative: int f(x)*delta'(x-a) = -f'(a)")
deriv_tests = [
    (x**3,   2,  -3*4,    "x^3: -[3x^2]_{x=2} = -12"),
    (sp.sin(x), pi/2, -sp.cos(pi/2), "sin: -cos(pi/2) = 0"),
    (sp.exp(x), 0,  -1,   "e^x: -e^0 = -1"),
]
for f_expr, a_val, expected, label in deriv_tests:
    result = integrate(f_expr * DiracDelta(x - a_val, 1), (x, -oo, oo))
    show(result, f"  int {label}:")
    chk(float(result.evalf()), float(sp.sympify(expected).evalf()), label)

# STEP 5 — Composition
print("\n  STEP 5 — Composition: delta(g(x))")
g = x**2 - 9    # zeros at x=+-3, g'=2x
result_comp = integrate((x+1)*DiracDelta(x**2-9), (x, -oo, oo))
# Manual: sum over x=3: (3+1)/|6| + x=-3: (-3+1)/|-6| = 4/6 + (-2)/6 = 2/6 = 1/3
show(result_comp, "int (x+1)*delta(x^2-9) dx =")
chk(float(result_comp), 1/3, "(x+1)*delta(x^2-9) = 1/3")

# STEP 6 — Fourier: F{delta} = 1
print("\n  STEP 6 — Fourier: F{delta(t)} = 1 (white spectrum)")
ft_delta = integrate(DiracDelta(t)*sp.exp(-sp.I*k*t), (t, -oo, oo))
show(ft_delta, "F{delta(t)}(k) =")
chk(float(ft_delta.evalf()), 1.0, "FT of delta = 1")

# F{delta(t-t0)} = e^{-ikt0}  (phase shift only, magnitude still 1)
t0 = symbols('t_0', real=True)
ft_shifted = integrate(DiracDelta(t-t0)*sp.exp(-sp.I*k*t), (t, -oo, oo))
show(ft_shifted, "F{delta(t-t0)}(k) =")
print("  |F{delta(t-t0)}| = 1 for all k  (delay = phase shift, not magnitude change)")
chk(float(sp.Abs(ft_shifted.subs([(k,1),(t0,2)])).evalf()), 1.0, "|FT shifted delta|=1")

# STEP 7 — Convolution identity
print("\n  STEP 7 — Convolution: f * delta = f")
tau = symbols('tau', real=True)
f_test = sp.exp(-t**2)
conv_result = integrate(f_test.subs(t,tau) * DiracDelta(t - tau), (tau, -oo, oo))
show(conv_result, "(f * delta)(t) =")
chk(float(conv_result.subs(t,1).evalf()), float(f_test.subs(t,1).evalf()), "f*delta = f at t=1")

# %% [markdown]
# ---
# ## §3 · Discrete δ[n] — The Qualcomm Hexagon DSP View
#
# In DSP hardware (Snapdragon Hexagon, TI C6x, ARM Cortex-M CMSIS-DSP):
# $$\delta[n] = \begin{cases}1 & n=0 \\ 0 & n\neq 0\end{cases}$$
# This is EXACT — no limits, no distributions. Just a 1 in a sea of zeros.

# %%
hdr("§3 — Discrete delta[n]: exact, implementable, Hexagon DSP")

N = 32
n_arr = np.arange(-N//2, N//2)

# Discrete delta
delta_d = (n_arr == 0).astype(float)

print("  delta[n] around n=0:")
print(f"  n  = {n_arr[N//2-4:N//2+5].tolist()}")
print(f"  d  = {delta_d[N//2-4:N//2+5].astype(int).tolist()}")
print(f"  sum(delta[n]) = {delta_d.sum():.0f}  (= 1, discrete norm)")
chk(delta_d.sum(), 1.0, "discrete delta sums to 1")

# Sifting in discrete: sum f[n]*delta[n-k] = f[k]
f_discrete = np.sin(2*np.pi*0.1*n_arr) + 0.3*np.cos(2*np.pi*0.3*n_arr)
k_sift = 5
delta_shifted = (n_arr == k_sift).astype(float)
sifted_value = np.sum(f_discrete * delta_shifted)
print(f"\n  Discrete sifting: sum f[n]*delta[n-{k_sift}] = f[{k_sift}]")
print(f"  Computed: {sifted_value:.6f}")
print(f"  Direct:   {f_discrete[n_arr==k_sift][0]:.6f}")
chk(sifted_value, f_discrete[n_arr==k_sift][0], "discrete sifting")

# Discrete Fourier transform of delta[n]
DTFT_delta = np.ones(N)   # F{delta[n]} = 1 for all omega
print(f"\n  DTFT of delta[n] = 1 for ALL omega  (white — every frequency present equally)")
print(f"  |DTFT[delta]| = {np.abs(DTFT_delta).mean():.1f}  (flat)")

# Shift property: delta[n-n0] -> e^{-j*omega*n0}
omega = np.linspace(0, 2*np.pi, 256)
n0 = 8
DTFT_shifted = np.exp(-1j * omega * n0)
print(f"  DTFT of delta[n-{n0}] = e^{{-j*omega*{n0}}}  (magnitude = 1, phase = -{n0}*omega)")
chk(np.abs(DTFT_shifted).mean(), 1.0, "shifted delta DTFT magnitude = 1")

# Z-transform of delta[n] = 1
print("\n  Z-transform: Z{delta[n]} = sum delta[n]*z^{-n} = z^0 = 1")
print("  Z{delta[n-k]} = z^{-k}  (pure delay of k samples)")
print("  This is why a unit delay in a filter = multiply by z^{-1}")

# %% [markdown]
# ---
# ## §4 · Convolution = Sum of Shifted Impulses — The Core DSP Identity
#
# Every signal can be written as:
# $$x[n] = \sum_{k=-\infty}^{\infty} x[k]\,\delta[n-k]$$
# Apply a LINEAR TIME-INVARIANT system $\mathcal{H}$:
# $$y[n] = \mathcal{H}\{x[n]\} = \sum_k x[k]\,\mathcal{H}\{\delta[n-k]\}
#        = \sum_k x[k]\,h[n-k] = (x * h)[n]$$
# **This is why impulse response determines everything about an LTI system.**

# %%
hdr("§4 — Convolution = sum of shifted impulses (LTI system identity)")

print("""
  Mental model for convolution:
    1. Decompose x[n] = sum x[k]*delta[n-k]   (every signal = weighted deltas)
    2. System is LINEAR -> output = sum of responses to each delta
    3. System is TIME-INVARIANT -> response to delta[n-k] = h[n-k]
    4. Therefore: y[n] = sum x[k]*h[n-k] = (x*h)[n]

  That's the ENTIRE derivation. Linearity + time-invariance -> convolution.
""")

# Demonstrate: build a signal from shifted deltas
x_signal = np.array([0, 0, 3, -1, 2, 0, 4, -2, 0, 0], dtype=float)
n_sig    = np.arange(len(x_signal))
print(f"  x[n] = {x_signal.tolist()}")
print(f"  = 3*delta[n-2] + (-1)*delta[n-3] + 2*delta[n-4] + 4*delta[n-6] + (-2)*delta[n-7]")

# Reconstruct from sum of weighted deltas
x_recon = np.zeros_like(x_signal)
for k in range(len(x_signal)):
    if x_signal[k] != 0:
        delta_k = (n_sig == k).astype(float)
        x_recon += x_signal[k] * delta_k

chk(np.max(np.abs(x_signal - x_recon)), 0, "signal = sum of weighted deltas", tol=1e-10)
print(f"  Reconstruction error: {np.max(np.abs(x_signal - x_recon)):.2e}  [PASS]")

# Impulse response of a simple low-pass filter
h_lpf = np.array([0.25, 0.5, 0.25])   # 3-tap FIR, triangular window
print(f"\n  3-tap FIR impulse response h[n] = {h_lpf.tolist()}")
print(f"  (This IS the system. Know h[n] -> know everything about the system.)")

# Convolve x with h
y_direct = np.convolve(x_signal, h_lpf, mode='same')

# Verify: manual sum-of-shifted-responses
y_manual = np.zeros(len(x_signal) + len(h_lpf) - 1)
for k in range(len(x_signal)):
    if x_signal[k] != 0:
        y_manual[k:k+len(h_lpf)] += x_signal[k] * h_lpf

# mode='same' is center-aligned: y_direct[i] == y_manual[i + offset]
# where offset = (len(h_lpf)-1)//2
offset = (len(h_lpf) - 1) // 2
N = len(x_signal)
chk(np.max(np.abs(y_direct - y_manual[offset:offset+N])), 0,
    "conv interior == sum of shifted responses", tol=1e-10)
print(f"  y[n] = {np.round(y_direct,3).tolist()}")

# %% [markdown]
# ---
# ## §5 · FIR Filter Design via δ[n] — Qualcomm Hexagon Implementation
#
# FIR filter: $H(z) = \sum_{k=0}^{N-1} h[k]\,z^{-k}$
# Coefficients $h[k]$ ARE the impulse response.
# Design = choosing h[k] to shape frequency response.

# %%
hdr("§5 — FIR Filter Design: impulse response = filter coefficients")

# Design a low-pass FIR using windowed sinc
def design_fir_lpf(N, fc, window='hann'):
    """N-tap low-pass FIR, cutoff fc (normalized 0..0.5)."""
    n_fir = np.arange(N) - (N-1)/2
    h = np.sinc(2*fc*n_fir)    # ideal sinc = IDTFT of rectangular window
    w = np.hanning(N) if window=='hann' else np.ones(N)
    h *= w
    h /= h.sum()               # normalize to unity DC gain
    return h

h_lpf50 = design_fir_lpf(63, 0.1)    # 63-tap, fc=0.1 (10% of Nyquist)
h_bpf   = design_fir_lpf(63, 0.3) - design_fir_lpf(63, 0.1)  # bandpass

print(f"  63-tap LPF: h[0..4] = {np.round(h_lpf50[:5],5).tolist()}")
print(f"  sum(h) = {h_lpf50.sum():.6f}  (unity DC gain)")
chk(h_lpf50.sum(), 1.0, "LPF DC gain = 1", tol=1e-5)

# Frequency response
w_fir, H_lpf = sig.freqz(h_lpf50, 1, worN=1024)
w_fir, H_bpf = sig.freqz(h_bpf,  1, worN=1024)

print(f"\n  LPF passband gain at DC:      {abs(H_lpf[0]):.4f}  (target 1.0)")
print(f"  LPF stopband attenuation at 0.3: {20*np.log10(abs(H_lpf[307])+1e-12):.1f} dB")
print(f"  BPF peak gain:                {abs(H_bpf).max():.4f}")

# Delta -> impulse response demo
print("""
  Qualcomm Hexagon DSP implementation (pseudocode):
    // FIR filter: y[n] = sum_{k=0}^{N-1} h[k]*x[n-k]
    // Coefficients h[] ARE delta[n] passed through the filter design
    void fir_filter(const int16_t *h, int16_t *x_buf,
                    int16_t *y, int N, int n_samples) {
        for (int n = 0; n < n_samples; n++) {
            int32_t acc = 0;
            for (int k = 0; k < N; k++)
                acc += (int32_t)h[k] * x_buf[n-k];  // MAC: multiply-accumulate
            y[n] = (int16_t)(acc >> 15);              // Q15 fixed-point
        }
    }
    // Hexagon SIMD: vmpa(v0.ub, v1.b):acc  -> 32 MACs per cycle
    // At 1 GHz: 32e9 MACs/s  =>  63-tap filter at 500 MHz sample rate: trivial
""")

# CMAC throughput estimate (Hexagon V65)
N_taps    = 63
fs_Hz     = 500e6     # 500 MHz sample rate (5G NR)
CMAC_per_cycle = 32   # Hexagon V65 vector MACs
f_clock   = 1e9       # 1 GHz
MACs_per_sample = N_taps
cycles_per_sample = MACs_per_sample / CMAC_per_cycle
max_sample_rate = f_clock / cycles_per_sample
print(f"  Hexagon V65 throughput analysis:")
print(f"  {N_taps}-tap FIR needs {N_taps} MACs/sample")
print(f"  32 MACs/cycle -> {cycles_per_sample:.2f} cycles/sample")
print(f"  At 1 GHz: max sample rate = {max_sample_rate/1e6:.0f} MHz >> {fs_Hz/1e6:.0f} MHz target  [headroom OK]")

# %% [markdown]
# ---
# ## §6 · Sampling Theorem — Multiplying by a Pulse Train
#
# $$x_s(t) = x(t) \cdot \sum_{n=-\infty}^{\infty} \delta(t - nT_s)
#           = \sum_n x(nT_s)\,\delta(t-nT_s)$$
# Fourier transform of a pulse train is ALSO a pulse train:
# $$\mathcal{F}\!\left\{\sum_n\delta(t-nT_s)\right\} = f_s\sum_k\delta(f-kf_s)$$
# Sampling = REPLICATION of spectrum with period $f_s$.
# Aliasing = replicas overlap when $f_s < 2f_{max}$ (Nyquist).

# %%
hdr("§6 — Sampling Theorem: delta train + spectrum replication")

# Continuous signal
t_cont = np.linspace(-2, 2, 8000)
f_sig  = 3.0   # Hz signal frequency
x_cont = np.sin(2*np.pi*f_sig*t_cont) + 0.5*np.cos(2*np.pi*5*t_cont)

# Sample at different rates
fs_good = 20.0   # > 2*5Hz = 10Hz: no aliasing
fs_bad  = 8.0    # < 2*5Hz: aliasing

t_good = np.arange(-2, 2, 1/fs_good)
t_bad  = np.arange(-2, 2, 1/fs_bad)
x_good = np.sin(2*np.pi*f_sig*t_good) + 0.5*np.cos(2*np.pi*5*t_good)
x_bad  = np.sin(2*np.pi*f_sig*t_bad)  + 0.5*np.cos(2*np.pi*5*t_bad)

print(f"  Signal: {f_sig}Hz + 5Hz  (f_max = 5Hz, Nyquist = 10Hz)")
print(f"  fs={fs_good}Hz: ABOVE Nyquist -> no aliasing")
print(f"  fs={fs_bad}Hz:  BELOW Nyquist -> aliasing (5Hz folds to {5-fs_bad}Hz)")
print(f"  Aliased frequency: {abs(5 - round(5/fs_bad)*fs_bad):.1f} Hz  (= 5 mod {fs_bad} = {5%fs_bad:.1f})")

# SymPy: pulse train FT
print("\n  SymPy: FT of pulse train")
T_s, f_s_sym = symbols('T_s f_s', positive=True)
print("  sum_n delta(t-n*T_s)  <--FT-->  f_s * sum_k delta(f - k*f_s)")
print("  Proof: Fourier series of periodic delta train")
print("  -> a_n = (1/T_s) * integral_{-T/2}^{T/2} delta(t) e^{-j*2pi*n*t/T} dt")
print("         = 1/T_s  for all n")
print("  -> F{train} = (1/T_s) * sum e^{-j*2pi*n*f*T_s} = f_s * sum delta(f-n*f_s)  QED")

# Nyquist-Shannon statement
print("""
  NYQUIST-SHANNON SAMPLING THEOREM:
  If x(t) is bandlimited to B Hz (X(f)=0 for |f|>B),
  then x(t) is COMPLETELY determined by samples at fs >= 2B.
  Reconstruction: x(t) = sum_n x[n] * sinc(fs*(t - n/fs))
  The sinc is the IDTFT of the ideal rectangular lowpass filter.
  At Qualcomm: 5G NR uses up to 100MHz BW -> needs >= 200 MSPS ADC.
""")

# 5Hz aliases to |5 - round(5/8)*8| = |5-8| = 3Hz
chk(abs(5 - round(5/fs_bad)*fs_bad), 3.0, "alias: 5Hz folds to 3Hz at fs=8", tol=1e-9)

# %% [markdown]
# ---
# ## §7 · CDMA Spreading — Near-Delta Autocorrelation
#
# In CDMA (Code Division Multiple Access — Qualcomm's core patent):
# The spreading code is chosen so its autocorrelation ≈ δ[n].
# This lets multiple users share the same frequency band.
# $$R_{cc}[k] = \sum_n c[n]\,c[n-k] \approx N\,\delta[k]$$

# %%
hdr("§7 — CDMA: spreading codes with near-delta autocorrelation")

# Gold code generator (simplified m-sequence based)
def m_sequence(taps, length, seed=None):
    """Maximal-length sequence via LFSR (Fibonacci form, shift-right).
    taps: list of feedback positions (1-indexed).  Polynomial: x^n + x^(n-taps[1]+1) + ...
    For n=10, primitive poly x^10+x^8+x^5+x^1+1 uses taps [10,5,2,1].
    Seed: defaults to all-ones (safe start for any primitive poly).
    """
    n_taps = max(taps)
    if seed is None:
        seed = (1 << n_taps) - 1   # all-ones: guarantees non-zero feedback bit
    state = seed & ((1 << n_taps) - 1)
    mask = (1 << n_taps) - 1
    seq = []
    for _ in range(length):
        seq.append(state & 1)
        new_bit = 0
        for t in taps:
            new_bit ^= (state >> (t-1)) & 1
        state = ((state >> 1) | (new_bit << (n_taps-1))) & mask
    return np.array(seq, dtype=float)*2 - 1   # map {0,1} -> {-1,+1}

# CDMA IS-95 uses 64-chip Walsh codes and 32768-chip PN codes
# Primitive poly for degree 10: x^10+x^8+x^5+x+1 (verified taps [10,5,2,1])
c1 = m_sequence([10, 5, 2, 1], 1023)           # User A spreading code
c2 = m_sequence([10, 5, 2, 1], 1023, seed=0b0101010101)  # User B: different phase

# Circular autocorrelation via FFT — this is what the m-seq property guarantees:
#   R_circ[0] = N,  R_circ[k] = -1 for k != 0
C1_f     = np.fft.fft(c1)
R_circ   = np.real(np.fft.ifft(C1_f * np.conj(C1_f)))
peak_auto = R_circ[0]
off_auto  = np.max(np.abs(R_circ[1:]))

# Linear correlation for cross-code (for DSP illustration only)
R_cross   = np.correlate(c1, c2, mode='full')
peak_cross = np.max(np.abs(R_cross))

N_seq = len(c1)
print(f"  m-sequence length N={N_seq}")
print(f"  Circular autocorr peak (lag=0): {peak_auto:.1f}  (= N = {N_seq})")
print(f"  Circular autocorr off-peak max: {off_auto:.4f}   (ideal = 1)")
print(f"  Peak-to-sidelobe ratio: {peak_auto/off_auto:.1f}  (= N = {N_seq})")
print(f"  Cross-correlation max: {peak_cross:.0f}  (different code, different user)")
chk(peak_auto, N_seq, "autocorr peak = N")
chk(peak_auto / off_auto, N_seq, "peak-to-sidelobe ratio = N", tol=0.01)

# Processing gain
PG_dB = 10*np.log10(len(c1))
print(f"\n  Processing gain: PG = 10*log10(N) = {PG_dB:.1f} dB")
print(f"  This means signal buried {PG_dB:.0f} dB BELOW noise floor is still recoverable!")
print(f"  The near-delta autocorrelation IS what gives you this gain.")

print("""
  Mental model for CDMA:
    User A transmits data*c_A(t)  -> spread across bandwidth B
    User B transmits data*c_B(t)  -> spread across same bandwidth B
    Receiver for A: multiply received signal by c_A(t), then integrate
      = data_A * R_{c_A,c_A}(0) + data_B * R_{c_B,c_A}(0) + noise
      ~ data_A * N  +  data_B * 1  + noise
      -> correlator output dominated by user A's data  [QED]

  R_auto ~ N*delta[k] is the MATHEMATICAL reason CDMA works.
  Qualcomm's patents (1986-1992) were essentially:
    "Use m-sequences with near-delta autocorrelation for cellular."
""")

# %% [markdown]
# ---
# ## §8 · Figure (12 panels)

# %%
fig = plt.figure(figsize=(20, 15))
gf  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.35)
c4  = ["#4C72B0","#DD8452","#55A868","#C44E52"]

# P1: Gaussian approximation families
ax1 = fig.add_subplot(gf[0,0])
x_plt = np.linspace(-1.5, 1.5, 1000)
for eps, col in zip([0.5, 0.25, 0.1, 0.04], c4):
    y = (1/(eps*np.sqrt(np.pi)))*np.exp(-x_plt**2/eps**2)
    ax1.plot(x_plt, y, color=col, lw=1.5, label=f'eps={eps}')
ax1.set_title("STEP 1: Gaussian -> delta(x)", fontsize=9)
ax1.set_xlabel("x"); ax1.legend(fontsize=6); ax1.set_ylim(0,12)

# P2: Sifting visualization
ax2 = fig.add_subplot(gf[0,1])
x_s = np.linspace(-1, 5, 500)
f_s_plot = np.sin(x_s) + 0.5*x_s
eps_s = 0.08
delta_at3 = (1/(eps_s*np.sqrt(np.pi)))*np.exp(-(x_s-3)**2/eps_s**2)
ax2.plot(x_s, f_s_plot, 'b-', lw=2, label='f(x)=sin+x/2')
ax2.fill_between(x_s, f_s_plot*delta_at3/10, alpha=0.5, color='orange',
                  label='f*delta(x-3)/10')
ax2.axvline(3, color='r', ls='--', lw=1, label='x=3')
ax2.axhline(np.sin(3)+1.5, color='g', ls=':', lw=1, label=f'f(3)={np.sin(3)+1.5:.2f}')
ax2.set_title("STEP 2: Sifting at x=3", fontsize=9)
ax2.legend(fontsize=6); ax2.set_xlabel("x")

# P3: Scaling — narrower base, taller peak
ax3 = fig.add_subplot(gf[0,2])
for a_val, col in zip([0.5,1,2,4], c4):
    y = (1/(0.1*np.sqrt(np.pi)))*np.exp(-(a_val*x_plt)**2/0.01)/(a_val+1e-10)
    y = a_val*(1/(0.1*np.sqrt(np.pi)))*np.exp(-(a_val*x_plt)**2/(0.01))
    # delta(ax): peak = |a|/(eps*sqrt(pi)) -> narrower
    eps_v = 0.15
    y2 = (a_val/(eps_v*np.sqrt(np.pi)))*np.exp(-(a_val*x_plt)**2/eps_v**2)/a_val
    ax3.plot(x_plt, y2, color=col, lw=1.5, label=f'delta({a_val}x)')
ax3.set_xlim(-1,1); ax3.set_title("STEP 3: Scaling delta(ax)", fontsize=9)
ax3.legend(fontsize=6); ax3.set_xlabel("x")

# P4: Discrete delta[n]
ax4 = fig.add_subplot(gf[0,3])
n_d = np.arange(-8, 9)
delta_n = (n_d == 0).astype(float)
ax4.stem(n_d, delta_n, markerfmt='ro', linefmt='b-', basefmt='k-')
ax4.set_title("delta[n]: exact 1 at n=0", fontsize=9)
ax4.set_xlabel("n"); ax4.set_ylabel("delta[n]")

# P5: Signal decomposition into shifted deltas
ax5 = fig.add_subplot(gf[1,0])
n_decomp = np.arange(10)
x_decomp = np.array([0, 0, 3, -1, 2, 0, 4, -2, 0, 0], dtype=float)
colors_d = plt.cm.tab10(np.linspace(0,1,10))
for k in range(10):
    if x_decomp[k] != 0:
        y_k = np.zeros(10); y_k[k] = x_decomp[k]
        ax5.stem(n_decomp, y_k, markerfmt='.', linefmt='-',
                 basefmt=' ', label=f'{x_decomp[k]:+.0f}*d[n-{k}]')
ax5.stem(n_decomp, x_decomp, markerfmt='ko', linefmt='k--',
         basefmt='k-', label='x[n]')
ax5.set_title("x[n] = sum of shifted deltas", fontsize=9)
ax5.set_xlabel("n"); ax5.legend(fontsize=5)

# P6: Convolution step by step
ax6 = fig.add_subplot(gf[1,1])
n_conv = np.arange(len(x_decomp))
ax6.stem(n_conv, x_decomp, markerfmt='bs', linefmt='b-',
         basefmt='k-', label='x[n]')
ax6.stem(np.arange(3), np.array([0.25,0.5,0.25]), markerfmt='r^',
         linefmt='r--', basefmt='k-', label='h[n]=[.25,.5,.25]')
ax6.stem(n_conv, y_direct, markerfmt='go', linefmt='g-',
         basefmt='k-', label='y=x*h')
ax6.set_title("Convolution y = x * h", fontsize=9)
ax6.set_xlabel("n"); ax6.legend(fontsize=6)

# P7: FIR frequency responses
ax7 = fig.add_subplot(gf[1,2])
ax7.plot(w_fir/np.pi*0.5, 20*np.log10(np.abs(H_lpf)+1e-12), 'b-', lw=2, label='LPF fc=0.1')
ax7.plot(w_fir/np.pi*0.5, 20*np.log10(np.abs(H_bpf)+1e-12), 'r-', lw=2, label='BPF 0.1-0.3')
ax7.set_xlabel("Normalized freq"); ax7.set_ylabel("dB")
ax7.set_ylim(-80, 5); ax7.set_title("FIR filter frequency response", fontsize=9)
ax7.legend(fontsize=7); ax7.grid(True, alpha=0.3)

# P8: Sampling — no aliasing vs aliasing
ax8 = fig.add_subplot(gf[1,3])
ax8.plot(t_cont, x_cont, 'b-', alpha=0.4, lw=1, label='continuous')
ax8.stem(t_good, x_good, markerfmt='go', linefmt='g-',
         basefmt='k-', label=f'fs={fs_good}Hz OK')
ax8.stem(t_bad,  x_bad,  markerfmt='r^', linefmt='r--',
         basefmt='k-', label=f'fs={fs_bad}Hz ALIAS')
ax8.set_xlim(-0.5, 0.5); ax8.set_title("Sampling: Nyquist vs aliasing", fontsize=9)
ax8.set_xlabel("t (s)"); ax8.legend(fontsize=6)

# P9: Pulse train spectrum (conceptual)
ax9 = fig.add_subplot(gf[2,0])
f_range = np.linspace(-3, 3, 1000)
fs_demo = 1.0
# Replicated spectrum
for k in range(-2, 3):
    f_center = k*fs_demo
    ax9.fill_between(f_range, np.where(np.abs(f_range-f_center)<0.2, 0.8, 0),
                     alpha=0.4, color=c4[abs(k)%4])
ax9.set_xlabel("f (normalized)"); ax9.set_ylabel("X_s(f)")
ax9.set_title("Sampling = spectrum replication", fontsize=9)
ax9.axvline(-0.5, color='k', ls='--', lw=1, label='Nyquist')
ax9.axvline(0.5,  color='k', ls='--', lw=1)
ax9.legend(fontsize=7)

# P10: CDMA autocorrelation
ax10 = fig.add_subplot(gf[2,1])
lags_circ = np.arange(-N_seq//2, N_seq//2)
R_circ_shift = np.roll(R_circ, N_seq//2)
mask50 = (lags_circ>=-50)&(lags_circ<=50)
ax10.plot(lags_circ[mask50], R_circ_shift[mask50], 'b-', lw=1)
ax10.axhline(1, color='r', ls='--', lw=1, label='off-peak = 1')
ax10.axhline(-1,color='r', ls='--', lw=1)
ax10.set_title(f"CDMA circ-autocorr (N={N_seq})", fontsize=9)
ax10.set_xlabel("lag k"); ax10.set_ylabel("R[k]")
ax10.legend(fontsize=7)

# P11: Cross-correlation (different users = orthogonal)
ax11 = fig.add_subplot(gf[2,2])
lags_lin = np.arange(-(N_seq-1), N_seq)
lags_c = lags_lin[(lags_lin>=-50)&(lags_lin<=50)]
R_c_p  = R_cross[(lags_lin>=-50)&(lags_lin<=50)]
ax11.plot(lags_c, R_c_p, 'r-', lw=1)
ax11.set_title(f"CDMA cross-corr (users A,B)", fontsize=9)
ax11.set_xlabel("lag k"); ax11.set_ylabel("R_cross[k]")
ax11.text(0.05,0.9,f"max={peak_cross:.0f} vs N={len(c1)}",
          transform=ax11.transAxes, fontsize=8)

# P12: The 7 mental steps diagram
ax12 = fig.add_subplot(gf[2,3])
ax12.axis('off')
steps = [
    "1. Not a function — a testing machine",
    "2. Sifting: int f*delta(x-a) = f(a)",
    "3. Scaling: delta(ax) = delta(x)/|a|",
    "4. Derivative: int f*delta' = -f'(a)",
    "5. Composition: find zeros of g(x)",
    "6. Fourier: F{delta} = 1  (white)",
    "7. Impulse response: h = system ID",
]
for i, step in enumerate(steps):
    ax12.text(0.02, 0.88-i*0.13, f"{step}",
             transform=ax12.transAxes, fontsize=8,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.2',
                       facecolor=['#EEF','#EFE','#FEE','#FFE',
                                  '#EEF','#EFE','#FEE'][i],
                       alpha=0.8))
ax12.set_title("7 Mental Steps", fontsize=9)

fig.suptitle(
    "Dirac Delta: 7 Mental Steps + Qualcomm DSP  "
    "(Sifting · Scaling · Derivative · Fourier · Convolution · Sampling · CDMA)",
    fontsize=11, fontweight='bold', y=1.01
)

import pathlib
out_dir  = pathlib.Path(__file__).parent if "__file__" in dir() else pathlib.Path("repl")
out_path = out_dir / "_out_dirac_mental_dsp.png"
fig.savefig(out_path, dpi=130, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close(fig)
print("\n=== All 7 mental steps + DSP sections complete ===")
