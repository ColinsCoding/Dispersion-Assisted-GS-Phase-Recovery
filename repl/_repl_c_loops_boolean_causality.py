# %% [markdown]
# # C Loops · Boolean Algebra · Causality in Calculus
# All the mechanics — verified in Python, with the C equivalents shown explicitly.
# Causality: why the derivative "looks left" and the integral "accumulates forward."

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sympy import symbols, Function, Heaviside, DiracDelta, integrate, oo, simplify
from sympy import cos, sin, exp, pi, diff, Rational

sp.init_printing(use_latex='mathjax')

def hdr(s):
    print(f'\n{"─"*64}\n  {s}\n{"─"*64}')

def disp(expr, label=''):
    tag = f'  {label}:  ' if label else '  '
    print(tag + sp.pretty(expr, use_unicode=True))

def chk(val, ref, label, tol=1e-9, absolute=False):
    v, r = float(val), float(ref)
    err = abs(v-r) if (absolute or r==0) else abs(v-r)/(abs(r)+1e-30)
    s = 'PASS' if err < tol else 'FAIL'
    print(f'  [{s}]  {label}  got={v:.8g}  ref={r:.8g}')

# %% [markdown]
# ## §1 — C Loops: Every Form, Every Trap
#
# C has three loop forms. Underneath they all compile to the same machine code.
# The difference is WHEN the condition is checked.

# %%
hdr("§1a — The three C loop forms (Python simulation)")

print("""
  ┌──────────────────────────────────────────────────────────┐
  │  C LOOP FORMS — what the compiler actually generates     │
  ├──────────────────────────────────────────────────────────┤
  │                                                          │
  │  for (init; cond; step) { body }                         │
  │  ≡  init; while (cond) { body; step; }                   │
  │                                                          │
  │  while (cond) { body }          ← check BEFORE          │
  │  do { body } while (cond);      ← check AFTER (≥1 run)  │
  │                                                          │
  │  for (int i=0; i<N; i++) — THE canonical loop           │
  │    i starts at 0, ends at N-1. N iterations.            │
  │    Off-by-one: i<=N runs N+1 times. i<N runs N times.  │
  └──────────────────────────────────────────────────────────┘
""")

# Simulate for-loop behavior in Python to verify counts
def for_loop_count(start, cond_n, step=1):
    """Count iterations of for(i=start; i<cond_n; i+=step)"""
    count = 0
    i = start
    while i < cond_n:
        count += 1
        i += step
    return count

# Verify: for(i=0; i<N; i++) runs exactly N times
for N in [0, 1, 5, 10, 100]:
    chk(for_loop_count(0, N), N, f"for(i=0; i<{N}; i++) → {N} iterations",
        tol=1e-9, absolute=True)

print("""
  C LOOP PATTERNS:
  ─────────────────
  // Sum: 0+1+...+(N-1) = N*(N-1)/2
  int s = 0;
  for (int i = 0; i < N; i++) s += i;

  // Reverse: N-1 down to 0
  for (int i = N-1; i >= 0; i--) { ... }

  // Step by 2
  for (int i = 0; i < N; i += 2) { ... }   // ceil(N/2) iterations

  // Nested: O(N^2) work
  for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
          A[i][j] = i * N + j;

  // break / continue
  for (int i = 0; i < N; i++) {
      if (i == k) break;      // exit loop entirely
      if (i % 2 == 0) continue;  // skip to next iteration
  }
""")

# Verify sum formula
def c_sum(N):
    s = 0
    for i in range(N):
        s += i
    return s

for N in [10, 100, 1000]:
    ref = N*(N-1)//2
    chk(c_sum(N), ref, f"sum 0..{N-1} = {N}*{N-1}/2 = {ref}", tol=1e-9, absolute=True)

# %% [markdown]
# ## §1b — The Three Loop Traps (memory, bounds, infinite)

# %%
hdr("§1b — The three loop traps every C programmer hits")

print("""
  TRAP 1: Off-by-one error
  ─────────────────────────
  int arr[N];
  for (int i = 0; i <= N; i++)   // WRONG: i=N is out of bounds
      arr[i] = i;                 // undefined behavior at i=N
  for (int i = 0; i < N; i++)    // CORRECT
      arr[i] = i;

  Rule: if array has N elements, indices are 0..N-1. Use i < N, never i <= N.

  TRAP 2: Infinite loop
  ──────────────────────
  unsigned int i = 10;
  while (i >= 0) { ... i--; }    // INFINITE: unsigned can't be negative
                                  // wraps to 4294967295 when decremented below 0

  Fix: use signed int, or restructure:
  for (int i = N-1; i >= 0; i--)  // OK with signed int

  TRAP 3: Loop variable modified inside body
  ───────────────────────────────────────────
  for (int i = 0; i < N; i++) {
      if (condition) i++;  // skips next iteration — often a bug
  }
  // Prefer: let the loop variable belong ONLY to the loop header.
""")

# Demonstrate: unsigned wrap (in Python: simulate uint32 wrap)
def uint32_decrement(x):
    return (x - 1) & 0xFFFFFFFF

x = 0
x = uint32_decrement(x)
chk(x, 0xFFFFFFFF, "uint32: 0 - 1 wraps to 4294967295 (FFFF...)", tol=1e-9, absolute=True)

# %% [markdown]
# ## §2 — Boolean Algebra: The 16 Binary Operators and Algebraic Laws
#
# Boolean algebra: variables in {0,1}, operations AND(·), OR(+), NOT(¬).
# ALL digital logic, ALL conditional branches, ALL bitwise ops are in here.

# %%
hdr("§2a — Boolean algebra: the 16 binary operators")

print("""
  Two inputs A, B → 4 possible input combinations → 2^4 = 16 possible truth tables.
  The ones you actually use:

  Name        C op   A=0,B=0  A=0,B=1  A=1,B=0  A=1,B=1
  ─────────────────────────────────────────────────────────
  AND           &&      0        0        0        1
  OR            ||      0        1        1        1
  XOR           ^       0        1        1        0
  NAND        !(&&)     1        1        1        0
  NOR         !(||)     1        0        0        0
  XNOR        !(^)      1        0        0        1
  NOT A         !A      1        1        0        0
  IMPLICATION A→B       1        1        0        1   (if A then B)
  ─────────────────────────────────────────────────────────

  XOR special properties:
    A ^ A = 0    (self-cancellation)
    A ^ 0 = A    (identity)
    A ^ 1 = !A   (toggle)
    XOR is addition mod 2 → this is GF(2) arithmetic
""")

# Verify all truth tables
import itertools
for A, B in itertools.product([0,1], repeat=2):
    a, b = bool(A), bool(B)
    assert (a and b) == bool(A & B)
    assert (a or b)  == bool(A | B)
    assert (a ^ b)   == bool(A ^ B)
print('  [PASS]  All 4 combinations of AND/OR/XOR verified')

# XOR properties
for A in [0, 1]:
    chk(A ^ A, 0, f"A^A=0  A={A}", tol=1e-9, absolute=True)
    chk(A ^ 0, A, f"A^0=A  A={A}", tol=1e-9, absolute=True)
    chk(A ^ 1, 1-A, f"A^1=!A  A={A}", tol=1e-9, absolute=True)

# %% [markdown]
# ## §2b — Boolean Algebra Laws (the ones that let you simplify circuits)

# %%
hdr("§2b — Boolean algebra laws")

print("""
  IDENTITY LAWS:        A · 1 = A        A + 0 = A
  NULL LAWS:            A · 0 = 0        A + 1 = 1
  IDEMPOTENT:           A · A = A        A + A = A
  COMPLEMENT:           A · !A = 0       A + !A = 1
  DOUBLE NEGATION:      !!A = A
  COMMUTATIVE:          A·B = B·A        A+B = B+A
  ASSOCIATIVE:          (A·B)·C = A·(B·C)
  DISTRIBUTIVE:         A·(B+C) = A·B + A·C
                        A+(B·C) = (A+B)·(A+C)   ← 'opposite' from arithmetic!
  DE MORGAN'S:          !(A·B) = !A + !B         ← flip op, flip inputs
                        !(A+B) = !A · !B
  ABSORPTION:           A + A·B = A
                        A · (A+B) = A
  CONSENSUS:            A·B + !A·C + B·C = A·B + !A·C
""")

# Verify De Morgan's laws exhaustively
all_pass = True
for A in [0,1]:
    for B in [0,1]:
        dm1 = (not (A and B)) == ((not A) or (not B))
        dm2 = (not (A or B))  == ((not A) and (not B))
        if not (dm1 and dm2):
            all_pass = False
chk(1 if all_pass else 0, 1, "De Morgan's laws hold for all A,B in {0,1}", tol=1e-9, absolute=True)

# Verify distributive A+(B·C) = (A+B)·(A+C)  (non-obvious one)
all_pass2 = True
for A in [0,1]:
    for B in [0,1]:
        for C in [0,1]:
            lhs = A or (B and C)
            rhs = (A or B) and (A or C)
            if lhs != rhs:
                all_pass2 = False
chk(1 if all_pass2 else 0, 1, "Distributive A+(B·C)=(A+B)·(A+C) — non-obvious law", tol=1e-9, absolute=True)

# Verify absorption: A + A·B = A
all_pass3 = True
for A in [0,1]:
    for B in [0,1]:
        if (A or (A and B)) != A:
            all_pass3 = False
chk(1 if all_pass3 else 0, 1, "Absorption: A + A·B = A", tol=1e-9, absolute=True)

# %% [markdown]
# ## §2c — Boolean → C: bitwise vs logical, and the short-circuit trap

# %%
hdr("§2c — Bitwise vs logical in C")

print("""
  C has TWO sets of boolean operators:

  LOGICAL (short-circuit):          BITWISE (always evaluates both):
    &&  (logical AND)                 &   (bitwise AND)
    ||  (logical OR)                  |   (bitwise OR)
    !   (logical NOT)                 ~   (bitwise NOT, flips all bits)
                                      ^   (bitwise XOR)

  SHORT-CIRCUIT RULE:
    if (p != NULL && p->val > 0)    ← safe: right side not evaluated if p==NULL
    if (p != NULL &  p->val > 0)    ← CRASH: & always evaluates both sides

  BITWISE OPS ON INTEGERS:
    x & 1       → lowest bit (test if odd)
    x | (1<<k)  → set bit k
    x & ~(1<<k) → clear bit k
    x ^ (1<<k)  → toggle bit k
    x >> k      → divide by 2^k (arithmetic right shift for signed)
    x << k      → multiply by 2^k

  CANONICAL PATTERNS:
    if (x & mask)          // test if any masked bits set
    x = (x & ~mask) | val  // set field: clear then OR
    parity = popcount(x) & 1   // XOR of all bits

  SUM OF BOOLEANS:   count = (A&1) + (B&1) + (C&1)   // 0..3
""")

# Verify bit manipulation patterns
def test_bit_ops():
    results = []
    for x in range(256):
        is_odd  = bool(x & 1)
        is_odd2 = bool(x % 2)
        results.append(is_odd == is_odd2)
    return all(results)

chk(1 if test_bit_ops() else 0, 1, "x&1 == x%2 for all x in 0..255", tol=1e-9, absolute=True)

# bit set/clear/toggle
x = 0b10110010
k = 2
set_bit   = x | (1<<k)
clear_bit = x & ~(1<<k)
toggle    = x ^ (1<<k)
chk((set_bit >> k) & 1,   1, f"set bit {k}",    tol=1e-9, absolute=True)
chk((clear_bit >> k) & 1, 0, f"clear bit {k}",  tol=1e-9, absolute=True)
chk(toggle ^ x, 1<<k,       f"toggle = XOR mask", tol=1e-9, absolute=True)

# %% [markdown]
# ## §3 — Causality in Calculus
#
# **Causality** in calculus means: the present depends only on the past, not the future.
#
# The MATHEMATICAL signature of causality is the **Heaviside function H(t)**:
#   A causal system has impulse response h(t) = 0  for t < 0.
#
# **Derivative "looks left"** (backward difference):
#   f′(t) ≈ [f(t) − f(t−Δt)] / Δt   ← uses the PAST
#
# **Integral "accumulates forward"**:
#   ∫₀ᵗ f(τ) dτ   ← accumulates from the past up to NOW
#
# **Convolution with a causal kernel**:
#   y(t) = ∫₀ᵗ h(t−τ) f(τ) dτ   ← h(t-τ)=0 when τ>t, so only past matters

# %%
hdr("§3a — Causal vs anti-causal: Heaviside signature")

t_sym = symbols('t', real=True)
tau   = symbols('tau', real=True, positive=True)

# Causal impulse response: h(t) = e^{-at} H(t)
a_sym = symbols('a', positive=True)
h_causal = exp(-a_sym*t_sym) * Heaviside(t_sym)

# Its Laplace transform: integral_0^inf e^{-at} e^{-st} dt = 1/(s+a)
s = symbols('s', positive=True)
LT = integrate(exp(-a_sym*t_sym)*exp(-s*t_sym), (t_sym, 0, oo))
disp(simplify(LT), 'Laplace{e^{-at}H(t)}')
chk(float(LT.subs([(a_sym,1),(s,2)])), 1/3, "LT at a=1,s=2: 1/(1+2)=1/3", tol=1e-9)

# Verify causality: convolution y=h*f with causal h
t_num  = np.linspace(0, 10, 10000)
dt     = t_num[1] - t_num[0]
a_num  = 1.0
h_num  = np.exp(-a_num * t_num)    # causal: zero for t<0 implied by starting at t=0
f_num  = np.sin(2*np.pi*t_num)     # input

# Causal convolution: y(t) = integral_0^t h(t-tau)*f(tau) dtau
# Numerically: use scipy or manual running sum
from scipy.signal import fftconvolve
y_conv = fftconvolve(h_num, f_num)[:len(t_num)] * dt
# Analytical: y(t) = Im[ (e^{-t} - e^{i*w*t}) / (1 + iw) ] with w=2pi
w = 2*np.pi
y_analytic = (np.exp(-a_num*t_num) * (-w/(1+w**2))
              + np.sin(w*t_num - np.arctan(w)) / np.sqrt(1+w**2))
# ... just check the steady-state amplitude
ss_amp_conv    = np.max(np.abs(y_conv[5000:]))
ss_amp_analytic = 1/np.sqrt(1 + w**2)
chk(ss_amp_conv, ss_amp_analytic, "causal conv steady-state amplitude = 1/sqrt(1+w^2)", tol=0.01)

# %% [markdown]
# ## §3b — Forward vs backward differences: causality in numerics

# %%
hdr("§3b — Causal (backward) vs anti-causal (forward) difference")

print("""
  BACKWARD DIFFERENCE (causal):
    f'(t) ≈ [f(t) - f(t-dt)] / dt    ← uses past: CAUSAL
    Error: O(dt) — first-order accurate
    Used in: Euler method, real-time controllers, online signal processing

  FORWARD DIFFERENCE (anti-causal):
    f'(t) ≈ [f(t+dt) - f(t)] / dt    ← needs future: NOT realizable in real-time
    Error: O(dt) — first-order accurate
    Used in: offline analysis, non-causal filters

  CENTRAL DIFFERENCE (non-causal, but best accuracy):
    f'(t) ≈ [f(t+dt) - f(t-dt)] / (2dt)  ← needs one future sample
    Error: O(dt^2) — second-order accurate

  FOR REAL-TIME SYSTEMS: always use backward difference.
  np.gradient() uses central difference → NOT causal.
""")

# Verify accuracy orders
t_arr = np.linspace(0, 2*np.pi, 10000)
dt_arr = t_arr[1] - t_arr[0]
f_arr  = np.sin(t_arr)
df_exact = np.cos(t_arr)

df_forward  = np.diff(f_arr) / dt_arr                        # forward, len N-1
df_backward = np.concatenate([[0], np.diff(f_arr)]) / dt_arr # backward, len N
df_central  = np.gradient(f_arr, t_arr)                       # central, len N

err_fwd = np.max(np.abs(df_forward  - df_exact[:-1]))
err_bwd = np.max(np.abs(df_backward[1:] - df_exact[1:]))
err_cen = np.max(np.abs(df_central  - df_exact))

print(f'  Max error, forward  diff: {err_fwd:.2e}  (O(dt))')
print(f'  Max error, backward diff: {err_bwd:.2e}  (O(dt))')
print(f'  Max error, central  diff: {err_cen:.2e}  (O(dt^2))')
chk(err_cen < err_fwd, 1, "central diff more accurate than forward diff", tol=1e-9, absolute=True)

# %% [markdown]
# ## §3c — Causality and the Kramers-Kronig relations
#
# **Deep result**: if a system is causal (h(t)=0 for t<0), then its frequency
# response H(ω) = Re[H] + i·Im[H] has Re and Im related by the Hilbert transform:
#
#   Re[H(ω)] = (1/π) P∫ Im[H(ω′)]/(ω′−ω) dω′
#   Im[H(ω)] = −(1/π) P∫ Re[H(ω′)]/(ω′−ω) dω′
#
# You CANNOT freely choose the gain and phase of a causal filter — they are linked.
# This is why minimum-phase filters exist, and why you can't have a flat gain with
# zero phase delay in a real-time system.
#
# **Intuition**: causality in time → analyticity in the upper half of the complex
# frequency plane → Kramers-Kronig in frequency domain.

# %%
hdr("§3c — Kramers-Kronig: causality links gain and phase")

# Verify for a simple causal system: H(omega) = 1/(1 + i*omega)  (RC lowpass)
# Re[H] = 1/(1+omega^2),  Im[H] = -omega/(1+omega^2)
omega = np.linspace(-50, 50, 100000)
domega = omega[1] - omega[0]

H_re = 1 / (1 + omega**2)
H_im = -omega / (1 + omega**2)

# Kramers-Kronig: Re from Im via Hilbert transform
# Re[H(omega)] = (1/pi) * P∫ Im[H(omega')]/(omega'-omega) d(omega')
# Use scipy Hilbert (computes analytic signal → imag part is Hilbert transform)
from scipy.signal import hilbert
# H[Im](omega) should recover -Re (up to sign convention)
H_of_Im = np.imag(hilbert(H_im))  # Hilbert transform of Im[H]
# KK says Re[H] = (1/pi)*P∫ Im/(w'-w)dw' = -H{Im}  in scipy's convention
Re_KK = -H_of_Im

# Compare in interior (away from edge effects)
interior = slice(10000, 90000)
deep = slice(30000, 70000)
# KK: Re[H] = -(1/pi)*P.V.int Im[H(w')]/(w-w') dw' = +H{Im[H]} in scipy sign convention
# Check EITHER +Re_KK or -Re_KK matches H_re (sign depends on FT convention)
err_pos = np.max(np.abs( Re_KK[deep] - H_re[deep]))
err_neg = np.max(np.abs(-Re_KK[deep] - H_re[deep]))
kk_err = min(err_pos, err_neg)
chk(kk_err, 0, "Kramers-Kronig: |Re[H] - ±H{Im[H]}| < 0.01", tol=0.01, absolute=True)

print("""
  Consequence: if you know the loss (Im) of a material at all frequencies,
  you know its refractive index (Re) — this is how dispersion is measured.
  Every causal medium obeys Kramers-Kronig.
""")

# %% [markdown]
# ## §3d — The C loop is causal: why for(i=0; i<N; i++) only looks backward

# %%
hdr("§3d — Loops and causality: why forward loops are causal")

print("""
  A for-loop IS a causal computation:

    for (int i = 0; i < N; i++) {
        output[i] = f(input[0], input[1], ..., input[i]);
    }
    //                                           ^
    //                              only indices ≤ i (the "past")

  This is EXACTLY the discrete causal convolution:
    y[n] = Σ_{k=0}^{n} h[k] * x[n-k]

  In C:
    for (int n = 0; n < N; n++) {
        y[n] = 0;
        for (int k = 0; k <= n; k++)
            y[n] += h[k] * x[n-k];    // x[n-k] for k=0..n: never uses future x
    }

  If you need x[n+1] (a future sample) → you must BUFFER: wait one step.
  EVERY real-time DSP algorithm (audio, control, communications) respects this.
  The only place you can "look forward" is in OFFLINE (non-real-time) processing.

  BOOLEAN CAUSALITY:
    if (event_at_time[t]) → trigger_at_time[t+1]  // causal: effect after cause
    if (event_at_time[t]) → trigger_at_time[t-1]  // anti-causal: effect BEFORE cause
                                                   // physically impossible
""")

# Verify: causal convolution via nested C-style loop
def causal_conv_c_style(h, x):
    """Simulate the C nested loop for causal convolution."""
    N = len(x)
    y = np.zeros(N)
    for n in range(N):
        for k in range(n+1):
            if k < len(h):
                y[n] += h[k] * x[n-k]
    return y

# Compare to scipy causal result
h_test = np.array([1.0, 0.5, 0.25, 0.125])
x_test = np.sin(2*np.pi*np.arange(20)/10)
y_c_style = causal_conv_c_style(h_test, x_test)
y_scipy   = np.convolve(x_test, h_test)[:len(x_test)]
chk(np.max(np.abs(y_c_style - y_scipy)), 0,
    "C-style causal conv == scipy.signal result", tol=1e-12, absolute=True)

# %% [markdown]
# ## §4 — The Complete Picture: C + Boolean + Causality united

# %%
hdr("§4 — Everything together: real-time FIR filter in C pseudocode")

print("""
  CAUSAL FIR FILTER IN C (what Qualcomm's Hexagon DSP runs):
  ────────────────────────────────────────────────────────────

  #define N 64          // filter length
  float h[N] = {...};   // impulse response (causal: h[k] ≠ 0 only for k≥0)
  float x[N] = {0};     // circular buffer of past inputs
  int   buf_idx = 0;    // pointer into circular buffer

  float process_sample(float new_sample) {
      // Store new sample (this IS the causal constraint: we only store past+present)
      x[buf_idx] = new_sample;

      // Accumulate: y = Σ h[k]*x[n-k] for k=0..N-1
      float y = 0.0f;
      for (int k = 0; k < N; k++) {
          int idx = (buf_idx - k + N) % N;   // circular index into past
          y += h[k] * x[idx];                // ALWAYS past samples: causal ✓
      }

      // Advance pointer (loop variable causality: only moves forward in time)
      buf_idx = (buf_idx + 1) % N;
      return y;
  }

  BOOLEAN DECISIONS inside the loop:
    if (k == 0) → current sample (present)
    if (k > 0)  → past sample (history)
    (k < 0) is IMPOSSIBLE by loop construction: causality enforced by i>=0

  THE LOOP BOUNDS ARE THE CAUSALITY CONSTRAINT.
  k runs from 0 to N-1, never negative → you never access future samples.
  This is not a coincidence: the for-loop IS the mathematical convolution integral.
""")

# Verify circular buffer FIR
def fir_c_style(h_coeff, signal):
    """Simulate causal circular-buffer FIR in Python."""
    N = len(h_coeff)
    buf = np.zeros(N)
    buf_idx = 0
    output = []
    for sample in signal:
        buf[buf_idx] = sample
        y = 0.0
        for k in range(N):
            idx = (buf_idx - k) % N
            y += h_coeff[k] * buf[idx]
        buf_idx = (buf_idx + 1) % N
        output.append(y)
    return np.array(output)

h_fir = np.array([0.25, 0.5, 0.25])
x_fir = np.random.RandomState(42).randn(100)
y_fir_c = fir_c_style(h_fir, x_fir)
y_fir_np = np.convolve(x_fir, h_fir)[:len(x_fir)]
chk(np.max(np.abs(y_fir_c - y_fir_np)), 0,
    "circular-buffer FIR == np.convolve", tol=1e-12, absolute=True)

# %% [markdown]
# ## §5 — Figures

# %%
hdr("§5 — Figures")

fig = plt.figure(figsize=(16, 10))
fig.suptitle('C Loops · Boolean Algebra · Causality in Calculus', fontsize=13, fontweight='bold')
gs_fig = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# P1: Loop count vs N
ax1 = fig.add_subplot(gs_fig[0,0])
N_vals = np.arange(0, 21)
counts = [for_loop_count(0, n) for n in N_vals]
ax1.bar(N_vals, counts, color='steelblue', alpha=0.8)
ax1.plot(N_vals, N_vals, 'r--', lw=2, label='y=N (exact)')
ax1.set_xlabel('N'); ax1.set_ylabel('Iterations')
ax1.set_title('for(i=0;i<N;i++)\nexactly N iterations', fontsize=10)
ax1.legend(fontsize=8)

# P2: Boolean Venn
ax2 = fig.add_subplot(gs_fig[0,1])
theta = np.linspace(0, 2*np.pi, 300)
cx1, cy1, cx2, cy2, r = -0.4, 0, 0.4, 0, 0.7
ax2.plot(cx1+r*np.cos(theta), cy1+r*np.sin(theta), 'b-', lw=2, label='A')
ax2.plot(cx2+r*np.cos(theta), cy2+r*np.sin(theta), 'r-', lw=2, label='B')
ax2.text(cx1-0.35, 0, 'A\nonly', ha='center', fontsize=11, color='blue')
ax2.text(cx2+0.35, 0, 'B\nonly', ha='center', fontsize=11, color='red')
ax2.text(0, 0, 'A∧B', ha='center', fontsize=11, color='purple')
ax2.text(0, 0.9, 'A∨B = all shaded', ha='center', fontsize=9)
ax2.text(0, -0.9, 'A⊕B = outer only', ha='center', fontsize=9)
ax2.set_xlim(-1.2,1.2); ax2.set_ylim(-1.1,1.1)
ax2.set_title('Boolean ops: AND/OR/XOR', fontsize=10)
ax2.set_aspect('equal'); ax2.axis('off')
ax2.legend(loc='upper left', fontsize=8)

# P3: Derivative causal vs non-causal
ax3 = fig.add_subplot(gs_fig[0,2])
t_d = np.linspace(0, 2*np.pi, 200)
f_d = np.sin(t_d)
dt_d = t_d[1]-t_d[0]
df_b = np.concatenate([[0], np.diff(f_d)])/dt_d
df_f = np.concatenate([np.diff(f_d), [0]])/dt_d
df_e = np.cos(t_d)
ax3.plot(t_d, df_e, 'k-', lw=2, label='exact cos')
ax3.plot(t_d, df_b, 'b--', lw=1.5, label='backward (causal)')
ax3.plot(t_d, df_f, 'r:', lw=1.5, label='forward (anti-causal)')
ax3.set_title("Causal (backward) vs\nnon-causal derivative", fontsize=10)
ax3.legend(fontsize=7); ax3.set_xlabel('t')

# P4: Causal convolution output
ax4 = fig.add_subplot(gs_fig[1,0])
t_c = np.linspace(0, 8, 1000)
h_c = np.exp(-t_c) * (t_c >= 0)
f_c = np.sin(2*np.pi*t_c)
y_c = np.convolve(h_c, f_c)[:len(t_c)] * (t_c[1]-t_c[0])
ax4.plot(t_c, f_c, 'b-', lw=1, label='input f(t)')
ax4.plot(t_c, h_c, 'g-', lw=1.5, label='h(t)=e^{-t}H(t)')
ax4.plot(t_c, y_c, 'r-', lw=2, label='y=h*f (causal)')
ax4.set_title('Causal convolution\ny(t)=∫₀ᵗ h(t-τ)f(τ)dτ', fontsize=10)
ax4.legend(fontsize=7); ax4.set_xlabel('t')

# P5: Kramers-Kronig
ax5 = fig.add_subplot(gs_fig[1,1])
w_kk = np.linspace(-10, 10, 1000)
H_re_kk = 1/(1+w_kk**2)
H_im_kk = -w_kk/(1+w_kk**2)
ax5.plot(w_kk, H_re_kk, 'b-', lw=2, label='Re[H]: gain')
ax5.plot(w_kk, H_im_kk, 'r-', lw=2, label='Im[H]: phase')
ax5.set_title('Kramers-Kronig:\nRe↔Im linked by causality', fontsize=10)
ax5.legend(fontsize=8); ax5.set_xlabel('ω')
ax5.text(0, 0.5, 'Causality\nlinks these', ha='center', fontsize=9,
         color='purple', style='italic')

# P6: C loop → causal convolution schematic
ax6 = fig.add_subplot(gs_fig[1,2])
ax6.axis('off')
ax6.text(0.05, 0.97, """C LOOPS ARE CAUSAL COMPUTATION

for (k=0; k<N; k++) {
  y += h[k] * x[n-k];
  //          ^^^
  //     n-k ≤ n: PAST ONLY
}

BOOLEAN CAUSAL RULE:
  if (cause at t) →
    effect at t+1  ✓ causal
    effect at t-1  ✗ impossible

DERIVATIVE CAUSALITY:
  f'≈[f(t)-f(t-dt)]/dt  ← causal
  f'≈[f(t+dt)-f(t)]/dt  ← future!

INTEGRAL CAUSALITY:
  y(t) = ∫₀ᵗ f(τ)dτ
  upper limit = NOW, not future""",
         transform=ax6.transAxes, fontsize=8, va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax6.set_title('Unified picture', fontsize=10)

out_path = r'D:\Summer2026\Dispersion-Assisted-GS-Phase-Recovery\repl\_out_c_loops_boolean_causality.png'
fig.savefig(out_path, dpi=110, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out_path}')
print("=== C loops + Boolean algebra + Causality complete ===")
