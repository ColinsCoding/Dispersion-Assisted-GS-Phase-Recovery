"""
repl/_repl_slicing_pipeline.py
[m:n] slicing, strides, views. Data pipeline pattern.
Reaction kinetics ODE. BLUF: result first, then derivation.
"""
import numpy as np
import sympy as sp
import pandas as pd
sp.init_printing(use_unicode=False, wrap_line=False)

print("=" * 60)
print("SLICING + PIPELINE + REACTION KINETICS")
print("=" * 60)
print()

# ============================================================
# RESULT FIRST (BLUF)
# ============================================================
print("=== RESULT FIRST ===")
print("""
Bottom Line Up Front:

  x[m:n]       -> elements m to n-1           (half-open interval)
  x[m:n:s]     -> every s-th element           (stride)
  x[..., m:n]  -> last axis slice              (numpy broadcast)
  x.view()     -> same memory, different shape (zero copy)
  x.copy()     -> new memory                  (safe to modify)

  Pipeline pattern: chain of pure functions
    output = f3(f2(f1(input)))
    your GS: E = P_C2(P_C1(P_C2(P_C1(... E_init ...))))

  Reaction kinetics:
    d[A]/dt = -k1*[A]          (first order: exponential decay)
    d[A]/dt = -k2*[A]*[B]      (second order: product rate)
    steady state: d[X]/dt = 0  -> [X]_ss = k_in / k_out
""")

# ============================================================
# 1. Slicing mechanics
# ============================================================
print("=== 1. [m:n] Slicing ===")

x = np.arange(16)
print(f"x        = {x}")
print(f"x[3:8]   = {x[3:8]}    (elements 3,4,5,6,7)")
print(f"x[::2]   = {x[::2]}  (even indices)")
print(f"x[1::2]  = {x[1::2]}  (odd indices)")
print(f"x[::-1]  = {x[::-1]}  (reversed)")
print(f"x[-4:]   = {x[-4:]}              (last 4)")
print(f"x[2:10:3]= {x[2:10:3]}                  (start=2, stop=10, step=3)")
print()

# 2D slicing
A = np.arange(24).reshape(4, 6)
print(f"A (4x6):\n{A}")
print(f"A[1:3, 2:5]  (rows 1-2, cols 2-4):\n{A[1:3, 2:5]}")
print(f"A[:, ::2]    (all rows, even cols):\n{A[:, ::2]}")
print(f"A[..., -1]   (last col): {A[..., -1]}")
print()

# ============================================================
# 2. Views vs copies
# ============================================================
print("=== 2. Views vs Copies (memory) ===")
x = np.arange(10, dtype=float)
view  = x[2:7]       # same memory
copy  = x[2:7].copy()

view[0] = 999
print(f"x after view[0]=999:  {x}")    # x changed
print(f"copy after view[0]=999: {copy}")  # copy unchanged
print()

# strides: the real mechanics
print("Strides (bytes between elements):")
x32 = np.arange(12, dtype=np.float64).reshape(3, 4)
print(f"  shape={x32.shape}  strides={x32.strides}  (bytes)")
print(f"  itemsize={x32.itemsize} bytes (float64)")
print(f"  strides = (4*8, 8) = (32, 8): row step=32B, col step=8B")

# stride trick: sliding window (no copy)
def sliding_window(x, w):
    n = len(x)
    shape   = (n - w + 1, w)
    strides = (x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

sig = np.array([1.,2.,3.,4.,5.,6.,7.,8.])
wins = sliding_window(sig, 3)
print(f"\nSliding window (w=3) on {sig}:")
print(wins)
print("No copy: each row is a view into the original array")
print()

# ============================================================
# 3. Pipeline pattern
# ============================================================
print("=== 3. Pipeline Pattern ===")
print("""
Pure function pipeline: each stage transforms data, no side effects.
Chain with compose() or simple function calls.
""")

# GS as a pipeline
def disperse(E, D):
    N = len(E); nu = np.fft.fftfreq(N)
    return np.fft.ifft(np.fft.fft(E) * np.exp(1j*np.pi*D*nu**2))

def proj_intensity(E, I):
    return np.sqrt(I) * np.exp(1j*np.angle(E))

def proj_unit(E):
    return np.exp(1j*np.angle(E))

def gs_step(E, I1, I2, D1, D2):
    """One GS iteration as a pure pipeline."""
    E = disperse(E, D1)           # stage 1: disperse
    E = proj_intensity(E, I1)     # stage 2: enforce I1
    E = disperse(E, -D1)          # stage 3: un-disperse
    E = disperse(E, D2)           # stage 4: disperse D2
    E = proj_intensity(E, I2)     # stage 5: enforce I2
    E = disperse(E, -D2)          # stage 6: un-disperse
    return E

# demo pipeline with synthetic signal
N = 256
rng = np.random.default_rng(0)
phi_true = rng.uniform(-np.pi, np.pi, N)
E_true = np.exp(1j*phi_true)
D1, D2 = 5000, -5000
I1 = np.abs(disperse(E_true, D1))**2
I2 = np.abs(disperse(E_true, D2))**2

E = np.sqrt(I1) * np.exp(1j*rng.uniform(0,2*np.pi,N))  # random init
errs = []
for _ in range(100):
    E = gs_step(E, I1, I2, D1, D2)
    errs.append(float(np.mean(np.abs(np.abs(disperse(E,D1))**2 - I1))))

# slice error history
print(f"GS pipeline errors [m:n] slices:")
print(f"  first 5:  {[round(e,4) for e in errs[:5]]}")
print(f"  [10:15]:  {[round(e,4) for e in errs[10:15]]}")
print(f"  last 5:   {[round(e,4) for e in errs[-5:]]}")
print(f"  [::20]:   {[round(e,4) for e in errs[::20]]}  (every 20th)")
print()

# ============================================================
# 4. Reaction kinetics ODE
# ============================================================
print("=== 4. Reaction Kinetics ===")
print("""
First order:    A -> B      d[A]/dt = -k*[A]
                            [A](t) = [A]0 * exp(-k*t)   (same as RC circuit)

Second order:   A + B -> C  d[A]/dt = -k*[A]*[B]
                            harder: need numerical ODE solver

Michaelis-Menten (enzyme kinetics / personalized medicine):
  E + S <-> ES -> E + P
  v = Vmax*[S] / (Km + [S])
  at [S] << Km:  v ~ (Vmax/Km)*[S]   (first order)
  at [S] >> Km:  v ~ Vmax             (zero order, saturated)
""")

# SymPy: first order
t_s, k_s, A0 = sp.symbols('t k A0', positive=True)
A_func = sp.Function('A')
ode = sp.Eq(A_func(t_s).diff(t_s), -k_s*A_func(t_s))
sol = sp.dsolve(ode, A_func(t_s))
C1_val = sp.solve(sol.rhs.subs(t_s, 0) - A0, 'C1')[0]
sol_iv = sol.subs('C1', C1_val)
print("First order A->B:"); sp.pprint(sol_iv)
print(f"  Half-life: t_1/2 = ln(2)/k")
print()

# Michaelis-Menten
S = np.logspace(-3, 2, 200)
Vmax, Km = 1.0, 1.0
v = Vmax*S / (Km + S)

print("Michaelis-Menten v([S]) sampled:")
for s_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
    v_val = Vmax*s_val/(Km+s_val)
    regime = 'first-order' if s_val < 0.1*Km else ('saturated' if s_val > 10*Km else 'transition')
    print(f"  [S]={s_val:7.3f}  v={v_val:.4f}  ({regime})")
print()

# ============================================================
# 5. Personalized medicine connection
# ============================================================
print("=== 5. Personalized Medicine ===")
print("""
Pharmacokinetics (PK): same math as GS pipeline
  dose -> absorption -> distribution -> metabolism -> excretion
  each stage: convolution with an impulse response h(t)

  C(t) = D * F * ka/(ka-ke) * (exp(-ke*t) - exp(-ka*t))
  D  = dose, F = bioavailability, ka = absorption, ke = elimination

GS connection:
  - patient data [m:n] slices: time series of drug concentration
  - FNO on PK time series: same architecture as your GS FNO
  - D parameter in GS <-> clearance rate in PK
  - diversity |D1-D2| <-> difference in patient metabolizer types
    (CYP2D6 poor vs ultra-rapid: 10x difference in clearance)

Data pipeline in clinical ML:
  raw EHR -> [m:n] select features -> normalize -> model -> dose recommendation
  same pattern as:
  I1.npy  -> [::2] downsample     -> normalize -> FNO  -> phase
""")

# simple 2-compartment PK model
print("Two-compartment PK (IV bolus):")
t_pk = np.linspace(0, 24, 200)   # hours
D_dose = 100.0   # mg
ka = 1.5   # /hr (absorption)
ke = 0.3   # /hr (elimination)
F  = 0.8   # bioavailability

C = D_dose * F * ka/(ka-ke) * (np.exp(-ke*t_pk) - np.exp(-ka*t_pk))
C_max_idx = np.argmax(C)
print(f"  Dose={D_dose}mg  ka={ka}/hr  ke={ke}/hr  F={F}")
print(f"  C_max = {C.max():.2f} mg/L at t = {t_pk[C_max_idx]:.2f} h")
print(f"  AUC   = {np.trapezoid(C, t_pk):.2f} mg*h/L")
print(f"  t_1/2 = {np.log(2)/ke:.2f} h  (elimination half-life)")

# slicing the PK curve
print(f"\n  C(t) [m:n] slices:")
print(f"  t[0:5]   = {np.round(t_pk[:5],1)}  C = {np.round(C[:5],2)}")
print(f"  t[::40]  = {np.round(t_pk[::40],1)}  C = {np.round(C[::40],2)}")
