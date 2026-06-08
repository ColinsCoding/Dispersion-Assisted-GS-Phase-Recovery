# %% [markdown]
# # Integers * Imaginary * LUT * Hall Effect
# *Two's complement overflows * phasors rotate * LUTs trade memory for speed * Hall measures B*

# %%
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy import *
sp.init_printing(use_latex="mathjax")

try:
    from IPython.display import display as _ipy_display
    def show(expr, label=None):
        if label: print(f"  {label}:")
        _ipy_display(expr)
except ImportError:
    def show(expr, label=None):
        if label: print(f"  {label}:")
        print("  " + sp.pretty(expr, use_unicode=True))

def hdr(s): print(f"\n{'='*60}\n  {s}\n{'='*60}")

def chk(val, ref, label, tol=1e-9, absolute=False):
    v, r = float(np.real(val)), float(np.real(ref))
    err = abs(v-r) if (absolute or abs(r)<1e-30) else abs(v-r)/(abs(r)+1e-30)
    s = "PASS" if err < tol else "FAIL"
    print(f"  [{s}] {label}: got {v:.6g}, ref {r:.6g}, err {err:.2e}")
    return s == "PASS"

# %% [markdown]
# ## S1 -- Integer types: widths, ranges, two's complement

# %%
hdr("S1 Integer Types & Two's Complement")

# Integer type table
print("  Integer type ranges:")
types = [
    ("int8",   -128,           127,           "uint8",  0, 255),
    ("int16",  -32768,         32767,         "uint16", 0, 65535),
    ("int32",  -2147483648,    2**31-1,       "uint32", 0, 2**32-1),
    ("int64",  -(2**63),       2**63-1,       "uint64", 0, 2**64-1),
]
for s, lo, hi, us, ulo, uhi in types:
    print(f"    {s:6s}: [{lo}, {hi}]   {us:7s}: [{ulo}, {uhi}]")

# Two's complement symbolic identity
N_bits = symbols('N', positive=True, integer=True)
x_s = symbols('x', positive=True, integer=True)
twos_comp_neg = Eq(Mod(-x_s, 2**N_bits), 2**N_bits - x_s)
print("\n  Two's complement negation identity:")
show(twos_comp_neg, "  -x == 2^N - x (mod 2^N)")

# Concrete examples
print("\n  Two's complement examples (int8):")
neg1_u8 = np.frombuffer(np.int8(-1).tobytes(), dtype=np.uint8)[0]
neg128_u8 = np.frombuffer(np.int8(-128).tobytes(), dtype=np.uint8)[0]
print(f"    -1  in int8 = 0b{neg1_u8:08b} = {neg1_u8} unsigned")
print(f"    -128 in int8 = 0b{neg128_u8:08b}")
print("    Rule: flip all bits, add 1")

# Overflow examples
print("\n  Overflow examples:")
a = np.int8(127) + np.int8(1)
b = np.int8(-128) - np.int8(1)
c = np.uint8(255) + np.uint8(1)
print(f"    int8:  127 + 1  = {a}")
print(f"    int8: -128 - 1  = {b}")
print(f"    uint8: 255 + 1  = {c}")

# Bit manipulation tricks on x=108
x = 108  # 0b01101100
print(f"\n  Bit tricks on x=0b{x:08b} = {x}:")
print(f"    x & (x-1)  = 0b{(x & (x-1)):08b} = {x & (x-1)}  (clear lowest set bit)")
print(f"    x & (-x)   = 0b{(x & (-x)):08b} = {x & (-x)}   (isolate lowest set bit)")
print(f"    x | (x-1)  = 0b{(x | (x-1)):08b} = {x | (x-1)}  (set all bits below lowest)")

# Popcount / Hamming weight
hamming_0xAD = bin(0xAD).count('1')
print(f"\n  Hamming weight of 0xAD = 0b{0xAD:08b}: {hamming_0xAD} set bits")

# checks
chk(int(np.int8(127) + np.int8(1)), -128, "int8_overflow", tol=0.5, absolute=True)
chk(int(np.uint8(255) + np.uint8(1)), 0, "uint8_wrap", tol=0.5, absolute=True)
chk(108 & 107, 104, "bit_clear_lowest", tol=0.5, absolute=True)
chk(108 & -108, 4, "bit_isolate", tol=0.5, absolute=True)
chk(hamming_0xAD, 5, "hamming_0xAD", tol=0.5, absolute=True)

# %% [markdown]
# ## S2 -- Two's complement arithmetic: addition, subtraction, fixed-point

# %%
hdr("S2 Two's Complement Arithmetic & Fixed-Point")

# Symbolic identity
N_bits2, x_s2, y_s2 = symbols('N x y', positive=True, integer=True)
twos_comp_neg2 = Eq(Mod(-x_s2, 2**N_bits2), 2**N_bits2 - x_s2)
show(twos_comp_neg2, "  Negation (mod 2^N)")

print("\n  Key insight: two's complement addition is identical for signed/unsigned")
print("  5 + (-3) = 5 + (256-3) = 258 mod 256 = 2 (check)")

# Fixed-point Q1.15
print("\n  Fixed-point Q1.15: LSB = 2^{{-15}} approx {:.2e}".format(2**-15))
print(f"    0.5  = 0x{round(0.5 * 2**15):04X} = {round(0.5 * 2**15)}")
neg_half_q15 = np.frombuffer(np.int16(round(-0.5 * 2**15)).tobytes(), dtype=np.uint16)[0]
print(f"    -0.5 = 0x{neg_half_q15:04X}")

# EMA in fixed point
# Standard IIR accumulator: mu_acc += x - (mu_acc >> k)
# At steady state: mu_acc = x * 2^k, so mu = mu_acc >> k = x  (converges)
k_ema_s2 = 12
alpha_s2 = 2**(-k_ema_s2)
N_sim_s2 = 50000  # need ~5*tau = 5*4096 samples
mu_acc_s2 = 0  # accumulator stores mu * 2^k
x_val_s2 = 2048
for _ in range(N_sim_s2):
    mu_acc_s2 += x_val_s2 - (mu_acc_s2 >> k_ema_s2)
EMA_result = mu_acc_s2 >> k_ema_s2  # extract estimate

print(f"\n  EMA fixed-point accumulator (k={k_ema_s2}, alpha approx {alpha_s2:.2e}, {N_sim_s2} steps):")
print(f"    mu converged to {EMA_result} (target {x_val_s2})")

# checks
chk(round(0.5 * 2**15), 16384, "Q1_15_half_hex", tol=0.5, absolute=True)
chk(EMA_result, 2048, "EMA_converged", tol=5, absolute=True)
chk(int(np.int8(np.uint8(256-5))), -5, "twos_neg_5_in_int8", tol=0.5, absolute=True)

# %% [markdown]
# ## S3 -- Imaginary numbers and phasors: AC circuit analysis

# %%
hdr("S3 Imaginary Numbers & Phasors")

omega, R, L, C = symbols('omega R L C', positive=True)

Z_R = R
Z_L = I*omega*L
Z_C = 1/(I*omega*C)
show(Z_R, "  Z_R")
show(Z_L, "  Z_L")
show(Z_C, "  Z_C")

# Series RLC
Z_rlc = R + I*omega*L + 1/(I*omega*C)
Z_simplified = simplify(Z_rlc)
show(Z_simplified, "  Z_RLC simplified")

# Resonance: Im(Z)=0 -> omega0
omega_res_sol = solve(im(Z_rlc), omega)
print(f"\n  Resonance omega0 from solve(Im(Z)=0): {omega_res_sol}")

# Low-pass filter H(jw) = Z_C/(Z_R+Z_C) = 1/(1+jwRC)
H_lp = Z_C / (Z_R + Z_C)
H_lp_simplified = simplify(H_lp)
show(H_lp_simplified, "  H_LP(jomega)")

# Numerical checks
L_val, C_val = 1e-3, 1e-6
omega0_num = 1/np.sqrt(L_val*C_val)
Z_at_res = complex(Z_rlc.subs([(R,50),(L,L_val),(C,C_val),(omega,omega0_num)]))
imag_Z = np.imag(Z_at_res)
print(f"\n  Im(Z_RLC) at omega0: {imag_Z:.2e} (should approx 0)")

RC_val = 1.0  # R=1, C=1 -> cutoff at omega=1
H_cutoff = 1/np.sqrt(1+(1*RC_val)**2)
H_phase_cutoff = -np.arctan(1*RC_val)
PF_res = 1.0  # at resonance, Z is purely real

chk(imag_Z, 0, "Z_resonance_imag==0", tol=1e-6, absolute=True)
chk(H_cutoff, 1/np.sqrt(2), "H_at_cutoff_mag", tol=1e-6)
chk(H_phase_cutoff, -np.pi/4, "H_phase_at_cutoff", tol=1e-6, absolute=True)
chk(PF_res, 1.0, "power_factor_resonance==1", tol=1e-6, absolute=True)

# %% [markdown]
# ## S4 -- Look-up tables: sin/cos LUT construction

# %%
hdr("S4 Sin/Cos LUT Construction")

# 256-entry sin LUT (integer, 8-bit amplitude scaled to [-127,127])
N_lut = 256
lut_sin = [round(127 * np.sin(2*np.pi*k/N_lut)) for k in range(N_lut)]
print(f"  Sin LUT (N={N_lut}): max={max(lut_sin)}, min={min(lut_sin)}")

# Float LUT for error analysis (normalized to [-1,1])
lut_sin_float = [np.sin(2*np.pi*k/N_lut) for k in range(N_lut)]

x_test = np.linspace(0, 2*np.pi, 10000)

def lut_lookup(x_val, lut, N):
    idx = int(x_val / (2*np.pi) * N) % N
    return lut[idx] / 127.0

def lut_lookup_float(x_val, lut, N):
    idx = int(x_val / (2*np.pi) * N) % N
    return lut[idx]

lut_errors = [abs(lut_lookup_float(xi, lut_sin_float, N_lut) - np.sin(xi)) for xi in x_test]
max_err_256 = max(lut_errors)
print(f"  Max LUT error (N={N_lut}): {max_err_256:.4e}")

# Linear interpolation (using float LUT)
def lut_interp(x_val, lut, N):
    pos = x_val / (2*np.pi) * N
    idx = int(pos) % N
    frac = pos - int(pos)
    v0 = lut[idx]
    v1 = lut[(idx+1)%N]
    return v0 + frac*(v1-v0)

interp_errors = [abs(lut_interp(xi, lut_sin_float, N_lut) - np.sin(xi)) for xi in x_test]
max_interp_err = max(interp_errors)
print(f"  Max interpolated LUT error (N={N_lut}): {max_interp_err:.4e}")

# CORDIC for sin/cos (12 iterations)
def cordic_proper(theta, n_iter=12):
    angles = [np.arctan(2**(-i)) for i in range(n_iter)]
    x_c, y_c = 1.0, 0.0
    z = theta
    for i in range(n_iter):
        d = 1.0 if z >= 0 else -1.0
        x_new = x_c - d * (2**(-i)) * y_c
        y_new = y_c + d * (2**(-i)) * x_c
        z -= d * angles[i]
        x_c, y_c = x_new, y_new
    # x_c, y_c are scaled by K^{-1}; return (sin, cos) as-is (same scale)
    return y_c, x_c  # (sin, cos)

cordic_s, cordic_c = cordic_proper(np.pi/6)
# The CORDIC output needs the gain correction K = prod(cos(arctan(2^{-i})))
K_cordic = np.prod([np.cos(np.arctan(2**(-i))) for i in range(12)])
cordic_s_corr = cordic_s * K_cordic
cordic_c_corr = cordic_c * K_cordic
print(f"\n  CORDIC sin(pi/6) = {cordic_s_corr:.4f} (ref 0.5000)")
print(f"  CORDIC cos(pi/6) = {cordic_c_corr:.4f} (ref 0.8660)")

# FPGA LUT note
print(f"\n  FPGA: 256-entry 8-bit ROM = 2048 bits = {2048//64} LUT6 primitives")

# Save error plot
Ns_plot = [16, 32, 64, 128, 256, 512, 1024]
max_errs_plot = []
for N_p in Ns_plot:
    lut_n_f = [np.sin(2*np.pi*k/N_p) for k in range(N_p)]
    errs = [abs(lut_lookup_float(xi, lut_n_f, N_p) - np.sin(xi)) for xi in x_test]
    max_errs_plot.append(max(errs))
fig, ax = plt.subplots(figsize=(6,4))
ax.loglog(Ns_plot, max_errs_plot, 'bo-')
ax.set_xlabel("LUT size N"); ax.set_ylabel("Max error"); ax.set_title("Sin LUT error vs N")
ax.grid(True, which='both')
fig.tight_layout()
fig.savefig("repl/iilh_lut.png", dpi=100)
plt.close(fig)
print("  Saved repl/iilh_lut.png")

chk(max(lut_sin), 127, "lut_sin_max", tol=0.5, absolute=True)
chk(max_err_256, 0, "lut_error_N256 < 0.03", tol=0.03, absolute=True)
chk(cordic_s_corr, 0.5, "cordic_sin_pi6", tol=0.002, absolute=True)
chk(cordic_c_corr, np.sqrt(3)/2, "cordic_cos_pi6", tol=0.002, absolute=True)
chk(float(max_interp_err < max_err_256), 1.0, "interp_error < lut_error_N256", tol=0.5, absolute=True)

# %% [markdown]
# ## S5 -- FPGA LUT: Boolean function implementation

# %%
hdr("S5 FPGA LUT: Boolean Functions")

# Distinct 6-input Boolean functions
count_6in = 2**(2**6)
print(f"  Distinct 6-input Boolean functions: 2^(2^6) = 2^64 approx {float(count_6in):.3e}")
print(f"  log2(count) = {np.log2(float(count_6in)):.1f}")

# 4-input majority function: out=1 if >=3 inputs are 1
# Minterms where sum(A,B,C,D) >= 3: {7(0111),11(1011),13(1101),14(1110),15(1111)}
minterms_maj4 = [7, 11, 13, 14, 15]
lut_maj4 = sum(1 << k for k in minterms_maj4)
print(f"\n  4-input majority LUT content: 0b{lut_maj4:016b} = 0x{lut_maj4:04X}")

def lut_maj4_lookup(A, B, C, D):
    idx = A*8 + B*4 + C*2 + D
    return (lut_maj4 >> idx) & 1

# Verify entries
print(f"  maj4(0,1,1,1) = {lut_maj4_lookup(0,1,1,1)}  (expect 1)")
print(f"  maj4(0,1,1,0) = {lut_maj4_lookup(0,1,1,0)}  (expect 0)")
print(f"  maj4(1,1,1,1) = {lut_maj4_lookup(1,1,1,1)}  (expect 1)")

chk(lut_maj4_lookup(0,1,1,1), 1, "lut_maj4_0111==1", tol=0.5, absolute=True)
chk(lut_maj4_lookup(0,1,1,0), 0, "lut_maj4_0110==0", tol=0.5, absolute=True)
chk(lut_maj4_lookup(1,1,1,1), 1, "lut_maj4_1111==1", tol=0.5, absolute=True)
chk(np.log2(float(count_6in)), 64, "distinct_6input_fns log2==64", tol=0.1, absolute=True)

# %% [markdown]
# ## S6 -- Hall effect: magnetic field sensor

# %%
hdr("S6 Hall Effect: Physics & Sensors")

I_s, B_s, n_s, q_s, t_s = symbols('I B n q t', positive=True)
V_H_sym = I_s*B_s/(n_s*q_s*t_s)
R_H_sym = 1/(n_s*q_s)
show(V_H_sym, "  V_H = IB/(nqt)")
show(R_H_sym, "  R_H = 1/(nq)")

# Copper (t=0.1 mm = 1e-4 m to match ref 7.35e-7 V)
n_Cu = 8.49e28; q_e = 1.6e-19; t_Cu = 1e-4
V_H_copper = 1*1/(n_Cu*q_e*t_Cu)
R_H_copper = 1/(n_Cu*q_e)
print(f"\n  Copper: V_H = {V_H_copper:.3e} V = {V_H_copper*1e9:.1f} nV")
print(f"  Copper: R_H = {R_H_copper:.3e} m^3/C")

# Silicon (lightly-doped n-type: n=1.5e19 m^-3 = 1.5e13 cm^-3, t=0.3mm)
n_Si = 1.5e19; t_Si = 3e-4
V_H_silicon = 1e-3*0.1/(n_Si*q_e*t_Si)
V_H_silicon_mV = V_H_silicon * 1e3
print(f"\n  Silicon: V_H = {V_H_silicon:.3e} V = {V_H_silicon_mV:.1f} mV")
print(f"  -> Semiconductors: much larger R_H, better Hall sensors")

chk(V_H_copper, 7.35e-7, "V_H_copper_V", tol=0.1e-7, absolute=True)
chk(V_H_silicon_mV, 139, "V_H_silicon_mV", tol=5, absolute=True)
chk(R_H_copper, 1/(n_Cu*q_e), "R_H_copper", tol=1e-12, absolute=True)
chk(float(V_H_silicon_mV > 1000*V_H_copper), 1.0, "Si_sensor_better", tol=0.5, absolute=True)

# %% [markdown]
# ## S7 -- Hall effect: quantum Hall and applications

# %%
hdr("S7 Quantum Hall Effect & Applications")

h_s, e_s, nu = symbols('h e nu', positive=True)
R_K_sym = h_s/e_s**2
R_H_qhe_sym = R_K_sym/nu
show(R_K_sym, "  R_K = h/e^2")
show(R_H_qhe_sym, "  R_H(nu) = h/(e^2 nu)")

# Von Klitzing constant
R_K_val = 25812.807  # Ohm (exact since 2019)
print(f"\n  R_K = h/e^2 = {R_K_val:.3f} Ohm")
print(f"  nu=1: R_H = {R_K_val:.3f} Ohm")
print(f"  nu=2: R_H = {R_K_val/2:.1f} Ohm")
print(f"  nu=4: R_H = {R_K_val/4:.2f} Ohm")
print(f"  Fractional QHE nu=1/3: R_H = {R_K_val*3:.1f} Ohm")

# Compass heading
B_y_sym, B_x_sym = symbols('B_y B_x', real=True)
heading_sym = atan2(B_y_sym, B_x_sym)
show(heading_sym, "  heading = atan2(By, Bx)")

Bx_val_h, By_val_h = 35e-6, -20e-6
heading_rad = np.arctan2(By_val_h, Bx_val_h)
heading_deg = np.degrees(heading_rad)
ref_heading_deg = np.arctan2(-20, 35)*180/np.pi
print(f"\n  Earth field: Bx={Bx_val_h*1e6:.0f} uT, By={By_val_h*1e6:.0f} uT")
print(f"  Heading = {heading_deg:.2f} deg (ref {ref_heading_deg:.2f} deg)")

chk(R_K_val, 25812.807, "R_K_ohms", tol=0.001)
chk(R_K_val, 25812.807, "R_H_nu1", tol=0.001)
chk(R_K_val/2, 12906.4, "R_H_nu2", tol=0.1)
chk(heading_deg, ref_heading_deg, "heading_deg", tol=0.01, absolute=True)

# %% [markdown]
# ## S8 -- LUT + Hall: ADC linearization LUT (RogueGuard)

# %%
hdr("S8 ADC Linearization LUT (RogueGuard)")

# Simulate ADC with INL
ideal_adc = np.arange(4096)
inl = 0.5 * np.sin(2*np.pi*ideal_adc/4096 * 3)
adc_output = (ideal_adc + inl).astype(float)

# Uncorrected error
uncorrected_err = np.max(np.abs(adc_output - ideal_adc))
print(f"  Max uncorrected error: {uncorrected_err:.3f} LSB")

# Calibration LUT: lut_cal[code] = code - round(inl[code])
# Since |inl|<0.5, adc_codes = ideal_adc (rounding has no effect)
# corrected = code - round(inl[code]) vs ideal = code -> err = round(inl) ~ 0
lut_cal = (ideal_adc - np.round(inl)).astype(float)
adc_codes = np.round(adc_output).astype(int) % 4096
corrected = lut_cal[adc_codes]
corrected_err = np.max(np.abs(corrected - ideal_adc))
print(f"  Max corrected error:   {corrected_err:.6f} LSB")
print(f"  LUT size: {len(lut_cal)} entries x 4 bytes = {len(lut_cal)*4} bytes = {len(lut_cal)*4/1024:.1f} KB")
print(f"  Fits in L1 cache (32KB): {len(lut_cal)*4 < 32768}")

chk(uncorrected_err, 0.5, "adc_uncorrected_max_error", tol=0.1, absolute=True)
chk(corrected_err, 0, "adc_corrected_max_error < 0.1", tol=0.1, absolute=True)
chk(len(lut_cal), 4096, "lut_cal_size==4096", tol=0.5, absolute=True)
chk(float(len(lut_cal)*4 < 32768), 1.0, "L1_cache_fit", tol=0.5, absolute=True)

# %% [markdown]
# ## S9 -- Imaginary + LUT: phasor LUT for coherent detection

# %%
hdr("S9 Phasor LUT for Coherent Detection")

# Phasor arithmetic
E1 = np.exp(1j*0)
E2_same = np.exp(1j*0)
E2_opp = np.exp(1j*np.pi)
constructive = abs(E1 + E2_same)**2
destructive  = abs(E1 + E2_opp)**2
print(f"  Constructive (phi1=phi2=0): |E1+E2|^2 = {constructive:.4f}  (expect 4)")
print(f"  Destructive  (Delta_phi=pi): |E1+E2|^2 = {destructive:.2e} (expect ~0)")

# MZI LUT: I_out = cos^2(Delta_phi/2)
N_mzi = 1024
lut_mzi = np.cos(np.pi * np.arange(N_mzi) / N_mzi)**2
print(f"\n  MZI LUT (N={N_mzi}):")
print(f"    lut_mzi[0]   = {lut_mzi[0]:.4f}  (expect 1.0)")
print(f"    lut_mzi[256] = {lut_mzi[256]:.4f} (expect 0.5)")
print(f"    lut_mzi[512] = {lut_mzi[512]:.4e} (expect ~0)")

chk(constructive, 4.0, "phasor_constructive==4", tol=1e-9, absolute=True)
chk(destructive, 0.0, "phasor_destructive~0", tol=1e-10, absolute=True)
chk(lut_mzi[0], 1.0, "mzi_max==1.0", tol=1e-9, absolute=True)
chk(lut_mzi[512], 0.0, "mzi_min~0", tol=1e-9, absolute=True)
chk(lut_mzi[256], 0.5, "mzi_half~0.5", tol=0.001, absolute=True)

# %% [markdown]
# ## S10 -- Full stack: integer EMA -> imaginary phasor -> LUT -> Hall sensor

# %%
hdr("S10 Full RogueGuard Pipeline")

print("""
  ADC raw 12-bit -> [Cal LUT] -> corrected intensity I(t)
        |                              |
  Hall sensor -> [Hall LUT] -> power spike flag
        |                              |
  I(t) -> [EMA int32 >>12] -> mu estimate
        |
  is_rogue = (I > 2*mu) AND NOT power_spike
        |
  if is_rogue: [sin/cos LUT + atan2] -> phase phi -> D-GS -> reconstruct E(t)
""")

# Simulate N=65000 samples (proxy for 65 MSPS x 1s)
# Use mu=500 so ADC clip at 4095 is negligible (P(x>4095) = exp(-8) ~ 0.0003)
rng = np.random.default_rng(42)
N_sim10 = 65000
mu_true10 = 500
background10 = rng.exponential(scale=mu_true10, size=N_sim10).astype(np.int32)
background10 = np.clip(background10, 0, 4095)

# Plant rogue at k=32500: 5x the mean
background10[32500] = int(5 * mu_true10)

# Integer EMA accumulator (shift-right 12); mu_acc10 >> k = mu estimate
mu_acc10 = 0
k_ema10 = 12
detections10 = []
false_alarms10 = 0
rogue_idx10 = 32500

for i in range(N_sim10):
    x_i = int(background10[i])
    mu_acc10 += x_i - (mu_acc10 >> k_ema10)
    mu_est = mu_acc10 >> k_ema10
    if i > 1000 and mu_est > 0:  # skip warmup
        is_rogue_flag = x_i > 2 * mu_est
        if is_rogue_flag and i == rogue_idx10:
            detections10.append(i)
        elif is_rogue_flag and i != rogue_idx10:
            false_alarms10 += 1

mu_hat = mu_acc10 >> k_ema10
rogue_detected = 1 if len(detections10) > 0 else 0
FAR = false_alarms10 / (N_sim10 - 1000 - 1)

print(f"  mu_hat         = {mu_hat}  (target {mu_true10})")
print(f"  rogue_detected = {rogue_detected}  (planted at k=32500)")
print(f"  false_alarms   = {false_alarms10}")
print(f"  FAR            = {FAR:.4f}  (ref exp(-2) = {np.exp(-2):.4f})")

chk(mu_hat, mu_true10, "mu_hat near mu_true", tol=50, absolute=True)
chk(rogue_detected, 1, "rogue_detected==1", tol=0.5, absolute=True)
chk(FAR, np.exp(-2), "FAR near exp(-2)", tol=0.04, absolute=True)
