"""
_repl_c_verilog_ml.py
C/Verilog 64-bit operators, symbolic linear regression (SymPy),
numeric linear regression (torch), FFT energy demo.
"""
import numpy as np
import sympy as sp
import torch
sp.init_printing(use_unicode=False, wrap_line=False)

# ============================================================
# 1. C & Verilog 64-bit operators side by side
# ============================================================
print("=== C / Verilog 64-bit operators ===")
print("""
Operator   C (int64_t)          Verilog (reg [63:0])    Python (int)
--------   -----------          --------------------    ------------
AND        a & b                a & b                   a & b
OR         a | b                a | b                   a | b
XOR        a ^ b                a ^ b                   a ^ b
NOT        ~a                   ~a                      ~a
NAND       ~(a & b)             ~(a & b)                ~(a & b)
NOR        ~(a | b)             ~(a | b)                ~(a | b)
XNOR       ~(a ^ b)             ~(a ^ b)                ~(a ^ b)
LSHIFT     a << n               a << n                  a << n
RSHIFT     a >> n               a >> n (logical)        a >> n
ARITH_R    a >> n (signed C)    $signed(a) >>> n        (arithmetic in C)
ADD        a + b                a + b                   a + b
MUL        a * b                a * b                   a * b
MOD        a % b                a % b                   a % b
TERNARY    a ? b : c            a ? b : c               b if a else c
CONCAT     --                   {a[3:0], b[3:0]}        (a<<4)|b  (Python)
""")

# Python demo of 64-bit ops
a = np.uint64(0xDEADBEEFCAFEBABE)
b = np.uint64(0x0F0F0F0F0F0F0F0F)
print(f"a        = {int(a):#018x}")
print(f"b        = {int(b):#018x}")
print(f"a & b    = {int(a & b):#018x}")
print(f"a | b    = {int(a | b):#018x}")
print(f"a ^ b    = {int(a ^ b):#018x}")
print(f"a >> 4   = {int(a >> np.uint64(4)):#018x}")
print(f"a << 4   = {int(a << np.uint64(4)) & 0xFFFFFFFFFFFFFFFF:#018x}")
print(f"popcount = {bin(int(a)).count('1')} ones")
print()

# ============================================================
# 2. Linear regression: SymPy symbolic
# ============================================================
print("=== Linear Regression: SymPy symbolic derivation ===")
m_s, b_s = sp.symbols('m b', real=True)
xi_s, yi_s, n_s = sp.symbols('x_i y_i n', real=True)

# Loss = sum (y_i - m*x_i - b)^2 / n
loss = (yi_s - m_s*xi_s - b_s)**2
print("Single sample loss:", sp.pretty(loss))

dL_dm = sp.diff(loss, m_s)
dL_db = sp.diff(loss, b_s)
print("dL/dm =", sp.pretty(dL_dm))
print("dL/db =", sp.pretty(dL_db))
print()
print("Set to zero -> normal equations:")
print("  m = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)")
print("  b = (sum(y) - m*sum(x)) / n")
print()

# ============================================================
# 3. Linear regression: 5x numeric loop (torch)
# ============================================================
print("=== Linear Regression: torch numeric (5 datasets) ===")

rng = np.random.default_rng(42)
results = []

for trial in range(5):
    N = 50
    x_true = float(rng.uniform(0.5, 3.0))
    b_true = float(rng.uniform(-2.0, 2.0))
    noise  = float(rng.uniform(0.1, 0.5))

    X = torch.linspace(0, 5, N).unsqueeze(1)
    Y = x_true * X + b_true + noise * torch.randn(N, 1)

    # single linear layer, no bias trick: [x,1] -> y
    X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)
    # normal equations: theta = (X^T X)^-1 X^T Y
    theta = torch.linalg.lstsq(X_aug, Y).solution
    m_hat = float(theta[0])
    b_hat = float(theta[1])

    # gradient descent version
    w = torch.zeros(2, 1, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.01)
    for _ in range(500):
        pred = X_aug @ w
        loss_t = ((pred - Y)**2).mean()
        opt.zero_grad()
        loss_t.backward()
        opt.step()
    m_gd = float(w[0])
    b_gd = float(w[1])

    results.append({
        'trial': trial,
        'm_true': round(x_true,3), 'b_true': round(b_true,3),
        'm_lstsq': round(m_hat,3), 'b_lstsq': round(b_hat,3),
        'm_gd':    round(m_gd,3),  'b_gd':    round(b_gd,3),
    })
    print(f"  trial {trial}: true=({x_true:.3f},{b_true:.3f})  "
          f"lstsq=({m_hat:.3f},{b_hat:.3f})  gd=({m_gd:.3f},{b_gd:.3f})")

print()

# ============================================================
# 4. FFT energy demo: discrete Parseval
# ============================================================
print("=== FFT energy: discrete Parseval theorem ===")
print("  sum|x[n]|^2 = (1/N) * sum|X[k]|^2")
print()

for label, sig in [
    ("pure tone 100 Hz",  lambda t: np.sin(2*np.pi*100*t)),
    ("two tones",         lambda t: np.sin(2*np.pi*100*t) + 0.5*np.sin(2*np.pi*300*t)),
    ("Gaussian pulse",    lambda t: np.exp(-((t-0.5)**2)/(2*0.02**2))),
    ("QPSK-like",         lambda t: np.exp(1j * np.repeat([0,np.pi/2,np.pi,3*np.pi/2], len(t)//4)[:len(t)])),
    ("white noise",       lambda t: np.random.default_rng(7).standard_normal(len(t))),
]:
    N  = 1024
    t  = np.linspace(0, 1, N, endpoint=False)
    x  = sig(t)
    X  = np.fft.fft(x)
    E_time = float(np.sum(np.abs(x)**2))
    E_freq = float(np.sum(np.abs(X)**2) / N)
    print(f"  {label:25s}  E_time={E_time:10.4f}  E_freq={E_freq:10.4f}  "
          f"ratio={E_time/E_freq:.6f}  {'OK' if abs(E_time-E_freq)<0.01 else 'FAIL'}")

print()
print("Ratio = 1.000000 confirms Parseval: FFT preserves energy exactly.")
print()

# ============================================================
# 5. Modern physics energy quantization loop
# ============================================================
print("=== Discrete energy levels: particle in a box ===")
print("  E_n = (n^2 * pi^2 * hbar^2) / (2 * m * L^2)")
print()

hbar_v = 1.055e-34
me_v   = 9.109e-31
eV_v   = 1.602e-19

for L_nm in [0.1, 0.5, 1.0, 5.0]:
    L = L_nm * 1e-9
    print(f"  L = {L_nm} nm:")
    for n in range(1, 6):
        E_J  = (n**2 * np.pi**2 * hbar_v**2) / (2 * me_v * L**2)
        E_eV = E_J / eV_v
        print(f"    n={n}  E={E_eV:8.3f} eV", end="")
        if n == 1:
            print("  <- ground state", end="")
        print()
    E21 = ((4-1) * np.pi**2 * hbar_v**2) / (2 * me_v * L**2) / eV_v
    lam = 1240 / E21   # nm, E(eV)*lambda(nm) = 1240
    print(f"    n=1->2 transition: {E21:.3f} eV  lambda={lam:.1f} nm")
    print()
