import numpy as np
import torch
import time
from functools import lru_cache

# ============================================================
# 1. Recursion
# ============================================================
print("=== Recursion: canonical forms ===")

def factorial(n):
    if n <= 1: return 1
    return n * factorial(n-1)

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)

print("factorial(12) =", factorial(12))
print("fib(50)       =", fib(50), "  (memoized, instant)")

# GS as recursive fixed-point -- tail recursion
def gs_recursive(E, I1, I2, D1, D2, n):
    if n == 0:
        return E
    N  = len(E)
    nu = np.fft.fftfreq(N)
    H1 = np.exp(1j * np.pi * D1 * nu**2)
    H2 = np.exp(1j * np.pi * D2 * nu**2)
    # forward pass D1
    E1 = np.fft.ifft(np.fft.fft(E) * H1)
    E1 = np.sqrt(np.maximum(I1, 0.0)) * np.exp(1j * np.angle(E1))
    En = np.fft.ifft(np.fft.fft(E1) * np.conj(H1))
    # forward pass D2
    E2 = np.fft.ifft(np.fft.fft(En) * H2)
    E2 = np.sqrt(np.maximum(I2, 0.0)) * np.exp(1j * np.angle(E2))
    En = np.fft.ifft(np.fft.fft(E2) * np.conj(H2))
    return gs_recursive(En, I1, I2, D1, D2, n - 1)

rng = np.random.default_rng(0)
N   = 64
E0  = rng.standard_normal(N) + 1j * rng.standard_normal(N)
I1  = np.abs(E0) ** 2
I2  = np.abs(E0) ** 2
E_r = gs_recursive(E0.copy(), I1, I2, -5000.0, -5750.0, 10)
print("GS recursive 10 iters OK, |E|^2 mean = {:.4f}".format(
      float(np.mean(np.abs(E_r)**2))))

# ============================================================
# 2. Operator overloading
# ============================================================
print()
print("=== Operator overloading: Q(value, error) ===")

class Q:
    def __init__(self, val, err=0.0):
        self.val = float(val)
        self.err = float(abs(err))
    def __add__(self, o):
        o = o if isinstance(o, Q) else Q(o)
        return Q(self.val + o.val, np.sqrt(self.err**2 + o.err**2))
    def __sub__(self, o):
        o = o if isinstance(o, Q) else Q(o)
        return Q(self.val - o.val, np.sqrt(self.err**2 + o.err**2))
    def __mul__(self, o):
        o = o if isinstance(o, Q) else Q(o)
        v = self.val * o.val
        e = abs(v) * np.sqrt((self.err/self.val)**2 + (o.err/o.val)**2) if v else 0
        return Q(v, e)
    def __truediv__(self, o):
        o = o if isinstance(o, Q) else Q(o)
        v = self.val / o.val
        e = abs(v) * np.sqrt((self.err/self.val)**2 + (o.err/o.val)**2) if v else 0
        return Q(v, e)
    def __pow__(self, n):
        v = self.val ** n
        e = abs(n) * abs(v) * self.err / abs(self.val) if self.val else 0
        return Q(v, e)
    # | parallel resistors
    def __or__(self, o):
        v = 1.0 / (1.0/self.val + 1.0/o.val)
        e = v**2 * np.sqrt((self.err/self.val**2)**2 + (o.err/o.val**2)**2)
        return Q(v, e)
    # & series
    def __and__(self, o):
        return self.__add__(o)
    def __repr__(self):
        return "{:.4g} +/- {:.2g}".format(self.val, self.err)

R1 = Q(100, 2)
R2 = Q(150, 3)
V  = Q(5.0, 0.05)
print("R1 =", R1, "ohm")
print("R2 =", R2, "ohm")
print("R1 & R2  (series)   =", R1 & R2)
print("R1 | R2  (parallel) =", R1 | R2)
print("P = V^2/R1          =", V**2 / R1, "W")

# ============================================================
# 3. CUDA inference timing -- correct vs wrong
# ============================================================
print()
print("=== CUDA inference timing ===")

from gs_fno import FNO1d, make_fno_dataset, train_fno

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

X_tr, Y_tr = make_fno_dataset(["QPSK"], n_per_format=40, N_t=512,
                               snr_db=35, D1=-5000, D2=-5750)
model = FNO1d(in_channels=2, out_channels=1, modes=32, width=32, n_layers=3)
train_fno(model, X_tr, Y_tr, n_epochs=20, lr=3e-3, batch_size=32)
model = model.to(device)
model.eval()

X_t = torch.tensor(X_tr[:32], dtype=torch.float32).to(device)

# warm up
with torch.no_grad():
    for _ in range(10):
        _ = model(X_t)
if device == "cuda":
    torch.cuda.synchronize()

# WRONG: no synchronize
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(100):
        _ = model(X_t)
t_wrong = (time.perf_counter() - t0) / 100 * 1000

# CORRECT: synchronize before stopping clock
if device == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(200):
        _ = model(X_t)
if device == "cuda":
    torch.cuda.synchronize()
t_correct = (time.perf_counter() - t0) / 200 * 1000

print()
print("WRONG  (no synchronize): {:.3f} ms/batch  <- CPU launch overhead only".format(t_wrong))
print("CORRECT (synchronize):   {:.3f} ms/batch  <- actual GPU compute time".format(t_correct))
print("Throughput:              {:.0f} inferences/sec".format(32 / (t_correct/1000)))
if device == "cuda":
    print("VRAM used:               {} MB / {} MB".format(
        torch.cuda.memory_allocated()//1024**2,
        torch.cuda.get_device_properties(0).total_memory//1024**2))
print()
print("Rule: ALWAYS torch.cuda.synchronize() before perf_counter() stop")
print("      Otherwise you measure kernel LAUNCH time (~0.01ms), not compute")
