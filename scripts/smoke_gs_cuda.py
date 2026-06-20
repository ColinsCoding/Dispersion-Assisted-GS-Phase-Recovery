"""Smoke-test CUDA GS: matches CPU gs_core, and benchmark GPU vs CPU (py -3.12)."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import gs_core as gs
import gs_cuda
import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

# correctness: CUDA recovery matches the CPU gs_core recovery
data = gs.make_qpsk_measurements(n_symbols=64, sps=8, D1=-5000.0, D2=-5750.0, snr_db=40.0)
phi_true = data["phi_true"]
phi_cpu, _ = gs.retrieve_phase(data["I1"], data["I2"], data["D1"], data["D2"], n_iter=80)
phi_gpu = gs_cuda.gerchberg_saxton_cuda(data["I1"], data["I2"], data["D1"], data["D2"], n_iter=80)

def rms(a, b):
    return float(np.sqrt(np.mean(np.angle(np.exp(1j*(a-b)))**2)))
print(f"CPU vs GPU phase agreement: {rms(phi_cpu, phi_gpu):.4f} rad (should be tiny)")
assert rms(phi_cpu, phi_gpu) < 0.05

# benchmark on a larger signal
N = 8192
big = gs.make_qpsk_measurements(n_symbols=1024, sps=8, D1=-5000.0, D2=-5750.0)
t0 = time.time(); gs.retrieve_phase(big["I1"], big["I2"], big["D1"], big["D2"], n_iter=120); t_cpu = time.time()-t0
# warm up GPU
gs_cuda.gerchberg_saxton_cuda(big["I1"], big["I2"], big["D1"], big["D2"], n_iter=5)
if dev == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
gs_cuda.gerchberg_saxton_cuda(big["I1"], big["I2"], big["D1"], big["D2"], n_iter=120)
if dev == "cuda":
    torch.cuda.synchronize()
t_gpu = time.time()-t0
print(f"\nN={len(big['I1'])}, 120 iterations:")
print(f"  CPU (gs_core, numpy): {t_cpu*1e3:.1f} ms")
print(f"  GPU (gs_cuda, {dev}):    {t_gpu*1e3:.1f} ms   speedup {t_cpu/t_gpu:.1f}x")

try:
    gs_cuda.gerchberg_saxton_cuda(data["I1"], data["I2"], -5000, -5000)
except ValueError as e:
    print("err ok:", e)
print("SMOKE PASS")
