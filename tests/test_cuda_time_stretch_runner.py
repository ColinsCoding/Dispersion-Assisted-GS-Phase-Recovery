"""Test the real, compiled-and-executed CUDA time-stretch/dispersive-
propagation kernel against an independent NumPy reference. Requires nvcc
AND cl.exe (MSVC) on PATH -- run from PowerShell with the MSVC bin
directory prepended, per dgs.cuda_time_stretch_runner's module docstring."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tempfile
import numpy as np
from dgs import cuda_time_stretch_runner as ctsr

N = 64
dt = 1.0 / N
t = (np.arange(N) - N // 2) * dt

# 1. a Gaussian pulse through the dispersive kernel matches the NumPy
#    reference to float32 precision (not float64-tight -- genuine GPU
#    single-precision roundoff through an FFT+multiply+IFFT chain)
E_in = np.exp(-(t ** 2) / (2 * 0.1 ** 2)).astype(complex)
beta2L = 0.02
E_ref = ctsr.numpy_reference_dispersion(E_in, dt, beta2L)
with tempfile.TemporaryDirectory() as tmp:
    E_cuda = ctsr.run_cuda_dispersion(E_in, dt, beta2L, tmp)
max_err = np.max(np.abs(E_cuda - E_ref))
max_scale = np.max(np.abs(E_ref))
assert max_err / max_scale < 1e-3

# 2. energy is conserved (dispersion is phase-only, |H(f)|=1) to float32 precision
input_energy = np.sum(np.abs(E_in) ** 2)
output_energy = np.sum(np.abs(E_cuda) ** 2)
assert abs(input_energy - output_energy) / input_energy < 1e-3

# 3. zero dispersion (beta2L=0) must return the input essentially unchanged
#    (H(f)=1 everywhere when beta2L=0) -- a real discriminating check, not
#    just "small error is always fine"
with tempfile.TemporaryDirectory() as tmp:
    E_cuda_zero_disp = ctsr.run_cuda_dispersion(E_in, dt, 0.0, tmp)
assert np.max(np.abs(E_cuda_zero_disp - E_in)) < 1e-3

# 4. a DIFFERENT beta2L genuinely changes the output (confirms the kernel
#    is actually applying dispersion, not silently ignoring the parameter --
#    echoes the real bug this session found in dgs.cuda_photonic_ai, where
#    D_ps_nm_km/L_km were computed but then discarded)
with tempfile.TemporaryDirectory() as tmp:
    E_cuda_other_disp = ctsr.run_cuda_dispersion(E_in, dt, 0.1, tmp)
assert np.max(np.abs(E_cuda_other_disp - E_cuda)) > 1e-3

print("all dgs.cuda_time_stretch_runner tests passed")
