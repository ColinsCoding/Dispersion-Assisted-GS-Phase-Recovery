"""Smoke-test the quantum Fourier transform: unitary, == circuit, == sqrt(N)*ifft."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import qubits as q
from dgs import fourier_tools as ft

rng = np.random.default_rng(0)

# 1. QFT matrix is unitary
for n in (1, 2, 3, 4):
    W = q.qft_matrix(n)
    assert np.allclose(W @ W.conj().T, np.eye(2**n), atol=1e-12), n

# 2. QFT equals sqrt(N) * numpy.fft.ifft (the classical FFT, just normalized)
for n in (2, 3, 4):
    psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    psi /= np.linalg.norm(psi)
    assert np.allclose(q.qft(psi), np.sqrt(2**n) * np.fft.ifft(psi), atol=1e-10), n

# 3. the GATE CIRCUIT (H + controlled phases + swaps) implements the same QFT
for n in (1, 2, 3, 4):
    psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    psi /= np.linalg.norm(psi)
    assert np.allclose(q.qft_circuit(psi), q.qft(psi), atol=1e-10), n

# 4. QFT of |0...0> is the uniform superposition (all amplitudes 1/sqrt(N))
n = 3
unif = q.qft(q.ket("000"))
assert np.allclose(unif, np.ones(8) / np.sqrt(8), atol=1e-12)

# 5. QFT turns a pure "frequency" state into a single computational basis spike
#    (inverse of #4): feed a uniform-phase ramp, get a delta
N = 2**n
ramp = np.exp(2j * np.pi * 1 * np.arange(N) / N) / np.sqrt(N)   # frequency k=1
out = q.inverse_qft(ramp)
peak = np.argmax(np.abs(out))
assert np.abs(out[peak])**2 > 0.999 and peak == 1               # all weight on |001>

# 6. QFT then inverse QFT is the identity
psi = rng.standard_normal(N) + 1j * rng.standard_normal(N)
psi /= np.linalg.norm(psi)
assert np.allclose(q.inverse_qft(q.qft(psi)), psi, atol=1e-10)

# 7. it stays normalized (unitary evolution)
assert abs(np.linalg.norm(q.qft_circuit(psi)) - 1.0) < 1e-10

print(f"SMOKE PASS  (QFT unitary; circuit == matrix == sqrt(N)*ifft; "
      f"QFT|000> = uniform; gate count O(n^2) = {n*(n+1)//2} for n={n})")
