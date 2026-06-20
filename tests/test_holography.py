"""Smoke-test computer-generated holography: GS (numpy) and torch-autograd CGH.
Run with py -3.12 (needs torch)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

def make_target(N=128):
    # a sparse, recognizable target: a ring + a dot (clean phase-only reconstruction)
    yy, xx = np.mgrid[0:N, 0:N]
    cy = cx = N // 2
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    T = np.zeros((N, N))
    T[(r > 24) & (r < 30)] = 1.0
    T[(yy - cy + 12)**2 + (xx - cx)**2 < 16] = 1.0
    return T / T.sum()          # normalize to unit total intensity

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

N = 128
T = make_target(N)
target_amp = np.sqrt(T)

# --- 1. Gerchberg-Saxton phase-only CGH (numpy) ---
rng = np.random.default_rng(0)
phi = rng.uniform(-np.pi, np.pi, (N, N))
for _ in range(60):
    slm = np.exp(1j * phi)                     # phase-only field at the SLM
    far = np.fft.fftshift(np.fft.fft2(slm))    # far-field reconstruction
    far = target_amp * np.exp(1j * np.angle(far))   # enforce target amplitude
    back = np.fft.ifft2(np.fft.ifftshift(far))
    phi = np.angle(back)                        # enforce unit amplitude (phase-only)
recon_gs = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * phi))))**2
recon_gs /= recon_gs.sum()
print(f"GS CGH:    reconstruction-vs-target correlation = {corr(recon_gs, T):.3f}")

# --- 2. torch autograd CGH (gradient descent on the phase mask) ---
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
Tt = torch.tensor(T, dtype=torch.float32, device=dev)
phi0 = torch.tensor(rng.uniform(-np.pi, np.pi, (N, N)), dtype=torch.float32, device=dev)
phi_t = phi0.clone().requires_grad_(True)     # random init (zero init -> central spike)
opt = torch.optim.Adam([phi_t], lr=0.05)
Tf = Tt.flatten()
for it in range(400):
    opt.zero_grad()
    field = torch.exp(1j * phi_t.to(torch.complex64))
    far = torch.fft.fftshift(torch.fft.fft2(field))
    recon = (far.abs()**2)
    recon = recon / recon.sum()
    # negative cosine similarity: directly maximize overlap (MSE on a sparse
    # target has a trivial uniform-spread minimum -- this avoids it)
    rf = recon.flatten()
    loss = 1 - (rf @ Tf) / (rf.norm() * Tf.norm() + 1e-12)
    loss.backward()
    opt.step()
recon_torch = recon.detach().cpu().numpy()
print(f"torch CGH: device={dev}, final loss={loss.item():.3e}, "
      f"correlation = {corr(recon_torch, T):.3f}")

assert corr(recon_gs, T) > 0.5, "GS CGH failed"
assert corr(recon_torch, T) > 0.5, "torch CGH failed"
print("SMOKE PASS")
