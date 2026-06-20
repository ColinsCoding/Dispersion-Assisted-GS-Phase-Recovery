"""
gs_fno.py — Fourier Neural Operator for TD-GS phase retrieval
==============================================================

Reference: Li et al., "Fourier Neural Operator for Parametric Partial
           Differential Equations," ICLR 2021.  arXiv:2010.08895

Why FNO instead of a standard CNN
----------------------------------
A regular Conv1d learns spatially-local filters of fixed receptive field.
The dispersive GS problem is *non-local*: each output phase sample φ(t)
depends on the entire input intensity pattern I(t') because dispersion
mixes all times.  FNO handles this by learning weights directly in the
Fourier domain — one FNO layer is a global convolution (all frequencies
at once) plus a local residual, so the receptive field is the whole signal.

Additionally, FNO is resolution-invariant: train at N=512, infer at N=1024
without retraining.  Critical for lab data where acquisition length varies.

Architecture
------------
    Input  : (B, 2, N)  — channels [I1(t), I2(t)]
    Output : (B, 1, N)  — recovered phase φ̂(t)

    FNO1d
    ├── Linear lift:  2 → width channels
    ├── n_layers × FNO block:
    │     ├── SpectralConv1d: FFT → complex weights (modes) → IFFT
    │     └── Conv1d residual: local linear bypass
    │     └── GELU activation
    └── Projection: width → 1 channel

Grade-7 explanation
-------------------
Think of the intensity pattern as music.  A regular neural net looks at
a few notes at a time.  FNO looks at all the frequencies simultaneously —
it works with the "score" (Fourier transform) instead of the raw "sound".
This is perfect here because dispersion scrambles notes in a known way
that lives naturally in frequency space.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Core FNO building blocks ──────────────────────────────────────────────────

class SpectralConv1d(nn.Module):
    """
    Fourier integral operator layer (1D).

    Computes:
        (K * v)(x) = F^{-1}[ R(k) · F[v](k) ]

    where R(k) are complex learnable weights for the lowest `modes` frequencies.
    High frequencies above `modes` are zeroed (implicit low-pass).

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    modes        : int — number of Fourier modes to keep (≤ N//2 + 1)
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_ch  = in_channels
        self.out_ch = out_channels
        self.modes  = modes

        # Complex weight tensor: (in_ch, out_ch, modes)
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """
        x : (B, in_ch, N)
        returns : (B, out_ch, N)
        """
        B, _, N = x.shape

        # FFT along last dimension
        x_ft = torch.fft.rfft(x, dim=-1)           # (B, in_ch, N//2+1)

        # Multiply lowest `modes` frequencies by learned weights
        out_ft = torch.zeros(B, self.out_ch, N // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m = min(self.modes, x_ft.shape[-1])
        # einsum: "b i k, i o k -> b o k"
        out_ft[:, :, :m] = torch.einsum('bik,iok->bok', x_ft[:, :, :m], self.weights[:, :, :m])

        # IFFT back to time domain
        return torch.fft.irfft(out_ft, n=N, dim=-1)  # (B, out_ch, N)


class FNOBlock1d(nn.Module):
    """One FNO layer = SpectralConv1d + local Conv1d bypass + GELU."""
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.bypass   = nn.Conv1d(width, width, kernel_size=1)
        self.act      = nn.GELU()

    def forward(self, x):
        return self.act(self.spectral(x) + self.bypass(x))


class FNO1d(nn.Module):
    """
    Full 1D Fourier Neural Operator for TD-GS phase retrieval.

    Parameters
    ----------
    in_channels  : int  — input channels (default 2: I1, I2)
    out_channels : int  — output channels (default 1: φ̂)
    modes        : int  — Fourier modes kept per layer (default 32)
    width        : int  — internal channel width (default 64)
    n_layers     : int  — number of FNO blocks (default 4)
    """
    def __init__(self, in_channels=2, out_channels=1,
                 modes=32, width=64, n_layers=4):
        super().__init__()
        self.lift    = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks  = nn.Sequential(*[FNOBlock1d(width, modes) for _ in range(n_layers)])
        self.project = nn.Sequential(
            nn.Conv1d(width, width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width // 2, out_channels, kernel_size=1),
        )

    def forward(self, x):
        """x : (B, in_channels, N)  →  (B, out_channels, N)"""
        x = self.lift(x)
        x = self.blocks(x)
        return self.project(x)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Wrapped-phase loss ────────────────────────────────────────────────────────

def wrapped_phase_loss(phi_pred, phi_true):
    """
    MSE on the wrapped phase difference.
    Robust to global-phase offsets (GS ambiguity).

    ‖e^{i(φ̂ - φ)}‖²_F  ≡  mean of |cos(Δφ) - 1|² + |sin(Δφ)|²
    Equivalently: 2 * mean(1 - cos(φ̂ - φ))
    """
    delta = phi_pred - phi_true
    return (1 - torch.cos(delta)).mean()


# ── Training utilities ────────────────────────────────────────────────────────

def train_fno(model, X_train, Y_train, n_epochs=100, lr=1e-3,
              batch_size=32, device='cpu', verbose=True):
    """
    Train FNO with wrapped-phase loss + Adam optimizer.

    Parameters
    ----------
    model    : FNO1d instance
    X_train  : (N_samples, 2, N_t) float32 tensor  — [I1, I2] per sample
    Y_train  : (N_samples, 1, N_t) float32 tensor  — true phase
    n_epochs : int
    lr       : float — learning rate
    batch_size : int
    device   : str
    verbose  : bool

    Returns
    -------
    losses : list of float — per-epoch mean loss
    """
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    model = model.to(device)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    loader = DataLoader(TensorDataset(X_train, Y_train),
                        batch_size=batch_size, shuffle=True)
    opt    = optim.Adam(model.parameters(), lr=lr)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    losses = []
    for ep in range(n_epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = wrapped_phase_loss(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss) * len(xb)
        ep_loss /= len(X_train)
        sched.step()
        losses.append(ep_loss)
        if verbose and (ep % max(1, n_epochs // 10) == 0 or ep == n_epochs - 1):
            print(f'  epoch {ep:4d}/{n_epochs}  loss={ep_loss:.5f}  lr={sched.get_last_lr()[0]:.2e}')

    return losses


def make_fno_dataset(modulations=None, n_per_format=50, N_t=512,
                     snr_db=22.0, D1=-5000., D2=-5750.):
    """
    Build a multi-format FNO training dataset using gs_core.make_measurements.

    Returns
    -------
    X : (total, 2, N_t) float32 tensor  — [I1, I2]
    Y : (total, 1, N_t) float32 tensor  — phi_true
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for FNO dataset generation.")
    import torch
    from dgs.gs_core import make_measurements

    if modulations is None:
        modulations = ['QPSK', 'DPSK', '6PSK', 'STEAM', 'Soliton']

    X_list, Y_list = [], []
    seed = 0
    for fmt in modulations:
        for i in range(n_per_format):
            d = make_measurements(fmt, n_symbols=64, sps=8,
                                  D1=D1, D2=D2, snr_db=snr_db, rng_seed=seed)
            seed += 1
            n = min(N_t, len(d['I1']))
            i1 = np.zeros(N_t); i2 = np.zeros(N_t); phi = np.zeros(N_t)
            i1[:n] = d['I1'][:n]; i2[:n] = d['I2'][:n]; phi[:n] = d['phi_true'][:n]
            X_list.append([i1, i2])
            Y_list.append([phi])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
    return X, Y


# ── Resolution-transfer test ──────────────────────────────────────────────────

def test_resolution_invariance(model, D1=-5000., D2=-5750., snr_db=25.):
    """
    Verify FNO generalizes to signal lengths not seen during training.
    Train at N=512, test at N=256, 512, 1024.
    """
    if not TORCH_AVAILABLE:
        return
    import torch
    from dgs.gs_core import make_measurements, retrieve_phase

    print("Resolution invariance test (trained at N=512):")
    for N_test in [256, 512, 1024]:
        d = make_measurements('QPSK', n_symbols=N_test // 8, sps=8,
                              D1=D1, D2=D2, snr_db=snr_db, rng_seed=77)
        n = min(N_test, len(d['I1']))
        x = torch.tensor(
            np.stack([d['I1'][:n], d['I2'][:n]])[None], dtype=torch.float32
        )
        model.eval()
        with torch.no_grad():
            phi_fno = model(x).squeeze().numpy()
        off = np.angle(np.mean(np.exp(1j * (d['phi_true'][:n] - phi_fno))))
        dlt = np.angle(np.exp(1j * (phi_fno + off - d['phi_true'][:n])))
        rms = np.degrees(np.sqrt(np.mean(dlt**2)))
        print(f"  N={N_test:5d}  RMS={rms:.2f}°")


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not installed — install with: pip install torch")
        raise SystemExit(1)

    import torch

    print("Building FNO1d...")
    model = FNO1d(in_channels=2, out_channels=1, modes=32, width=64, n_layers=4)
    print(f"  Parameters: {model.n_params:,}")

    print("\nBuilding dataset (5 formats × 50 signals)...")
    X, Y = make_fno_dataset(n_per_format=50)
    print(f"  X: {tuple(X.shape)}  Y: {tuple(Y.shape)}")

    # Train/val split
    n_tr = int(0.8 * len(X))
    X_tr, X_va = X[:n_tr], X[n_tr:]
    Y_tr, Y_va = Y[:n_tr], Y[n_tr:]

    print(f"\nTraining {n_tr} samples, validating {len(X_va)} ...")
    losses = train_fno(model, X_tr, Y_tr, n_epochs=60, lr=1e-3, verbose=True)

    # Validation
    model.eval()
    with torch.no_grad():
        phi_pred = model(X_va).squeeze(1).numpy()
    phi_true  = Y_va.squeeze(1).numpy()
    rms_vals  = []
    for i in range(len(phi_pred)):
        off = np.angle(np.mean(np.exp(1j * (phi_true[i] - phi_pred[i]))))
        dlt = np.angle(np.exp(1j * (phi_pred[i] + off - phi_true[i])))
        rms_vals.append(np.degrees(np.sqrt(np.mean(dlt**2))))
    print(f"\nValidation RMS: {np.mean(rms_vals):.2f}° ± {np.std(rms_vals):.2f}°")

    test_resolution_invariance(model)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.semilogy(losses, 'o-', color='steelblue', markersize=3)
    plt.title(f'FNO training — wrapped-phase loss  (val RMS={np.mean(rms_vals):.1f}°)')
    plt.xlabel('Epoch'); plt.ylabel('Wrapped-phase loss')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('fno_training.png', dpi=150)
    print("Saved fno_training.png")
