"""
PyTorch differentiable implementation of Coppinger 1999 photonic time stretch.
Extends dgs/coppinger1999.py with autograd-capable forward model.

Use py-3.12 ONLY (torch not available on py-3.13).

Architecture:
  CoppingerForward: H(f) = exp(j*pi*D*f^2) as nn.Module with learnable D
  TimeStretchEncoder: full pipeline (chirp -> MZM -> stretch -> detect)
  PhaseRetrievalNet: GS unrolled as differentiable neural network
  Training loop: recover phase from intensity measurements

Hermitian connection (moments):
  E[|x|^2] = first moment of intensity (mean power)
  E[t * |x|^2] = centroid (first temporal moment)
  E[t^2 * |x|^2] = second moment = pulse duration squared
  These are Hermitian observables <psi|O|psi> where O is t, t^2, etc.
  GS algorithm minimizes discrepancy between measured and estimated moments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor


# ---------------------------------------------------------------------------
# Core dispersive operator H(f) = exp(j*pi*D*f^2)
# ---------------------------------------------------------------------------
class DispersivePhaseFilter(nn.Module):
    """
    H(f) = exp(j*pi*D*f^2) as a differentiable torch layer.
    D is learnable if learn_D=True -- gradient flows through the phase.

    Coppinger 1999 notation: beta2 * L2 maps to D via D = 2*pi^2*L2*beta2
    This repo notation: D [ps^2] directly.
    """
    def __init__(self, D_init: float = -5000.0, N: int = 256, learn_D: bool = False):
        super().__init__()
        self.N = N
        if learn_D:
            self.D = nn.Parameter(torch.tensor(D_init, dtype=torch.float64))
        else:
            self.register_buffer('D', torch.tensor(D_init, dtype=torch.float64))

        # Frequency axis (normalized: [-0.5, 0.5])
        f_np = np.fft.fftfreq(N)
        self.register_buffer('f_axis', torch.tensor(f_np, dtype=torch.float64))

    def forward(self, E: Tensor) -> Tensor:
        """
        Apply dispersive phase: E_out(f) = E_in(f) * exp(j*pi*D*f^2)
        E: complex tensor of shape (..., N)
        """
        phase = torch.pi * self.D * self.f_axis**2  # shape (N,)
        H = torch.polar(torch.ones_like(phase), phase)  # |H|=1, angle=phase
        # Apply in frequency domain
        E_f = torch.fft.fft(E, dim=-1)
        E_f_out = E_f * H
        return torch.fft.ifft(E_f_out, dim=-1)

    def transfer_function(self) -> Tensor:
        """Return H(f) directly for inspection."""
        phase = torch.pi * self.D * self.f_axis**2
        return torch.polar(torch.ones_like(phase), phase)

    def extra_repr(self) -> str:
        return f'D={float(self.D):.1f} ps^2, N={self.N}, learn_D={self.D.requires_grad}'


# ---------------------------------------------------------------------------
# MZM modulator (Coppinger Eq 3)
# ---------------------------------------------------------------------------
class MachZehnderModulator(nn.Module):
    """
    Differentiable MZM: E_out = E_in * (1 + a*cos(2*pi*fm*t))
    a and fm can be learned (e.g., to infer input RF signal).

    Coppinger Eq(3): E_in(L1,t) = E_ch(L1,t) * [1 + a*cos(2*pi*fm*t)]
    LiNbO3 connection: a = V_rf/V_pi * sin(pi*V_bias/V_pi)
    """
    def __init__(self, a: float = 0.3, fm_normalized: float = 0.05,
                 learn_params: bool = False, N: int = 256):
        super().__init__()
        self.N = N
        t_np = np.linspace(-1, 1, N)
        self.register_buffer('t_axis', torch.tensor(t_np, dtype=torch.float64))

        if learn_params:
            self.a = nn.Parameter(torch.tensor(a, dtype=torch.float64))
            self.fm = nn.Parameter(torch.tensor(fm_normalized, dtype=torch.float64))
        else:
            self.register_buffer('a', torch.tensor(a, dtype=torch.float64))
            self.register_buffer('fm', torch.tensor(fm_normalized, dtype=torch.float64))

    def forward(self, E: Tensor) -> Tensor:
        """Apply MZM modulation. E shape: (..., N), complex."""
        mod = 1 + self.a * torch.cos(2 * torch.pi * self.fm * self.t_axis)
        return E * mod.to(E.dtype)


# ---------------------------------------------------------------------------
# Chirped Gaussian source (Eq 1-2 of Coppinger)
# ---------------------------------------------------------------------------
class ChirpedGaussianSource(nn.Module):
    """
    Generate chirped Gaussian pulse: E_ch(t) = exp(-t^2/tau^2) * exp(-j*phi_chirp*t^2)
    tau: pulse half-width (normalized)
    phi_chirp: quadratic phase (chirp, from first dispersive fiber L1)
    """
    def __init__(self, tau: float = 0.2, phi_chirp: float = 0.0, N: int = 256):
        super().__init__()
        self.N = N
        t_np = np.linspace(-1, 1, N)
        self.register_buffer('t_axis', torch.tensor(t_np, dtype=torch.float64))
        self.tau = tau
        self.phi_chirp = phi_chirp

    def forward(self, batch_size: int = 1) -> Tensor:
        """Generate batch of chirped Gaussian pulses. Shape: (batch, N)."""
        t = self.t_axis
        envelope = torch.exp(-t**2 / self.tau**2)
        chirp = torch.exp(torch.tensor(1j * self.phi_chirp, dtype=torch.complex128) * t**2)
        E = (envelope * chirp).unsqueeze(0).expand(batch_size, -1)
        return E.clone()


# ---------------------------------------------------------------------------
# Full Coppinger forward model (Fig.1 of paper)
# ---------------------------------------------------------------------------
class CoppingerForward(nn.Module):
    """
    Complete photonic time-stretch pipeline (Coppinger 1999, Fig.1):
      1. Chirped Gaussian source (first fiber L1 already applied)
      2. Mach-Zehnder modulator (RF signal encoding)
      3. Second dispersive fiber L2: H(f) = exp(j*pi*D*f^2)
      4. Photodetector: I = |E_out|^2

    Differentiable end-to-end: gradients flow from I_detected back to D and a.
    This enables:
      - Learning D from measured intensities (calibration)
      - Inverting the forward model (phase retrieval)
      - GS algorithm as unrolled neural net (PhaseRetrievalNet below)
    """
    def __init__(self, D: float = -5000.0, a: float = 0.3,
                 fm_normalized: float = 0.05, tau: float = 0.2,
                 N: int = 256, learn_D: bool = False, learn_mzm: bool = False):
        super().__init__()
        self.N = N
        self.source = ChirpedGaussianSource(tau=tau, N=N)
        self.mzm = MachZehnderModulator(a=a, fm_normalized=fm_normalized,
                                         learn_params=learn_mzm, N=N)
        self.dispersive = DispersivePhaseFilter(D_init=D, N=N, learn_D=learn_D)

    def forward(self, batch_size: int = 1):
        """
        Forward pass: source -> MZM -> disperse -> detect
        Returns: (I_detected, E_out_complex)
        I_detected: (batch, N) real intensities (what ADC measures)
        E_out_complex: (batch, N) complex field (for phase retrieval)
        """
        E_ch = self.source(batch_size)                 # chirped pulse
        E_mod = self.mzm(E_ch)                         # after MZM (Eq 3)
        E_out = self.dispersive(E_mod)                 # after L2 (Eq 5)
        I_detected = torch.abs(E_out)**2               # photodetector (Eq 7)
        return I_detected, E_out

    def stretch_factor(self) -> float:
        """M = 1 + |D2|/|D1|. Approximate from source chirp."""
        return 1.0  # placeholder -- override with actual L1,L2 values

    def extra_repr(self):
        return f'N={self.N}'


# ---------------------------------------------------------------------------
# Unrolled GS algorithm as differentiable neural network (10 iterations)
# ---------------------------------------------------------------------------
class GSIteration(nn.Module):
    """One iteration of Gerchberg-Saxton as a differentiable layer."""
    def __init__(self, D1: float, D2: float, N: int = 256):
        super().__init__()
        self.H1 = DispersivePhaseFilter(D_init=D1, N=N, learn_D=False)
        self.H2 = DispersivePhaseFilter(D_init=D2, N=N, learn_D=False)

    def forward(self, E_est: Tensor, I1_sqrt: Tensor, I2_sqrt: Tensor) -> Tensor:
        """
        One GS iteration:
          1. Apply H1 -> plane 2
          2. Replace amplitude with sqrt(I2), keep phase
          3. Apply H2 (inverse) -> plane 1
          4. Replace amplitude with sqrt(I1), keep phase
        """
        # Forward: plane1 -> plane2
        E2 = self.H1(E_est)
        phase2 = E2 / (torch.abs(E2) + 1e-12)
        E2 = I2_sqrt * phase2

        # Backward: plane2 -> plane1
        E1 = self.H2(E2)
        phase1 = E1 / (torch.abs(E1) + 1e-12)
        E1 = I1_sqrt * phase1

        return E1


class PhaseRetrievalNet(nn.Module):
    """
    GS phase retrieval as unrolled neural network (n_iter iterations).
    Each iteration is one GSIteration module -- differentiable end-to-end.
    Gradients can train a learnable phase correction layer on top.

    Moments connection:
      The network implicitly minimizes the moment discrepancy:
        L = ||E[t^2 * |E_est|^2] - E[t^2 * |E_true|^2]||
      i.e., matching the second temporal moment (pulse duration).
      This is a Hermitian observable <psi|t^2|psi>.
    """
    def __init__(self, D1: float = 5000.0, D2: float = -5000.0,
                 n_iter: int = 10, N: int = 256):
        super().__init__()
        self.n_iter = n_iter
        self.N = N
        # Unrolled GS iterations (shared weights)
        self.gs_iter = GSIteration(D1, D2, N)
        # Optional learnable phase correction (1x1 conv in freq domain)
        self.phase_correction = nn.Parameter(
            torch.zeros(N, dtype=torch.float64)
        )

    def forward(self, I1: Tensor, I2: Tensor) -> tuple[Tensor, list]:
        """
        Recover phase from two intensity measurements.
        I1, I2: (batch, N) real intensity tensors
        Returns: (E_recovered, correlation_history)
        """
        I1_sqrt = torch.sqrt(I1.to(torch.float64) + 1e-15)
        I2_sqrt = torch.sqrt(I2.to(torch.float64) + 1e-15)

        # Initialize with random phase
        init_phase = torch.rand(I1.shape, dtype=torch.float64) * 2 * torch.pi
        E_est = I1_sqrt * torch.exp(1j * init_phase.to(torch.complex128))

        corr_history = []
        for _ in range(self.n_iter):
            E_est = self.gs_iter(E_est, I1_sqrt.to(torch.complex128),
                                  I2_sqrt.to(torch.complex128))

        # Apply learnable phase correction in freq domain
        E_f = torch.fft.fft(E_est, dim=-1)
        correction = torch.exp(1j * self.phase_correction.to(torch.complex128))
        E_corrected = torch.fft.ifft(E_f * correction, dim=-1)

        return E_corrected, corr_history

    def intensity_loss(self, I_pred: Tensor, I_target: Tensor) -> Tensor:
        """L2 loss on intensities -- differentiable training objective."""
        return F.mse_loss(I_pred.double(), I_target.double())

    def moment_loss(self, E_pred: Tensor, E_target: Tensor,
                    t_axis: Tensor, order: int = 2) -> Tensor:
        """
        Hermitian moment loss: ||<E_pred|t^order|E_pred> - <E_target|t^order|E_target>||
        order=1: centroid; order=2: duration^2 (RMS pulse width)
        These are Hermitian observables with real expectation values.
        """
        I_pred = torch.abs(E_pred)**2
        I_target = torch.abs(E_target)**2
        t = t_axis.to(E_pred.device)
        moment_pred = (I_pred * t**order).sum(dim=-1)
        moment_target = (I_target * t**order).sum(dim=-1)
        return F.mse_loss(moment_pred, moment_target)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_phase_retrieval(D1: float = 5000.0, D2: float = -5000.0,
                           N: int = 256, n_iter: int = 10,
                           n_epochs: int = 100, lr: float = 1e-3,
                           n_train: int = 32, verbose: bool = True):
    """
    Train PhaseRetrievalNet to recover phases from synthetic intensity data.
    Uses CoppingerForward to generate training pairs (I1, I2, E_true).

    Loop structure:
      for epoch in range(n_epochs):
          for batch in dataloader:
              E_recovered = net(I1, I2)
              loss = intensity_loss + moment_loss
              loss.backward()
              optimizer.step()
    """
    device = torch.device('cpu')

    # Forward model to generate training data
    forward = CoppingerForward(D=D2, a=0.3, fm_normalized=0.05, tau=0.2, N=N)
    forward.eval()

    # Phase retrieval network
    net = PhaseRetrievalNet(D1=D1, D2=D2, n_iter=n_iter, N=N)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    t_axis = torch.linspace(-1, 1, N, dtype=torch.float64)
    history = {'loss': [], 'epoch': []}

    for epoch in range(n_epochs):
        # Generate synthetic batch
        with torch.no_grad():
            I2, E_true = forward(batch_size=n_train)
            I1 = torch.abs(forward.source(batch_size=n_train))**2

        # Forward pass through retrieval net
        E_recovered, _ = net(I1, I2)
        I_reconstructed = torch.abs(E_recovered)**2

        # Loss: intensity match + second moment (pulse duration) match
        loss_I = net.intensity_loss(I_reconstructed, I2)
        loss_m = net.moment_loss(E_recovered, E_true.to(E_recovered.dtype),
                                  t_axis, order=2)
        loss = loss_I + 0.1 * loss_m

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history['loss'].append(loss.detach().item())
        history['epoch'].append(epoch)

        if verbose and (epoch % 20 == 0 or epoch == n_epochs-1):
            print(f"  Epoch {epoch:4d}/{n_epochs}: loss={float(loss):.6f} "
                  f"(I={float(loss_I):.6f}, moment={float(loss_m):.6f})")

    return net, history


def demo():
    print("=== CoppingerLayer Torch Demo (py-3.12) ===")

    N = 128
    D = -5000.0

    # 1. Dispersive filter
    H = DispersivePhaseFilter(D_init=D, N=N, learn_D=False)
    E_in = torch.ones(1, N, dtype=torch.complex128)
    E_out = H(E_in)
    print(f"[1] H(f): |E_in|={E_in.abs().mean():.3f} -> |E_out|={E_out.abs().mean():.3f} (unitary)")

    # 2. Full forward model
    model = CoppingerForward(D=D, a=0.3, fm_normalized=0.05, N=N)
    I_det, E_field = model(batch_size=4)
    print(f"[2] CoppingerForward: I shape={I_det.shape}, mean I={I_det.mean():.4f}")

    # 3. Phase retrieval net (no training)
    net = PhaseRetrievalNet(D1=5000, D2=D, n_iter=5, N=N)
    I1 = torch.abs(model.source(1))**2
    I2, _ = model(1)
    E_rec, _ = net(I1, I2)
    print(f"[3] PhaseRetrievalNet output shape: {E_rec.shape}")

    # 4. Short training loop
    print("[4] Training loop (10 epochs):")
    trained_net, hist = train_phase_retrieval(D1=5000, D2=D, N=N,
                                               n_iter=5, n_epochs=10,
                                               n_train=8, verbose=True)
    print(f"    Final loss: {hist['loss'][-1]:.6f}")
    print("Torch add-on to Coppinger1999 complete.")


if __name__ == '__main__':
    demo()
