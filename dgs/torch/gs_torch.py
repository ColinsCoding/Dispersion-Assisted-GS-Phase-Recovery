"""Torch-based dispersion-assisted phase retrieval.

Two complementary optimizers for recovering the phase of E(t) from two
intensity measurements I1 = |E|^2 and I2 = |D(E)|^2:

  1. TorchGS  -- alternating projections (Gerchberg-Saxton) as a
                 differentiable torch module.  Each projection is a
                 sub-gradient step; the whole sequence is end-to-end
                 differentiable through autograd.

  2. GradientPhaseRetrieval  -- direct gradient descent on the
                 self-consistency loss.  The complex field E = A * exp(i*phi)
                 is parameterised with learnable amplitude A and phase phi;
                 Adam minimises  L = ||A^2 - I1||^2 + ||D(A*exp(i*phi))^2 - I2||^2.

  3. HybridGS  -- runs TorchGS for warm-start, then switches to
                 GradientPhaseRetrieval for fine-tuning.  Best of both.

Why torch for phase retrieval?
  The alternating-projection GS algorithm converges reliably but can stall
  in local minima.  Gradient descent can escape those minima, but it requires
  a differentiable forward model.  By implementing the GVD propagator
  exp(i*beta2/2 * omega^2) as a torch.fft operation, the entire pipeline
  becomes differentiable and Adam can optimize directly over the phase.

Run with py -3.12 (torch not available on 3.13).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.fft as fft
from typing import Optional, Tuple

from dgs.torch.gs_layer import DispersionForward


# ---------------------------------------------------------------------------
# GVD operator (shared by all classes)
# ---------------------------------------------------------------------------

def _make_beta(N: int, D: float, device='cpu') -> torch.Tensor:
    """GVD phase: beta[k] = D/2 * omega_k^2 where omega = 2pi*fftfreq."""
    omega = torch.fft.fftfreq(N, d=1.0) * 2 * torch.pi
    return (D / 2) * omega ** 2


def _disperse(E: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Apply GVD: E_out = IFFT[ FFT(E) * exp(i*beta) ]."""
    return fft.ifft(fft.fft(E) * torch.exp(1j * beta))


def _loss(E: torch.Tensor, I1: torch.Tensor,
          I2: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Self-consistency loss: RMS mismatch on both intensity constraints."""
    err1 = torch.mean((torch.abs(E) ** 2 - I1) ** 2)
    err2 = torch.mean((torch.abs(_disperse(E, beta)) ** 2 - I2) ** 2)
    return torch.sqrt(err1) + torch.sqrt(err2)


# ---------------------------------------------------------------------------
# 1. TorchGS — alternating projections, fully differentiable
# ---------------------------------------------------------------------------

class TorchGS(nn.Module):
    """Gerchberg-Saxton as a differentiable torch module.

    Parameters
    ----------
    N      : signal length
    D      : dispersion value (|D| >= 100)
    n_iter : number of GS iterations per forward pass
    """

    def __init__(self, N: int, D: float, n_iter: int = 50):
        super().__init__()
        if abs(D) < 100:
            raise ValueError(f"|D| must be >= 100; got {D}")
        self.n_iter = n_iter
        beta = _make_beta(N, D)
        self.register_buffer('beta', beta)

    def forward(self, I1: torch.Tensor, I2: torch.Tensor,
                E0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run n_iter GS steps from initial field E0 (random if None).

        Returns the complex field E after n_iter projections.
        """
        N = I1.shape[-1]
        if E0 is None:
            phase0 = torch.rand(N, dtype=torch.float32,
                                device=I1.device) * 2 * torch.pi
            E0 = torch.sqrt(I1.clamp(min=0)) * torch.exp(1j * phase0)

        E = E0.to(torch.complex64)
        for _ in range(self.n_iter):
            # Enforce I1: replace amplitude, keep phase
            E = torch.sqrt(I1.clamp(min=1e-12)) * torch.exp(1j * torch.angle(E))
            # Disperse, enforce I2, undisperse
            Ed = _disperse(E, self.beta)
            Ed = torch.sqrt(I2.clamp(min=1e-12)) * torch.exp(1j * torch.angle(Ed))
            E  = _disperse(Ed, -self.beta)
        return E


# ---------------------------------------------------------------------------
# 2. GradientPhaseRetrieval — Adam on self-consistency loss
# ---------------------------------------------------------------------------

class GradientPhaseRetrieval(nn.Module):
    """Direct gradient descent on the self-consistency loss.

    The complex field is reparameterised as E = amplitude * exp(i * phase)
    with both amplitude and phase as learnable real parameters.  This
    avoids the non-holomorphic issue with complex parameters in autograd.

    Parameters
    ----------
    N      : signal length
    D      : dispersion value
    lr     : Adam learning rate
    n_steps: gradient descent steps per call to .optimize()
    """

    def __init__(self, N: int, D: float, lr: float = 1e-2, n_steps: int = 500):
        super().__init__()
        beta = _make_beta(N, D)
        self.register_buffer('beta', beta)
        self.lr      = lr
        self.n_steps = n_steps

        # learnable parameters: log-amplitude (ensures positivity) and phase
        self.log_amp = nn.Parameter(torch.zeros(N))
        self.phase   = nn.Parameter(torch.zeros(N))

    def _field(self) -> torch.Tensor:
        return torch.exp(self.log_amp) * torch.exp(1j * self.phase)

    def optimize(self, I1: torch.Tensor, I2: torch.Tensor,
                 E_init: Optional[torch.Tensor] = None,
                 verbose: bool = False) -> Tuple[torch.Tensor, list]:
        """Run Adam optimisation.  Returns (E_best, loss_history)."""
        if E_init is not None:
            with torch.no_grad():
                self.log_amp.data = torch.log(torch.abs(E_init).clamp(min=1e-8))
                self.phase.data   = torch.angle(E_init)
        else:
            nn.init.zeros_(self.log_amp)
            nn.init.uniform_(self.phase, -torch.pi, torch.pi)

        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        history = []
        best_loss = float('inf')
        best_E    = None

        for step in range(self.n_steps):
            opt.zero_grad()
            E   = self._field()
            lv  = _loss(E, I1, I2, self.beta)
            lv.backward()
            opt.step()

            lf = lv.item()
            history.append(lf)
            if lf < best_loss:
                best_loss = lf
                best_E    = E.detach().clone()

            if verbose and step % 100 == 0:
                print(f"  step {step:4d}  loss={lf:.4e}")

        return best_E, history


# ---------------------------------------------------------------------------
# 3. HybridGS — GS warm-start + gradient fine-tune
# ---------------------------------------------------------------------------

class HybridGS:
    """Combine TorchGS projection and GradientPhaseRetrieval fine-tuning.

    Strategy:
      Phase 1: Run TorchGS for n_gs iterations to get a good initial field.
               Alternating projections find the right basin fast.
      Phase 2: Refine with Adam inside that basin.
               Gradient descent polishes the phase detail that
               alternating projections miss.

    This mirrors the Mario Kart ghost strategy in gs_unsupervised.py:
    the GS ghost gives you the best basin; gradient descent is the
    final lap polishing.
    """

    def __init__(self, N: int, D: float,
                 n_gs: int = 50, n_grad: int = 300, lr: float = 5e-3):
        self.gs   = TorchGS(N, D, n_iter=n_gs)
        self.grad = GradientPhaseRetrieval(N, D, lr=lr, n_steps=n_grad)

    def retrieve(self, I1: torch.Tensor, I2: torch.Tensor,
                 verbose: bool = False) -> Tuple[torch.Tensor, dict]:
        """Full retrieval: GS warm-start + gradient fine-tune.

        Returns (E_recovered, info_dict).
        """
        # Phase 1: GS projections
        with torch.no_grad():
            E_gs = self.gs(I1, I2)
        beta = self.gs.beta
        loss_gs = _loss(E_gs, I1, I2, beta).item()
        if verbose:
            print(f"After GS warm-start: loss = {loss_gs:.4e}")

        # Phase 2: gradient fine-tune starting from GS result
        E_best, history = self.grad.optimize(I1, I2, E_init=E_gs, verbose=verbose)
        loss_grad = history[-1]
        if verbose:
            print(f"After gradient fine-tune: loss = {loss_grad:.4e}")

        return E_best, {
            'loss_gs': loss_gs,
            'loss_grad': loss_grad,
            'loss_history': history,
            'improvement_db': 20 * torch.log10(
                torch.tensor(loss_gs / (loss_grad + 1e-12))).item()
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    torch.manual_seed(0)
    N, D = 256, -800.0
    t = np.linspace(-8, 8, N)

    # True field: Gaussian pulse with quadratic phase
    E_np  = np.exp(-t**2 / 2) * np.exp(1j * 0.3 * t**2)
    omega = np.fft.fftfreq(N) * 2 * np.pi
    beta_np = D / 2 * omega**2
    Ed_np = np.fft.ifft(np.fft.fft(E_np) * np.exp(1j * beta_np))

    I1 = torch.tensor(np.abs(E_np)**2,  dtype=torch.float32)
    I2 = torch.tensor(np.abs(Ed_np)**2, dtype=torch.float32)

    print("=== TorchGS (alternating projections) ===")
    gs = TorchGS(N, D, n_iter=100)
    E_gs = gs(I1, I2)
    corr_gs = float(np.corrcoef(E_gs.abs().numpy()**2, I1.numpy())[0, 1])
    print(f"  Amplitude correlation: {corr_gs:.4f}")

    print("\n=== HybridGS (GS + gradient descent) ===")
    hybrid = HybridGS(N, D, n_gs=50, n_grad=300, lr=5e-3)
    E_hyb, info = hybrid.retrieve(I1, I2, verbose=True)
    corr_hyb = float(np.corrcoef(E_hyb.abs().numpy()**2, I1.numpy())[0, 1])
    print(f"  Amplitude correlation: {corr_hyb:.4f}")
    print(f"  Improvement from gradient step: {info['improvement_db']:.2f} dB")
