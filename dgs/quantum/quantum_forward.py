# quantum_forward.py
# Quantum-like forward model with shot noise, squeezed noise, and correlations.

import torch
import torch.nn as nn
import torch.fft as fft

class QuantumLikeForward(nn.Module):
    def __init__(self, beta, squeeze_factor=0.0, correlated=False):
        super().__init__()
        self.register_buffer("beta", beta)
        self.squeeze_factor = squeeze_factor
        self.correlated = correlated

    def classical_intensity(self, x):
        X = fft.fft(x, dim=-1)
        D = torch.exp(1j * self.beta)
        Y = X * D
        y_time = fft.ifft(Y, dim=-1)
        return torch.abs(y_time) ** 2

    def add_shot_noise(self, I):
        lam = I.clamp(min=1e-8)
        return torch.poisson(lam)

    def add_squeezed_noise(self, I):
        r = self.squeeze_factor
        lam = I.clamp(min=1e-8)
        var = lam * torch.exp(-2 * r)
        noise = torch.randn_like(I) * torch.sqrt(var + 1e-8)
        return (lam + noise).clamp(min=0.0)

    def forward(self, x):
        I = self.classical_intensity(x)

        if self.squeeze_factor > 0:
            I_noisy = self.add_squeezed_noise(I)
        else:
            I_noisy = self.add_shot_noise(I)

        if self.correlated:
            g = torch.randn_like(I_noisy.mean(dim=-1, keepdim=True))
            I_noisy = I_noisy + 0.1 * g

        return I_noisy
